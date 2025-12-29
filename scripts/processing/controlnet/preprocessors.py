"""
ControlNet control-image preprocessors.

Design goals:
- Local-first: prefer local model paths to avoid network downloads.
- Optional dependencies: only import heavy deps (cv2, transformers, controlnet_aux) when needed.
- Reusable: used by both dataset builder and generation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PIL import Image


class PreprocessorUnavailableError(RuntimeError):
    """Raised when a requested preprocessor cannot be constructed (missing deps/models)."""


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping (dict)")
    return data


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


@dataclass(frozen=True)
class PreprocessorContext:
    controlnet_config_path: str = "configs/generation/controlnet_config.yaml"
    prefer_local_models: bool = True
    allow_download: bool = False
    device: str = "cuda"
    canny_low_threshold: Optional[int] = None
    canny_high_threshold: Optional[int] = None


class ControlImagePreprocessor:
    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        raise NotImplementedError


class CannyLikePreprocessor(ControlImagePreprocessor):
    def __init__(
        self,
        *,
        mode: str,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ):
        self.mode = mode
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)

    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            raise PreprocessorUnavailableError("opencv-python is required for canny-like preprocessors") from e

        image = image.convert("RGB").resize((int(detect_resolution), int(detect_resolution)))
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        if self.mode == "softedge":
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        if self.mode == "scribble":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

        edges_rgb = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_rgb).resize((int(image_resolution), int(image_resolution)))


class TilePreprocessor(ControlImagePreprocessor):
    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        # Tile ControlNet is commonly conditioned on the image itself (possibly resized).
        return image.convert("RGB").resize((int(image_resolution), int(image_resolution)), Image.LANCZOS)


class OpenPosePreprocessor(ControlImagePreprocessor):
    def __init__(
        self,
        *,
        model_source: str,
        hand_and_face: bool = False,
    ):
        try:
            from controlnet_aux import OpenposeDetector
        except ImportError as e:
            raise PreprocessorUnavailableError(
                "controlnet_aux is required for OpenPose preprocessing. Install with: pip install controlnet-aux"
            ) from e

        self._detector = OpenposeDetector.from_pretrained(model_source)
        self._hand_and_face = bool(hand_and_face)

    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        return self._detector(
            image.convert("RGB"),
            hand_and_face=self._hand_and_face,
            detect_resolution=int(detect_resolution),
            image_resolution=int(image_resolution),
        )


class TransformersDepthPreprocessor(ControlImagePreprocessor):
    def __init__(
        self,
        *,
        model_source: str,
        device: str,
        allow_download: bool,
    ):
        try:
            import torch
        except ImportError as e:
            raise PreprocessorUnavailableError("torch is required for depth preprocessing") from e

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise PreprocessorUnavailableError(
                "transformers is required for depth preprocessing. Install with: pip install transformers"
            ) from e

        resolved_device = str(device)
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        self._device = torch.device(resolved_device)

        local_files_only = not bool(allow_download)
        torch_dtype = torch.float16 if self._device.type == "cuda" else torch.float32

        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model_source,
                local_files_only=local_files_only,
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                model_source,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
            )
        except Exception as e:
            raise PreprocessorUnavailableError(
                f"Failed to load depth model from '{model_source}'. "
                "Provide a local model directory via `preprocessing.depth.local_path` "
                "or run with --allow_download."
            ) from e

        self._model.to(self._device)
        self._model.eval()

    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        import numpy as np
        import torch
        import torch.nn.functional as F

        image = image.convert("RGB").resize((int(detect_resolution), int(detect_resolution)), Image.LANCZOS)

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            depth = getattr(outputs, "predicted_depth", None)
            if depth is None:
                raise PreprocessorUnavailableError("Depth model output missing `predicted_depth`")

        # depth: (B, H, W) -> (B, 1, H, W)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(
            depth,
            size=(int(image_resolution), int(image_resolution)),
            mode="bicubic",
            align_corners=False,
        )
        depth = depth.squeeze(1)

        # Normalize per image
        depth_min = depth.amin(dim=(1, 2), keepdim=True)
        depth_max = depth.amax(dim=(1, 2), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth = depth.clamp(0.0, 1.0)

        depth_u8 = (depth[0] * 255.0).to(torch.uint8).cpu().numpy()
        depth_rgb = np.stack([depth_u8] * 3, axis=-1)
        return Image.fromarray(depth_rgb)


class RembgSegmentationPreprocessor(ControlImagePreprocessor):
    def __init__(self, *, model_name: str = "isnet-general-use"):
        try:
            from rembg import new_session, remove
        except ImportError as e:
            raise PreprocessorUnavailableError(
                "rembg is required for segmentation preprocessing. Install with: pip install rembg"
            ) from e

        self._remove = remove
        self._session = new_session(str(model_name))

    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        import numpy as np

        image = image.convert("RGB").resize((int(detect_resolution), int(detect_resolution)), Image.LANCZOS)
        mask = self._remove(image, session=self._session, only_mask=True)
        mask = mask.resize((int(image_resolution), int(image_resolution)), Image.NEAREST)

        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_u8 = (mask_np > 127).astype("uint8") * 255
        seg_rgb = np.stack([mask_u8] * 3, axis=-1)
        return Image.fromarray(seg_rgb)


class NormalFromDepthPreprocessor(ControlImagePreprocessor):
    def __init__(self, depth_preprocessor: ControlImagePreprocessor):
        self._depth = depth_preprocessor

    def __call__(self, image: Image.Image, *, detect_resolution: int, image_resolution: int) -> Image.Image:
        import numpy as np

        depth_img = self._depth(image, detect_resolution=detect_resolution, image_resolution=image_resolution)
        depth = np.array(depth_img.convert("L")).astype("float32") / 255.0

        dzdy, dzdx = np.gradient(depth)
        nx = -dzdx
        ny = -dzdy
        nz = np.ones_like(depth, dtype="float32")

        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        normal = np.stack([(nx * 0.5 + 0.5), (ny * 0.5 + 0.5), (nz * 0.5 + 0.5)], axis=-1)
        normal_u8 = (normal * 255.0).clip(0, 255).astype("uint8")
        return Image.fromarray(normal_u8)


class ControlNetPreprocessorFactory:
    def __init__(self, context: PreprocessorContext):
        self.context = context
        self._cfg = _load_yaml(context.controlnet_config_path)
        self._cache: Dict[str, ControlImagePreprocessor] = {}

    def get(self, preprocess_type: str) -> ControlImagePreprocessor:
        key = str(preprocess_type).strip().lower()
        if key in self._cache:
            return self._cache[key]

        preprocessing_cfg = (self._cfg.get("preprocessing") or {}) if isinstance(self._cfg, dict) else {}
        defaults_cfg = (self._cfg.get("defaults") or {}) if isinstance(self._cfg, dict) else {}

        canny_cfg = preprocessing_cfg.get("canny") or {}
        low_thr = int(self.context.canny_low_threshold) if self.context.canny_low_threshold is not None else int(
            canny_cfg.get("low_threshold", 100)
        )
        high_thr = int(self.context.canny_high_threshold) if self.context.canny_high_threshold is not None else int(
            canny_cfg.get("high_threshold", 200)
        )

        if key in ("canny", "scribble", "softedge", "lineart"):
            pre = CannyLikePreprocessor(mode=key, low_threshold=low_thr, high_threshold=high_thr)
        elif key == "tile":
            pre = TilePreprocessor()
        elif key in ("pose", "openpose"):
            pose_cfg = preprocessing_cfg.get("pose") or {}
            model_id = pose_cfg.get("model_id") or pose_cfg.get("model") or "lllyasviel/ControlNet"
            local_path = pose_cfg.get("local_path")
            local_path = str(local_path) if local_path else None

            model_source: Optional[str] = None
            if self.context.prefer_local_models and local_path and Path(local_path).exists():
                model_source = local_path
            elif model_id and self.context.allow_download:
                model_source = str(model_id)

            if not model_source:
                raise PreprocessorUnavailableError(
                    "OpenPose preprocessor requires a local model. "
                    "Set `preprocessing.pose.local_path` in configs/generation/controlnet_config.yaml "
                    "or run with --allow_download."
                )

            hand_and_face = _coerce_bool(pose_cfg.get("hand"), False) or _coerce_bool(pose_cfg.get("face"), False)
            pre = OpenPosePreprocessor(
                model_source=model_source,
                hand_and_face=hand_and_face,
            )
        elif key in ("depth", "zoe_depth"):
            depth_cfg = preprocessing_cfg.get("depth") or {}
            model_id = depth_cfg.get("model_id") or depth_cfg.get("model") or "Intel/dpt-hybrid-midas"
            local_path = depth_cfg.get("local_path")
            local_path = str(local_path) if local_path else None

            model_source: Optional[str] = None
            if self.context.prefer_local_models and local_path and Path(local_path).exists():
                model_source = local_path
            elif model_id and self.context.allow_download:
                model_source = str(model_id)

            if not model_source:
                raise PreprocessorUnavailableError(
                    "Depth preprocessor requires a local model. "
                    "Set `preprocessing.depth.local_path` in configs/generation/controlnet_config.yaml "
                    "or run with --allow_download."
                )

            device = str(depth_cfg.get("device") or self.context.device)
            pre = TransformersDepthPreprocessor(
                model_source=model_source,
                device=device,
                allow_download=bool(self.context.allow_download),
            )
        elif key in ("seg", "segmentation"):
            seg_cfg = preprocessing_cfg.get("seg") or preprocessing_cfg.get("segmentation") or {}
            model_name = seg_cfg.get("model") or "isnet-general-use"
            pre = RembgSegmentationPreprocessor(model_name=str(model_name))
        elif key in ("normal", "normal_map"):
            depth_pre = self.get("depth")
            pre = NormalFromDepthPreprocessor(depth_pre)
        else:
            raise PreprocessorUnavailableError(f"Unknown preprocess_type: {preprocess_type}")

        self._cache[key] = pre
        return pre


def resolve_detect_resolution(controlnet_config_path: str, fallback: int = 512) -> int:
    cfg = _load_yaml(controlnet_config_path)
    defaults_cfg = (cfg.get("defaults") or {}) if isinstance(cfg, dict) else {}
    return int(defaults_cfg.get("detection_resolution", fallback))
