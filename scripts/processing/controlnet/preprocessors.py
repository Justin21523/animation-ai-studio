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
            raise PreprocessorUnavailableError(
                "pose/openpose preprocessor not enabled yet (requires controlnet_aux)."
            )
        elif key in ("depth", "zoe_depth"):
            raise PreprocessorUnavailableError(
                "depth preprocessor not enabled yet (requires a local depth-estimation model)."
            )
        elif key in ("seg", "segmentation"):
            raise PreprocessorUnavailableError(
                "segmentation preprocessor not enabled yet (provide --control_images_dir or enable rembg/SAM2)."
            )
        elif key in ("normal", "normal_map"):
            raise PreprocessorUnavailableError(
                "normal preprocessor not enabled yet (provide --control_images_dir or enable depth->normal)."
            )
        else:
            raise PreprocessorUnavailableError(f"Unknown preprocess_type: {preprocess_type}")

        self._cache[key] = pre
        return pre


def resolve_detect_resolution(controlnet_config_path: str, fallback: int = 512) -> int:
    cfg = _load_yaml(controlnet_config_path)
    defaults_cfg = (cfg.get("defaults") or {}) if isinstance(cfg, dict) else {}
    return int(defaults_cfg.get("detection_resolution", fallback))
