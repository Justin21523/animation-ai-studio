#!/usr/bin/env python3
"""
ControlNet Dataset Builder (SDXL)

Builds a reproducible on-disk dataset for ControlNet training:

output_dir/
  images/            # target images
  conditioning/      # control images (same index as images/)
  captions/          # text prompt/caption (same index as images/)
  dataset_metadata.json

Supports:
- Using precomputed control images (recommended for depth/normal/segmentation)
- Computing basic controls (canny/scribble/softedge/lineart) from target images

This module is designed to avoid network downloads and heavy model dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image
from tqdm import tqdm


logger = logging.getLogger(__name__)


SUPPORTED_IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict)")
    return data


def _iter_images(images_dir: Path) -> List[Path]:
    images_dir = Path(images_dir)
    files: List[Path] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in files if p.is_file()})


def _find_matching_file(stem: str, directory: Path) -> Optional[Path]:
    directory = Path(directory)
    for ext in SUPPORTED_IMAGE_EXTS:
        for cand in (directory / f"{stem}{ext}", directory / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def _read_caption_for_image(image_path: Path, captions_dir: Optional[Path], default_caption: str) -> str:
    # Sidecar caption next to image takes priority.
    sidecar = image_path.with_suffix(".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()

    if captions_dir:
        cand = Path(captions_dir) / f"{image_path.stem}.txt"
        if cand.exists():
            return cand.read_text(encoding="utf-8").strip()

    return default_caption.strip()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _resize_square(image: Image.Image, resolution: int) -> Image.Image:
    # Simple square resize; keep deterministic for reproducibility.
    return image.resize((resolution, resolution), Image.LANCZOS)


@dataclass(frozen=True)
class BuildConfig:
    images_dir: Path
    output_dir: Path
    control_type: str
    controlnet_config_path: str = "configs/generation/controlnet_config.yaml"
    detect_resolution: int = 512
    prefer_local_models: bool = True
    allow_download: bool = False
    device: str = "cuda"
    control_images_dir: Optional[Path] = None
    captions_dir: Optional[Path] = None
    resolution: int = 1024
    default_caption: str = ""
    overwrite: bool = False
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200


def build_dataset(config: BuildConfig) -> Dict[str, Any]:
    from scripts.processing.controlnet.preprocessors import (
        ControlNetPreprocessorFactory,
        PreprocessorContext,
        resolve_preprocess_as,
        resolve_detect_resolution,
    )

    images_dir = Path(config.images_dir)
    output_dir = Path(config.output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_out = output_dir / "images"
    cond_out = output_dir / "conditioning"
    captions_out = output_dir / "captions"
    for d in (images_out, cond_out, captions_out):
        if d.exists():
            if config.overwrite:
                shutil.rmtree(d)
            elif any(d.iterdir()):
                raise FileExistsError(f"Output directory not empty: {d} (use --overwrite to allow)")
        d.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "dataset_metadata.json"
    if meta_path.exists() and config.overwrite:
        meta_path.unlink()

    image_files = _iter_images(images_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir} (supported: {', '.join(SUPPORTED_IMAGE_EXTS)})")

    items: List[Dict[str, Any]] = []
    computed = 0
    skipped = 0

    detect_resolution = int(config.detect_resolution) if config.detect_resolution else resolve_detect_resolution(
        config.controlnet_config_path, fallback=512
    )
    preprocessor_factory = ControlNetPreprocessorFactory(
        PreprocessorContext(
            controlnet_config_path=str(config.controlnet_config_path),
            prefer_local_models=bool(config.prefer_local_models),
            allow_download=bool(config.allow_download),
            device=str(config.device),
            canny_low_threshold=int(config.canny_low_threshold),
            canny_high_threshold=int(config.canny_high_threshold),
        )
    )

    preprocessor = None
    preprocess_type = resolve_preprocess_as(str(config.controlnet_config_path), str(config.control_type))
    if not config.control_images_dir:
        # Fail fast for missing dependencies / missing local models.
        preprocessor = preprocessor_factory.get(str(preprocess_type))

    for idx, src_image_path in enumerate(tqdm(image_files, desc="Building ControlNet dataset")):
        try:
            with Image.open(src_image_path) as img:
                img_rgb = _ensure_rgb(img)
                img_rgb = _resize_square(img_rgb, int(config.resolution))

            caption = _read_caption_for_image(src_image_path, config.captions_dir, config.default_caption)

            # Control image: prefer precomputed dir; otherwise compute if supported.
            src_control_path: Optional[Path] = None
            if config.control_images_dir:
                src_control_path = _find_matching_file(src_image_path.stem, Path(config.control_images_dir))
                if not src_control_path:
                    raise FileNotFoundError(
                        f"Missing control image for '{src_image_path.name}' in {config.control_images_dir}"
                    )
                with Image.open(src_control_path) as cimg:
                    control_rgb = _ensure_rgb(cimg)
                    control_rgb = _resize_square(control_rgb, int(config.resolution))
            else:
                control_rgb = preprocessor(
                    img_rgb,
                    detect_resolution=int(detect_resolution),
                    image_resolution=int(config.resolution),
                )
                computed += 1

            base_name = f"{idx:06d}"
            out_image = images_out / f"{base_name}.png"
            out_cond = cond_out / f"{base_name}.png"
            out_caption = captions_out / f"{base_name}.txt"

            img_rgb.save(out_image, format="PNG", optimize=True)
            control_rgb.save(out_cond, format="PNG", optimize=True)
            out_caption.write_text(caption, encoding="utf-8")

            items.append(
                {
                    "id": int(idx),
                    "image": str(out_image.relative_to(output_dir)),
                    "conditioning_image": str(out_cond.relative_to(output_dir)),
                    "caption": caption,
                    "source_image": str(src_image_path),
                    "source_control_image": str(src_control_path) if src_control_path else None,
                }
            )
        except Exception as e:
            skipped += 1
            logger.warning("Skipping %s: %s", src_image_path, e)

    meta = {
        "control_type": str(config.control_type),
        "preprocess_type": str(preprocess_type),
        "resolution": int(config.resolution),
        "images_dir": str(images_dir),
        "control_images_dir": str(config.control_images_dir) if config.control_images_dir else None,
        "captions_dir": str(config.captions_dir) if config.captions_dir else None,
        "default_caption": config.default_caption,
        "items": items,
        "stats": {
            "total_found": len(image_files),
            "written": len(items),
            "skipped": int(skipped),
            "computed_controls": int(computed),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d items to %s", len(items), output_dir)
    return meta


def _build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a ControlNet training dataset on disk (SDXL).")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file")

    parser.add_argument("--images_dir", type=Path, required=defaults.get("images_dir") is None)
    parser.add_argument("--output_dir", type=Path, required=defaults.get("output_dir") is None)
    parser.add_argument("--control_type", type=str, required=defaults.get("control_type") is None)
    parser.add_argument(
        "--controlnet_config_path",
        type=str,
        default=str(defaults.get("controlnet_config_path", "configs/generation/controlnet_config.yaml")),
    )
    parser.add_argument("--detect_resolution", type=int, default=int(defaults.get("detect_resolution", 512)))
    parser.add_argument(
        "--prefer_local_models",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("prefer_local_models", True)),
    )
    parser.add_argument(
        "--allow_download",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("allow_download", False)),
    )
    parser.add_argument("--device", type=str, default=str(defaults.get("device", "cuda")))
    parser.add_argument("--control_images_dir", type=Path, default=defaults.get("control_images_dir"))
    parser.add_argument("--captions_dir", type=Path, default=defaults.get("captions_dir"))

    parser.add_argument("--resolution", type=int, default=int(defaults.get("resolution", 1024)))
    parser.add_argument("--default_caption", type=str, default=str(defaults.get("default_caption", "")))
    parser.add_argument("--overwrite", action="store_true", default=bool(defaults.get("overwrite", False)))

    parser.add_argument("--canny_low_threshold", type=int, default=int(defaults.get("canny_low_threshold", 100)))
    parser.add_argument("--canny_high_threshold", type=int, default=int(defaults.get("canny_high_threshold", 200)))

    parser.add_argument("--log_level", type=str, default=str(defaults.get("log_level", "INFO")))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = _load_yaml(known.config)

    parser = _build_parser(defaults)
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    cfg = BuildConfig(
        images_dir=Path(args.images_dir or defaults.get("images_dir")),
        output_dir=Path(args.output_dir or defaults.get("output_dir")),
        control_type=str(args.control_type or defaults.get("control_type")),
        controlnet_config_path=str(args.controlnet_config_path),
        detect_resolution=int(args.detect_resolution),
        prefer_local_models=bool(args.prefer_local_models),
        allow_download=bool(args.allow_download),
        device=str(args.device),
        control_images_dir=Path(args.control_images_dir) if args.control_images_dir else None,
        captions_dir=Path(args.captions_dir) if args.captions_dir else None,
        resolution=int(args.resolution),
        default_caption=str(args.default_caption),
        overwrite=bool(args.overwrite),
        canny_low_threshold=int(args.canny_low_threshold),
        canny_high_threshold=int(args.canny_high_threshold),
    )

    build_dataset(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
