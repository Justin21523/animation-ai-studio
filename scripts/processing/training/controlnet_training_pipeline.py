#!/usr/bin/env python3
"""
End-to-end ControlNet Training Pipeline (SDXL)

Given an input images directory, this pipeline will:
1) Build a ControlNet dataset on disk (auto-generate control maps if needed)
2) Train an SDXL ControlNet (diffusers + accelerate)
3) Upsert the trained model into configs/generation/controlnet_config.yaml

This is intended to make "only provide images → get a usable ControlNet model" reproducible and automated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


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


def _iter_images(images_dir: Path) -> Sequence[Path]:
    images_dir = Path(images_dir)
    files = []
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


def _is_main_process() -> bool:
    # Conservative check: if launched by a distributed runner, only rank 0 should touch filesystem outputs/registry.
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "ACCELERATE_PROCESS_INDEX"):
        val = os.environ.get(key)
        if val is not None and str(val) not in ("", "0"):
            return False
    return True


@dataclass(frozen=True)
class PipelineConfig:
    images_dir: Path
    control_type: str

    captions_dir: Optional[Path] = None
    control_images_dir: Optional[Path] = None
    default_caption: str = ""

    controlnet_config_path: str = "configs/generation/controlnet_config.yaml"
    detect_resolution: int = 512
    resolution: int = 1024
    prefer_local_models: bool = True
    allow_download: bool = False
    device: str = "cuda"
    overwrite_dataset: bool = False

    dataset_output_dir: Path = Path("outputs/controlnet_datasets")
    dataset_name: Optional[str] = None

    trainer_config_path: Path = Path("configs/training/controlnet/trainer.yaml")

    registry_path: Path = Path("configs/generation/controlnet_config.yaml")
    controlnet_key: Optional[str] = None
    preprocess_as: Optional[str] = None
    description: str = ""
    use_case: str = ""
    stage_to_dir: Optional[Path] = None
    stage_name: Optional[str] = None
    stage_move: bool = False
    stage_overwrite: bool = False


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_registry_local_path(controlnet_config_path: Path, key: str) -> Optional[str]:
    """
    Resolve a local ControlNet directory from the registry.

    This is useful for initializing training from an existing ControlNet of the same type.
    """
    key = str(key).strip()
    if not key:
        return None

    cfg = _load_yaml(controlnet_config_path)
    models = cfg.get("controlnet_models") or {}
    if not isinstance(models, dict):
        return None

    entry = models.get(key) or {}
    if not isinstance(entry, dict):
        return None

    local_path = entry.get("local_path")
    if not local_path:
        return None

    p = Path(str(local_path))
    return str(p) if p.exists() else None


def validate_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """
    Validate config + dependencies without writing outputs or training.

    Intended for "only images → train" automation preflight checks.
    """
    from scripts.processing.controlnet.preprocessors import (
        ControlNetPreprocessorFactory,
        PreprocessorContext,
        resolve_preprocess_as,
    )

    images_dir = Path(config.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    controlnet_config_path = Path(config.controlnet_config_path)
    if not controlnet_config_path.exists():
        raise FileNotFoundError(f"controlnet_config_path not found: {controlnet_config_path}")

    image_files = list(_iter_images(images_dir))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir} (supported: {', '.join(SUPPORTED_IMAGE_EXTS)})")

    trainer_cfg = _load_yaml(config.trainer_config_path)
    if not trainer_cfg:
        raise ValueError(f"Empty trainer config: {config.trainer_config_path}")

    missing_keys = [k for k in ("base_model_path", "output_dir", "output_name") if not trainer_cfg.get(k)]
    if missing_keys:
        raise ValueError(f"trainer config missing required keys: {', '.join(missing_keys)}")

    base_model_path = Path(str(trainer_cfg["base_model_path"]))
    if not base_model_path.exists():
        raise FileNotFoundError(f"base_model_path not found: {base_model_path}")

    base_repo_path = trainer_cfg.get("base_repo_path")
    if base_model_path.is_file():
        if not base_repo_path or not Path(str(base_repo_path)).exists():
            raise FileNotFoundError(
                "Single-file SDXL checkpoints require a local base_repo_path (tokenizers/text encoders/scheduler), "
                f"but got base_repo_path={base_repo_path!r}"
            )

    control_images_dir = Path(config.control_images_dir) if config.control_images_dir else None
    if control_images_dir:
        if not control_images_dir.exists():
            raise FileNotFoundError(f"control_images_dir not found: {control_images_dir}")
        missing_controls = []
        for img in image_files:
            if not _find_matching_file(img.stem, control_images_dir):
                missing_controls.append(img.name)
                if len(missing_controls) >= 20:
                    break
        if missing_controls:
            raise FileNotFoundError(
                "Missing control images for some targets in control_images_dir. "
                f"Examples: {', '.join(missing_controls[:5])}"
            )
        preprocessor_info: Optional[Dict[str, Any]] = None
        preprocess_type = None
    else:
        preprocess_type = resolve_preprocess_as(str(controlnet_config_path), str(config.control_type))
        factory = ControlNetPreprocessorFactory(
            PreprocessorContext(
                controlnet_config_path=str(controlnet_config_path),
                prefer_local_models=bool(config.prefer_local_models),
                allow_download=bool(config.allow_download),
                device=str(config.device),
            )
        )
        preprocessor_info = factory.validate(str(preprocess_type))

    init_controlnet_resolved = None
    if not trainer_cfg.get("init_controlnet_path"):
        init_controlnet_resolved = _resolve_registry_local_path(controlnet_config_path, preprocess_type or "")
        if init_controlnet_resolved is None:
            init_controlnet_resolved = _resolve_registry_local_path(controlnet_config_path, str(config.control_type))

    return {
        "dry_run": True,
        "images_dir": str(images_dir),
        "images_found": len(image_files),
        "control_type": str(config.control_type),
        "control_images_dir": str(control_images_dir) if control_images_dir else None,
        "preprocess_type": str(preprocess_type) if preprocess_type else None,
        "preprocessor": preprocessor_info,
        "trainer_config_path": str(config.trainer_config_path),
        "trainer_output_name": str(trainer_cfg.get("output_name")),
        "base_model_path": str(base_model_path),
        "base_repo_path": str(base_repo_path) if base_repo_path else None,
        "init_controlnet_path": str(trainer_cfg.get("init_controlnet_path")) if trainer_cfg.get("init_controlnet_path") else None,
        "init_controlnet_resolved": init_controlnet_resolved,
    }


def _maybe_resolve_init_controlnet(
    *,
    trainer_cfg: Dict[str, Any],
    controlnet_config_path: Path,
    control_type: str,
    preprocess_type: str,
) -> Optional[str]:
    if trainer_cfg.get("init_controlnet_path"):
        return str(trainer_cfg.get("init_controlnet_path"))

    resolved = _resolve_registry_local_path(controlnet_config_path, preprocess_type)
    if resolved is not None:
        return resolved
    return _resolve_registry_local_path(controlnet_config_path, control_type)


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    from scripts.processing.training.controlnet_dataset_builder import BuildConfig, build_dataset
    from scripts.processing.training.sdxl_controlnet_trainer import TrainConfig, train
    from scripts.processing.training.controlnet_registry_updater import upsert_controlnet_entry
    from scripts.processing.controlnet.preprocessors import resolve_preprocess_as

    if not _is_main_process():
        raise RuntimeError(
            "controlnet_training_pipeline.py must be run as a single process (not under a distributed launcher). "
            "Run the trainer separately with `accelerate launch` if you need multi-GPU."
        )

    images_dir = Path(config.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    trainer_cfg = _load_yaml(config.trainer_config_path)
    if not trainer_cfg:
        raise ValueError(f"Empty trainer config: {config.trainer_config_path}")

    trainer_output_name = str(trainer_cfg.get("output_name") or "").strip() or "controlnet_model"

    preprocess_type = resolve_preprocess_as(str(config.controlnet_config_path), str(config.control_type))

    dataset_output_root = _ensure_dir(Path(config.dataset_output_dir))
    dataset_name = config.dataset_name or config.controlnet_key or trainer_output_name
    dataset_dir = dataset_output_root / str(dataset_name)

    logger.info("Building dataset: %s", dataset_dir)
    dataset_meta = build_dataset(
        BuildConfig(
            images_dir=images_dir,
            output_dir=dataset_dir,
            control_type=str(config.control_type),
            controlnet_config_path=str(config.controlnet_config_path),
            detect_resolution=int(config.detect_resolution),
            prefer_local_models=bool(config.prefer_local_models),
            allow_download=bool(config.allow_download),
            device=str(config.device),
            control_images_dir=Path(config.control_images_dir) if config.control_images_dir else None,
            captions_dir=Path(config.captions_dir) if config.captions_dir else None,
            resolution=int(config.resolution),
            default_caption=str(config.default_caption),
            overwrite=bool(config.overwrite_dataset),
        )
    )

    # Fill dataset_dir in trainer config (without mutating the file on disk).
    trainer_cfg = dict(trainer_cfg)
    trainer_cfg["dataset_dir"] = str(dataset_dir)
    init_controlnet_path = _maybe_resolve_init_controlnet(
        trainer_cfg=trainer_cfg,
        controlnet_config_path=Path(config.controlnet_config_path),
        control_type=str(config.control_type),
        preprocess_type=str(preprocess_type),
    )

    train_cfg = TrainConfig(
        dataset_dir=Path(trainer_cfg["dataset_dir"]),
        base_model_path=str(trainer_cfg["base_model_path"]),
        base_repo_path=str(trainer_cfg.get("base_repo_path")) if trainer_cfg.get("base_repo_path") else None,
        output_dir=Path(trainer_cfg["output_dir"]),
        output_name=str(trainer_cfg["output_name"]),
        resolution=int(trainer_cfg.get("resolution", config.resolution)),
        init_controlnet_path=str(init_controlnet_path) if init_controlnet_path else None,
        train_batch_size=int(trainer_cfg.get("train_batch_size", 1)),
        gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(trainer_cfg.get("learning_rate", 1e-5)),
        max_train_steps=int(trainer_cfg.get("max_train_steps", 1000)),
        save_steps=int(trainer_cfg.get("save_steps", 500)),
        max_grad_norm=float(trainer_cfg.get("max_grad_norm", 1.0)),
        mixed_precision=str(trainer_cfg.get("mixed_precision", "fp16")),
        seed=int(trainer_cfg.get("seed", 42)),
        num_workers=int(trainer_cfg.get("num_workers", 0)),
        enable_gradient_checkpointing=bool(trainer_cfg.get("enable_gradient_checkpointing", True)),
    )

    logger.info("Training ControlNet: output=%s name=%s", train_cfg.output_dir, train_cfg.output_name)
    final_dir = train(train_cfg)

    key = config.controlnet_key or str(train_cfg.output_name)
    preprocess_as = config.preprocess_as or str(config.control_type)

    logger.info("Updating registry: key=%s model_dir=%s", key, final_dir)
    result = upsert_controlnet_entry(
        registry_path=Path(config.registry_path),
        controlnet_key=str(key),
        model_dir=Path(final_dir),
        model_id=None,
        description=str(config.description),
        use_case=str(config.use_case),
        preprocess_as=str(preprocess_as) if preprocess_as else None,
        stage_to_dir=Path(config.stage_to_dir) if config.stage_to_dir else None,
        stage_name=str(config.stage_name) if config.stage_name else None,
        stage_move=bool(config.stage_move),
        stage_overwrite=bool(config.stage_overwrite),
    )

    return {
        "dataset_dir": str(dataset_dir),
        "dataset_written": int((dataset_meta.get("stats") or {}).get("written", 0)),
        "trained_model_dir": str(final_dir),
        "registry_path": str(result.registry_path),
        "controlnet_key": str(result.controlnet_key),
        "registry_local_path": str(result.final_model_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end SDXL ControlNet training pipeline.")
    p.add_argument("--config", type=Path, required=True, help="Pipeline YAML config")

    # Common overrides
    p.add_argument("--images_dir", type=Path, default=None)
    p.add_argument("--control_type", type=str, default=None)
    p.add_argument("--trainer_config_path", type=Path, default=None)
    p.add_argument("--controlnet_key", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", default=False, help="Validate config/deps without training")

    p.add_argument("--log_level", type=str, default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    raw = _load_yaml(args.config)

    images_dir = Path(args.images_dir or raw.get("images_dir"))
    control_type = str(args.control_type or raw.get("control_type"))

    if not images_dir or str(images_dir) in ("None", ""):
        raise ValueError("Missing images_dir")
    if not control_type or control_type in ("None", ""):
        raise ValueError("Missing control_type")

    pipeline_cfg = PipelineConfig(
        images_dir=images_dir,
        control_type=control_type,
        captions_dir=Path(raw["captions_dir"]) if raw.get("captions_dir") else None,
        control_images_dir=Path(raw["control_images_dir"]) if raw.get("control_images_dir") else None,
        default_caption=str(raw.get("default_caption", "")),
        controlnet_config_path=str(raw.get("controlnet_config_path", "configs/generation/controlnet_config.yaml")),
        detect_resolution=int(raw.get("detect_resolution", 512)),
        resolution=int(raw.get("resolution", 1024)),
        prefer_local_models=bool(raw.get("prefer_local_models", True)),
        allow_download=bool(raw.get("allow_download", False)),
        device=str(raw.get("device", "cuda")),
        overwrite_dataset=bool(raw.get("overwrite_dataset", False)),
        dataset_output_dir=Path(raw.get("dataset_output_dir", "outputs/controlnet_datasets")),
        dataset_name=str(raw.get("dataset_name")) if raw.get("dataset_name") else None,
        trainer_config_path=Path(args.trainer_config_path or raw.get("trainer_config_path", "configs/training/controlnet/trainer.yaml")),
        registry_path=Path(raw.get("registry_path", "configs/generation/controlnet_config.yaml")),
        controlnet_key=str(args.controlnet_key or raw.get("controlnet_key")) if (args.controlnet_key or raw.get("controlnet_key")) else None,
        preprocess_as=str(raw.get("preprocess_as")) if raw.get("preprocess_as") else None,
        description=str(raw.get("description", "")),
        use_case=str(raw.get("use_case", "")),
        stage_to_dir=Path(raw["stage_to_dir"]) if raw.get("stage_to_dir") else None,
        stage_name=str(raw.get("stage_name")) if raw.get("stage_name") else None,
        stage_move=bool(raw.get("stage_move", False)),
        stage_overwrite=bool(raw.get("stage_overwrite", False)),
    )

    if args.dry_run:
        result = validate_pipeline(pipeline_cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result = run_pipeline(pipeline_cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
