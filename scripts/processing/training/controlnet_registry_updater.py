#!/usr/bin/env python3
"""
ControlNet Registry Updater

Automates updating `configs/generation/controlnet_config.yaml` after a ControlNet training run.

Workflow (P4):
- Training completed → locate model directory (typically .../final)
- Optional "落盤": copy/move model directory to a stable location
- Upsert entry into `controlnet_models` so generation tools/agents can use it immediately

Registry schema (existing):
  controlnet_models:
    <key>:
      model_id: <optional HF id>
      local_path: <optional local path (directory)>
      description: <string>
      use_case: <string>
      preprocess_as: <optional, e.g. canny/depth/pose>
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


logger = logging.getLogger(__name__)


DEFAULT_REGISTRY_PATH = "configs/generation/controlnet_config.yaml"


@dataclass(frozen=True)
class RegistryUpdateResult:
    registry_path: Path
    controlnet_key: str
    created: bool
    updated: bool
    final_model_dir: Path


def _ensure_registry_shape(data: Any) -> Dict[str, Any]:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("ControlNet registry must be a YAML mapping (dict)")
    if "controlnet_models" not in data or data["controlnet_models"] is None:
        data["controlnet_models"] = {}
    if not isinstance(data["controlnet_models"], dict):
        raise ValueError("ControlNet registry key `controlnet_models` must be a YAML mapping (dict)")
    return data


def load_registry(registry_path: Path) -> Dict[str, Any]:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return _ensure_registry_shape({})
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _ensure_registry_shape(data)


def save_registry(registry_path: Path, data: Dict[str, Any]) -> None:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def stage_model_dir(
    model_dir: Path,
    dest_dir: Optional[Path] = None,
    dest_name: Optional[str] = None,
    *,
    move: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Optional "落盤": copy/move trained model directory to a stable destination.

    Returns:
        Final model directory (original if dest_dir is None).
    """
    model_dir = Path(model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    if dest_dir is None:
        return model_dir

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = dest_name or model_dir.name
    dest_path = dest_dir / dest_name

    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dest_path}")
        shutil.rmtree(dest_path)

    if move:
        shutil.move(str(model_dir), str(dest_path))
    else:
        shutil.copytree(model_dir, dest_path)

    return dest_path


def _find_existing_entry(models: Dict[str, Any], key: str) -> Tuple[bool, Dict[str, Any]]:
    existing = models.get(key)
    if isinstance(existing, dict):
        return True, existing
    return False, {}


def upsert_controlnet_entry(
    *,
    registry_path: Path,
    controlnet_key: str,
    model_dir: Path,
    model_id: Optional[str] = None,
    description: str = "",
    use_case: str = "",
    preprocess_as: Optional[str] = None,
    stage_to_dir: Optional[Path] = None,
    stage_name: Optional[str] = None,
    stage_move: bool = False,
    stage_overwrite: bool = False,
) -> RegistryUpdateResult:
    registry_path = Path(registry_path)
    data = load_registry(registry_path)

    final_model_dir = stage_model_dir(
        model_dir,
        dest_dir=stage_to_dir,
        dest_name=stage_name,
        move=stage_move,
        overwrite=stage_overwrite,
    )

    models = data["controlnet_models"]
    existed, existing_entry = _find_existing_entry(models, controlnet_key)

    entry: Dict[str, Any] = dict(existing_entry) if existed else {}
    entry["model_id"] = model_id if model_id is not None else entry.get("model_id")
    entry["local_path"] = str(final_model_dir)
    if description:
        entry["description"] = description
    if use_case:
        entry["use_case"] = use_case
    if preprocess_as:
        entry["preprocess_as"] = preprocess_as

    models[controlnet_key] = entry
    save_registry(registry_path, data)

    return RegistryUpdateResult(
        registry_path=registry_path,
        controlnet_key=controlnet_key,
        created=not existed,
        updated=True,
        final_model_dir=final_model_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upsert a trained ControlNet into controlnet_config.yaml")
    p.add_argument("--registry_path", type=Path, default=Path(DEFAULT_REGISTRY_PATH))
    p.add_argument("--controlnet_key", type=str, required=True, help="Key under controlnet_models (e.g. depth_game)")
    p.add_argument("--model_dir", type=Path, required=True, help="Directory containing ControlNet weights/config.json")
    p.add_argument("--model_id", type=str, default=None, help="Optional HF model id (kept for reference)")
    p.add_argument("--description", type=str, default="")
    p.add_argument("--use_case", type=str, default="")
    p.add_argument("--preprocess_as", type=str, default=None, help="Optional: canny|depth|pose|seg|normal")

    p.add_argument("--stage_to_dir", type=Path, default=None, help="Optional stable destination dir")
    p.add_argument("--stage_name", type=str, default=None, help="Optional destination folder name")
    p.add_argument("--stage_move", action="store_true", default=False)
    p.add_argument("--stage_overwrite", action="store_true", default=False)

    p.add_argument("--log_level", type=str, default="INFO")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    result = upsert_controlnet_entry(
        registry_path=args.registry_path,
        controlnet_key=args.controlnet_key,
        model_dir=args.model_dir,
        model_id=args.model_id,
        description=args.description,
        use_case=args.use_case,
        preprocess_as=args.preprocess_as,
        stage_to_dir=args.stage_to_dir,
        stage_name=args.stage_name,
        stage_move=bool(args.stage_move),
        stage_overwrite=bool(args.stage_overwrite),
    )

    logger.info(
        "Updated registry=%s key=%s local_path=%s",
        result.registry_path,
        result.controlnet_key,
        result.final_model_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

