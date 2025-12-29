#!/usr/bin/env python3
"""
ControlNet Preprocessor Model Setup (local-first)

Helps you make the "only provide images" pipeline fully reproducible by ensuring
OpenPose / Depth models are available locally and wired into:
  configs/generation/controlnet_config.yaml (preprocessing.*.local_path)

Supports:
- Locate existing HuggingFace cache snapshot for a given model_id
- (Optional) Download via huggingface_hub.snapshot_download (requires network)
- Update YAML config with resolved local_path
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


DEFAULT_CONTROLNET_CONFIG = Path("configs/generation/controlnet_config.yaml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping (dict)")
    return data


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _hf_cache_root() -> Path:
    # Priority order: HF_HUB_CACHE (direct), HF_HOME (root), default.
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_cached_snapshot(model_id: str) -> Optional[Path]:
    model_id = str(model_id).strip()
    if not model_id or "/" not in model_id:
        return None

    org, repo = model_id.split("/", 1)
    root = _hf_cache_root() / f"models--{org}--{repo}"
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None

    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Latest by mtime.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def snapshot_download_to(model_id: str, dest_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=str(model_id),
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
    )


@dataclass(frozen=True)
class UpdateRequest:
    key: str  # "pose" or "depth"
    model_id: str
    dest_dir: Optional[Path]
    allow_download: bool


def _get_preprocessing_entry(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    preprocessing = data.setdefault("preprocessing", {}) or {}
    if not isinstance(preprocessing, dict):
        raise ValueError("controlnet_config.yaml: `preprocessing` must be a mapping")
    entry = preprocessing.setdefault(key, {}) or {}
    if not isinstance(entry, dict):
        raise ValueError(f"controlnet_config.yaml: `preprocessing.{key}` must be a mapping")
    return entry


def update_local_path(controlnet_config_path: Path, req: UpdateRequest) -> Path:
    data = _load_yaml(controlnet_config_path)
    entry = _get_preprocessing_entry(data, req.key)

    # Prefer explicit dest_dir download target if allowed.
    resolved: Optional[Path] = None

    # 1) Use existing local_path if it exists.
    existing = entry.get("local_path")
    if existing and Path(str(existing)).exists():
        resolved = Path(str(existing))

    # 2) Try HF cache snapshot.
    if resolved is None:
        cached = find_cached_snapshot(req.model_id)
        if cached and cached.exists():
            resolved = cached

    # 3) Optional download.
    if resolved is None and req.allow_download:
        if not req.dest_dir:
            raise ValueError("--dest_dir is required when --download is enabled")
        resolved = snapshot_download_to(req.model_id, req.dest_dir)

    if resolved is None:
        raise FileNotFoundError(
            f"Cannot resolve a local model for preprocessing.{req.key} model_id={req.model_id!r}. "
            "Either download it (with --download) or set preprocessing.<key>.local_path manually."
        )

    entry["local_path"] = str(resolved)
    _save_yaml(controlnet_config_path, data)
    return resolved


def main() -> int:
    p = argparse.ArgumentParser(description="Setup local model paths for ControlNet preprocessors.")
    p.add_argument("--controlnet_config_path", type=Path, default=DEFAULT_CONTROLNET_CONFIG)
    p.add_argument("--preprocessor", type=str, choices=["pose", "depth"], required=True)
    p.add_argument("--model_id", type=str, default=None, help="Override model_id (defaults to config preprocessing.<key>.model_id)")
    p.add_argument("--dest_dir", type=Path, default=None, help="Download destination (used with --download)")
    p.add_argument("--download", action="store_true", default=False, help="Allow downloading missing models (needs network)")
    p.add_argument("--log_level", type=str, default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    data = _load_yaml(args.controlnet_config_path)
    entry = _get_preprocessing_entry(data, args.preprocessor)
    model_id = str(args.model_id or entry.get("model_id") or entry.get("model") or "").strip()
    if not model_id:
        raise ValueError(f"Missing model_id for preprocessing.{args.preprocessor} (set in config or pass --model_id)")

    resolved = update_local_path(
        args.controlnet_config_path,
        UpdateRequest(
            key=args.preprocessor,
            model_id=model_id,
            dest_dir=args.dest_dir,
            allow_download=bool(args.download),
        ),
    )

    logger.info("Updated %s preprocessing.%s.local_path=%s", args.controlnet_config_path, args.preprocessor, resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

