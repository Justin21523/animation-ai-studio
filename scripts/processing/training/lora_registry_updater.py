#!/usr/bin/env python3
"""
LoRA Registry Updater

Automates updating `configs/generation/lora_registry.yaml` after a Kohya_ss LoRA training job.

Goals (P1):
- Training completed → find/check checkpoint artifact
- Optional "落盤" (copy/move artifact to a stable location)
- Upsert LoRA entry in registry so generation tools/agents can use it immediately

This module is designed to be imported by training launchers/orchestrators, and also provides a CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


logger = logging.getLogger(__name__)


DEFAULT_REGISTRY_PATH = "configs/generation/lora_registry.yaml"


@dataclass(frozen=True)
class RegistryUpdateResult:
    """Result of an upsert operation."""

    registry_path: Path
    lora_name: str
    created: bool
    updated: bool
    entry_index: int
    artifact_path: Path


def _ensure_registry_shape(data: Any) -> Dict[str, Any]:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("LoRA registry must be a YAML mapping (dict)")

    if "loras" not in data or data["loras"] is None:
        data["loras"] = []
    if not isinstance(data["loras"], list):
        raise ValueError("LoRA registry key `loras` must be a YAML list")

    if "combinations" not in data or data["combinations"] is None:
        data["combinations"] = {}
    if not isinstance(data["combinations"], dict):
        raise ValueError("LoRA registry key `combinations` must be a YAML mapping (dict)")

    return data


def load_registry(registry_path: Path) -> Dict[str, Any]:
    """Load registry YAML, creating an empty structure if file does not exist."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return _ensure_registry_shape({})

    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _ensure_registry_shape(data)


def save_registry(registry_path: Path, data: Dict[str, Any]) -> None:
    """Write registry YAML back to disk."""
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep a stable top-level key order (loras first).
    normalized = {
        "loras": data.get("loras", []),
        "combinations": data.get("combinations", {}),
    }
    # Preserve any other unexpected keys without dropping them.
    for key, value in data.items():
        if key not in normalized:
            normalized[key] = value

    with open(registry_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            normalized,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def stage_artifact(
    artifact_path: Path,
    dest_dir: Optional[Path] = None,
    dest_filename: Optional[str] = None,
    *,
    move: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Optional "落盤": copy/move artifact to a stable destination directory.

    Returns:
        Final artifact path (original if dest_dir is None).
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    if dest_dir is None:
        return artifact_path

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = dest_filename or artifact_path.name
    dest_path = dest_dir / dest_filename

    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dest_path}")

    if move:
        shutil.move(str(artifact_path), str(dest_path))
    else:
        shutil.copy2(str(artifact_path), str(dest_path))

    return dest_path


def _find_entry_by_name(loras: List[Dict[str, Any]], name: str) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    for idx, entry in enumerate(loras):
        if isinstance(entry, dict) and entry.get("name") == name:
            return idx, entry
    return None, None


def upsert_lora_entry(
    *,
    registry_path: Path,
    lora_name: str,
    lora_path: Path,
    lora_type: str = "character",
    trigger_words: Optional[List[str]] = None,
    recommended_weight: float = 0.8,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    stage_to_dir: Optional[Path] = None,
    stage_filename: Optional[str] = None,
    stage_move: bool = False,
    stage_overwrite: bool = False,
) -> RegistryUpdateResult:
    """
    Add or update a LoRA entry in the registry.

    Notes:
    - If `stage_to_dir` is provided, the artifact will be copied/moved first, and the registry path
      will point to the staged file.
    """
    registry_path = Path(registry_path)
    lora_path = Path(lora_path)

    # "落盤" first (optional).
    final_artifact_path = stage_artifact(
        lora_path,
        dest_dir=stage_to_dir,
        dest_filename=stage_filename,
        move=stage_move,
        overwrite=stage_overwrite,
    )

    data = load_registry(registry_path)
    loras = data["loras"]

    idx, existing = _find_entry_by_name(loras, lora_name)
    created = existing is None
    updated = False

    trigger_words = trigger_words or [lora_name]
    metadata = metadata or {}

    new_fields: Dict[str, Any] = {
        "name": lora_name,
        "path": str(final_artifact_path),
        "type": lora_type,
        "trigger_words": trigger_words,
        "recommended_weight": float(recommended_weight),
        "description": description,
        "metadata": metadata,
    }

    if created:
        loras.append(new_fields)
        idx = len(loras) - 1
        updated = True
    else:
        # Only update fields we know about; keep any custom fields on existing entries.
        merged = dict(existing)
        for k, v in new_fields.items():
            if v is None:
                continue
            merged[k] = v
        loras[idx] = merged
        updated = True

    save_registry(registry_path, data)

    return RegistryUpdateResult(
        registry_path=registry_path,
        lora_name=lora_name,
        created=created,
        updated=updated,
        entry_index=int(idx),
        artifact_path=final_artifact_path,
    )


def _parse_trigger_words(args: argparse.Namespace) -> Optional[List[str]]:
    if getattr(args, "trigger_word", None):
        return list(args.trigger_word)
    if getattr(args, "trigger_words", None):
        words = [w.strip() for w in args.trigger_words.split(",")]
        return [w for w in words if w]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert a LoRA entry into lora_registry.yaml")
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY_PATH, help="Path to lora_registry.yaml")
    parser.add_argument("--name", type=str, required=True, help="LoRA name (registry key)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA .safetensors artifact")
    parser.add_argument("--type", type=str, default="character", help="LoRA type (character/style/background/pose)")

    parser.add_argument("--description", type=str, default="", help="Human readable description")
    parser.add_argument("--recommended-weight", type=float, default=0.8, help="Default LoRA weight")

    parser.add_argument("--trigger-word", action="append", help="Trigger word (repeatable)")
    parser.add_argument("--trigger-words", type=str, help="Comma-separated trigger words (alternative)")

    parser.add_argument("--metadata-json", type=str, help="JSON string for metadata")

    parser.add_argument("--stage-dir", type=str, help="Optional directory to copy/move artifact into")
    parser.add_argument("--stage-filename", type=str, help="Optional filename in stage dir")
    parser.add_argument("--move", action="store_true", help="Move artifact instead of copy")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite staged artifact if it exists")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    metadata: Optional[Dict[str, Any]] = None
    if args.metadata_json:
        metadata = json.loads(args.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must be a JSON object")

    result = upsert_lora_entry(
        registry_path=Path(args.registry),
        lora_name=args.name,
        lora_path=Path(args.checkpoint),
        lora_type=args.type,
        trigger_words=_parse_trigger_words(args),
        recommended_weight=args.recommended_weight,
        description=args.description,
        metadata=metadata,
        stage_to_dir=Path(args.stage_dir) if args.stage_dir else None,
        stage_filename=args.stage_filename,
        stage_move=args.move,
        stage_overwrite=args.overwrite,
    )

    action = "created" if result.created else "updated"
    logger.info(
        "Registry %s entry '%s' at index %s (artifact=%s)",
        action,
        result.lora_name,
        result.entry_index,
        result.artifact_path,
    )


if __name__ == "__main__":
    main()

