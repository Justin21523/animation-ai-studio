"""
Unit Tests for LoRA Registry Updater

Validates that training artifacts can be staged and the LoRA registry can be updated
in a reproducible way.
"""

from pathlib import Path
import sys

import yaml


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.processing.training.lora_registry_updater import upsert_lora_entry


def test_upsert_creates_new_entry(tmp_path: Path):
    registry_path = tmp_path / "lora_registry.yaml"
    artifact_path = tmp_path / "example.safetensors"
    artifact_path.write_bytes(b"dummy")

    registry_path.write_text(
        yaml.safe_dump(
            {
                "loras": [],
                "combinations": {"demo": {"loras": [{"name": "base", "weight": 1.0}]}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = upsert_lora_entry(
        registry_path=registry_path,
        lora_name="new_character",
        lora_path=artifact_path,
        lora_type="character",
        trigger_words=["new_character", "test token"],
        recommended_weight=0.75,
        description="Test LoRA",
        metadata={"training_steps": 123},
    )

    assert result.created is True
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "new_character" for entry in data["loras"])
    assert data["combinations"]["demo"]["loras"][0]["name"] == "base"


def test_upsert_updates_existing_entry(tmp_path: Path):
    registry_path = tmp_path / "lora_registry.yaml"
    old_artifact = tmp_path / "old.safetensors"
    new_artifact = tmp_path / "new.safetensors"
    old_artifact.write_bytes(b"old")
    new_artifact.write_bytes(b"new")

    registry_path.write_text(
        yaml.safe_dump(
            {
                "loras": [
                    {
                        "name": "luca",
                        "path": str(old_artifact),
                        "type": "character",
                        "trigger_words": ["luca"],
                        "recommended_weight": 0.8,
                        "description": "Old",
                        "metadata": {"training_steps": 1},
                    }
                ],
                "combinations": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = upsert_lora_entry(
        registry_path=registry_path,
        lora_name="luca",
        lora_path=new_artifact,
        lora_type="character",
        trigger_words=["luca", "updated"],
        recommended_weight=0.9,
        description="Updated",
        metadata={"training_steps": 999},
    )

    assert result.created is False
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    luca = next(entry for entry in data["loras"] if entry.get("name") == "luca")
    assert luca["path"] == str(new_artifact)
    assert luca["recommended_weight"] == 0.9
    assert "updated" in luca["trigger_words"]


def test_stage_artifact_updates_path(tmp_path: Path):
    registry_path = tmp_path / "lora_registry.yaml"
    artifact_path = tmp_path / "ckpt.safetensors"
    stage_dir = tmp_path / "staged"
    artifact_path.write_bytes(b"dummy")

    registry_path.write_text(
        yaml.safe_dump({"loras": [], "combinations": {}}, sort_keys=False),
        encoding="utf-8",
    )

    result = upsert_lora_entry(
        registry_path=registry_path,
        lora_name="staged_lora",
        lora_path=artifact_path,
        stage_to_dir=stage_dir,
        stage_filename="staged_lora.safetensors",
    )

    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = next(entry for entry in data["loras"] if entry.get("name") == "staged_lora")
    assert Path(entry["path"]).name == "staged_lora.safetensors"
    assert (stage_dir / "staged_lora.safetensors").exists()
    assert result.artifact_path == stage_dir / "staged_lora.safetensors"

