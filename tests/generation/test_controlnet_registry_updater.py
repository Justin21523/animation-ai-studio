"""
Unit Tests for ControlNet Registry Updater

Validates that trained ControlNet directories can be staged and the ControlNet registry
(`configs/generation/controlnet_config.yaml`) can be updated in a reproducible way.
"""

from pathlib import Path
import sys

import yaml


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.processing.training.controlnet_registry_updater import upsert_controlnet_entry


def _make_dummy_controlnet_dir(root: Path) -> Path:
    model_dir = root / "controlnet_dummy"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"dummy")
    return model_dir


def test_upsert_creates_new_entry(tmp_path: Path):
    registry_path = tmp_path / "controlnet_config.yaml"
    model_dir = _make_dummy_controlnet_dir(tmp_path)

    registry_path.write_text(
        yaml.safe_dump({"controlnet_models": {}}, sort_keys=False),
        encoding="utf-8",
    )

    result = upsert_controlnet_entry(
        registry_path=registry_path,
        controlnet_key="depth_game",
        model_dir=model_dir,
        model_id=None,
        description="Game depth ControlNet",
        use_case="Depth maps from engine",
        preprocess_as="depth",
    )

    assert result.created is True
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert "depth_game" in (data.get("controlnet_models") or {})
    assert data["controlnet_models"]["depth_game"]["local_path"] == str(model_dir)
    assert data["controlnet_models"]["depth_game"]["preprocess_as"] == "depth"


def test_upsert_updates_existing_entry(tmp_path: Path):
    registry_path = tmp_path / "controlnet_config.yaml"
    model_dir_a = _make_dummy_controlnet_dir(tmp_path / "a")
    model_dir_b = _make_dummy_controlnet_dir(tmp_path / "b")

    registry_path.write_text(
        yaml.safe_dump(
            {
                "controlnet_models": {
                    "canny_custom": {
                        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
                        "local_path": str(model_dir_a),
                        "description": "Old",
                        "use_case": "Old",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = upsert_controlnet_entry(
        registry_path=registry_path,
        controlnet_key="canny_custom",
        model_dir=model_dir_b,
        description="Updated",
        use_case="Updated use case",
        preprocess_as="canny",
    )

    assert result.created is False
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = data["controlnet_models"]["canny_custom"]
    assert entry["local_path"] == str(model_dir_b)
    assert entry["description"] == "Updated"
    assert entry["use_case"] == "Updated use case"
    assert entry["preprocess_as"] == "canny"


def test_stage_model_dir_updates_path(tmp_path: Path):
    registry_path = tmp_path / "controlnet_config.yaml"
    model_dir = _make_dummy_controlnet_dir(tmp_path / "src")
    stage_dir = tmp_path / "staged"

    registry_path.write_text(yaml.safe_dump({"controlnet_models": {}}, sort_keys=False), encoding="utf-8")

    result = upsert_controlnet_entry(
        registry_path=registry_path,
        controlnet_key="staged_model",
        model_dir=model_dir,
        stage_to_dir=stage_dir,
        stage_name="staged_model",
        stage_overwrite=True,
    )

    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = data["controlnet_models"]["staged_model"]
    assert Path(entry["local_path"]) == stage_dir / "staged_model"
    assert (stage_dir / "staged_model" / "config.json").exists()
    assert result.final_model_dir == stage_dir / "staged_model"

