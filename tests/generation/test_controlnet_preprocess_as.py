"""
Unit Tests for ControlNet preprocess_as resolution

Ensures custom registry keys can still reuse built-in preprocessing behavior.
"""

from pathlib import Path
import sys

import torch
import yaml


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image.controlnet_pipeline import ControlNetPipelineManager


def test_preprocess_as_is_applied(tmp_path: Path):
    cfg_path = tmp_path / "controlnet_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "controlnet_models": {
                    "my_custom_depth": {
                        "model_id": "dummy/controlnet",
                        "local_path": None,
                        "preprocess_as": "depth",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manager = ControlNetPipelineManager(
        sdxl_model_path="dummy.safetensors",
        base_repo_path=None,
        control_type="my_custom_depth",
        controlnet_model_path=None,
        controlnet_config_path=str(cfg_path),
        prefer_local_path=True,
        device="cpu",
        dtype=torch.float32,
        use_sdpa=True,
    )

    assert manager.preprocess_type == "depth"

