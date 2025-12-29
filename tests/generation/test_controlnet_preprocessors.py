"""
Unit Tests for ControlNet preprocessors (local-first behavior).

These tests avoid requiring heavy optional dependencies unless they are already installed.
"""

from pathlib import Path
import sys

import pytest
from PIL import Image
import yaml


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.processing.controlnet.preprocessors import (
    ControlNetPreprocessorFactory,
    PreprocessorContext,
    PreprocessorUnavailableError,
)


def test_pose_requires_local_path_when_download_disabled(tmp_path: Path):
    cfg_path = tmp_path / "controlnet_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {
                    "pose": {
                        "model_id": "lllyasviel/ControlNet",
                        "local_path": None,
                        "hand": False,
                        "face": False,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    factory = ControlNetPreprocessorFactory(
        PreprocessorContext(
            controlnet_config_path=str(cfg_path),
            prefer_local_models=True,
            allow_download=False,
        )
    )

    with pytest.raises(PreprocessorUnavailableError) as exc:
        factory.get("pose")

    assert "preprocessing.pose.local_path" in str(exc.value)


def test_depth_requires_local_path_when_download_disabled(tmp_path: Path):
    cfg_path = tmp_path / "controlnet_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {
                    "depth": {
                        "model_id": "Intel/dpt-hybrid-midas",
                        "local_path": None,
                        "device": "cpu",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    factory = ControlNetPreprocessorFactory(
        PreprocessorContext(
            controlnet_config_path=str(cfg_path),
            prefer_local_models=True,
            allow_download=False,
            device="cpu",
        )
    )

    with pytest.raises(PreprocessorUnavailableError) as exc:
        factory.get("depth")

    assert "preprocessing.depth.local_path" in str(exc.value)


def test_seg_preprocessor_runs_if_rembg_available(tmp_path: Path):
    cfg_path = tmp_path / "controlnet_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"preprocessing": {"seg": {"backend": "rembg", "model": "isnet-general-use"}}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    factory = ControlNetPreprocessorFactory(
        PreprocessorContext(
            controlnet_config_path=str(cfg_path),
            prefer_local_models=True,
            allow_download=False,
            device="cpu",
        )
    )

    try:
        pre = factory.get("seg")
    except PreprocessorUnavailableError as e:
        pytest.skip(str(e))
    img = Image.new("RGB", (32, 32), color=(120, 200, 80))
    out = pre(img, detect_resolution=32, image_resolution=32)
    assert out.size == (32, 32)
    assert out.mode == "RGB"
