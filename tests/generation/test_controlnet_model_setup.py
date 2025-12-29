"""
Unit Tests for ControlNet preprocessor model setup utility.

Verifies local-first resolution via HuggingFace cache snapshots and YAML updates.
"""

from pathlib import Path
import sys

import yaml


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.processing.controlnet.model_setup import update_local_path, UpdateRequest


def test_update_local_path_uses_cached_snapshot(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.delenv("HF_HOME", raising=False)

    cfg_path = tmp_path / "controlnet_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {
                    "pose": {
                        "model_id": "org/repo",
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

    resolved = update_local_path(
        cfg_path,
        UpdateRequest(
            key="pose",
            model_id="org/repo",
            dest_dir=None,
            allow_download=False,
        ),
    )

    assert resolved == snapshot
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert data["preprocessing"]["pose"]["local_path"] == str(snapshot)

