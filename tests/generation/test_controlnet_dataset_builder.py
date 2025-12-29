"""
Unit Tests for ControlNet Dataset Builder

Uses precomputed control images (no cv2 dependency) to verify on-disk dataset structure
and metadata generation.
"""

from pathlib import Path
import sys

from PIL import Image


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.processing.training.controlnet_dataset_builder import BuildConfig, build_dataset


def _write_rgb_image(path: Path, size: int, color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (size, size), color=color)
    img.save(path, format="PNG")


def test_build_dataset_with_precomputed_controls(tmp_path: Path):
    images_dir = tmp_path / "images"
    controls_dir = tmp_path / "controls"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    controls_dir.mkdir()

    _write_rgb_image(images_dir / "a.png", 64, (255, 0, 0))
    _write_rgb_image(images_dir / "b.png", 64, (0, 255, 0))
    _write_rgb_image(controls_dir / "a.png", 64, (0, 0, 0))
    _write_rgb_image(controls_dir / "b.png", 64, (255, 255, 255))
    (images_dir / "a.txt").write_text("caption a", encoding="utf-8")
    (images_dir / "b.txt").write_text("caption b", encoding="utf-8")

    meta = build_dataset(
        BuildConfig(
            images_dir=images_dir,
            output_dir=out_dir,
            control_type="depth",
            control_images_dir=controls_dir,
            resolution=32,
            overwrite=False,
        )
    )

    assert (out_dir / "images" / "000000.png").exists()
    assert (out_dir / "conditioning" / "000000.png").exists()
    assert (out_dir / "captions" / "000000.txt").read_text(encoding="utf-8") == "caption a"
    assert meta["stats"]["written"] == 2
    assert (out_dir / "dataset_metadata.json").exists()


def test_overwrite_requires_flag(tmp_path: Path):
    images_dir = tmp_path / "images"
    controls_dir = tmp_path / "controls"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    controls_dir.mkdir()

    _write_rgb_image(images_dir / "a.png", 64, (255, 0, 0))
    _write_rgb_image(controls_dir / "a.png", 64, (0, 0, 0))

    build_dataset(
        BuildConfig(
            images_dir=images_dir,
            output_dir=out_dir,
            control_type="depth",
            control_images_dir=controls_dir,
            resolution=32,
            overwrite=False,
        )
    )

    try:
        build_dataset(
            BuildConfig(
                images_dir=images_dir,
                output_dir=out_dir,
                control_type="depth",
                control_images_dir=controls_dir,
                resolution=32,
                overwrite=False,
            )
        )
        assert False, "Expected FileExistsError"
    except FileExistsError:
        pass

    build_dataset(
        BuildConfig(
            images_dir=images_dir,
            output_dir=out_dir,
            control_type="depth",
            control_images_dir=controls_dir,
            resolution=32,
            overwrite=True,
        )
    )

