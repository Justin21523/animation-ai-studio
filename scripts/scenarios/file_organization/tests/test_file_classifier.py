"""
Test FileClassifier

Unit tests for file type classification.

Author: Animation AI Studio
Date: 2025-12-03
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.scenarios.file_organization.analyzers.file_classifier import FileClassifier
from scripts.scenarios.file_organization.common import FileType


def test_extension_classification():
    """Test classification by file extension"""
    print("Testing extension-based classification...")

    classifier = FileClassifier()

    test_cases = [
        ("test.jpg", FileType.IMAGE),
        ("video.mp4", FileType.VIDEO),
        ("audio.mp3", FileType.AUDIO),
        ("document.pdf", FileType.DOCUMENT),
        ("script.py", FileType.CODE),
        ("archive.zip", FileType.ARCHIVE),
        ("model.safetensors", FileType.MODEL),
        ("config.yaml", FileType.CODE),
        ("app.log", FileType.LOG),
        ("unknown.xyz", FileType.OTHER),
    ]

    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for filename, expected_type in test_cases:
            # Create dummy file
            filepath = tmpdir_path / filename
            filepath.write_bytes(b"test content")

            # Classify
            result = classifier.classify_file(filepath)

            if result == expected_type:
                print(f"  ✓ {filename:25s} -> {result.value:10s}")
                passed += 1
            else:
                print(f"  ✗ {filename:25s} -> {result.value:10s} (expected {expected_type.value})")
                failed += 1

    print(f"\nExtension tests: {passed} passed, {failed} failed")
    return failed == 0


def test_ai_model_detection():
    """Test AI model file detection"""
    print("\nTesting AI model detection...")

    classifier = FileClassifier()

    test_cases = [
        ("character_lora.safetensors", True),
        ("model_checkpoint.pt", True),
        ("weights.pth", True),
        ("trained_model.ckpt", True),
        ("network.onnx", True),
        ("regular_file.txt", False),
        ("not_a_model.jpg", False),
    ]

    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for filename, should_be_model in test_cases:
            filepath = tmpdir_path / filename
            filepath.write_bytes(b"test content")

            result = classifier.is_ai_model(filepath)

            if result == should_be_model:
                print(f"  ✓ {filename:30s} -> {'MODEL' if result else 'NOT MODEL':10s}")
                passed += 1
            else:
                print(f"  ✗ {filename:30s} -> {'MODEL' if result else 'NOT MODEL':10s} (expected {'MODEL' if should_be_model else 'NOT MODEL'})")
                failed += 1

    print(f"\nAI model tests: {passed} passed, {failed} failed")
    return failed == 0


def test_dataset_detection():
    """Test dataset directory detection"""
    print("\nTesting dataset directory detection...")

    classifier = FileClassifier()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create dataset-like structure
        dataset_dir = tmpdir_path / "training_data"
        dataset_dir.mkdir()

        # Add images and labels
        (dataset_dir / "image1.jpg").write_bytes(b"fake image")
        (dataset_dir / "image2.jpg").write_bytes(b"fake image")
        (dataset_dir / "label1.txt").write_bytes(b"label")
        (dataset_dir / "label2.txt").write_bytes(b"label")

        # Test dataset detection
        is_dataset = classifier.is_dataset(dataset_dir)

        if is_dataset:
            print(f"  ✓ Dataset directory correctly identified")
            passed = 1
        else:
            print(f"  ✗ Dataset directory NOT identified")
            passed = 0

        # Create non-dataset directory
        regular_dir = tmpdir_path / "regular_folder"
        regular_dir.mkdir()
        (regular_dir / "file1.txt").write_bytes(b"content")

        is_not_dataset = not classifier.is_dataset(regular_dir)

        if is_not_dataset:
            print(f"  ✓ Regular directory correctly NOT identified as dataset")
            passed += 1
        else:
            print(f"  ✗ Regular directory incorrectly identified as dataset")

        total = 2
        failed = total - passed

        print(f"\nDataset tests: {passed}/{total} passed")
        return failed == 0


def test_supported_extensions():
    """Test getting supported extensions"""
    print("\nTesting supported extensions query...")

    classifier = FileClassifier()

    # Test a few file types
    image_exts = classifier.get_supported_extensions(FileType.IMAGE)
    video_exts = classifier.get_supported_extensions(FileType.VIDEO)
    model_exts = classifier.get_supported_extensions(FileType.MODEL)

    checks = [
        ('.jpg' in image_exts, "IMAGE extensions include .jpg"),
        ('.png' in image_exts, "IMAGE extensions include .png"),
        ('.mp4' in video_exts, "VIDEO extensions include .mp4"),
        ('.safetensors' in model_exts, "MODEL extensions include .safetensors"),
    ]

    passed = sum(1 for check, _ in checks if check)
    failed = len(checks) - passed

    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")

    print(f"\nExtensions tests: {passed}/{len(checks)} passed")
    return failed == 0


def run_all_tests():
    """Run all FileClassifier tests"""
    print("=" * 70)
    print("FILECLASSIFIER UNIT TESTS")
    print("=" * 70)

    results = []

    results.append(("Extension Classification", test_extension_classification()))
    results.append(("AI Model Detection", test_ai_model_detection()))
    results.append(("Dataset Detection", test_dataset_detection()))
    results.append(("Supported Extensions", test_supported_extensions()))

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:40s} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} test suites passed")
    print("=" * 70)

    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
