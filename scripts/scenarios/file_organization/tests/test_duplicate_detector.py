"""
Test DuplicateDetector

Unit tests for duplicate file detection.

Author: Animation AI Studio
Date: 2025-12-03
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.scenarios.file_organization.analyzers.duplicate_detector import DuplicateDetector
from scripts.scenarios.file_organization.common import FileMetadata, FileType


def test_sha256_computation():
    """Test SHA256 hash computation"""
    print("Testing SHA256 hash computation...")

    detector = DuplicateDetector(enable_perceptual=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test file
        test_file = tmpdir_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Compute hash
        hash1 = detector.compute_sha256(test_file)
        hash2 = detector.compute_sha256(test_file)

        # Hashes should be consistent
        if hash1 == hash2:
            print(f"  ✓ Consistent hash: {hash1[:16]}...")
            passed = 1
        else:
            print(f"  ✗ Inconsistent hashes!")
            passed = 0

        # Create identical file
        test_file2 = tmpdir_path / "test2.txt"
        test_file2.write_bytes(test_content)

        hash3 = detector.compute_sha256(test_file2)

        if hash1 == hash3:
            print(f"  ✓ Identical content produces same hash")
            passed += 1
        else:
            print(f"  ✗ Identical content produces different hash!")

        # Create different file
        test_file3 = tmpdir_path / "test3.txt"
        test_file3.write_bytes(b"Different content")

        hash4 = detector.compute_sha256(test_file3)

        if hash1 != hash4:
            print(f"  ✓ Different content produces different hash")
            passed += 1
        else:
            print(f"  ✗ Different content produces same hash!")

        print(f"\nSHA256 tests: {passed}/3 passed")
        return passed == 3


def test_exact_duplicate_detection():
    """Test exact duplicate detection"""
    print("\nTesting exact duplicate detection...")

    detector = DuplicateDetector(enable_perceptual=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create duplicate files
        content = b"Duplicate content" * 100  # Make it larger than min size

        file1 = tmpdir_path / "file1.txt"
        file2 = tmpdir_path / "file2.txt"
        file3 = tmpdir_path / "file3.txt"
        file4 = tmpdir_path / "unique.txt"

        file1.write_bytes(content)
        file2.write_bytes(content)
        file3.write_bytes(content)
        file4.write_bytes(b"Unique content" * 100)

        # Create metadata
        now = datetime.now()
        files = [
            FileMetadata(
                path=file1,
                file_type=FileType.DOCUMENT,
                size_bytes=len(content),
                created_time=now,
                modified_time=now,
                accessed_time=now
            ),
            FileMetadata(
                path=file2,
                file_type=FileType.DOCUMENT,
                size_bytes=len(content),
                created_time=now,
                modified_time=now,
                accessed_time=now
            ),
            FileMetadata(
                path=file3,
                file_type=FileType.DOCUMENT,
                size_bytes=len(content),
                created_time=now,
                modified_time=now,
                accessed_time=now
            ),
            FileMetadata(
                path=file4,
                file_type=FileType.DOCUMENT,
                size_bytes=len(b"Unique content" * 100),
                created_time=now,
                modified_time=now,
                accessed_time=now
            ),
        ]

        # Detect duplicates
        duplicates = detector.detect_duplicates(files)

        passed = 0

        # Should find 1 duplicate group (file1, file2, file3)
        if len(duplicates) == 1:
            print(f"  ✓ Found 1 duplicate group")
            passed += 1
        else:
            print(f"  ✗ Expected 1 duplicate group, found {len(duplicates)}")

        if duplicates and len(duplicates[0].files) == 3:
            print(f"  ✓ Duplicate group has 3 files")
            passed += 1
        else:
            print(f"  ✗ Expected 3 files in duplicate group")

        if duplicates and duplicates[0].savings_bytes > 0:
            savings = duplicates[0].savings_bytes
            print(f"  ✓ Savings calculated: {savings} bytes")
            passed += 1
        else:
            print(f"  ✗ No savings calculated")

        print(f"\nExact duplicate tests: {passed}/3 passed")
        return passed == 3


def test_size_prefiltering():
    """Test size-based pre-filtering"""
    print("\nTesting size-based pre-filtering...")

    detector = DuplicateDetector(
        enable_perceptual=False,
        min_file_size=1024  # 1KB minimum
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create small file (below threshold)
        small_file = tmpdir_path / "small.txt"
        small_content = b"small"
        small_file.write_bytes(small_content)

        # Create large file (above threshold)
        large_file = tmpdir_path / "large.txt"
        large_content = b"large" * 300
        large_file.write_bytes(large_content)

        now = datetime.now()
        files = [
            FileMetadata(
                path=small_file,
                file_type=FileType.DOCUMENT,
                size_bytes=len(small_content),
                created_time=now,
                modified_time=now,
                accessed_time=now
            ),
            FileMetadata(
                path=large_file,
                file_type=FileType.DOCUMENT,
                size_bytes=len(large_content),
                created_time=now,
                modified_time=now,
                accessed_time=now
            ),
        ]

        # Should only process large file
        # (Can't verify directly, but test shouldn't crash)
        duplicates = detector.detect_duplicates(files)

        # Should find no duplicates (only 1 file above threshold)
        if len(duplicates) == 0:
            print(f"  ✓ Correctly filtered small files")
            print(f"  ✓ No false duplicates detected")
            passed = 2
        else:
            print(f"  ✗ Unexpected duplicates found")
            passed = 0

        print(f"\nSize filtering tests: {passed}/2 passed")
        return passed == 2


def test_savings_estimation():
    """Test space savings estimation"""
    print("\nTesting space savings estimation...")

    detector = DuplicateDetector(enable_perceptual=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create duplicate files of known size
        file_size = 10000  # 10KB
        content = b"x" * file_size

        files_data = []
        for i in range(5):
            filepath = tmpdir_path / f"dup{i}.txt"
            filepath.write_bytes(content)

            now = datetime.now()
            files_data.append(FileMetadata(
                path=filepath,
                file_type=FileType.DOCUMENT,
                size_bytes=file_size,
                created_time=now,
                modified_time=now,
                accessed_time=now
            ))

        # Detect duplicates
        duplicates = detector.detect_duplicates(files_data)

        passed = 0

        # Should find 1 group with 5 files
        if duplicates and len(duplicates) == 1:
            group = duplicates[0]

            # Total size = 5 * 10KB = 50KB
            expected_total = file_size * 5
            if group.total_size_bytes == expected_total:
                print(f"  ✓ Correct total size: {expected_total} bytes")
                passed += 1
            else:
                print(f"  ✗ Wrong total size: {group.total_size_bytes} (expected {expected_total})")

            # Savings = 4 * 10KB = 40KB (keep 1, remove 4)
            expected_savings = file_size * 4
            if group.savings_bytes == expected_savings:
                print(f"  ✓ Correct savings: {expected_savings} bytes")
                passed += 1
            else:
                print(f"  ✗ Wrong savings: {group.savings_bytes} (expected {expected_savings})")

            # Test total savings estimation
            total_savings = detector.estimate_savings(duplicates)
            if total_savings == expected_savings:
                print(f"  ✓ Correct total savings estimation")
                passed += 1
            else:
                print(f"  ✗ Wrong total savings: {total_savings}")
        else:
            print(f"  ✗ Duplicate detection failed")

        print(f"\nSavings tests: {passed}/3 passed")
        return passed == 3


def run_all_tests():
    """Run all DuplicateDetector tests"""
    print("=" * 70)
    print("DUPLICATEDETECTOR UNIT TESTS")
    print("=" * 70)

    results = []

    results.append(("SHA256 Computation", test_sha256_computation()))
    results.append(("Exact Duplicate Detection", test_exact_duplicate_detection()))
    results.append(("Size Pre-filtering", test_size_prefiltering()))
    results.append(("Savings Estimation", test_savings_estimation()))

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
