"""
Test StructureAnalyzer

Unit tests for directory structure analysis.

Author: Animation AI Studio
Date: 2025-12-03
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.scenarios.file_organization.analyzers.structure_analyzer import StructureAnalyzer


def test_depth_calculation():
    """Test directory depth calculation"""
    print("Testing directory depth calculation...")

    analyzer = StructureAnalyzer()

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create nested structure
        level1 = root / "level1"
        level2 = level1 / "level2"
        level3 = level2 / "level3"

        level1.mkdir()
        level2.mkdir(parents=True)
        level3.mkdir(parents=True)

        # Test depth calculations
        tests = [
            (root, 0, "Root directory"),
            (level1, 1, "Level 1 directory"),
            (level2, 2, "Level 2 directory"),
            (level3, 3, "Level 3 directory"),
        ]

        passed = 0
        for path, expected_depth, desc in tests:
            depth = analyzer.calculate_depth(path, root)

            if depth == expected_depth:
                print(f"  ✓ {desc:30s} -> depth {depth}")
                passed += 1
            else:
                print(f"  ✗ {desc:30s} -> depth {depth} (expected {expected_depth})")

        print(f"\nDepth tests: {passed}/{len(tests)} passed")
        return passed == len(tests)


def test_project_detection():
    """Test project type detection"""
    print("\nTesting project type detection...")

    analyzer = StructureAnalyzer()

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create Python project structure
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "docs").mkdir()
        (root / "requirements.txt").write_text("numpy==1.0.0")
        (root / "setup.py").write_text("# setup")

        # Detect project type
        project_type = analyzer.detect_project_structure(root)

        if project_type == "python":
            print(f"  ✓ Correctly identified Python project")
            passed = 1
        else:
            print(f"  ✗ Failed to identify Python project (detected: {project_type})")
            passed = 0

        # Clean up and create ML project structure
        for item in root.iterdir():
            if item.is_dir():
                for subitem in item.iterdir():
                    subitem.unlink()
                item.rmdir()
            else:
                item.unlink()

        (root / "data").mkdir()
        (root / "models").mkdir()
        (root / "notebooks").mkdir()
        (root / "scripts").mkdir()
        (root / "requirements.txt").write_text("torch==2.0.0")

        project_type = analyzer.detect_project_structure(root)

        if project_type == "ml_project":
            print(f"  ✓ Correctly identified ML project")
            passed += 1
        else:
            print(f"  ✗ Failed to identify ML project (detected: {project_type})")

        print(f"\nProject detection tests: {passed}/2 passed")
        return passed == 2


def test_orphaned_directory_detection():
    """Test orphaned directory detection"""
    print("\nTesting orphaned directory detection...")

    analyzer = StructureAnalyzer(min_files_per_dir=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create empty directory
        empty_dir = root / "empty"
        empty_dir.mkdir()

        # Create directory with one file (below threshold)
        sparse_dir = root / "sparse"
        sparse_dir.mkdir()
        (sparse_dir / "file1.txt").write_text("content")

        # Create directory with enough files
        normal_dir = root / "normal"
        normal_dir.mkdir()
        (normal_dir / "file1.txt").write_text("content")
        (normal_dir / "file2.txt").write_text("content")
        (normal_dir / "file3.txt").write_text("content")

        # Find orphaned directories
        orphaned = analyzer.find_orphaned_directories(root)

        passed = 0

        # Should find empty and sparse dirs
        orphaned_names = {d.name for d in orphaned}

        if "empty" in orphaned_names:
            print(f"  ✓ Detected empty directory")
            passed += 1
        else:
            print(f"  ✗ Failed to detect empty directory")

        if "sparse" in orphaned_names:
            print(f"  ✓ Detected sparse directory")
            passed += 1
        else:
            print(f"  ✗ Failed to detect sparse directory")

        if "normal" not in orphaned_names:
            print(f"  ✓ Did not flag normal directory")
            passed += 1
        else:
            print(f"  ✗ Incorrectly flagged normal directory")

        print(f"\nOrphaned detection tests: {passed}/3 passed")
        return passed == 3


def test_structure_analysis():
    """Test complete structure analysis"""
    print("\nTesting complete structure analysis...")

    analyzer = StructureAnalyzer(
        max_depth=10,
        excessive_nesting_threshold=3
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create test structure
        (root / "dir1").mkdir()
        (root / "dir1" / "subdir1").mkdir()
        (root / "dir1" / "subdir1" / "subsubdir1").mkdir()
        (root / "dir1" / "subdir1" / "subsubdir1" / "deep1").mkdir()
        (root / "dir1" / "subdir1" / "subsubdir1" / "deep1" / "deep2").mkdir()

        (root / "dir2").mkdir()
        (root / "dir2" / "file1.txt").write_text("content")

        # Run analysis
        analysis, issues = analyzer.analyze_structure(root)

        passed = 0

        # Check analysis results
        if analysis.total_directories > 0:
            print(f"  ✓ Detected {analysis.total_directories} directories")
            passed += 1
        else:
            print(f"  ✗ Failed to count directories")

        if analysis.max_depth >= 5:
            print(f"  ✓ Calculated max depth: {analysis.max_depth}")
            passed += 1
        else:
            print(f"  ✗ Incorrect max depth: {analysis.max_depth}")

        if analysis.avg_depth > 0:
            print(f"  ✓ Calculated average depth: {analysis.avg_depth:.1f}")
            passed += 1
        else:
            print(f"  ✗ Failed to calculate average depth")

        # Should detect excessive nesting
        has_nesting_issue = any(
            issue.category.value == "nested_excessive"
            for issue in issues
        )

        if has_nesting_issue:
            print(f"  ✓ Detected excessive nesting")
            passed += 1
        else:
            print(f"  ✗ Failed to detect excessive nesting")

        print(f"\nStructure analysis tests: {passed}/4 passed")
        return passed == 4


def run_all_tests():
    """Run all StructureAnalyzer tests"""
    print("=" * 70)
    print("STRUCTUREANALYZER UNIT TESTS")
    print("=" * 70)

    results = []

    results.append(("Depth Calculation", test_depth_calculation()))
    results.append(("Project Detection", test_project_detection()))
    results.append(("Orphaned Directory Detection", test_orphaned_directory_detection()))
    results.append(("Structure Analysis", test_structure_analysis()))

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
