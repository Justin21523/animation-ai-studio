"""
Run All File Organization Tests

Master test runner for all file organization unit tests.

Author: Animation AI Studio
Date: 2025-12-03
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.scenarios.file_organization.tests.test_file_classifier import run_all_tests as test_classifier
from scripts.scenarios.file_organization.tests.test_duplicate_detector import run_all_tests as test_duplicate
from scripts.scenarios.file_organization.tests.test_structure_analyzer import run_all_tests as test_structure


def main():
    """Run all test suites"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FILE ORGANIZATION SCENARIO - COMPREHENSIVE TEST SUITE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    results = []

    # Run all test suites
    print("Running FileClassifier tests...")
    print("-" * 70)
    results.append(("FileClassifier", test_classifier()))
    print("\n\n")

    print("Running DuplicateDetector tests...")
    print("-" * 70)
    results.append(("DuplicateDetector", test_duplicate()))
    print("\n\n")

    print("Running StructureAnalyzer tests...")
    print("-" * 70)
    results.append(("StructureAnalyzer", test_structure()))
    print("\n\n")

    # Final summary
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FINAL TEST SUMMARY".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    for suite_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        status_color = "\033[92m" if passed else "\033[91m"  # Green or Red
        reset_color = "\033[0m"

        print(f"  {suite_name:30s} {status_color}{status}{reset_color}")

    print()

    total_passed = sum(1 for _, passed in results if passed)
    total_suites = len(results)

    if total_passed == total_suites:
        print(f"  \033[92m✓ All {total_suites} test suites PASSED!\033[0m")
        print()
        print("  " + "🎉" * 20)
        print()
        exit_code = 0
    else:
        failed = total_suites - total_passed
        print(f"  \033[91m✗ {failed}/{total_suites} test suites FAILED\033[0m")
        exit_code = 1

    print("=" * 70)
    print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
