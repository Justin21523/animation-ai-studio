#!/usr/bin/env python3
"""
Master Test Runner - Animation AI Studio

Runs all tests across all modules with comprehensive reporting.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
    python tests/run_all_tests.py --module agent
    python tests/run_all_tests.py --coverage

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Master test runner for all modules"""

    def __init__(self, verbose=False, coverage=False):
        self.verbose = verbose
        self.coverage = coverage
        self.project_root = project_root
        self.results = {}

    def run_tests(self, module_name=None):
        """Run tests for specified module or all modules"""
        print("=" * 70)
        print("🧪 ANIMATION AI STUDIO - TEST SUITE")
        print("=" * 70)
        print(f"Project Root: {self.project_root}")
        print(f"Verbose: {self.verbose}")
        print(f"Coverage: {self.coverage}")
        print("=" * 70)

        if module_name:
            self._run_module_tests(module_name)
        else:
            self._run_all_module_tests()

        self._print_summary()

    def _run_all_module_tests(self):
        """Run tests for all modules"""
        modules = [
            ("Agent Framework", "scripts/agent/test_agent_phase2.py"),
            ("Video Editing", "scripts/editing/tests/test_module8.py"),
            ("Creative Studio", "scripts/applications/creative_studio/tests/test_creative_studio.py")
        ]

        for name, test_path in modules:
            self._run_test_file(name, test_path)

    def _run_module_tests(self, module_name):
        """Run tests for specific module"""
        module_map = {
            "agent": ("Agent Framework", "scripts/agent/test_agent_phase2.py"),
            "editing": ("Video Editing", "scripts/editing/tests/test_module8.py"),
            "creative": ("Creative Studio", "scripts/applications/creative_studio/tests/test_creative_studio.py")
        }

        if module_name not in module_map:
            print(f"❌ Unknown module: {module_name}")
            print(f"Available modules: {', '.join(module_map.keys())}")
            return

        name, test_path = module_map[module_name]
        self._run_test_file(name, test_path)

    def _run_test_file(self, name, test_path):
        """Run a specific test file"""
        print(f"\n{'=' * 70}")
        print(f"🧪 Testing: {name}")
        print(f"{'=' * 70}")

        full_path = self.project_root / test_path

        if not full_path.exists():
            print(f"⚠️  Test file not found: {full_path}")
            self.results[name] = {"status": "SKIP", "reason": "File not found"}
            return

        start_time = time.time()

        try:
            cmd = [sys.executable, str(full_path)]

            if self.coverage:
                cmd = ["pytest", str(full_path), "--cov", "-v"]
            elif self.verbose:
                cmd = ["pytest", str(full_path), "-v"]

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ PASSED ({elapsed:.1f}s)")
                self.results[name] = {"status": "PASS", "time": elapsed}
            else:
                print(f"❌ FAILED ({elapsed:.1f}s)")
                print("\nSTDOUT:")
                print(result.stdout)
                print("\nSTDERR:")
                print(result.stderr)
                self.results[name] = {"status": "FAIL", "time": elapsed}

        except subprocess.TimeoutExpired:
            print(f"⏱️  TIMEOUT (>300s)")
            self.results[name] = {"status": "TIMEOUT", "time": 300}

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results[name] = {"status": "ERROR", "error": str(e)}

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results.values() if r["status"] == "SKIP")
        errors = sum(1 for r in self.results.values() if r["status"] == "ERROR")
        timeouts = sum(1 for r in self.results.values() if r["status"] == "TIMEOUT")

        for name, result in self.results.items():
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌",
                "SKIP": "⚠️ ",
                "ERROR": "💥",
                "TIMEOUT": "⏱️ "
            }.get(result["status"], "❓")

            time_str = f"({result.get('time', 0):.1f}s)" if "time" in result else ""
            print(f"{status_icon} {name}: {result['status']} {time_str}")

        print("\n" + "-" * 70)
        print(f"Total: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Skipped: {skipped} ⚠️")
        print(f"Errors: {errors} 💥")
        print(f"Timeouts: {timeouts} ⏱️")
        print("=" * 70)

        if passed == total:
            print("\n🎉 ALL TESTS PASSED! 🎉\n")
            return 0
        else:
            print(f"\n⚠️  {failed + errors + timeouts} TESTS DID NOT PASS\n")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Animation AI Studio test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting (requires pytest-cov)"
    )

    parser.add_argument(
        "--module", "-m",
        choices=["agent", "editing", "creative"],
        help="Run tests for specific module only"
    )

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, coverage=args.coverage)
    exit_code = runner.run_tests(module_name=args.module)

    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
