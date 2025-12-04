"""
Preflight Checks Module

Comprehensive pre-execution safety checks before starting automation workflows.
Integrates GPU isolation, memory monitoring, disk space, and dependency verification.

Preflight Checklist:
  1. GPU Isolation: Verify CPU-only environment is properly configured
  2. Memory Safety: Check available RAM meets minimum requirements
  3. Disk Space: Ensure sufficient space for outputs and temp files
  4. Dependencies: Verify required Python packages are installed
  5. Environment: Check required environment variables are set
  6. Permissions: Verify write access to output directories

Usage:
  - Call run_preflight() before starting any automation workflow
  - Use strict=True (default) to raise errors on violations
  - Use strict=False to get detailed report without exiting

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import os
import sys
import shutil
import importlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import logging

# Import safety modules
from .gpu_isolation import (
    enforce_cpu_only,
    verify_no_gpu_usage,
    get_gpu_status,
    GPUIsolationError,
)
from .memory_monitor import (
    MemoryMonitor,
    MemoryThresholds,
    MemoryStats,
)

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PreflightResult:
    """Result of preflight checks."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format result as string."""
        lines = []
        lines.append("=" * 80)
        lines.append("Preflight Check Result")
        lines.append("=" * 80)

        if self.passed:
            lines.append("✓ Status: PASSED")
        else:
            lines.append("✗ Status: FAILED")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        if self.info:
            lines.append("\nInformation:")
            for key, value in self.info.items():
                lines.append(f"  • {key}: {value}")

        lines.append("=" * 80)
        return "\n".join(lines)


class PreflightError(Exception):
    """Raised when preflight checks fail in strict mode."""
    pass


# ============================================================================
# Preflight Checks
# ============================================================================

def check_gpu_isolation(result: PreflightResult) -> None:
    """
    Check GPU isolation is properly configured.

    Args:
        result: PreflightResult to update

    Modifies result in-place with errors/warnings/info.
    """
    logger.info("Checking GPU isolation...")

    try:
        # Enforce CPU-only environment
        previous_env = enforce_cpu_only(strict=False)

        # Verify no GPU usage
        is_safe, msg = verify_no_gpu_usage(raise_on_violation=False)

        if is_safe:
            logger.info("✓ GPU isolation verified")
            result.info['gpu_isolation'] = 'verified'
        else:
            logger.error(f"✗ GPU isolation violated: {msg}")
            result.errors.append(f"GPU isolation violated: {msg}")

        # Get GPU status for info
        gpu_status = get_gpu_status()
        result.info['cuda_visible_devices'] = gpu_status['cuda_visible_devices']
        result.info['torch_device'] = gpu_status['torch_device']

    except Exception as e:
        logger.error(f"✗ GPU isolation check failed: {e}")
        result.errors.append(f"GPU isolation check failed: {e}")


def check_memory_safety(
    result: PreflightResult,
    min_available_gb: float = 10.0
) -> None:
    """
    Check memory safety and availability.

    Args:
        result: PreflightResult to update
        min_available_gb: Minimum required available RAM (GB)

    Modifies result in-place with errors/warnings/info.
    """
    logger.info("Checking memory safety...")

    try:
        monitor = MemoryMonitor()

        # Get memory stats
        stats = monitor.get_memory_stats()

        # Check safety level
        is_safe, level, info = monitor.check_safety(force=True)

        # Record stats
        result.info['total_ram_gb'] = f"{stats.total_gb:.1f}"
        result.info['available_ram_gb'] = f"{stats.available_gb:.1f}"
        result.info['ram_percent_used'] = f"{stats.percent_used:.1f}%"
        result.info['memory_safety_level'] = level

        # Check minimum availability
        if stats.available_gb < min_available_gb:
            result.errors.append(
                f"Insufficient RAM: {stats.available_gb:.1f} GB < {min_available_gb:.1f} GB required"
            )

        # Check safety level
        if level == 'emergency':
            result.errors.append(
                f"Memory at emergency level ({stats.percent_used:.1f}%), cannot start"
            )
        elif level == 'critical':
            result.warnings.append(
                f"Memory at critical level ({stats.percent_used:.1f}%), reduce workload"
            )
        elif level == 'warning':
            result.warnings.append(
                f"Memory at warning level ({stats.percent_used:.1f}%), monitor closely"
            )
        else:
            logger.info(f"✓ Memory safety OK ({stats.percent_used:.1f}%)")

        # Check swap
        if stats.swap_percent > 20.0:
            result.warnings.append(
                f"High swap usage ({stats.swap_percent:.1f}%), may impact performance"
            )

    except Exception as e:
        logger.error(f"✗ Memory safety check failed: {e}")
        result.errors.append(f"Memory safety check failed: {e}")


def check_disk_space(
    result: PreflightResult,
    paths: Optional[List[Path]] = None,
    min_free_gb: float = 50.0
) -> None:
    """
    Check disk space availability.

    Args:
        result: PreflightResult to update
        paths: List of paths to check (defaults to /mnt/data and /mnt/c)
        min_free_gb: Minimum free space required (GB)

    Modifies result in-place with errors/warnings/info.
    """
    logger.info("Checking disk space...")

    if paths is None:
        paths = [
            Path('/mnt/data'),  # Datasets and training outputs
            Path('/mnt/c'),     # Models and projects
        ]

    for path in paths:
        try:
            if not path.exists():
                result.warnings.append(f"Path does not exist: {path}")
                continue

            # Get disk usage
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            percent_used = (usage.used / usage.total) * 100

            # Record info
            result.info[f'disk_{path.name}_free_gb'] = f"{free_gb:.1f}"
            result.info[f'disk_{path.name}_total_gb'] = f"{total_gb:.1f}"
            result.info[f'disk_{path.name}_percent_used'] = f"{percent_used:.1f}%"

            # Check minimum
            if free_gb < min_free_gb:
                result.errors.append(
                    f"Insufficient disk space on {path}: {free_gb:.1f} GB < {min_free_gb:.1f} GB required"
                )
            elif free_gb < min_free_gb * 2:
                result.warnings.append(
                    f"Low disk space on {path}: {free_gb:.1f} GB (warning threshold: {min_free_gb * 2:.1f} GB)"
                )
            else:
                logger.info(f"✓ Disk space OK on {path} ({free_gb:.1f} GB free)")

        except Exception as e:
            logger.error(f"✗ Disk space check failed for {path}: {e}")
            result.errors.append(f"Disk space check failed for {path}: {e}")


def check_dependencies(
    result: PreflightResult,
    required_packages: Optional[List[str]] = None
) -> None:
    """
    Check required Python packages are installed.

    Args:
        result: PreflightResult to update
        required_packages: List of package names (uses default list if None)

    Modifies result in-place with errors/warnings/info.
    """
    logger.info("Checking dependencies...")

    if required_packages is None:
        # Default required packages for CPU-only automation
        required_packages = [
            'psutil',       # Memory/CPU monitoring
            'yaml',         # Config files (PyYAML)
            'anthropic',    # Claude API
            'openai',       # OpenAI API (fallback)
            'numpy',        # Array operations
            'pillow',       # Image processing
            'moviepy',      # Video editing (CPU)
            'scenedetect',  # Scene detection (CPU)
        ]

    missing_packages = []
    installed_packages = []

    for package_name in required_packages:
        # Handle package import names that differ from pip names
        import_name = package_name
        if package_name == 'yaml':
            import_name = 'yaml'  # PyYAML imports as yaml
        elif package_name == 'pillow':
            import_name = 'PIL'  # Pillow imports as PIL
        elif package_name == 'scenedetect':
            import_name = 'scenedetect'

        try:
            importlib.import_module(import_name)
            installed_packages.append(package_name)
            logger.debug(f"✓ Package installed: {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            logger.warning(f"✗ Package missing: {package_name}")

    # Record results
    result.info['dependencies_installed'] = len(installed_packages)
    result.info['dependencies_total'] = len(required_packages)

    if missing_packages:
        result.errors.append(
            f"Missing required packages: {', '.join(missing_packages)}"
        )
        result.info['missing_packages'] = ', '.join(missing_packages)
    else:
        logger.info(f"✓ All {len(required_packages)} dependencies installed")


def check_environment_variables(result: PreflightResult) -> None:
    """
    Check required environment variables are set.

    Args:
        result: PreflightResult to update

    Modifies result in-place with errors/warnings/info.
    """
    logger.info("Checking environment variables...")

    required_vars = {
        'HF_HOME': '/mnt/c/ai_cache/huggingface',
        'TRANSFORMERS_CACHE': '/mnt/c/ai_cache/huggingface',
        'TORCH_HOME': '/mnt/c/ai_cache/torch',
        'XDG_CACHE_HOME': '/mnt/c/ai_cache',
    }

    for var_name, expected_value in required_vars.items():
        actual_value = os.environ.get(var_name)

        if actual_value is None:
            result.warnings.append(
                f"Environment variable not set: {var_name} (expected: {expected_value})"
            )
        elif actual_value != expected_value:
            result.warnings.append(
                f"Environment variable mismatch: {var_name}='{actual_value}' "
                f"(expected: '{expected_value}')"
            )
        else:
            logger.debug(f"✓ Environment variable OK: {var_name}")

        result.info[f'env_{var_name}'] = actual_value or '<not set>'


def check_output_permissions(
    result: PreflightResult,
    output_dirs: Optional[List[Path]] = None
) -> None:
    """
    Check write permissions for output directories.

    Args:
        result: PreflightResult to update
        output_dirs: List of output directories to check

    Modifies result in-place with errors/warnings/info.
    """
    logger.info("Checking output permissions...")

    if output_dirs is None:
        output_dirs = [
            Path('/mnt/data/datasets'),
            Path('/mnt/data/training'),
            Path('/mnt/data/tmp'),
        ]

    for output_dir in output_dirs:
        try:
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Test write permission
            test_file = output_dir / '.preflight_test'
            test_file.write_text('test')
            test_file.unlink()

            logger.debug(f"✓ Write permission OK: {output_dir}")
            result.info[f'writable_{output_dir.name}'] = 'yes'

        except Exception as e:
            logger.error(f"✗ Write permission failed for {output_dir}: {e}")
            result.errors.append(
                f"Cannot write to {output_dir}: {e}"
            )
            result.info[f'writable_{output_dir.name}'] = 'no'


# ============================================================================
# Main Preflight Function
# ============================================================================

def run_preflight(
    strict: bool = True,
    min_ram_gb: float = 10.0,
    min_disk_gb: float = 50.0,
    output_dirs: Optional[List[Path]] = None
) -> PreflightResult:
    """
    Run comprehensive preflight checks.

    Args:
        strict: If True, raises PreflightError on any errors.
                If False, returns result with errors listed.
        min_ram_gb: Minimum required available RAM (GB)
        min_disk_gb: Minimum required free disk space (GB)
        output_dirs: Optional list of output directories to check

    Returns:
        PreflightResult object

    Raises:
        PreflightError: If strict=True and any checks fail

    Example:
        >>> # Strict mode (raises on errors)
        >>> try:
        >>>     result = run_preflight()
        >>>     print("✓ All preflight checks passed")
        >>> except PreflightError as e:
        >>>     print(f"✗ Preflight failed: {e}")
        >>>
        >>> # Non-strict mode (returns report)
        >>> result = run_preflight(strict=False)
        >>> if not result.passed:
        >>>     print(result)
    """
    logger.info("Starting preflight checks...")

    result = PreflightResult(passed=True)

    # Run all checks
    check_gpu_isolation(result)
    check_memory_safety(result, min_available_gb=min_ram_gb)
    check_disk_space(result, min_free_gb=min_disk_gb)
    check_dependencies(result)
    check_environment_variables(result)
    check_output_permissions(result, output_dirs=output_dirs)

    # Determine overall pass/fail
    if result.errors:
        result.passed = False

    # Log summary
    if result.passed:
        logger.info(
            f"✓ Preflight checks passed "
            f"({len(result.warnings)} warnings, {len(result.errors)} errors)"
        )
    else:
        logger.error(
            f"✗ Preflight checks failed "
            f"({len(result.warnings)} warnings, {len(result.errors)} errors)"
        )

    # Raise error in strict mode
    if strict and not result.passed:
        error_msg = f"Preflight checks failed:\n" + "\n".join(
            f"  - {error}" for error in result.errors
        )
        raise PreflightError(error_msg)

    return result


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("Preflight Checks Module - Self Test")
    print("=" * 80)

    # Test 1: Run preflight in non-strict mode
    print("\n[Test 1] Running preflight checks (non-strict mode)...")
    try:
        result = run_preflight(strict=False)
        print(result)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Test 2: Run preflight in strict mode
    print("\n[Test 2] Running preflight checks (strict mode)...")
    try:
        result = run_preflight(strict=True)
        print("✓ All preflight checks passed")
    except PreflightError as e:
        print(f"✗ Preflight failed (expected if environment not fully configured):")
        print(f"  {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("\n" + "=" * 80)
    print("Self-test complete")
    print("=" * 80)
