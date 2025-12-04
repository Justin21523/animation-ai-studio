"""
Memory Monitor Module

Provides real-time memory monitoring and safety checks for CPU-only automation.
Integrates with existing system-level memory watchdog (80%/90% thresholds) by
implementing more conservative thresholds (70%/80%/85%) for early intervention.

Memory Management Strategy:
  - 70% threshold: Warning - reduce batch size by 50%
  - 80% threshold: Critical - minimal batch, enable streaming mode
  - 85% threshold: Emergency - save checkpoint and exit gracefully
  - Reserve 18GB for system at all times

Integration with Auto_Protection_Guide.txt:
  - System watchdog: 80% warning, 90% emergency
  - This module: 70%/80%/85% for proactive management
  - Automation OOM priority: +500 (killable before training -300)

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MemoryStats:
    """Memory statistics snapshot."""
    total_gb: float
    used_gb: float
    available_gb: float
    percent_used: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    timestamp: float


@dataclass
class MemoryThresholds:
    """Memory threshold configuration."""
    warning_percent: float = 70.0    # Reduce batch size
    critical_percent: float = 80.0   # Minimal batch, streaming
    emergency_percent: float = 85.0  # Save and exit
    reserve_system_gb: float = 18.0  # Always keep 18GB free


# ============================================================================
# Memory Monitoring
# ============================================================================

class MemoryMonitor:
    """
    Real-time memory monitoring with tiered safety thresholds.

    Example:
        >>> monitor = MemoryMonitor()
        >>> is_safe, level, stats = monitor.check_safety()
        >>> if level == 'critical':
        >>>     batch_size = monitor.estimate_batch_size(per_item_mb=512)
    """

    def __init__(
        self,
        thresholds: Optional[MemoryThresholds] = None,
        enable_swap_warning: bool = True
    ):
        """
        Initialize memory monitor.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
            enable_swap_warning: Warn if swap usage > 10%
        """
        self.thresholds = thresholds or MemoryThresholds()
        self.enable_swap_warning = enable_swap_warning
        self.last_check_time = 0.0
        self.check_interval = 5.0  # seconds between checks

        # Verify psutil is available
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            raise ImportError(
                "psutil not installed. Install with: pip install psutil"
            )

        logger.info(
            f"MemoryMonitor initialized with thresholds: "
            f"warning={self.thresholds.warning_percent}%, "
            f"critical={self.thresholds.critical_percent}%, "
            f"emergency={self.thresholds.emergency_percent}%"
        )

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats object with current memory usage

        Example:
            >>> stats = monitor.get_memory_stats()
            >>> print(f"RAM: {stats.used_gb:.1f}/{stats.total_gb:.1f} GB "
            ...       f"({stats.percent_used:.1f}%)")
        """
        # Virtual memory (RAM)
        vm = self.psutil.virtual_memory()

        # Swap memory
        swap = self.psutil.swap_memory()

        stats = MemoryStats(
            total_gb=vm.total / (1024 ** 3),
            used_gb=vm.used / (1024 ** 3),
            available_gb=vm.available / (1024 ** 3),
            percent_used=vm.percent,
            swap_total_gb=swap.total / (1024 ** 3),
            swap_used_gb=swap.used / (1024 ** 3),
            swap_percent=swap.percent,
            timestamp=time.time()
        )

        return stats

    def check_safety(
        self,
        force: bool = False
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check memory safety with tiered thresholds.

        Args:
            force: Force check even if within check_interval

        Returns:
            Tuple of (is_safe, level, stats_dict):
              - is_safe: True if memory usage is safe
              - level: 'normal' | 'warning' | 'critical' | 'emergency'
              - stats_dict: Dict with memory stats and recommendations

        Example:
            >>> is_safe, level, info = monitor.check_safety()
            >>> if not is_safe:
            >>>     logger.error(f"Memory {level}: {info['message']}")
            >>>     if level == 'emergency':
            >>>         save_checkpoint_and_exit()
        """
        # Rate limit checks (unless forced)
        current_time = time.time()
        if not force and (current_time - self.last_check_time) < self.check_interval:
            return True, 'normal', {'cached': True}

        self.last_check_time = current_time

        # Get current stats
        stats = self.get_memory_stats()

        # Determine safety level
        level = self._determine_safety_level(stats)

        # Build result dict
        result = {
            'total_gb': stats.total_gb,
            'used_gb': stats.used_gb,
            'available_gb': stats.available_gb,
            'percent_used': stats.percent_used,
            'swap_percent': stats.swap_percent,
            'level': level,
            'message': '',
            'recommendations': [],
        }

        # Level-specific messages and recommendations
        if level == 'emergency':
            result['message'] = (
                f"EMERGENCY: Memory usage at {stats.percent_used:.1f}% "
                f"(threshold: {self.thresholds.emergency_percent}%)"
            )
            result['recommendations'] = [
                'Save checkpoint immediately',
                'Exit gracefully to prevent system crash',
                'Reduce batch size for next run',
            ]
            logger.error(result['message'])
            return False, level, result

        elif level == 'critical':
            result['message'] = (
                f"CRITICAL: Memory usage at {stats.percent_used:.1f}% "
                f"(threshold: {self.thresholds.critical_percent}%)"
            )
            result['recommendations'] = [
                'Switch to minimal batch size',
                'Enable streaming/chunked processing',
                'Clear unnecessary caches',
                'Monitor closely for emergency threshold',
            ]
            logger.warning(result['message'])
            return False, level, result

        elif level == 'warning':
            result['message'] = (
                f"WARNING: Memory usage at {stats.percent_used:.1f}% "
                f"(threshold: {self.thresholds.warning_percent}%)"
            )
            result['recommendations'] = [
                'Reduce batch size by 50%',
                'Clear caches if possible',
                'Monitor memory trend',
            ]
            logger.warning(result['message'])
            return False, level, result

        else:  # normal
            result['message'] = (
                f"Memory usage normal: {stats.percent_used:.1f}%"
            )

            # Swap warning (even if memory is normal)
            if self.enable_swap_warning and stats.swap_percent > 10.0:
                result['recommendations'].append(
                    f'Swap usage high ({stats.swap_percent:.1f}%) - may impact performance'
                )
                logger.warning(
                    f"Swap usage high: {stats.swap_used_gb:.1f}/{stats.swap_total_gb:.1f} GB "
                    f"({stats.swap_percent:.1f}%)"
                )

            return True, level, result

    def _determine_safety_level(self, stats: MemoryStats) -> str:
        """
        Determine safety level from memory stats.

        Args:
            stats: MemoryStats object

        Returns:
            'normal' | 'warning' | 'critical' | 'emergency'
        """
        percent = stats.percent_used

        if percent >= self.thresholds.emergency_percent:
            return 'emergency'
        elif percent >= self.thresholds.critical_percent:
            return 'critical'
        elif percent >= self.thresholds.warning_percent:
            return 'warning'
        else:
            return 'normal'

    def get_memory_budget(self) -> float:
        """
        Calculate available memory budget (GB) for processing.

        Budget = Available RAM - System Reserve (18GB)

        Returns:
            Available memory in GB (or 0 if insufficient)

        Example:
            >>> budget_gb = monitor.get_memory_budget()
            >>> if budget_gb < 1.0:
            >>>     logger.error("Insufficient memory budget")
        """
        stats = self.get_memory_stats()

        # Calculate budget
        budget_gb = stats.available_gb - self.thresholds.reserve_system_gb

        if budget_gb < 0:
            logger.warning(
                f"Memory budget negative: {budget_gb:.1f} GB "
                f"(available: {stats.available_gb:.1f} GB, "
                f"reserve: {self.thresholds.reserve_system_gb:.1f} GB)"
            )
            return 0.0

        return budget_gb

    def estimate_batch_size(
        self,
        per_item_mb: float,
        safety_factor: float = 0.8
    ) -> int:
        """
        Estimate safe batch size based on memory budget.

        Args:
            per_item_mb: Memory consumption per item (in MB)
            safety_factor: Use this fraction of budget (default 0.8 = 80%)

        Returns:
            Recommended batch size (minimum 1)

        Example:
            >>> # Estimate batch size for image processing (512 MB per image)
            >>> batch_size = monitor.estimate_batch_size(per_item_mb=512)
            >>> logger.info(f"Using batch size: {batch_size}")
        """
        budget_gb = self.get_memory_budget()

        # Apply safety factor
        usable_budget_gb = budget_gb * safety_factor

        # Convert to MB
        usable_budget_mb = usable_budget_gb * 1024

        # Calculate batch size
        batch_size = int(usable_budget_mb / per_item_mb)

        # Ensure minimum of 1
        batch_size = max(1, batch_size)

        logger.debug(
            f"Estimated batch size: {batch_size} "
            f"(budget: {budget_gb:.1f} GB, per_item: {per_item_mb:.0f} MB)"
        )

        return batch_size

    def adjust_batch_size_for_level(
        self,
        current_batch_size: int,
        level: str
    ) -> int:
        """
        Adjust batch size based on safety level.

        Args:
            current_batch_size: Current batch size
            level: Safety level ('normal' | 'warning' | 'critical' | 'emergency')

        Returns:
            Adjusted batch size

        Example:
            >>> is_safe, level, _ = monitor.check_safety()
            >>> new_batch = monitor.adjust_batch_size_for_level(batch_size, level)
        """
        if level == 'emergency':
            # Emergency: batch size = 1
            return 1

        elif level == 'critical':
            # Critical: batch size = 2 (minimal but allows some parallelism)
            return max(1, min(2, current_batch_size))

        elif level == 'warning':
            # Warning: reduce by 50%
            return max(1, current_batch_size // 2)

        else:
            # Normal: no change
            return current_batch_size

    def wait_for_memory(
        self,
        required_gb: float,
        timeout: float = 60.0,
        check_interval: float = 2.0
    ) -> bool:
        """
        Wait until required memory becomes available.

        Args:
            required_gb: Required available memory (GB)
            timeout: Maximum wait time (seconds)
            check_interval: Check interval (seconds)

        Returns:
            True if memory available, False if timeout

        Example:
            >>> # Wait for 10 GB to become available
            >>> if monitor.wait_for_memory(required_gb=10.0, timeout=30.0):
            >>>     # Proceed with processing
            >>> else:
            >>>     logger.error("Timeout waiting for memory")
        """
        start_time = time.time()

        while True:
            stats = self.get_memory_stats()

            if stats.available_gb >= required_gb:
                logger.info(
                    f"Required memory available: {stats.available_gb:.1f} GB >= {required_gb:.1f} GB"
                )
                return True

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    f"Timeout waiting for memory: {stats.available_gb:.1f} GB < {required_gb:.1f} GB "
                    f"(waited {elapsed:.1f}s)"
                )
                return False

            logger.debug(
                f"Waiting for memory: {stats.available_gb:.1f} GB < {required_gb:.1f} GB "
                f"(elapsed: {elapsed:.1f}s)"
            )

            time.sleep(check_interval)


# ============================================================================
# Utilities
# ============================================================================

def load_thresholds_from_yaml(config_path: Path) -> MemoryThresholds:
    """
    Load memory thresholds from YAML config.

    Args:
        config_path: Path to resource_limits.yaml

    Returns:
        MemoryThresholds object

    Example:
        >>> thresholds = load_thresholds_from_yaml(
        ...     Path('configs/automation/resource_limits.yaml')
        ... )
        >>> monitor = MemoryMonitor(thresholds=thresholds)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return MemoryThresholds()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    memory_config = config.get('memory', {})
    thresholds_config = memory_config.get('thresholds', {})

    return MemoryThresholds(
        warning_percent=thresholds_config.get('warning_percent', 70.0),
        critical_percent=thresholds_config.get('critical_percent', 80.0),
        emergency_percent=thresholds_config.get('emergency_percent', 85.0),
        reserve_system_gb=memory_config.get('reserve_system_ram_gb', 18.0),
    )


def print_memory_report(monitor: Optional[MemoryMonitor] = None) -> None:
    """
    Print formatted memory report.

    Args:
        monitor: MemoryMonitor instance (creates new one if None)

    Example:
        >>> print_memory_report()
        Memory Report:
          Total RAM: 128.0 GB
          Used RAM: 64.0 GB (50.0%)
          Available RAM: 64.0 GB
          Swap: 32.0 GB (10.0% used)
          Safety Level: normal
          Memory Budget: 46.0 GB
    """
    if monitor is None:
        monitor = MemoryMonitor()

    stats = monitor.get_memory_stats()
    is_safe, level, info = monitor.check_safety(force=True)
    budget = monitor.get_memory_budget()

    print("Memory Report:")
    print(f"  Total RAM: {stats.total_gb:.1f} GB")
    print(f"  Used RAM: {stats.used_gb:.1f} GB ({stats.percent_used:.1f}%)")
    print(f"  Available RAM: {stats.available_gb:.1f} GB")
    print(f"  Swap: {stats.swap_total_gb:.1f} GB ({stats.swap_percent:.1f}% used)")
    print(f"  Safety Level: {level}")
    print(f"  Memory Budget: {budget:.1f} GB")

    if info.get('recommendations'):
        print("  Recommendations:")
        for rec in info['recommendations']:
            print(f"    - {rec}")


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
    print("Memory Monitor Module - Self Test")
    print("=" * 80)

    # Test 1: Initialize monitor
    print("\n[Test 1] Initializing MemoryMonitor...")
    try:
        monitor = MemoryMonitor()
        print("✓ MemoryMonitor initialized")
    except ImportError as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)

    # Test 2: Get memory stats
    print("\n[Test 2] Getting memory statistics...")
    stats = monitor.get_memory_stats()
    print(f"✓ Total RAM: {stats.total_gb:.1f} GB")
    print(f"  Used RAM: {stats.used_gb:.1f} GB ({stats.percent_used:.1f}%)")
    print(f"  Available RAM: {stats.available_gb:.1f} GB")
    print(f"  Swap: {stats.swap_total_gb:.1f} GB ({stats.swap_percent:.1f}% used)")

    # Test 3: Check safety
    print("\n[Test 3] Checking memory safety...")
    is_safe, level, info = monitor.check_safety(force=True)
    print(f"  Safety Level: {level}")
    print(f"  Message: {info['message']}")
    if info.get('recommendations'):
        print("  Recommendations:")
        for rec in info['recommendations']:
            print(f"    - {rec}")

    # Test 4: Memory budget
    print("\n[Test 4] Calculating memory budget...")
    budget = monitor.get_memory_budget()
    print(f"✓ Memory budget: {budget:.1f} GB")
    print(f"  (Available: {stats.available_gb:.1f} GB - Reserve: {monitor.thresholds.reserve_system_gb:.1f} GB)")

    # Test 5: Estimate batch size
    print("\n[Test 5] Estimating batch sizes...")
    for per_item_mb in [128, 256, 512, 1024]:
        batch_size = monitor.estimate_batch_size(per_item_mb=per_item_mb)
        print(f"  {per_item_mb} MB per item → batch size {batch_size}")

    # Test 6: Batch size adjustment
    print("\n[Test 6] Testing batch size adjustment...")
    for test_level in ['normal', 'warning', 'critical', 'emergency']:
        adjusted = monitor.adjust_batch_size_for_level(current_batch_size=32, level=test_level)
        print(f"  Level '{test_level}': 32 → {adjusted}")

    # Test 7: Full memory report
    print("\n[Test 7] Full memory report:")
    print_memory_report(monitor)

    print("\n" + "=" * 80)
    print("Self-test complete")
    print("=" * 80)
