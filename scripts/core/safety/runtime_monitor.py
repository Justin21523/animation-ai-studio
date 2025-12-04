"""
Runtime Monitor Module

Provides continuous monitoring during workflow execution with automatic safety actions.
Runs in background thread to check memory/GPU status without blocking main workflow.

Features:
  - Periodic memory monitoring (configurable interval)
  - Automatic batch size reduction on memory warning
  - Emergency checkpoint save on critical memory
  - GPU usage detection and alerts
  - Performance metrics logging

Integration:
  - Use as context manager: with RuntimeMonitor() as monitor:
  - Or manual start/stop: monitor.start() ... monitor.stop()
  - Callbacks for safety events (warning, critical, emergency)

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import os
import time
import threading
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .memory_monitor import MemoryMonitor, MemoryStats
from .gpu_isolation import verify_no_gpu_usage, GPUIsolationError

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MonitoringStats:
    """Runtime monitoring statistics."""
    start_time: float
    checks_performed: int = 0
    memory_warnings: int = 0
    memory_criticals: int = 0
    memory_emergencies: int = 0
    gpu_violations: int = 0
    last_check_time: float = 0.0
    last_memory_percent: float = 0.0


# ============================================================================
# Runtime Monitor
# ============================================================================

class RuntimeMonitor:
    """
    Background monitoring thread for runtime safety checks.

    Example:
        >>> # Context manager usage (recommended)
        >>> with RuntimeMonitor(check_interval=30.0) as monitor:
        >>>     # Your workflow code here
        >>>     process_data()
        >>>
        >>> # Manual usage
        >>> monitor = RuntimeMonitor()
        >>> monitor.start()
        >>> try:
        >>>     process_data()
        >>> finally:
        >>>     monitor.stop()
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        enable_gpu_checks: bool = True,
        enable_memory_checks: bool = True,
        on_warning: Optional[Callable] = None,
        on_critical: Optional[Callable] = None,
        on_emergency: Optional[Callable] = None,
        on_gpu_violation: Optional[Callable] = None,
    ):
        """
        Initialize runtime monitor.

        Args:
            check_interval: Seconds between safety checks (default 30.0)
            enable_gpu_checks: Monitor for GPU usage violations
            enable_memory_checks: Monitor memory safety levels
            on_warning: Callback for memory warning level
            on_critical: Callback for memory critical level
            on_emergency: Callback for memory emergency level
            on_gpu_violation: Callback for GPU isolation violation
        """
        self.check_interval = check_interval
        self.enable_gpu_checks = enable_gpu_checks
        self.enable_memory_checks = enable_memory_checks

        # Callbacks
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.on_emergency = on_emergency
        self.on_gpu_violation = on_gpu_violation

        # Internal state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Statistics
        self.stats = MonitoringStats(start_time=time.time())

        # Memory monitor
        if self.enable_memory_checks:
            self.memory_monitor = MemoryMonitor()
        else:
            self.memory_monitor = None

        logger.info(
            f"RuntimeMonitor initialized (interval={check_interval}s, "
            f"gpu_checks={enable_gpu_checks}, memory_checks={enable_memory_checks})"
        )

    def start(self) -> None:
        """
        Start background monitoring thread.

        Example:
            >>> monitor = RuntimeMonitor()
            >>> monitor.start()
        """
        with self._lock:
            if self._running:
                logger.warning("Monitor already running")
                return

            self._running = True
            self._thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name='RuntimeMonitor'
            )
            self._thread.start()
            logger.info("✓ Runtime monitor started")

    def stop(self) -> None:
        """
        Stop background monitoring thread.

        Example:
            >>> monitor.stop()
        """
        with self._lock:
            if not self._running:
                logger.warning("Monitor not running")
                return

            self._running = False

        # Wait for thread to finish (with timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.check_interval + 5.0)

        logger.info("✓ Runtime monitor stopped")

    def is_running(self) -> bool:
        """Check if monitor is currently running."""
        with self._lock:
            return self._running

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current monitoring statistics.

        Returns:
            Dict with runtime statistics

        Example:
            >>> stats = monitor.get_stats()
            >>> print(f"Checks: {stats['checks_performed']}, "
            ...       f"Warnings: {stats['memory_warnings']}")
        """
        elapsed = time.time() - self.stats.start_time
        return {
            'running_time_seconds': elapsed,
            'checks_performed': self.stats.checks_performed,
            'memory_warnings': self.stats.memory_warnings,
            'memory_criticals': self.stats.memory_criticals,
            'memory_emergencies': self.stats.memory_emergencies,
            'gpu_violations': self.stats.gpu_violations,
            'last_memory_percent': self.stats.last_memory_percent,
            'checks_per_minute': (self.stats.checks_performed / elapsed * 60.0) if elapsed > 0 else 0,
        }

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop (runs in background thread).

        This method runs continuously while self._running is True,
        performing safety checks at regular intervals.
        """
        logger.info("Monitoring loop started")

        while self._running:
            try:
                # Perform safety checks
                self._perform_safety_check()

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Continue monitoring despite errors
                time.sleep(self.check_interval)

        logger.info("Monitoring loop stopped")

    def _perform_safety_check(self) -> None:
        """
        Perform one round of safety checks.

        Called periodically by monitoring loop.
        """
        self.stats.checks_performed += 1
        self.stats.last_check_time = time.time()

        # Check memory safety
        if self.enable_memory_checks and self.memory_monitor:
            self._check_memory_safety()

        # Check GPU isolation
        if self.enable_gpu_checks:
            self._check_gpu_isolation()

    def _check_memory_safety(self) -> None:
        """Check memory safety and trigger callbacks."""
        try:
            is_safe, level, info = self.memory_monitor.check_safety(force=True)

            # Update stats
            stats = self.memory_monitor.get_memory_stats()
            self.stats.last_memory_percent = stats.percent_used

            # Handle safety levels
            if level == 'emergency':
                self.stats.memory_emergencies += 1
                logger.critical(
                    f"EMERGENCY: Memory at {stats.percent_used:.1f}% - "
                    f"Save checkpoint and exit!"
                )
                if self.on_emergency:
                    try:
                        self.on_emergency(level, info)
                    except Exception as e:
                        logger.error(f"Emergency callback failed: {e}")

            elif level == 'critical':
                self.stats.memory_criticals += 1
                logger.error(
                    f"CRITICAL: Memory at {stats.percent_used:.1f}% - "
                    f"Reduce batch size immediately"
                )
                if self.on_critical:
                    try:
                        self.on_critical(level, info)
                    except Exception as e:
                        logger.error(f"Critical callback failed: {e}")

            elif level == 'warning':
                self.stats.memory_warnings += 1
                logger.warning(
                    f"WARNING: Memory at {stats.percent_used:.1f}% - "
                    f"Reduce batch size by 50%"
                )
                if self.on_warning:
                    try:
                        self.on_warning(level, info)
                    except Exception as e:
                        logger.error(f"Warning callback failed: {e}")

            else:
                logger.debug(f"Memory OK: {stats.percent_used:.1f}%")

        except Exception as e:
            logger.error(f"Memory safety check failed: {e}", exc_info=True)

    def _check_gpu_isolation(self) -> None:
        """Check GPU isolation is still valid."""
        try:
            is_safe, msg = verify_no_gpu_usage(raise_on_violation=False)

            if not is_safe:
                self.stats.gpu_violations += 1
                logger.error(f"GPU isolation violated: {msg}")

                if self.on_gpu_violation:
                    try:
                        self.on_gpu_violation(msg)
                    except Exception as e:
                        logger.error(f"GPU violation callback failed: {e}")

        except Exception as e:
            logger.error(f"GPU isolation check failed: {e}", exc_info=True)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False  # Don't suppress exceptions


# ============================================================================
# Utilities
# ============================================================================

def create_checkpoint_callback(checkpoint_dir: str) -> Callable:
    """
    Create emergency callback that saves checkpoint and exits.

    Args:
        checkpoint_dir: Directory to save emergency checkpoint

    Returns:
        Callback function for on_emergency parameter

    Example:
        >>> from pathlib import Path
        >>> checkpoint_callback = create_checkpoint_callback('/tmp/checkpoints')
        >>> monitor = RuntimeMonitor(on_emergency=checkpoint_callback)
    """
    def emergency_callback(level: str, info: Dict[str, Any]) -> None:
        """Emergency callback: save checkpoint and exit."""
        from pathlib import Path
        import json

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save emergency checkpoint
        checkpoint_file = checkpoint_path / f'emergency_{int(time.time())}.json'
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'reason': 'memory_emergency',
                'memory_info': info,
            }, f, indent=2)

        logger.critical(f"Emergency checkpoint saved: {checkpoint_file}")
        logger.critical("Exiting due to memory emergency")

        # Exit process
        import sys
        sys.exit(1)

    return emergency_callback


def create_batch_size_callback(reduce_batch_fn: Callable[[str], None]) -> Callable:
    """
    Create callback that reduces batch size on warning/critical.

    Args:
        reduce_batch_fn: Function that takes safety level and reduces batch size

    Returns:
        Callback function for on_warning/on_critical parameters

    Example:
        >>> def my_batch_reducer(level):
        ...     if level == 'warning':
        ...         batch_size = batch_size // 2
        ...     elif level == 'critical':
        ...         batch_size = 1
        >>>
        >>> callback = create_batch_size_callback(my_batch_reducer)
        >>> monitor = RuntimeMonitor(on_warning=callback, on_critical=callback)
    """
    def batch_size_callback(level: str, info: Dict[str, Any]) -> None:
        """Batch size reduction callback."""
        logger.info(f"Reducing batch size (level: {level})")
        reduce_batch_fn(level)

    return batch_size_callback


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
    print("Runtime Monitor Module - Self Test")
    print("=" * 80)

    # Test 1: Basic start/stop
    print("\n[Test 1] Basic start/stop...")
    monitor = RuntimeMonitor(check_interval=2.0)
    monitor.start()
    time.sleep(5.0)
    monitor.stop()
    stats = monitor.get_stats()
    print(f"✓ Performed {stats['checks_performed']} checks in {stats['running_time_seconds']:.1f}s")

    # Test 2: Context manager
    print("\n[Test 2] Context manager usage...")
    with RuntimeMonitor(check_interval=2.0) as mon:
        print("  Monitor running...")
        time.sleep(5.0)
    print("✓ Monitor stopped automatically")

    # Test 3: Callbacks
    print("\n[Test 3] Testing callbacks...")

    callback_triggered = {'warning': False, 'critical': False}

    def test_warning(level, info):
        print(f"  ⚠ Warning callback triggered: {level}")
        callback_triggered['warning'] = True

    def test_critical(level, info):
        print(f"  ✗ Critical callback triggered: {level}")
        callback_triggered['critical'] = True

    with RuntimeMonitor(
        check_interval=2.0,
        on_warning=test_warning,
        on_critical=test_critical
    ) as mon:
        time.sleep(5.0)

    print(f"✓ Callbacks configured (triggered: {callback_triggered})")

    # Test 4: Statistics
    print("\n[Test 4] Statistics report...")
    monitor = RuntimeMonitor(check_interval=1.0)
    monitor.start()
    time.sleep(3.5)
    stats = monitor.get_stats()
    monitor.stop()

    print(f"  Checks performed: {stats['checks_performed']}")
    print(f"  Running time: {stats['running_time_seconds']:.1f}s")
    print(f"  Checks per minute: {stats['checks_per_minute']:.1f}")
    print(f"  Memory warnings: {stats['memory_warnings']}")
    print(f"  Memory criticals: {stats['memory_criticals']}")
    print(f"  Memory emergencies: {stats['memory_emergencies']}")
    print(f"  GPU violations: {stats['gpu_violations']}")
    print(f"  Last memory %: {stats['last_memory_percent']:.1f}%")

    print("\n" + "=" * 80)
    print("Self-test complete")
    print("=" * 80)
