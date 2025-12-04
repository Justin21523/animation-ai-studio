"""
Memory Budget Manager

RAM usage monitoring and enforcement system for CPU-only automation.
Ensures automation processes stay within memory budgets and don't impact
LoRA training performance.

Features:
- Real-time RAM usage monitoring
- Per-module memory tracking
- Soft/hard limit enforcement
- Graceful degradation triggers
- Memory leak detection
- Automatic memory reporting

Author: Animation AI Studio
Date: 2025-12-02
"""

import os
import sys
import time
import logging
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class MemoryLevel(Enum):
    """Memory usage levels"""
    NORMAL = "normal"          # < soft_limit
    WARNING = "warning"        # >= soft_limit, < hard_limit
    CRITICAL = "critical"      # >= hard_limit
    EMERGENCY = "emergency"    # >= emergency_limit


class MemoryAction(Enum):
    """Actions to take on memory threshold breach"""
    LOG = "log"                    # Log warning only
    NOTIFY = "notify"              # Trigger notification event
    DEGRADE = "degrade"            # Trigger performance degradation
    THROTTLE = "throttle"          # Reduce concurrency
    EMERGENCY_STOP = "emergency"   # Stop all automation


@dataclass
class MemoryThresholds:
    """Memory limit configuration"""
    soft_limit_mb: int = 4096      # Warning threshold (4GB)
    hard_limit_mb: int = 6144      # Degradation threshold (6GB)
    emergency_limit_mb: int = 8192 # Emergency stop threshold (8GB)

    soft_action: MemoryAction = MemoryAction.NOTIFY
    hard_action: MemoryAction = MemoryAction.DEGRADE
    emergency_action: MemoryAction = MemoryAction.EMERGENCY_STOP

    def get_level(self, usage_mb: float) -> MemoryLevel:
        """Determine memory level based on usage"""
        if usage_mb >= self.emergency_limit_mb:
            return MemoryLevel.EMERGENCY
        elif usage_mb >= self.hard_limit_mb:
            return MemoryLevel.CRITICAL
        elif usage_mb >= self.soft_limit_mb:
            return MemoryLevel.WARNING
        else:
            return MemoryLevel.NORMAL

    def get_action(self, level: MemoryLevel) -> MemoryAction:
        """Get action for memory level"""
        action_map = {
            MemoryLevel.NORMAL: MemoryAction.LOG,
            MemoryLevel.WARNING: self.soft_action,
            MemoryLevel.CRITICAL: self.hard_action,
            MemoryLevel.EMERGENCY: self.emergency_action
        }
        return action_map.get(level, MemoryAction.LOG)


@dataclass
class MemoryStatus:
    """Current memory status snapshot"""
    timestamp: float = field(default_factory=time.time)

    # System memory
    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    percent: float = 0.0

    # Process memory
    process_rss_mb: float = 0.0      # Resident Set Size
    process_vms_mb: float = 0.0      # Virtual Memory Size
    process_percent: float = 0.0

    # Budget status
    level: MemoryLevel = MemoryLevel.NORMAL
    over_soft_limit: bool = False
    over_hard_limit: bool = False
    over_emergency_limit: bool = False

    # Module tracking
    module_usage: Dict[str, float] = field(default_factory=dict)

    def is_safe(self) -> bool:
        """Check if memory usage is safe"""
        return self.level in [MemoryLevel.NORMAL, MemoryLevel.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "system": {
                "total_mb": self.total_mb,
                "available_mb": self.available_mb,
                "used_mb": self.used_mb,
                "percent": self.percent
            },
            "process": {
                "rss_mb": self.process_rss_mb,
                "vms_mb": self.process_vms_mb,
                "percent": self.process_percent
            },
            "budget": {
                "level": self.level.value,
                "over_soft_limit": self.over_soft_limit,
                "over_hard_limit": self.over_hard_limit,
                "over_emergency_limit": self.over_emergency_limit
            },
            "modules": self.module_usage
        }


@dataclass
class MemoryLeak:
    """Memory leak detection result"""
    detected: bool
    trend_mb_per_min: float
    confidence: float
    samples_analyzed: int
    detection_time: float = field(default_factory=time.time)


class MemoryBudget:
    """
    Memory Budget Manager

    Monitors and enforces RAM usage limits for CPU-only automation.

    Features:
    - Real-time system and process memory monitoring
    - Per-module memory tracking (estimated)
    - Soft/hard/emergency limit enforcement
    - Memory leak detection via trend analysis
    - Configurable actions on threshold breach
    - Memory usage history and statistics

    Example:
        # Configure budget
        thresholds = MemoryThresholds(
            soft_limit_mb=4096,
            hard_limit_mb=6144,
            emergency_limit_mb=8192
        )

        budget = MemoryBudget(thresholds=thresholds)

        # Start monitoring
        budget.start_monitoring()

        # Check status
        status = budget.get_status()
        if not status.is_safe():
            budget.handle_threshold_breach(status)

        # Track module usage
        budget.register_module("agent", estimated_mb=512)
        budget.update_module_usage("agent", 650)

        # Detect leaks
        leak = budget.detect_memory_leak()
        if leak.detected:
            logger.warning(f"Memory leak: {leak.trend_mb_per_min:.2f} MB/min")
    """

    def __init__(
        self,
        thresholds: Optional[MemoryThresholds] = None,
        check_interval: float = 10.0,
        history_size: int = 100,
        leak_detection_window: int = 30,
        enable_per_module_tracking: bool = True
    ):
        """
        Initialize Memory Budget Manager

        Args:
            thresholds: Memory threshold configuration
            check_interval: Monitoring interval (seconds)
            history_size: Number of status snapshots to keep
            leak_detection_window: Window size for leak detection
            enable_per_module_tracking: Enable per-module memory tracking
        """
        self.thresholds = thresholds or MemoryThresholds()
        self.check_interval = check_interval
        self.history_size = history_size
        self.leak_detection_window = leak_detection_window
        self.enable_per_module_tracking = enable_per_module_tracking

        # Process handle
        self.process = psutil.Process(os.getpid())

        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # History tracking
        self._history: deque = deque(maxlen=history_size)

        # Module tracking
        self._modules: Dict[str, Dict[str, Any]] = {}
        self._module_baselines: Dict[str, float] = {}

        # Callbacks
        self._threshold_callbacks: List[Callable[[MemoryStatus], None]] = []

        # Statistics
        self._breach_count = 0
        self._last_breach_time: Optional[float] = None

        logger.info(
            f"MemoryBudget initialized: "
            f"soft={self.thresholds.soft_limit_mb}MB, "
            f"hard={self.thresholds.hard_limit_mb}MB, "
            f"emergency={self.thresholds.emergency_limit_mb}MB"
        )

    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self._monitoring:
            logger.warning("Memory monitoring already running")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval={self.check_interval}s)")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                status = self.get_status()
                self._history.append(status)

                # Check for threshold breaches
                if not status.is_safe():
                    self._breach_count += 1
                    self._last_breach_time = time.time()
                    self.handle_threshold_breach(status)

            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}", exc_info=True)

            time.sleep(self.check_interval)

    def get_status(self) -> MemoryStatus:
        """
        Get current memory status

        Returns:
            MemoryStatus snapshot
        """
        # System memory
        mem = psutil.virtual_memory()

        # Process memory
        proc_mem = self.process.memory_info()

        status = MemoryStatus(
            timestamp=time.time(),
            # System
            total_mb=mem.total / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            used_mb=mem.used / (1024 * 1024),
            percent=mem.percent,
            # Process
            process_rss_mb=proc_mem.rss / (1024 * 1024),
            process_vms_mb=proc_mem.vms / (1024 * 1024),
            process_percent=self.process.memory_percent()
        )

        # Determine level based on process RSS
        status.level = self.thresholds.get_level(status.process_rss_mb)
        status.over_soft_limit = status.process_rss_mb >= self.thresholds.soft_limit_mb
        status.over_hard_limit = status.process_rss_mb >= self.thresholds.hard_limit_mb
        status.over_emergency_limit = status.process_rss_mb >= self.thresholds.emergency_limit_mb

        # Per-module tracking
        if self.enable_per_module_tracking:
            status.module_usage = self._estimate_module_usage()

        return status

    def register_module(
        self,
        module_name: str,
        estimated_mb: Optional[float] = None,
        baseline_mb: Optional[float] = None
    ):
        """
        Register module for memory tracking

        Args:
            module_name: Module identifier
            estimated_mb: Estimated memory usage
            baseline_mb: Baseline memory before module init
        """
        if baseline_mb is None:
            baseline_mb = self.process.memory_info().rss / (1024 * 1024)

        self._modules[module_name] = {
            "estimated_mb": estimated_mb,
            "baseline_mb": baseline_mb,
            "registered_at": time.time(),
            "current_mb": estimated_mb or 0.0
        }

        self._module_baselines[module_name] = baseline_mb

        logger.debug(
            f"Module registered: {module_name} "
            f"(estimated={estimated_mb}MB, baseline={baseline_mb:.1f}MB)"
        )

    def update_module_usage(self, module_name: str, usage_mb: float):
        """
        Update module memory usage

        Args:
            module_name: Module identifier
            usage_mb: Current memory usage in MB
        """
        if module_name not in self._modules:
            logger.warning(f"Module {module_name} not registered, registering now")
            self.register_module(module_name)

        self._modules[module_name]["current_mb"] = usage_mb
        self._modules[module_name]["last_update"] = time.time()

    def _estimate_module_usage(self) -> Dict[str, float]:
        """
        Estimate per-module memory usage

        Returns:
            Dict mapping module names to estimated MB
        """
        result = {}
        for module_name, info in self._modules.items():
            result[module_name] = info["current_mb"]
        return result

    def handle_threshold_breach(self, status: MemoryStatus):
        """
        Handle memory threshold breach

        Args:
            status: Current memory status
        """
        action = self.thresholds.get_action(status.level)

        logger.warning(
            f"Memory threshold breach #{self._breach_count}: "
            f"level={status.level.value}, "
            f"usage={status.process_rss_mb:.1f}MB, "
            f"action={action.value}"
        )

        # Execute action
        if action == MemoryAction.LOG:
            pass  # Already logged above

        elif action == MemoryAction.NOTIFY:
            self._trigger_callbacks(status)

        elif action == MemoryAction.DEGRADE:
            logger.warning("Triggering performance degradation")
            self._trigger_callbacks(status)

        elif action == MemoryAction.THROTTLE:
            logger.warning("Throttling automation processes")
            self._trigger_callbacks(status)

        elif action == MemoryAction.EMERGENCY_STOP:
            logger.critical(
                f"EMERGENCY: Memory limit exceeded ({status.process_rss_mb:.1f}MB). "
                "Stopping automation to prevent system instability."
            )
            self._trigger_callbacks(status)
            # Give time to flush logs
            time.sleep(1)
            sys.exit(1)

    def register_threshold_callback(self, callback: Callable[[MemoryStatus], None]):
        """
        Register callback for threshold breaches

        Args:
            callback: Function to call on breach
        """
        self._threshold_callbacks.append(callback)
        logger.debug(f"Threshold callback registered: {callback.__name__}")

    def _trigger_callbacks(self, status: MemoryStatus):
        """Trigger all registered callbacks"""
        for callback in self._threshold_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Error in threshold callback: {e}", exc_info=True)

    def detect_memory_leak(
        self,
        window_size: Optional[int] = None,
        threshold_mb_per_min: float = 10.0
    ) -> MemoryLeak:
        """
        Detect memory leak via trend analysis

        Args:
            window_size: Number of recent samples to analyze
            threshold_mb_per_min: Leak threshold (MB/min)

        Returns:
            MemoryLeak detection result
        """
        if window_size is None:
            window_size = self.leak_detection_window

        if len(self._history) < window_size:
            return MemoryLeak(
                detected=False,
                trend_mb_per_min=0.0,
                confidence=0.0,
                samples_analyzed=len(self._history)
            )

        # Get recent samples
        recent = list(self._history)[-window_size:]

        # Calculate trend
        times = [s.timestamp for s in recent]
        memory = [s.process_rss_mb for s in recent]

        # Simple linear regression
        n = len(times)
        time_mean = statistics.mean(times)
        mem_mean = statistics.mean(memory)

        numerator = sum((times[i] - time_mean) * (memory[i] - mem_mean) for i in range(n))
        denominator = sum((times[i] - time_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Convert to MB/min
        trend_mb_per_min = slope * 60

        # Calculate confidence (R²)
        if len(memory) > 1:
            ss_tot = sum((m - mem_mean) ** 2 for m in memory)
            predictions = [mem_mean + slope * (times[i] - time_mean) for i in range(n)]
            ss_res = sum((memory[i] - predictions[i]) ** 2 for i in range(n))

            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0.0
        else:
            r_squared = 0.0

        # Detect leak
        detected = (
            trend_mb_per_min > threshold_mb_per_min and
            r_squared > 0.7  # High confidence in trend
        )

        return MemoryLeak(
            detected=detected,
            trend_mb_per_min=trend_mb_per_min,
            confidence=r_squared,
            samples_analyzed=n
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Statistics dictionary
        """
        if not self._history:
            return {
                "samples": 0,
                "monitoring": self._monitoring
            }

        recent = list(self._history)

        rss_values = [s.process_rss_mb for s in recent]

        return {
            "monitoring": self._monitoring,
            "samples": len(recent),
            "breach_count": self._breach_count,
            "last_breach": self._last_breach_time,
            "current": {
                "rss_mb": rss_values[-1] if rss_values else 0.0,
                "level": recent[-1].level.value if recent else "unknown"
            },
            "statistics": {
                "min_mb": min(rss_values) if rss_values else 0.0,
                "max_mb": max(rss_values) if rss_values else 0.0,
                "mean_mb": statistics.mean(rss_values) if rss_values else 0.0,
                "stdev_mb": statistics.stdev(rss_values) if len(rss_values) > 1 else 0.0
            },
            "modules": {
                name: info["current_mb"]
                for name, info in self._modules.items()
            }
        }

    def get_report(self) -> str:
        """
        Generate human-readable memory report

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        status = self.get_status()

        report = []
        report.append("=" * 60)
        report.append("Memory Budget Report")
        report.append("=" * 60)

        # Current status
        report.append("\n[Current Status]")
        report.append(f"  Level: {status.level.value.upper()}")
        report.append(f"  Process RSS: {status.process_rss_mb:.1f} MB")
        report.append(f"  System Usage: {status.percent:.1f}%")
        report.append(f"  Available: {status.available_mb:.1f} MB")

        # Thresholds
        report.append("\n[Thresholds]")
        report.append(f"  Soft Limit: {self.thresholds.soft_limit_mb} MB")
        report.append(f"  Hard Limit: {self.thresholds.hard_limit_mb} MB")
        report.append(f"  Emergency: {self.thresholds.emergency_limit_mb} MB")

        # Statistics
        if stats["samples"] > 0:
            report.append("\n[Statistics]")
            report.append(f"  Samples: {stats['samples']}")
            report.append(f"  Breach Count: {stats['breach_count']}")
            report.append(f"  Min/Max/Mean: {stats['statistics']['min_mb']:.1f} / "
                         f"{stats['statistics']['max_mb']:.1f} / "
                         f"{stats['statistics']['mean_mb']:.1f} MB")

        # Module tracking
        if self._modules:
            report.append("\n[Module Usage]")
            for module_name, usage_mb in stats["modules"].items():
                report.append(f"  {module_name}: {usage_mb:.1f} MB")

        # Leak detection
        if len(self._history) >= self.leak_detection_window:
            leak = self.detect_memory_leak()
            report.append("\n[Leak Detection]")
            report.append(f"  Detected: {'YES' if leak.detected else 'NO'}")
            report.append(f"  Trend: {leak.trend_mb_per_min:+.2f} MB/min")
            report.append(f"  Confidence: {leak.confidence:.2%}")

        report.append("=" * 60)
        return "\n".join(report)

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"MemoryBudget(level={status.level.value}, "
            f"usage={status.process_rss_mb:.1f}MB, "
            f"monitoring={self._monitoring})"
        )
