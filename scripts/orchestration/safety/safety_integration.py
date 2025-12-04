"""
Safety Integration Layer

Unified safety system integrating GPU isolation, memory budgeting,
degradation management, and emergency handling.

Provides single interface for orchestration layer safety.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

from .gpu_isolation import GPUIsolation, IsolationLevel, GPUCheckResult
from .memory_budget import MemoryBudget, MemoryThresholds, MemoryStatus, MemoryLevel
from .degradation_manager import DegradationManager, DegradationLevel
from .emergency_handler import EmergencyHandler, EmergencyType, EmergencySeverity

logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    """Overall safety status"""
    SAFE = "safe"                # All systems operational
    WARNING = "warning"          # Minor issues detected
    DEGRADED = "degraded"        # Running in degraded mode
    CRITICAL = "critical"        # Critical issues, limited operation
    EMERGENCY = "emergency"      # Emergency situation


@dataclass
class SystemSafetyReport:
    """Comprehensive safety status report"""
    timestamp: float
    overall_status: SafetyStatus

    # Component statuses
    gpu_safe: bool
    memory_safe: bool
    degradation_level: DegradationLevel
    active_emergencies: int

    # Details
    gpu_violations: int
    memory_level: MemoryLevel
    memory_usage_mb: float
    memory_percent: float

    # Recommendations
    can_continue: bool
    should_degrade: bool
    should_pause: bool
    should_stop: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "component_status": {
                "gpu_safe": self.gpu_safe,
                "memory_safe": self.memory_safe,
                "degradation_level": self.degradation_level.value,
                "active_emergencies": self.active_emergencies
            },
            "details": {
                "gpu_violations": self.gpu_violations,
                "memory_level": self.memory_level.value,
                "memory_usage_mb": self.memory_usage_mb,
                "memory_percent": self.memory_percent
            },
            "recommendations": {
                "can_continue": self.can_continue,
                "should_degrade": self.should_degrade,
                "should_pause": self.should_pause,
                "should_stop": self.should_stop
            }
        }


class SafetyIntegration:
    """
    Safety Integration Layer

    Unified safety system coordinating all safety components.

    Components:
    - GPUIsolation: 4-layer GPU isolation
    - MemoryBudget: RAM usage monitoring and limits
    - DegradationManager: Graceful performance reduction
    - EmergencyHandler: Critical situation response

    Features:
    - Single interface for all safety checks
    - Coordinated multi-component responses
    - Automatic degradation on resource constraints
    - Emergency escalation when needed
    - Comprehensive safety reporting

    Example:
        # Initialize safety system
        safety = SafetyIntegration(
            gpu_isolation_level=IsolationLevel.STRICT,
            memory_soft_limit_mb=4096,
            memory_hard_limit_mb=6144
        )

        # Start monitoring
        safety.start()

        # Periodic safety check
        report = safety.check_safety()
        if not report.can_continue:
            safety.handle_safety_violation(report)

        # Check before starting task
        if safety.is_safe_to_proceed():
            # Execute task
            pass

        # Cleanup
        safety.stop()
    """

    def __init__(
        self,
        gpu_isolation_level: IsolationLevel = IsolationLevel.STRICT,
        memory_soft_limit_mb: int = 4096,
        memory_hard_limit_mb: int = 6144,
        memory_emergency_limit_mb: int = 8192,
        enable_auto_degradation: bool = True,
        enable_emergency_response: bool = True
    ):
        """
        Initialize Safety Integration

        Args:
            gpu_isolation_level: GPU isolation security level
            memory_soft_limit_mb: Memory warning threshold
            memory_hard_limit_mb: Memory degradation threshold
            memory_emergency_limit_mb: Memory emergency threshold
            enable_auto_degradation: Auto-degrade on resource constraints
            enable_emergency_response: Enable emergency response system
        """
        self.enable_auto_degradation = enable_auto_degradation
        self.enable_emergency_response = enable_emergency_response

        # Initialize components
        logger.info("Initializing Safety Integration...")

        # GPU Isolation
        self.gpu_isolation = GPUIsolation(
            level=gpu_isolation_level,
            check_interval=30.0,
            auto_terminate=True
        )

        # Memory Budget
        self.memory_budget = MemoryBudget(
            thresholds=MemoryThresholds(
                soft_limit_mb=memory_soft_limit_mb,
                hard_limit_mb=memory_hard_limit_mb,
                emergency_limit_mb=memory_emergency_limit_mb
            ),
            check_interval=10.0
        )

        # Degradation Manager
        self.degradation = DegradationManager(
            auto_recovery=True,
            recovery_delay=60.0
        )

        # Emergency Handler
        self.emergency = EmergencyHandler(
            auto_checkpoint=True,
            enable_alerts=True
        )

        # Connect components
        self._connect_components()

        # State
        self._running = False
        self._last_check: Optional[float] = None

        logger.info("Safety Integration initialized successfully")

    def _connect_components(self):
        """Connect safety components with callbacks"""
        # Memory Budget → Degradation Manager
        def memory_threshold_callback(status: MemoryStatus):
            if not self.enable_auto_degradation:
                return

            if status.level == MemoryLevel.WARNING and not self.degradation.is_degraded():
                logger.warning("Memory warning detected, applying light degradation")
                self.degradation.apply_degradation(
                    DegradationLevel.LIGHT,
                    reason="Memory usage above soft limit"
                )

            elif status.level == MemoryLevel.CRITICAL:
                logger.error("Memory critical, applying heavy degradation")
                self.degradation.apply_degradation(
                    DegradationLevel.HEAVY,
                    reason="Memory usage above hard limit"
                )

            elif status.level == MemoryLevel.EMERGENCY:
                logger.critical("Memory emergency, triggering emergency response")
                self.emergency.trigger_emergency(
                    EmergencyType.MEMORY_EXHAUSTION,
                    description=f"Memory usage: {status.process_rss_mb:.1f}MB",
                    context=status.to_dict()
                )

        self.memory_budget.register_threshold_callback(memory_threshold_callback)

        # GPU Isolation → Emergency Handler
        def gpu_violation_callback(result: GPUCheckResult):
            if not result.is_safe() and self.enable_emergency_response:
                logger.critical("GPU isolation violation detected")
                self.emergency.trigger_emergency(
                    EmergencyType.GPU_VIOLATION,
                    description=f"GPU violations: {', '.join(result.violations)}",
                    context={
                        "cuda_available": result.cuda_available,
                        "gpu_processes": result.gpu_processes
                    }
                )

        # Note: GPU isolation doesn't have callbacks, so we check manually

    def start(self):
        """Start safety monitoring"""
        if self._running:
            logger.warning("Safety monitoring already running")
            return

        logger.info("Starting safety monitoring...")

        # Enforce GPU isolation
        self.gpu_isolation.enforce()

        # Start memory monitoring
        self.memory_budget.start_monitoring()

        self._running = True
        logger.info("Safety monitoring started")

    def stop(self):
        """Stop safety monitoring"""
        if not self._running:
            return

        logger.info("Stopping safety monitoring...")

        # Stop monitoring
        self.gpu_isolation.stop_monitoring()
        self.memory_budget.stop_monitoring()

        self._running = False
        logger.info("Safety monitoring stopped")

    def check_safety(self) -> SystemSafetyReport:
        """
        Perform comprehensive safety check

        Returns:
            SystemSafetyReport with current status
        """
        self._last_check = time.time()

        # Check GPU
        gpu_result = self.gpu_isolation.check_gpu_usage()
        gpu_safe = gpu_result.is_safe()

        # Check Memory
        memory_status = self.memory_budget.get_status()
        memory_safe = memory_status.is_safe()

        # Get degradation level
        degradation_level = self.degradation.get_level()

        # Get active emergencies
        active_emergencies = len(self.emergency.get_active_emergencies())

        # Determine overall status
        if active_emergencies > 0 or not gpu_safe:
            overall_status = SafetyStatus.EMERGENCY
        elif memory_status.level == MemoryLevel.CRITICAL:
            overall_status = SafetyStatus.CRITICAL
        elif degradation_level in [DegradationLevel.MODERATE, DegradationLevel.HEAVY]:
            overall_status = SafetyStatus.DEGRADED
        elif memory_status.level == MemoryLevel.WARNING or degradation_level == DegradationLevel.LIGHT:
            overall_status = SafetyStatus.WARNING
        else:
            overall_status = SafetyStatus.SAFE

        # Generate recommendations
        can_continue = overall_status not in [SafetyStatus.EMERGENCY]
        should_degrade = memory_status.level in [MemoryLevel.WARNING, MemoryLevel.CRITICAL]
        should_pause = overall_status == SafetyStatus.CRITICAL
        should_stop = overall_status == SafetyStatus.EMERGENCY

        report = SystemSafetyReport(
            timestamp=self._last_check,
            overall_status=overall_status,
            gpu_safe=gpu_safe,
            memory_safe=memory_safe,
            degradation_level=degradation_level,
            active_emergencies=active_emergencies,
            gpu_violations=len(gpu_result.violations),
            memory_level=memory_status.level,
            memory_usage_mb=memory_status.process_rss_mb,
            memory_percent=memory_status.process_percent,
            can_continue=can_continue,
            should_degrade=should_degrade,
            should_pause=should_pause,
            should_stop=should_stop
        )

        return report

    def is_safe_to_proceed(self) -> bool:
        """
        Quick safety check for task execution

        Returns:
            True if safe to proceed with automation tasks
        """
        report = self.check_safety()
        return report.can_continue

    def handle_safety_violation(self, report: SystemSafetyReport):
        """
        Handle safety violation based on report

        Args:
            report: Safety report with violation details
        """
        if report.should_stop:
            logger.critical("Safety violation requires stop")
            self.emergency.trigger_emergency(
                EmergencyType.CRITICAL_FAILURE,
                description="Multiple safety violations detected",
                context=report.to_dict()
            )

        elif report.should_pause:
            logger.error("Safety violation requires pause")
            if self.enable_auto_degradation:
                self.degradation.apply_degradation(
                    DegradationLevel.HEAVY,
                    reason="Critical safety violations"
                )

        elif report.should_degrade:
            logger.warning("Safety violation requires degradation")
            if self.enable_auto_degradation:
                self.degradation.apply_degradation(
                    DegradationLevel.MODERATE,
                    reason="Resource constraints detected"
                )

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all safety systems"""
        return {
            "running": self._running,
            "last_check": self._last_check,
            "gpu_isolation": self.gpu_isolation.get_status(),
            "memory_budget": self.memory_budget.get_statistics(),
            "degradation": self.degradation.get_statistics(),
            "emergency": self.emergency.get_statistics()
        }

    def get_report(self) -> str:
        """Generate comprehensive safety report"""
        report = []
        report.append("=" * 70)
        report.append("SAFETY SYSTEM COMPREHENSIVE REPORT")
        report.append("=" * 70)

        # Overall status
        current_report = self.check_safety()
        report.append(f"\n[Overall Status: {current_report.overall_status.value.upper()}]")
        report.append(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_report.timestamp))}")
        report.append(f"  Can Continue: {'YES' if current_report.can_continue else 'NO'}")

        # Component reports
        report.append("\n" + "-" * 70)
        report.append("GPU ISOLATION")
        report.append("-" * 70)
        # Add summary of GPU status
        gpu_status = self.gpu_isolation.get_status()
        report.append(f"Safe: {gpu_status['safe']}")
        report.append(f"Violations: {gpu_status['violation_count']}")

        report.append("\n" + "-" * 70)
        report.append("MEMORY BUDGET")
        report.append("-" * 70)
        report.append(self.memory_budget.get_report())

        report.append("\n" + "-" * 70)
        report.append("DEGRADATION MANAGER")
        report.append("-" * 70)
        report.append(self.degradation.get_report())

        report.append("\n" + "-" * 70)
        report.append("EMERGENCY HANDLER")
        report.append("-" * 70)
        report.append(self.emergency.get_report())

        report.append("=" * 70)
        return "\n".join(report)

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

    def __repr__(self) -> str:
        report = self.check_safety() if self._running else None
        status = report.overall_status.value if report else "not running"
        return f"SafetyIntegration(status={status}, running={self._running})"
