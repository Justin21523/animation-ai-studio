"""
Degradation Manager

Graceful performance degradation system for CPU-only automation.
Reduces resource usage when memory/system constraints are detected.

Features:
- Multi-level degradation strategies
- Per-module degradation policies
- Automatic and manual degradation control
- Performance impact tracking
- Graceful recovery when resources available

Author: Animation AI Studio
Date: 2025-12-02
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Performance degradation levels"""
    NONE = "none"              # Full performance
    LIGHT = "light"            # Minor optimizations (10-20% reduction)
    MODERATE = "moderate"      # Noticeable reduction (30-50% reduction)
    HEAVY = "heavy"            # Significant reduction (50-70% reduction)
    EMERGENCY = "emergency"    # Minimal operations only (>70% reduction)


class DegradationStrategy(Enum):
    """Degradation strategies"""
    REDUCE_CONCURRENCY = "reduce_concurrency"     # Limit parallel tasks
    REDUCE_BATCH_SIZE = "reduce_batch_size"       # Smaller batch processing
    REDUCE_QUALITY = "reduce_quality"             # Lower quality settings
    DISABLE_CACHING = "disable_caching"           # Reduce memory footprint
    THROTTLE_RATE = "throttle_rate"               # Slow down processing
    PAUSE_NON_CRITICAL = "pause_non_critical"     # Stop non-essential tasks
    EMERGENCY_STOP = "emergency_stop"             # Stop all automation


@dataclass
class DegradationPolicy:
    """Degradation policy for a specific level"""
    level: DegradationLevel
    strategies: List[DegradationStrategy]
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"Degradation policy for {self.level.value}"


@dataclass
class DegradationState:
    """Current degradation state"""
    level: DegradationLevel = DegradationLevel.NONE
    active_strategies: List[DegradationStrategy] = field(default_factory=list)
    applied_at: Optional[float] = None
    reason: str = ""

    # Performance metrics
    original_concurrency: int = 0
    current_concurrency: int = 0
    original_batch_size: int = 0
    current_batch_size: int = 0

    # Statistics
    degradation_count: int = 0
    total_degraded_time: float = 0.0

    def is_degraded(self) -> bool:
        """Check if any degradation is active"""
        return self.level != DegradationLevel.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "level": self.level.value,
            "active_strategies": [s.value for s in self.active_strategies],
            "applied_at": self.applied_at,
            "reason": self.reason,
            "performance": {
                "original_concurrency": self.original_concurrency,
                "current_concurrency": self.current_concurrency,
                "original_batch_size": self.original_batch_size,
                "current_batch_size": self.current_batch_size
            },
            "statistics": {
                "degradation_count": self.degradation_count,
                "total_degraded_time": self.total_degraded_time
            }
        }


class DegradationManager:
    """
    Degradation Manager

    Provides graceful performance degradation when resource constraints detected.

    Features:
    - Multi-level degradation (NONE → LIGHT → MODERATE → HEAVY → EMERGENCY)
    - Configurable degradation policies per level
    - Automatic degradation based on memory/system status
    - Manual degradation control
    - Performance tracking and statistics
    - Graceful recovery when resources available

    Degradation Strategies:
    - REDUCE_CONCURRENCY: Limit number of parallel tasks
    - REDUCE_BATCH_SIZE: Process smaller batches
    - REDUCE_QUALITY: Lower quality settings (e.g., smaller images)
    - DISABLE_CACHING: Clear caches to free memory
    - THROTTLE_RATE: Add delays between operations
    - PAUSE_NON_CRITICAL: Stop non-essential background tasks
    - EMERGENCY_STOP: Stop all automation

    Example:
        # Configure policies
        policies = {
            DegradationLevel.LIGHT: DegradationPolicy(
                level=DegradationLevel.LIGHT,
                strategies=[DegradationStrategy.REDUCE_CONCURRENCY],
                parameters={"max_concurrent_tasks": 2}
            ),
            DegradationLevel.MODERATE: DegradationPolicy(
                level=DegradationLevel.MODERATE,
                strategies=[
                    DegradationStrategy.REDUCE_CONCURRENCY,
                    DegradationStrategy.REDUCE_BATCH_SIZE
                ],
                parameters={
                    "max_concurrent_tasks": 1,
                    "batch_size_multiplier": 0.5
                }
            )
        }

        manager = DegradationManager(policies=policies)

        # Apply degradation
        manager.apply_degradation(
            DegradationLevel.MODERATE,
            reason="Memory usage at 85%"
        )

        # Check if degraded
        if manager.is_degraded():
            concurrency = manager.get_max_concurrency()
            batch_size = manager.get_batch_size(original=32)

        # Recover when safe
        manager.recover(reason="Memory usage normalized")
    """

    def __init__(
        self,
        policies: Optional[Dict[DegradationLevel, DegradationPolicy]] = None,
        auto_recovery: bool = True,
        recovery_delay: float = 60.0
    ):
        """
        Initialize Degradation Manager

        Args:
            policies: Degradation policies per level
            auto_recovery: Automatically recover when safe
            recovery_delay: Delay before attempting recovery (seconds)
        """
        self.policies = policies or self._create_default_policies()
        self.auto_recovery = auto_recovery
        self.recovery_delay = recovery_delay

        # Current state
        self.state = DegradationState()

        # History
        self._history: List[DegradationState] = []

        # Module-specific overrides
        self._module_overrides: Dict[str, DegradationLevel] = {}

        # Callbacks
        self._degradation_callbacks: List[Callable[[DegradationState], None]] = []
        self._recovery_callbacks: List[Callable[[DegradationState], None]] = []

        logger.info(f"DegradationManager initialized with {len(self.policies)} policies")

    def _create_default_policies(self) -> Dict[DegradationLevel, DegradationPolicy]:
        """Create default degradation policies"""
        return {
            DegradationLevel.NONE: DegradationPolicy(
                level=DegradationLevel.NONE,
                strategies=[],
                description="Full performance, no degradation"
            ),
            DegradationLevel.LIGHT: DegradationPolicy(
                level=DegradationLevel.LIGHT,
                strategies=[DegradationStrategy.REDUCE_CONCURRENCY],
                parameters={
                    "max_concurrent_tasks": 2,
                    "concurrency_multiplier": 0.8
                },
                description="Light degradation: reduce concurrency by 20%"
            ),
            DegradationLevel.MODERATE: DegradationPolicy(
                level=DegradationLevel.MODERATE,
                strategies=[
                    DegradationStrategy.REDUCE_CONCURRENCY,
                    DegradationStrategy.REDUCE_BATCH_SIZE
                ],
                parameters={
                    "max_concurrent_tasks": 1,
                    "concurrency_multiplier": 0.5,
                    "batch_size_multiplier": 0.6
                },
                description="Moderate degradation: reduce concurrency 50%, batch size 40%"
            ),
            DegradationLevel.HEAVY: DegradationPolicy(
                level=DegradationLevel.HEAVY,
                strategies=[
                    DegradationStrategy.REDUCE_CONCURRENCY,
                    DegradationStrategy.REDUCE_BATCH_SIZE,
                    DegradationStrategy.DISABLE_CACHING,
                    DegradationStrategy.THROTTLE_RATE
                ],
                parameters={
                    "max_concurrent_tasks": 1,
                    "concurrency_multiplier": 0.3,
                    "batch_size_multiplier": 0.4,
                    "throttle_delay": 0.5
                },
                description="Heavy degradation: minimal concurrency, small batches, throttling"
            ),
            DegradationLevel.EMERGENCY: DegradationPolicy(
                level=DegradationLevel.EMERGENCY,
                strategies=[
                    DegradationStrategy.PAUSE_NON_CRITICAL,
                    DegradationStrategy.EMERGENCY_STOP
                ],
                parameters={
                    "max_concurrent_tasks": 0,
                    "allow_critical_only": True
                },
                description="Emergency: stop all non-critical automation"
            )
        }

    def apply_degradation(
        self,
        level: DegradationLevel,
        reason: str = "",
        original_concurrency: Optional[int] = None,
        original_batch_size: Optional[int] = None
    ):
        """
        Apply degradation at specified level

        Args:
            level: Degradation level to apply
            reason: Reason for degradation
            original_concurrency: Original max concurrent tasks
            original_batch_size: Original batch size
        """
        if level not in self.policies:
            logger.error(f"Unknown degradation level: {level}")
            return

        policy = self.policies[level]

        # Save previous state to history
        if self.state.applied_at is not None:
            previous_state = DegradationState(
                level=self.state.level,
                active_strategies=self.state.active_strategies.copy(),
                applied_at=self.state.applied_at,
                reason=self.state.reason
            )
            self._history.append(previous_state)

        # Update state
        old_level = self.state.level
        self.state.level = level
        self.state.active_strategies = policy.strategies.copy()
        self.state.applied_at = time.time()
        self.state.reason = reason
        self.state.degradation_count += 1

        # Track original values
        if original_concurrency is not None:
            self.state.original_concurrency = original_concurrency
        if original_batch_size is not None:
            self.state.original_batch_size = original_batch_size

        # Calculate degraded values
        if DegradationStrategy.REDUCE_CONCURRENCY in policy.strategies:
            multiplier = policy.parameters.get("concurrency_multiplier", 0.5)
            self.state.current_concurrency = max(
                1,
                int(self.state.original_concurrency * multiplier)
            )

        if DegradationStrategy.REDUCE_BATCH_SIZE in policy.strategies:
            multiplier = policy.parameters.get("batch_size_multiplier", 0.5)
            self.state.current_batch_size = max(
                1,
                int(self.state.original_batch_size * multiplier)
            )

        logger.warning(
            f"Degradation applied: {old_level.value} → {level.value} "
            f"(reason: {reason})"
        )
        logger.info(f"Active strategies: {[s.value for s in policy.strategies]}")

        # Trigger callbacks
        self._trigger_degradation_callbacks()

    def recover(self, reason: str = ""):
        """
        Recover from degradation back to normal performance

        Args:
            reason: Reason for recovery
        """
        if not self.is_degraded():
            logger.debug("Already at normal performance, no recovery needed")
            return

        # Calculate total degraded time
        if self.state.applied_at is not None:
            degraded_duration = time.time() - self.state.applied_at
            self.state.total_degraded_time += degraded_duration

        old_level = self.state.level

        # Reset to normal
        self.state.level = DegradationLevel.NONE
        self.state.active_strategies = []
        self.state.reason = reason
        self.state.current_concurrency = self.state.original_concurrency
        self.state.current_batch_size = self.state.original_batch_size

        logger.info(
            f"Recovered from degradation: {old_level.value} → NONE "
            f"(reason: {reason})"
        )

        # Trigger callbacks
        self._trigger_recovery_callbacks()

    def is_degraded(self) -> bool:
        """Check if any degradation is currently active"""
        return self.state.is_degraded()

    def get_level(self) -> DegradationLevel:
        """Get current degradation level"""
        return self.state.level

    def get_max_concurrency(self, original: int = 4) -> int:
        """
        Get max concurrent tasks based on degradation level

        Args:
            original: Original max concurrent tasks

        Returns:
            Adjusted max concurrent tasks
        """
        if not self.is_degraded():
            return original

        policy = self.policies[self.state.level]

        if DegradationStrategy.REDUCE_CONCURRENCY in policy.strategies:
            # Use configured max or calculate from multiplier
            if "max_concurrent_tasks" in policy.parameters:
                return policy.parameters["max_concurrent_tasks"]
            else:
                multiplier = policy.parameters.get("concurrency_multiplier", 0.5)
                return max(1, int(original * multiplier))

        return original

    def get_batch_size(self, original: int) -> int:
        """
        Get batch size based on degradation level

        Args:
            original: Original batch size

        Returns:
            Adjusted batch size
        """
        if not self.is_degraded():
            return original

        policy = self.policies[self.state.level]

        if DegradationStrategy.REDUCE_BATCH_SIZE in policy.strategies:
            multiplier = policy.parameters.get("batch_size_multiplier", 0.5)
            return max(1, int(original * multiplier))

        return original

    def should_throttle(self) -> bool:
        """Check if throttling is active"""
        return DegradationStrategy.THROTTLE_RATE in self.state.active_strategies

    def get_throttle_delay(self) -> float:
        """Get throttle delay in seconds"""
        if not self.should_throttle():
            return 0.0

        policy = self.policies[self.state.level]
        return policy.parameters.get("throttle_delay", 0.5)

    def is_caching_disabled(self) -> bool:
        """Check if caching is disabled"""
        return DegradationStrategy.DISABLE_CACHING in self.state.active_strategies

    def is_strategy_active(self, strategy: DegradationStrategy) -> bool:
        """Check if specific strategy is active"""
        return strategy in self.state.active_strategies

    def set_module_override(self, module_name: str, level: DegradationLevel):
        """
        Set degradation override for specific module

        Args:
            module_name: Module identifier
            level: Degradation level for this module
        """
        self._module_overrides[module_name] = level
        logger.info(f"Module override set: {module_name} → {level.value}")

    def get_module_level(self, module_name: str) -> DegradationLevel:
        """
        Get degradation level for specific module

        Args:
            module_name: Module identifier

        Returns:
            Degradation level (override or global)
        """
        return self._module_overrides.get(module_name, self.state.level)

    def register_degradation_callback(self, callback: Callable[[DegradationState], None]):
        """Register callback for degradation events"""
        self._degradation_callbacks.append(callback)

    def register_recovery_callback(self, callback: Callable[[DegradationState], None]):
        """Register callback for recovery events"""
        self._recovery_callbacks.append(callback)

    def _trigger_degradation_callbacks(self):
        """Trigger degradation callbacks"""
        for callback in self._degradation_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}", exc_info=True)

    def _trigger_recovery_callbacks(self):
        """Trigger recovery callbacks"""
        for callback in self._recovery_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get degradation statistics"""
        return {
            "current_level": self.state.level.value,
            "is_degraded": self.is_degraded(),
            "degradation_count": self.state.degradation_count,
            "total_degraded_time": self.state.total_degraded_time,
            "history_count": len(self._history),
            "module_overrides": {
                name: level.value for name, level in self._module_overrides.items()
            }
        }

    def get_report(self) -> str:
        """Generate human-readable degradation report"""
        report = []
        report.append("=" * 60)
        report.append("Degradation Manager Report")
        report.append("=" * 60)

        # Current state
        report.append("\n[Current State]")
        report.append(f"  Level: {self.state.level.value.upper()}")
        report.append(f"  Degraded: {'YES' if self.is_degraded() else 'NO'}")

        if self.is_degraded():
            report.append(f"  Reason: {self.state.reason}")
            report.append(f"  Applied: {time.time() - self.state.applied_at:.1f}s ago")
            report.append(f"  Strategies: {', '.join([s.value for s in self.state.active_strategies])}")

        # Performance impact
        if self.state.original_concurrency > 0:
            report.append("\n[Performance Impact]")
            report.append(
                f"  Concurrency: {self.state.original_concurrency} → "
                f"{self.state.current_concurrency} "
                f"({(1 - self.state.current_concurrency/self.state.original_concurrency)*100:.0f}% reduction)"
            )

        if self.state.original_batch_size > 0:
            report.append(
                f"  Batch Size: {self.state.original_batch_size} → "
                f"{self.state.current_batch_size} "
                f"({(1 - self.state.current_batch_size/self.state.original_batch_size)*100:.0f}% reduction)"
            )

        # Statistics
        report.append("\n[Statistics]")
        report.append(f"  Degradation Count: {self.state.degradation_count}")
        report.append(f"  Total Degraded Time: {self.state.total_degraded_time:.1f}s")
        report.append(f"  History Records: {len(self._history)}")

        # Module overrides
        if self._module_overrides:
            report.append("\n[Module Overrides]")
            for module, level in self._module_overrides.items():
                report.append(f"  {module}: {level.value}")

        report.append("=" * 60)
        return "\n".join(report)

    def __repr__(self) -> str:
        return (
            f"DegradationManager(level={self.state.level.value}, "
            f"degraded={self.is_degraded()}, "
            f"count={self.state.degradation_count})"
        )
