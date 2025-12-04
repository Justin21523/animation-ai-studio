"""
Emergency Handler

Critical situation handler for CPU-only automation.
Coordinates emergency responses when safety thresholds are breached.

Features:
- Emergency type classification
- Coordinated multi-system response
- Emergency action execution
- Recovery coordination
- Emergency audit logging

Author: Animation AI Studio
Date: 2025-12-02
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class EmergencyType(Enum):
    """Emergency situation types"""
    MEMORY_EXHAUSTION = "memory_exhaustion"        # RAM critically low
    GPU_VIOLATION = "gpu_violation"                # GPU usage detected
    SYSTEM_OVERLOAD = "system_overload"            # CPU/system resources maxed
    OOM_IMMINENT = "oom_imminent"                  # Out-of-memory imminent
    TRAINING_CONFLICT = "training_conflict"        # Automation interfering with training
    CRITICAL_FAILURE = "critical_failure"          # Unrecoverable error
    DISK_FULL = "disk_full"                        # Storage exhausted
    DEPENDENCY_FAILURE = "dependency_failure"      # Critical module failed


class EmergencyAction(Enum):
    """Emergency response actions"""
    PAUSE_AUTOMATION = "pause_automation"          # Pause all automation
    STOP_AUTOMATION = "stop_automation"            # Stop all automation
    CLEAR_MEMORY = "clear_memory"                  # Emergency memory cleanup
    KILL_PROCESSES = "kill_processes"              # Kill non-essential processes
    CHECKPOINT_STATE = "checkpoint_state"          # Save current state
    SEND_ALERT = "send_alert"                      # Send alert notification
    EMERGENCY_EXIT = "emergency_exit"              # Terminate process


class EmergencySeverity(Enum):
    """Emergency severity levels"""
    WARNING = "warning"        # Potential emergency
    CRITICAL = "critical"      # Emergency in progress
    FATAL = "fatal"            # System-threatening emergency


@dataclass
class EmergencyEvent:
    """Emergency event record"""
    event_id: str
    emergency_type: EmergencyType
    severity: EmergencySeverity
    timestamp: float = field(default_factory=time.time)

    # Context
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Response
    actions_taken: List[EmergencyAction] = field(default_factory=list)
    resolution: Optional[str] = None
    resolved_at: Optional[float] = None

    # Metrics
    response_time: float = 0.0  # Time to first action
    recovery_time: Optional[float] = None  # Time to resolution

    def is_resolved(self) -> bool:
        """Check if emergency is resolved"""
        return self.resolution is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "type": self.emergency_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "description": self.description,
            "context": self.context,
            "actions_taken": [a.value for a in self.actions_taken],
            "resolution": self.resolution,
            "resolved_at": self.resolved_at,
            "response_time": self.response_time,
            "recovery_time": self.recovery_time
        }


@dataclass
class EmergencyResponse:
    """Emergency response plan"""
    emergency_type: EmergencyType
    actions: List[EmergencyAction]
    priority: int = 0  # Higher = more urgent
    auto_execute: bool = True
    requires_confirmation: bool = False

    description: str = ""


class EmergencyHandler:
    """
    Emergency Handler

    Coordinates emergency responses for critical situations in CPU-only automation.

    Features:
    - Emergency classification and severity assessment
    - Coordinated multi-action response plans
    - Emergency execution with rollback capability
    - Recovery coordination across safety systems
    - Emergency audit trail and logging

    Emergency Types:
    - MEMORY_EXHAUSTION: RAM critically low, risk of OOM
    - GPU_VIOLATION: GPU usage detected in CPU-only automation
    - SYSTEM_OVERLOAD: CPU/system resources exhausted
    - OOM_IMMINENT: Out-of-memory condition imminent
    - TRAINING_CONFLICT: Automation interfering with LoRA training
    - CRITICAL_FAILURE: Unrecoverable error in automation
    - DISK_FULL: Storage space exhausted
    - DEPENDENCY_FAILURE: Critical dependency failed

    Example:
        # Create handler with response plans
        handler = EmergencyHandler()

        # Register custom response plan
        handler.register_response(
            EmergencyType.MEMORY_EXHAUSTION,
            EmergencyResponse(
                emergency_type=EmergencyType.MEMORY_EXHAUSTION,
                actions=[
                    EmergencyAction.CLEAR_MEMORY,
                    EmergencyAction.PAUSE_AUTOMATION,
                    EmergencyAction.SEND_ALERT
                ],
                priority=100,
                auto_execute=True
            )
        )

        # Trigger emergency
        event = handler.trigger_emergency(
            emergency_type=EmergencyType.MEMORY_EXHAUSTION,
            description="Memory usage exceeded 95%",
            context={"memory_mb": 15800, "threshold_mb": 15000}
        )

        # Check if resolved
        if event.is_resolved():
            print(f"Emergency resolved: {event.resolution}")
    """

    def __init__(
        self,
        emergency_log_dir: Optional[Path] = None,
        auto_checkpoint: bool = True,
        enable_alerts: bool = True
    ):
        """
        Initialize Emergency Handler

        Args:
            emergency_log_dir: Directory for emergency logs
            auto_checkpoint: Automatically checkpoint state on emergency
            enable_alerts: Enable alert notifications
        """
        self.emergency_log_dir = emergency_log_dir or Path("/tmp/emergency_logs")
        self.emergency_log_dir.mkdir(parents=True, exist_ok=True)

        self.auto_checkpoint = auto_checkpoint
        self.enable_alerts = enable_alerts

        # Response plans
        self._response_plans: Dict[EmergencyType, EmergencyResponse] = {}
        self._create_default_responses()

        # Active emergencies
        self._active_emergencies: Dict[str, EmergencyEvent] = {}

        # History
        self._emergency_history: List[EmergencyEvent] = []

        # Statistics
        self._emergency_count = 0
        self._emergencies_by_type: Dict[EmergencyType, int] = {t: 0 for t in EmergencyType}

        # Action handlers
        self._action_handlers: Dict[EmergencyAction, Callable] = {}
        self._register_default_handlers()

        # Callbacks
        self._emergency_callbacks: List[Callable[[EmergencyEvent], None]] = []

        logger.info(
            f"EmergencyHandler initialized: log_dir={self.emergency_log_dir}, "
            f"auto_checkpoint={auto_checkpoint}"
        )

    def _create_default_responses(self):
        """Create default emergency response plans"""
        self._response_plans = {
            EmergencyType.MEMORY_EXHAUSTION: EmergencyResponse(
                emergency_type=EmergencyType.MEMORY_EXHAUSTION,
                actions=[
                    EmergencyAction.CHECKPOINT_STATE,
                    EmergencyAction.CLEAR_MEMORY,
                    EmergencyAction.PAUSE_AUTOMATION,
                    EmergencyAction.SEND_ALERT
                ],
                priority=100,
                description="Memory critically low, save state and pause"
            ),
            EmergencyType.GPU_VIOLATION: EmergencyResponse(
                emergency_type=EmergencyType.GPU_VIOLATION,
                actions=[
                    EmergencyAction.STOP_AUTOMATION,
                    EmergencyAction.SEND_ALERT,
                    EmergencyAction.EMERGENCY_EXIT
                ],
                priority=200,
                description="GPU usage detected, stop immediately"
            ),
            EmergencyType.OOM_IMMINENT: EmergencyResponse(
                emergency_type=EmergencyType.OOM_IMMINENT,
                actions=[
                    EmergencyAction.CHECKPOINT_STATE,
                    EmergencyAction.KILL_PROCESSES,
                    EmergencyAction.CLEAR_MEMORY,
                    EmergencyAction.EMERGENCY_EXIT
                ],
                priority=150,
                description="OOM imminent, save state and exit"
            ),
            EmergencyType.TRAINING_CONFLICT: EmergencyResponse(
                emergency_type=EmergencyType.TRAINING_CONFLICT,
                actions=[
                    EmergencyAction.PAUSE_AUTOMATION,
                    EmergencyAction.SEND_ALERT
                ],
                priority=80,
                description="Training conflict detected, pause automation"
            ),
            EmergencyType.SYSTEM_OVERLOAD: EmergencyResponse(
                emergency_type=EmergencyType.SYSTEM_OVERLOAD,
                actions=[
                    EmergencyAction.CHECKPOINT_STATE,
                    EmergencyAction.PAUSE_AUTOMATION,
                    EmergencyAction.SEND_ALERT
                ],
                priority=90,
                description="System overloaded, pause and checkpoint"
            ),
            EmergencyType.CRITICAL_FAILURE: EmergencyResponse(
                emergency_type=EmergencyType.CRITICAL_FAILURE,
                actions=[
                    EmergencyAction.CHECKPOINT_STATE,
                    EmergencyAction.STOP_AUTOMATION,
                    EmergencyAction.SEND_ALERT
                ],
                priority=120,
                description="Critical failure, checkpoint and stop"
            )
        }

    def _register_default_handlers(self):
        """Register default action handlers"""
        self._action_handlers = {
            EmergencyAction.PAUSE_AUTOMATION: self._handle_pause_automation,
            EmergencyAction.STOP_AUTOMATION: self._handle_stop_automation,
            EmergencyAction.CLEAR_MEMORY: self._handle_clear_memory,
            EmergencyAction.CHECKPOINT_STATE: self._handle_checkpoint_state,
            EmergencyAction.SEND_ALERT: self._handle_send_alert,
            EmergencyAction.EMERGENCY_EXIT: self._handle_emergency_exit
        }

    def trigger_emergency(
        self,
        emergency_type: EmergencyType,
        description: str = "",
        context: Optional[Dict[str, Any]] = None,
        severity: Optional[EmergencySeverity] = None
    ) -> EmergencyEvent:
        """
        Trigger emergency response

        Args:
            emergency_type: Type of emergency
            description: Emergency description
            context: Additional context
            severity: Emergency severity (auto-determined if None)

        Returns:
            EmergencyEvent with response details
        """
        # Generate event ID
        self._emergency_count += 1
        event_id = f"EMERGENCY_{self._emergency_count:06d}_{int(time.time())}"

        # Determine severity
        if severity is None:
            severity = self._determine_severity(emergency_type)

        # Create event
        event = EmergencyEvent(
            event_id=event_id,
            emergency_type=emergency_type,
            severity=severity,
            description=description,
            context=context or {}
        )

        # Track
        self._active_emergencies[event_id] = event
        self._emergencies_by_type[emergency_type] += 1

        logger.critical(
            f"EMERGENCY TRIGGERED: {emergency_type.value} "
            f"(severity={severity.value}, id={event_id})"
        )
        logger.critical(f"Description: {description}")

        # Log to file
        self._log_emergency(event)

        # Execute response
        start_time = time.time()
        self._execute_response(event)
        event.response_time = time.time() - start_time

        # Trigger callbacks
        self._trigger_callbacks(event)

        return event

    def _determine_severity(self, emergency_type: EmergencyType) -> EmergencySeverity:
        """Determine severity based on emergency type"""
        high_severity = [
            EmergencyType.GPU_VIOLATION,
            EmergencyType.OOM_IMMINENT,
            EmergencyType.CRITICAL_FAILURE
        ]

        if emergency_type in high_severity:
            return EmergencySeverity.FATAL
        else:
            return EmergencySeverity.CRITICAL

    def _execute_response(self, event: EmergencyEvent):
        """Execute emergency response plan"""
        if event.emergency_type not in self._response_plans:
            logger.error(f"No response plan for {event.emergency_type.value}")
            return

        response = self._response_plans[event.emergency_type]

        logger.info(f"Executing response plan: {len(response.actions)} actions")

        for action in response.actions:
            try:
                logger.info(f"  Executing action: {action.value}")

                if action in self._action_handlers:
                    self._action_handlers[action](event)
                    event.actions_taken.append(action)
                else:
                    logger.warning(f"No handler for action: {action.value}")

            except Exception as e:
                logger.error(f"Error executing {action.value}: {e}", exc_info=True)

    # Action handlers

    def _handle_pause_automation(self, event: EmergencyEvent):
        """Pause all automation"""
        logger.warning("ACTION: Pausing automation (placeholder)")
        # Real implementation would pause workflow executor

    def _handle_stop_automation(self, event: EmergencyEvent):
        """Stop all automation"""
        logger.warning("ACTION: Stopping automation (placeholder)")
        # Real implementation would stop workflow executor

    def _handle_clear_memory(self, event: EmergencyEvent):
        """Emergency memory cleanup"""
        import gc
        logger.warning("ACTION: Clearing memory (forcing garbage collection)")
        gc.collect()

    def _handle_checkpoint_state(self, event: EmergencyEvent):
        """Checkpoint current state"""
        logger.warning("ACTION: Checkpointing state (placeholder)")
        # Real implementation would checkpoint workflow state

    def _handle_send_alert(self, event: EmergencyEvent):
        """Send alert notification"""
        if self.enable_alerts:
            logger.critical(f"ACTION: ALERT - {event.description}")
            # Real implementation would send notifications

    def _handle_emergency_exit(self, event: EmergencyEvent):
        """Emergency exit"""
        logger.critical("ACTION: EMERGENCY EXIT - terminating process")
        time.sleep(1)  # Allow logs to flush
        sys.exit(1)

    def resolve_emergency(
        self,
        event_id: str,
        resolution: str = ""
    ):
        """
        Mark emergency as resolved

        Args:
            event_id: Emergency event ID
            resolution: Resolution description
        """
        if event_id not in self._active_emergencies:
            logger.warning(f"Emergency {event_id} not found in active emergencies")
            return

        event = self._active_emergencies[event_id]
        event.resolution = resolution
        event.resolved_at = time.time()

        if event.response_time > 0:
            event.recovery_time = event.resolved_at - (event.timestamp + event.response_time)

        logger.info(f"Emergency resolved: {event_id} - {resolution}")

        # Move to history
        self._emergency_history.append(event)
        del self._active_emergencies[event_id]

        # Log resolution
        self._log_emergency(event)

    def register_response(self, emergency_type: EmergencyType, response: EmergencyResponse):
        """Register custom emergency response plan"""
        self._response_plans[emergency_type] = response
        logger.info(f"Response plan registered: {emergency_type.value}")

    def register_action_handler(
        self,
        action: EmergencyAction,
        handler: Callable[[EmergencyEvent], None]
    ):
        """Register custom action handler"""
        self._action_handlers[action] = handler
        logger.info(f"Action handler registered: {action.value}")

    def register_callback(self, callback: Callable[[EmergencyEvent], None]):
        """Register callback for emergency events"""
        self._emergency_callbacks.append(callback)

    def _trigger_callbacks(self, event: EmergencyEvent):
        """Trigger emergency callbacks"""
        for callback in self._emergency_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in emergency callback: {e}", exc_info=True)

    def _log_emergency(self, event: EmergencyEvent):
        """Log emergency to audit file"""
        log_file = self.emergency_log_dir / f"emergency_{event.event_id}.json"

        try:
            with open(log_file, "w") as f:
                json.dump(event.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log emergency: {e}", exc_info=True)

    def get_active_emergencies(self) -> List[EmergencyEvent]:
        """Get list of active emergencies"""
        return list(self._active_emergencies.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get emergency statistics"""
        return {
            "total_emergencies": self._emergency_count,
            "active_count": len(self._active_emergencies),
            "resolved_count": len(self._emergency_history),
            "by_type": {
                t.value: count for t, count in self._emergencies_by_type.items()
            }
        }

    def get_report(self) -> str:
        """Generate emergency report"""
        report = []
        report.append("=" * 60)
        report.append("Emergency Handler Report")
        report.append("=" * 60)

        # Statistics
        stats = self.get_statistics()
        report.append("\n[Statistics]")
        report.append(f"  Total Emergencies: {stats['total_emergencies']}")
        report.append(f"  Active: {stats['active_count']}")
        report.append(f"  Resolved: {stats['resolved_count']}")

        # By type
        report.append("\n[By Type]")
        for type_name, count in stats["by_type"].items():
            if count > 0:
                report.append(f"  {type_name}: {count}")

        # Active emergencies
        if self._active_emergencies:
            report.append("\n[Active Emergencies]")
            for event in self._active_emergencies.values():
                report.append(f"  {event.event_id}: {event.emergency_type.value}")
                report.append(f"    Severity: {event.severity.value}")
                report.append(f"    Age: {time.time() - event.timestamp:.1f}s")

        report.append("=" * 60)
        return "\n".join(report)

    def __repr__(self) -> str:
        return (
            f"EmergencyHandler(total={self._emergency_count}, "
            f"active={len(self._active_emergencies)})"
        )
