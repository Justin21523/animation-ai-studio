"""
Safety System for CPU-Only Automation

Ensures 100% safe operation without interfering with GPU training.

Components:
- GPU Isolation: 4-layer protection against GPU usage
- Memory Budget: RAM usage monitoring and limits
- Degradation Manager: Graceful performance reduction
- Emergency Handler: Critical situation handling
- Safety Integration: Unified safety layer

Author: Animation AI Studio
Date: 2025-12-02
"""

from .gpu_isolation import GPUIsolation, GPUCheckResult
from .memory_budget import MemoryBudget, MemoryStatus
from .degradation_manager import DegradationManager, DegradationLevel
from .emergency_handler import EmergencyHandler, EmergencyType
from .safety_integration import SafetyIntegration

__all__ = [
    "GPUIsolation",
    "GPUCheckResult",
    "MemoryBudget",
    "MemoryStatus",
    "DegradationManager",
    "DegradationLevel",
    "EmergencyHandler",
    "EmergencyType",
    "SafetyIntegration"
]
