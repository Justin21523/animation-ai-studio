"""
Model Management Module

Dynamic model loading/unloading for VRAM-constrained environments.

Components:
- VRAMMonitor: Real-time VRAM usage tracking and safety checks
- ServiceController: LLM/SDXL service start/stop control
- ModelManager: Dynamic model switching and service orchestration

Author: Animation AI Studio
Date: 2025-11-17
"""

from .vram_monitor import (
    VRAMMonitor,
    VRAMSnapshot,
    VRAMEstimate,
    check_vram_requirements
)
from .service_controller import (
    ServiceController,
    ServiceStatus
)
from .model_manager import (
    ModelManager,
    ModelState
)

__all__ = [
    # VRAM Monitoring
    "VRAMMonitor",
    "VRAMSnapshot",
    "VRAMEstimate",
    "check_vram_requirements",

    # Service Control
    "ServiceController",
    "ServiceStatus",

    # Model Management
    "ModelManager",
    "ModelState",
]
