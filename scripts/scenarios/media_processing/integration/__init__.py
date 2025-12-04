"""
Media Processing Integration

Integration adapters for orchestration and safety systems.

Author: Animation AI Studio
Date: 2025-12-03
"""

from .orchestration_integration import MediaProcessingEventAdapter
from .safety_integration import MediaProcessingSafetyAdapter

__all__ = [
    "MediaProcessingEventAdapter",
    "MediaProcessingSafetyAdapter"
]
