"""
Pose Detectors
Provides pose detection using different backends.

Components:
- mediapipe_detector: MediaPipe-based pose detection
- rtmpose_detector: RTMPose-based pose detection
"""

from .mediapipe_detector import MediaPipePoseDetector
from .rtmpose_detector import RTMPoseDetector

__all__ = [
    'MediaPipePoseDetector',
    'RTMPoseDetector',
]
