"""
Segmentation Modules
Provides character segmentation tools for 2D and 3D animation.

Available Methods:
1. Two-Stage Strategy (RECOMMENDED for efficiency):
   - YOLO Detection → Fine Segmentation (MobileSAM / ToonOut)
   - Optimized for both 2D anime and 3D animated characters
   - Much faster, lower compute requirements

2. SAM2 Instance Segmentation (Alternative option):
   - Full-frame instance segmentation
   - More thorough but slower
   - Better for complex multi-character scenes

Components:
- yolo_sam_segmentation: Two-stage YOLO + SAM/ToonOut (RECOMMENDED)
- instance_segmentation: SAM2-based instance segmentation (ALTERNATIVE)
- reprocess_frames: Frame reprocessing utilities
- sample_frames: Frame sampling tools
"""

# Note: Individual scripts are standalone CLI tools
# Import classes if needed for programmatic use

__all__ = [
    # Main modules are used as CLI scripts
]
