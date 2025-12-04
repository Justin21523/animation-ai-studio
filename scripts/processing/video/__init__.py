"""
Video Processing Modules
Provides comprehensive video processing tools for frame extraction, interpolation, and synthesis.

Video Processing Components:

1. Frame Extraction:
   - universal_frame_extractor: Universal frame extraction with multiple modes
     * Scene detection mode: Extract frames at scene changes
     * Interval mode: Extract frames at fixed intervals
     * Hybrid mode: Combine scene detection with interval sampling
     * Quality filtering: Blur detection and quality scoring

2. Frame Interpolation:
   - frame_interpolator: High-quality frame interpolation using RIFE
     * Smooth slow-motion effects
     * Frame rate upscaling (e.g., 24fps → 60fps)
     * Multiple quality presets

3. Frame Restoration:
   - frame_restoration_pipeline: Comprehensive frame restoration workflow
     * Super-resolution (RealESRGAN)
     * Denoising and deblurring
     * Face restoration (CodeFormer)
     * Batch processing support

4. Video Synthesis:
   - video_synthesizer: Create videos from image sequences
     * Multiple codec support (H.264, H.265, VP9, ProRes)
     * Audio track integration
     * Custom encoding parameters
     * Subtitle support

5. Dataset Preparation:
   - video_dataset_preparer: Prepare video datasets for training
     * Automatic scene segmentation
     * Quality filtering and deduplication
     * Metadata extraction and organization

Usage Examples:

# Extract frames from video (scene detection)
python scripts/processing/video/universal_frame_extractor.py \
  --input video.mp4 \
  --output frames/ \
  --mode scene \
  --scene-threshold 0.3 \
  --quality high

# Interpolate frames (2x frame rate)
python scripts/processing/video/frame_interpolator.py \
  --input frames/ \
  --output interpolated/ \
  --factor 2 \
  --quality high

# Restore and enhance frames
python scripts/processing/video/frame_restoration_pipeline.py \
  --input frames/ \
  --output restored/ \
  --upscale-factor 2 \
  --face-restore

# Synthesize video from frames
python scripts/processing/video/video_synthesizer.py \
  --input frames/ \
  --output video.mp4 \
  --fps 24 \
  --codec h264 \
  --quality high
"""

# Individual scripts are standalone CLI tools

__all__ = [
    # Main modules are used as CLI scripts
]
