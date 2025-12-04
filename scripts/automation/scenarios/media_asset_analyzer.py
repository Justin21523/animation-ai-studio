"""
Media Asset Analyzer

Analyzes video/image media assets to extract metadata, detect scenes, and assess quality.
All processing is CPU-only to avoid interfering with GPU training.

Features:
  - Scene detection using PySceneDetect (CPU-only)
  - Video metadata extraction (resolution, FPS, duration, codec)
  - Quality assessment (blur detection, brightness, contrast)
  - Frame sampling with quality filtering
  - JSON report generation

Usage:
  python scripts/automation/scenarios/media_asset_analyzer.py \
    --input /path/to/video.mp4 \
    --output /path/to/analysis_report.json \
    --extract-frames \
    --frames-dir /path/to/frames/

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure
from scripts.core.safety import (
    enforce_cpu_only,
    verify_no_gpu_usage,
    MemoryMonitor,
    RuntimeMonitor,
    run_preflight,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Video Metadata Extraction
# ============================================================================

def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe (CPU-only).

    Args:
        video_path: Path to video file

    Returns:
        Dict with metadata (duration, resolution, fps, codec, bitrate, etc.)
    """
    logger.info(f"Extracting metadata from: {video_path}")

    try:
        # Use ffprobe to get video info
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = next(
            (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
            None
        )

        if not video_stream:
            raise ValueError("No video stream found in file")

        # Extract key metadata
        metadata = {
            'filepath': str(video_path),
            'filesize_mb': video_path.stat().st_size / (1024 * 1024),
            'duration_seconds': float(data.get('format', {}).get('duration', 0)),
            'bitrate_kbps': int(data.get('format', {}).get('bit_rate', 0)) / 1000,

            # Video stream info
            'codec': video_stream.get('codec_name'),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),  # e.g., "24/1" -> 24.0
            'total_frames': int(video_stream.get('nb_frames', 0)),
            'pixel_format': video_stream.get('pix_fmt'),

            # Calculated fields
            'resolution': f"{video_stream.get('width')}x{video_stream.get('height')}",
            'aspect_ratio': f"{video_stream.get('width', 0) / max(video_stream.get('height', 1), 1):.2f}",
        }

        logger.info(f"  Duration: {metadata['duration_seconds']:.1f}s")
        logger.info(f"  Resolution: {metadata['resolution']} @ {metadata['fps']:.1f} FPS")
        logger.info(f"  Codec: {metadata['codec']}")

        return metadata

    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        raise


# ============================================================================
# Scene Detection (CPU-only)
# ============================================================================

def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_length: int = 15
) -> List[Dict[str, Any]]:
    """
    Detect scenes using PySceneDetect (CPU-only).

    Args:
        video_path: Path to video file
        threshold: Scene detection threshold (default 27.0 for content-aware)
        min_scene_length: Minimum scene length in frames

    Returns:
        List of scenes with start/end frames and timecodes
    """
    logger.info(f"Detecting scenes in: {video_path}")
    logger.info(f"  Threshold: {threshold}, Min length: {min_scene_length} frames")

    try:
        from scenedetect import open_video, SceneManager, ContentDetector

        # Open video
        video = open_video(str(video_path))

        # Create scene manager and add detector
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_length)
        )

        # Detect scenes
        scene_manager.detect_scenes(video)

        # Get scene list
        scene_list = scene_manager.get_scene_list()

        # Convert to dict format
        scenes = []
        for i, (start_time, end_time) in enumerate(scene_list):
            scene = {
                'scene_id': i,
                'start_frame': start_time.get_frames(),
                'end_frame': end_time.get_frames(),
                'start_time': start_time.get_seconds(),
                'end_time': end_time.get_seconds(),
                'duration_frames': end_time.get_frames() - start_time.get_frames(),
                'duration_seconds': end_time.get_seconds() - start_time.get_seconds(),
            }
            scenes.append(scene)

        logger.info(f"  Detected {len(scenes)} scenes")

        return scenes

    except ImportError:
        logger.error("scenedetect not installed. Install with: pip install scenedetect[opencv]")
        raise
    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        raise


# ============================================================================
# Frame Quality Assessment (CPU-only)
# ============================================================================

def assess_frame_quality(frame_path: Path) -> Dict[str, Any]:
    """
    Assess frame quality using PIL and NumPy (CPU-only).

    Args:
        frame_path: Path to frame image

    Returns:
        Dict with quality metrics (blur, brightness, contrast, etc.)
    """
    try:
        from PIL import Image
        import numpy as np

        # Open image
        img = Image.open(frame_path)

        # Convert to grayscale for analysis
        gray = img.convert('L')
        gray_array = np.array(gray)

        # Calculate metrics
        metrics = {
            'width': img.width,
            'height': img.height,
            'mode': img.mode,

            # Brightness (mean intensity)
            'brightness': float(np.mean(gray_array)),

            # Contrast (std deviation of intensity)
            'contrast': float(np.std(gray_array)),

            # Blur detection (Laplacian variance - lower = more blurry)
            'sharpness': _calculate_laplacian_variance(gray_array),
        }

        return metrics

    except Exception as e:
        logger.warning(f"Failed to assess frame quality for {frame_path}: {e}")
        return {}


def _calculate_laplacian_variance(gray_array: 'np.ndarray') -> float:
    """
    Calculate Laplacian variance for blur detection.
    Lower values indicate more blur.
    """
    try:
        import cv2

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        variance = laplacian.var()

        return float(variance)

    except ImportError:
        # Fallback: use scipy if opencv not available
        logger.warning("OpenCV not available, using scipy for blur detection")
        from scipy import ndimage

        # Simple Laplacian kernel
        laplacian = ndimage.laplace(gray_array)
        variance = laplacian.var()

        return float(variance)


# ============================================================================
# Frame Extraction with Quality Filtering
# ============================================================================

def extract_frames_from_scenes(
    video_path: Path,
    scenes: List[Dict[str, Any]],
    output_dir: Path,
    frames_per_scene: int = 5,
    quality_threshold: float = 100.0
) -> List[Dict[str, Any]]:
    """
    Extract representative frames from each scene with quality filtering.

    Args:
        video_path: Path to video file
        scenes: List of detected scenes
        output_dir: Directory to save extracted frames
        frames_per_scene: Number of frames to extract per scene
        quality_threshold: Minimum sharpness threshold

    Returns:
        List of extracted frames with metadata
    """
    logger.info(f"Extracting {frames_per_scene} frames per scene to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames = []

    try:
        import cv2

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        for scene in scenes:
            scene_id = scene['scene_id']
            start_frame = scene['start_frame']
            end_frame = scene['end_frame']
            duration = end_frame - start_frame

            # Calculate frame indices to extract (evenly distributed)
            frame_indices = [
                start_frame + int(duration * (i / (frames_per_scene - 1)))
                for i in range(frames_per_scene)
            ]

            for frame_idx in frame_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}")
                    continue

                # Save frame
                frame_filename = f"scene{scene_id:04d}_frame{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                # Assess quality
                quality = assess_frame_quality(frame_path)

                # Check quality threshold
                if quality.get('sharpness', 0) < quality_threshold:
                    logger.debug(f"Frame {frame_idx} below quality threshold (sharpness={quality.get('sharpness', 0):.1f})")
                    # Still keep frame but mark as low quality
                    quality['low_quality'] = True

                # Add to list
                extracted_frames.append({
                    'scene_id': scene_id,
                    'frame_index': frame_idx,
                    'frame_path': str(frame_path),
                    'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                    'quality': quality,
                })

        cap.release()
        logger.info(f"  Extracted {len(extracted_frames)} frames")

        return extracted_frames

    except ImportError:
        logger.error("opencv-python not installed. Install with: pip install opencv-python")
        raise
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        raise


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_media_asset(
    video_path: Path,
    output_path: Path,
    extract_frames: bool = False,
    frames_dir: Optional[Path] = None,
    scene_threshold: float = 27.0,
    frames_per_scene: int = 5,
) -> Dict[str, Any]:
    """
    Perform comprehensive media asset analysis.

    Args:
        video_path: Path to video file
        output_path: Path to save analysis report (JSON)
        extract_frames: Whether to extract frames
        frames_dir: Directory to save extracted frames
        scene_threshold: Scene detection threshold
        frames_per_scene: Frames to extract per scene

    Returns:
        Analysis report dict
    """
    logger.info("=" * 80)
    logger.info("MEDIA ASSET ANALYZER")
    logger.info("=" * 80)
    logger.info(f"Input: {video_path}")
    logger.info(f"Output: {output_path}")

    # Initialize report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(video_path),
        'analysis_version': '1.0.0',
    }

    try:
        # 1. Extract metadata
        logger.info("\n[1/3] Extracting video metadata...")
        metadata = extract_video_metadata(video_path)
        report['metadata'] = metadata

        # 2. Detect scenes
        logger.info("\n[2/3] Detecting scenes...")
        scenes = detect_scenes(
            video_path,
            threshold=scene_threshold,
            min_scene_length=15
        )
        report['scenes'] = scenes
        report['scene_count'] = len(scenes)

        # 3. Extract frames (optional)
        if extract_frames:
            if frames_dir is None:
                frames_dir = output_path.parent / f"{video_path.stem}_frames"

            logger.info(f"\n[3/3] Extracting frames to: {frames_dir}...")
            extracted_frames = extract_frames_from_scenes(
                video_path,
                scenes,
                frames_dir,
                frames_per_scene=frames_per_scene
            )
            report['extracted_frames'] = extracted_frames
            report['extracted_frame_count'] = len(extracted_frames)

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n✓ Analysis complete: {output_path}")
        logger.info(f"  Scenes: {report['scene_count']}")
        if extract_frames:
            logger.info(f"  Frames extracted: {report['extracted_frame_count']}")

        return report

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze media assets (video/image) for metadata, scenes, and quality'
    )

    # Input/output
    parser.add_argument('--input', type=Path, required=True,
                       help='Input video file path')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output analysis report path (JSON)')

    # Frame extraction
    parser.add_argument('--extract-frames', action='store_true',
                       help='Extract representative frames from scenes')
    parser.add_argument('--frames-dir', type=Path,
                       help='Directory to save extracted frames (default: <output>_frames/)')
    parser.add_argument('--frames-per-scene', type=int, default=5,
                       help='Number of frames to extract per scene (default: 5)')

    # Scene detection
    parser.add_argument('--scene-threshold', type=float, default=27.0,
                       help='Scene detection threshold (default: 27.0)')

    # Safety
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight safety checks (not recommended)')

    args = parser.parse_args()

    # Enforce CPU-only
    enforce_cpu_only()

    # Run preflight checks
    if not args.skip_preflight:
        logger.info("Running preflight checks...")
        try:
            run_preflight(strict=True)
        except Exception as e:
            logger.warning(f"Preflight checks failed: {e}")
            logger.warning("Continuing anyway (use --skip-preflight to suppress this)")

    # Start runtime monitoring
    with RuntimeMonitor(check_interval=30.0) as monitor:
        # Run analysis
        analyze_media_asset(
            video_path=args.input,
            output_path=args.output,
            extract_frames=args.extract_frames,
            frames_dir=args.frames_dir,
            scene_threshold=args.scene_threshold,
            frames_per_scene=args.frames_per_scene,
        )


if __name__ == '__main__':
    main()
