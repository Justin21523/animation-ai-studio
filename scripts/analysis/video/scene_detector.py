"""
Scene Detector for Video Analysis

Uses PySceneDetect for content-aware scene detection with adaptive thresholding.

Features:
- Content-aware scene detection
- Adaptive threshold optimization
- Keyframe extraction (representative frame per scene)
- JSON output for Agent Framework integration

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# PySceneDetect imports
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.video_splitter import split_video_ffmpeg
except ImportError:
    raise ImportError(
        "PySceneDetect not installed. Install with: pip install scenedetect[opencv]"
    )


logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Single scene representation"""
    scene_id: int
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds
    keyframe_index: int  # Representative frame index
    keyframe_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scene_id": self.scene_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "keyframe_index": self.keyframe_index,
            "keyframe_path": self.keyframe_path,
            "metadata": self.metadata
        }


@dataclass
class SceneDetectionResult:
    """Complete scene detection result"""
    video_path: str
    total_scenes: int
    scenes: List[Scene]
    video_duration: float
    fps: float
    resolution: Tuple[int, int]
    avg_scene_duration: float
    detection_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_path": self.video_path,
            "total_scenes": self.total_scenes,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "video_duration": self.video_duration,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "avg_scene_duration": self.avg_scene_duration,
            "detection_time": self.detection_time,
            "metadata": self.metadata
        }

    def save_json(self, output_path: str):
        """Save result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scene detection result to: {output_path}")


class SceneDetector:
    """
    Scene Detector using PySceneDetect

    Detects scene changes in video using content-aware analysis.
    Extracts keyframes for each detected scene.

    Usage:
        detector = SceneDetector(threshold=27.0)
        result = detector.detect(video_path, extract_keyframes=True)
        result.save_json("scenes.json")
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_length: int = 15,
        adaptive_threshold: bool = True
    ):
        """
        Initialize scene detector

        Args:
            threshold: Scene detection threshold (default: 27.0)
                      Lower = more sensitive, higher = less sensitive
            min_scene_length: Minimum scene length in frames
            adaptive_threshold: Auto-adjust threshold based on video content
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.adaptive_threshold = adaptive_threshold

        logger.info(f"SceneDetector initialized (threshold={threshold}, "
                   f"min_length={min_scene_length}, adaptive={adaptive_threshold})")

    def detect(
        self,
        video_path: str,
        extract_keyframes: bool = True,
        keyframe_output_dir: Optional[str] = None
    ) -> SceneDetectionResult:
        """
        Detect scenes in video

        Args:
            video_path: Path to video file
            extract_keyframes: Whether to extract keyframes
            keyframe_output_dir: Directory to save keyframes (auto-generated if None)

        Returns:
            SceneDetectionResult
        """
        start_time = time.time()

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Detecting scenes in: {video_path}")

        # Create video manager
        video_manager = VideoManager([video_path])

        # Create scene manager with ContentDetector
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_length
            )
        )

        # Start video manager
        video_manager.start()

        # Get video properties
        fps = video_manager.get_framerate()
        resolution = video_manager.get_framesize()
        total_frames = video_manager.get_duration().get_frames()
        duration_seconds = video_manager.get_duration().get_seconds()

        logger.info(f"Video: {fps:.2f} FPS, {resolution[0]}x{resolution[1]}, "
                   f"{duration_seconds:.2f}s ({total_frames} frames)")

        # Detect scenes
        logger.info("Running scene detection...")
        video_manager.set_downscale_factor()  # Default downscale for performance
        scene_manager.detect_scenes(frame_source=video_manager)

        # Get scene list
        scene_list = scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode())

        video_manager.release()

        logger.info(f"Detected {len(scene_list)} scenes")

        # Convert to Scene objects
        scenes = []
        for i, (start_time_obj, end_time_obj) in enumerate(scene_list, 1):
            start_frame = start_time_obj.get_frames()
            end_frame = end_time_obj.get_frames()
            start_sec = start_time_obj.get_seconds()
            end_sec = end_time_obj.get_seconds()
            duration = end_sec - start_sec

            # Calculate keyframe index (middle of scene)
            keyframe_idx = start_frame + (end_frame - start_frame) // 2

            scene = Scene(
                scene_id=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_sec,
                end_time=end_sec,
                duration=duration,
                keyframe_index=keyframe_idx
            )

            scenes.append(scene)

        # Extract keyframes if requested
        if extract_keyframes and scenes:
            if keyframe_output_dir is None:
                # Auto-generate output directory
                video_name = Path(video_path).stem
                keyframe_output_dir = f"outputs/analysis/{video_name}/keyframes"

            Path(keyframe_output_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting {len(scenes)} keyframes to: {keyframe_output_dir}")
            self._extract_keyframes(video_path, scenes, keyframe_output_dir)

        # Calculate statistics
        avg_scene_duration = np.mean([scene.duration for scene in scenes]) if scenes else 0.0
        detection_time = time.time() - start_time

        result = SceneDetectionResult(
            video_path=video_path,
            total_scenes=len(scenes),
            scenes=scenes,
            video_duration=duration_seconds,
            fps=fps,
            resolution=resolution,
            avg_scene_duration=avg_scene_duration,
            detection_time=detection_time,
            metadata={
                "threshold": self.threshold,
                "min_scene_length": self.min_scene_length,
                "adaptive_threshold": self.adaptive_threshold
            }
        )

        logger.info(f"Scene detection completed in {detection_time:.2f}s")
        logger.info(f"Average scene duration: {avg_scene_duration:.2f}s")

        return result

    def _extract_keyframes(
        self,
        video_path: str,
        scenes: List[Scene],
        output_dir: str
    ):
        """Extract keyframes for each scene"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        video_name = Path(video_path).stem

        for scene in scenes:
            # Seek to keyframe
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene.keyframe_index)
            ret, frame = cap.read()

            if ret:
                # Save keyframe
                keyframe_filename = f"{video_name}_scene_{scene.scene_id:03d}_frame_{scene.keyframe_index:06d}.jpg"
                keyframe_path = Path(output_dir) / keyframe_filename

                cv2.imwrite(str(keyframe_path), frame)
                scene.keyframe_path = str(keyframe_path)

                logger.debug(f"Extracted keyframe for scene {scene.scene_id}: {keyframe_path}")
            else:
                logger.warning(f"Failed to extract keyframe for scene {scene.scene_id}")

        cap.release()
        logger.info(f"Keyframe extraction completed")

    def optimize_threshold(
        self,
        video_path: str,
        target_scene_count: Optional[int] = None,
        target_avg_duration: Optional[float] = None
    ) -> float:
        """
        Optimize detection threshold for desired results

        Args:
            video_path: Path to video file
            target_scene_count: Desired number of scenes (optional)
            target_avg_duration: Desired average scene duration in seconds (optional)

        Returns:
            Optimized threshold value
        """
        logger.info("Optimizing scene detection threshold...")

        # Try different thresholds
        thresholds = [15.0, 20.0, 25.0, 27.0, 30.0, 35.0, 40.0]
        results = []

        for threshold in thresholds:
            temp_detector = SceneDetector(
                threshold=threshold,
                min_scene_length=self.min_scene_length,
                adaptive_threshold=False
            )

            result = temp_detector.detect(video_path, extract_keyframes=False)
            results.append((threshold, result))

            logger.info(f"Threshold {threshold}: {result.total_scenes} scenes, "
                       f"avg {result.avg_scene_duration:.2f}s")

        # Find best threshold
        if target_scene_count:
            # Minimize difference from target scene count
            best_threshold = min(results, key=lambda x: abs(x[1].total_scenes - target_scene_count))[0]
        elif target_avg_duration:
            # Minimize difference from target average duration
            best_threshold = min(results, key=lambda x: abs(x[1].avg_scene_duration - target_avg_duration))[0]
        else:
            # Use default (27.0 is generally good)
            best_threshold = 27.0

        logger.info(f"Optimized threshold: {best_threshold}")
        self.threshold = best_threshold

        return best_threshold


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example video path (replace with actual video)
    video_path = "path/to/your/video.mp4"

    if not Path(video_path).exists():
        logger.warning(f"Example video not found: {video_path}")
        logger.info("Please provide a valid video path")
        return

    # Create detector
    detector = SceneDetector(
        threshold=27.0,
        min_scene_length=15,
        adaptive_threshold=True
    )

    # Detect scenes
    result = detector.detect(
        video_path=video_path,
        extract_keyframes=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("SCENE DETECTION RESULTS")
    print("=" * 60)
    print(f"Video: {result.video_path}")
    print(f"Duration: {result.video_duration:.2f}s")
    print(f"FPS: {result.fps:.2f}")
    print(f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
    print(f"Total Scenes: {result.total_scenes}")
    print(f"Average Scene Duration: {result.avg_scene_duration:.2f}s")
    print(f"Detection Time: {result.detection_time:.2f}s")

    print("\nScenes:")
    for scene in result.scenes[:10]:  # Show first 10
        print(f"  Scene {scene.scene_id}: "
              f"{scene.start_time:.2f}s - {scene.end_time:.2f}s "
              f"({scene.duration:.2f}s)")

    if result.total_scenes > 10:
        print(f"  ... and {result.total_scenes - 10} more scenes")

    # Save to JSON
    output_json = "outputs/analysis/scenes.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    result.save_json(output_json)

    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
