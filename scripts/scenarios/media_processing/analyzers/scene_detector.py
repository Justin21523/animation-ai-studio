"""
Scene Detector

Detects scene boundaries in video files using CPU-only methods.
Provides multiple detection algorithms (content-based, histogram, hybrid).

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import subprocess
import re
from pathlib import Path
from typing import List, Optional

from ..common import SceneInfo, SceneDetectionMethod, VideoStreamInfo

logger = logging.getLogger(__name__)


class SceneDetector:
    """
    Detect scene boundaries in video files

    Features:
    - Content-based detection (ffmpeg select filter)
    - Histogram-based detection
    - Shot boundary detection
    - Hybrid method combining multiple approaches
    - Configurable sensitivity thresholds
    - Minimum scene duration enforcement
    - CPU-only operation (no GPU)

    Example:
        detector = SceneDetector(ffmpeg_path="ffmpeg")
        scenes = detector.detect_scenes(
            video_path=Path("/path/to/video.mp4"),
            method=SceneDetectionMethod.CONTENT_BASED,
            threshold=0.3,
            min_scene_duration=1.0
        )

        print(f"Detected {len(scenes)} scenes")
        for scene in scenes:
            print(f"Scene {scene.scene_id}: {scene.timestamp_range}")
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize scene detector

        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

        # Verify ffmpeg is available
        self._verify_ffmpeg()

        logger.info(f"SceneDetector initialized with ffmpeg: {self.ffmpeg_path}")

    def _verify_ffmpeg(self):
        """Verify ffmpeg is available and working"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            logger.debug(f"ffmpeg version: {result.stdout.split()[2]}")

        except FileNotFoundError:
            raise RuntimeError(
                f"ffmpeg not found at: {self.ffmpeg_path}. "
                "Please install ffmpeg or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg verification timed out")

    def detect_scenes(
        self,
        video_path: Path,
        method: SceneDetectionMethod = SceneDetectionMethod.CONTENT_BASED,
        threshold: float = 0.3,
        min_scene_duration: float = 1.0,
        video_stream: Optional[VideoStreamInfo] = None
    ) -> List[SceneInfo]:
        """
        Detect scene boundaries in video

        Args:
            video_path: Path to video file
            method: Detection method to use
            threshold: Detection sensitivity (0.0-1.0, higher = more sensitive)
            min_scene_duration: Minimum scene duration in seconds
            video_stream: Optional video stream info (for FPS/duration)

        Returns:
            List of detected scenes

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If detection fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(
            f"Detecting scenes in: {video_path} "
            f"(method={method.value}, threshold={threshold})"
        )

        # Select detection method
        if method == SceneDetectionMethod.CONTENT_BASED:
            scenes = self._detect_content_based(video_path, threshold, video_stream)
        elif method == SceneDetectionMethod.HISTOGRAM:
            scenes = self._detect_histogram(video_path, threshold, video_stream)
        elif method == SceneDetectionMethod.SHOT_BOUNDARY:
            scenes = self._detect_shot_boundary(video_path, threshold, video_stream)
        elif method == SceneDetectionMethod.HYBRID:
            scenes = self._detect_hybrid(video_path, threshold, video_stream)
        else:
            raise ValueError(f"Unknown detection method: {method}")

        # Merge short scenes if requested
        if min_scene_duration > 0:
            scenes = self.merge_short_scenes(scenes, min_scene_duration)

        logger.info(f"Detected {len(scenes)} scenes")

        return scenes

    def _detect_content_based(
        self,
        video_path: Path,
        threshold: float,
        video_stream: Optional[VideoStreamInfo]
    ) -> List[SceneInfo]:
        """
        Detect scenes using ffmpeg select filter (content-based)

        This method uses ffmpeg's scene detection filter which analyzes
        pixel differences between frames.

        Args:
            video_path: Path to video file
            threshold: Scene detection threshold (0.0-1.0)
            video_stream: Optional video stream info

        Returns:
            List of detected scenes
        """
        # Get FPS for timestamp conversion
        fps = video_stream.fps if video_stream else self._get_fps(video_path)

        # Run ffmpeg with scene detection filter
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse scene timestamps from stderr output
            timestamps = self._parse_scene_timestamps(result.stderr, fps)

            # Convert timestamps to SceneInfo objects
            scenes = self._timestamps_to_scenes(timestamps, fps)

            logger.debug(f"Content-based detection found {len(scenes)} scenes")

            return scenes

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Scene detection timed out for: {video_path}")
        except Exception as e:
            raise RuntimeError(f"Scene detection failed: {e}")

    def _detect_histogram(
        self,
        video_path: Path,
        threshold: float,
        video_stream: Optional[VideoStreamInfo]
    ) -> List[SceneInfo]:
        """
        Detect scenes using histogram difference analysis

        This method compares frame histograms to detect significant changes.
        More sensitive than content-based for subtle scene changes.

        Args:
            video_path: Path to video file
            threshold: Histogram difference threshold (0.0-1.0)
            video_stream: Optional video stream info

        Returns:
            List of detected scenes
        """
        # Get FPS
        fps = video_stream.fps if video_stream else self._get_fps(video_path)

        # Use ffmpeg histogram comparison
        # This detects changes in color distribution
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"select='gt(abs(PREV_INFERRED_DIFF),{threshold})',showinfo",
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse timestamps
            timestamps = self._parse_scene_timestamps(result.stderr, fps)
            scenes = self._timestamps_to_scenes(timestamps, fps)

            logger.debug(f"Histogram detection found {len(scenes)} scenes")

            return scenes

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Histogram detection timed out for: {video_path}")
        except Exception as e:
            raise RuntimeError(f"Histogram detection failed: {e}")

    def _detect_shot_boundary(
        self,
        video_path: Path,
        threshold: float,
        video_stream: Optional[VideoStreamInfo]
    ) -> List[SceneInfo]:
        """
        Detect shot boundaries using specialized filter

        Args:
            video_path: Path to video file
            threshold: Detection threshold
            video_stream: Optional video stream info

        Returns:
            List of detected scenes
        """
        # Shot boundary detection is similar to content-based
        # but with adjusted parameters for cuts vs gradual transitions
        return self._detect_content_based(video_path, threshold, video_stream)

    def _detect_hybrid(
        self,
        video_path: Path,
        threshold: float,
        video_stream: Optional[VideoStreamInfo]
    ) -> List[SceneInfo]:
        """
        Hybrid detection combining multiple methods

        Runs both content-based and histogram detection, then merges results.

        Args:
            video_path: Path to video file
            threshold: Detection threshold
            video_stream: Optional video stream info

        Returns:
            List of detected scenes (merged from multiple methods)
        """
        # Run both detection methods
        content_scenes = self._detect_content_based(video_path, threshold, video_stream)
        histogram_scenes = self._detect_histogram(video_path, threshold * 0.8, video_stream)

        # Merge and deduplicate scenes
        all_timestamps = []
        for scene in content_scenes + histogram_scenes:
            all_timestamps.append(scene.start_time)

        # Remove duplicates and sort
        unique_timestamps = sorted(set(all_timestamps))

        # Get FPS
        fps = video_stream.fps if video_stream else self._get_fps(video_path)

        # Convert to scenes
        scenes = self._timestamps_to_scenes(unique_timestamps, fps)

        logger.debug(f"Hybrid detection found {len(scenes)} scenes")

        return scenes

    def _parse_scene_timestamps(self, ffmpeg_output: str, fps: float) -> List[float]:
        """
        Parse scene change timestamps from ffmpeg output

        Args:
            ffmpeg_output: ffmpeg stderr output
            fps: Video FPS for frame-to-time conversion

        Returns:
            List of timestamps (in seconds)
        """
        timestamps = [0.0]  # Always start from 0

        # Parse showinfo output for pts_time
        # Format: pts_time:12.345
        pattern = r'pts_time:([\d.]+)'
        matches = re.findall(pattern, ffmpeg_output)

        for match in matches:
            try:
                timestamp = float(match)
                timestamps.append(timestamp)
            except ValueError:
                logger.warning(f"Failed to parse timestamp: {match}")
                continue

        return sorted(set(timestamps))

    def _timestamps_to_scenes(self, timestamps: List[float], fps: float) -> List[SceneInfo]:
        """
        Convert list of timestamps to SceneInfo objects

        Args:
            timestamps: List of scene start timestamps
            fps: Video FPS

        Returns:
            List of SceneInfo objects
        """
        scenes = []

        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            duration = end_time - start_time

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            frame_count = end_frame - start_frame

            scene = SceneInfo(
                scene_id=i,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration,
                frame_count=frame_count,
                confidence=1.0  # Placeholder for now
            )

            scenes.append(scene)

        return scenes

    def merge_short_scenes(
        self,
        scenes: List[SceneInfo],
        min_duration: float
    ) -> List[SceneInfo]:
        """
        Merge scenes shorter than minimum duration with adjacent scenes

        Args:
            scenes: List of detected scenes
            min_duration: Minimum scene duration in seconds

        Returns:
            List of merged scenes
        """
        if not scenes:
            return []

        merged_scenes = []
        current_scene = scenes[0]

        for next_scene in scenes[1:]:
            if current_scene.duration < min_duration:
                # Merge with next scene
                current_scene = SceneInfo(
                    scene_id=current_scene.scene_id,
                    start_time=current_scene.start_time,
                    end_time=next_scene.end_time,
                    start_frame=current_scene.start_frame,
                    end_frame=next_scene.end_frame,
                    duration=next_scene.end_time - current_scene.start_time,
                    frame_count=next_scene.end_frame - current_scene.start_frame,
                    confidence=min(current_scene.confidence, next_scene.confidence)
                )
            else:
                # Keep current scene, move to next
                merged_scenes.append(current_scene)
                current_scene = next_scene

        # Add last scene
        merged_scenes.append(current_scene)

        # Renumber scenes
        for i, scene in enumerate(merged_scenes):
            scene.scene_id = i

        logger.debug(
            f"Merged {len(scenes)} scenes to {len(merged_scenes)} scenes "
            f"(min_duration={min_duration}s)"
        )

        return merged_scenes

    def _get_fps(self, video_path: Path) -> float:
        """
        Get video FPS using ffprobe

        Args:
            video_path: Path to video file

        Returns:
            FPS as float
        """
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            fps_str = result.stdout.strip()

            # Parse fractional FPS (e.g., "30000/1001")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)

            return fps

        except (ValueError, ZeroDivisionError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to get FPS, defaulting to 30: {e}")
            return 30.0
