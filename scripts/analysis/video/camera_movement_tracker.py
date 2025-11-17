"""
Camera Movement Tracker for Video Analysis

Tracks camera movements in video using optical flow:
- Pan (horizontal camera movement)
- Tilt (vertical camera movement)
- Zoom (camera moving closer/farther)
- Roll (camera rotation)
- Static/Handheld/Smooth classification
- Movement velocity and acceleration

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


logger = logging.getLogger(__name__)


@dataclass
class CameraMovement:
    """Camera movement for a frame interval"""
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds

    # Movement components
    pan_x: float  # Horizontal pan (pixels, + = right, - = left)
    pan_y: float  # Vertical tilt (pixels, + = down, - = up)
    zoom_factor: float  # Zoom change (1.0 = no change, >1 = zoom in, <1 = zoom out)
    rotation: float  # Rotation in degrees

    # Velocity
    pan_velocity: float  # pixels/second
    tilt_velocity: float  # pixels/second

    # Movement classification
    movement_type: str  # "static", "pan", "tilt", "zoom", "pan_tilt", "complex"
    movement_intensity: float  # 0.0 to 1.0
    is_smooth: bool  # Whether movement is smooth or jerky

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "zoom_factor": self.zoom_factor,
            "rotation": self.rotation,
            "pan_velocity": self.pan_velocity,
            "tilt_velocity": self.tilt_velocity,
            "movement_type": self.movement_type,
            "movement_intensity": self.movement_intensity,
            "is_smooth": self.is_smooth,
            "metadata": self.metadata
        }


@dataclass
class CameraShot:
    """Extended camera shot with movement profile"""
    shot_id: int
    start_frame: int
    end_frame: int
    duration: float  # seconds

    # Aggregate movement statistics
    dominant_movement: str  # Most common movement type
    avg_movement_intensity: float
    is_mostly_static: bool
    is_handheld: bool  # Detected shakiness
    smoothness_score: float  # 0.0 (jerky) to 1.0 (smooth)

    # Individual movements within shot
    movements: List[CameraMovement]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration": self.duration,
            "dominant_movement": self.dominant_movement,
            "avg_movement_intensity": self.avg_movement_intensity,
            "is_mostly_static": self.is_mostly_static,
            "is_handheld": self.is_handheld,
            "smoothness_score": self.smoothness_score,
            "movements": [m.to_dict() for m in self.movements],
            "metadata": self.metadata
        }


@dataclass
class CameraTrackingResult:
    """Complete camera tracking result"""
    video_path: str
    total_frames: int
    fps: float
    duration: float

    # All detected movements
    movements: List[CameraMovement]

    # Shot-level analysis
    shots: List[CameraShot]

    # Global statistics
    total_static_duration: float  # seconds
    total_moving_duration: float  # seconds
    avg_movement_intensity: float
    camera_style: str  # "static", "smooth", "dynamic", "handheld"

    analysis_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "movements": [m.to_dict() for m in self.movements],
            "shots": [s.to_dict() for s in self.shots],
            "total_static_duration": self.total_static_duration,
            "total_moving_duration": self.total_moving_duration,
            "avg_movement_intensity": self.avg_movement_intensity,
            "camera_style": self.camera_style,
            "analysis_time": self.analysis_time,
            "metadata": self.metadata
        }

    def save_json(self, output_path: str):
        """Save result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved camera tracking result to: {output_path}")


class CameraMovementTracker:
    """
    Camera Movement Tracker using Optical Flow

    Uses Lucas-Kanade optical flow to track feature points
    between frames and estimate camera movement.

    Method:
    1. Detect good features to track (Shi-Tomasi corners)
    2. Track features using Lucas-Kanade optical flow
    3. Estimate camera transformation (affine/perspective)
    4. Classify movement type and intensity

    Usage:
        tracker = CameraMovementTracker()
        result = tracker.track_video(video_path)
        result.save_json("camera_tracking.json")
    """

    def __init__(
        self,
        feature_params: Optional[Dict[str, Any]] = None,
        lk_params: Optional[Dict[str, Any]] = None,
        movement_threshold: float = 2.0,
        static_threshold: float = 0.5
    ):
        """
        Initialize camera movement tracker

        Args:
            feature_params: Parameters for Shi-Tomasi corner detection
            lk_params: Parameters for Lucas-Kanade optical flow
            movement_threshold: Minimum movement to detect (pixels)
            static_threshold: Maximum movement to consider static (pixels)
        """
        # Default feature detection parameters
        self.feature_params = feature_params or {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }

        # Default Lucas-Kanade parameters
        self.lk_params = lk_params or {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        self.movement_threshold = movement_threshold
        self.static_threshold = static_threshold

        logger.info(f"CameraMovementTracker initialized")

    def track_video(
        self,
        video_path: str,
        sample_interval: int = 1
    ) -> CameraTrackingResult:
        """
        Track camera movements in video

        Args:
            video_path: Path to video file
            sample_interval: Analyze every Nth frame

        Returns:
            CameraTrackingResult
        """
        start_time = time.time()

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Tracking camera movements: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Detect initial features
        prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)

        movements = []
        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate optical flow
                if prev_points is not None and len(prev_points) > 0:
                    curr_points, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_points, None, **self.lk_params
                    )

                    # Select good points
                    if curr_points is not None:
                        good_prev = prev_points[status == 1]
                        good_curr = curr_points[status == 1]

                        if len(good_prev) >= 4:  # Need at least 4 points for affine
                            # Estimate camera movement
                            movement = self._estimate_camera_movement(
                                good_prev, good_curr,
                                frame_idx - sample_interval, frame_idx,
                                fps
                            )

                            movements.append(movement)

                            # Update points for next iteration
                            prev_points = good_curr.reshape(-1, 1, 2)
                        else:
                            # Too few points, re-detect
                            prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                    else:
                        # Optical flow failed, re-detect features
                        prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                else:
                    # No points, detect new features
                    prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

                prev_gray = gray

            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()

        logger.info(f"Detected {len(movements)} movements")

        # Analyze shots
        shots = self._analyze_shots(movements, fps)

        # Calculate global statistics
        static_duration = sum(m.duration for m in movements if m.movement_type == "static")
        moving_duration = duration - static_duration
        avg_intensity = np.mean([m.movement_intensity for m in movements]) if movements else 0.0

        # Determine camera style
        camera_style = self._determine_camera_style(movements, shots)

        analysis_time = time.time() - start_time

        result = CameraTrackingResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            movements=movements,
            shots=shots,
            total_static_duration=static_duration,
            total_moving_duration=moving_duration,
            avg_movement_intensity=avg_intensity,
            camera_style=camera_style,
            analysis_time=analysis_time,
            metadata={
                "sample_interval": sample_interval,
                "movement_threshold": self.movement_threshold,
                "static_threshold": self.static_threshold
            }
        )

        logger.info(f"Camera tracking completed in {analysis_time:.2f}s")
        logger.info(f"Camera style: {camera_style}")
        logger.info(f"Static: {static_duration:.1f}s, Moving: {moving_duration:.1f}s")

        return result

    def _estimate_camera_movement(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        start_frame: int,
        end_frame: int,
        fps: float
    ) -> CameraMovement:
        """
        Estimate camera movement from point correspondences

        Args:
            prev_points: Points in previous frame (Nx2)
            curr_points: Points in current frame (Nx2)
            start_frame: Start frame index
            end_frame: End frame index
            fps: Frames per second

        Returns:
            CameraMovement
        """
        # Calculate movement vectors
        displacement = curr_points - prev_points

        # Median displacement (robust to outliers)
        median_dx = np.median(displacement[:, 0])
        median_dy = np.median(displacement[:, 1])

        # Calculate zoom using median distance change
        prev_center = np.mean(prev_points, axis=0)
        curr_center = np.mean(curr_points, axis=0)

        prev_distances = np.linalg.norm(prev_points - prev_center, axis=1)
        curr_distances = np.linalg.norm(curr_points - curr_center, axis=1)

        median_prev_dist = np.median(prev_distances)
        median_curr_dist = np.median(curr_distances)

        zoom_factor = median_curr_dist / median_prev_dist if median_prev_dist > 0 else 1.0

        # Estimate rotation (simplified - use affine transform)
        try:
            transform = cv2.estimateAffinePartial2D(prev_points, curr_points)[0]
            if transform is not None:
                # Extract rotation from affine matrix
                rotation = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
            else:
                rotation = 0.0
        except:
            rotation = 0.0

        # Calculate time information
        duration = (end_frame - start_frame) / fps
        start_time = start_frame / fps
        end_time = end_frame / fps

        # Calculate velocities
        pan_velocity = abs(median_dx) / duration if duration > 0 else 0.0
        tilt_velocity = abs(median_dy) / duration if duration > 0 else 0.0

        # Classify movement type
        movement_type = self._classify_movement(
            median_dx, median_dy, zoom_factor, rotation
        )

        # Calculate movement intensity
        movement_intensity = self._calculate_movement_intensity(
            median_dx, median_dy, zoom_factor, rotation
        )

        # Determine if movement is smooth
        # Use standard deviation of displacement as smoothness metric
        displacement_std = np.std(displacement, axis=0)
        is_smooth = np.mean(displacement_std) < 2.0

        return CameraMovement(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            pan_x=float(median_dx),
            pan_y=float(median_dy),
            zoom_factor=float(zoom_factor),
            rotation=float(rotation),
            pan_velocity=float(pan_velocity),
            tilt_velocity=float(tilt_velocity),
            movement_type=movement_type,
            movement_intensity=float(movement_intensity),
            is_smooth=is_smooth
        )

    def _classify_movement(
        self,
        dx: float,
        dy: float,
        zoom: float,
        rotation: float
    ) -> str:
        """Classify movement type"""
        # Calculate movement magnitudes
        pan_mag = abs(dx)
        tilt_mag = abs(dy)
        zoom_mag = abs(zoom - 1.0)
        rot_mag = abs(rotation)

        # Check if static
        if pan_mag < self.static_threshold and tilt_mag < self.static_threshold and zoom_mag < 0.01:
            return "static"

        # Check for zoom
        if zoom_mag > 0.05:
            if pan_mag > self.movement_threshold or tilt_mag > self.movement_threshold:
                return "complex"
            else:
                return "zoom"

        # Check for pan and tilt
        if pan_mag > self.movement_threshold and tilt_mag > self.movement_threshold:
            return "pan_tilt"
        elif pan_mag > self.movement_threshold:
            return "pan"
        elif tilt_mag > self.movement_threshold:
            return "tilt"

        # Small movements
        return "static"

    def _calculate_movement_intensity(
        self,
        dx: float,
        dy: float,
        zoom: float,
        rotation: float
    ) -> float:
        """Calculate movement intensity (0.0 to 1.0)"""
        # Normalize each component
        pan_intensity = min(1.0, abs(dx) / 50.0)  # 50 pixels = max
        tilt_intensity = min(1.0, abs(dy) / 50.0)
        zoom_intensity = min(1.0, abs(zoom - 1.0) * 5.0)  # 0.2 zoom change = max
        rot_intensity = min(1.0, abs(rotation) / 10.0)  # 10 degrees = max

        # Combined intensity
        intensity = max(pan_intensity, tilt_intensity, zoom_intensity, rot_intensity)

        return intensity

    def _analyze_shots(
        self,
        movements: List[CameraMovement],
        fps: float
    ) -> List[CameraShot]:
        """Group movements into shots and analyze"""
        if not movements:
            return []

        shots = []
        shot_id = 1

        # Simple shot detection: consecutive movements with similar characteristics
        current_shot_movements = [movements[0]]
        shot_start_frame = movements[0].start_frame

        for i in range(1, len(movements)):
            prev_mov = movements[i - 1]
            curr_mov = movements[i]

            # Check if part of same shot (consecutive frames, similar movement)
            frame_gap = curr_mov.start_frame - prev_mov.end_frame
            movement_similar = (
                prev_mov.movement_type == curr_mov.movement_type or
                (prev_mov.movement_type == "static" and curr_mov.movement_intensity < 0.3) or
                (curr_mov.movement_type == "static" and prev_mov.movement_intensity < 0.3)
            )

            if frame_gap <= 5 and movement_similar:
                # Same shot
                current_shot_movements.append(curr_mov)
            else:
                # New shot, finalize previous
                shot = self._create_camera_shot(
                    shot_id, current_shot_movements, shot_start_frame, fps
                )
                shots.append(shot)

                shot_id += 1
                current_shot_movements = [curr_mov]
                shot_start_frame = curr_mov.start_frame

        # Finalize last shot
        if current_shot_movements:
            shot = self._create_camera_shot(
                shot_id, current_shot_movements, shot_start_frame, fps
            )
            shots.append(shot)

        logger.info(f"Detected {len(shots)} camera shots")

        return shots

    def _create_camera_shot(
        self,
        shot_id: int,
        movements: List[CameraMovement],
        start_frame: int,
        fps: float
    ) -> CameraShot:
        """Create CameraShot from movements"""
        end_frame = movements[-1].end_frame
        duration = (end_frame - start_frame) / fps

        # Determine dominant movement
        movement_types = [m.movement_type for m in movements]
        dominant_movement = max(set(movement_types), key=movement_types.count)

        # Calculate average intensity
        avg_intensity = np.mean([m.movement_intensity for m in movements])

        # Check if mostly static
        static_count = sum(1 for m in movements if m.movement_type == "static")
        is_mostly_static = static_count / len(movements) > 0.7

        # Detect handheld (lots of small jerky movements)
        jerky_count = sum(1 for m in movements if not m.is_smooth and m.movement_intensity < 0.3)
        is_handheld = jerky_count / len(movements) > 0.5

        # Calculate smoothness score
        smooth_count = sum(1 for m in movements if m.is_smooth)
        smoothness_score = smooth_count / len(movements)

        return CameraShot(
            shot_id=shot_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration=duration,
            dominant_movement=dominant_movement,
            avg_movement_intensity=float(avg_intensity),
            is_mostly_static=is_mostly_static,
            is_handheld=is_handheld,
            smoothness_score=float(smoothness_score),
            movements=movements
        )

    def _determine_camera_style(
        self,
        movements: List[CameraMovement],
        shots: List[CameraShot]
    ) -> str:
        """Determine overall camera style"""
        if not movements:
            return "unknown"

        # Calculate statistics
        static_ratio = sum(1 for m in movements if m.movement_type == "static") / len(movements)
        avg_intensity = np.mean([m.movement_intensity for m in movements])
        avg_smoothness = np.mean([s.smoothness_score for s in shots]) if shots else 0.0

        # Classify style
        if static_ratio > 0.8:
            return "static"
        elif avg_intensity > 0.6:
            return "dynamic"
        elif avg_smoothness > 0.7:
            return "smooth"
        else:
            return "handheld"


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

    # Create tracker
    tracker = CameraMovementTracker()

    # Track camera movements
    result = tracker.track_video(
        video_path=video_path,
        sample_interval=1  # Analyze every frame
    )

    # Print results
    print("\n" + "=" * 60)
    print("CAMERA MOVEMENT TRACKING RESULTS")
    print("=" * 60)
    print(f"Video: {result.video_path}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"FPS: {result.fps:.2f}")
    print(f"Camera Style: {result.camera_style}")
    print(f"Static Duration: {result.total_static_duration:.2f}s")
    print(f"Moving Duration: {result.total_moving_duration:.2f}s")
    print(f"Avg Movement Intensity: {result.avg_movement_intensity:.3f}")
    print(f"Total Movements Detected: {len(result.movements)}")
    print(f"Total Shots Detected: {len(result.shots)}")

    # Show sample shots
    print("\nSample Shots:")
    for shot in result.shots[:5]:  # Show first 5
        print(f"  Shot {shot.shot_id}: {shot.start_frame}-{shot.end_frame} "
              f"({shot.duration:.2f}s)")
        print(f"    Movement: {shot.dominant_movement}, "
              f"Intensity: {shot.avg_movement_intensity:.3f}")
        print(f"    Static: {shot.is_mostly_static}, "
              f"Handheld: {shot.is_handheld}, "
              f"Smoothness: {shot.smoothness_score:.3f}")

    if len(result.shots) > 5:
        print(f"  ... and {len(result.shots) - 5} more shots")

    # Save to JSON
    output_json = "outputs/analysis/camera_tracking.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    result.save_json(output_json)

    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
