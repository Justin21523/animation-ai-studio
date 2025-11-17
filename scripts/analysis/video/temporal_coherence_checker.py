"""
Temporal Coherence Checker for Video Analysis

Analyzes temporal consistency and quality of video:
- Color stability across frames
- Motion smoothness and continuity
- Flicker detection
- Abrupt transition detection
- Frame-to-frame similarity
- Temporal artifacts (ghosting, judder)

Critical for AI-generated videos where temporal coherence is often poor.

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
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


@dataclass
class FrameCoherence:
    """Temporal coherence metrics for a frame pair"""
    frame_a: int
    frame_b: int
    timestamp: float  # seconds

    # Color stability
    color_difference: float  # Mean color change (0-255)
    color_stability_score: float  # 0.0 (unstable) to 1.0 (stable)

    # Motion metrics
    motion_magnitude: float  # Average optical flow magnitude
    motion_smoothness: float  # 0.0 (jerky) to 1.0 (smooth)

    # Structural similarity
    ssim_score: float  # 0.0 to 1.0

    # Flicker detection
    brightness_change: float  # Change in mean brightness
    has_flicker: bool

    # Transition detection
    is_abrupt_transition: bool

    # Overall coherence
    coherence_score: float  # 0.0 (poor) to 1.0 (excellent)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "frame_a": self.frame_a,
            "frame_b": self.frame_b,
            "timestamp": self.timestamp,
            "color_difference": self.color_difference,
            "color_stability_score": self.color_stability_score,
            "motion_magnitude": self.motion_magnitude,
            "motion_smoothness": self.motion_smoothness,
            "ssim_score": self.ssim_score,
            "brightness_change": self.brightness_change,
            "has_flicker": self.has_flicker,
            "is_abrupt_transition": self.is_abrupt_transition,
            "coherence_score": self.coherence_score,
            "metadata": self.metadata
        }


@dataclass
class TemporalSegment:
    """Segment of video with consistent temporal properties"""
    segment_id: int
    start_frame: int
    end_frame: int
    duration: float  # seconds

    # Aggregate metrics
    avg_coherence_score: float
    min_coherence_score: float
    has_issues: bool  # Whether segment has temporal artifacts

    # Issue counts
    flicker_count: int
    abrupt_transition_count: int
    low_coherence_count: int

    # Segment quality
    quality_rating: str  # "excellent", "good", "fair", "poor"

    frame_coherences: List[FrameCoherence]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "segment_id": self.segment_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration": self.duration,
            "avg_coherence_score": self.avg_coherence_score,
            "min_coherence_score": self.min_coherence_score,
            "has_issues": self.has_issues,
            "flicker_count": self.flicker_count,
            "abrupt_transition_count": self.abrupt_transition_count,
            "low_coherence_count": self.low_coherence_count,
            "quality_rating": self.quality_rating,
            "frame_coherences": [fc.to_dict() for fc in self.frame_coherences],
            "metadata": self.metadata
        }


@dataclass
class TemporalCoherenceResult:
    """Complete temporal coherence analysis result"""
    video_path: str
    total_frames: int
    fps: float
    duration: float

    # All frame-to-frame coherence metrics
    frame_coherences: List[FrameCoherence]

    # Segmented analysis
    segments: List[TemporalSegment]

    # Global statistics
    avg_coherence_score: float
    min_coherence_score: float
    total_flicker_frames: int
    total_abrupt_transitions: int
    temporal_stability_rating: str  # "excellent", "good", "fair", "poor"

    # Problem areas (low coherence segments)
    problem_segments: List[int]  # Segment IDs

    analysis_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "frame_coherences": [fc.to_dict() for fc in self.frame_coherences],
            "segments": [s.to_dict() for s in self.segments],
            "avg_coherence_score": self.avg_coherence_score,
            "min_coherence_score": self.min_coherence_score,
            "total_flicker_frames": self.total_flicker_frames,
            "total_abrupt_transitions": self.total_abrupt_transitions,
            "temporal_stability_rating": self.temporal_stability_rating,
            "problem_segments": self.problem_segments,
            "analysis_time": self.analysis_time,
            "metadata": self.metadata
        }

    def save_json(self, output_path: str):
        """Save result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved temporal coherence result to: {output_path}")


class TemporalCoherenceChecker:
    """
    Temporal Coherence Checker

    Analyzes temporal consistency of video by examining
    frame-to-frame changes in color, motion, and structure.

    Key Metrics:
    - SSIM (Structural Similarity): Measures perceptual similarity
    - Color Stability: Tracks color consistency
    - Motion Smoothness: Analyzes optical flow continuity
    - Flicker Detection: Identifies brightness oscillations

    Usage:
        checker = TemporalCoherenceChecker()
        result = checker.check_video(video_path)
        result.save_json("temporal_coherence.json")
    """

    def __init__(
        self,
        flicker_threshold: float = 10.0,
        abrupt_transition_threshold: float = 0.3,
        coherence_threshold: float = 0.7,
        segment_length: int = 90  # frames per segment
    ):
        """
        Initialize temporal coherence checker

        Args:
            flicker_threshold: Brightness change threshold for flicker detection
            abrupt_transition_threshold: SSIM threshold for abrupt transitions
            coherence_threshold: Minimum coherence score for quality
            segment_length: Frames per segment for analysis
        """
        self.flicker_threshold = flicker_threshold
        self.abrupt_transition_threshold = abrupt_transition_threshold
        self.coherence_threshold = coherence_threshold
        self.segment_length = segment_length

        logger.info(f"TemporalCoherenceChecker initialized")

    def check_video(
        self,
        video_path: str,
        sample_interval: int = 1
    ) -> TemporalCoherenceResult:
        """
        Check temporal coherence of video

        Args:
            video_path: Path to video file
            sample_interval: Analyze every Nth frame

        Returns:
            TemporalCoherenceResult
        """
        start_time = time.time()

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Checking temporal coherence: {video_path}")

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
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

        frame_coherences = []
        frame_idx = 1

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_idx % sample_interval == 0:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)

                # Analyze frame pair
                coherence = self._analyze_frame_pair(
                    prev_frame, curr_frame,
                    prev_gray, curr_gray,
                    prev_hsv, curr_hsv,
                    frame_idx - sample_interval, frame_idx,
                    fps
                )

                frame_coherences.append(coherence)

                # Update previous frame
                prev_frame = curr_frame
                prev_gray = curr_gray
                prev_hsv = curr_hsv

            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()

        logger.info(f"Analyzed {len(frame_coherences)} frame pairs")

        # Segment analysis
        segments = self._segment_analysis(frame_coherences, fps)

        # Calculate global statistics
        if frame_coherences:
            avg_coherence = np.mean([fc.coherence_score for fc in frame_coherences])
            min_coherence = min(fc.coherence_score for fc in frame_coherences)
            flicker_count = sum(1 for fc in frame_coherences if fc.has_flicker)
            transition_count = sum(1 for fc in frame_coherences if fc.is_abrupt_transition)
        else:
            avg_coherence = 0.0
            min_coherence = 0.0
            flicker_count = 0
            transition_count = 0

        # Determine stability rating
        stability_rating = self._rate_stability(avg_coherence, flicker_count, transition_count, len(frame_coherences))

        # Identify problem segments
        problem_segments = [s.segment_id for s in segments if s.has_issues]

        analysis_time = time.time() - start_time

        result = TemporalCoherenceResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            frame_coherences=frame_coherences,
            segments=segments,
            avg_coherence_score=avg_coherence,
            min_coherence_score=min_coherence,
            total_flicker_frames=flicker_count,
            total_abrupt_transitions=transition_count,
            temporal_stability_rating=stability_rating,
            problem_segments=problem_segments,
            analysis_time=analysis_time,
            metadata={
                "sample_interval": sample_interval,
                "flicker_threshold": self.flicker_threshold,
                "abrupt_transition_threshold": self.abrupt_transition_threshold,
                "coherence_threshold": self.coherence_threshold
            }
        )

        logger.info(f"Temporal coherence check completed in {analysis_time:.2f}s")
        logger.info(f"Stability rating: {stability_rating}")
        logger.info(f"Avg coherence: {avg_coherence:.3f}, Min: {min_coherence:.3f}")
        logger.info(f"Flicker frames: {flicker_count}, Abrupt transitions: {transition_count}")

        return result

    def _analyze_frame_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        gray_a: np.ndarray,
        gray_b: np.ndarray,
        hsv_a: np.ndarray,
        hsv_b: np.ndarray,
        idx_a: int,
        idx_b: int,
        fps: float
    ) -> FrameCoherence:
        """Analyze temporal coherence between two frames"""

        timestamp = idx_b / fps

        # 1. Color Stability
        color_diff, color_stability = self._analyze_color_stability(hsv_a, hsv_b)

        # 2. Motion Analysis
        motion_mag, motion_smooth = self._analyze_motion(gray_a, gray_b)

        # 3. Structural Similarity (SSIM)
        ssim_score = self._calculate_ssim(gray_a, gray_b)

        # 4. Flicker Detection
        brightness_a = np.mean(gray_a)
        brightness_b = np.mean(gray_b)
        brightness_change = abs(brightness_b - brightness_a)
        has_flicker = brightness_change > self.flicker_threshold

        # 5. Abrupt Transition Detection
        is_abrupt = ssim_score < self.abrupt_transition_threshold

        # Calculate overall coherence score
        coherence_score = self._calculate_coherence_score(
            color_stability, motion_smooth, ssim_score, has_flicker, is_abrupt
        )

        return FrameCoherence(
            frame_a=idx_a,
            frame_b=idx_b,
            timestamp=timestamp,
            color_difference=float(color_diff),
            color_stability_score=float(color_stability),
            motion_magnitude=float(motion_mag),
            motion_smoothness=float(motion_smooth),
            ssim_score=float(ssim_score),
            brightness_change=float(brightness_change),
            has_flicker=has_flicker,
            is_abrupt_transition=is_abrupt,
            coherence_score=float(coherence_score)
        )

    def _analyze_color_stability(
        self,
        hsv_a: np.ndarray,
        hsv_b: np.ndarray
    ) -> Tuple[float, float]:
        """Analyze color stability between frames"""
        # Calculate color difference in HSV space
        h_diff = np.mean(np.abs(hsv_a[:, :, 0].astype(float) - hsv_b[:, :, 0].astype(float)))
        s_diff = np.mean(np.abs(hsv_a[:, :, 1].astype(float) - hsv_b[:, :, 1].astype(float)))
        v_diff = np.mean(np.abs(hsv_a[:, :, 2].astype(float) - hsv_b[:, :, 2].astype(float)))

        # Combined color difference
        color_diff = (h_diff * 2 + s_diff + v_diff) / 4  # Weight hue more

        # Stability score (inverse of difference)
        color_stability = 1.0 - min(1.0, color_diff / 50.0)

        return color_diff, color_stability

    def _analyze_motion(
        self,
        gray_a: np.ndarray,
        gray_b: np.ndarray
    ) -> Tuple[float, float]:
        """Analyze motion smoothness using optical flow"""
        try:
            # Calculate dense optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Calculate motion magnitude
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitude = np.mean(mag)

            # Motion smoothness: lower standard deviation = smoother
            motion_std = np.std(mag)
            motion_smoothness = 1.0 - min(1.0, motion_std / 10.0)

        except Exception as e:
            logger.warning(f"Motion analysis failed: {e}")
            motion_magnitude = 0.0
            motion_smoothness = 1.0

        return motion_magnitude, motion_smoothness

    def _calculate_ssim(
        self,
        gray_a: np.ndarray,
        gray_b: np.ndarray
    ) -> float:
        """Calculate Structural Similarity Index"""
        # Simplified SSIM implementation
        # Full implementation would use skimage.metrics.structural_similarity

        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Convert to float
        img1 = gray_a.astype(np.float64)
        img2 = gray_b.astype(np.float64)

        # Means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        ssim_score = np.mean(ssim_map)

        return float(ssim_score)

    def _calculate_coherence_score(
        self,
        color_stability: float,
        motion_smoothness: float,
        ssim_score: float,
        has_flicker: bool,
        is_abrupt: bool
    ) -> float:
        """Calculate overall coherence score"""
        # Weighted combination
        score = (
            color_stability * 0.25 +
            motion_smoothness * 0.25 +
            ssim_score * 0.35 +
            0.15  # Base score
        )

        # Penalties
        if has_flicker:
            score -= 0.2
        if is_abrupt:
            score -= 0.3

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return score

    def _segment_analysis(
        self,
        frame_coherences: List[FrameCoherence],
        fps: float
    ) -> List[TemporalSegment]:
        """Segment video and analyze each segment"""
        if not frame_coherences:
            return []

        segments = []
        segment_id = 1

        # Divide into fixed-length segments
        for i in range(0, len(frame_coherences), self.segment_length):
            segment_coherences = frame_coherences[i:i + self.segment_length]

            if not segment_coherences:
                continue

            start_frame = segment_coherences[0].frame_a
            end_frame = segment_coherences[-1].frame_b
            duration = (end_frame - start_frame) / fps

            # Calculate segment statistics
            coherence_scores = [fc.coherence_score for fc in segment_coherences]
            avg_coherence = np.mean(coherence_scores)
            min_coherence = min(coherence_scores)

            # Count issues
            flicker_count = sum(1 for fc in segment_coherences if fc.has_flicker)
            transition_count = sum(1 for fc in segment_coherences if fc.is_abrupt_transition)
            low_coherence_count = sum(1 for fc in segment_coherences if fc.coherence_score < self.coherence_threshold)

            has_issues = (
                flicker_count > len(segment_coherences) * 0.1 or
                low_coherence_count > len(segment_coherences) * 0.2
            )

            # Quality rating
            if avg_coherence >= 0.85:
                quality_rating = "excellent"
            elif avg_coherence >= 0.7:
                quality_rating = "good"
            elif avg_coherence >= 0.5:
                quality_rating = "fair"
            else:
                quality_rating = "poor"

            segment = TemporalSegment(
                segment_id=segment_id,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration,
                avg_coherence_score=avg_coherence,
                min_coherence_score=min_coherence,
                has_issues=has_issues,
                flicker_count=flicker_count,
                abrupt_transition_count=transition_count,
                low_coherence_count=low_coherence_count,
                quality_rating=quality_rating,
                frame_coherences=segment_coherences
            )

            segments.append(segment)
            segment_id += 1

        logger.info(f"Created {len(segments)} temporal segments")

        return segments

    def _rate_stability(
        self,
        avg_coherence: float,
        flicker_count: int,
        transition_count: int,
        total_frames: int
    ) -> str:
        """Rate overall temporal stability"""
        flicker_ratio = flicker_count / total_frames if total_frames > 0 else 0
        transition_ratio = transition_count / total_frames if total_frames > 0 else 0

        # High coherence, low artifacts
        if avg_coherence >= 0.85 and flicker_ratio < 0.05 and transition_ratio < 0.02:
            return "excellent"
        # Good coherence, few artifacts
        elif avg_coherence >= 0.7 and flicker_ratio < 0.1 and transition_ratio < 0.05:
            return "good"
        # Moderate coherence, some artifacts
        elif avg_coherence >= 0.5 and flicker_ratio < 0.2:
            return "fair"
        # Low coherence or many artifacts
        else:
            return "poor"


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

    # Create checker
    checker = TemporalCoherenceChecker()

    # Check temporal coherence
    result = checker.check_video(
        video_path=video_path,
        sample_interval=1  # Analyze every frame
    )

    # Print results
    print("\n" + "=" * 60)
    print("TEMPORAL COHERENCE CHECK RESULTS")
    print("=" * 60)
    print(f"Video: {result.video_path}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Temporal Stability: {result.temporal_stability_rating.upper()}")
    print(f"Average Coherence Score: {result.avg_coherence_score:.3f}")
    print(f"Minimum Coherence Score: {result.min_coherence_score:.3f}")
    print(f"Flicker Frames: {result.total_flicker_frames}")
    print(f"Abrupt Transitions: {result.total_abrupt_transitions}")
    print(f"Problem Segments: {len(result.problem_segments)}")

    # Show segment summary
    if result.segments:
        print("\nSegment Quality Summary:")
        quality_counts = {}
        for seg in result.segments:
            quality_counts[seg.quality_rating] = quality_counts.get(seg.quality_rating, 0) + 1

        for quality in ["excellent", "good", "fair", "poor"]:
            count = quality_counts.get(quality, 0)
            if count > 0:
                print(f"  {quality.capitalize()}: {count} segments")

    # Show problem segments
    if result.problem_segments:
        print(f"\nProblem Segments (IDs): {result.problem_segments[:10]}")
        if len(result.problem_segments) > 10:
            print(f"  ... and {len(result.problem_segments) - 10} more")

    # Save to JSON
    output_json = "outputs/analysis/temporal_coherence.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    result.save_json(output_json)

    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
