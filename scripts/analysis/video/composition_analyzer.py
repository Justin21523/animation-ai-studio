"""
Composition Analyzer for Video Analysis

Analyzes visual composition of video frames:
- Rule of thirds compliance scoring
- Visual balance analysis (left/right, top/bottom)
- Depth layer detection (foreground, midground, background)
- Subject position identification
- Frame power points (intersection of thirds)

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
class RuleOfThirdsScore:
    """Rule of thirds compliance scoring"""
    overall_score: float  # 0.0 to 1.0
    horizontal_alignment: float  # How well content aligns with horizontal thirds
    vertical_alignment: float    # How well content aligns with vertical thirds
    power_point_usage: float     # How well subjects placed at power points
    power_points: List[Tuple[int, int]] = field(default_factory=list)  # 4 intersection points


@dataclass
class VisualBalance:
    """Visual balance metrics"""
    left_right_balance: float  # -1.0 (left-heavy) to 1.0 (right-heavy), 0.0 is balanced
    top_bottom_balance: float  # -1.0 (top-heavy) to 1.0 (bottom-heavy), 0.0 is balanced
    overall_balance_score: float  # 0.0 (unbalanced) to 1.0 (perfectly balanced)
    left_weight: float
    right_weight: float
    top_weight: float
    bottom_weight: float


@dataclass
class DepthLayers:
    """Depth layer detection results"""
    has_foreground: bool
    has_midground: bool
    has_background: bool
    foreground_ratio: float  # Percentage of frame
    midground_ratio: float
    background_ratio: float
    depth_complexity: float  # 0.0 to 1.0, how well-separated layers are


@dataclass
class SubjectInfo:
    """Dominant subject information"""
    position: Tuple[int, int]  # (x, y) center of subject
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_ratio: float  # Subject area / frame area
    position_label: str  # "center", "left-third", "right-third", etc.
    at_power_point: bool  # Whether subject is near a power point


@dataclass
class CompositionMetrics:
    """Complete composition analysis for a single frame"""
    frame_index: int
    timestamp: float  # seconds
    rule_of_thirds: RuleOfThirdsScore
    visual_balance: VisualBalance
    depth_layers: DepthLayers
    subjects: List[SubjectInfo]
    overall_composition_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "rule_of_thirds": {
                "overall_score": self.rule_of_thirds.overall_score,
                "horizontal_alignment": self.rule_of_thirds.horizontal_alignment,
                "vertical_alignment": self.rule_of_thirds.vertical_alignment,
                "power_point_usage": self.rule_of_thirds.power_point_usage,
                "power_points": self.rule_of_thirds.power_points
            },
            "visual_balance": {
                "left_right_balance": self.visual_balance.left_right_balance,
                "top_bottom_balance": self.visual_balance.top_bottom_balance,
                "overall_balance_score": self.visual_balance.overall_balance_score,
                "left_weight": self.visual_balance.left_weight,
                "right_weight": self.visual_balance.right_weight,
                "top_weight": self.visual_balance.top_weight,
                "bottom_weight": self.visual_balance.bottom_weight
            },
            "depth_layers": {
                "has_foreground": self.depth_layers.has_foreground,
                "has_midground": self.depth_layers.has_midground,
                "has_background": self.depth_layers.has_background,
                "foreground_ratio": self.depth_layers.foreground_ratio,
                "midground_ratio": self.depth_layers.midground_ratio,
                "background_ratio": self.depth_layers.background_ratio,
                "depth_complexity": self.depth_layers.depth_complexity
            },
            "subjects": [
                {
                    "position": list(s.position),
                    "bounding_box": list(s.bounding_box),
                    "area_ratio": s.area_ratio,
                    "position_label": s.position_label,
                    "at_power_point": s.at_power_point
                }
                for s in self.subjects
            ],
            "overall_composition_score": self.overall_composition_score,
            "metadata": self.metadata
        }


@dataclass
class CompositionAnalysisResult:
    """Complete composition analysis result"""
    video_path: str
    total_frames_analyzed: int
    frame_metrics: List[CompositionMetrics]
    avg_composition_score: float
    best_composition_frame: int
    worst_composition_frame: int
    analysis_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_path": self.video_path,
            "total_frames_analyzed": self.total_frames_analyzed,
            "frame_metrics": [m.to_dict() for m in self.frame_metrics],
            "avg_composition_score": self.avg_composition_score,
            "best_composition_frame": self.best_composition_frame,
            "worst_composition_frame": self.worst_composition_frame,
            "analysis_time": self.analysis_time,
            "metadata": self.metadata
        }

    def save_json(self, output_path: str):
        """Save result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved composition analysis to: {output_path}")


class CompositionAnalyzer:
    """
    Composition Analyzer for Video Frames

    Analyzes visual composition using computer vision techniques:
    - Rule of thirds grid analysis
    - Visual weight distribution
    - Depth estimation via edge detection and blur analysis
    - Subject detection via saliency maps

    Usage:
        analyzer = CompositionAnalyzer()
        result = analyzer.analyze_video(video_path, sample_rate=30)
        result.save_json("composition_analysis.json")
    """

    def __init__(
        self,
        enable_visualization: bool = False
    ):
        """
        Initialize composition analyzer

        Args:
            enable_visualization: Whether to generate visualization images
        """
        self.enable_visualization = enable_visualization

        logger.info(f"CompositionAnalyzer initialized (visualization={enable_visualization})")

    def analyze_video(
        self,
        video_path: str,
        sample_rate: int = 30,
        visualization_output_dir: Optional[str] = None
    ) -> CompositionAnalysisResult:
        """
        Analyze composition of video frames

        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame
            visualization_output_dir: Directory to save visualizations

        Returns:
            CompositionAnalysisResult
        """
        start_time = time.time()

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Analyzing composition: {video_path}")
        logger.info(f"Sample rate: every {sample_rate} frames")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {fps:.2f} FPS, {total_frames} frames")

        # Create visualization directory if needed
        if self.enable_visualization and visualization_output_dir:
            Path(visualization_output_dir).mkdir(parents=True, exist_ok=True)

        # Analyze frames
        frame_metrics = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps

                # Analyze this frame
                metrics = self._analyze_frame(frame, frame_idx, timestamp)
                frame_metrics.append(metrics)

                # Generate visualization
                if self.enable_visualization and visualization_output_dir:
                    vis_path = Path(visualization_output_dir) / f"composition_frame_{frame_idx:06d}.jpg"
                    self._visualize_composition(frame, metrics, str(vis_path))

                if len(frame_metrics) % 10 == 0:
                    logger.info(f"Analyzed {len(frame_metrics)} frames ({frame_idx}/{total_frames})")

            frame_idx += 1

        cap.release()

        logger.info(f"Analyzed {len(frame_metrics)} frames total")

        # Calculate aggregate statistics
        if frame_metrics:
            avg_score = np.mean([m.overall_composition_score for m in frame_metrics])
            best_idx = max(range(len(frame_metrics)), key=lambda i: frame_metrics[i].overall_composition_score)
            worst_idx = min(range(len(frame_metrics)), key=lambda i: frame_metrics[i].overall_composition_score)
            best_frame = frame_metrics[best_idx].frame_index
            worst_frame = frame_metrics[worst_idx].frame_index
        else:
            avg_score = 0.0
            best_frame = 0
            worst_frame = 0

        analysis_time = time.time() - start_time

        result = CompositionAnalysisResult(
            video_path=video_path,
            total_frames_analyzed=len(frame_metrics),
            frame_metrics=frame_metrics,
            avg_composition_score=avg_score,
            best_composition_frame=best_frame,
            worst_composition_frame=worst_frame,
            analysis_time=analysis_time,
            metadata={
                "sample_rate": sample_rate,
                "total_video_frames": total_frames,
                "fps": fps
            }
        )

        logger.info(f"Composition analysis completed in {analysis_time:.2f}s")
        logger.info(f"Average composition score: {avg_score:.3f}")
        logger.info(f"Best frame: {best_frame}, Worst frame: {worst_frame}")

        return result

    def _analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> CompositionMetrics:
        """
        Analyze composition of a single frame

        Args:
            frame: Frame image (BGR)
            frame_idx: Frame index
            timestamp: Timestamp in seconds

        Returns:
            CompositionMetrics
        """
        h, w = frame.shape[:2]

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Rule of Thirds Analysis
        rule_of_thirds = self._analyze_rule_of_thirds(frame, gray)

        # 2. Visual Balance Analysis
        visual_balance = self._analyze_visual_balance(gray)

        # 3. Depth Layer Detection
        depth_layers = self._analyze_depth_layers(frame, gray)

        # 4. Subject Detection
        subjects = self._detect_subjects(frame, gray, w, h)

        # Calculate overall composition score
        overall_score = self._calculate_overall_score(
            rule_of_thirds, visual_balance, depth_layers, subjects
        )

        return CompositionMetrics(
            frame_index=frame_idx,
            timestamp=timestamp,
            rule_of_thirds=rule_of_thirds,
            visual_balance=visual_balance,
            depth_layers=depth_layers,
            subjects=subjects,
            overall_composition_score=overall_score
        )

    def _analyze_rule_of_thirds(
        self,
        frame: np.ndarray,
        gray: np.ndarray
    ) -> RuleOfThirdsScore:
        """Analyze rule of thirds compliance"""
        h, w = frame.shape[:2]

        # Calculate third lines
        third_h = h // 3
        third_w = w // 3

        horizontal_lines = [third_h, 2 * third_h]
        vertical_lines = [third_w, 2 * third_w]

        # Power points (intersections)
        power_points = [
            (vertical_lines[0], horizontal_lines[0]),
            (vertical_lines[1], horizontal_lines[0]),
            (vertical_lines[0], horizontal_lines[1]),
            (vertical_lines[1], horizontal_lines[1])
        ]

        # Detect edges to find important content
        edges = cv2.Canny(gray, 50, 150)

        # Calculate alignment scores
        horizontal_score = 0.0
        for line_y in horizontal_lines:
            # Sample pixels along horizontal third line
            line_pixels = edges[max(0, line_y-5):min(h, line_y+5), :]
            horizontal_score += np.sum(line_pixels) / (w * 10 * 255)

        vertical_score = 0.0
        for line_x in vertical_lines:
            # Sample pixels along vertical third line
            line_pixels = edges[:, max(0, line_x-5):min(w, line_x+5)]
            vertical_score += np.sum(line_pixels) / (h * 10 * 255)

        # Normalize scores
        horizontal_score = min(1.0, horizontal_score / 2)
        vertical_score = min(1.0, vertical_score / 2)

        # Calculate power point usage
        power_point_score = 0.0
        for px, py in power_points:
            # Check 20x20 region around power point
            region = edges[max(0, py-10):min(h, py+10), max(0, px-10):min(w, px+10)]
            power_point_score += np.sum(region) / (20 * 20 * 255)

        power_point_score = min(1.0, power_point_score / 4)

        # Overall score
        overall_score = (horizontal_score + vertical_score + power_point_score) / 3

        return RuleOfThirdsScore(
            overall_score=overall_score,
            horizontal_alignment=horizontal_score,
            vertical_alignment=vertical_score,
            power_point_usage=power_point_score,
            power_points=power_points
        )

    def _analyze_visual_balance(self, gray: np.ndarray) -> VisualBalance:
        """Analyze visual weight distribution"""
        h, w = gray.shape

        # Calculate visual weight using pixel intensity
        # Brighter/higher-contrast areas have more visual weight

        # Split frame into halves
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]

        # Calculate weights (sum of pixel intensities)
        left_weight = np.sum(left_half) / (h * (w // 2))
        right_weight = np.sum(right_half) / (h * (w // 2))
        top_weight = np.sum(top_half) / ((h // 2) * w)
        bottom_weight = np.sum(bottom_half) / ((h // 2) * w)

        # Calculate balance scores (-1 to 1, where 0 is balanced)
        left_right_balance = (right_weight - left_weight) / max(right_weight + left_weight, 1e-6)
        top_bottom_balance = (bottom_weight - top_weight) / max(bottom_weight + top_weight, 1e-6)

        # Overall balance score (0 to 1, where 1 is perfectly balanced)
        overall_balance = 1.0 - (abs(left_right_balance) + abs(top_bottom_balance)) / 2

        return VisualBalance(
            left_right_balance=left_right_balance,
            top_bottom_balance=top_bottom_balance,
            overall_balance_score=overall_balance,
            left_weight=float(left_weight),
            right_weight=float(right_weight),
            top_weight=float(top_weight),
            bottom_weight=float(bottom_weight)
        )

    def _analyze_depth_layers(
        self,
        frame: np.ndarray,
        gray: np.ndarray
    ) -> DepthLayers:
        """Estimate depth layers using blur and edge analysis"""
        h, w = gray.shape

        # Method: Use blur detection and edge density
        # Foreground: Sharp, high edge density
        # Background: Blurred, low edge density
        # Midground: Intermediate

        # Calculate local sharpness using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)

        # Normalize
        if laplacian.max() > 0:
            laplacian = laplacian / laplacian.max()

        # Threshold into layers
        foreground_mask = laplacian > 0.6
        background_mask = laplacian < 0.2
        midground_mask = ~(foreground_mask | background_mask)

        # Calculate ratios
        total_pixels = h * w
        foreground_ratio = np.sum(foreground_mask) / total_pixels
        midground_ratio = np.sum(midground_mask) / total_pixels
        background_ratio = np.sum(background_mask) / total_pixels

        # Determine presence
        has_foreground = foreground_ratio > 0.05
        has_midground = midground_ratio > 0.05
        has_background = background_ratio > 0.05

        # Depth complexity: How well-separated are the layers?
        layer_count = sum([has_foreground, has_midground, has_background])
        layer_variance = np.var([foreground_ratio, midground_ratio, background_ratio])
        depth_complexity = min(1.0, (layer_count / 3) * (layer_variance * 10))

        return DepthLayers(
            has_foreground=has_foreground,
            has_midground=has_midground,
            has_background=has_background,
            foreground_ratio=float(foreground_ratio),
            midground_ratio=float(midground_ratio),
            background_ratio=float(background_ratio),
            depth_complexity=float(depth_complexity)
        )

    def _detect_subjects(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        w: int,
        h: int
    ) -> List[SubjectInfo]:
        """Detect dominant subjects using saliency detection"""
        subjects = []

        try:
            # Use static saliency detection (spectral residual method)
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(frame)

            if not success:
                logger.warning("Saliency detection failed")
                return subjects

            # Convert to uint8
            saliency_map = (saliency_map * 255).astype(np.uint8)

            # Threshold to find salient regions
            _, thresh = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and process contours
            min_area = (w * h) * 0.01  # Minimum 1% of frame
            max_area = (w * h) * 0.8   # Maximum 80% of frame

            for contour in contours:
                area = cv2.contourArea(contour)

                if min_area <= area <= max_area:
                    # Get bounding box
                    x, y, bw, bh = cv2.boundingRect(contour)

                    # Calculate center
                    cx = x + bw // 2
                    cy = y + bh // 2

                    # Determine position label
                    position_label = self._get_position_label(cx, cy, w, h)

                    # Check if near power point
                    third_w = w // 3
                    third_h = h // 3
                    power_points = [
                        (third_w, third_h),
                        (2 * third_w, third_h),
                        (third_w, 2 * third_h),
                        (2 * third_w, 2 * third_h)
                    ]

                    at_power_point = any(
                        abs(cx - px) < w * 0.1 and abs(cy - py) < h * 0.1
                        for px, py in power_points
                    )

                    # Area ratio
                    area_ratio = area / (w * h)

                    subjects.append(SubjectInfo(
                        position=(cx, cy),
                        bounding_box=(x, y, x + bw, y + bh),
                        area_ratio=float(area_ratio),
                        position_label=position_label,
                        at_power_point=at_power_point
                    ))

            # Sort by area (largest first)
            subjects.sort(key=lambda s: s.area_ratio, reverse=True)

            # Keep only top 3 subjects
            subjects = subjects[:3]

        except Exception as e:
            logger.warning(f"Subject detection failed: {e}")

        return subjects

    def _get_position_label(self, x: int, y: int, w: int, h: int) -> str:
        """Get position label for coordinates"""
        third_w = w // 3
        third_h = h // 3

        # Determine horizontal position
        if x < third_w:
            h_pos = "left"
        elif x < 2 * third_w:
            h_pos = "center"
        else:
            h_pos = "right"

        # Determine vertical position
        if y < third_h:
            v_pos = "top"
        elif y < 2 * third_h:
            v_pos = "middle"
        else:
            v_pos = "bottom"

        # Combine
        if h_pos == "center" and v_pos == "middle":
            return "center"
        elif h_pos == "center":
            return f"{v_pos}-center"
        elif v_pos == "middle":
            return f"{h_pos}-middle"
        else:
            return f"{v_pos}-{h_pos}"

    def _calculate_overall_score(
        self,
        rule_of_thirds: RuleOfThirdsScore,
        visual_balance: VisualBalance,
        depth_layers: DepthLayers,
        subjects: List[SubjectInfo]
    ) -> float:
        """Calculate overall composition score"""
        # Weighted combination of factors
        score = 0.0

        # Rule of thirds (30%)
        score += rule_of_thirds.overall_score * 0.3

        # Visual balance (25%)
        score += visual_balance.overall_balance_score * 0.25

        # Depth complexity (20%)
        score += depth_layers.depth_complexity * 0.2

        # Subject positioning (25%)
        if subjects:
            # Bonus for subjects at power points
            power_point_bonus = sum(1 for s in subjects if s.at_power_point) / len(subjects)
            # Bonus for subjects in third-aligned positions
            third_positions = ["left-middle", "right-middle", "top-center", "bottom-center", "center"]
            position_bonus = sum(1 for s in subjects if s.position_label in third_positions) / len(subjects)

            subject_score = (power_point_bonus + position_bonus) / 2
            score += subject_score * 0.25
        else:
            # No subjects detected, neutral score
            score += 0.125

        return min(1.0, max(0.0, score))

    def _visualize_composition(
        self,
        frame: np.ndarray,
        metrics: CompositionMetrics,
        output_path: str
    ):
        """Generate composition visualization"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw rule of thirds grid
        third_h = h // 3
        third_w = w // 3

        # Horizontal lines
        cv2.line(vis, (0, third_h), (w, third_h), (0, 255, 0), 2)
        cv2.line(vis, (0, 2 * third_h), (w, 2 * third_h), (0, 255, 0), 2)

        # Vertical lines
        cv2.line(vis, (third_w, 0), (third_w, h), (0, 255, 0), 2)
        cv2.line(vis, (2 * third_w, 0), (2 * third_w, h), (0, 255, 0), 2)

        # Draw power points
        for px, py in metrics.rule_of_thirds.power_points:
            cv2.circle(vis, (px, py), 10, (0, 255, 255), -1)

        # Draw subjects
        for i, subject in enumerate(metrics.subjects):
            x1, y1, x2, y2 = subject.bounding_box
            color = (255, 0, 0) if subject.at_power_point else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            cv2.circle(vis, subject.position, 5, color, -1)

            # Label
            label = f"S{i+1}: {subject.position_label}"
            cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add metrics text
        y_offset = 30
        cv2.putText(vis, f"Overall: {metrics.overall_composition_score:.3f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(vis, f"RoT: {metrics.rule_of_thirds.overall_score:.3f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(vis, f"Balance: {metrics.visual_balance.overall_balance_score:.3f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(vis, f"Depth: {metrics.depth_layers.depth_complexity:.3f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save visualization
        cv2.imwrite(output_path, vis)


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

    # Create analyzer
    analyzer = CompositionAnalyzer(enable_visualization=True)

    # Analyze video
    result = analyzer.analyze_video(
        video_path=video_path,
        sample_rate=30,  # Analyze every 30th frame
        visualization_output_dir="outputs/analysis/composition_vis"
    )

    # Print results
    print("\n" + "=" * 60)
    print("COMPOSITION ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Video: {result.video_path}")
    print(f"Frames Analyzed: {result.total_frames_analyzed}")
    print(f"Average Composition Score: {result.avg_composition_score:.3f}")
    print(f"Best Frame: {result.best_composition_frame}")
    print(f"Worst Frame: {result.worst_composition_frame}")
    print(f"Analysis Time: {result.analysis_time:.2f}s")

    # Show sample metrics
    if result.frame_metrics:
        print("\nSample Frame Analysis (first frame):")
        m = result.frame_metrics[0]
        print(f"  Frame: {m.frame_index}, Time: {m.timestamp:.2f}s")
        print(f"  Overall Score: {m.overall_composition_score:.3f}")
        print(f"  Rule of Thirds: {m.rule_of_thirds.overall_score:.3f}")
        print(f"  Visual Balance: {m.visual_balance.overall_balance_score:.3f}")
        print(f"  Depth Complexity: {m.depth_layers.depth_complexity:.3f}")
        print(f"  Subjects Detected: {len(m.subjects)}")

    # Save to JSON
    output_json = "outputs/analysis/composition_analysis.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    result.save_json(output_json)

    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
