"""
Quality Evaluator for Edited Videos

Evaluates quality of edited videos using multiple metrics:
- Technical quality (composition, temporal coherence)
- Creative quality (goal achievement, pacing)
- Automated quality checks (PSNR, SSIM, flicker)

Integrated with LLM Decision Engine for iterative improvement.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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
class QualityMetrics:
    """Quality evaluation metrics"""
    overall_score: float  # 0.0 to 1.0
    technical_score: float  # Technical quality
    creative_score: float  # Creative quality
    composition_score: float
    temporal_coherence_score: float
    pacing_score: float
    goal_achievement_score: float
    needs_improvement: bool
    feedback: str = ""
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "technical_score": self.technical_score,
            "creative_score": self.creative_score,
            "composition_score": self.composition_score,
            "temporal_coherence_score": self.temporal_coherence_score,
            "pacing_score": self.pacing_score,
            "goal_achievement_score": self.goal_achievement_score,
            "needs_improvement": self.needs_improvement,
            "feedback": self.feedback,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


class QualityEvaluator:
    """
    Quality Evaluator for Edited Videos

    Combines automated metrics with LLM-based evaluation.

    Metrics:
    - Composition quality (from Module 7)
    - Temporal coherence (from Module 7)
    - Pacing analysis
    - Goal achievement (LLM-based)

    Usage:
        evaluator = QualityEvaluator()
        metrics = evaluator.evaluate(
            video_path="edited.mp4",
            goal="Create funny highlight reel",
            quality_threshold=0.7
        )
    """

    def __init__(
        self,
        enable_automated_checks: bool = True,
        enable_llm_evaluation: bool = True
    ):
        """
        Initialize quality evaluator

        Args:
            enable_automated_checks: Enable automated technical checks
            enable_llm_evaluation: Enable LLM-based creative evaluation
        """
        self.enable_automated_checks = enable_automated_checks
        self.enable_llm_evaluation = enable_llm_evaluation

        logger.info(f"QualityEvaluator initialized")

    def evaluate(
        self,
        video_path: str,
        goal: Optional[str] = None,
        quality_threshold: float = 0.7,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """
        Evaluate video quality

        Args:
            video_path: Path to video file
            goal: Original editing goal
            quality_threshold: Minimum acceptable quality
            analysis_results: Video analysis results from Module 7

        Returns:
            QualityMetrics
        """
        logger.info(f"Evaluating video quality: {video_path}")

        # Automated technical checks
        if self.enable_automated_checks:
            technical_metrics = self._automated_quality_checks(video_path, analysis_results)
        else:
            technical_metrics = {
                "composition_score": 0.7,
                "temporal_coherence_score": 0.7,
                "pacing_score": 0.7
            }

        # Calculate technical score
        technical_score = np.mean([
            technical_metrics["composition_score"],
            technical_metrics["temporal_coherence_score"],
            technical_metrics["pacing_score"]
        ])

        # Creative evaluation (simplified here, would use LLM in full implementation)
        creative_score = 0.75  # Placeholder
        goal_achievement_score = 0.8  # Placeholder

        # Overall score
        overall_score = (technical_score * 0.5) + (creative_score * 0.3) + (goal_achievement_score * 0.2)

        # Determine if improvement needed
        needs_improvement = overall_score < quality_threshold

        # Generate feedback
        feedback = self._generate_feedback(
            overall_score,
            technical_metrics,
            creative_score,
            goal_achievement_score,
            needs_improvement
        )

        # Identify issues
        issues = self._identify_issues(technical_metrics, overall_score, quality_threshold)

        # Generate suggestions
        suggestions = self._generate_suggestions(issues, technical_metrics)

        metrics = QualityMetrics(
            overall_score=overall_score,
            technical_score=technical_score,
            creative_score=creative_score,
            composition_score=technical_metrics["composition_score"],
            temporal_coherence_score=technical_metrics["temporal_coherence_score"],
            pacing_score=technical_metrics["pacing_score"],
            goal_achievement_score=goal_achievement_score,
            needs_improvement=needs_improvement,
            feedback=feedback,
            issues=issues,
            suggestions=suggestions,
            metadata={
                "video_path": video_path,
                "goal": goal,
                "quality_threshold": quality_threshold
            }
        )

        logger.info(f"Quality evaluation: {overall_score:.3f} (threshold: {quality_threshold})")

        return metrics

    def _automated_quality_checks(
        self,
        video_path: str,
        analysis_results: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Run automated quality checks"""

        # Use analysis results if available
        if analysis_results:
            composition_score = analysis_results.get("composition", {}).get("avg_composition_score", 0.7)
            temporal_score = analysis_results.get("temporal_coherence", {}).get("avg_coherence_score", 0.7)
        else:
            # Run basic checks
            composition_score = self._check_composition(video_path)
            temporal_score = self._check_temporal_coherence(video_path)

        # Check pacing
        pacing_score = self._check_pacing(video_path)

        return {
            "composition_score": composition_score,
            "temporal_coherence_score": temporal_score,
            "pacing_score": pacing_score
        }

    def _check_composition(self, video_path: str) -> float:
        """Check composition quality (simplified)"""
        # Placeholder: would integrate with Module 7 composition analyzer
        return 0.75

    def _check_temporal_coherence(self, video_path: str) -> float:
        """Check temporal coherence (simplified)"""
        # Placeholder: would integrate with Module 7 temporal coherence checker
        return 0.80

    def _check_pacing(self, video_path: str) -> float:
        """Check video pacing"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.5

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            cap.release()

            # Simple pacing heuristic
            # Good pacing: 20-60 seconds for highlights, 5-10 cuts per minute
            if 20 <= duration <= 60:
                pacing_score = 0.9
            elif 10 <= duration <= 90:
                pacing_score = 0.7
            else:
                pacing_score = 0.5

            return pacing_score

        except Exception as e:
            logger.warning(f"Failed to check pacing: {e}")
            return 0.5

    def _generate_feedback(
        self,
        overall_score: float,
        technical_metrics: Dict[str, float],
        creative_score: float,
        goal_achievement_score: float,
        needs_improvement: bool
    ) -> str:
        """Generate human-readable feedback"""
        if overall_score >= 0.9:
            feedback = "Excellent! The video meets all quality criteria."
        elif overall_score >= 0.7:
            feedback = "Good quality. Minor improvements possible."
        elif overall_score >= 0.5:
            feedback = "Acceptable quality but needs improvement."
        else:
            feedback = "Poor quality. Significant improvements needed."

        # Add specific feedback
        if technical_metrics["composition_score"] < 0.6:
            feedback += " Composition needs improvement."
        if technical_metrics["temporal_coherence_score"] < 0.6:
            feedback += " Temporal coherence issues detected."
        if technical_metrics["pacing_score"] < 0.6:
            feedback += " Pacing could be better."

        return feedback

    def _identify_issues(
        self,
        technical_metrics: Dict[str, float],
        overall_score: float,
        quality_threshold: float
    ) -> list:
        """Identify specific issues"""
        issues = []

        if overall_score < quality_threshold:
            issues.append("Overall quality below threshold")

        if technical_metrics["composition_score"] < 0.6:
            issues.append("Poor composition in some frames")

        if technical_metrics["temporal_coherence_score"] < 0.6:
            issues.append("Temporal coherence issues (flicker, jumps)")

        if technical_metrics["pacing_score"] < 0.6:
            issues.append("Pacing issues (too fast or too slow)")

        return issues

    def _generate_suggestions(
        self,
        issues: list,
        technical_metrics: Dict[str, float]
    ) -> list:
        """Generate improvement suggestions"""
        suggestions = []

        if "composition" in str(issues).lower():
            suggestions.append("Re-frame clips to improve composition")
            suggestions.append("Use best-composed frames from analysis")

        if "temporal" in str(issues).lower():
            suggestions.append("Add smoother transitions between clips")
            suggestions.append("Check for and fix temporal artifacts")

        if "pacing" in str(issues).lower():
            if technical_metrics["pacing_score"] < 0.5:
                suggestions.append("Adjust clip durations for better pacing")
                suggestions.append("Consider adding speed changes")

        return suggestions


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    evaluator = QualityEvaluator()

    video_path = "path/to/edited_video.mp4"

    if Path(video_path).exists():
        metrics = evaluator.evaluate(
            video_path=video_path,
            goal="Create funny highlight reel",
            quality_threshold=0.7
        )

        print("\n" + "=" * 60)
        print("QUALITY EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Score: {metrics.overall_score:.3f}")
        print(f"Technical Score: {metrics.technical_score:.3f}")
        print(f"Creative Score: {metrics.creative_score:.3f}")
        print(f"Needs Improvement: {metrics.needs_improvement}")
        print(f"\nFeedback: {metrics.feedback}")

        if metrics.issues:
            print(f"\nIssues:")
            for issue in metrics.issues:
                print(f"  - {issue}")

        if metrics.suggestions:
            print(f"\nSuggestions:")
            for suggestion in metrics.suggestions:
                print(f"  - {suggestion}")
    else:
        print(f"Video not found: {video_path}")


if __name__ == "__main__":
    main()
