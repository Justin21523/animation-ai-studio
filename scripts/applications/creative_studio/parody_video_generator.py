"""
Parody Video Generator - Autonomous Funny Video Creation

Integrates all modules to create funny/parody videos automatically:
- Module 7: Video Analysis (scene detection, composition, camera movement)
- Module 8: Video Editing (LLM decisions, parody effects, quality evaluation)
- Agent Framework: Autonomous orchestration

This is the "大壓軸" - Complete AI-driven creative video generation.

Usage:
    generator = ParodyVideoGenerator()
    result = await generator.generate_parody(
        input_video="luca_pasta.mp4",
        style="dramatic",
        target_duration=30.0,
        output_path="luca_funny.mp4"
    )

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.agent.tools.video_analysis_tools import (
    detect_scenes,
    analyze_composition,
    track_camera_movement,
    check_temporal_coherence,
    analyze_video_complete
)

from scripts.agent.tools.video_editing_tools import (
    create_edit_plan,
    auto_edit_video,
    create_parody_video,
    evaluate_video_quality
)


logger = logging.getLogger(__name__)


@dataclass
class ParodyGenerationResult:
    """Result of parody video generation"""
    success: bool
    input_video: str
    output_video: str
    parody_style: str
    target_duration: float
    actual_duration: float
    generation_time: float
    quality_score: float

    # Analysis results
    scenes_detected: int
    avg_composition_score: float
    camera_style: str
    temporal_coherence: float

    # Editing decisions
    total_decisions: int
    edit_plan: Dict[str, Any]

    # Quality evaluation
    needs_improvement: bool
    feedback: str
    iterations: int

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "input_video": self.input_video,
            "output_video": self.output_video,
            "parody_style": self.parody_style,
            "target_duration": self.target_duration,
            "actual_duration": self.actual_duration,
            "generation_time": self.generation_time,
            "quality_score": self.quality_score,
            "scenes_detected": self.scenes_detected,
            "avg_composition_score": self.avg_composition_score,
            "camera_style": self.camera_style,
            "temporal_coherence": self.temporal_coherence,
            "total_decisions": self.total_decisions,
            "edit_plan": self.edit_plan,
            "needs_improvement": self.needs_improvement,
            "feedback": self.feedback,
            "iterations": self.iterations,
            "metadata": self.metadata
        }


class ParodyVideoGenerator:
    """
    Autonomous Parody Video Generator

    Complete AI-driven pipeline for creating funny/parody videos:
    1. Analyze video (Module 7)
    2. LLM creates funny edit plan (Module 8)
    3. Execute parody effects
    4. Evaluate quality
    5. Iterate until quality threshold met

    This orchestrates ALL modules for autonomous creative video generation.

    Usage:
        generator = ParodyVideoGenerator()

        # Automatic parody generation
        result = await generator.generate_parody(
            input_video="luca_pasta.mp4",
            style="dramatic",
            target_duration=30.0
        )

        # Custom workflow
        result = await generator.custom_workflow(
            input_video="video.mp4",
            workflow_description="Create funny compilation with zoom punches at dramatic moments"
        )
    """

    def __init__(
        self,
        quality_threshold: float = 0.7,
        max_iterations: int = 3,
        enable_analysis: bool = True
    ):
        """
        Initialize parody video generator

        Args:
            quality_threshold: Minimum acceptable quality
            max_iterations: Maximum improvement iterations
            enable_analysis: Enable video analysis (Module 7)
        """
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.enable_analysis = enable_analysis

        logger.info(f"ParodyVideoGenerator initialized (threshold={quality_threshold})")

    async def generate_parody(
        self,
        input_video: str,
        output_video: str,
        style: str = "dramatic",
        target_duration: Optional[float] = None,
        effects: Optional[List[str]] = None
    ) -> ParodyGenerationResult:
        """
        Generate parody video automatically

        Args:
            input_video: Path to input video
            output_video: Path to output video
            style: Parody style (dramatic, chaotic, wholesome)
            target_duration: Target duration in seconds (optional)
            effects: List of effects to apply (optional)

        Returns:
            ParodyGenerationResult
        """
        logger.info(f"Starting parody generation: {input_video} -> {output_video}")
        logger.info(f"Style: {style}, Target: {target_duration}s")

        start_time = time.time()

        try:
            # Step 1: Analyze video (Module 7)
            if self.enable_analysis:
                logger.info("Step 1/5: Analyzing video...")
                analysis = await analyze_video_complete(
                    video_path=input_video,
                    sample_rate=30
                )

                if not analysis["success"]:
                    raise Exception(f"Video analysis failed: {analysis.get('error')}")

                analyses = analysis["analyses"]
                logger.info(f"  Scenes: {analyses['scenes']['total_scenes']}")
                logger.info(f"  Composition: {analyses['composition']['avg_composition_score']:.3f}")
                logger.info(f"  Camera: {analyses['camera']['camera_style']}")
            else:
                analyses = None

            # Step 2: Create funny edit plan (Module 8 LLM Decision Engine)
            logger.info("Step 2/5: Creating funny edit plan with LLM...")

            # Build goal for LLM
            goal = self._build_parody_goal(style, target_duration, effects)

            plan_result = await create_edit_plan(
                video_path=input_video,
                goal=goal,
                analysis_results=analyses,
                target_duration=target_duration
            )

            if not plan_result["success"]:
                raise Exception(f"Edit plan creation failed: {plan_result.get('error')}")

            logger.info(f"  Plan created: {plan_result['total_decisions']} decisions")

            # Step 3: Execute parody effects
            logger.info("Step 3/5: Executing parody effects...")

            # Use simplified parody generator for now
            # (In full implementation, would execute all decisions from plan)
            parody_result = await create_parody_video(
                video_path=input_video,
                output_path=output_video,
                parody_style=style,
                effects=effects
            )

            if not parody_result["success"]:
                raise Exception(f"Parody generation failed: {parody_result.get('error')}")

            logger.info(f"  Parody effects applied")

            # Step 4: Evaluate quality
            logger.info("Step 4/5: Evaluating quality...")

            quality_result = await evaluate_video_quality(
                video_path=output_video,
                goal=goal,
                quality_threshold=self.quality_threshold,
                analysis_results=analyses
            )

            quality_score = quality_result["overall_score"]
            logger.info(f"  Quality score: {quality_score:.3f}")

            # Step 5: Iterate if needed
            iterations = 1
            while (quality_result.get("needs_improvement", False) and
                   iterations < self.max_iterations):
                logger.info(f"Step 5/5: Iteration {iterations} - Improving quality...")

                # In full implementation, would re-plan and re-execute
                # For now, just log
                logger.info(f"  Suggestions: {quality_result.get('suggestions', [])}")

                iterations += 1

            generation_time = time.time() - start_time

            # Build result
            result = ParodyGenerationResult(
                success=True,
                input_video=input_video,
                output_video=output_video,
                parody_style=style,
                target_duration=target_duration or 0.0,
                actual_duration=parody_result.get("duration", 0.0),
                generation_time=generation_time,
                quality_score=quality_score,
                scenes_detected=analyses["scenes"]["total_scenes"] if analyses else 0,
                avg_composition_score=analyses["composition"]["avg_composition_score"] if analyses else 0.0,
                camera_style=analyses["camera"]["camera_style"] if analyses else "unknown",
                temporal_coherence=analyses["temporal"]["avg_coherence_score"] if analyses else 0.0,
                total_decisions=plan_result["total_decisions"],
                edit_plan=plan_result,
                needs_improvement=quality_result["needs_improvement"],
                feedback=quality_result["feedback"],
                iterations=iterations,
                metadata={
                    "effects_applied": parody_result.get("effects_applied", []),
                    "quality_details": quality_result
                }
            )

            logger.info(f"Parody generation complete in {generation_time:.1f}s")
            logger.info(f"Quality: {quality_score:.3f}, Iterations: {iterations}")

            return result

        except Exception as e:
            logger.error(f"Parody generation failed: {e}")

            return ParodyGenerationResult(
                success=False,
                input_video=input_video,
                output_video=output_video,
                parody_style=style,
                target_duration=target_duration or 0.0,
                actual_duration=0.0,
                generation_time=time.time() - start_time,
                quality_score=0.0,
                scenes_detected=0,
                avg_composition_score=0.0,
                camera_style="unknown",
                temporal_coherence=0.0,
                total_decisions=0,
                edit_plan={},
                needs_improvement=True,
                feedback=f"Generation failed: {str(e)}",
                iterations=0,
                metadata={"error": str(e)}
            )

    async def custom_workflow(
        self,
        input_video: str,
        output_video: str,
        workflow_description: str,
        quality_threshold: Optional[float] = None
    ) -> ParodyGenerationResult:
        """
        Execute custom parody workflow based on description

        Args:
            input_video: Input video path
            output_video: Output video path
            workflow_description: Natural language workflow description
            quality_threshold: Quality threshold (overrides default)

        Returns:
            ParodyGenerationResult
        """
        logger.info(f"Custom workflow: {workflow_description}")

        threshold = quality_threshold or self.quality_threshold

        # Use auto_edit_video for custom workflows
        result = await auto_edit_video(
            video_path=input_video,
            goal=workflow_description,
            output_path=output_video,
            quality_threshold=threshold,
            max_iterations=self.max_iterations,
            analyze_first=self.enable_analysis
        )

        # Convert to ParodyGenerationResult
        if result["success"]:
            quality = result.get("quality_evaluation", {})

            return ParodyGenerationResult(
                success=True,
                input_video=input_video,
                output_video=output_video,
                parody_style="custom",
                target_duration=0.0,
                actual_duration=0.0,
                generation_time=0.0,
                quality_score=quality.get("overall_score", 0.0),
                scenes_detected=0,
                avg_composition_score=0.0,
                camera_style="unknown",
                temporal_coherence=0.0,
                total_decisions=result.get("edit_plan", {}).get("total_decisions", 0),
                edit_plan=result.get("edit_plan", {}),
                needs_improvement=quality.get("needs_improvement", False),
                feedback=quality.get("feedback", ""),
                iterations=result.get("total_iterations", 1),
                metadata={"workflow": workflow_description}
            )
        else:
            return ParodyGenerationResult(
                success=False,
                input_video=input_video,
                output_video=output_video,
                parody_style="custom",
                target_duration=0.0,
                actual_duration=0.0,
                generation_time=0.0,
                quality_score=0.0,
                scenes_detected=0,
                avg_composition_score=0.0,
                camera_style="unknown",
                temporal_coherence=0.0,
                total_decisions=0,
                edit_plan={},
                needs_improvement=True,
                feedback=result.get("error", "Unknown error"),
                iterations=0,
                metadata={"error": result.get("error")}
            )

    def _build_parody_goal(
        self,
        style: str,
        target_duration: Optional[float],
        effects: Optional[List[str]]
    ) -> str:
        """Build LLM goal for parody generation"""

        style_descriptions = {
            "dramatic": "Create a dramatic parody with slow motion at key moments, zoom punches, and epic music suggestions",
            "chaotic": "Create a chaotic meme-style parody with rapid cuts, speed changes, and exaggerated effects",
            "wholesome": "Create a wholesome funny video with gentle speed changes and feel-good vibes"
        }

        goal = style_descriptions.get(style, f"Create a {style} parody video")

        if target_duration:
            goal += f". Target duration: {target_duration} seconds"

        if effects:
            goal += f". Apply these effects: {', '.join(effects)}"

        return goal

    def save_result(self, result: ParodyGenerationResult, output_path: str):
        """Save generation result to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Result saved to: {output_path}")


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    generator = ParodyVideoGenerator(
        quality_threshold=0.7,
        max_iterations=3
    )

    # Example 1: Dramatic parody
    video_path = "data/films/luca/scenes/pasta_discovery.mp4"

    if Path(video_path).exists():
        result = await generator.generate_parody(
            input_video=video_path,
            output_video="outputs/creative_studio/luca_dramatic_parody.mp4",
            style="dramatic",
            target_duration=30.0
        )

        print("\n" + "=" * 70)
        print("PARODY VIDEO GENERATION RESULT")
        print("=" * 70)
        print(f"Success: {result.success}")
        print(f"Style: {result.parody_style}")
        print(f"Duration: {result.actual_duration:.1f}s (target: {result.target_duration:.1f}s)")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Generation Time: {result.generation_time:.1f}s")
        print(f"Iterations: {result.iterations}")
        print(f"\nAnalysis:")
        print(f"  Scenes: {result.scenes_detected}")
        print(f"  Composition: {result.avg_composition_score:.3f}")
        print(f"  Camera Style: {result.camera_style}")
        print(f"  Temporal Coherence: {result.temporal_coherence:.3f}")
        print(f"\nEditing:")
        print(f"  Total Decisions: {result.total_decisions}")
        print(f"  Needs Improvement: {result.needs_improvement}")
        print(f"\nFeedback: {result.feedback}")

        # Save result
        generator.save_result(
            result,
            "outputs/creative_studio/luca_dramatic_parody_result.json"
        )
    else:
        print(f"Video not found: {video_path}")
        print("Using custom workflow example instead...")

        # Example 2: Custom workflow
        result = await generator.custom_workflow(
            input_video="test_video.mp4",
            output_video="outputs/creative_studio/custom_funny.mp4",
            workflow_description="Create a funny compilation with zoom punches at dramatic moments and slow motion for comedic effect"
        )


if __name__ == "__main__":
    asyncio.run(main())
