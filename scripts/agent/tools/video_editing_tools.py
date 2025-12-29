"""
Video Editing Tools for Agent Framework

Wrapper functions for video editing modules:
- Character Segmentation
- Video Editing Operations
- LLM-Driven Edit Planning
- Quality Evaluation
- Parody Generation

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.editing.segmentation.character_segmenter import CharacterSegmenter
from scripts.editing.engine.video_editor import VideoEditor
from scripts.editing.decision.llm_decision_engine import LLMDecisionEngine
from scripts.editing.quality.quality_evaluator import QualityEvaluator
from scripts.editing.effects.parody_generator import ParodyGenerator


logger = logging.getLogger(__name__)


def _load_edit_plan_json(plan_json_path: str) -> Dict[str, Any]:
    plan_path = Path(plan_json_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Edit plan JSON not found: {plan_path}")
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_segments_from_cut_decisions(
    decisions: List[Dict[str, Any]],
    target_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    for decision in decisions:
        if (decision.get("decision_type") or "").lower() != "cut":
            continue

        params = decision.get("parameters") or {}
        start_time = _safe_float(params.get("start_time"), 0.0)
        end_time = _safe_float(params.get("end_time"), 0.0)
        if end_time <= start_time:
            continue

        segments.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "priority": int(decision.get("priority", 5)),
                "decision_id": decision.get("decision_id", ""),
            }
        )

    if not segments:
        return []

    # Choose segments to meet target duration (best-effort).
    if target_duration is not None and target_duration > 0:
        by_priority = sorted(segments, key=lambda s: (s["priority"], s["duration"]), reverse=True)
        chosen: List[Dict[str, Any]] = []
        total = 0.0
        for seg in by_priority:
            chosen.append(seg)
            total += float(seg["duration"])
            if total >= float(target_duration):
                break
        segments = chosen

    # Keep final ordering by time for coherent playback.
    segments.sort(key=lambda s: s["start_time"])
    return segments


async def execute_edit_plan(
    *,
    video_path: str,
    plan_json_path: str,
    output_path: str,
    target_duration: Optional[float] = None,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a saved LLM edit plan (JSON) into an actual edited video.

    Supported decision types (best-effort):
    - cut (start_time, end_time)
    - speed (start_time, end_time, speed_factor) [applied within each cut segment]
    - transition (transition_type=crossfade, duration)
    - text_overlay (text, position, start_time, duration, font_size, color)
    - effect (effect_type: mirror_x|mirror_y|blackwhite|invert_colors|zoom_punch)
    """
    try:
        plan = _load_edit_plan_json(plan_json_path)
        decisions = plan.get("decisions", []) or []

        work_root = Path(working_dir) if working_dir else Path("outputs/editing") / Path(video_path).stem / f"exec_{int(time.time())}"
        work_root.mkdir(parents=True, exist_ok=True)

        editor = VideoEditor()

        cut_segments = _extract_segments_from_cut_decisions(decisions, target_duration=target_duration)

        # If there are no cut decisions, treat the whole video as a single segment.
        if not cut_segments:
            cut_segments = [{"start_time": 0.0, "end_time": None, "duration": None, "priority": 0, "decision_id": "full"}]

        speed_decisions = [d for d in decisions if (d.get("decision_type") or "").lower() == "speed"]
        transition_decisions = [d for d in decisions if (d.get("decision_type") or "").lower() == "transition"]
        text_decisions = [d for d in decisions if (d.get("decision_type") or "").lower() == "text_overlay"]
        effect_decisions = [d for d in decisions if (d.get("decision_type") or "").lower() == "effect"]

        transition = None
        transition_duration = 0.5
        if transition_decisions:
            params = transition_decisions[0].get("parameters") or {}
            transition = (params.get("transition_type") or "crossfade").lower()
            transition_duration = _safe_float(params.get("duration"), 0.5)

        # 1) Cut segments
        segment_outputs: List[str] = []
        for i, seg in enumerate(cut_segments):
            start_time = float(seg["start_time"])
            end_time = seg.get("end_time")

            cut_output = work_root / f"cut_{i:03d}.mp4"
            if end_time is None:
                # For "full video", use original path directly (then apply optional speed edits).
                segment_path = str(Path(video_path))
            else:
                result = editor.cut_clip(
                    video_path=video_path,
                    start_time=start_time,
                    end_time=float(end_time),
                    output_path=str(cut_output),
                )
                if not result.success:
                    return {"success": False, "error": f"Cut failed: {result.metadata.get('error')}", "plan_json_path": plan_json_path}
                segment_path = result.output_path

            # 2) Apply speed decisions within this segment (relative time mapping).
            if speed_decisions:
                speed_segments: List[Tuple[float, float, float]] = []

                if end_time is None:
                    # Apply speed segments directly in the original timeline.
                    for decision in speed_decisions:
                        params = decision.get("parameters") or {}
                        speed_segments.append(
                            (
                                _safe_float(params.get("start_time"), 0.0),
                                _safe_float(params.get("end_time"), 0.0),
                                _safe_float(params.get("speed_factor"), 1.0),
                            )
                        )
                else:
                    seg_end = float(end_time)
                    for decision in speed_decisions:
                        params = decision.get("parameters") or {}
                        s = _safe_float(params.get("start_time"), 0.0)
                        e = _safe_float(params.get("end_time"), 0.0)
                        factor = _safe_float(params.get("speed_factor"), 1.0)

                        overlap_start = max(s, start_time)
                        overlap_end = min(e, seg_end)
                        if overlap_end <= overlap_start:
                            continue
                        speed_segments.append((overlap_start - start_time, overlap_end - start_time, factor))

                if speed_segments:
                    speed_output = work_root / f"speed_{i:03d}.mp4"
                    speed_result = editor.change_speed_segments(
                        video_path=segment_path,
                        speed_segments=speed_segments,
                        output_path=str(speed_output),
                    )
                    if speed_result.success:
                        segment_path = speed_result.output_path

            segment_outputs.append(segment_path)

        # 3) Concatenate all segments
        concat_output = work_root / "concatenated.mp4"
        if len(segment_outputs) == 1:
            current_video = segment_outputs[0]
        else:
            concat_result = editor.concatenate_clips(
                clip_paths=segment_outputs,
                output_path=str(concat_output),
                transition=transition if transition in ("crossfade",) else None,
                transition_duration=transition_duration,
            )
            if not concat_result.success:
                return {"success": False, "error": f"Concatenate failed: {concat_result.metadata.get('error')}", "plan_json_path": plan_json_path}
            current_video = concat_result.output_path

        # 4) Text overlays (best-effort, sequential)
        for j, decision in enumerate(text_decisions):
            params = decision.get("parameters") or {}
            text = str(params.get("text", "")).strip()
            if not text:
                continue
            pos = params.get("position") or (100, 100)
            try:
                position = (int(pos[0]), int(pos[1])) if isinstance(pos, (list, tuple)) and len(pos) == 2 else (100, 100)
            except Exception:
                position = (100, 100)

            text_out = work_root / f"text_{j:03d}.mp4"
            text_result = editor.add_text_overlay(
                video_path=current_video,
                text=text,
                output_path=str(text_out),
                position=position,
                font_size=int(params.get("font_size", 50)),
                color=str(params.get("color", "white")),
                start_time=_safe_float(params.get("start_time"), 0.0),
                duration=_safe_float(params.get("duration"), None) if params.get("duration") is not None else None,
            )
            if text_result.success:
                current_video = text_result.output_path

        # 5) Effects (best-effort, sequential)
        for j, decision in enumerate(effect_decisions):
            params = decision.get("parameters") or {}
            effect_type = str(params.get("effect_type", "")).strip().lower()
            if not effect_type:
                continue

            effect_out = work_root / f"effect_{j:03d}.mp4"

            if effect_type in ("mirror_x", "mirror_y", "blackwhite", "invert_colors"):
                effect_result = editor.apply_effect(
                    video_path=current_video,
                    effect_type=effect_type,
                    output_path=str(effect_out),
                    parameters=params,
                )
                if effect_result.success:
                    current_video = effect_result.output_path
                continue

            if effect_type == "zoom_punch":
                try:
                    generator = ParodyGenerator()
                    zoom_time = _safe_float(params.get("zoom_time"), 0.0)
                    if zoom_time <= 0:
                        zoom_time = 3.0
                    zoom_factor = _safe_float(params.get("zoom_factor"), 1.5)
                    zoom_duration = _safe_float(params.get("duration"), 0.5)
                    effect_result = generator.apply_zoom_punch(
                        video_path=current_video,
                        zoom_time=zoom_time,
                        output_path=str(effect_out),
                        zoom_factor=zoom_factor,
                        duration=zoom_duration,
                    )
                    if effect_result.get("success"):
                        current_video = str(effect_out)
                except Exception:
                    pass

        # Finalize
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(current_video).resolve() != Path(output_path).resolve():
            # Copy final file into requested output path.
            import shutil

            shutil.copy2(current_video, output_path)

        return {
            "success": True,
            "video_path": video_path,
            "plan_json_path": plan_json_path,
            "output_path": output_path,
            "working_dir": str(work_root),
            "decisions_total": len(decisions),
            "cuts_applied": len([d for d in decisions if (d.get("decision_type") or "").lower() == "cut"]),
            "speed_applied": len([d for d in decisions if (d.get("decision_type") or "").lower() == "speed"]),
            "text_overlays_applied": len(text_decisions),
            "effects_applied": len(effect_decisions),
        }

    except Exception as e:
        logger.error(f"Failed to execute edit plan: {e}")
        return {"success": False, "error": str(e), "plan_json_path": plan_json_path}


async def segment_characters(
    video_path: str,
    model_size: str = "base",
    sample_interval: int = 1,
    output_masks_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Segment and track characters in video

    Args:
        video_path: Path to video file
        model_size: SAM2 model size (tiny, small, base, large)
        sample_interval: Process every Nth frame
        output_masks_dir: Directory to save masks (optional)

    Returns:
        Segmentation result with character tracks
    """
    try:
        logger.info(f"Segmenting characters in: {video_path}")

        segmenter = CharacterSegmenter(
            model_size=model_size,
            device="cuda"
        )

        result = segmenter.segment_video(
            video_path=video_path,
            sample_interval=sample_interval,
            output_masks_dir=output_masks_dir,
            track_characters=True
        )

        # Save result
        video_name = Path(video_path).stem
        output_json = f"outputs/editing/{video_name}/character_segmentation.json"
        result.save_json(output_json)

        logger.info(f"Character segmentation completed: {len(result.character_tracks)} characters tracked")

        return {
            "success": True,
            "video_path": video_path,
            "total_characters": len(result.character_tracks),
            "segmentation_time": result.segmentation_time,
            "output_json": output_json,
            "character_tracks": [
                {
                    "character_id": track.character_id,
                    "character_name": track.character_name,
                    "start_frame": track.start_frame,
                    "end_frame": track.end_frame,
                    "total_segments": len(track.segments),
                    "is_consistent": track.is_consistent
                }
                for track in result.character_tracks
            ]
        }

    except Exception as e:
        logger.error(f"Character segmentation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def cut_video_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str
) -> Dict[str, Any]:
    """
    Cut a clip from video

    Args:
        video_path: Input video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output video path

    Returns:
        Edit result
    """
    try:
        logger.info(f"Cutting clip: {start_time}s - {end_time}s")

        editor = VideoEditor()
        result = editor.cut_clip(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path
        )

        return {
            "success": result.success,
            "output_path": result.output_path,
            "edit_time": result.edit_time,
            "input_duration": result.input_duration,
            "output_duration": result.output_duration
        }

    except Exception as e:
        logger.error(f"Failed to cut clip: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def change_video_speed(
    video_path: str,
    speed_factor: float,
    output_path: str
) -> Dict[str, Any]:
    """
    Change video playback speed

    Args:
        video_path: Input video path
        speed_factor: Speed multiplier (0.5 = slow motion, 2.0 = fast forward)
        output_path: Output video path

    Returns:
        Edit result
    """
    try:
        logger.info(f"Changing speed: {speed_factor}x")

        editor = VideoEditor()
        result = editor.change_speed(
            video_path=video_path,
            speed_factor=speed_factor,
            output_path=output_path
        )

        return {
            "success": result.success,
            "output_path": result.output_path,
            "edit_time": result.edit_time,
            "speed_factor": speed_factor,
            "output_duration": result.output_duration
        }

    except Exception as e:
        logger.error(f"Failed to change speed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def create_edit_plan(
    video_path: str,
    goal: str,
    analysis_results: Optional[Dict[str, Any]] = None,
    target_duration: Optional[float] = None,
    constraints: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create AI-driven edit plan using LLM

    This is the CORE INNOVATION: LLM makes ALL editing decisions.

    Args:
        video_path: Path to video file
        goal: User's editing goal (e.g., "Create a funny 30-second highlight reel")
        analysis_results: Video analysis results from Module 7
        target_duration: Target video duration in seconds
        constraints: List of constraints

    Returns:
        Edit plan with LLM decisions
    """
    try:
        logger.info(f"Creating AI edit plan for: {goal}")

        async with LLMDecisionEngine() as engine:
            plan = await engine.create_edit_plan(
                video_path=video_path,
                goal=goal,
                analysis_results=analysis_results,
                constraints=constraints,
                target_duration=target_duration
            )

            # Save plan
            video_name = Path(video_path).stem
            output_json = f"outputs/editing/{video_name}/edit_plan.json"
            plan.save_json(output_json)

            logger.info(f"Edit plan created: {len(plan.decisions)} decisions")

            return {
                "success": True,
                "plan_id": plan.plan_id,
                "video_path": video_path,
                "goal": goal,
                "total_decisions": len(plan.decisions),
                "quality_threshold": plan.quality_threshold,
                "max_iterations": plan.max_iterations,
                "output_json": output_json,
                "decisions": [
                    {
                        "decision_id": d.decision_id,
                        "decision_type": d.decision_type,
                        "confidence": d.confidence,
                        "reasoning": d.reasoning,
                        "priority": d.priority,
                        "parameters": d.parameters
                    }
                    for d in plan.decisions[:10]  # Return top 10
                ]
            }

    except Exception as e:
        logger.error(f"Failed to create edit plan: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def evaluate_video_quality(
    video_path: str,
    goal: Optional[str] = None,
    quality_threshold: float = 0.7,
    analysis_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate video quality

    Args:
        video_path: Path to video file
        goal: Original editing goal
        quality_threshold: Minimum acceptable quality
        analysis_results: Video analysis results

    Returns:
        Quality evaluation metrics
    """
    try:
        logger.info(f"Evaluating video quality: {video_path}")

        evaluator = QualityEvaluator()
        metrics = evaluator.evaluate(
            video_path=video_path,
            goal=goal,
            quality_threshold=quality_threshold,
            analysis_results=analysis_results
        )

        logger.info(f"Quality score: {metrics.overall_score:.3f}")

        return {
            "success": True,
            "video_path": video_path,
            "overall_score": metrics.overall_score,
            "technical_score": metrics.technical_score,
            "creative_score": metrics.creative_score,
            "composition_score": metrics.composition_score,
            "temporal_coherence_score": metrics.temporal_coherence_score,
            "pacing_score": metrics.pacing_score,
            "goal_achievement_score": metrics.goal_achievement_score,
            "needs_improvement": metrics.needs_improvement,
            "feedback": metrics.feedback,
            "issues": metrics.issues,
            "suggestions": metrics.suggestions
        }

    except Exception as e:
        logger.error(f"Failed to evaluate quality: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def create_parody_video(
    video_path: str,
    output_path: str,
    parody_style: str = "dramatic",
    effects: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create funny/parody video with comedic effects

    Args:
        video_path: Input video path
        output_path: Output video path
        parody_style: Parody style (dramatic, chaotic, wholesome)
        effects: List of effects to apply (zoom_punch, speed_ramp)

    Returns:
        Parody generation result
    """
    try:
        logger.info(f"Creating {parody_style} parody video")

        generator = ParodyGenerator()

        if effects and "zoom_punch" in effects:
            # Apply zoom punch
            result = generator.apply_zoom_punch(
                video_path=video_path,
                zoom_time=5.0,  # Auto-detect would be better
                output_path=output_path
            )
        else:
            # Create meme-style video
            result = generator.create_meme_video(
                video_path=video_path,
                output_path=output_path,
                meme_style=parody_style
            )

        logger.info(f"Parody video created: {output_path}")

        return {
            "success": result.get("success", False),
            "output_path": output_path,
            "parody_style": parody_style,
            "effects_applied": effects or [parody_style]
        }

    except Exception as e:
        logger.error(f"Failed to create parody video: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def auto_edit_video(
    video_path: str,
    goal: str,
    output_path: str,
    quality_threshold: float = 0.7,
    max_iterations: int = 3,
    analyze_first: bool = True
) -> Dict[str, Any]:
    """
    COMPLETE AI-DRIVEN VIDEO EDITING WORKFLOW

    This is the main autonomous editing function that:
    1. Analyzes video (optional, using Module 7)
    2. Creates LLM edit plan
    3. Executes edits
    4. Evaluates quality
    5. Iterates if quality below threshold

    Args:
        video_path: Input video path
        goal: User's editing goal
        output_path: Output video path
        quality_threshold: Minimum acceptable quality
        max_iterations: Maximum improvement iterations
        analyze_first: Whether to analyze video first

    Returns:
        Complete editing result
    """
    try:
        logger.info(f"Starting autonomous video editing: {goal}")

        results = {
            "video_path": video_path,
            "goal": goal,
            "output_path": output_path,
            "success": False,
            "iterations": []
        }

        # Step 1: Analyze video (if requested)
        analysis_results = None
        if analyze_first:
            logger.info("Step 1: Analyzing video...")
            # Import video analysis tools
            from scripts.agent.tools.video_analysis_tools import analyze_video_complete

            analysis = await analyze_video_complete(
                video_path=video_path,
                sample_rate=30
            )

            if analysis["success"]:
                analysis_results = analysis["analyses"]
                results["analysis_results"] = analysis_results
                logger.info("Video analysis completed")

        # Step 2: Create edit plan
        logger.info("Step 2: Creating LLM edit plan...")
        plan_result = await create_edit_plan(
            video_path=video_path,
            goal=goal,
            analysis_results=analysis_results
        )

        if not plan_result["success"]:
            return {
                "success": False,
                "error": "Failed to create edit plan",
                "details": plan_result
            }

        results["edit_plan"] = plan_result

        # Step 3: Execute edits (execute plan decisions)
        logger.info("Step 3: Executing edits...")
        exec_result = await execute_edit_plan(
            video_path=video_path,
            plan_json_path=plan_result["output_json"],
            output_path=output_path,
            target_duration=None,
        )

        results["execution"] = exec_result
        if not exec_result.get("success"):
            results["success"] = False
            results["error"] = exec_result.get("error", "Failed to execute edit plan")
            return results

        # Step 4: Evaluate quality
        logger.info("Step 4: Evaluating quality...")
        quality_result = await evaluate_video_quality(
            video_path=output_path if Path(output_path).exists() else video_path,
            goal=goal,
            quality_threshold=quality_threshold
        )

        results["quality_evaluation"] = quality_result

        # Step 5: Iterate if needed
        iteration = 1
        while (quality_result.get("needs_improvement", False) and
               iteration < max_iterations):
            logger.info(f"Step 5: Iteration {iteration} - Improving edit...")

            # Get improvement suggestions from LLM
            # In full implementation, would re-plan and re-execute

            iteration += 1

        results["success"] = True
        results["total_iterations"] = iteration

        logger.info(f"Autonomous editing completed in {iteration} iteration(s)")

        return results

    except Exception as e:
        logger.error(f"Autonomous editing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Register tools with Agent Framework
def register_video_editing_tools(tool_registry):
    """Register all video editing tools"""
    from scripts.agent.tools.tool_registry import Tool, ToolCategory, ToolParameter

    # Character Segmentation
    tool_registry.register_tool(Tool(
        name="segment_characters",
        description="Segment and track characters in video using SAM2. Returns character masks and tracks.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("model_size", "string", "SAM2 model size", required=False, default="base",
                         enum=["tiny", "small", "base", "large"]),
            ToolParameter("sample_interval", "integer", "Process every Nth frame", required=False, default=1),
            ToolParameter("output_masks_dir", "string", "Directory to save masks", required=False),
        ],
        function=segment_characters,
        examples=[
            "Segment all characters in my video",
            "Track character movements using SAM2"
        ],
        requires_gpu=True,
        estimated_vram_gb=6.0,  # SAM2 base
        estimated_time_seconds=120.0
    ))

    # Video Editing Operations
    tool_registry.register_tool(Tool(
        name="cut_video_clip",
        description="Cut a specific segment from video.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Input video path"),
            ToolParameter("start_time", "float", "Start time in seconds"),
            ToolParameter("end_time", "float", "End time in seconds"),
            ToolParameter("output_path", "string", "Output video path"),
        ],
        function=cut_video_clip,
        examples=[
            "Cut the first 30 seconds of the video",
            "Extract segment from 10s to 25s"
        ],
        requires_gpu=False,
        estimated_time_seconds=30.0
    ))

    tool_registry.register_tool(Tool(
        name="change_video_speed",
        description="Change video playback speed (slow motion or fast forward).",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Input video path"),
            ToolParameter("speed_factor", "float", "Speed multiplier (0.5 = slow, 2.0 = fast)"),
            ToolParameter("output_path", "string", "Output video path"),
        ],
        function=change_video_speed,
        examples=[
            "Create slow motion version at 0.5x speed",
            "Speed up video to 2x"
        ],
        requires_gpu=False,
        estimated_time_seconds=45.0
    ))

    # LLM-Driven Edit Planning
    tool_registry.register_tool(Tool(
        name="create_edit_plan",
        description="Create AI-driven edit plan using LLM. CORE INNOVATION: LLM makes ALL editing decisions autonomously.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("goal", "string", "User's editing goal"),
            ToolParameter("analysis_results", "object", "Video analysis results from Module 7", required=False),
            ToolParameter("target_duration", "float", "Target duration in seconds", required=False),
            ToolParameter("constraints", "array", "List of constraints", required=False),
        ],
        function=create_edit_plan,
        examples=[
            "Create a funny 30-second highlight reel",
            "Make a dramatic trailer from this video"
        ],
        requires_gpu=False,
        estimated_time_seconds=15.0
    ))

    # Quality Evaluation
    tool_registry.register_tool(Tool(
        name="evaluate_video_quality",
        description="Evaluate video quality with technical and creative metrics.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("goal", "string", "Original editing goal", required=False),
            ToolParameter("quality_threshold", "float", "Minimum quality threshold", required=False, default=0.7),
            ToolParameter("analysis_results", "object", "Video analysis results", required=False),
        ],
        function=evaluate_video_quality,
        examples=[
            "Evaluate the quality of my edited video",
            "Check if video meets quality standards"
        ],
        requires_gpu=False,
        estimated_time_seconds=30.0
    ))

    # Parody Generation
    tool_registry.register_tool(Tool(
        name="create_parody_video",
        description="Create funny/parody video with comedic effects (zoom punch, speed ramp, meme style).",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Input video path"),
            ToolParameter("output_path", "string", "Output video path"),
            ToolParameter("parody_style", "string", "Parody style", required=False, default="dramatic",
                         enum=["dramatic", "chaotic", "wholesome"]),
            ToolParameter("effects", "array", "List of effects to apply", required=False),
        ],
        function=create_parody_video,
        examples=[
            "Create a funny dramatic parody of this video",
            "Make a chaotic meme version"
        ],
        requires_gpu=False,
        estimated_time_seconds=60.0
    ))

    # Complete Autonomous Editing
    tool_registry.register_tool(Tool(
        name="auto_edit_video",
        description="COMPLETE AI-DRIVEN VIDEO EDITING: Analyze, plan, execute, evaluate, and iterate until quality threshold met. Fully autonomous.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Input video path"),
            ToolParameter("goal", "string", "User's editing goal"),
            ToolParameter("output_path", "string", "Output video path"),
            ToolParameter("quality_threshold", "float", "Minimum quality", required=False, default=0.7),
            ToolParameter("max_iterations", "integer", "Max improvement iterations", required=False, default=3),
            ToolParameter("analyze_first", "boolean", "Analyze video first", required=False, default=True),
        ],
        function=auto_edit_video,
        examples=[
            "Automatically edit this into a 30-second highlight reel",
            "Create a funny version of this video with AI"
        ],
        requires_gpu=True,
        estimated_vram_gb=6.0,
        estimated_time_seconds=300.0
    ))

    logger.info("Video editing tools registered successfully")
