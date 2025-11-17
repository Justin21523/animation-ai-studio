"""
Video Analysis Tools for Agent Framework

Wrapper functions for video analysis modules:
- Scene Detection
- Composition Analysis
- Camera Movement Tracking
- Temporal Coherence Checking

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.analysis.video.scene_detector import SceneDetector
from scripts.analysis.video.composition_analyzer import CompositionAnalyzer
from scripts.analysis.video.camera_movement_tracker import CameraMovementTracker
from scripts.analysis.video.temporal_coherence_checker import TemporalCoherenceChecker


logger = logging.getLogger(__name__)


async def detect_scenes(
    video_path: str,
    extract_keyframes: bool = True,
    keyframe_output_dir: Optional[str] = None,
    threshold: float = 27.0,
    min_scene_length: int = 15
) -> Dict[str, Any]:
    """
    Detect scenes in video using PySceneDetect

    Args:
        video_path: Path to video file
        extract_keyframes: Whether to extract keyframes
        keyframe_output_dir: Directory to save keyframes (auto-generated if None)
        threshold: Scene detection threshold (27.0 is default, lower = more sensitive)
        min_scene_length: Minimum scene length in frames

    Returns:
        Scene detection result as dictionary
    """
    try:
        logger.info(f"Detecting scenes in: {video_path}")

        detector = SceneDetector(
            threshold=threshold,
            min_scene_length=min_scene_length,
            adaptive_threshold=True
        )

        result = detector.detect(
            video_path=video_path,
            extract_keyframes=extract_keyframes,
            keyframe_output_dir=keyframe_output_dir
        )

        # Save to JSON
        video_name = Path(video_path).stem
        output_json = f"outputs/analysis/{video_name}/scene_detection.json"
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_json)

        logger.info(f"Scene detection completed: {result.total_scenes} scenes detected")

        return {
            "success": True,
            "video_path": video_path,
            "total_scenes": result.total_scenes,
            "avg_scene_duration": result.avg_scene_duration,
            "detection_time": result.detection_time,
            "output_json": output_json,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": s.duration,
                    "keyframe_path": s.keyframe_path
                }
                for s in result.scenes
            ]
        }

    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def analyze_composition(
    video_path: str,
    sample_rate: int = 30,
    enable_visualization: bool = False,
    visualization_output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze composition of video frames

    Evaluates:
    - Rule of thirds compliance
    - Visual balance
    - Depth layers
    - Subject positioning

    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame
        enable_visualization: Whether to generate visualization images
        visualization_output_dir: Directory to save visualizations

    Returns:
        Composition analysis result as dictionary
    """
    try:
        logger.info(f"Analyzing composition: {video_path}")

        analyzer = CompositionAnalyzer(
            enable_visualization=enable_visualization
        )

        result = analyzer.analyze_video(
            video_path=video_path,
            sample_rate=sample_rate,
            visualization_output_dir=visualization_output_dir
        )

        # Save to JSON
        video_name = Path(video_path).stem
        output_json = f"outputs/analysis/{video_name}/composition_analysis.json"
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_json)

        logger.info(f"Composition analysis completed: {result.total_frames_analyzed} frames analyzed")

        return {
            "success": True,
            "video_path": video_path,
            "frames_analyzed": result.total_frames_analyzed,
            "avg_composition_score": result.avg_composition_score,
            "best_frame": result.best_composition_frame,
            "worst_frame": result.worst_composition_frame,
            "analysis_time": result.analysis_time,
            "output_json": output_json,
            "summary": {
                "avg_rule_of_thirds": sum(m.rule_of_thirds.overall_score for m in result.frame_metrics) / len(result.frame_metrics) if result.frame_metrics else 0.0,
                "avg_visual_balance": sum(m.visual_balance.overall_balance_score for m in result.frame_metrics) / len(result.frame_metrics) if result.frame_metrics else 0.0,
                "avg_depth_complexity": sum(m.depth_layers.depth_complexity for m in result.frame_metrics) / len(result.frame_metrics) if result.frame_metrics else 0.0
            }
        }

    except Exception as e:
        logger.error(f"Composition analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def track_camera_movement(
    video_path: str,
    sample_interval: int = 1
) -> Dict[str, Any]:
    """
    Track camera movements in video using optical flow

    Detects:
    - Pan (horizontal movement)
    - Tilt (vertical movement)
    - Zoom
    - Camera style (static, smooth, dynamic, handheld)

    Args:
        video_path: Path to video file
        sample_interval: Analyze every Nth frame

    Returns:
        Camera tracking result as dictionary
    """
    try:
        logger.info(f"Tracking camera movements: {video_path}")

        tracker = CameraMovementTracker()

        result = tracker.track_video(
            video_path=video_path,
            sample_interval=sample_interval
        )

        # Save to JSON
        video_name = Path(video_path).stem
        output_json = f"outputs/analysis/{video_name}/camera_tracking.json"
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_json)

        logger.info(f"Camera tracking completed: {result.camera_style} style")

        return {
            "success": True,
            "video_path": video_path,
            "camera_style": result.camera_style,
            "total_movements": len(result.movements),
            "total_shots": len(result.shots),
            "static_duration": result.total_static_duration,
            "moving_duration": result.total_moving_duration,
            "avg_movement_intensity": result.avg_movement_intensity,
            "analysis_time": result.analysis_time,
            "output_json": output_json,
            "shots_summary": [
                {
                    "shot_id": s.shot_id,
                    "duration": s.duration,
                    "dominant_movement": s.dominant_movement,
                    "is_handheld": s.is_handheld,
                    "smoothness": s.smoothness_score
                }
                for s in result.shots[:10]  # First 10 shots
            ]
        }

    except Exception as e:
        logger.error(f"Camera tracking failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def check_temporal_coherence(
    video_path: str,
    sample_interval: int = 1
) -> Dict[str, Any]:
    """
    Check temporal coherence and quality of video

    Analyzes:
    - Color stability across frames
    - Motion smoothness
    - Flicker detection
    - Abrupt transitions
    - Frame-to-frame similarity (SSIM)

    Critical for AI-generated videos.

    Args:
        video_path: Path to video file
        sample_interval: Analyze every Nth frame

    Returns:
        Temporal coherence result as dictionary
    """
    try:
        logger.info(f"Checking temporal coherence: {video_path}")

        checker = TemporalCoherenceChecker()

        result = checker.check_video(
            video_path=video_path,
            sample_interval=sample_interval
        )

        # Save to JSON
        video_name = Path(video_path).stem
        output_json = f"outputs/analysis/{video_name}/temporal_coherence.json"
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_json)

        logger.info(f"Temporal coherence check completed: {result.temporal_stability_rating}")

        return {
            "success": True,
            "video_path": video_path,
            "temporal_stability_rating": result.temporal_stability_rating,
            "avg_coherence_score": result.avg_coherence_score,
            "min_coherence_score": result.min_coherence_score,
            "total_flicker_frames": result.total_flicker_frames,
            "total_abrupt_transitions": result.total_abrupt_transitions,
            "problem_segments": result.problem_segments,
            "analysis_time": result.analysis_time,
            "output_json": output_json,
            "segment_quality_summary": {
                "excellent": sum(1 for s in result.segments if s.quality_rating == "excellent"),
                "good": sum(1 for s in result.segments if s.quality_rating == "good"),
                "fair": sum(1 for s in result.segments if s.quality_rating == "fair"),
                "poor": sum(1 for s in result.segments if s.quality_rating == "poor")
            }
        }

    except Exception as e:
        logger.error(f"Temporal coherence check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def analyze_video_complete(
    video_path: str,
    scene_detection: bool = True,
    composition_analysis: bool = True,
    camera_tracking: bool = True,
    temporal_coherence: bool = True,
    sample_rate: int = 30
) -> Dict[str, Any]:
    """
    Run complete video analysis suite

    Performs all available video analyses:
    - Scene detection
    - Composition analysis
    - Camera movement tracking
    - Temporal coherence checking

    Args:
        video_path: Path to video file
        scene_detection: Enable scene detection
        composition_analysis: Enable composition analysis
        camera_tracking: Enable camera tracking
        temporal_coherence: Enable temporal coherence check
        sample_rate: Sample rate for frame-based analyses

    Returns:
        Combined analysis results
    """
    logger.info(f"Running complete video analysis: {video_path}")

    results = {
        "video_path": video_path,
        "success": True,
        "analyses": {}
    }

    # Scene Detection
    if scene_detection:
        scene_result = await detect_scenes(video_path)
        results["analyses"]["scene_detection"] = scene_result
        if not scene_result["success"]:
            results["success"] = False

    # Composition Analysis
    if composition_analysis:
        comp_result = await analyze_composition(video_path, sample_rate=sample_rate)
        results["analyses"]["composition"] = comp_result
        if not comp_result["success"]:
            results["success"] = False

    # Camera Tracking
    if camera_tracking:
        camera_result = await track_camera_movement(video_path)
        results["analyses"]["camera_tracking"] = camera_result
        if not camera_result["success"]:
            results["success"] = False

    # Temporal Coherence
    if temporal_coherence:
        coherence_result = await check_temporal_coherence(video_path)
        results["analyses"]["temporal_coherence"] = coherence_result
        if not coherence_result["success"]:
            results["success"] = False

    # Generate summary
    if results["success"]:
        summary = _generate_analysis_summary(results["analyses"])
        results["summary"] = summary

        logger.info("Complete video analysis succeeded")
    else:
        logger.warning("Some analyses failed")

    return results


def _generate_analysis_summary(analyses: Dict[str, Any]) -> Dict[str, Any]:
    """Generate human-readable summary of all analyses"""
    summary = {}

    # Scene Detection Summary
    if "scene_detection" in analyses and analyses["scene_detection"]["success"]:
        sd = analyses["scene_detection"]
        summary["scenes"] = {
            "total_scenes": sd["total_scenes"],
            "avg_duration": f"{sd['avg_scene_duration']:.2f}s",
            "description": f"Video contains {sd['total_scenes']} distinct scenes with average duration of {sd['avg_scene_duration']:.2f} seconds."
        }

    # Composition Summary
    if "composition" in analyses and analyses["composition"]["success"]:
        comp = analyses["composition"]
        score = comp["avg_composition_score"]
        if score >= 0.8:
            quality = "excellent"
        elif score >= 0.6:
            quality = "good"
        elif score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"

        summary["composition"] = {
            "quality": quality,
            "score": f"{score:.3f}",
            "description": f"Composition quality is {quality} with average score of {score:.3f}. Best composition at frame {comp['best_frame']}."
        }

    # Camera Tracking Summary
    if "camera_tracking" in analyses and analyses["camera_tracking"]["success"]:
        cam = analyses["camera_tracking"]
        summary["camera"] = {
            "style": cam["camera_style"],
            "shots": cam["total_shots"],
            "description": f"Camera style is {cam['camera_style']} with {cam['total_shots']} distinct shots. {cam['static_duration']:.1f}s static, {cam['moving_duration']:.1f}s moving."
        }

    # Temporal Coherence Summary
    if "temporal_coherence" in analyses and analyses["temporal_coherence"]["success"]:
        tc = analyses["temporal_coherence"]
        summary["temporal_quality"] = {
            "rating": tc["temporal_stability_rating"],
            "score": f"{tc['avg_coherence_score']:.3f}",
            "issues": {
                "flicker_frames": tc["total_flicker_frames"],
                "abrupt_transitions": tc["total_abrupt_transitions"]
            },
            "description": f"Temporal stability is {tc['temporal_stability_rating']} with average coherence {tc['avg_coherence_score']:.3f}. Detected {tc['total_flicker_frames']} flicker frames and {tc['total_abrupt_transitions']} abrupt transitions."
        }

    return summary


# Register tools with Agent Framework
def register_video_analysis_tools(tool_registry):
    """Register all video analysis tools"""
    from scripts.agent.tools.tool_registry import Tool, ToolCategory, ToolParameter

    # Scene Detection
    tool_registry.register_tool(Tool(
        name="detect_scenes",
        description="Detect scenes in video using content-aware analysis. Returns scene boundaries and keyframes.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("extract_keyframes", "boolean", "Whether to extract keyframe images", required=False, default=True),
            ToolParameter("keyframe_output_dir", "string", "Directory to save keyframes", required=False),
            ToolParameter("threshold", "float", "Detection sensitivity (lower = more sensitive)", required=False, default=27.0),
            ToolParameter("min_scene_length", "integer", "Minimum scene length in frames", required=False, default=15),
        ],
        function=detect_scenes,
        examples=[
            "Detect all scenes in the video and extract keyframes",
            "Find scene changes in my animation video"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=30.0
    ))

    # Composition Analysis
    tool_registry.register_tool(Tool(
        name="analyze_composition",
        description="Analyze visual composition of video frames. Evaluates rule of thirds, visual balance, depth layers, and subject positioning.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("sample_rate", "integer", "Analyze every Nth frame", required=False, default=30),
            ToolParameter("enable_visualization", "boolean", "Generate composition visualization images", required=False, default=False),
            ToolParameter("visualization_output_dir", "string", "Directory to save visualizations", required=False),
        ],
        function=analyze_composition,
        examples=[
            "Analyze the composition quality of my video",
            "Check rule of thirds compliance in animation frames"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=45.0
    ))

    # Camera Movement Tracking
    tool_registry.register_tool(Tool(
        name="track_camera_movement",
        description="Track camera movements using optical flow. Detects pan, tilt, zoom, and determines camera style.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("sample_interval", "integer", "Analyze every Nth frame", required=False, default=1),
        ],
        function=track_camera_movement,
        examples=[
            "Track camera movements in my video",
            "Detect if the video has static or dynamic camera work"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=60.0
    ))

    # Temporal Coherence Check
    tool_registry.register_tool(Tool(
        name="check_temporal_coherence",
        description="Check temporal coherence and quality. Analyzes color stability, motion smoothness, flicker, and frame-to-frame consistency. Critical for AI-generated videos.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("sample_interval", "integer", "Analyze every Nth frame", required=False, default=1),
        ],
        function=check_temporal_coherence,
        examples=[
            "Check if my AI-generated video has temporal artifacts",
            "Detect flicker and abrupt transitions in the video"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=90.0
    ))

    # Complete Video Analysis
    tool_registry.register_tool(Tool(
        name="analyze_video_complete",
        description="Run complete video analysis suite with all available analyses. Provides comprehensive video quality and content assessment.",
        category=ToolCategory.VIDEO_ANALYSIS,
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter("scene_detection", "boolean", "Enable scene detection", required=False, default=True),
            ToolParameter("composition_analysis", "boolean", "Enable composition analysis", required=False, default=True),
            ToolParameter("camera_tracking", "boolean", "Enable camera tracking", required=False, default=True),
            ToolParameter("temporal_coherence", "boolean", "Enable temporal coherence check", required=False, default=True),
            ToolParameter("sample_rate", "integer", "Sample rate for frame-based analyses", required=False, default=30),
        ],
        function=analyze_video_complete,
        examples=[
            "Run full analysis on my animation video",
            "Comprehensively analyze video quality and content"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=180.0
    ))

    logger.info("Video analysis tools registered successfully")
