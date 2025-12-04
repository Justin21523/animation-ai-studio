"""
Data Pipeline Automation - Frame Extraction Executor

Wraps universal_frame_extractor.py for pipeline execution.

Author: Animation AI Studio
Date: 2025-12-04
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from .script_executor import ScriptExecutor
from ..common import StageResult, ExecutionStatus


class FrameExtractionExecutor(ScriptExecutor):
    """
    Frame extraction stage executor

    Wraps universal_frame_extractor.py to extract frames from videos
    using scene-based, interval-based, or hybrid strategies.

    Required config keys:
        - input_dir: Directory containing video files
        - mode: Extraction mode (scene, interval, hybrid)

    Optional config keys:
        - output_dir: Output directory for frames (default: input_dir/extracted_frames)
        - temp_dir: Temporary directory (default: input_dir/temp)
        - scene_threshold: Scene detection sensitivity (default: 27.0)
        - frames_per_scene: Frames to extract per scene (default: 3)
        - skip_scene_boundaries: Whether to skip scene boundaries (default: true)
        - interval_seconds: Extract one frame every N seconds
        - interval_frames: Extract one frame every N frames
        - jpeg_quality: JPEG quality 1-100 (default: 95)
        - workers: Number of parallel workers (default: auto)
        - episode_pattern: Regex pattern for episode numbers (default: r'(\d+)')
        - start_episode: Start episode number (optional)
        - end_episode: End episode number (optional)

    Outputs:
        - output_dir: Directory containing extracted frames
        - frame_count: Total number of frames extracted
        - fps: Average FPS of processed videos
        - total_episodes: Total episodes processed
        - successful_episodes: Successfully processed episodes

    Metrics:
        - frames_extracted: Total frames extracted
        - extraction_time: Total execution time in seconds
        - frames_per_second: Processing speed (frames/sec)
        - episodes_processed: Number of episodes successfully processed
    """

    # Script configuration
    script_path = "scripts/processing/extraction/universal_frame_extractor.py"
    required_config_keys = ["input_dir", "mode"]
    output_keys = ["output_dir", "frame_count", "fps", "total_episodes", "successful_episodes"]

    def validate_config(self) -> bool:
        """
        Validate frame extraction configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation
        super().validate_config()

        # Validate input directory exists
        input_dir = Path(self._get_config_value("input_dir", required=True))
        self._validate_path_exists(input_dir, "Input directory")

        # Validate mode
        mode = self._get_config_value("mode", required=True)
        valid_modes = ["scene", "interval", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Validate scene threshold (if scene-based or hybrid)
        if mode in ["scene", "hybrid"]:
            scene_threshold = self._get_config_value("scene_threshold", default=27.0)
            if not (0.0 < scene_threshold < 100.0):
                raise ValueError(f"Scene threshold must be between 0 and 100, got {scene_threshold}")

            frames_per_scene = self._get_config_value("frames_per_scene", default=3)
            if frames_per_scene < 1:
                raise ValueError(f"Frames per scene must be >= 1, got {frames_per_scene}")

        # Validate interval parameters (if interval-based or hybrid)
        if mode in ["interval", "hybrid"]:
            interval_seconds = self._get_config_value("interval_seconds")
            interval_frames = self._get_config_value("interval_frames")

            if interval_seconds is not None and interval_seconds <= 0:
                raise ValueError(f"Interval seconds must be > 0, got {interval_seconds}")

            if interval_frames is not None and interval_frames <= 0:
                raise ValueError(f"Interval frames must be > 0, got {interval_frames}")

        # Validate JPEG quality
        jpeg_quality = self._get_config_value("jpeg_quality", default=95)
        if not (1 <= jpeg_quality <= 100):
            raise ValueError(f"JPEG quality must be between 1 and 100, got {jpeg_quality}")

        return True

    def _build_command(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Build command to execute frame extraction script

        Args:
            inputs: Input data from previous stages (not used for frame extraction)

        Returns:
            Command line arguments list
        """
        # Start with base command
        command = ["python", self.script_path]

        # Add required positional argument (input directory)
        input_dir = self._get_config_value("input_dir", required=True)
        command.append(str(input_dir))

        # Add mode
        mode = self._get_config_value("mode", required=True)
        command.extend(["--mode", mode])

        # Add output directory
        output_dir = self._get_config_value("output_dir")
        if output_dir:
            command.extend(["--output-dir", str(output_dir)])

        # Add temp directory
        temp_dir = self._get_config_value("temp_dir")
        if temp_dir:
            command.extend(["--temp-dir", str(temp_dir)])

        # Add scene-based parameters
        if mode in ["scene", "hybrid"]:
            scene_threshold = self._get_config_value("scene_threshold", default=27.0)
            command.extend(["--scene-threshold", str(scene_threshold)])

            frames_per_scene = self._get_config_value("frames_per_scene", default=3)
            command.extend(["--frames-per-scene", str(frames_per_scene)])

            # Handle skip_scene_boundaries (inverse flag)
            skip_boundaries = self._get_config_value("skip_scene_boundaries", default=True)
            if not skip_boundaries:
                command.append("--no-skip-boundaries")

        # Add interval-based parameters
        if mode in ["interval", "hybrid"]:
            interval_seconds = self._get_config_value("interval_seconds")
            if interval_seconds is not None:
                command.extend(["--interval-seconds", str(interval_seconds)])

            interval_frames = self._get_config_value("interval_frames")
            if interval_frames is not None:
                command.extend(["--interval-frames", str(interval_frames)])

        # Add JPEG quality
        jpeg_quality = self._get_config_value("jpeg_quality", default=95)
        command.extend(["--jpeg-quality", str(jpeg_quality)])

        # Add workers
        workers = self._get_config_value("workers")
        if workers is not None:
            command.extend(["--workers", str(workers)])

        # Add episode pattern
        episode_pattern = self._get_config_value("episode_pattern")
        if episode_pattern is not None:
            command.extend(["--episode-pattern", str(episode_pattern)])

        # Add episode range
        start_episode = self._get_config_value("start_episode")
        if start_episode is not None:
            command.extend(["--start", str(start_episode)])

        end_episode = self._get_config_value("end_episode")
        if end_episode is not None:
            command.extend(["--end", str(end_episode)])

        return command

    def _parse_outputs(self) -> Dict[str, Any]:
        """
        Parse frame extraction outputs

        Returns:
            Dictionary containing:
                - output_dir: Path to extracted frames
                - frame_count: Total frames extracted
                - fps: Average FPS
                - total_episodes: Total episodes
                - successful_episodes: Successfully processed episodes

        Raises:
            RuntimeError: If output parsing fails
        """
        outputs = {}

        # Determine output directory
        output_dir = self._get_config_value("output_dir")
        if output_dir:
            output_path = Path(output_dir)
        else:
            input_dir = Path(self._get_config_value("input_dir", required=True))
            output_path = input_dir / "extracted_frames"

        # Validate output directory exists
        if not output_path.exists():
            raise RuntimeError(f"Output directory not found: {output_path}")

        outputs["output_dir"] = str(output_path)

        # Load extraction_results.json
        results_file = output_path / "extraction_results.json"
        if results_file.exists():
            try:
                results_data = self._load_json_file(results_file)
                outputs["frame_count"] = results_data.get("total_frames", 0)
                outputs["total_episodes"] = results_data.get("total_episodes", 0)
                outputs["successful_episodes"] = results_data.get("successful_episodes", 0)

                # Calculate average FPS (if available in config)
                if "config" in results_data:
                    outputs["fps"] = 24.0  # Default FPS placeholder
                else:
                    outputs["fps"] = 24.0

            except Exception as e:
                raise RuntimeError(f"Failed to parse extraction_results.json: {e}")
        else:
            # Fallback: count frames manually
            frame_count = self._count_files(output_path, "**/*.jpg")
            outputs["frame_count"] = frame_count
            outputs["total_episodes"] = 1
            outputs["successful_episodes"] = 1 if frame_count > 0 else 0
            outputs["fps"] = 24.0

        return outputs

    def _extract_metrics(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract performance metrics

        Args:
            outputs: Parsed outputs from _parse_outputs()

        Returns:
            Dictionary containing:
                - frames_extracted: Total frames extracted
                - extraction_time: Total execution time in seconds
                - frames_per_second: Processing speed
                - episodes_processed: Successfully processed episodes
        """
        metrics = {}

        # Frame count metric
        frame_count = outputs.get("frame_count", 0)
        metrics["frames_extracted"] = float(frame_count)

        # Episodes processed
        successful = outputs.get("successful_episodes", 0)
        metrics["episodes_processed"] = float(successful)

        # Calculate extraction time
        if self.start_time and self.end_time:
            extraction_time = self.end_time - self.start_time
            metrics["extraction_time"] = extraction_time

            # Calculate processing speed
            if extraction_time > 0:
                metrics["frames_per_second"] = frame_count / extraction_time
            else:
                metrics["frames_per_second"] = 0.0
        else:
            metrics["extraction_time"] = 0.0
            metrics["frames_per_second"] = 0.0

        return metrics

    def estimate_duration(self) -> float:
        """
        Estimate frame extraction duration

        Returns:
            Estimated duration in seconds

        Notes:
            Estimation based on:
            - Scene mode: ~30-60 seconds per video
            - Interval mode: ~20-40 seconds per video
            - Hybrid mode: ~50-90 seconds per video
            - Default: 5 minutes for safety
        """
        mode = self._get_config_value("mode", default="scene")

        # Rough estimates per video
        estimates = {
            "scene": 45.0,      # ~45 seconds per video
            "interval": 30.0,   # ~30 seconds per video
            "hybrid": 70.0      # ~70 seconds per video (both modes)
        }

        base_duration = estimates.get(mode, 45.0)

        # Try to count videos in input directory
        try:
            input_dir = Path(self._get_config_value("input_dir", required=True))
            video_extensions = ['.mp4', '.mkv', '.avi', '.flv', '.mov', '.wmv']
            video_count = sum(
                len(list(input_dir.glob(f'*{ext}'))) + len(list(input_dir.glob(f'*{ext.upper()}')))
                for ext in video_extensions
            )

            if video_count > 0:
                # Estimate total duration
                return base_duration * video_count
        except Exception:
            pass

        # Default: 5 minutes
        return 300.0
