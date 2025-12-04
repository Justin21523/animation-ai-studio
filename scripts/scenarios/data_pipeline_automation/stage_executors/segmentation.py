"""
Data Pipeline Automation - SAM2 Segmentation Executor

Wraps instance_segmentation.py for pipeline execution.

Author: Animation AI Studio
Date: 2025-12-04
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from .script_executor import ScriptExecutor
from ..common import StageResult, ExecutionStatus


class SegmentationExecutor(ScriptExecutor):
    """
    SAM2-based instance segmentation stage executor

    Wraps instance_segmentation.py to extract character instances
    from frames using SAM2 automatic mask generation.

    Required config keys:
        - input_dir: Directory containing input frames

    Optional config keys:
        - output_dir: Output directory for instances (required if project not specified)
        - project: Project/film name (auto-constructs paths)
        - model: SAM2 model type (default: sam2_hiera_base)
            - sam2_hiera_large: Highest quality, most VRAM (~7GB)
            - sam2_hiera_base: Balanced quality/VRAM (~6GB) [RECOMMENDED]
            - sam2_hiera_small: Faster, less VRAM (~4GB)
            - sam2_hiera_tiny: Fastest, minimal VRAM (~3GB)
        - device: Device to use (default: cuda)
        - min_size: Minimum instance area in pixels (default: 16384 = 128x128)
        - visualize: Save visualization of detected instances (default: false)
        - save_masks: Save instance masks as grayscale PNG (default: false)
        - save_backgrounds: Save inpainted backgrounds (default: false)
        - context_mode: Background handling (default: all)
            - transparent: Alpha channel only
            - context: Keep original background
            - blurred: Blur background
            - all: Generate all three versions
        - context_padding: Padding around bbox for context (default: 20)

    Outputs:
        - output_dir: Directory containing extracted instances
        - character_dir: Main instances directory (transparent)
        - instance_count: Total number of instances extracted
        - frames_processed: Number of frames successfully processed
        - avg_instances_per_frame: Average instances detected per frame

    Metrics:
        - instances_extracted: Total instances extracted
        - frames_processed: Frames successfully segmented
        - segmentation_time: Total execution time in seconds
        - instances_per_second: Processing speed
        - frames_with_multiple_chars: Frames containing multiple characters
    """

    # Script configuration
    script_path = "scripts/processing/segmentation/instance_segmentation.py"
    required_config_keys = ["input_dir"]
    output_keys = ["output_dir", "character_dir", "instance_count", "frames_processed", "avg_instances_per_frame"]

    def validate_config(self) -> bool:
        """
        Validate segmentation configuration

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

        # Check for image files in input directory
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_count = sum(
            len(list(input_dir.glob(f'**/*{ext}'))) + len(list(input_dir.glob(f'**/*{ext.upper()}')))
            for ext in image_extensions
        )
        if image_count == 0:
            raise ValueError(f"No image files found in input directory: {input_dir}")

        # Validate output_dir or project is specified
        output_dir = self._get_config_value("output_dir")
        project = self._get_config_value("project")
        if not output_dir and not project:
            raise ValueError("Either 'output_dir' or 'project' must be specified")

        # Validate model type
        model = self._get_config_value("model", default="sam2_hiera_base")
        valid_models = ["sam2_hiera_large", "sam2_hiera_base", "sam2_hiera_small", "sam2_hiera_tiny"]
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")

        # Validate device
        device = self._get_config_value("device", default="cuda")
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got '{device}'")

        # Validate min_size
        min_size = self._get_config_value("min_size", default=128 * 128)
        if min_size < 64 * 64:
            raise ValueError(f"Minimum instance size too small (< 4096 pixels), got {min_size}")

        # Validate context_mode
        context_mode = self._get_config_value("context_mode", default="all")
        valid_modes = ["transparent", "context", "blurred", "all"]
        if context_mode not in valid_modes:
            raise ValueError(f"Invalid context_mode '{context_mode}'. Must be one of: {valid_modes}")

        # Validate context_padding
        context_padding = self._get_config_value("context_padding", default=20)
        if context_padding < 0:
            raise ValueError(f"Context padding must be >= 0, got {context_padding}")

        return True

    def _build_command(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Build command to execute SAM2 segmentation script

        Args:
            inputs: Input data from previous stages
                   Can contain 'output_dir' from frame extraction stage

        Returns:
            Command line arguments list
        """
        # Start with base command
        command = ["python", self.script_path]

        # Add required positional argument (input directory)
        # Check if input_dir comes from previous stage output
        input_dir = self.config.get("input_dir")
        if input_dir and isinstance(input_dir, str) and "{" in input_dir:
            # Template string like "{extract_frames.output_dir}"
            # Should be resolved by orchestrator before execution
            # For now, treat as literal if not yet resolved
            from ..common import parse_stage_outputs
            input_dir = parse_stage_outputs(input_dir, inputs)
        else:
            input_dir = self._get_config_value("input_dir", required=True)

        command.append(str(input_dir))

        # Add output directory (if specified)
        output_dir = self._get_config_value("output_dir")
        if output_dir:
            command.extend(["--output-dir", str(output_dir)])

        # Add project name (if specified)
        project = self._get_config_value("project")
        if project:
            command.extend(["--project", str(project)])

        # Add model type
        model = self._get_config_value("model", default="sam2_hiera_base")
        command.extend(["--model", model])

        # Add device
        device = self._get_config_value("device", default="cuda")
        command.extend(["--device", device])

        # Add minimum instance size
        min_size = self._get_config_value("min_size", default=128 * 128)
        command.extend(["--min-size", str(min_size)])

        # Add context mode
        context_mode = self._get_config_value("context_mode", default="all")
        command.extend(["--context-mode", context_mode])

        # Add context padding
        context_padding = self._get_config_value("context_padding", default=20)
        command.extend(["--context-padding", str(context_padding)])

        # Add boolean flags
        if self._get_config_value("visualize", default=False):
            command.append("--visualize")

        if self._get_config_value("save_masks", default=False):
            command.append("--save-masks")

        if self._get_config_value("save_backgrounds", default=False):
            command.append("--save-backgrounds")

        return command

    def _parse_outputs(self) -> Dict[str, Any]:
        """
        Parse segmentation outputs

        Returns:
            Dictionary containing:
                - output_dir: Root output directory
                - character_dir: Main instances directory (transparent)
                - instance_count: Total instances extracted
                - frames_processed: Frames successfully processed
                - avg_instances_per_frame: Average instances per frame

        Raises:
            RuntimeError: If output parsing fails
        """
        outputs = {}

        # Determine output directory
        output_dir = self._get_config_value("output_dir")
        project = self._get_config_value("project")

        if project and not output_dir:
            # Auto-constructed path
            base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
            output_path = base_dir / project / "instances"
        elif output_dir:
            output_path = Path(output_dir)
        else:
            raise RuntimeError("Cannot determine output directory")

        # Validate output directory exists
        if not output_path.exists():
            raise RuntimeError(f"Output directory not found: {output_path}")

        outputs["output_dir"] = str(output_path)

        # Determine character directory based on context_mode
        context_mode = self._get_config_value("context_mode", default="all")
        if context_mode == "all" or context_mode == "transparent":
            character_dir = output_path / "instances"
        elif context_mode == "context":
            character_dir = output_path / "instances_context"
        elif context_mode == "blurred":
            character_dir = output_path / "instances_blurred"
        else:
            character_dir = output_path / "instances"

        outputs["character_dir"] = str(character_dir)

        # Load instances_metadata.json
        metadata_file = output_path / "instances_metadata.json"
        if metadata_file.exists():
            try:
                metadata = self._load_json_file(metadata_file)

                # Extract statistics
                stats = metadata.get("statistics", {})
                outputs["instance_count"] = stats.get("total_instances", 0)
                outputs["frames_processed"] = stats.get("frames_processed", 0)

                # Calculate average instances per frame
                instances_per_frame = stats.get("instances_per_frame", [])
                if instances_per_frame:
                    avg_instances = sum(instances_per_frame) / len(instances_per_frame)
                    outputs["avg_instances_per_frame"] = round(avg_instances, 2)
                else:
                    outputs["avg_instances_per_frame"] = 0.0

                # Store additional stats for metrics
                outputs["_stats"] = stats  # Store for _extract_metrics

            except Exception as e:
                raise RuntimeError(f"Failed to parse instances_metadata.json: {e}")
        else:
            # Fallback: count instance files manually
            instance_count = self._count_files(character_dir, "*.png")
            outputs["instance_count"] = instance_count
            outputs["frames_processed"] = instance_count  # Rough estimate
            outputs["avg_instances_per_frame"] = 0.0
            outputs["_stats"] = {}

        return outputs

    def _extract_metrics(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract performance metrics

        Args:
            outputs: Parsed outputs from _parse_outputs()

        Returns:
            Dictionary containing:
                - instances_extracted: Total instances
                - frames_processed: Frames successfully segmented
                - segmentation_time: Total execution time
                - instances_per_second: Processing speed
                - frames_with_multiple_chars: Frames with multiple characters
        """
        metrics = {}

        # Instance count metric
        instance_count = outputs.get("instance_count", 0)
        metrics["instances_extracted"] = float(instance_count)

        # Frames processed
        frames_processed = outputs.get("frames_processed", 0)
        metrics["frames_processed"] = float(frames_processed)

        # Extract from stored stats
        stats = outputs.get("_stats", {})
        frames_with_multiple = stats.get("frames_with_multiple", 0)
        metrics["frames_with_multiple_chars"] = float(frames_with_multiple)

        # Calculate segmentation time
        if self.start_time and self.end_time:
            segmentation_time = self.end_time - self.start_time
            metrics["segmentation_time"] = segmentation_time

            # Calculate processing speed
            if segmentation_time > 0:
                metrics["instances_per_second"] = instance_count / segmentation_time
            else:
                metrics["instances_per_second"] = 0.0
        else:
            metrics["segmentation_time"] = 0.0
            metrics["instances_per_second"] = 0.0

        return metrics

    def estimate_duration(self) -> float:
        """
        Estimate segmentation duration

        Returns:
            Estimated duration in seconds

        Notes:
            SAM2 segmentation is GPU-intensive:
            - sam2_hiera_tiny: ~0.5-1 sec/frame
            - sam2_hiera_small: ~1-2 sec/frame
            - sam2_hiera_base: ~2-3 sec/frame
            - sam2_hiera_large: ~3-5 sec/frame
            Default: 60 minutes for safety
        """
        model = self._get_config_value("model", default="sam2_hiera_base")
        device = self._get_config_value("device", default="cuda")

        # Time per frame estimates (seconds)
        time_per_frame = {
            "sam2_hiera_tiny": 0.75 if device == "cuda" else 5.0,
            "sam2_hiera_small": 1.5 if device == "cuda" else 10.0,
            "sam2_hiera_base": 2.5 if device == "cuda" else 15.0,
            "sam2_hiera_large": 4.0 if device == "cuda" else 20.0,
        }

        base_duration = time_per_frame.get(model, 2.5)

        # Try to count images in input directory
        try:
            input_dir = Path(self._get_config_value("input_dir", required=True))
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_count = sum(
                len(list(input_dir.glob(f'**/*{ext}'))) + len(list(input_dir.glob(f'**/*{ext.upper()}')))
                for ext in image_extensions
            )

            if image_count > 0:
                # Estimate total duration
                estimated = base_duration * image_count
                # Add 20% overhead for initialization and I/O
                return estimated * 1.2
        except Exception:
            pass

        # Default: 60 minutes
        return 3600.0
