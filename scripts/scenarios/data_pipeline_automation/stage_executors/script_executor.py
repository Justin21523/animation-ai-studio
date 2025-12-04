"""
Data Pipeline Automation - Generic Script Executor

Base class for executors that wrap Python scripts.

Author: Animation AI Studio
Date: 2025-12-03
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_executor import StageExecutor
from ..common import StageResult, ExecutionStatus


class ScriptExecutor(StageExecutor):
    """
    Generic executor for wrapping Python scripts

    Subclasses only need to define:
    - script_path: Path to the Python script
    - required_config_keys: List of required configuration keys
    - output_keys: Expected output keys from the script
    """

    # To be overridden by subclasses
    script_path: str = None
    required_config_keys: List[str] = []
    output_keys: List[str] = []

    def validate_config(self) -> bool:
        """Validate configuration has required keys"""
        for key in self.required_config_keys:
            self._get_config_value(key, required=True)
        return True

    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        """Execute the wrapped script"""
        self._mark_started()

        try:
            # Build command
            command = self._build_command(inputs)

            # Execute
            result = self._run_subprocess(command, timeout=3600)

            # Parse outputs
            outputs = self._parse_outputs()

            # Extract metrics
            metrics = self._extract_metrics(outputs)

            self._mark_completed()
            return self._create_success_result(outputs, metrics)

        except Exception as e:
            self._mark_completed()
            return self._create_failure_result(str(e))

    def _build_command(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Build command to execute script

        Override this in subclasses for custom argument building
        """
        command = ["python", self.script_path]

        # Add config arguments
        for key, value in self.config.items():
            if key.startswith('_'):  # Skip internal keys
                continue

            # Convert to command line argument
            arg_name = f"--{key.replace('_', '-')}"
            command.extend([arg_name, str(value)])

        return command

    def _parse_outputs(self) -> Dict[str, Any]:
        """
        Parse script outputs

        Override this in subclasses for custom output parsing
        """
        outputs = {}

        # Try to find output JSON file
        output_dir = Path(self._get_config_value("output_dir", required=True))

        for key in self.output_keys:
            # Check if output exists
            if output_dir.exists():
                outputs[key] = str(output_dir)

        return outputs

    def _extract_metrics(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract metrics from outputs

        Override this in subclasses for custom metrics
        """
        metrics = {}

        # Count files in output directory
        output_dir = self._get_config_value("output_dir")
        if output_dir:
            output_path = Path(output_dir)
            if output_path.exists():
                metrics['output_files'] = float(self._count_files(output_path))

        return metrics

    def estimate_duration(self) -> float:
        """
        Estimate execution duration

        Override this in subclasses for more accurate estimates
        """
        # Default: 5 minutes
        return 300.0
