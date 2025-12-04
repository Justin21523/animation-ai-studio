"""
Data Pipeline Automation - Base Stage Executor

Abstract base class for all stage executors.

Author: Animation AI Studio
Date: 2025-12-03
"""

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import StageResult, ExecutionStatus

logger = logging.getLogger(__name__)


class StageExecutor(ABC):
    """
    Abstract base class for stage executors

    All stage executors must implement:
    - validate_config(): Validate stage configuration
    - execute(): Execute the stage
    - estimate_duration(): Estimate execution duration
    """

    def __init__(self, stage_id: str, config: Dict[str, Any]):
        """
        Initialize stage executor

        Args:
            stage_id: Unique stage identifier
            config: Stage configuration
        """
        self.stage_id = stage_id
        self.config = config
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        logger.debug(f"Initialized executor for stage '{stage_id}'")

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate stage configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        """
        Execute the stage

        Args:
            inputs: Input data from previous stages

        Returns:
            Stage execution result

        Raises:
            RuntimeError: If execution fails
        """
        pass

    @abstractmethod
    def estimate_duration(self) -> float:
        """
        Estimate stage execution duration

        Returns:
            Estimated duration in seconds
        """
        pass

    def cleanup(self):
        """
        Cleanup resources after execution

        Override this method if cleanup is needed.
        """
        pass

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_config_value(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value with optional default

        Args:
            key: Configuration key
            default: Default value if key not found
            required: Whether key is required

        Returns:
            Configuration value

        Raises:
            ValueError: If required key is missing
        """
        if key not in self.config:
            if required:
                raise ValueError(f"Required config key '{key}' missing for stage '{self.stage_id}'")
            return default

        return self.config[key]

    def _validate_path_exists(self, path: Path, path_type: str = "path") -> bool:
        """
        Validate path exists

        Args:
            path: Path to validate
            path_type: Type of path (for error message)

        Returns:
            True if path exists

        Raises:
            ValueError: If path does not exist
        """
        if not path.exists():
            raise ValueError(f"{path_type} does not exist: {path}")
        return True

    def _run_subprocess(self,
                       command: List[str],
                       timeout: Optional[int] = None,
                       capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run subprocess command

        Args:
            command: Command to run as list of strings
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            Completed process

        Raises:
            RuntimeError: If command fails
        """
        try:
            logger.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=True
            )
            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed: {e.stderr if e.stderr else str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except subprocess.TimeoutExpired as e:
            error_msg = f"Command timed out after {timeout}s"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _load_json_file(self, json_path: Path) -> Dict[str, Any]:
        """
        Load JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            RuntimeError: If JSON loading fails
        """
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON from {json_path}: {e}")

    def _count_files(self, directory: Path, pattern: str = "*") -> int:
        """
        Count files in directory

        Args:
            directory: Directory to count files in
            pattern: Glob pattern for files

        Returns:
            Number of files
        """
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))

    def _create_success_result(self,
                               outputs: Dict[str, Any],
                               metrics: Optional[Dict[str, float]] = None) -> StageResult:
        """
        Create successful stage result

        Args:
            outputs: Stage outputs
            metrics: Performance metrics

        Returns:
            Stage result
        """
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        return StageResult(
            stage_id=self.stage_id,
            status=ExecutionStatus.SUCCESS,
            duration=duration,
            outputs=outputs,
            metrics=metrics or {},
            error_message=None
        )

    def _create_failure_result(self, error_message: str) -> StageResult:
        """
        Create failed stage result

        Args:
            error_message: Error message

        Returns:
            Stage result
        """
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        return StageResult(
            stage_id=self.stage_id,
            status=ExecutionStatus.FAILED,
            duration=duration,
            outputs={},
            metrics={},
            error_message=error_message
        )

    def _mark_started(self):
        """Mark stage as started"""
        self.start_time = time.time()
        logger.info(f"Stage '{self.stage_id}' started")

    def _mark_completed(self):
        """Mark stage as completed"""
        self.end_time = time.time()
        duration = self.end_time - (self.start_time or self.end_time)
        logger.info(f"Stage '{self.stage_id}' completed in {duration:.2f}s")
