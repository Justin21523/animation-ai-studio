"""
Scenario Automation Adapter

Adapter for integrating automation scenario scripts with the orchestration layer.
Wraps 11 existing scenario scripts to provide standardized Task/TaskResult interface.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.orchestration.module_registry import (
    ModuleAdapter,
    ModuleType,
    ModuleStatus,
    ModuleCapabilities,
    HealthCheckResult,
    Task,
    TaskResult
)

logger = logging.getLogger(__name__)


# Scenario script mappings
SCENARIO_SCRIPTS = {
    "video_processor": "scripts/automation/scenarios/video_processor.py",
    "audio_processor": "scripts/automation/scenarios/audio_processor.py",
    "image_processor": "scripts/automation/scenarios/image_processor.py",
    "subtitle_automation": "scripts/automation/scenarios/subtitle_automation.py",
    "file_organizer": "scripts/automation/scenarios/file_organizer.py",
    "dataset_builder": "scripts/automation/scenarios/dataset_builder.py",
    "data_augmentation": "scripts/automation/scenarios/data_augmentation.py",
    "annotation_tool": "scripts/automation/scenarios/annotation_tool.py",
    "auto_categorizer": "scripts/automation/scenarios/auto_categorizer.py",
    "knowledge_base_builder": "scripts/automation/scenarios/knowledge_base_builder.py",
    "media_asset_analyzer": "scripts/automation/scenarios/media_asset_analyzer.py"
}


class ScenarioAdapter(ModuleAdapter):
    """
    Adapter for Automation Scenario Scripts

    Wraps 11 existing scenario scripts to provide standardized interface for:
    - Video processing (cut, concat, convert, effects)
    - Audio processing (extract, normalize, mix)
    - Image processing (resize, filter, convert)
    - Subtitle automation (transcribe, translate, sync)
    - File organization (categorize, deduplicate, archive)
    - Dataset building (prepare, validate, split)
    - Data augmentation (transform, generate, balance)
    - Annotation tools (label, review, export)
    - Auto categorization (classify, tag, index)
    - Knowledge base building (index, embed, search)
    - Media analysis (metadata, quality, content)

    Task Types Supported:
    - "video": Video processing operations
    - "audio": Audio processing operations
    - "image": Image processing operations
    - "subtitle": Subtitle automation operations
    - "file": File organization operations
    - "dataset": Dataset building operations
    - "augment": Data augmentation operations
    - "annotate": Annotation operations
    - "categorize": Auto categorization operations
    - "knowledge": Knowledge base operations
    - "analyze": Media analysis operations

    Task Parameters:
    - scenario (str): Scenario name (video_processor, audio_processor, etc.)
    - operation (str): Operation to perform (cut, extract, resize, etc.)
    - Additional parameters specific to each scenario (converted to CLI args)

    Example:
        adapter = ScenarioAdapter()
        await adapter.initialize()

        # Video processing task
        task = Task(
            task_id="1",
            task_type="video",
            parameters={
                "scenario": "video_processor",
                "operation": "cut",
                "input": "/path/to/video.mp4",
                "output": "/path/to/output.mp4",
                "start_time": "00:01:30",
                "end_time": "00:05:45"
            }
        )

        result = await adapter.execute(task)
    """

    def __init__(self, python_exe: str = "python"):
        """
        Initialize Scenario Adapter

        Args:
            python_exe: Python executable to use for running scripts
        """
        super().__init__(
            module_name="scenario",
            module_type=ModuleType.SCENARIO
        )

        self.python_exe = python_exe
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_time = 0.0
        self.scenario_counts = {scenario: 0 for scenario in SCENARIO_SCRIPTS.keys()}

        logger.info("ScenarioAdapter created")

    async def initialize(self) -> bool:
        """
        Initialize Scenario Adapter

        Verifies all scenario scripts exist.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Scenario Adapter...")

            # Verify all scripts exist
            missing_scripts = []
            for scenario, script_path in SCENARIO_SCRIPTS.items():
                full_path = self.project_root / script_path
                if not full_path.exists():
                    missing_scripts.append(f"{scenario} ({script_path})")

            if missing_scripts:
                logger.warning(
                    f"Missing {len(missing_scripts)} scenario scripts: "
                    f"{', '.join(missing_scripts)}"
                )
            else:
                logger.info(f"All {len(SCENARIO_SCRIPTS)} scenario scripts found")

            self._initialized = True
            logger.info("Scenario Adapter initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Scenario Adapter: {e}", exc_info=True)
            return False

    async def execute(self, task: Task) -> TaskResult:
        """
        Execute task using scenario scripts

        Args:
            task: Task to execute

        Returns:
            TaskResult with scenario output
        """
        if not self._initialized:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error="Scenario Adapter not initialized"
            )

        start_time = time.time()
        self.total_tasks += 1

        try:
            # Extract scenario name
            scenario = task.parameters.get("scenario")
            if not scenario:
                raise ValueError("Missing required parameter: scenario")

            if scenario not in SCENARIO_SCRIPTS:
                raise ValueError(f"Unknown scenario: {scenario}")

            # Track scenario usage
            self.scenario_counts[scenario] += 1

            # Get script path
            script_path = self.project_root / SCENARIO_SCRIPTS[scenario]
            if not script_path.exists():
                raise FileNotFoundError(f"Scenario script not found: {script_path}")

            # Build command line arguments
            cmd = [self.python_exe, str(script_path)]

            # Convert task parameters to CLI arguments
            for key, value in task.parameters.items():
                if key == "scenario":
                    continue  # Skip scenario itself

                # Convert parameter name to CLI flag format
                flag = f"--{key.replace('_', '-')}"

                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        cmd.extend([flag, str(item)])
                else:
                    cmd.extend([flag, str(value)])

            # Execute scenario script
            logger.info(f"Executing {scenario}: {' '.join(cmd[:5])}...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.project_root),
                env={**subprocess.os.environ, "PYTHONPATH": str(self.project_root)}
            )

            stdout, stderr = process.communicate()
            return_code = process.returncode

            execution_time = time.time() - start_time
            self.total_time += execution_time

            # Parse output
            output = self._parse_output(stdout, stderr)

            if return_code == 0:
                self.successful_tasks += 1
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    output={
                        "scenario": scenario,
                        "stdout": stdout,
                        "stderr": stderr,
                        "parsed_output": output
                    },
                    execution_time=execution_time,
                    metadata={
                        "scenario": scenario,
                        "operation": task.parameters.get("operation"),
                        "return_code": return_code
                    }
                )
            else:
                self.failed_tasks += 1
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Scenario failed with return code {return_code}: {stderr}",
                    execution_time=execution_time,
                    metadata={
                        "scenario": scenario,
                        "return_code": return_code,
                        "stdout": stdout,
                        "stderr": stderr
                    }
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.total_time += execution_time
            self.failed_tasks += 1

            logger.error(f"Scenario execution failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def _parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Parse scenario output

        Attempts to extract structured data from stdout/stderr.

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            Parsed output dictionary
        """
        result = {}

        # Try to parse JSON output
        for line in stdout.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass

        # Extract key information from text output
        if not result:
            result = {
                "success": "error" not in stderr.lower() and "failed" not in stderr.lower(),
                "output_lines": stdout.strip().split("\n") if stdout.strip() else [],
                "error_lines": stderr.strip().split("\n") if stderr.strip() else []
            }

        return result

    async def health_check(self) -> HealthCheckResult:
        """
        Check Scenario Adapter health

        Returns:
            HealthCheckResult with status and details
        """
        try:
            if not self._initialized:
                return HealthCheckResult(
                    module_name=self.module_name,
                    status=ModuleStatus.UNHEALTHY,
                    message="Scenario Adapter not initialized"
                )

            # Check script availability
            missing = []
            available = []

            for scenario, script_path in SCENARIO_SCRIPTS.items():
                full_path = self.project_root / script_path
                if full_path.exists():
                    available.append(scenario)
                else:
                    missing.append(scenario)

            # Determine status
            if not available:
                status = ModuleStatus.UNHEALTHY
                message = "No scenario scripts available"
            elif missing:
                status = ModuleStatus.DEGRADED
                message = f"Missing {len(missing)} scenario scripts"
            else:
                status = ModuleStatus.HEALTHY
                message = "All scenario scripts available"

            success_rate = (
                self.successful_tasks / self.total_tasks * 100
                if self.total_tasks > 0 else 100.0
            )

            avg_time = (
                self.total_time / self.total_tasks
                if self.total_tasks > 0 else 0.0
            )

            return HealthCheckResult(
                module_name=self.module_name,
                status=status,
                message=message,
                details={
                    "total_scenarios": len(SCENARIO_SCRIPTS),
                    "available_scenarios": len(available),
                    "missing_scenarios": len(missing),
                    "missing_list": missing,
                    "total_tasks": self.total_tasks,
                    "successful_tasks": self.successful_tasks,
                    "failed_tasks": self.failed_tasks,
                    "success_rate": f"{success_rate:.1f}%",
                    "avg_execution_time": f"{avg_time:.2f}s",
                    "scenario_usage": self.scenario_counts
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return HealthCheckResult(
                module_name=self.module_name,
                status=ModuleStatus.UNKNOWN,
                message=f"Health check error: {str(e)}"
            )

    def get_capabilities(self) -> ModuleCapabilities:
        """
        Get Scenario Adapter capabilities

        Returns:
            ModuleCapabilities describing scenario features
        """
        return ModuleCapabilities(
            module_name=self.module_name,
            module_type=self.module_type,
            supported_operations=[
                "video",        # Video processing
                "audio",        # Audio processing
                "image",        # Image processing
                "subtitle",     # Subtitle automation
                "file",         # File organization
                "dataset",      # Dataset building
                "augment",      # Data augmentation
                "annotate",     # Annotation
                "categorize",   # Auto categorization
                "knowledge",    # Knowledge base
                "analyze"       # Media analysis
            ],
            requires_gpu=False,  # All scenarios are CPU-only
            max_concurrent_tasks=4,  # Limit concurrent CPU-intensive tasks
            estimated_memory_mb=512,  # Varies by scenario, ~512MB average
            description=(
                "Automation Scenario Adapter for 11 CPU-only processing workflows. "
                "Includes video/audio/image processing, file organization, "
                "dataset building, annotation, categorization, and media analysis. "
                "All operations run via CLI scripts with standardized interface."
            )
        )

    async def cleanup(self):
        """Cleanup Scenario Adapter resources"""
        logger.info("Scenario Adapter cleaned up (no resources to release)")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get adapter statistics

        Returns:
            Statistics dictionary
        """
        success_rate = (
            self.successful_tasks / self.total_tasks * 100
            if self.total_tasks > 0 else 0.0
        )

        avg_time = (
            self.total_time / self.total_tasks
            if self.total_tasks > 0 else 0.0
        )

        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": success_rate,
            "total_time": self.total_time,
            "avg_execution_time": avg_time,
            "scenario_usage": self.scenario_counts,
            "available_scenarios": list(SCENARIO_SCRIPTS.keys())
        }

    def __repr__(self) -> str:
        available = sum(
            1 for script_path in SCENARIO_SCRIPTS.values()
            if (self.project_root / script_path).exists()
        )
        return (
            f"ScenarioAdapter(initialized={self._initialized}, "
            f"scenarios={available}/{len(SCENARIO_SCRIPTS)}, "
            f"tasks={self.total_tasks})"
        )
