"""
Data Pipeline Automation - Pipeline Orchestrator

Orchestrates pipeline execution with checkpoint management and progress tracking.

Author: Animation AI Studio
Date: 2025-12-03
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..common import (
    Pipeline,
    PipelineStage,
    PipelineResult,
    StageResult,
    PipelineState,
    ExecutionStatus,
    OrchestratorConfig,
    CheckpointData,
    topological_sort,
    build_parallel_groups,
    generate_pipeline_id,
)
from ..stage_executors import StageExecutor

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages pipeline checkpoints for resume capability

    Handles saving/loading pipeline state and pruning old checkpoints.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Initialized checkpoint manager at {checkpoint_dir}")

    def save_checkpoint(
        self,
        pipeline: Pipeline,
        completed_stages: List[str],
        stage_outputs: Dict[str, Dict[str, Any]]
    ) -> Path:
        """
        Save pipeline checkpoint

        Args:
            pipeline: Pipeline being executed
            completed_stages: List of completed stage IDs
            stage_outputs: Outputs from completed stages

        Returns:
            Path to checkpoint file
        """
        checkpoint_data = CheckpointData(
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            completed_stages=completed_stages,
            stage_outputs=stage_outputs,
            timestamp=time.time()
        )

        checkpoint_file = self.checkpoint_dir / f"{pipeline.id}_checkpoint_{int(time.time())}.json"

        # Atomic write
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data.__dict__, f, indent=2)

        temp_file.rename(checkpoint_file)

        logger.info(f"Saved checkpoint: {checkpoint_file}")
        return checkpoint_file

    def load_checkpoint(self, checkpoint_path: Path) -> CheckpointData:
        """
        Load checkpoint from file

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint data is invalid
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            checkpoint = CheckpointData(**data)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint

        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")

    def get_latest_checkpoint(self, pipeline_id: str) -> Optional[Path]:
        """
        Get latest checkpoint for pipeline

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{pipeline_id}_checkpoint_*.json"))

        if not checkpoints:
            return None

        # Sort by timestamp (embedded in filename)
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest

    def list_checkpoints(self, pipeline_id: str) -> List[Path]:
        """
        List all checkpoints for pipeline

        Args:
            pipeline_id: Pipeline ID

        Returns:
            List of checkpoint paths sorted by timestamp
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{pipeline_id}_checkpoint_*.json"))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)

    def prune_checkpoints(self, pipeline_id: str, keep_last: int = 3):
        """
        Remove old checkpoints, keeping only the last N

        Args:
            pipeline_id: Pipeline ID
            keep_last: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints(pipeline_id)

        if len(checkpoints) <= keep_last:
            return

        # Remove oldest checkpoints
        for checkpoint in checkpoints[keep_last:]:
            checkpoint.unlink()
            logger.debug(f"Pruned old checkpoint: {checkpoint}")

        logger.info(f"Pruned {len(checkpoints) - keep_last} old checkpoints for pipeline '{pipeline_id}'")


class PipelineProgressMonitor:
    """
    Monitors and tracks pipeline execution progress

    Calculates metrics, ETA, and emits progress events.
    """

    def __init__(self):
        """Initialize progress monitor"""
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized pipeline progress monitor")

    def start_pipeline(self, pipeline: Pipeline):
        """
        Start tracking pipeline

        Args:
            pipeline: Pipeline to track
        """
        self.pipelines[pipeline.id] = {
            'pipeline': pipeline,
            'start_time': time.time(),
            'stage_times': {},
            'stage_metrics': {},
            'completed_stages': [],
        }

        logger.info(f"Started tracking pipeline '{pipeline.name}' ({pipeline.id})")

    def update_stage_progress(
        self,
        pipeline_id: str,
        stage_id: str,
        progress: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Update stage progress

        Args:
            pipeline_id: Pipeline ID
            stage_id: Stage ID
            progress: Progress percentage (0-100)
            metrics: Optional stage metrics
        """
        if pipeline_id not in self.pipelines:
            logger.warning(f"Unknown pipeline: {pipeline_id}")
            return

        if metrics:
            self.pipelines[pipeline_id]['stage_metrics'][stage_id] = metrics

        logger.debug(f"Stage '{stage_id}' progress: {progress:.1f}%")

    def complete_stage(self, pipeline_id: str, stage_id: str, result: StageResult):
        """
        Mark stage as completed

        Args:
            pipeline_id: Pipeline ID
            stage_id: Stage ID
            result: Stage execution result
        """
        if pipeline_id not in self.pipelines:
            logger.warning(f"Unknown pipeline: {pipeline_id}")
            return

        self.pipelines[pipeline_id]['stage_times'][stage_id] = result.duration
        self.pipelines[pipeline_id]['completed_stages'].append(stage_id)

        if result.metrics:
            self.pipelines[pipeline_id]['stage_metrics'][stage_id] = result.metrics

        logger.info(f"Stage '{stage_id}' completed in {result.duration:.2f}s")

    def calculate_eta(self, pipeline_id: str) -> float:
        """
        Calculate estimated time remaining

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Estimated seconds remaining
        """
        if pipeline_id not in self.pipelines:
            return 0.0

        data = self.pipelines[pipeline_id]
        pipeline = data['pipeline']
        completed = len(data['completed_stages'])
        total = len(pipeline.stages)

        if completed == 0:
            return 0.0

        # Average time per stage
        elapsed = time.time() - data['start_time']
        avg_time_per_stage = elapsed / completed

        # Estimate remaining
        remaining_stages = total - completed
        return remaining_stages * avg_time_per_stage

    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get current pipeline status

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Status dictionary with metrics and progress
        """
        if pipeline_id not in self.pipelines:
            return {}

        data = self.pipelines[pipeline_id]
        pipeline = data['pipeline']

        elapsed = time.time() - data['start_time']
        completed = len(data['completed_stages'])
        total = len(pipeline.stages)
        progress = (completed / total * 100) if total > 0 else 0

        return {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline.name,
            'state': pipeline.state.value,
            'progress_percent': progress,
            'completed_stages': completed,
            'total_stages': total,
            'elapsed_seconds': elapsed,
            'eta_seconds': self.calculate_eta(pipeline_id),
            'stage_metrics': data['stage_metrics'],
        }


class PipelineOrchestrator:
    """
    Main pipeline orchestrator

    Coordinates stage execution, manages dependencies, handles checkpoints,
    and tracks progress.
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize orchestrator

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.progress_monitor = PipelineProgressMonitor()
        self.executor_registry: Dict[str, type] = {}

        # Active pipelines
        self.active_pipelines: Dict[str, Pipeline] = {}

        logger.info(f"Initialized pipeline orchestrator (max_parallel={config.max_parallel_stages})")

    def register_executor(self, stage_type: str, executor_class: type):
        """
        Register stage executor

        Args:
            stage_type: Stage type identifier
            executor_class: Executor class (must inherit from StageExecutor)
        """
        if not issubclass(executor_class, StageExecutor):
            raise ValueError(f"Executor must inherit from StageExecutor: {executor_class}")

        self.executor_registry[stage_type] = executor_class
        logger.debug(f"Registered executor for stage type '{stage_type}'")

    def execute_pipeline(
        self,
        pipeline: Pipeline,
        resume: bool = False
    ) -> PipelineResult:
        """
        Execute pipeline

        Args:
            pipeline: Pipeline to execute
            resume: Whether to resume from checkpoint

        Returns:
            Pipeline execution result
        """
        logger.info(f"Starting pipeline execution: '{pipeline.name}' ({pipeline.id})")

        # Track pipeline
        self.active_pipelines[pipeline.id] = pipeline
        self.progress_monitor.start_pipeline(pipeline)

        # Resume state
        completed_stages: Set[str] = set()
        stage_outputs: Dict[str, Dict[str, Any]] = {}

        if resume:
            checkpoint = self._load_checkpoint_for_resume(pipeline.id)
            if checkpoint:
                completed_stages = set(checkpoint.completed_stages)
                stage_outputs = checkpoint.stage_outputs
                logger.info(f"Resuming from checkpoint ({len(completed_stages)} stages completed)")

        # Sort stages by dependencies
        sorted_stages = topological_sort(pipeline.stages)
        pipeline.stages = sorted_stages

        # Build parallel execution groups
        parallel_groups = build_parallel_groups(sorted_stages)

        # Execute
        pipeline.state = PipelineState.RUNNING
        start_time = time.time()

        try:
            for group_idx, group in enumerate(parallel_groups):
                logger.info(f"Executing stage group {group_idx + 1}/{len(parallel_groups)} ({len(group)} stages)")

                # Filter out already completed stages
                stages_to_run = [
                    stage for stage in sorted_stages
                    if stage.id in group and stage.id not in completed_stages
                ]

                if not stages_to_run:
                    logger.debug(f"Group {group_idx + 1} already completed, skipping")
                    continue

                # Execute stages in parallel
                group_results = self._execute_stage_group(
                    stages_to_run,
                    stage_outputs
                )

                # Process results
                for stage_id, result in group_results.items():
                    if result.status == ExecutionStatus.SUCCESS:
                        completed_stages.add(stage_id)
                        stage_outputs[stage_id] = result.outputs
                        self.progress_monitor.complete_stage(pipeline.id, stage_id, result)

                    elif result.status == ExecutionStatus.FAILED:
                        # Handle failure
                        if self._should_retry(result):
                            logger.warning(f"Stage '{stage_id}' failed, retrying...")
                            # Retry logic would go here
                        else:
                            logger.error(f"Stage '{stage_id}' failed: {result.error_message}")
                            pipeline.state = PipelineState.FAILED
                            raise RuntimeError(f"Stage '{stage_id}' failed: {result.error_message}")

                # Save checkpoint after each group
                if self.config.enable_checkpoints:
                    self.checkpoint_manager.save_checkpoint(
                        pipeline,
                        list(completed_stages),
                        stage_outputs
                    )

            # Success
            pipeline.state = PipelineState.COMPLETED
            duration = time.time() - start_time

            logger.info(f"Pipeline '{pipeline.name}' completed successfully in {duration:.2f}s")

            return PipelineResult(
                pipeline_id=pipeline.id,
                pipeline_name=pipeline.name,
                status=ExecutionStatus.SUCCESS,
                duration=duration,
                stage_results=[],  # Would collect from group_results
                total_stages=len(pipeline.stages),
                successful_stages=len(completed_stages),
                failed_stages=0,
                error_message=None
            )

        except Exception as e:
            pipeline.state = PipelineState.FAILED
            duration = time.time() - start_time

            logger.error(f"Pipeline '{pipeline.name}' failed: {e}")

            return PipelineResult(
                pipeline_id=pipeline.id,
                pipeline_name=pipeline.name,
                status=ExecutionStatus.FAILED,
                duration=duration,
                stage_results=[],
                total_stages=len(pipeline.stages),
                successful_stages=len(completed_stages),
                failed_stages=1,
                error_message=str(e)
            )

        finally:
            # Cleanup
            if pipeline.id in self.active_pipelines:
                del self.active_pipelines[pipeline.id]

            # Prune old checkpoints
            if self.config.enable_checkpoints:
                self.checkpoint_manager.prune_checkpoints(pipeline.id, keep_last=3)

    def _execute_stage_group(
        self,
        stages: List[PipelineStage],
        stage_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, StageResult]:
        """
        Execute a group of stages in parallel

        Args:
            stages: Stages to execute
            stage_outputs: Outputs from previous stages

        Returns:
            Dict mapping stage_id to result
        """
        results: Dict[str, StageResult] = {}

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_stages) as executor:
            # Submit all stages
            futures = {
                executor.submit(self._execute_stage, stage, stage_outputs): stage
                for stage in stages
            }

            # Collect results
            for future in as_completed(futures):
                stage = futures[future]
                try:
                    result = future.result()
                    results[stage.id] = result

                except Exception as e:
                    logger.error(f"Stage '{stage.id}' raised exception: {e}")
                    results[stage.id] = StageResult(
                        stage_id=stage.id,
                        status=ExecutionStatus.FAILED,
                        duration=0.0,
                        outputs={},
                        metrics={},
                        error_message=str(e)
                    )

        return results

    def _execute_stage(
        self,
        stage: PipelineStage,
        stage_outputs: Dict[str, Dict[str, Any]]
    ) -> StageResult:
        """
        Execute a single stage

        Args:
            stage: Stage to execute
            stage_outputs: Outputs from previous stages

        Returns:
            Stage execution result
        """
        logger.info(f"Executing stage: '{stage.id}' (type: {stage.type.value})")

        # Get executor
        executor_class = self.executor_registry.get(stage.type.value)
        if not executor_class:
            raise ValueError(f"No executor registered for stage type '{stage.type.value}'")

        # Create executor
        executor = executor_class(stage.id, stage.config)

        # Validate config
        try:
            executor.validate_config()
        except Exception as e:
            logger.error(f"Stage '{stage.id}' config validation failed: {e}")
            return StageResult(
                stage_id=stage.id,
                status=ExecutionStatus.FAILED,
                duration=0.0,
                outputs={},
                metrics={},
                error_message=f"Config validation failed: {e}"
            )

        # Prepare inputs from dependency outputs
        inputs = self._prepare_stage_inputs(stage, stage_outputs)

        # Execute
        try:
            result = executor.execute(inputs)
            return result

        except Exception as e:
            logger.error(f"Stage '{stage.id}' execution failed: {e}")
            return StageResult(
                stage_id=stage.id,
                status=ExecutionStatus.FAILED,
                duration=0.0,
                outputs={},
                metrics={},
                error_message=str(e)
            )

        finally:
            executor.cleanup()

    def _prepare_stage_inputs(
        self,
        stage: PipelineStage,
        stage_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare stage inputs from dependency outputs

        Args:
            stage: Stage to prepare inputs for
            stage_outputs: Outputs from previous stages

        Returns:
            Input dict for stage
        """
        inputs = {}

        for dep_id in stage.depends_on:
            if dep_id not in stage_outputs:
                logger.warning(f"Dependency '{dep_id}' has no outputs for stage '{stage.id}'")
                continue

            # Add dependency outputs to inputs
            dep_outputs = stage_outputs[dep_id]
            inputs[dep_id] = dep_outputs

        return inputs

    def _load_checkpoint_for_resume(self, pipeline_id: str) -> Optional[CheckpointData]:
        """
        Load checkpoint for resuming pipeline

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Checkpoint data or None
        """
        checkpoint_path = self.checkpoint_manager.get_latest_checkpoint(pipeline_id)

        if not checkpoint_path:
            logger.info("No checkpoint found for resume")
            return None

        try:
            return self.checkpoint_manager.load_checkpoint(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _should_retry(self, result: StageResult) -> bool:
        """
        Determine if stage should be retried

        Args:
            result: Stage result

        Returns:
            True if should retry
        """
        # Simple retry logic - could be enhanced with retry policy
        return self.config.retry_failed_stages and result.status == ExecutionStatus.FAILED

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel running pipeline

        Args:
            pipeline_id: Pipeline ID

        Returns:
            True if cancelled
        """
        if pipeline_id not in self.active_pipelines:
            logger.warning(f"Pipeline not active: {pipeline_id}")
            return False

        pipeline = self.active_pipelines[pipeline_id]
        pipeline.state = PipelineState.CANCELLED

        logger.info(f"Cancelled pipeline: {pipeline_id}")
        return True

    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get pipeline status

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Status dictionary
        """
        return self.progress_monitor.get_pipeline_status(pipeline_id)
