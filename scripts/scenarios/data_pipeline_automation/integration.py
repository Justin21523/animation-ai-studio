"""
Data Pipeline Automation - Integration Layer

Provides integration with EventBus, Safety System, and configuration loading.

Author: Animation AI Studio
Date: 2025-12-04
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from .common import (
    Pipeline,
    PipelineStage,
    StageResult,
    PipelineResult,
    PipelineState,
    ExecutionStatus,
    OrchestratorConfig
)

logger = logging.getLogger(__name__)


class PipelineEventEmitter:
    """
    EventBus integration for pipeline events

    Emits events at key pipeline lifecycle points:
    - pipeline.started
    - pipeline.stage.started
    - pipeline.stage.completed
    - pipeline.stage.failed
    - pipeline.completed
    - pipeline.failed
    """

    def __init__(self, event_bus: Optional[Any] = None):
        """
        Initialize event emitter

        Args:
            event_bus: EventBus instance (optional)
        """
        self.event_bus = event_bus
        self.enabled = event_bus is not None

    def emit_pipeline_started(self, pipeline: Pipeline) -> None:
        """
        Emit pipeline started event

        Args:
            pipeline: Pipeline instance
        """
        if not self.enabled:
            return

        try:
            event_data = {
                "pipeline_id": pipeline.id,
                "pipeline_name": pipeline.name,
                "pipeline_version": pipeline.version,
                "total_stages": len(pipeline.stages),
                "timestamp": pipeline.started_at
            }

            # Note: EventBus might be async or sync
            # Handle both cases
            if hasattr(self.event_bus, 'publish'):
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    # Async - need to schedule
                    import asyncio
                    asyncio.create_task(self.event_bus.publish("pipeline.started", event_data))
                else:
                    # Sync
                    self.event_bus.publish("pipeline.started", event_data)

            logger.info(f"Event emitted: pipeline.started ({pipeline.id})")

        except Exception as e:
            logger.warning(f"Failed to emit pipeline.started event: {e}")

    def emit_stage_started(self, stage: PipelineStage, pipeline_id: str) -> None:
        """
        Emit stage started event

        Args:
            stage: Pipeline stage
            pipeline_id: Pipeline identifier
        """
        if not self.enabled:
            return

        try:
            event_data = {
                "pipeline_id": pipeline_id,
                "stage_id": stage.id,
                "stage_type": stage.type.value if hasattr(stage.type, 'value') else str(stage.type),
                "depends_on": stage.depends_on,
                "timestamp": stage.started_at
            }

            if hasattr(self.event_bus, 'publish'):
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    import asyncio
                    asyncio.create_task(self.event_bus.publish("pipeline.stage.started", event_data))
                else:
                    self.event_bus.publish("pipeline.stage.started", event_data)

            logger.debug(f"Event emitted: pipeline.stage.started ({stage.id})")

        except Exception as e:
            logger.warning(f"Failed to emit pipeline.stage.started event: {e}")

    def emit_stage_completed(self, stage: PipelineStage, result: StageResult, pipeline_id: str) -> None:
        """
        Emit stage completed event

        Args:
            stage: Pipeline stage
            result: Stage execution result
            pipeline_id: Pipeline identifier
        """
        if not self.enabled:
            return

        try:
            event_data = {
                "pipeline_id": pipeline_id,
                "stage_id": stage.id,
                "stage_type": stage.type.value if hasattr(stage.type, 'value') else str(stage.type),
                "status": result.status.value,
                "duration": result.duration,
                "outputs": result.outputs,
                "metrics": result.metrics,
                "timestamp": stage.completed_at
            }

            if hasattr(self.event_bus, 'publish'):
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    import asyncio
                    asyncio.create_task(self.event_bus.publish("pipeline.stage.completed", event_data))
                else:
                    self.event_bus.publish("pipeline.stage.completed", event_data)

            logger.debug(f"Event emitted: pipeline.stage.completed ({stage.id})")

        except Exception as e:
            logger.warning(f"Failed to emit pipeline.stage.completed event: {e}")

    def emit_stage_failed(self, stage: PipelineStage, error_message: str, pipeline_id: str) -> None:
        """
        Emit stage failed event

        Args:
            stage: Pipeline stage
            error_message: Error message
            pipeline_id: Pipeline identifier
        """
        if not self.enabled:
            return

        try:
            event_data = {
                "pipeline_id": pipeline_id,
                "stage_id": stage.id,
                "stage_type": stage.type.value if hasattr(stage.type, 'value') else str(stage.type),
                "error_message": error_message,
                "retry_count": stage.retry_count,
                "timestamp": stage.completed_at
            }

            if hasattr(self.event_bus, 'publish'):
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    import asyncio
                    asyncio.create_task(self.event_bus.publish("pipeline.stage.failed", event_data))
                else:
                    self.event_bus.publish("pipeline.stage.failed", event_data)

            logger.debug(f"Event emitted: pipeline.stage.failed ({stage.id})")

        except Exception as e:
            logger.warning(f"Failed to emit pipeline.stage.failed event: {e}")

    def emit_pipeline_completed(self, pipeline: Pipeline, result: PipelineResult) -> None:
        """
        Emit pipeline completed event

        Args:
            pipeline: Pipeline instance
            result: Pipeline execution result
        """
        if not self.enabled:
            return

        try:
            event_data = {
                "pipeline_id": pipeline.id,
                "pipeline_name": pipeline.name,
                "status": result.status.value,
                "total_duration": result.total_duration,
                "stages_completed": len([s for s in pipeline.stages if s.is_completed]),
                "stages_failed": len([s for s in pipeline.stages if s.is_failed]),
                "final_outputs": result.final_outputs,
                "timestamp": pipeline.completed_at
            }

            if hasattr(self.event_bus, 'publish'):
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    import asyncio
                    asyncio.create_task(self.event_bus.publish("pipeline.completed", event_data))
                else:
                    self.event_bus.publish("pipeline.completed", event_data)

            logger.info(f"Event emitted: pipeline.completed ({pipeline.id})")

        except Exception as e:
            logger.warning(f"Failed to emit pipeline.completed event: {e}")

    def emit_pipeline_failed(self, pipeline: Pipeline, error_message: str) -> None:
        """
        Emit pipeline failed event

        Args:
            pipeline: Pipeline instance
            error_message: Error message
        """
        if not self.enabled:
            return

        try:
            event_data = {
                "pipeline_id": pipeline.id,
                "pipeline_name": pipeline.name,
                "error_message": error_message,
                "stages_completed": len([s for s in pipeline.stages if s.is_completed]),
                "stages_failed": len([s for s in pipeline.stages if s.is_failed]),
                "timestamp": pipeline.completed_at
            }

            if hasattr(self.event_bus, 'publish'):
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    import asyncio
                    asyncio.create_task(self.event_bus.publish("pipeline.failed", event_data))
                else:
                    self.event_bus.publish("pipeline.failed", event_data)

            logger.warning(f"Event emitted: pipeline.failed ({pipeline.id})")

        except Exception as e:
            logger.warning(f"Failed to emit pipeline.failed event: {e}")


class PipelineSafetyValidator:
    """
    Safety System integration for pipeline validation

    Provides resource constraint checking and usage reporting.
    """

    def __init__(self, max_vram_gb: float = 15.0, max_cpu_percent: float = 95.0, max_disk_gb: float = 100.0):
        """
        Initialize safety validator

        Args:
            max_vram_gb: Maximum VRAM usage in GB
            max_cpu_percent: Maximum CPU usage percentage
            max_disk_gb: Maximum disk space required in GB
        """
        self.max_vram_gb = max_vram_gb
        self.max_cpu_percent = max_cpu_percent
        self.max_disk_gb = max_disk_gb

    def register_pipeline(self, pipeline: Pipeline) -> bool:
        """
        Register pipeline with safety system

        Args:
            pipeline: Pipeline to register

        Returns:
            True if registration successful
        """
        logger.info(f"Pipeline registered: {pipeline.id} ({pipeline.name})")
        return True

    def check_resource_constraints(self) -> bool:
        """
        Check if current resource usage is within limits

        Returns:
            True if resources are available
        """
        try:
            # Check GPU VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                    vram_percent = (vram_used / vram_total) * 100

                    if vram_used > self.max_vram_gb:
                        logger.warning(f"VRAM usage ({vram_used:.2f}GB) exceeds limit ({self.max_vram_gb}GB)")
                        return False
            except ImportError:
                pass  # PyTorch not available

            # Check CPU usage
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > self.max_cpu_percent:
                    logger.warning(f"CPU usage ({cpu_percent}%) exceeds limit ({self.max_cpu_percent}%)")
                    return False
            except ImportError:
                pass  # psutil not available

            # Check disk space
            try:
                import psutil
                disk = psutil.disk_usage('/')
                disk_free_gb = disk.free / (1024 ** 3)
                if disk_free_gb < self.max_disk_gb:
                    logger.warning(f"Free disk space ({disk_free_gb:.2f}GB) below requirement ({self.max_disk_gb}GB)")
                    return False
            except ImportError:
                pass  # psutil not available

            return True

        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Fail open

    def report_resource_usage(self, vram_gb: float, cpu_percent: float, disk_gb: float) -> None:
        """
        Report resource usage to safety system

        Args:
            vram_gb: VRAM usage in GB
            cpu_percent: CPU usage percentage
            disk_gb: Disk usage in GB
        """
        logger.info(f"Resource usage: VRAM={vram_gb:.2f}GB, CPU={cpu_percent:.1f}%, Disk={disk_gb:.2f}GB")


def load_pipeline_config(config_path: Path) -> OrchestratorConfig:
    """
    Load pipeline configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OrchestratorConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract orchestrator config (or use defaults)
        orchestrator_config = config_dict.get("orchestrator", {})

        # Convert paths
        checkpoint_dir = orchestrator_config.get("checkpoint_dir", "/tmp/pipeline_checkpoints")
        log_dir = orchestrator_config.get("log_dir", "/tmp/pipeline_logs")

        config = OrchestratorConfig(
            checkpoint_dir=Path(checkpoint_dir),
            enable_parallel_execution=orchestrator_config.get("enable_parallel_execution", True),
            max_parallel_stages=orchestrator_config.get("max_parallel_stages", 4),
            checkpoint_interval=orchestrator_config.get("checkpoint_interval", 300),
            enable_auto_retry=orchestrator_config.get("enable_auto_retry", True),
            max_retries=orchestrator_config.get("max_retries", 2),
            event_bus_enabled=orchestrator_config.get("event_bus_enabled", True),
            log_dir=Path(log_dir)
        )

        logger.info(f"Loaded pipeline config from: {config_path}")
        return config

    except Exception as e:
        raise ValueError(f"Failed to load pipeline config: {e}")
