"""
Data Pipeline Automation - Common Types and Utilities

This module defines core enums, dataclasses, and helper functions for
the data pipeline automation system.

Author: Animation AI Studio
Date: 2025-12-03
"""

import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ========================================================================
# Enums
# ========================================================================

class PipelineState(Enum):
    """Pipeline execution state"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionStatus(Enum):
    """Stage execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RUNNING = "running"
    PENDING = "pending"


class StageType(Enum):
    """Stage type identifiers"""
    FRAME_EXTRACTION = "frame_extraction"
    SEGMENTATION = "segmentation"
    CLUSTERING = "clustering"
    TRAINING_DATA_PREP = "training_data_prep"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Resource types for monitoring"""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"


# ========================================================================
# Configuration Dataclasses
# ========================================================================

@dataclass
class StageConfig:
    """Base stage configuration"""
    input_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class RetryPolicy:
    """Retry policy for failed stages"""
    max_retries: int = 2
    retry_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    retry_on_errors: List[str] = field(default_factory=lambda: ["timeout", "resource_exhausted"])


@dataclass
class OrchestratorConfig:
    """Pipeline orchestrator configuration"""
    checkpoint_dir: Path = field(default_factory=lambda: Path("/tmp/pipeline_checkpoints"))
    enable_parallel_execution: bool = True
    max_parallel_stages: int = 4
    checkpoint_interval: int = 300  # Save checkpoint every 5 minutes
    enable_auto_retry: bool = True
    max_retries: int = 2
    event_bus_enabled: bool = True
    log_dir: Path = field(default_factory=lambda: Path("/tmp/pipeline_logs"))

    def __post_init__(self):
        """Ensure paths are Path objects"""
        if not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if not isinstance(self.log_dir, Path):
            self.log_dir = Path(self.log_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ========================================================================
# Core Pipeline Dataclasses
# ========================================================================

@dataclass
class PipelineStage:
    """
    Pipeline stage definition

    Attributes:
        id: Unique stage identifier
        type: Stage type (frame_extraction, segmentation, etc.)
        depends_on: List of stage IDs this stage depends on
        config: Stage-specific configuration
        status: Current execution status
        outputs: Stage outputs (paths, metrics, etc.)
        duration: Stage execution duration in seconds
        error_message: Error message if failed
        retry_count: Number of retry attempts
        started_at: Timestamp when stage started
        completed_at: Timestamp when stage completed
    """
    id: str
    type: StageType
    depends_on: List[str]
    config: Dict[str, Any]
    status: ExecutionStatus = ExecutionStatus.PENDING
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def is_completed(self) -> bool:
        """Check if stage completed successfully"""
        return self.status == ExecutionStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """Check if stage failed"""
        return self.status == ExecutionStatus.FAILED

    @property
    def is_running(self) -> bool:
        """Check if stage is running"""
        return self.status == ExecutionStatus.RUNNING

    def mark_started(self):
        """Mark stage as started"""
        self.started_at = time.time()
        self.status = ExecutionStatus.RUNNING

    def mark_completed(self, outputs: Dict[str, Any]):
        """Mark stage as completed"""
        self.completed_at = time.time()
        self.duration = self.completed_at - (self.started_at or self.completed_at)
        self.status = ExecutionStatus.SUCCESS
        self.outputs = outputs

    def mark_failed(self, error_message: str):
        """Mark stage as failed"""
        self.completed_at = time.time()
        self.duration = self.completed_at - (self.started_at or self.completed_at)
        self.status = ExecutionStatus.FAILED
        self.error_message = error_message


@dataclass
class Pipeline:
    """
    Complete pipeline definition

    Attributes:
        id: Unique pipeline identifier
        name: Pipeline name
        version: Pipeline version
        stages: List of pipeline stages
        config: Pipeline-wide configuration
        created_at: Creation timestamp
        state: Current pipeline state
        checkpoint_dir: Checkpoint storage directory
        started_at: Pipeline start timestamp
        completed_at: Pipeline completion timestamp
    """
    id: str
    name: str
    version: str
    stages: List[PipelineStage]
    config: Dict[str, Any]
    created_at: float
    state: PipelineState = PipelineState.PENDING
    checkpoint_dir: Optional[Path] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def progress_percent(self) -> float:
        """Calculate pipeline progress percentage"""
        if not self.stages:
            return 0.0
        completed = sum(1 for s in self.stages if s.status == ExecutionStatus.SUCCESS)
        return (completed / len(self.stages)) * 100

    @property
    def duration(self) -> Optional[float]:
        """Calculate total pipeline duration"""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    @property
    def completed_stages(self) -> List[str]:
        """Get list of completed stage IDs"""
        return [s.id for s in self.stages if s.is_completed]

    @property
    def failed_stages(self) -> List[str]:
        """Get list of failed stage IDs"""
        return [s.id for s in self.stages if s.is_failed]

    @property
    def running_stages(self) -> List[str]:
        """Get list of running stage IDs"""
        return [s.id for s in self.stages if s.is_running]

    def get_stage(self, stage_id: str) -> Optional[PipelineStage]:
        """Get stage by ID"""
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        return None

    def mark_started(self):
        """Mark pipeline as started"""
        self.started_at = time.time()
        self.state = PipelineState.RUNNING

    def mark_completed(self):
        """Mark pipeline as completed"""
        self.completed_at = time.time()
        self.state = PipelineState.COMPLETED

    def mark_failed(self):
        """Mark pipeline as failed"""
        self.completed_at = time.time()
        self.state = PipelineState.FAILED


@dataclass
class StageResult:
    """
    Stage execution result

    Attributes:
        stage_id: Stage identifier
        status: Execution status
        duration: Execution duration in seconds
        outputs: Stage outputs (paths, counts, etc.)
        metrics: Performance metrics
        error_message: Error message if failed
        checkpoints: List of checkpoint files created
    """
    stage_id: str
    status: ExecutionStatus
    duration: float
    outputs: Dict[str, Any]
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    checkpoints: List[Path] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if execution succeeded"""
        return self.status == ExecutionStatus.SUCCESS


@dataclass
class PipelineResult:
    """
    Complete pipeline execution result

    Attributes:
        pipeline_id: Pipeline identifier
        status: Final pipeline state
        total_duration: Total execution duration
        stage_results: Results from all stages
        final_outputs: Final pipeline outputs
        error_message: Error message if failed
    """
    pipeline_id: str
    status: PipelineState
    total_duration: float
    stage_results: List[StageResult]
    final_outputs: Dict[str, Any]
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if pipeline succeeded"""
        return self.status == PipelineState.COMPLETED

    @property
    def failed_stages(self) -> List[str]:
        """Get list of failed stage IDs"""
        return [r.stage_id for r in self.stage_results if not r.success]


@dataclass
class CheckpointData:
    """
    Pipeline checkpoint data

    Attributes:
        pipeline_id: Pipeline identifier
        checkpoint_time: Checkpoint creation timestamp
        completed_stages: List of completed stage IDs
        stage_outputs: Outputs from completed stages
        pipeline_state: Current pipeline state
        next_stage: Next stage to execute
    """
    pipeline_id: str
    checkpoint_time: float
    completed_stages: List[str]
    stage_outputs: Dict[str, Dict[str, Any]]
    pipeline_state: PipelineState
    next_stage: Optional[str] = None


# ========================================================================
# Helper Functions
# ========================================================================

def generate_pipeline_id(name: str) -> str:
    """
    Generate unique pipeline ID from name and timestamp

    Args:
        name: Pipeline name

    Returns:
        Unique pipeline ID (format: name_YYYYMMDD_HHMMSS)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean name (remove special characters)
    clean_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return f"{clean_name}_{timestamp}"


def generate_stage_hash(stage: PipelineStage) -> str:
    """
    Generate hash for stage configuration (for cache key)

    Args:
        stage: Pipeline stage

    Returns:
        MD5 hash of stage config
    """
    import json
    config_str = json.dumps(stage.config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def validate_dag(stages: List[PipelineStage]) -> bool:
    """
    Validate that stages form a valid DAG (no cycles)

    Args:
        stages: List of pipeline stages

    Returns:
        True if valid DAG, False if cycles detected
    """
    # Build adjacency list
    stage_ids = {s.id for s in stages}
    adjacency: Dict[str, List[str]] = {s.id: s.depends_on for s in stages}

    # Check all dependencies exist
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in stage_ids:
                return False

    # Detect cycles using DFS
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def has_cycle(node: str) -> bool:
        """DFS cycle detection"""
        visited.add(node)
        rec_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for stage_id in stage_ids:
        if stage_id not in visited:
            if has_cycle(stage_id):
                return False

    return True


def topological_sort(stages: List[PipelineStage]) -> List[PipelineStage]:
    """
    Sort stages in topological order (dependency order)

    Uses Kahn's algorithm for topological sorting.

    Args:
        stages: List of pipeline stages

    Returns:
        Stages sorted in execution order

    Raises:
        ValueError: If cycle detected in dependencies
    """
    # Build stage map and in-degree
    stage_map = {s.id: s for s in stages}
    in_degree = {s.id: 0 for s in stages}
    adjacency: Dict[str, List[str]] = {s.id: [] for s in stages}

    # Build graph
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in stage_map:
                raise ValueError(f"Stage {stage.id} depends on unknown stage {dep}")
            adjacency[dep].append(stage.id)
            in_degree[stage.id] += 1

    # Kahn's algorithm
    queue = [sid for sid in in_degree if in_degree[sid] == 0]
    sorted_stages = []

    while queue:
        # Sort queue for deterministic ordering
        queue.sort()
        stage_id = queue.pop(0)
        sorted_stages.append(stage_map[stage_id])

        # Reduce in-degree for dependents
        for dependent_id in adjacency[stage_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)

    # Check for cycle
    if len(sorted_stages) != len(stages):
        raise ValueError("Circular dependency detected in pipeline stages")

    return sorted_stages


def build_parallel_groups(stages: List[PipelineStage]) -> List[List[str]]:
    """
    Group stages that can be executed in parallel

    Args:
        stages: List of pipeline stages (topologically sorted)

    Returns:
        List of stage ID groups that can run in parallel
    """
    # Build dependency map
    stage_map = {s.id: s for s in stages}
    dependencies = {s.id: set(s.depends_on) for s in stages}

    groups: List[List[str]] = []
    completed: Set[str] = set()

    while len(completed) < len(stages):
        # Find stages that can run (all dependencies met)
        ready = [
            sid for sid in stage_map
            if sid not in completed and dependencies[sid].issubset(completed)
        ]

        if not ready:
            break  # No progress possible (shouldn't happen with valid DAG)

        groups.append(ready)
        completed.update(ready)

    return groups


def format_pipeline_duration(seconds: float) -> str:
    """
    Format pipeline duration in human-readable form

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 34m 56s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {secs}s"


def parse_stage_outputs(template: str, outputs: Dict[str, Dict[str, Any]]) -> str:
    """
    Parse stage output template and replace variables

    Example: "{extract_frames.output_dir}" -> "/path/to/frames"

    Args:
        template: Template string with {stage_id.output_key} placeholders
        outputs: Stage outputs dictionary

    Returns:
        Resolved template string
    """
    import re

    # Find all {stage_id.key} patterns
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, template)

    result = template
    for match in matches:
        parts = match.split(".")
        if len(parts) == 2:
            stage_id, key = parts
            if stage_id in outputs and key in outputs[stage_id]:
                value = str(outputs[stage_id][key])
                result = result.replace(f"{{{match}}}", value)

    return result


def calculate_stage_eta(stage: PipelineStage, avg_duration_per_item: float, total_items: int, processed_items: int) -> float:
    """
    Calculate estimated time remaining for stage

    Args:
        stage: Pipeline stage
        avg_duration_per_item: Average duration per item in seconds
        total_items: Total number of items to process
        processed_items: Number of items already processed

    Returns:
        Estimated time remaining in seconds
    """
    if processed_items >= total_items:
        return 0.0

    remaining_items = total_items - processed_items
    return remaining_items * avg_duration_per_item


def merge_stage_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge stage configurations (override takes precedence)

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    result = base_config.copy()
    result.update(override_config)
    return result
