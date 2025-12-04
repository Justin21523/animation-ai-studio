"""
Data Pipeline Automation - Package Exports

Provides DAG-based pipeline orchestration for multi-stage data processing.

Author: Animation AI Studio
Date: 2025-12-03
Version: 1.0.0
"""

from .common import (
    # Enums
    PipelineState,
    ExecutionStatus,
    StageType,
    ResourceType,

    # Configuration
    StageConfig,
    RetryPolicy,
    OrchestratorConfig,

    # Core Pipeline Types
    PipelineStage,
    Pipeline,
    StageResult,
    PipelineResult,
    CheckpointData,

    # Helper Functions
    generate_pipeline_id,
    generate_stage_hash,
    validate_dag,
    topological_sort,
    build_parallel_groups,
    format_pipeline_duration,
    parse_stage_outputs,
    calculate_stage_eta,
    merge_stage_configs,
)

__version__ = "1.0.0"
__author__ = "Animation AI Studio"

__all__ = [
    # Enums
    "PipelineState",
    "ExecutionStatus",
    "StageType",
    "ResourceType",

    # Configuration
    "StageConfig",
    "RetryPolicy",
    "OrchestratorConfig",

    # Core Pipeline Types
    "PipelineStage",
    "Pipeline",
    "StageResult",
    "PipelineResult",
    "CheckpointData",

    # Helper Functions
    "generate_pipeline_id",
    "generate_stage_hash",
    "validate_dag",
    "topological_sort",
    "build_parallel_groups",
    "format_pipeline_duration",
    "parse_stage_outputs",
    "calculate_stage_eta",
    "merge_stage_configs",
]
