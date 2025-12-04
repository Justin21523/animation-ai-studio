"""
Batch Training Orchestrator Scenario

Distributed batch training orchestrator for LoRA jobs with GPU resource management,
job scheduling, progress monitoring, and automatic checkpoint evaluation.

Features:
- Multi-GPU job scheduling and execution
- Resource allocation and conflict resolution
- Progress monitoring and logging
- Automatic checkpoint evaluation
- Job queue management (priority, dependencies)
- Failure recovery and retry logic

Author: Animation AI Studio
Date: 2025-12-03
Version: 1.0.0
"""

from .common import (
    # Enums
    JobState,
    SchedulingStrategy,
    ResourceType,
    JobPriority,

    # Resource Dataclasses
    GPUInfo,
    SystemResources,
    ResourceRequirements,

    # Job Dataclasses
    TrainingJob,
    JobResult,

    # Configuration
    SchedulerConfig,
    MonitorConfig,
    ResourceConfig,
    OrchestratorConfig,

    # Statistics
    JobStatistics,

    # Helper Functions
    generate_job_id,
    validate_gpu_ids,
    format_duration,
    format_memory
)

__version__ = "1.0.0"

__all__ = [
    # Enums
    "JobState",
    "SchedulingStrategy",
    "ResourceType",
    "JobPriority",

    # Resource Dataclasses
    "GPUInfo",
    "SystemResources",
    "ResourceRequirements",

    # Job Dataclasses
    "TrainingJob",
    "JobResult",

    # Configuration
    "SchedulerConfig",
    "MonitorConfig",
    "ResourceConfig",
    "OrchestratorConfig",

    # Statistics
    "JobStatistics",

    # Helper Functions
    "generate_job_id",
    "validate_gpu_ids",
    "format_duration",
    "format_memory"
]
