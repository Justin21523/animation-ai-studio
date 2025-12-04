"""
Batch Training Orchestrator - Common Types and Data Structures

Shared enums, dataclasses, and configuration for distributed training orchestration.

Author: Animation AI Studio
Date: 2025-12-03
"""

import hashlib
import time
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ============================================================================
# Enums
# ============================================================================

class JobState(Enum):
    """Training job states"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class SchedulingStrategy(Enum):
    """Job scheduling strategies"""
    FIFO = "fifo"  # First-In-First-Out
    PRIORITY = "priority"  # Priority-based
    FAIR_SHARE = "fair_share"  # Equal resource distribution
    SHORTEST_JOB_FIRST = "shortest_job_first"  # Estimated duration


class ResourceType(Enum):
    """Resource types"""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


# ============================================================================
# Resource Dataclasses
# ============================================================================

@dataclass
class GPUInfo:
    """GPU device information"""
    id: int
    name: str
    total_memory: int  # MB
    free_memory: int  # MB
    utilization: float  # 0-100%
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    is_available: bool = True


@dataclass
class SystemResources:
    """System resource snapshot"""
    gpus: List[GPUInfo]
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory: int  # MB
    available_memory: int  # MB
    total_disk: int  # MB
    available_disk: int  # MB
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceRequirements:
    """Job resource requirements"""
    gpu_count: int = 1
    gpu_memory: int = 16384  # MB (16GB default)
    cpu_cores: int = 4
    system_memory: int = 32768  # MB (32GB default)
    disk_space: int = 102400  # MB (100GB default)
    estimated_duration: Optional[int] = None  # seconds

    def __post_init__(self):
        """Validate resource requirements"""
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be non-negative")
        if self.gpu_memory < 0:
            raise ValueError("gpu_memory must be non-negative")
        if self.cpu_cores < 0:
            raise ValueError("cpu_cores must be non-negative")


# ============================================================================
# Job Dataclasses
# ============================================================================

@dataclass
class TrainingJob:
    """Training job definition"""
    id: str
    name: str
    config_path: Path
    output_dir: Path
    log_file: Path

    # Scheduling
    priority: int = JobPriority.NORMAL.value
    dependencies: List[str] = field(default_factory=list)

    # Resources
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    allocated_gpus: List[int] = field(default_factory=list)

    # State
    state: JobState = JobState.PENDING
    created_at: float = field(default_factory=time.time)
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Execution
    retry_count: int = 3
    current_retry: int = 0
    timeout: Optional[int] = None  # seconds
    command: Optional[str] = None
    process_id: Optional[int] = None
    tmux_session: Optional[str] = None

    # Results
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    checkpoints: List[Path] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate job definition"""
        if not self.name:
            raise ValueError("Job name cannot be empty")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    @property
    def is_terminal(self) -> bool:
        """Check if job is in terminal state"""
        return self.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]

    @property
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self.state == JobState.RUNNING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "log_file": str(self.log_file),
            "priority": self.priority,
            "dependencies": self.dependencies,
            "resources": {
                "gpu_count": self.resources.gpu_count,
                "gpu_memory": self.resources.gpu_memory,
                "cpu_cores": self.resources.cpu_cores,
                "system_memory": self.resources.system_memory,
                "disk_space": self.resources.disk_space,
                "estimated_duration": self.resources.estimated_duration
            },
            "allocated_gpus": self.allocated_gpus,
            "state": self.state.value,
            "created_at": self.created_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "current_retry": self.current_retry,
            "timeout": self.timeout,
            "command": self.command,
            "process_id": self.process_id,
            "tmux_session": self.tmux_session,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "checkpoints": [str(p) for p in self.checkpoints],
            "metadata": self.metadata,
            "tags": self.tags
        }


@dataclass
class JobResult:
    """Training job result"""
    job_id: str
    success: bool
    state: JobState
    duration: float
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    checkpoints: List[Path] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[Path] = None
    log_file: Optional[Path] = None


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    max_concurrent_jobs: int = 4
    check_interval: int = 10  # seconds
    enable_dependencies: bool = True
    enable_priorities: bool = True
    fairness_window: int = 3600  # seconds (1 hour)


@dataclass
class MonitorConfig:
    """Monitoring configuration"""
    log_parse_interval: int = 5  # seconds
    metrics_retention_days: int = 7
    enable_event_emission: bool = True
    enable_dashboard: bool = False
    dashboard_port: int = 8000


@dataclass
class ResourceConfig:
    """Resource management configuration"""
    gpu_ids: List[int] = field(default_factory=list)  # Empty = use all
    max_vram_per_gpu: int = 24576  # MB (24GB default)
    max_system_ram: int = 65536  # MB (64GB default)
    max_disk_space: int = 1048576  # MB (1TB default)
    enable_gpu_monitoring: bool = True
    monitoring_interval: int = 5  # seconds
    reserve_system_memory: int = 8192  # MB (8GB reserve)


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration"""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Storage
    job_db_path: Path = Path("jobs.db")
    log_dir: Path = Path("logs/training")
    checkpoint_dir: Path = Path("checkpoints")

    # Integration
    enable_orchestration_layer: bool = False
    enable_safety_system: bool = False

    def __post_init__(self):
        """Ensure directories exist"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Statistics Dataclasses
# ============================================================================

@dataclass
class JobStatistics:
    """Job queue statistics"""
    total_jobs: int
    pending_jobs: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int

    total_gpu_hours: float = 0.0
    average_job_duration: float = 0.0
    success_rate: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.completed_jobs + self.failed_jobs > 0:
            self.success_rate = self.completed_jobs / (self.completed_jobs + self.failed_jobs)


# ============================================================================
# Helper Functions
# ============================================================================

def generate_job_id(name: str, timestamp: Optional[float] = None) -> str:
    """
    Generate unique job ID

    Args:
        name: Job name
        timestamp: Optional timestamp (default: current time)

    Returns:
        Unique job ID (hash-based)
    """
    if timestamp is None:
        timestamp = time.time()

    content = f"{name}_{timestamp}".encode('utf-8')
    hash_digest = hashlib.sha256(content).hexdigest()
    return f"job_{hash_digest[:16]}"


def validate_gpu_ids(gpu_ids: List[int], available_gpus: List[GPUInfo]) -> bool:
    """
    Validate GPU IDs against available GPUs

    Args:
        gpu_ids: List of GPU IDs to validate
        available_gpus: List of available GPU info

    Returns:
        True if all GPU IDs are valid
    """
    available_ids = {gpu.id for gpu in available_gpus}
    return all(gpu_id in available_ids for gpu_id in gpu_ids)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 34m 56s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_memory(mb: int) -> str:
    """
    Format memory in human-readable form

    Args:
        mb: Memory in MB

    Returns:
        Formatted string (e.g., "16.5 GB")
    """
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"
