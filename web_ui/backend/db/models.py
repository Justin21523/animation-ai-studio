"""
Pydantic models for Web UI API.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class JobSubmission(BaseModel):
    """Job submission request."""
    film_name: str = Field(..., description="Name of the film")
    pipeline_type: str = Field(
        "cpu_only",
        description="Pipeline type: cpu_only, cpu_gpu_full, custom, action_*, or creative_*"
    )
    input_video_path: str = Field(..., description="Path to input video file")
    output_base_dir: str = Field(..., description="Base directory for outputs")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional options (workers, enable_gpu, sam2_model, etc.)"
    )


class JobInfo(BaseModel):
    """Job information response."""
    job_id: str
    workflow_id: Optional[str] = None
    film_name: str
    pipeline_type: str
    status: str
    progress_percent: float = 0.0
    current_stage: Optional[str] = None
    input_video_path: str
    output_base_dir: str
    options: Optional[Dict[str, Any]] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


class JobOutput(BaseModel):
    """Job output information."""
    stage: str
    output_type: str
    path: str
    file_count: Optional[int] = None
    size_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ProgressEvent(BaseModel):
    """Progress event for SSE streaming."""
    job_id: str
    progress: float
    stage: Optional[str] = None
    message: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class SystemMetrics(BaseModel):
    """System resource metrics."""
    timestamp: float
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    gpu_percent: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
    gpu_temperature_c: Optional[int] = None
    disk_usage_percent: Optional[float] = None
