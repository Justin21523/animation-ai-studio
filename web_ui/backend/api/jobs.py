"""
Jobs API: Job submission, monitoring, and management endpoints.

Provides REST API for job CRUD operations and SSE endpoint for real-time progress streaming.
"""
import asyncio
import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..db.models import JobSubmission, JobInfo, JobOutput
from ..services.job_service import JobService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# Dependency injection for JobService
# This will be set by main.py
_job_service: Optional[JobService] = None


def set_job_service(service: JobService):
    """Set the global job service instance."""
    global _job_service
    _job_service = service


def get_job_service() -> JobService:
    """Get job service dependency."""
    if _job_service is None:
        raise RuntimeError("JobService not initialized")
    return _job_service


@router.post("", response_model=dict, status_code=201)
async def submit_job(
    submission: JobSubmission,
    service: JobService = Depends(get_job_service)
):
    """
    Submit a new processing job.

    Args:
        submission: Job submission details

    Returns:
        {"job_id": "uuid", "status": "pending"}

    Raises:
        400: Invalid input (file not found, invalid pipeline type)
        500: Internal server error
    """
    try:
        job_id = await service.submit_job(submission)

        logger.info(f"Job submitted: {job_id} ({submission.film_name})")

        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"Job submitted successfully: {submission.film_name}"
        }

    except ValueError as e:
        logger.error(f"Invalid job submission: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Error submitting job")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("/{job_id}", response_model=JobInfo)
async def get_job(
    job_id: str,
    service: JobService = Depends(get_job_service)
):
    """
    Get job information by ID.

    Args:
        job_id: Job identifier

    Returns:
        JobInfo with current status and progress

    Raises:
        404: Job not found
    """
    job = await service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job


@router.get("", response_model=dict)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: JobService = Depends(get_job_service)
):
    """
    List jobs with optional filtering and pagination.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum number of jobs to return (1-200)
        offset: Pagination offset

    Returns:
        Paginated job list response.
    """
    jobs = await service.list_jobs(status=status, limit=limit, offset=offset)
    total = await service.db.count_jobs(status=status)
    return {
        "jobs": jobs,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/{job_id}", response_model=dict)
async def cancel_job(
    job_id: str,
    service: JobService = Depends(get_job_service)
):
    """
    Cancel a running job.

    Args:
        job_id: Job identifier

    Returns:
        {"success": true, "message": "..."}

    Raises:
        404: Job not found
        400: Job not running
    """
    # Check if job exists
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Attempt cancellation
    success = await service.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not running (current status: {job.status})"
        )

    logger.info(f"Job cancelled: {job_id}")

    return {
        "success": True,
        "message": f"Job {job_id} cancelled successfully"
    }


@router.get("/{job_id}/outputs", response_model=List[JobOutput])
async def get_job_outputs(
    job_id: str,
    service: JobService = Depends(get_job_service)
):
    """
    Get all outputs for a completed job.

    Args:
        job_id: Job identifier

    Returns:
        List of JobOutput objects

    Raises:
        404: Job not found
    """
    # Verify job exists
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    outputs = await service.db.get_job_outputs(job_id)
    return outputs


@router.get("/{job_id}/progress")
async def stream_job_progress(
    job_id: str,
    request: Request,
    service: JobService = Depends(get_job_service)
):
    """
    Stream real-time job progress via Server-Sent Events (SSE).

    Args:
        job_id: Job identifier

    Returns:
        SSE stream with progress events

    Event Types:
        - progress: {"type": "progress", "progress": 0.5, "stage": "2/3", "message": "..."}
        - stage_complete: {"type": "stage_complete", "message": "..."}
        - gpu_metrics: {"type": "gpu_metrics", "vram_used_mb": 8500, "vram_total_mb": 16384}
        - completed: {"type": "completed", "job_id": "..."}
        - failed: {"type": "failed", "error": "..."}

    Raises:
        404: Job not found
    """
    # Verify job exists
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_generator():
        """Generate SSE events for job progress."""

        # Create event queue for this connection
        event_queue = asyncio.Queue()

        # Progress callback that pushes events to queue
        async def progress_callback(event: dict):
            await event_queue.put(event)

        # Subscribe to job progress
        service.subscribe_progress(job_id, progress_callback)

        try:
            # Send initial status
            yield {
                "event": "status",
                "data": json.dumps({
                    "type": "status",
                    "job_id": job_id,
                    "status": job.status,
                    "progress": job.progress_percent
                })
            }

            # Stream progress events
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from job {job_id} progress stream")
                    break

                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)

                    event_type = event.get('type', 'progress')

                    # Yield SSE event
                    yield {
                        "event": event_type,
                        "data": json.dumps(event)
                    }

                    # Check for terminal events
                    if event_type in ('completed', 'failed', 'cancelled'):
                        logger.info(f"Job {job_id} reached terminal state: {event_type}")
                        break

                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield {"comment": "keepalive"}
                    continue

        finally:
            # Unsubscribe when connection closes
            service.unsubscribe_progress(job_id, progress_callback)
            logger.info(f"Unsubscribed from job {job_id} progress")

    return EventSourceResponse(event_generator())


@router.get("/stats", response_model=dict)
async def get_job_statistics(
    service: JobService = Depends(get_job_service)
):
    """
    Get job statistics and analytics.

    Returns:
        Statistics including:
        - Total jobs
        - Jobs by status
        - Completion rate
        - Average processing time
        - Pipeline type distribution
        - Recent activity

    Example:
        GET /api/jobs/stats
    """
    # Get all jobs
    all_jobs = await service.list_jobs(limit=10000)  # Get all jobs

    total_jobs = len(all_jobs)

    # Count jobs by status
    status_counts = {}
    pipeline_counts = {}
    completed_times = []

    for job in all_jobs:
        # Count by status
        status_counts[job.status] = status_counts.get(job.status, 0) + 1

        # Count by pipeline type
        pipeline_counts[job.pipeline_type] = pipeline_counts.get(job.pipeline_type, 0) + 1

        # Calculate processing time for completed jobs
        if job.status == "completed" and job.started_at and job.completed_at:
            completed_times.append(float(job.completed_at) - float(job.started_at))

    # Calculate completion rate
    completed_count = status_counts.get("completed", 0)
    failed_count = status_counts.get("failed", 0)
    total_finished = completed_count + failed_count

    completion_rate = (completed_count / total_finished * 100) if total_finished > 0 else 0

    # Calculate average processing time
    avg_time_seconds = sum(completed_times) / len(completed_times) if completed_times else 0
    avg_time_minutes = round(avg_time_seconds / 60, 2)

    # Get recent jobs (last 10)
    recent_jobs = all_jobs[:10]

    return {
        "total_jobs": total_jobs,
        "by_status": status_counts,
        "by_pipeline": pipeline_counts,
        "completion_rate_percent": round(completion_rate, 2),
        "average_processing_time_minutes": avg_time_minutes,
        "recent_activity": [
            {
                "job_id": job.job_id,
                "film_name": job.film_name,
                "status": job.status,
                "created_at": job.created_at
            }
            for job in recent_jobs
        ]
    }


@router.get("/{job_id}/logs", response_model=dict)
async def get_job_logs(
    job_id: str,
    tail: int = Query(100, ge=1, le=1000, description="Number of lines to return"),
    service: JobService = Depends(get_job_service)
):
    """
    Get job execution logs.

    Args:
        job_id: Job identifier
        tail: Number of lines to return (default: 100)

    Returns:
        {"stdout": [...], "stderr": [...]}

    Raises:
        404: Job not found
        501: Not implemented (logs stored in database in future)

    Note:
        Currently returns placeholder. In future, logs will be stored
        in database or file system for retrieval.
    """
    # Verify job exists
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    def _tail_lines(path: Path, max_lines: int) -> List[str]:
        if not path.exists():
            return []
        out: deque[str] = deque(maxlen=max_lines)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    out.append(line.rstrip("\n"))
        except Exception:
            return []
        return list(out)

    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    log_dir = Path(job.output_base_dir) / "webui_logs" / job_id
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    return {
        "job_id": job_id,
        "log_dir": str(log_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout": _tail_lines(stdout_path, tail),
        "stderr": _tail_lines(stderr_path, tail),
    }
