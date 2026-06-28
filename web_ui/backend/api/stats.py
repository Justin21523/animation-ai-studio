"""Portfolio demo statistics API."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..services.job_service import JobService
from .jobs import get_job_service


router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/summary")
async def get_stats_summary(
    time_range: str = Query("7d"),
    service: JobService = Depends(get_job_service),
):
    """Return a compact job summary for portfolio dashboards."""
    jobs = await service.list_jobs(limit=10000)
    by_status = Counter(job.status for job in jobs)
    by_pipeline = Counter(job.pipeline_type for job in jobs)
    completed = by_status.get("completed", 0)
    failed = by_status.get("failed", 0)
    finished = completed + failed
    success_rate = round((completed / finished) * 100, 1) if finished else 0.0
    return {
        "time_range": time_range,
        "total_jobs": len(jobs),
        "by_status": dict(by_status),
        "by_pipeline": dict(by_pipeline),
        "success_rate_percent": success_rate,
        "recent_jobs": [job.model_dump() for job in jobs[:8]],
    }


@router.get("/charts")
async def get_stats_chart_data(
    type: str = Query("jobs"),
    time_range: str = Query("7d"),
    service: JobService = Depends(get_job_service),
):
    """Return simple chart-ready aggregates."""
    jobs = await service.list_jobs(limit=10000)
    if type == "success_rate":
        completed = sum(1 for job in jobs if job.status == "completed")
        failed = sum(1 for job in jobs if job.status == "failed")
        return {
            "type": type,
            "time_range": time_range,
            "labels": ["completed", "failed"],
            "values": [completed, failed],
        }

    counter: Counter[str]
    if type == "duration":
        values = [
            round((float(job.completed_at) - float(job.started_at)) / 60, 2)
            for job in jobs
            if job.started_at and job.completed_at
        ]
        return {
            "type": type,
            "time_range": time_range,
            "labels": [f"job-{index + 1}" for index in range(len(values))],
            "values": values,
        }

    counter = Counter(job.pipeline_type for job in jobs)
    return {
        "type": "jobs",
        "time_range": time_range,
        "labels": list(counter.keys()),
        "values": list(counter.values()),
    }
