"""
System Monitoring API - Monitor system resources and GPU status.

Provides endpoints for:
- CPU and RAM metrics
- GPU information and utilization
- Disk usage statistics
"""
import os
import subprocess
import asyncio
import json
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import psutil


# Pydantic models
class CPUMetrics(BaseModel):
    """CPU metrics."""
    percent: float
    count: int
    per_cpu: List[float]
    frequency_mhz: Optional[float] = None


class MemoryMetrics(BaseModel):
    """Memory metrics."""
    total_mb: int
    used_mb: int
    available_mb: int
    percent: float


class DiskMetrics(BaseModel):
    """Disk metrics."""
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float


class GPUInfo(BaseModel):
    """GPU information."""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    memory_percent: float
    gpu_utilization_percent: int
    temperature_c: Optional[int] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None


class SystemMetrics(BaseModel):
    """Complete system metrics."""
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: List[DiskMetrics]
    gpu: Optional[List[GPUInfo]] = None
    timestamp: str


# Create router
router = APIRouter(prefix="/api/system", tags=["system"])
METRICS_HISTORY = deque(maxlen=300)


def get_cpu_metrics() -> CPUMetrics:
    """Get CPU metrics using psutil."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count()
    per_cpu = psutil.cpu_percent(interval=0.5, percpu=True)

    freq = psutil.cpu_freq()
    frequency = freq.current if freq else None

    return CPUMetrics(
        percent=cpu_percent,
        count=cpu_count,
        per_cpu=per_cpu,
        frequency_mhz=frequency
    )


def get_memory_metrics() -> MemoryMetrics:
    """Get memory metrics using psutil."""
    mem = psutil.virtual_memory()

    return MemoryMetrics(
        total_mb=int(mem.total / (1024 * 1024)),
        used_mb=int(mem.used / (1024 * 1024)),
        available_mb=int(mem.available / (1024 * 1024)),
        percent=mem.percent
    )


def get_disk_metrics(paths: List[str] = None) -> List[DiskMetrics]:
    """
    Get disk usage metrics for specified paths.

    Args:
        paths: List of paths to check (default: ['/mnt/data', '/'])

    Returns:
        List of DiskMetrics
    """
    if paths is None:
        paths = ['/mnt/data', '/']

    disk_metrics = []

    for path in paths:
        if not os.path.exists(path):
            continue

        try:
            usage = psutil.disk_usage(path)
            disk_metrics.append(DiskMetrics(
                path=path,
                total_gb=round(usage.total / (1024**3), 2),
                used_gb=round(usage.used / (1024**3), 2),
                free_gb=round(usage.free / (1024**3), 2),
                percent=usage.percent
            ))
        except (PermissionError, OSError):
            continue

    return disk_metrics


def get_gpu_info() -> Optional[List[GPUInfo]]:
    """
    Get GPU information using nvidia-smi.

    Returns:
        List of GPUInfo objects, or None if no GPU or nvidia-smi not available
    """
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 9:
                continue

            try:
                index = int(parts[0])
                name = parts[1]
                memory_total = int(float(parts[2]))
                memory_used = int(float(parts[3]))
                memory_free = int(float(parts[4]))
                gpu_util = int(float(parts[5]))
                temp = int(float(parts[6])) if parts[6] not in ['N/A', ''] else None
                power_draw = float(parts[7]) if parts[7] not in ['N/A', ''] else None
                power_limit = float(parts[8]) if parts[8] not in ['N/A', ''] else None

                memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0

                gpus.append(GPUInfo(
                    index=index,
                    name=name,
                    memory_total_mb=memory_total,
                    memory_used_mb=memory_used,
                    memory_free_mb=memory_free,
                    memory_percent=round(memory_percent, 2),
                    gpu_utilization_percent=gpu_util,
                    temperature_c=temp,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit
                ))
            except (ValueError, IndexError):
                continue

        return gpus if gpus else None

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return None


def build_system_metrics_payload() -> Dict[str, Any]:
    """Build nested and frontend-compatible flat metrics."""
    cpu = get_cpu_metrics()
    memory = get_memory_metrics()
    disk = get_disk_metrics()
    gpu = get_gpu_info()
    first_disk = disk[0] if disk else None
    first_gpu = gpu[0] if gpu else None

    payload: Dict[str, Any] = {
        "cpu": cpu.model_dump(),
        "memory": memory.model_dump(),
        "disk": [item.model_dump() for item in disk],
        "gpu": [item.model_dump() for item in gpu] if gpu else None,
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu.percent,
        "cpu_count": cpu.count,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used_mb / 1024, 2),
        "ram_total_gb": round(memory.total_mb / 1024, 2),
        "gpu_available": bool(first_gpu),
        "disk_percent": first_disk.percent if first_disk else 0,
        "disk_used_gb": first_disk.used_gb if first_disk else 0,
        "disk_total_gb": first_disk.total_gb if first_disk else 0,
    }
    if first_gpu:
        payload.update(
            {
                "gpu_name": first_gpu.name,
                "gpu_percent": first_gpu.gpu_utilization_percent,
                "gpu_memory_used_mb": first_gpu.memory_used_mb,
                "gpu_memory_total_mb": first_gpu.memory_total_mb,
                "gpu_temperature_c": first_gpu.temperature_c,
            }
        )
    METRICS_HISTORY.append(payload)
    return payload


@router.get("/metrics", response_model=dict)
async def get_system_metrics():
    """
    Get current system metrics (CPU, RAM, Disk, GPU).

    Returns:
        SystemMetrics with current resource usage

    Example:
        GET /api/system/metrics
    """
    return build_system_metrics_payload()


@router.get("/metrics/history")
async def get_system_metrics_history(limit: int = Query(100, ge=1, le=300)):
    """Get recent in-memory metric snapshots for dashboard charts."""
    if not METRICS_HISTORY:
        build_system_metrics_payload()
    return list(METRICS_HISTORY)[-limit:]


@router.get("/metrics/stream")
async def stream_system_metrics(request: Request):
    """Stream system metrics as SSE events for the dashboard."""

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            payload = build_system_metrics_payload()
            yield {
                "event": "metrics",
                "data": json.dumps(payload),
            }
            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@router.get("/gpu", response_model=List[GPUInfo])
async def get_gpu_status():
    """
    Get GPU information and utilization.

    Returns:
        List of GPUInfo objects

    Example:
        GET /api/system/gpu

    Raises:
        404: If no GPU is available or nvidia-smi is not installed
    """
    gpus = get_gpu_info()

    if gpus is None:
        raise HTTPException(
            status_code=404,
            detail="No GPU available or nvidia-smi not found"
        )

    return gpus


@router.get("/cpu")
async def get_cpu_status():
    """
    Get detailed CPU information.

    Returns:
        CPUMetrics with CPU usage details

    Example:
        GET /api/system/cpu
    """
    return get_cpu_metrics()


@router.get("/memory")
async def get_memory_status():
    """
    Get detailed memory information.

    Returns:
        MemoryMetrics with RAM usage details

    Example:
        GET /api/system/memory
    """
    return get_memory_metrics()


@router.get("/disk")
async def get_disk_status(
    paths: Optional[str] = None
):
    """
    Get disk usage information.

    Query Parameters:
        paths: Comma-separated list of paths to check (default: /mnt/data,/)

    Returns:
        List of DiskMetrics

    Example:
        GET /api/system/disk?paths=/mnt/data,/
    """
    path_list = paths.split(',') if paths else None
    return get_disk_metrics(path_list)


@router.get("/processes")
async def get_process_info(
    limit: int = 10,
    sort_by: str = "memory"
):
    """
    Get top processes by CPU or memory usage.

    Query Parameters:
        limit: Number of processes to return (default: 10)
        sort_by: Sort by 'cpu' or 'memory' (default: memory)

    Returns:
        List of top processes

    Example:
        GET /api/system/processes?limit=10&sort_by=memory
    """
    processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
        try:
            pinfo = proc.info
            processes.append({
                'pid': pinfo['pid'],
                'name': pinfo['name'],
                'cpu_percent': pinfo['cpu_percent'] or 0,
                'memory_percent': pinfo['memory_percent'] or 0,
                'memory_mb': int(pinfo['memory_info'].rss / (1024 * 1024)) if pinfo.get('memory_info') else 0
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort processes
    if sort_by == "cpu":
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
    else:
        processes.sort(key=lambda x: x['memory_percent'], reverse=True)

    return {
        "total_processes": len(processes),
        "sort_by": sort_by,
        "top_processes": processes[:limit]
    }
