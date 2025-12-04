"""
Batch Training Orchestrator - Resource Manager

Manages GPU discovery, allocation, and VRAM tracking.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import subprocess
import json
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from threading import Lock

from ..common import (
    GPUInfo,
    SystemResources,
    ResourceRequirements,
    validate_gpu_ids
)


logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Track GPU allocation to jobs"""
    gpu_id: int
    job_id: str
    allocated_memory: int  # MB
    allocated_at: float


class ResourceManager:
    """
    Manages system resources (GPUs, VRAM, CPU, RAM)

    Features:
    - GPU discovery via nvidia-smi or torch.cuda
    - VRAM usage monitoring
    - Resource allocation and locking
    - Conflict resolution
    - Resource release and cleanup
    """

    def __init__(self,
                 allowed_gpu_ids: Optional[List[int]] = None,
                 max_vram_per_gpu: int = 24576,
                 reserve_system_memory: int = 8192):
        """
        Initialize resource manager

        Args:
            allowed_gpu_ids: List of GPU IDs to use (None = all available)
            max_vram_per_gpu: Maximum VRAM to allocate per GPU (MB)
            reserve_system_memory: System RAM to reserve (MB)
        """
        self.allowed_gpu_ids = allowed_gpu_ids
        self.max_vram_per_gpu = max_vram_per_gpu
        self.reserve_system_memory = reserve_system_memory

        # GPU allocation tracking
        self.allocations: Dict[int, GPUAllocation] = {}
        self.allocation_lock = Lock()

        # Discover available GPUs
        self.available_gpus = self._discover_gpus()

        # Filter by allowed_gpu_ids
        if self.allowed_gpu_ids:
            self.available_gpus = [g for g in self.available_gpus if g.id in self.allowed_gpu_ids]

        logger.info(f"ResourceManager initialized with {len(self.available_gpus)} GPUs")

    def get_system_resources(self) -> SystemResources:
        """
        Get current system resource snapshot

        Returns:
            System resources including GPU, CPU, memory, disk
        """
        # Update GPU info
        gpus = self._discover_gpus()

        # Filter by allowed GPUs
        if self.allowed_gpu_ids:
            gpus = [g for g in gpus if g.id in self.allowed_gpu_ids]

        # Get CPU/memory info via psutil (if available)
        try:
            import psutil

            total_cpu_cores = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=1)
            available_cpu_cores = int(total_cpu_cores * (100 - cpu_percent) / 100)

            mem = psutil.virtual_memory()
            total_memory = mem.total // (1024 * 1024)  # MB
            available_memory = mem.available // (1024 * 1024)  # MB

            disk = psutil.disk_usage('/')
            total_disk = disk.total // (1024 * 1024)  # MB
            available_disk = disk.free // (1024 * 1024)  # MB

        except ImportError:
            logger.warning("psutil not available, using defaults for CPU/memory")
            total_cpu_cores = 8
            available_cpu_cores = 4
            total_memory = 65536  # 64GB
            available_memory = 32768  # 32GB
            total_disk = 1048576  # 1TB
            available_disk = 524288  # 512GB

        return SystemResources(
            gpus=gpus,
            total_cpu_cores=total_cpu_cores,
            available_cpu_cores=available_cpu_cores,
            total_memory=total_memory,
            available_memory=available_memory,
            total_disk=total_disk,
            available_disk=available_disk
        )

    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """
        Check if resources can be allocated

        Args:
            requirements: Resource requirements to check

        Returns:
            True if resources are available
        """
        with self.allocation_lock:
            # Check GPU count
            available_gpu_count = len([g for g in self.available_gpus if g.id not in self.allocations])

            if requirements.gpu_count > available_gpu_count:
                return False

            # Check VRAM requirements
            available_gpus_with_vram = []
            for gpu in self.available_gpus:
                if gpu.id in self.allocations:
                    continue

                if gpu.free_memory >= requirements.gpu_memory:
                    available_gpus_with_vram.append(gpu)

            if len(available_gpus_with_vram) < requirements.gpu_count:
                return False

            # Check system resources
            resources = self.get_system_resources()

            if requirements.cpu_cores > resources.available_cpu_cores:
                return False

            if requirements.system_memory > (resources.available_memory - self.reserve_system_memory):
                return False

            if requirements.disk_space > resources.available_disk:
                return False

            return True

    def allocate(self, job_id: str, requirements: ResourceRequirements) -> Optional[List[int]]:
        """
        Allocate resources for job

        Args:
            job_id: Job identifier
            requirements: Resource requirements

        Returns:
            List of allocated GPU IDs or None if allocation failed
        """
        with self.allocation_lock:
            # Check if resources available
            if not self.can_allocate(requirements):
                logger.warning(f"Cannot allocate resources for job {job_id}")
                return None

            # Find suitable GPUs
            allocated_gpus = []
            for gpu in self.available_gpus:
                if gpu.id in self.allocations:
                    continue

                if gpu.free_memory >= requirements.gpu_memory:
                    allocated_gpus.append(gpu.id)

                    # Create allocation record
                    import time
                    allocation = GPUAllocation(
                        gpu_id=gpu.id,
                        job_id=job_id,
                        allocated_memory=requirements.gpu_memory,
                        allocated_at=time.time()
                    )
                    self.allocations[gpu.id] = allocation

                    if len(allocated_gpus) >= requirements.gpu_count:
                        break

            if len(allocated_gpus) < requirements.gpu_count:
                # Rollback allocations
                for gpu_id in allocated_gpus:
                    del self.allocations[gpu_id]
                logger.error(f"Failed to allocate {requirements.gpu_count} GPUs for job {job_id}")
                return None

            logger.info(f"Allocated GPUs {allocated_gpus} to job {job_id}")
            return allocated_gpus

    def release(self, job_id: str):
        """
        Release resources allocated to job

        Args:
            job_id: Job identifier
        """
        with self.allocation_lock:
            # Find and remove allocations for this job
            released_gpus = []
            for gpu_id, allocation in list(self.allocations.items()):
                if allocation.job_id == job_id:
                    del self.allocations[gpu_id]
                    released_gpus.append(gpu_id)

            if released_gpus:
                logger.info(f"Released GPUs {released_gpus} from job {job_id}")
            else:
                logger.warning(f"No GPU allocations found for job {job_id}")

    def get_allocation(self, job_id: str) -> List[int]:
        """
        Get GPU IDs allocated to job

        Args:
            job_id: Job identifier

        Returns:
            List of allocated GPU IDs
        """
        with self.allocation_lock:
            allocated_gpus = []
            for gpu_id, allocation in self.allocations.items():
                if allocation.job_id == job_id:
                    allocated_gpus.append(gpu_id)
            return allocated_gpus

    def get_available_gpus(self) -> List[GPUInfo]:
        """
        Get list of available (unallocated) GPUs

        Returns:
            List of available GPU info
        """
        with self.allocation_lock:
            gpus = self._discover_gpus()

            # Filter by allowed GPUs
            if self.allowed_gpu_ids:
                gpus = [g for g in gpus if g.id in self.allowed_gpu_ids]

            # Filter out allocated GPUs
            available = [g for g in gpus if g.id not in self.allocations]

            return available

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _discover_gpus(self) -> List[GPUInfo]:
        """
        Discover available GPUs using nvidia-smi or torch.cuda

        Returns:
            List of GPU info
        """
        # Try nvidia-smi first
        try:
            return self._discover_gpus_nvidia_smi()
        except Exception as e:
            logger.warning(f"nvidia-smi discovery failed: {e}")

        # Fallback to torch.cuda
        try:
            return self._discover_gpus_torch()
        except Exception as e:
            logger.warning(f"torch.cuda discovery failed: {e}")

        # No GPUs available
        logger.error("GPU discovery failed, no GPUs available")
        return []

    def _discover_gpus_nvidia_smi(self) -> List[GPUInfo]:
        """
        Discover GPUs using nvidia-smi

        Returns:
            List of GPU info
        """
        query_cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(query_cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')

        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]

            if len(parts) < 5:
                continue

            gpu_id = int(parts[0])
            name = parts[1]
            total_memory = int(float(parts[2]))
            free_memory = int(float(parts[3]))
            utilization = float(parts[4])
            temperature = float(parts[5]) if len(parts) > 5 and parts[5] != 'N/A' else None
            power_usage = float(parts[6]) if len(parts) > 6 and parts[6] != 'N/A' else None

            gpu = GPUInfo(
                id=gpu_id,
                name=name,
                total_memory=total_memory,
                free_memory=free_memory,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                is_available=True
            )
            gpus.append(gpu)

        return gpus

    def _discover_gpus_torch(self) -> List[GPUInfo]:
        """
        Discover GPUs using torch.cuda

        Returns:
            List of GPU info
        """
        import torch

        if not torch.cuda.is_available():
            return []

        gpus = []
        for gpu_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_id)
            name = props.name
            total_memory = props.total_memory // (1024 * 1024)  # MB

            # Get free memory (approximate)
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            free_memory = (props.total_memory - torch.cuda.memory_allocated()) // (1024 * 1024)

            gpu = GPUInfo(
                id=gpu_id,
                name=name,
                total_memory=total_memory,
                free_memory=free_memory,
                utilization=0.0,  # Not available via torch
                temperature=None,
                power_usage=None,
                is_available=True
            )
            gpus.append(gpu)

        return gpus
