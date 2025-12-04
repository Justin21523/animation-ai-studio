"""
Batch Training Orchestrator - GPU Monitor

Real-time GPU metrics collection and monitoring.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import time
from typing import List, Dict, Optional
from threading import Thread, Event

from ..common import GPUInfo


logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    Monitors GPU metrics in real-time

    Features:
    - Periodic GPU metrics collection
    - Memory usage tracking
    - Temperature and power monitoring
    - Background monitoring thread
    - Metrics history retention
    """

    def __init__(self,
                 interval: int = 5,
                 max_history: int = 100):
        """
        Initialize GPU monitor

        Args:
            interval: Monitoring interval in seconds
            max_history: Maximum number of historical measurements to retain
        """
        self.interval = interval
        self.max_history = max_history

        # Metrics storage
        self.current_metrics: Dict[int, GPUInfo] = {}
        self.metrics_history: Dict[int, List[GPUInfo]] = {}

        # Monitoring thread
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.is_running = False

        logger.info(f"GPUMonitor initialized (interval={interval}s)")

    def start(self):
        """Start background monitoring"""
        if self.is_running:
            logger.warning("GPU monitor already running")
            return

        self.stop_event.clear()
        self.is_running = True

        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("GPU monitoring started")

    def stop(self):
        """Stop background monitoring"""
        if not self.is_running:
            return

        self.stop_event.set()
        self.is_running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("GPU monitoring stopped")

    def get_current_metrics(self, gpu_id: Optional[int] = None) -> Dict[int, GPUInfo]:
        """
        Get current GPU metrics

        Args:
            gpu_id: Specific GPU ID (None = all GPUs)

        Returns:
            Dictionary mapping GPU ID to current metrics
        """
        if gpu_id is not None:
            if gpu_id in self.current_metrics:
                return {gpu_id: self.current_metrics[gpu_id]}
            return {}

        return self.current_metrics.copy()

    def get_metrics_history(self, gpu_id: int, limit: Optional[int] = None) -> List[GPUInfo]:
        """
        Get historical metrics for GPU

        Args:
            gpu_id: GPU identifier
            limit: Maximum number of historical entries to return

        Returns:
            List of GPU metrics (newest first)
        """
        history = self.metrics_history.get(gpu_id, [])

        if limit is not None:
            history = history[:limit]

        return history

    def get_peak_usage(self, gpu_id: int) -> Optional[Dict]:
        """
        Get peak memory and utilization for GPU

        Args:
            gpu_id: GPU identifier

        Returns:
            Dictionary with peak_memory_used and peak_utilization
        """
        history = self.metrics_history.get(gpu_id, [])

        if not history:
            return None

        peak_memory_used = max(gpu.total_memory - gpu.free_memory for gpu in history)
        peak_utilization = max(gpu.utilization for gpu in history)

        return {
            "peak_memory_used": peak_memory_used,
            "peak_utilization": peak_utilization
        }

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()

                # Update current metrics
                self.current_metrics = {gpu.id: gpu for gpu in metrics}

                # Update history
                for gpu in metrics:
                    if gpu.id not in self.metrics_history:
                        self.metrics_history[gpu.id] = []

                    # Add to history (newest first)
                    self.metrics_history[gpu.id].insert(0, gpu)

                    # Trim history
                    if len(self.metrics_history[gpu.id]) > self.max_history:
                        self.metrics_history[gpu.id] = self.metrics_history[gpu.id][:self.max_history]

            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")

            # Wait for next interval
            self.stop_event.wait(self.interval)

    def _collect_metrics(self) -> List[GPUInfo]:
        """
        Collect current GPU metrics

        Returns:
            List of GPU info
        """
        # Try nvidia-smi
        try:
            return self._collect_nvidia_smi()
        except Exception as e:
            logger.debug(f"nvidia-smi collection failed: {e}")

        # Fallback to torch.cuda
        try:
            return self._collect_torch()
        except Exception as e:
            logger.debug(f"torch.cuda collection failed: {e}")

        return []

    def _collect_nvidia_smi(self) -> List[GPUInfo]:
        """
        Collect metrics via nvidia-smi

        Returns:
            List of GPU info
        """
        import subprocess

        query_cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(query_cmd, capture_output=True, text=True, check=True, timeout=10)
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

    def _collect_torch(self) -> List[GPUInfo]:
        """
        Collect metrics via torch.cuda

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

            # Get memory usage
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) // (1024 * 1024)
            free_memory = total_memory - allocated

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
