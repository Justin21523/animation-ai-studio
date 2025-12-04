"""
Batch Training Orchestrator - Progress Monitor

Monitors training job progress by parsing logs and extracting metrics.

Author: Animation AI Studio
Date: 2025-12-03
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Thread, Event
from collections import deque

from ..common import TrainingJob


logger = logging.getLogger(__name__)


class ProgressMonitor:
    """
    Monitors training progress via log parsing

    Features:
    - Real-time log parsing
    - Metrics extraction (epoch, loss, learning rate, ETA)
    - Event emission for monitoring systems
    - Progress aggregation for dashboard
    """

    def __init__(self, interval: int = 5):
        """
        Initialize progress monitor

        Args:
            interval: Log parsing interval in seconds
        """
        self.interval = interval

        # Metrics storage (job_id -> metrics)
        self.current_metrics: Dict[str, Dict[str, Any]] = {}

        # Monitoring thread
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.is_running = False

        # Jobs being monitored (job_id -> log_file)
        self.monitored_jobs: Dict[str, Path] = {}
        self.log_positions: Dict[str, int] = {}  # Track file read position

        logger.info(f"ProgressMonitor initialized (interval={interval}s)")

    def start(self):
        """Start background monitoring"""
        if self.is_running:
            logger.warning("Progress monitor already running")
            return

        self.stop_event.clear()
        self.is_running = True

        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Progress monitoring started")

    def stop(self):
        """Stop background monitoring"""
        if not self.is_running:
            return

        self.stop_event.set()
        self.is_running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("Progress monitoring stopped")

    def add_job(self, job: TrainingJob):
        """
        Add job to monitoring

        Args:
            job: Training job to monitor
        """
        self.monitored_jobs[job.id] = job.log_file
        self.log_positions[job.id] = 0

        # Initialize metrics
        self.current_metrics[job.id] = {
            "job_id": job.id,
            "job_name": job.name,
            "current_epoch": 0,
            "total_epochs": None,
            "current_step": 0,
            "total_steps": None,
            "loss": None,
            "learning_rate": None,
            "eta": None,
            "progress_percent": 0.0
        }

        logger.info(f"Added job {job.id} to progress monitoring")

    def remove_job(self, job_id: str):
        """
        Remove job from monitoring

        Args:
            job_id: Job identifier
        """
        if job_id in self.monitored_jobs:
            del self.monitored_jobs[job_id]
            del self.log_positions[job_id]
            logger.info(f"Removed job {job_id} from progress monitoring")

    def get_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current metrics for job

        Args:
            job_id: Job identifier

        Returns:
            Metrics dictionary or None if job not found
        """
        return self.current_metrics.get(job_id)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all monitored jobs

        Returns:
            Dictionary mapping job_id to metrics
        """
        return self.current_metrics.copy()

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Parse logs for all monitored jobs
                for job_id, log_file in list(self.monitored_jobs.items()):
                    if not log_file.exists():
                        continue

                    # Read new log lines
                    new_lines = self._read_new_lines(job_id, log_file)

                    if new_lines:
                        # Parse metrics from new lines
                        metrics = self._parse_metrics(new_lines)

                        # Update current metrics
                        if metrics:
                            self.current_metrics[job_id].update(metrics)

            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")

            # Wait for next interval
            self.stop_event.wait(self.interval)

    def _read_new_lines(self, job_id: str, log_file: Path) -> list:
        """
        Read new lines from log file since last check

        Args:
            job_id: Job identifier
            log_file: Log file path

        Returns:
            List of new log lines
        """
        try:
            with open(log_file, 'r', errors='ignore') as f:
                # Seek to last position
                last_position = self.log_positions.get(job_id, 0)
                f.seek(last_position)

                # Read new lines
                new_lines = f.readlines()

                # Update position
                self.log_positions[job_id] = f.tell()

                return new_lines

        except Exception as e:
            logger.error(f"Failed to read log file {log_file}: {e}")
            return []

    def _parse_metrics(self, lines: list) -> Dict[str, Any]:
        """
        Parse metrics from log lines

        Args:
            lines: Log lines

        Returns:
            Extracted metrics
        """
        metrics = {}

        for line in lines:
            # Parse epoch progress (Kohya_ss format)
            # Example: "epoch 3/10"
            epoch_match = re.search(r'epoch\s+(\d+)/(\d+)', line, re.IGNORECASE)
            if epoch_match:
                metrics["current_epoch"] = int(epoch_match.group(1))
                metrics["total_epochs"] = int(epoch_match.group(2))

            # Parse step progress
            # Example: "step 150/500"
            step_match = re.search(r'step\s+(\d+)/(\d+)', line, re.IGNORECASE)
            if step_match:
                metrics["current_step"] = int(step_match.group(1))
                metrics["total_steps"] = int(step_match.group(2))

            # Parse loss
            # Example: "loss: 0.0523" or "loss=0.0523"
            loss_match = re.search(r'loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
            if loss_match:
                metrics["loss"] = float(loss_match.group(1))

            # Parse learning rate
            # Example: "lr: 1e-4" or "lr=0.0001"
            lr_match = re.search(r'lr[:\s=]+([0-9.e-]+)', line, re.IGNORECASE)
            if lr_match:
                metrics["learning_rate"] = float(lr_match.group(1))

            # Parse ETA
            # Example: "ETA: 2h 34m" or "eta: 1:23:45"
            eta_match = re.search(r'eta[:\s]+([0-9hms:]+)', line, re.IGNORECASE)
            if eta_match:
                metrics["eta"] = eta_match.group(1)

        # Calculate progress percentage
        if "current_epoch" in metrics and "total_epochs" in metrics:
            if metrics["total_epochs"] > 0:
                metrics["progress_percent"] = (metrics["current_epoch"] / metrics["total_epochs"]) * 100

        elif "current_step" in metrics and "total_steps" in metrics:
            if metrics["total_steps"] > 0:
                metrics["progress_percent"] = (metrics["current_step"] / metrics["total_steps"]) * 100

        return metrics
