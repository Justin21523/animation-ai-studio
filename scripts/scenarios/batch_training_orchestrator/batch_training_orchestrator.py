"""
Batch Training Orchestrator - Main Orchestrator

Coordinates all components for distributed batch training orchestration.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import time
from pathlib import Path
from typing import Optional, List
from threading import Thread, Event

from .common import (
    TrainingJob,
    JobState,
    JobResult,
    OrchestratorConfig,
    JobStatistics,
    ResourceRequirements,
    generate_job_id
)
from .jobs import JobManager, JobExecutor
from .resources import ResourceManager, GPUMonitor
from .monitors import ProgressMonitor
from .schedulers import JobScheduler


logger = logging.getLogger(__name__)


class BatchTrainingOrchestrator:
    """
    Main orchestrator for distributed batch training

    Coordinates:
    - Job submission and management
    - Resource allocation
    - Job scheduling
    - Progress monitoring
    - Automatic execution
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize batch training orchestrator

        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()

        # Initialize components
        self.job_manager = JobManager(self.config.job_db_path)
        self.job_executor = JobExecutor(use_tmux=False)
        self.resource_manager = ResourceManager(
            allowed_gpu_ids=self.config.resources.gpu_ids,
            max_vram_per_gpu=self.config.resources.max_vram_per_gpu,
            reserve_system_memory=self.config.resources.reserve_system_memory
        )
        self.scheduler = JobScheduler(self.config.scheduler)
        self.progress_monitor = ProgressMonitor(self.config.monitor.log_parse_interval)
        self.gpu_monitor = GPUMonitor(self.config.resources.monitoring_interval)

        # Orchestration loop
        self.orchestration_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.is_running = False

        logger.info("BatchTrainingOrchestrator initialized")

    def start(self):
        """Start orchestrator (begins automatic job execution)"""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return

        self.stop_event.clear()
        self.is_running = True

        # Start monitors
        self.progress_monitor.start()
        if self.config.resources.enable_gpu_monitoring:
            self.gpu_monitor.start()

        # Start orchestration loop
        self.orchestration_thread = Thread(target=self._orchestration_loop, daemon=True)
        self.orchestration_thread.start()

        logger.info("Batch training orchestrator started")

    def stop(self):
        """Stop orchestrator"""
        if not self.is_running:
            return

        self.stop_event.set()
        self.is_running = False

        # Stop monitors
        self.progress_monitor.stop()
        self.gpu_monitor.stop()

        # Wait for orchestration thread
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=10)

        logger.info("Batch training orchestrator stopped")

    def submit_job(self, job: TrainingJob) -> str:
        """
        Submit training job

        Args:
            job: Training job to submit

        Returns:
            Job ID
        """
        job_id = self.job_manager.submit(job)
        self.scheduler.update_submission_count(job)
        return job_id

    def submit_job_from_config(self,
                               name: str,
                               config_path: Path,
                               output_dir: Path,
                               priority: int = 5,
                               requirements: Optional[ResourceRequirements] = None) -> str:
        """
        Submit job from training config file

        Args:
            name: Job name
            config_path: Path to training config (TOML)
            output_dir: Output directory for checkpoints
            priority: Job priority (0-20)
            requirements: Resource requirements

        Returns:
            Job ID
        """
        job_id = generate_job_id(name)
        log_file = self.config.log_dir / f"{job_id}.log"

        job = TrainingJob(
            id=job_id,
            name=name,
            config_path=config_path,
            output_dir=output_dir,
            log_file=log_file,
            priority=priority,
            resources=requirements or ResourceRequirements()
        )

        return self.submit_job(job)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel job

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled
        """
        job = self.job_manager.get_job(job_id)

        if job and job.is_running:
            # Terminate running job
            self.job_executor.terminate(job_id)

            # Release resources
            self.resource_manager.release(job_id)

            # Remove from progress monitoring
            self.progress_monitor.remove_job(job_id)

        return self.job_manager.cancel_job(job_id)

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """
        Get detailed job status

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary or None if job not found
        """
        job = self.job_manager.get_job(job_id)

        if not job:
            return None

        # Get progress metrics
        progress = self.progress_monitor.get_metrics(job_id) or {}

        # Get allocated resources
        allocated_gpus = self.resource_manager.get_allocation(job_id)

        status = {
            "job_id": job.id,
            "name": job.name,
            "state": job.state.value,
            "priority": job.priority,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration": job.duration,
            "allocated_gpus": allocated_gpus,
            "progress": progress,
            "exit_code": job.exit_code,
            "error_message": job.error_message
        }

        return status

    def get_statistics(self) -> dict:
        """
        Get overall orchestrator statistics

        Returns:
            Statistics dictionary
        """
        job_stats = self.job_manager.get_statistics()
        system_resources = self.resource_manager.get_system_resources()

        stats = {
            "jobs": job_stats.__dict__,
            "resources": {
                "total_gpus": len(system_resources.gpus),
                "available_gpus": len(self.resource_manager.get_available_gpus()),
                "total_memory_mb": system_resources.total_memory,
                "available_memory_mb": system_resources.available_memory
            }
        }

        return stats

    def list_jobs(self, state: Optional[JobState] = None) -> List[dict]:
        """
        List jobs with optional state filter

        Args:
            state: Filter by job state

        Returns:
            List of job status dictionaries
        """
        jobs = self.job_manager.list_jobs(state=state, limit=100)

        return [
            {
                "job_id": job.id,
                "name": job.name,
                "state": job.state.value,
                "priority": job.priority,
                "created_at": job.created_at,
                "duration": job.duration
            }
            for job in jobs
        ]

    # ========================================================================
    # Private Orchestration Loop
    # ========================================================================

    def _orchestration_loop(self):
        """Main orchestration loop"""
        logger.info("Orchestration loop started")

        while not self.stop_event.is_set():
            try:
                # Get queued jobs
                queued_jobs = self.job_manager.list_jobs(state=JobState.QUEUED)

                # Get completed jobs for dependency resolution
                completed_jobs = self.job_manager.list_jobs(state=JobState.COMPLETED)
                completed_job_ids = {job.id for job in completed_jobs}

                # Select next job to execute
                next_job = self.scheduler.select_next_job(queued_jobs, completed_job_ids)

                if next_job:
                    # Try to allocate resources
                    if self.resource_manager.can_allocate(next_job.resources):
                        # Allocate GPUs
                        allocated_gpus = self.resource_manager.allocate(next_job.id, next_job.resources)

                        if allocated_gpus:
                            next_job.allocated_gpus = allocated_gpus

                            # Update job state
                            self.job_manager.update_job_state(next_job.id, JobState.RUNNING)

                            # Start progress monitoring
                            self.progress_monitor.add_job(next_job)

                            # Execute job in background thread
                            Thread(target=self._execute_job, args=(next_job,), daemon=True).start()

            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")

            # Wait before next iteration
            self.stop_event.wait(self.config.scheduler.check_interval)

        logger.info("Orchestration loop stopped")

    def _execute_job(self, job: TrainingJob):
        """
        Execute job (runs in background thread)

        Args:
            job: Training job
        """
        try:
            logger.info(f"Executing job {job.id}")

            # Execute job
            result = self.job_executor.execute(job)

            # Update job with result
            job.exit_code = result.exit_code
            job.error_message = result.error_message
            job.checkpoints = result.checkpoints

            # Update job state
            if result.success:
                self.job_manager.update_job_state(job.id, JobState.COMPLETED)
                logger.info(f"Job {job.id} completed successfully")
            else:
                self.job_manager.update_job_state(job.id, JobState.FAILED)
                logger.error(f"Job {job.id} failed: {result.error_message}")

                # Retry if retries remaining
                if job.current_retry < job.retry_count:
                    logger.info(f"Retrying job {job.id}")
                    self.job_manager.retry_job(job.id)

        except Exception as e:
            logger.error(f"Job execution error: {e}")
            self.job_manager.update_job_state(job.id, JobState.FAILED)

        finally:
            # Release resources
            self.resource_manager.release(job.id)

            # Remove from progress monitoring
            self.progress_monitor.remove_job(job.id)
