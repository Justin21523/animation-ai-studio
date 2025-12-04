"""
Batch Training Orchestrator - Job Manager

Manages job queue, state transitions, persistence, and dependency resolution.

Author: Animation AI Studio
Date: 2025-12-03
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

from ..common import (
    TrainingJob,
    JobState,
    JobPriority,
    JobStatistics,
    generate_job_id
)


logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages training job queue and lifecycle

    Features:
    - Job submission and validation
    - Priority-based queueing
    - Dependency resolution
    - State tracking and transitions
    - Persistent storage (JSON)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize job manager

        Args:
            db_path: Path to job database JSON file
        """
        self.db_path = db_path or Path("jobs.db")
        self.jobs: Dict[str, TrainingJob] = {}
        self.queue: deque = deque()

        # Load existing jobs if database exists
        if self.db_path.exists():
            self.load()

        logger.info(f"JobManager initialized with {len(self.jobs)} jobs")

    def submit(self, job: TrainingJob) -> str:
        """
        Submit new job to queue

        Args:
            job: Training job to submit

        Returns:
            Job ID

        Raises:
            ValueError: If job validation fails
        """
        # Validate job
        self._validate_job(job)

        # Check for duplicate job ID
        if job.id in self.jobs:
            raise ValueError(f"Job with ID {job.id} already exists")

        # Add to jobs registry
        self.jobs[job.id] = job

        # Update state to QUEUED
        self._transition_state(job.id, JobState.QUEUED)
        job.queued_at = datetime.now().timestamp()

        # Add to queue (will be sorted by priority)
        self.queue.append(job.id)
        self._sort_queue()

        # Persist to disk
        self.save()

        logger.info(f"Job {job.id} ({job.name}) submitted with priority {job.priority}")
        return job.id

    def get_next_job(self, max_dependencies_depth: int = 10) -> Optional[TrainingJob]:
        """
        Get next job from queue that has all dependencies satisfied

        Args:
            max_dependencies_depth: Maximum depth for dependency resolution

        Returns:
            Next executable job or None if queue is empty or all jobs blocked
        """
        if not self.queue:
            return None

        # Find first job with satisfied dependencies
        for job_id in list(self.queue):
            job = self.jobs[job_id]

            # Check if dependencies are satisfied
            if self._are_dependencies_satisfied(job_id, max_dependencies_depth):
                # Remove from queue
                self.queue.remove(job_id)
                return job

        # All jobs in queue are blocked by dependencies
        logger.warning("All queued jobs blocked by dependencies")
        return None

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get job by ID

        Args:
            job_id: Job identifier

        Returns:
            Training job or None if not found
        """
        return self.jobs.get(job_id)

    def list_jobs(self,
                  state: Optional[JobState] = None,
                  limit: Optional[int] = None) -> List[TrainingJob]:
        """
        List jobs filtered by state

        Args:
            state: Filter by job state (None = all jobs)
            limit: Maximum number of jobs to return

        Returns:
            List of training jobs
        """
        jobs = list(self.jobs.values())

        # Filter by state
        if state is not None:
            jobs = [j for j in jobs if j.state == state]

        # Sort by created_at (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply limit
        if limit is not None:
            jobs = jobs[:limit]

        return jobs

    def update_job_state(self, job_id: str, new_state: JobState) -> bool:
        """
        Update job state with validation

        Args:
            job_id: Job identifier
            new_state: New job state

        Returns:
            True if state transition succeeded
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False

        job = self.jobs[job_id]
        old_state = job.state

        # Validate state transition
        if not self._is_valid_transition(old_state, new_state):
            logger.error(f"Invalid state transition: {old_state.value} -> {new_state.value}")
            return False

        # Perform transition
        self._transition_state(job_id, new_state)

        # Update timestamps
        if new_state == JobState.RUNNING:
            job.started_at = datetime.now().timestamp()
        elif new_state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
            job.completed_at = datetime.now().timestamp()

        # Persist
        self.save()

        logger.info(f"Job {job_id} state: {old_state.value} -> {new_state.value}")
        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel job (if not already terminal)

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False

        job = self.jobs[job_id]

        # Check if job is in terminal state
        if job.is_terminal:
            logger.warning(f"Job {job_id} already in terminal state: {job.state.value}")
            return False

        # Remove from queue if queued
        if job_id in self.queue:
            self.queue.remove(job_id)

        # Update state
        return self.update_job_state(job_id, JobState.CANCELLED)

    def retry_job(self, job_id: str) -> bool:
        """
        Retry failed job

        Args:
            job_id: Job identifier

        Returns:
            True if job was re-queued
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False

        job = self.jobs[job_id]

        # Check if job can be retried
        if job.state != JobState.FAILED:
            logger.error(f"Only FAILED jobs can be retried (current: {job.state.value})")
            return False

        if job.current_retry >= job.retry_count:
            logger.error(f"Job {job_id} exceeded retry limit ({job.retry_count})")
            return False

        # Increment retry counter
        job.current_retry += 1

        # Reset state and timestamps
        job.state = JobState.PENDING
        job.started_at = None
        job.completed_at = None
        job.exit_code = None
        job.error_message = None

        # Re-queue
        self.queue.append(job_id)
        self._sort_queue()

        # Update state to QUEUED
        self._transition_state(job_id, JobState.QUEUED)
        job.queued_at = datetime.now().timestamp()

        # Persist
        self.save()

        logger.info(f"Job {job_id} re-queued (retry {job.current_retry}/{job.retry_count})")
        return True

    def get_statistics(self) -> JobStatistics:
        """
        Compute job queue statistics

        Returns:
            Job statistics
        """
        total_jobs = len(self.jobs)

        # Count by state
        state_counts = {state: 0 for state in JobState}
        for job in self.jobs.values():
            state_counts[job.state] += 1

        # Compute GPU hours and average duration
        total_gpu_hours = 0.0
        completed_durations = []

        for job in self.jobs.values():
            if job.duration is not None:
                duration_hours = job.duration / 3600.0
                gpu_hours = duration_hours * len(job.allocated_gpus) if job.allocated_gpus else duration_hours
                total_gpu_hours += gpu_hours

                if job.state == JobState.COMPLETED:
                    completed_durations.append(job.duration)

        average_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0.0

        return JobStatistics(
            total_jobs=total_jobs,
            pending_jobs=state_counts[JobState.PENDING],
            queued_jobs=state_counts[JobState.QUEUED],
            running_jobs=state_counts[JobState.RUNNING],
            completed_jobs=state_counts[JobState.COMPLETED],
            failed_jobs=state_counts[JobState.FAILED],
            cancelled_jobs=state_counts[JobState.CANCELLED],
            total_gpu_hours=total_gpu_hours,
            average_job_duration=average_duration
        )

    def save(self):
        """Save job database to disk"""
        data = {
            "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
            "queue": list(self.queue),
            "saved_at": datetime.now().isoformat()
        }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Job database saved to {self.db_path}")

    def load(self):
        """Load job database from disk"""
        if not self.db_path.exists():
            logger.warning(f"Job database not found: {self.db_path}")
            return

        with open(self.db_path, 'r') as f:
            data = json.load(f)

        # Reconstruct jobs from dict
        self.jobs = {}
        for job_id, job_dict in data.get("jobs", {}).items():
            self.jobs[job_id] = self._job_from_dict(job_dict)

        # Restore queue
        self.queue = deque(data.get("queue", []))

        logger.info(f"Loaded {len(self.jobs)} jobs from {self.db_path}")

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _validate_job(self, job: TrainingJob):
        """
        Validate job definition

        Args:
            job: Training job

        Raises:
            ValueError: If validation fails
        """
        if not job.name:
            raise ValueError("Job name cannot be empty")

        if not job.config_path.exists():
            raise ValueError(f"Config file not found: {job.config_path}")

        # Validate dependencies exist
        for dep_id in job.dependencies:
            if dep_id not in self.jobs:
                raise ValueError(f"Dependency job {dep_id} not found")

    def _transition_state(self, job_id: str, new_state: JobState):
        """Update job state without validation"""
        self.jobs[job_id].state = new_state

    def _is_valid_transition(self, old_state: JobState, new_state: JobState) -> bool:
        """
        Check if state transition is valid

        Args:
            old_state: Current state
            new_state: Target state

        Returns:
            True if transition is allowed
        """
        # Define allowed transitions
        allowed_transitions = {
            JobState.PENDING: [JobState.QUEUED, JobState.CANCELLED],
            JobState.QUEUED: [JobState.RUNNING, JobState.CANCELLED],
            JobState.RUNNING: [JobState.COMPLETED, JobState.FAILED, JobState.PAUSED, JobState.CANCELLED],
            JobState.PAUSED: [JobState.RUNNING, JobState.CANCELLED],
            JobState.COMPLETED: [],  # Terminal
            JobState.FAILED: [JobState.PENDING],  # Can retry
            JobState.CANCELLED: []  # Terminal
        }

        return new_state in allowed_transitions.get(old_state, [])

    def _sort_queue(self):
        """Sort queue by priority (highest first)"""
        self.queue = deque(sorted(self.queue, key=lambda jid: self.jobs[jid].priority, reverse=True))

    def _are_dependencies_satisfied(self, job_id: str, max_depth: int) -> bool:
        """
        Check if all dependencies are completed

        Args:
            job_id: Job identifier
            max_depth: Maximum recursion depth

        Returns:
            True if all dependencies satisfied
        """
        if max_depth <= 0:
            logger.error(f"Max dependency depth exceeded for job {job_id}")
            return False

        job = self.jobs[job_id]

        for dep_id in job.dependencies:
            if dep_id not in self.jobs:
                logger.error(f"Dependency {dep_id} not found for job {job_id}")
                return False

            dep_job = self.jobs[dep_id]

            # Dependency must be completed
            if dep_job.state != JobState.COMPLETED:
                return False

            # Recursively check nested dependencies
            if not self._are_dependencies_satisfied(dep_id, max_depth - 1):
                return False

        return True

    def _job_from_dict(self, job_dict: Dict) -> TrainingJob:
        """
        Reconstruct TrainingJob from dictionary

        Args:
            job_dict: Serialized job data

        Returns:
            TrainingJob instance
        """
        from ..common import ResourceRequirements

        # Reconstruct ResourceRequirements
        resources_dict = job_dict.get("resources", {})
        resources = ResourceRequirements(
            gpu_count=resources_dict.get("gpu_count", 1),
            gpu_memory=resources_dict.get("gpu_memory", 16384),
            cpu_cores=resources_dict.get("cpu_cores", 4),
            system_memory=resources_dict.get("system_memory", 32768),
            disk_space=resources_dict.get("disk_space", 102400),
            estimated_duration=resources_dict.get("estimated_duration")
        )

        # Reconstruct TrainingJob
        job = TrainingJob(
            id=job_dict["id"],
            name=job_dict["name"],
            config_path=Path(job_dict["config_path"]),
            output_dir=Path(job_dict["output_dir"]),
            log_file=Path(job_dict["log_file"]),
            priority=job_dict.get("priority", JobPriority.NORMAL.value),
            dependencies=job_dict.get("dependencies", []),
            resources=resources,
            allocated_gpus=job_dict.get("allocated_gpus", []),
            state=JobState(job_dict["state"]),
            created_at=job_dict.get("created_at", datetime.now().timestamp()),
            queued_at=job_dict.get("queued_at"),
            started_at=job_dict.get("started_at"),
            completed_at=job_dict.get("completed_at"),
            retry_count=job_dict.get("retry_count", 3),
            current_retry=job_dict.get("current_retry", 0),
            timeout=job_dict.get("timeout"),
            command=job_dict.get("command"),
            process_id=job_dict.get("process_id"),
            tmux_session=job_dict.get("tmux_session"),
            exit_code=job_dict.get("exit_code"),
            error_message=job_dict.get("error_message"),
            checkpoints=[Path(p) for p in job_dict.get("checkpoints", [])],
            metadata=job_dict.get("metadata", {}),
            tags=job_dict.get("tags", [])
        )

        return job
