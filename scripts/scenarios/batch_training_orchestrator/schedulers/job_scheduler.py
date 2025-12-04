"""
Batch Training Orchestrator - Job Scheduler

Schedules jobs based on various strategies (FIFO, priority, fair-share, etc.).

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
from typing import List, Optional, Dict
from collections import defaultdict

from ..common import (
    TrainingJob,
    SchedulingStrategy,
    SchedulerConfig,
    ResourceRequirements
)


logger = logging.getLogger(__name__)


class JobScheduler:
    """
    Schedules jobs using various strategies

    Features:
    - Multiple scheduling strategies (FIFO, priority, fair-share, shortest-job-first)
    - Dependency resolution via topological sort
    - Load balancing across GPUs
    - Resource-aware scheduling
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize job scheduler

        Args:
            config: Scheduler configuration
        """
        self.config = config or SchedulerConfig()

        # Track job submissions per user/project for fair-share
        self.submission_counts: Dict[str, int] = defaultdict(int)

        logger.info(f"JobScheduler initialized (strategy={self.config.strategy.value})")

    def select_next_job(self, pending_jobs: List[TrainingJob],
                       completed_job_ids: set,
                       available_resources: Optional[Dict] = None) -> Optional[TrainingJob]:
        """
        Select next job to execute based on scheduling strategy

        Args:
            pending_jobs: List of pending jobs
            completed_job_ids: Set of completed job IDs (for dependency checking)
            available_resources: Available system resources

        Returns:
            Selected job or None if no eligible job
        """
        if not pending_jobs:
            return None

        # Filter jobs with satisfied dependencies
        eligible_jobs = []
        for job in pending_jobs:
            if self._dependencies_satisfied(job, completed_job_ids):
                eligible_jobs.append(job)

        if not eligible_jobs:
            logger.debug("No jobs with satisfied dependencies")
            return None

        # Apply scheduling strategy
        if self.config.strategy == SchedulingStrategy.FIFO:
            selected = self._schedule_fifo(eligible_jobs)

        elif self.config.strategy == SchedulingStrategy.PRIORITY:
            selected = self._schedule_priority(eligible_jobs)

        elif self.config.strategy == SchedulingStrategy.FAIR_SHARE:
            selected = self._schedule_fair_share(eligible_jobs)

        elif self.config.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            selected = self._schedule_shortest_job_first(eligible_jobs)

        else:
            # Default to FIFO
            selected = self._schedule_fifo(eligible_jobs)

        if selected:
            logger.info(f"Scheduled job {selected.id} ({selected.name}) using {self.config.strategy.value}")

        return selected

    def update_submission_count(self, job: TrainingJob):
        """
        Update submission count for fair-share scheduling

        Args:
            job: Training job
        """
        # Use job tags to identify user/project
        user_tag = None
        for tag in job.tags:
            if tag.startswith("user:"):
                user_tag = tag
                break

        if user_tag:
            self.submission_counts[user_tag] += 1

    # ========================================================================
    # Private Helper Methods - Scheduling Strategies
    # ========================================================================

    def _schedule_fifo(self, jobs: List[TrainingJob]) -> Optional[TrainingJob]:
        """
        Schedule using First-In-First-Out

        Args:
            jobs: Eligible jobs

        Returns:
            Selected job (oldest submission)
        """
        if not jobs:
            return None

        # Sort by created_at (oldest first)
        jobs_sorted = sorted(jobs, key=lambda j: j.created_at)
        return jobs_sorted[0]

    def _schedule_priority(self, jobs: List[TrainingJob]) -> Optional[TrainingJob]:
        """
        Schedule using priority-based selection

        Args:
            jobs: Eligible jobs

        Returns:
            Selected job (highest priority, then oldest)
        """
        if not jobs:
            return None

        # Sort by priority (highest first), then by created_at (oldest first)
        jobs_sorted = sorted(jobs, key=lambda j: (-j.priority, j.created_at))
        return jobs_sorted[0]

    def _schedule_fair_share(self, jobs: List[TrainingJob]) -> Optional[TrainingJob]:
        """
        Schedule using fair-share allocation

        Args:
            jobs: Eligible jobs

        Returns:
            Selected job (fairest distribution across users/projects)
        """
        if not jobs:
            return None

        # Calculate fairness score (lower = fairer)
        def fairness_score(job: TrainingJob) -> int:
            user_tag = None
            for tag in job.tags:
                if tag.startswith("user:"):
                    user_tag = tag
                    break

            if user_tag:
                return self.submission_counts.get(user_tag, 0)
            return 0

        # Sort by fairness score (lowest first), then priority (highest first)
        jobs_sorted = sorted(jobs, key=lambda j: (fairness_score(j), -j.priority, j.created_at))
        return jobs_sorted[0]

    def _schedule_shortest_job_first(self, jobs: List[TrainingJob]) -> Optional[TrainingJob]:
        """
        Schedule using shortest job first (based on estimated duration)

        Args:
            jobs: Eligible jobs

        Returns:
            Selected job (shortest estimated duration)
        """
        if not jobs:
            return None

        # Sort by estimated duration (shortest first), then priority (highest first)
        def sort_key(job: TrainingJob):
            estimated_duration = job.resources.estimated_duration or float('inf')
            return (estimated_duration, -job.priority, job.created_at)

        jobs_sorted = sorted(jobs, key=sort_key)
        return jobs_sorted[0]

    # ========================================================================
    # Private Helper Methods - Dependency Resolution
    # ========================================================================

    def _dependencies_satisfied(self, job: TrainingJob, completed_job_ids: set) -> bool:
        """
        Check if job dependencies are satisfied

        Args:
            job: Training job
            completed_job_ids: Set of completed job IDs

        Returns:
            True if all dependencies completed
        """
        if not self.config.enable_dependencies:
            return True

        for dep_id in job.dependencies:
            if dep_id not in completed_job_ids:
                return False

        return True

    def topological_sort(self, jobs: List[TrainingJob]) -> List[TrainingJob]:
        """
        Sort jobs by dependencies (topological order)

        Args:
            jobs: List of jobs with dependencies

        Returns:
            Sorted list of jobs
        """
        # Build dependency graph
        job_map = {job.id: job for job in jobs}
        in_degree = {job.id: 0 for job in jobs}
        adjacency = {job.id: [] for job in jobs}

        for job in jobs:
            for dep_id in job.dependencies:
                if dep_id in job_map:
                    adjacency[dep_id].append(job.id)
                    in_degree[job.id] += 1

        # Kahn's algorithm for topological sort
        queue = [job_id for job_id in in_degree if in_degree[job_id] == 0]
        sorted_jobs = []

        while queue:
            # Sort queue by priority before processing
            queue.sort(key=lambda jid: -job_map[jid].priority)

            job_id = queue.pop(0)
            sorted_jobs.append(job_map[job_id])

            # Reduce in-degree for dependent jobs
            for dependent_id in adjacency[job_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # Check for cycles
        if len(sorted_jobs) != len(jobs):
            logger.error("Circular dependency detected in job graph")
            # Return original list if cycle detected
            return jobs

        return sorted_jobs
