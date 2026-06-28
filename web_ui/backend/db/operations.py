"""
Database operations for job management.
"""
import sqlite3
import json
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from .models import JobInfo, JobOutput


class JobDatabase:
    """Database operations for jobs."""

    def __init__(self, db_path: str):
        """
        Initialize job database.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn

    async def close(self) -> None:
        """Compatibility hook for FastAPI lifespan shutdown."""
        return None

    async def create_job(
        self,
        job_id: str,
        film_name: str,
        pipeline_type: str,
        input_video_path: str,
        output_base_dir: str,
        options: Dict[str, Any]
    ) -> None:
        """
        Create a new job entry.

        Args:
            job_id: Unique job identifier
            film_name: Name of the film
            pipeline_type: Type of pipeline (cpu_only, cpu_gpu_full)
            input_video_path: Path to input video
            output_base_dir: Base output directory
            options: Additional options dict
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, film_name, pipeline_type, status,
                    input_video_path, output_base_dir, options_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    film_name,
                    pipeline_type,
                    'pending',
                    input_video_path,
                    output_base_dir,
                    json.dumps(options),
                    time.time()
                )
            )
            conn.commit()
        finally:
            conn.close()

    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """
        Get job information by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobInfo if found, None otherwise
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Parse JSON fields
            options = json.loads(row['options_json']) if row['options_json'] else {}

            return JobInfo(
                job_id=row['job_id'],
                workflow_id=row['workflow_id'],
                film_name=row['film_name'],
                pipeline_type=row['pipeline_type'],
                status=row['status'],
                progress_percent=row['progress_percent'] or 0.0,
                current_stage=row['current_stage'],
                input_video_path=row['input_video_path'],
                output_base_dir=row['output_base_dir'],
                options=options,
                created_at=row['created_at'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                error_message=row['error_message']
            )

        finally:
            conn.close()

    async def update_job(
        self,
        job_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update job fields.

        Args:
            job_id: Job identifier
            updates: Dictionary of fields to update
        """
        if not updates:
            return

        # Build UPDATE query dynamically
        set_clauses = []
        values = []

        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        values.append(job_id)

        query = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE job_id = ?"

        conn = self._connect()
        try:
            conn.execute(query, values)
            conn.commit()
        finally:
            conn.close()

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[JobInfo]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status (optional)
            limit: Maximum number of jobs to return
            offset: Pagination offset

        Returns:
            List of JobInfo objects
        """
        conn = self._connect()
        try:
            if status:
                query = """
                    SELECT * FROM jobs
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                cursor = conn.execute(query, (status, limit, offset))
            else:
                query = """
                    SELECT * FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                cursor = conn.execute(query, (limit, offset))

            rows = cursor.fetchall()

            jobs = []
            for row in rows:
                options = json.loads(row['options_json']) if row['options_json'] else {}

                jobs.append(JobInfo(
                    job_id=row['job_id'],
                    workflow_id=row['workflow_id'],
                    film_name=row['film_name'],
                    pipeline_type=row['pipeline_type'],
                    status=row['status'],
                    progress_percent=row['progress_percent'] or 0.0,
                    current_stage=row['current_stage'],
                    input_video_path=row['input_video_path'],
                    output_base_dir=row['output_base_dir'],
                    options=options,
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    error_message=row['error_message']
                ))

            return jobs

        finally:
            conn.close()

    async def count_jobs(self, status: Optional[str] = None) -> int:
        """Count jobs with optional status filtering."""
        conn = self._connect()
        try:
            if status:
                cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,))
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM jobs")
            return int(cursor.fetchone()[0])
        finally:
            conn.close()

    async def add_job_output(
        self,
        job_id: str,
        stage: str,
        output_type: str,
        path: str,
        file_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add job output entry.

        Args:
            job_id: Job identifier
            stage: Stage name (e.g., 'frame_extraction')
            output_type: Type of output (e.g., 'frames', 'json')
            path: Output path
            file_count: Number of files (optional)
            size_bytes: Total size in bytes (optional)
            metadata: Additional metadata (optional)
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO job_outputs (
                    job_id, stage, output_type, path,
                    file_count, size_bytes, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    stage,
                    output_type,
                    path,
                    file_count,
                    size_bytes,
                    json.dumps(metadata) if metadata else None,
                    time.time()
                )
            )
            conn.commit()
        finally:
            conn.close()

    async def get_job_outputs(self, job_id: str) -> List[JobOutput]:
        """
        Get all outputs for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of JobOutput objects
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT stage, output_type, path, file_count, size_bytes, metadata_json
                FROM job_outputs
                WHERE job_id = ?
                ORDER BY created_at ASC
                """,
                (job_id,)
            )

            outputs = []
            for row in cursor.fetchall():
                metadata = json.loads(row['metadata_json']) if row['metadata_json'] else None

                outputs.append(JobOutput(
                    stage=row['stage'],
                    output_type=row['output_type'],
                    path=row['path'],
                    file_count=row['file_count'],
                    size_bytes=row['size_bytes'],
                    metadata=metadata
                ))

            return outputs

        finally:
            conn.close()

    async def record_system_metrics(
        self,
        cpu_percent: Optional[float] = None,
        ram_percent: Optional[float] = None,
        gpu_percent: Optional[int] = None,
        gpu_memory_used_mb: Optional[int] = None,
        gpu_memory_total_mb: Optional[int] = None,
        gpu_temperature_c: Optional[int] = None,
        disk_usage_percent: Optional[float] = None
    ) -> None:
        """
        Record system metrics snapshot.

        Args:
            cpu_percent: CPU usage percentage
            ram_percent: RAM usage percentage
            gpu_percent: GPU utilization percentage
            gpu_memory_used_mb: GPU memory used in MB
            gpu_memory_total_mb: GPU memory total in MB
            gpu_temperature_c: GPU temperature in Celsius
            disk_usage_percent: Disk usage percentage
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, ram_percent, gpu_percent,
                    gpu_memory_used_mb, gpu_memory_total_mb,
                    gpu_temperature_c, disk_usage_percent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    cpu_percent,
                    ram_percent,
                    gpu_percent,
                    gpu_memory_used_mb,
                    gpu_memory_total_mb,
                    gpu_temperature_c,
                    disk_usage_percent
                )
            )
            conn.commit()
        finally:
            conn.close()

    async def list_system_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent system metric snapshots, oldest first."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT timestamp, cpu_percent, ram_percent, gpu_percent,
                       gpu_memory_used_mb, gpu_memory_total_mb,
                       gpu_temperature_c, disk_usage_percent
                FROM system_metrics
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            rows.reverse()
            return rows
        finally:
            conn.close()
