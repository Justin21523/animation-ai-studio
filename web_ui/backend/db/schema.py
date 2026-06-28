"""
Database schema for Web UI.
Extends existing workflows.db with new tables for job management.
"""
import sqlite3
from pathlib import Path
from typing import Optional


# SQL for creating jobs table
CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    workflow_id TEXT,
    film_name TEXT NOT NULL,
    pipeline_type TEXT NOT NULL CHECK(
        pipeline_type IN ('cpu_only', 'cpu_gpu_full', 'custom', 'image_generate')
        OR pipeline_type LIKE 'action_%'
        OR pipeline_type LIKE 'creative_%'
    ),
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress_percent REAL DEFAULT 0.0,
    current_stage TEXT,
    progress_json TEXT,
    input_video_path TEXT NOT NULL,
    output_base_dir TEXT NOT NULL,
    options_json TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    error_message TEXT,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id) ON DELETE SET NULL
)
"""

# SQL for creating job outputs table
CREATE_JOB_OUTPUTS_TABLE = """
CREATE TABLE IF NOT EXISTS job_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    output_type TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    file_count INTEGER,
    metadata_json TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
)
"""

# SQL for creating system metrics table
CREATE_SYSTEM_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cpu_percent REAL,
    ram_percent REAL,
    gpu_percent REAL,
    gpu_memory_used_mb INTEGER,
    gpu_memory_total_mb INTEGER,
    gpu_temperature_c INTEGER,
    disk_usage_percent REAL
)
"""

# Indexes for performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_job_outputs_job_id ON job_outputs(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp DESC)",
]


def init_database(db_path: str) -> None:
    """
    Initialize database with Web UI tables.

    Args:
        db_path: Path to SQLite database file

    Note:
        This extends the existing workflows.db used by StateManager.
        If the database doesn't exist, it will be created.
    """
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Migrate legacy jobs table if pipeline_type constraint is too strict.
        cursor.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='jobs'
            """
        )
        row = cursor.fetchone()
        if row and row[0]:
            existing_sql = str(row[0])
            needs_migration = (
                ("LIKE 'action_%'" not in existing_sql)
                or ("LIKE 'creative_%'" not in existing_sql)
                or ("'image_generate'" not in existing_sql)
            )
            if needs_migration:
                cursor.execute("ALTER TABLE jobs RENAME TO jobs_legacy")
                cursor.execute(CREATE_JOBS_TABLE)

                cursor.execute("PRAGMA table_info(jobs)")
                new_cols = [r[1] for r in cursor.fetchall()]
                cursor.execute("PRAGMA table_info(jobs_legacy)")
                old_cols = {r[1] for r in cursor.fetchall()}

                common = [c for c in new_cols if c in old_cols]
                if common:
                    cols_csv = ", ".join(common)
                    cursor.execute(f"INSERT INTO jobs ({cols_csv}) SELECT {cols_csv} FROM jobs_legacy")

                conn.commit()

        # Create tables
        cursor.execute(CREATE_JOBS_TABLE)
        cursor.execute(CREATE_JOB_OUTPUTS_TABLE)
        cursor.execute(CREATE_SYSTEM_METRICS_TABLE)

        # Create indexes
        for index_sql in CREATE_INDEXES:
            cursor.execute(index_sql)

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to initialize database: {e}")

    finally:
        conn.close()


def check_database_version(db_path: str) -> bool:
    """
    Check if database has Web UI tables.

    Args:
        db_path: Path to SQLite database

    Returns:
        True if all Web UI tables exist
    """
    if not Path(db_path).exists():
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if jobs table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='jobs'
        """)

        return cursor.fetchone() is not None

    finally:
        conn.close()
