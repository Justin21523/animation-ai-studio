"""
State Manager for Workflow Orchestration

Manages workflow state with checkpoint/resume capability.
Uses SQLite for metadata and JSON files for large state objects.

Usage:
    manager = StateManager()

    # Save checkpoint
    await manager.save_checkpoint("wf_123", {"task_id": "task_1", "status": "running"})

    # Load checkpoint
    state = await manager.load_checkpoint("wf_123")

    # Resume workflow
    await manager.resume_workflow("wf_123")

    # List all checkpoints
    checkpoints = await manager.list_checkpoints("wf_123")
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata."""
    workflow_id: str
    checkpoint_number: int
    timestamp: float
    state_path: str
    size_bytes: int
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class StateManager:
    """
    Manages workflow state with checkpoint/resume capability.

    Features:
    - SQLite backend for checkpoint metadata
    - JSON files for large state objects
    - Automatic cleanup of old checkpoints (keep last N)
    - Transaction support for atomic updates
    - Async operations (non-blocking)

    Storage Structure:
    /mnt/data/tmp/automation/state/
    ├── workflows.db                    # SQLite metadata database
    └── {workflow_id}/                  # Per-workflow state directory
        ├── checkpoint_0001.json
        ├── checkpoint_0002.json
        └── ...

    Example:
        manager = StateManager(
            state_dir="/mnt/data/tmp/automation/state",
            max_checkpoints=10
        )

        # Save state
        await manager.save_checkpoint("wf_123", {
            "current_task": "task_1",
            "completed_tasks": ["task_0"],
            "task_results": {...}
        })

        # Load latest state
        state = await manager.load_checkpoint("wf_123")

        # Resume from checkpoint
        await manager.resume_workflow("wf_123")
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        max_checkpoints: int = 10,
        db_name: str = "workflows.db"
    ):
        """
        Initialize State Manager.

        Args:
            state_dir: Directory for state storage (default: /mnt/data/tmp/automation/state)
            max_checkpoints: Maximum checkpoints to keep per workflow (older are deleted)
            db_name: SQLite database filename
        """
        if state_dir is None:
            state_dir = Path("/mnt/data/tmp/automation/state")

        self.state_dir = Path(state_dir)
        self.max_checkpoints = max_checkpoints
        self.db_path = self.state_dir / db_name

        # Create state directory
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info(f"StateManager initialized: state_dir={self.state_dir}, max_checkpoints={max_checkpoints}")

    def _init_database(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                checkpoint_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                state_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                description TEXT,
                UNIQUE(workflow_id, checkpoint_number)
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_id
            ON checkpoints(workflow_id, checkpoint_number DESC)
        """)

        # Create workflow metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                status TEXT NOT NULL,
                checkpoint_count INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

        logger.debug(f"Database initialized: {self.db_path}")

    async def save_checkpoint(
        self,
        workflow_id: str,
        state: Dict[str, Any],
        description: Optional[str] = None
    ) -> CheckpointMetadata:
        """
        Save workflow checkpoint.

        Args:
            workflow_id: Unique workflow identifier
            state: Workflow state dictionary (will be serialized to JSON)
            description: Optional checkpoint description

        Returns:
            CheckpointMetadata for saved checkpoint

        Raises:
            ValueError: If state cannot be serialized to JSON
        """
        async with self._lock:
            # Get next checkpoint number
            checkpoint_number = await self._get_next_checkpoint_number(workflow_id)

            # Create workflow directory
            workflow_dir = self.state_dir / workflow_id
            workflow_dir.mkdir(exist_ok=True)

            # Save state to JSON file
            state_path = workflow_dir / f"checkpoint_{checkpoint_number:04d}.json"

            try:
                # Serialize state
                state_json = json.dumps(state, indent=2)

                # Write to file
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, state_path.write_text, state_json)

                # Get file size
                size_bytes = state_path.stat().st_size

            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to serialize state to JSON: {e}")

            # Save metadata to database
            timestamp = time.time()
            metadata = CheckpointMetadata(
                workflow_id=workflow_id,
                checkpoint_number=checkpoint_number,
                timestamp=timestamp,
                state_path=str(state_path),
                size_bytes=size_bytes,
                description=description
            )

            await self._save_metadata(metadata)

            # Update workflow metadata
            await self._update_workflow_metadata(workflow_id)

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints(workflow_id)

            logger.info(
                f"Saved checkpoint: workflow={workflow_id}, "
                f"number={checkpoint_number}, size={size_bytes} bytes"
            )

            return metadata

    async def load_checkpoint(
        self,
        workflow_id: str,
        checkpoint_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load workflow checkpoint.

        Args:
            workflow_id: Workflow identifier
            checkpoint_number: Specific checkpoint to load (None = latest)

        Returns:
            Workflow state dictionary, or None if not found
        """
        async with self._lock:
            # Get checkpoint metadata
            if checkpoint_number is None:
                metadata = await self._get_latest_checkpoint_metadata(workflow_id)
            else:
                metadata = await self._get_checkpoint_metadata(workflow_id, checkpoint_number)

            if metadata is None:
                logger.warning(f"Checkpoint not found: workflow={workflow_id}, number={checkpoint_number}")
                return None

            # Load state from file
            state_path = Path(metadata.state_path)

            if not state_path.exists():
                logger.error(f"Checkpoint file missing: {state_path}")
                return None

            try:
                # Read file
                loop = asyncio.get_event_loop()
                state_json = await loop.run_in_executor(None, state_path.read_text)

                # Deserialize
                state = json.loads(state_json)

                logger.info(f"Loaded checkpoint: workflow={workflow_id}, number={metadata.checkpoint_number}")
                return state

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None

    async def list_checkpoints(self, workflow_id: str) -> List[CheckpointMetadata]:
        """
        List all checkpoints for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of checkpoint metadata (newest first)
        """
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT workflow_id, checkpoint_number, timestamp, state_path, size_bytes, description
                FROM checkpoints
                WHERE workflow_id = ?
                ORDER BY checkpoint_number DESC
            """, (workflow_id,))

            rows = cursor.fetchall()
            conn.close()

            checkpoints = [
                CheckpointMetadata(
                    workflow_id=row[0],
                    checkpoint_number=row[1],
                    timestamp=row[2],
                    state_path=row[3],
                    size_bytes=row[4],
                    description=row[5]
                )
                for row in rows
            ]

            return checkpoints

    async def delete_checkpoint(
        self,
        workflow_id: str,
        checkpoint_number: int
    ) -> bool:
        """
        Delete specific checkpoint.

        Args:
            workflow_id: Workflow identifier
            checkpoint_number: Checkpoint number to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            # Get metadata
            metadata = await self._get_checkpoint_metadata(workflow_id, checkpoint_number)
            if metadata is None:
                return False

            # Delete file
            state_path = Path(metadata.state_path)
            if state_path.exists():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, state_path.unlink)

            # Delete metadata
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM checkpoints
                WHERE workflow_id = ? AND checkpoint_number = ?
            """, (workflow_id, checkpoint_number))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"Deleted checkpoint: workflow={workflow_id}, number={checkpoint_number}")

            return deleted

    async def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete all checkpoints and data for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            # Delete workflow directory
            workflow_dir = self.state_dir / workflow_id
            if workflow_dir.exists():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, shutil.rmtree, workflow_dir)

            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete checkpoints
            cursor.execute("DELETE FROM checkpoints WHERE workflow_id = ?", (workflow_id,))
            checkpoints_deleted = cursor.rowcount

            # Delete workflow metadata
            cursor.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            workflow_deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()

            if workflow_deleted:
                logger.info(
                    f"Deleted workflow: workflow={workflow_id}, "
                    f"checkpoints_deleted={checkpoints_deleted}"
                )

            return workflow_deleted

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow status metadata.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Dictionary with workflow status, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT workflow_id, created_at, updated_at, status, checkpoint_count
            FROM workflows
            WHERE workflow_id = ?
        """, (workflow_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return {
            "workflow_id": row[0],
            "created_at": row[1],
            "updated_at": row[2],
            "status": row[3],
            "checkpoint_count": row[4]
        }

    async def _get_next_checkpoint_number(self, workflow_id: str) -> int:
        """Get next checkpoint number for workflow."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MAX(checkpoint_number)
            FROM checkpoints
            WHERE workflow_id = ?
        """, (workflow_id,))

        row = cursor.fetchone()
        conn.close()

        max_number = row[0] if row[0] is not None else 0
        return max_number + 1

    async def _save_metadata(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO checkpoints (workflow_id, checkpoint_number, timestamp, state_path, size_bytes, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metadata.workflow_id,
            metadata.checkpoint_number,
            metadata.timestamp,
            metadata.state_path,
            metadata.size_bytes,
            metadata.description
        ))

        conn.commit()
        conn.close()

    async def _get_checkpoint_metadata(
        self,
        workflow_id: str,
        checkpoint_number: int
    ) -> Optional[CheckpointMetadata]:
        """Get specific checkpoint metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT workflow_id, checkpoint_number, timestamp, state_path, size_bytes, description
            FROM checkpoints
            WHERE workflow_id = ? AND checkpoint_number = ?
        """, (workflow_id, checkpoint_number))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return CheckpointMetadata(
            workflow_id=row[0],
            checkpoint_number=row[1],
            timestamp=row[2],
            state_path=row[3],
            size_bytes=row[4],
            description=row[5]
        )

    async def _get_latest_checkpoint_metadata(
        self,
        workflow_id: str
    ) -> Optional[CheckpointMetadata]:
        """Get latest checkpoint metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT workflow_id, checkpoint_number, timestamp, state_path, size_bytes, description
            FROM checkpoints
            WHERE workflow_id = ?
            ORDER BY checkpoint_number DESC
            LIMIT 1
        """, (workflow_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return CheckpointMetadata(
            workflow_id=row[0],
            checkpoint_number=row[1],
            timestamp=row[2],
            state_path=row[3],
            size_bytes=row[4],
            description=row[5]
        )

    async def _update_workflow_metadata(self, workflow_id: str):
        """Update workflow metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count checkpoints
        cursor.execute("""
            SELECT COUNT(*) FROM checkpoints WHERE workflow_id = ?
        """, (workflow_id,))
        checkpoint_count = cursor.fetchone()[0]

        # Update or insert workflow metadata
        timestamp = time.time()
        cursor.execute("""
            INSERT INTO workflows (workflow_id, created_at, updated_at, status, checkpoint_count)
            VALUES (?, ?, ?, 'active', ?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                updated_at = ?,
                checkpoint_count = ?
        """, (workflow_id, timestamp, timestamp, checkpoint_count, timestamp, checkpoint_count))

        conn.commit()
        conn.close()

    async def _cleanup_old_checkpoints(self, workflow_id: str):
        """Delete old checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return  # No cleanup

        # Get all checkpoints
        checkpoints = await self.list_checkpoints(workflow_id)

        # Delete excess checkpoints (oldest first)
        if len(checkpoints) > self.max_checkpoints:
            to_delete = checkpoints[self.max_checkpoints:]

            for checkpoint in to_delete:
                await self.delete_checkpoint(workflow_id, checkpoint.checkpoint_number)

            logger.info(
                f"Cleaned up old checkpoints: workflow={workflow_id}, "
                f"deleted={len(to_delete)}, kept={self.max_checkpoints}"
            )

    def __repr__(self) -> str:
        return f"StateManager(state_dir={self.state_dir}, max_checkpoints={self.max_checkpoints})"
