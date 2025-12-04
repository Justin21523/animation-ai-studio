"""
Media Processing Safety Integration

Safety adapter for media processing scenario.
Enforces constraints, manages checkpoints, handles emergencies.

Author: Animation AI Studio
Date: 2025-12-03
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MediaProcessingSafetyAdapter:
    """
    Safety adapter for media processing integration

    Features:
    - File size and duration constraints
    - Memory budget enforcement
    - Processing checkpoint management
    - Emergency shutdown handling
    - Resource monitoring

    Example:
        from scripts.automation.safety.system_manager import SafetyManager

        safety_manager = SafetyManager()
        adapter = MediaProcessingSafetyAdapter(
            safety_manager=safety_manager,
            max_file_size_gb=10.0,
            max_duration_hours=4.0,
            checkpoint_interval=300
        )

        # Create MediaProcessor with safety
        processor = MediaProcessor(safety_manager=adapter)

        # Safety constraints are automatically enforced
        result = processor.transcode_video(...)
    """

    def __init__(
        self,
        safety_manager: Any,
        max_file_size_gb: float = 10.0,
        max_duration_hours: float = 4.0,
        checkpoint_interval: int = 300,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize safety adapter

        Args:
            safety_manager: SafetyManager from safety system
            max_file_size_gb: Maximum input file size in GB
            max_duration_hours: Maximum processing duration in hours
            checkpoint_interval: Checkpoint interval in seconds
            checkpoint_dir: Checkpoint directory (default: /tmp/media_processing_checkpoints)
        """
        self.safety_manager = safety_manager
        self.max_file_size_bytes = int(max_file_size_gb * 1024 * 1024 * 1024)
        self.max_duration_seconds = max_duration_hours * 3600
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/media_processing_checkpoints")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Processing state
        self._last_checkpoint_time = datetime.now()
        self._processing_start_time = None

        logger.info(
            f"MediaProcessingSafetyAdapter initialized: "
            f"max_size={max_file_size_gb}GB, max_duration={max_duration_hours}h"
        )

    def get_max_file_size(self) -> int:
        """
        Get maximum file size in bytes

        Returns:
            Max file size in bytes
        """
        return self.max_file_size_bytes

    def check_file_size(self, file_path: Path):
        """
        Check if file size is within limits

        Args:
            file_path: File to check

        Raises:
            RuntimeError: If file exceeds size limit
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size

        if file_size > self.max_file_size_bytes:
            raise RuntimeError(
                f"File size {file_size / 1024**3:.2f}GB exceeds limit "
                f"{self.max_file_size_bytes / 1024**3:.2f}GB"
            )

        logger.debug(f"File size check passed: {file_size / 1024**3:.2f}GB")

    def check_processing_duration(self):
        """
        Check if processing duration is within limits

        Raises:
            RuntimeError: If processing exceeds time limit
        """
        if not self._processing_start_time:
            self._processing_start_time = datetime.now()
            return

        elapsed = (datetime.now() - self._processing_start_time).total_seconds()

        if elapsed > self.max_duration_seconds:
            raise RuntimeError(
                f"Processing duration {elapsed / 3600:.2f}h exceeds limit "
                f"{self.max_duration_seconds / 3600:.2f}h"
            )

        logger.debug(f"Duration check passed: {elapsed / 3600:.2f}h")

    def check_memory_budget(self):
        """
        Check if memory usage is within budget

        Raises:
            RuntimeError: If memory exceeds budget
        """
        if not hasattr(self.safety_manager, "check_memory_usage"):
            return

        try:
            self.safety_manager.check_memory_usage()
        except Exception as e:
            logger.error(f"Memory budget check failed: {e}")
            raise

    def save_checkpoint(self, state: Dict[str, Any]):
        """
        Save processing checkpoint

        Args:
            state: Processing state to checkpoint
        """
        # Check checkpoint interval
        elapsed = (datetime.now() - self._last_checkpoint_time).total_seconds()
        if elapsed < self.checkpoint_interval:
            logger.debug(f"Skipping checkpoint (interval not reached: {elapsed}s)")
            return

        try:
            # Generate checkpoint filename
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Prepare checkpoint data
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "state": self._serialize_state(state)
            }

            # Write checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            self._last_checkpoint_time = datetime.now()

            logger.info(f"Saved checkpoint: {checkpoint_file}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint

        Returns:
            Checkpoint state or None if no checkpoints exist
        """
        try:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"), reverse=True)

            if not checkpoints:
                logger.info("No checkpoints found")
                return None

            # Load latest checkpoint
            with open(checkpoints[0], 'r') as f:
                checkpoint_data = json.load(f)

            logger.info(f"Loaded checkpoint: {checkpoints[0]}")

            return checkpoint_data.get("state")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoints(self):
        """Clear all checkpoints"""
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))

            for checkpoint in checkpoints:
                checkpoint.unlink()

            logger.info(f"Cleared {len(checkpoints)} checkpoints")

        except Exception as e:
            logger.warning(f"Failed to clear checkpoints: {e}")

    def handle_emergency(self, error: Exception):
        """
        Handle emergency shutdown

        Args:
            error: Exception that triggered emergency
        """
        logger.error(f"Emergency shutdown triggered: {error}")

        try:
            # Notify safety manager
            if hasattr(self.safety_manager, "handle_emergency"):
                self.safety_manager.handle_emergency(str(error))

            # Save emergency state
            emergency_file = self.checkpoint_dir / f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            emergency_data = {
                "timestamp": datetime.now().isoformat(),
                "error": str(error),
                "error_type": type(error).__name__
            }

            with open(emergency_file, 'w') as f:
                json.dump(emergency_data, f, indent=2)

            logger.info(f"Saved emergency state: {emergency_file}")

        except Exception as e:
            logger.error(f"Failed to handle emergency: {e}")

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize state for checkpoint

        Args:
            state: Processing state

        Returns:
            Serialized state
        """
        serialized = {}

        for key, value in state.items():
            # Convert Path to string
            if isinstance(value, Path):
                serialized[key] = str(value)
            # Convert datetime to ISO format
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            # Skip non-serializable objects
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                logger.debug(f"Skipping non-serializable key: {key}")

        return serialized

    def get_stats(self) -> Dict[str, Any]:
        """
        Get safety statistics

        Returns:
            Safety statistics dictionary
        """
        stats = {
            "max_file_size_gb": self.max_file_size_bytes / 1024**3,
            "max_duration_hours": self.max_duration_seconds / 3600,
            "checkpoint_interval_seconds": self.checkpoint_interval,
            "checkpoint_count": len(list(self.checkpoint_dir.glob("checkpoint_*.json"))),
            "emergency_count": len(list(self.checkpoint_dir.glob("emergency_*.json")))
        }

        if self._processing_start_time:
            elapsed = (datetime.now() - self._processing_start_time).total_seconds()
            stats["current_duration_hours"] = elapsed / 3600

        return stats
