"""
Unit tests for State Manager.

Tests cover:
- Checkpoint save/load
- Checkpoint listing
- Checkpoint deletion
- Workflow metadata
- Automatic cleanup
- Transaction atomicity
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
import json
import time

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.orchestration.state_manager import StateManager, CheckpointMetadata


class TestStateManager:
    """Test State Manager functionality."""

    @pytest.fixture
    async def state_manager(self):
        """Create state manager with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(
                state_dir=Path(tmpdir),
                max_checkpoints=5
            )
            yield manager

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, state_manager):
        """Test basic save and load operations."""
        workflow_id = "test_workflow_1"
        state = {
            "current_task": "task_1",
            "completed_tasks": ["task_0"],
            "variables": {"x": 123, "y": "hello"}
        }

        # Save checkpoint
        metadata = await state_manager.save_checkpoint(workflow_id, state)
        assert metadata is not None
        assert metadata.workflow_id == workflow_id
        assert metadata.checkpoint_number == 1
        assert metadata.size_bytes > 0

        # Load checkpoint
        loaded_state = await state_manager.load_checkpoint(workflow_id)
        assert loaded_state == state

    @pytest.mark.asyncio
    async def test_multiple_checkpoints(self, state_manager):
        """Test saving multiple checkpoints."""
        workflow_id = "test_workflow_multi"

        # Save 3 checkpoints
        states = []
        for i in range(3):
            state = {"step": i, "data": f"checkpoint_{i}"}
            states.append(state)
            await state_manager.save_checkpoint(workflow_id, state)

        # Load latest (should be last one)
        loaded_state = await state_manager.load_checkpoint(workflow_id)
        assert loaded_state == states[2]

        # Load specific checkpoint
        loaded_state = await state_manager.load_checkpoint(workflow_id, checkpoint_number=1)
        assert loaded_state == states[0]

        loaded_state = await state_manager.load_checkpoint(workflow_id, checkpoint_number=2)
        assert loaded_state == states[1]

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, state_manager):
        """Test listing checkpoints."""
        workflow_id = "test_workflow_list"

        # Save 3 checkpoints
        for i in range(3):
            await state_manager.save_checkpoint(
                workflow_id,
                {"step": i},
                description=f"Step {i}"
            )

        # List checkpoints
        checkpoints = await state_manager.list_checkpoints(workflow_id)
        assert len(checkpoints) == 3

        # Should be newest first
        assert checkpoints[0].checkpoint_number == 3
        assert checkpoints[1].checkpoint_number == 2
        assert checkpoints[2].checkpoint_number == 1

        # Check descriptions
        assert checkpoints[2].description == "Step 0"

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, state_manager):
        """Test deleting specific checkpoint."""
        workflow_id = "test_workflow_delete"

        # Save 3 checkpoints
        for i in range(3):
            await state_manager.save_checkpoint(workflow_id, {"step": i})

        # Delete middle checkpoint
        deleted = await state_manager.delete_checkpoint(workflow_id, checkpoint_number=2)
        assert deleted is True

        # List remaining
        checkpoints = await state_manager.list_checkpoints(workflow_id)
        assert len(checkpoints) == 2
        assert checkpoints[0].checkpoint_number == 3
        assert checkpoints[1].checkpoint_number == 1

    @pytest.mark.asyncio
    async def test_delete_workflow(self, state_manager):
        """Test deleting entire workflow."""
        workflow_id = "test_workflow_delete_all"

        # Save checkpoints
        for i in range(3):
            await state_manager.save_checkpoint(workflow_id, {"step": i})

        # Delete workflow
        deleted = await state_manager.delete_workflow(workflow_id)
        assert deleted is True

        # Verify deleted
        checkpoints = await state_manager.list_checkpoints(workflow_id)
        assert len(checkpoints) == 0

        loaded_state = await state_manager.load_checkpoint(workflow_id)
        assert loaded_state is None

    @pytest.mark.asyncio
    async def test_automatic_cleanup(self, state_manager):
        """Test automatic cleanup of old checkpoints."""
        workflow_id = "test_workflow_cleanup"

        # Save more than max_checkpoints (5)
        for i in range(8):
            await state_manager.save_checkpoint(workflow_id, {"step": i})

        # Should keep only last 5
        checkpoints = await state_manager.list_checkpoints(workflow_id)
        assert len(checkpoints) == 5

        # Should keep newest (4-8)
        assert checkpoints[0].checkpoint_number == 8
        assert checkpoints[4].checkpoint_number == 4

    @pytest.mark.asyncio
    async def test_workflow_metadata(self, state_manager):
        """Test workflow metadata tracking."""
        workflow_id = "test_workflow_metadata"

        # Save checkpoints
        for i in range(3):
            await state_manager.save_checkpoint(workflow_id, {"step": i})
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

        # Get workflow status
        status = await state_manager.get_workflow_status(workflow_id)
        assert status is not None
        assert status["workflow_id"] == workflow_id
        assert status["checkpoint_count"] == 3
        assert status["status"] == "active"
        assert status["created_at"] > 0
        assert status["updated_at"] >= status["created_at"]

    @pytest.mark.asyncio
    async def test_nonexistent_workflow(self, state_manager):
        """Test loading nonexistent workflow."""
        loaded_state = await state_manager.load_checkpoint("nonexistent_workflow")
        assert loaded_state is None

        status = await state_manager.get_workflow_status("nonexistent_workflow")
        assert status is None

    @pytest.mark.asyncio
    async def test_checkpoint_with_complex_state(self, state_manager):
        """Test checkpoint with complex nested state."""
        workflow_id = "test_workflow_complex"

        complex_state = {
            "tasks": [
                {"id": "task_1", "status": "completed", "result": {"value": 123}},
                {"id": "task_2", "status": "running", "result": None}
            ],
            "variables": {
                "nested": {
                    "deep": {
                        "value": [1, 2, 3, 4, 5]
                    }
                }
            },
            "metadata": {
                "started_at": time.time(),
                "user": "test_user"
            }
        }

        # Save and load
        await state_manager.save_checkpoint(workflow_id, complex_state)
        loaded_state = await state_manager.load_checkpoint(workflow_id)

        assert loaded_state == complex_state
        assert loaded_state["tasks"][0]["result"]["value"] == 123
        assert loaded_state["variables"]["nested"]["deep"]["value"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, state_manager):
        """Test concurrent checkpoint saves."""
        workflow_id = "test_workflow_concurrent"

        # Save multiple checkpoints concurrently
        tasks = []
        for i in range(5):
            task = state_manager.save_checkpoint(workflow_id, {"step": i})
            tasks.append(task)

        # Wait for all
        await asyncio.gather(*tasks)

        # Verify all saved
        checkpoints = await state_manager.list_checkpoints(workflow_id)
        assert len(checkpoints) == 5

    @pytest.mark.asyncio
    async def test_invalid_json_serialization(self, state_manager):
        """Test handling of non-serializable state."""
        workflow_id = "test_workflow_invalid"

        # State with non-serializable object
        class NonSerializable:
            pass

        invalid_state = {"obj": NonSerializable()}

        # Should raise ValueError
        with pytest.raises(ValueError, match="Failed to serialize"):
            await state_manager.save_checkpoint(workflow_id, invalid_state)

    @pytest.mark.asyncio
    async def test_checkpoint_file_integrity(self, state_manager):
        """Test checkpoint file structure."""
        workflow_id = "test_workflow_integrity"
        state = {"test": "data"}

        # Save checkpoint
        metadata = await state_manager.save_checkpoint(workflow_id, state)

        # Verify file exists
        state_path = Path(metadata.state_path)
        assert state_path.exists()

        # Verify file content
        with open(state_path) as f:
            file_content = json.load(f)
            assert file_content == state

        # Verify file size matches metadata
        actual_size = state_path.stat().st_size
        assert metadata.size_bytes == actual_size

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, state_manager):
        """Test loading specific checkpoint number."""
        workflow_id = "test_workflow_specific"

        # Save checkpoints with distinct data
        states = {
            1: {"version": "1.0", "data": "first"},
            2: {"version": "2.0", "data": "second"},
            3: {"version": "3.0", "data": "third"}
        }

        for i in [1, 2, 3]:
            await state_manager.save_checkpoint(workflow_id, states[i])

        # Load specific checkpoints
        for i in [1, 2, 3]:
            loaded = await state_manager.load_checkpoint(workflow_id, checkpoint_number=i)
            assert loaded == states[i]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, state_manager):
        """Test deleting nonexistent checkpoint."""
        workflow_id = "test_workflow_delete_nonexistent"

        # Try to delete nonexistent checkpoint
        deleted = await state_manager.delete_checkpoint(workflow_id, checkpoint_number=999)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_multiple_workflows(self, state_manager):
        """Test managing multiple workflows simultaneously."""
        workflows = {
            "workflow_1": [{"step": 1}, {"step": 2}],
            "workflow_2": [{"step": "a"}, {"step": "b"}, {"step": "c"}],
            "workflow_3": [{"step": "x"}]
        }

        # Save checkpoints for each workflow
        for wf_id, states in workflows.items():
            for state in states:
                await state_manager.save_checkpoint(wf_id, state)

        # Verify each workflow
        checkpoints_1 = await state_manager.list_checkpoints("workflow_1")
        assert len(checkpoints_1) == 2

        checkpoints_2 = await state_manager.list_checkpoints("workflow_2")
        assert len(checkpoints_2) == 3

        checkpoints_3 = await state_manager.list_checkpoints("workflow_3")
        assert len(checkpoints_3) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
