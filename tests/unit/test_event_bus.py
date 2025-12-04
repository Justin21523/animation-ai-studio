"""
Unit tests for Event Bus system.

Tests cover:
- Basic pub/sub
- Wildcard subscriptions
- Priority handling
- Error isolation
- Event history
- Async/sync handler support
"""

import asyncio
import pytest
import time
from pathlib import Path
import tempfile
import json
from typing import List

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.orchestration.event_bus import EventBus, Event, Priority


class TestEventBus:
    """Test Event Bus functionality."""

    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus(
                max_queue_size=100,
                max_history=50,
                persist_events=True,
                persist_path=Path(tmpdir) / "events.jsonl",
                enable_statistics=True
            )
            await bus.start()
            yield bus
            await bus.stop()

    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self, event_bus):
        """Test basic event publish and subscribe."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        # Subscribe
        sub_id = await event_bus.subscribe("test.event", handler)
        assert sub_id is not None

        # Publish
        await event_bus.publish("test.event", {"message": "hello"}, wait=True)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify
        assert len(received_events) == 1
        assert received_events[0].event_type == "test.event"
        assert received_events[0].data["message"] == "hello"

    @pytest.mark.asyncio
    async def test_wildcard_subscriptions(self, event_bus):
        """Test wildcard pattern matching."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event.event_type)

        # Subscribe with wildcard
        await event_bus.subscribe("task.*", handler)

        # Publish multiple events
        await event_bus.publish("task.started", {}, wait=True)
        await event_bus.publish("task.completed", {}, wait=True)
        await event_bus.publish("workflow.started", {}, wait=True)

        await asyncio.sleep(0.2)

        # Should receive only task.* events
        assert len(received_events) == 2
        assert "task.started" in received_events
        assert "task.completed" in received_events
        assert "workflow.started" not in received_events

    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus):
        """Test handlers are called in priority order."""
        call_order = []

        async def high_handler(event: Event):
            await asyncio.sleep(0.01)  # Small delay
            call_order.append("high")

        async def normal_handler(event: Event):
            await asyncio.sleep(0.01)
            call_order.append("normal")

        async def low_handler(event: Event):
            await asyncio.sleep(0.01)
            call_order.append("low")

        # Subscribe in reverse order
        await event_bus.subscribe("test.priority", low_handler, Priority.LOW)
        await event_bus.subscribe("test.priority", normal_handler, Priority.NORMAL)
        await event_bus.subscribe("test.priority", high_handler, Priority.HIGH)

        # Publish
        await event_bus.publish("test.priority", {}, wait=True)
        await asyncio.sleep(0.2)

        # Verify order: high, normal, low
        assert call_order == ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_error_isolation(self, event_bus):
        """Test that handler errors don't break other handlers."""
        successful_calls = []

        async def failing_handler(event: Event):
            raise ValueError("Handler error")

        async def successful_handler(event: Event):
            successful_calls.append(event.event_type)

        # Subscribe both handlers
        await event_bus.subscribe("test.error", failing_handler)
        await event_bus.subscribe("test.error", successful_handler)

        # Publish
        await event_bus.publish("test.error", {}, wait=True)
        await asyncio.sleep(0.2)

        # Successful handler should still be called
        assert len(successful_calls) == 1
        assert successful_calls[0] == "test.error"

        # Check error was tracked
        stats = event_bus.get_statistics()
        assert stats["handler_errors"] >= 1

    @pytest.mark.asyncio
    async def test_sync_handler_support(self, event_bus):
        """Test that synchronous handlers work."""
        received = []

        def sync_handler(event: Event):
            """Synchronous handler."""
            received.append(event.event_type)

        await event_bus.subscribe("test.sync", sync_handler)
        await event_bus.publish("test.sync", {}, wait=True)
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0] == "test.sync"

    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history tracking."""
        # Publish multiple events
        for i in range(5):
            await event_bus.publish(f"test.history.{i}", {"index": i}, wait=True)

        await asyncio.sleep(0.2)

        # Get all history
        history = event_bus.get_history()
        assert len(history) >= 5

        # Get filtered history
        history_filtered = event_bus.get_history(event_type="test.history.*", limit=3)
        assert len(history_filtered) == 3

        # Get recent history
        now = time.time()
        history_recent = event_bus.get_history(since=now - 10.0)
        assert len(history_recent) >= 5

    @pytest.mark.asyncio
    async def test_history_size_limit(self):
        """Test history respects max size."""
        bus = EventBus(max_history=10, persist_events=False)

        # Publish more than max
        for i in range(20):
            await bus.publish(f"test.{i}", {})

        # History should be trimmed
        history = bus.get_history()
        assert len(history) == 10

    @pytest.mark.asyncio
    async def test_event_persistence(self):
        """Test events are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "events.jsonl"
            bus = EventBus(persist_events=True, persist_path=persist_path)

            # Publish events
            await bus.publish("test.persist", {"message": "test1"})
            await bus.publish("test.persist", {"message": "test2"})

            await asyncio.sleep(0.1)

            # Verify file exists and contains events
            assert persist_path.exists()

            with open(persist_path) as f:
                lines = f.readlines()
                assert len(lines) == 2

                event1 = json.loads(lines[0])
                assert event1["event_type"] == "test.persist"
                assert event1["data"]["message"] == "test1"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing handlers."""
        received = []

        async def handler(event: Event):
            received.append(event.event_type)

        # Subscribe
        sub_id = await event_bus.subscribe("test.unsub", handler)

        # Publish (should receive)
        await event_bus.publish("test.unsub", {}, wait=True)
        await asyncio.sleep(0.1)
        assert len(received) == 1

        # Unsubscribe
        removed = await event_bus.unsubscribe(sub_id)
        assert removed is True

        # Publish again (should not receive)
        await event_bus.publish("test.unsub", {}, wait=True)
        await asyncio.sleep(0.1)
        assert len(received) == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_unsubscribe_pattern(self, event_bus):
        """Test unsubscribing by pattern."""
        received = []

        async def handler(event: Event):
            received.append(event.event_type)

        # Subscribe multiple handlers to same pattern
        await event_bus.subscribe("test.pattern", handler)
        await event_bus.subscribe("test.pattern", handler)

        # Publish (should receive twice)
        await event_bus.publish("test.pattern", {}, wait=True)
        await asyncio.sleep(0.1)
        assert len(received) == 2

        # Unsubscribe pattern
        removed_count = await event_bus.unsubscribe_pattern("test.pattern")
        assert removed_count == 2

        # Publish again (should not receive)
        received.clear()
        await event_bus.publish("test.pattern", {}, wait=True)
        await asyncio.sleep(0.1)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_wait_for_event(self, event_bus):
        """Test waiting for specific event."""
        # Publish event after delay
        async def delayed_publish():
            await asyncio.sleep(0.2)
            await event_bus.publish("test.wait", {"value": 123})

        asyncio.create_task(delayed_publish())

        # Wait for event
        event = await event_bus.wait_for_event("test.wait", timeout=1.0)
        assert event is not None
        assert event.event_type == "test.wait"
        assert event.data["value"] == 123

    @pytest.mark.asyncio
    async def test_wait_for_event_timeout(self, event_bus):
        """Test wait_for_event timeout."""
        # Don't publish event
        event = await event_bus.wait_for_event("test.timeout", timeout=0.1)
        assert event is None

    @pytest.mark.asyncio
    async def test_wait_for_event_predicate(self, event_bus):
        """Test wait_for_event with predicate."""
        # Publish multiple events
        async def publish_multiple():
            await asyncio.sleep(0.1)
            await event_bus.publish("test.pred", {"value": 1})
            await event_bus.publish("test.pred", {"value": 2})
            await event_bus.publish("test.pred", {"value": 3})

        asyncio.create_task(publish_multiple())

        # Wait for specific value
        event = await event_bus.wait_for_event(
            "test.pred",
            timeout=1.0,
            predicate=lambda e: e.data["value"] == 2
        )
        assert event is not None
        assert event.data["value"] == 2

    @pytest.mark.asyncio
    async def test_statistics(self, event_bus):
        """Test statistics tracking."""
        async def handler(event: Event):
            pass

        # Subscribe
        await event_bus.subscribe("test.stats", handler)

        # Publish
        await event_bus.publish("test.stats", {}, wait=True)
        await asyncio.sleep(0.2)

        # Get statistics
        stats = event_bus.get_statistics()
        assert stats["events_published"] >= 1
        assert stats["events_processed"] >= 1
        assert stats["subscription_count"] >= 1
        assert "queue_size" in stats
        assert "history_size" in stats

    @pytest.mark.asyncio
    async def test_clear_history(self, event_bus):
        """Test clearing event history."""
        # Publish events
        await event_bus.publish("test.clear", {})
        await asyncio.sleep(0.1)

        # Verify history
        history = event_bus.get_history()
        assert len(history) > 0

        # Clear
        event_bus.clear_history()

        # Verify cleared
        history = event_bus.get_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, event_bus):
        """Test graceful shutdown."""
        received = []

        async def slow_handler(event: Event):
            await asyncio.sleep(0.5)
            received.append(event.event_type)

        await event_bus.subscribe("test.shutdown", slow_handler)

        # Publish event
        await event_bus.publish("test.shutdown", {}, wait=True)

        # Stop immediately
        await event_bus.stop(timeout=1.0)

        # Should have processed event
        assert len(received) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
