"""
Event Bus System for Orchestration Layer

Lightweight pub/sub event bus using asyncio.Queue (no external dependencies).
Supports wildcard subscriptions, priority handlers, event history, and error isolation.

Usage:
    bus = EventBus()
    await bus.subscribe("task.started", my_handler)
    await bus.subscribe("task.*", wildcard_handler)
    await bus.publish("task.started", {"task_id": "123"})
"""

import asyncio
import fnmatch
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from enum import IntEnum
import traceback

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Handler priority levels (higher = executed first)."""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 0


@dataclass
class Event:
    """Event data structure."""
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time() * 1000000)}")
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Subscription:
    """Subscription information."""
    pattern: str
    handler: Callable
    priority: Priority
    is_async: bool
    subscription_id: str = field(default_factory=lambda: f"sub_{int(time.time() * 1000000)}")


class EventBus:
    """
    Central message bus for module communication.

    Features:
    - Async pub/sub pattern via asyncio.Queue
    - Wildcard subscriptions (e.g., "task.*", "*.completed")
    - Priority handlers (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
    - Event history for debugging (configurable max size)
    - Error isolation (one handler failure doesn't break others)
    - Event persistence to disk (optional)
    - Supports both sync and async handlers

    Example:
        bus = EventBus(max_history=1000, persist_events=True)

        # Subscribe to specific events
        await bus.subscribe("task.started", on_task_started)

        # Subscribe to all task events
        await bus.subscribe("task.*", on_any_task_event, priority=Priority.HIGH)

        # Publish event
        await bus.publish("task.started", {"task_id": "123", "name": "dataset_scan"})

        # Get event history
        history = bus.get_history(event_type="task.started", limit=10)
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        max_history: int = 10000,
        persist_events: bool = True,
        persist_path: Optional[Path] = None,
        enable_statistics: bool = True
    ):
        """
        Initialize Event Bus.

        Args:
            max_queue_size: Maximum events in processing queue
            max_history: Maximum events to keep in history (0 = unlimited)
            persist_events: Whether to persist events to disk
            persist_path: Path to event log file (default: /tmp/automation/events.jsonl)
            enable_statistics: Track event statistics
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._subscriptions: List[Subscription] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._persist_events = persist_events
        self._enable_statistics = enable_statistics

        # Setup persistence
        if persist_path is None:
            persist_path = Path("/tmp/automation/events.jsonl")
        self._persist_path = persist_path
        if self._persist_events:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "handler_errors": 0,
            "subscription_count": 0,
        }

        # Event processing task
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info(f"EventBus initialized: queue_size={max_queue_size}, history={max_history}, persist={persist_events}")

    async def start(self):
        """Start event processing loop."""
        if self._running:
            logger.warning("EventBus already running")
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")

    async def stop(self, timeout: float = 5.0):
        """
        Stop event processing loop gracefully.

        Args:
            timeout: Maximum time to wait for processing to complete
        """
        if not self._running:
            return

        self._running = False

        # Wait for processing task to complete
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"EventBus processing task did not complete within {timeout}s, cancelling")
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass

        logger.info("EventBus stopped")

    async def subscribe(
        self,
        pattern: str,
        handler: Callable,
        priority: Priority = Priority.NORMAL
    ) -> str:
        """
        Subscribe to events matching pattern.

        Args:
            pattern: Event type pattern (supports wildcards: "task.*", "*.completed")
            handler: Callback function (can be sync or async)
            priority: Handler priority (higher priority = executed first)

        Returns:
            Subscription ID for unsubscribing

        Example:
            # Subscribe to all task events
            sub_id = await bus.subscribe("task.*", my_handler, priority=Priority.HIGH)

            # Later unsubscribe
            await bus.unsubscribe(sub_id)
        """
        # Detect if handler is async
        is_async = asyncio.iscoroutinefunction(handler)

        subscription = Subscription(
            pattern=pattern,
            handler=handler,
            priority=priority,
            is_async=is_async
        )

        async with self._lock:
            self._subscriptions.append(subscription)
            # Sort by priority (highest first)
            self._subscriptions.sort(key=lambda s: s.priority, reverse=True)
            self._stats["subscription_count"] = len(self._subscriptions)

        logger.debug(f"Subscribed: pattern='{pattern}', priority={priority.name}, async={is_async}, id={subscription.subscription_id}")
        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe by subscription ID.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if subscription was found and removed
        """
        async with self._lock:
            original_count = len(self._subscriptions)
            self._subscriptions = [s for s in self._subscriptions if s.subscription_id != subscription_id]
            removed = len(self._subscriptions) < original_count
            self._stats["subscription_count"] = len(self._subscriptions)

        if removed:
            logger.debug(f"Unsubscribed: id={subscription_id}")
        else:
            logger.warning(f"Subscription not found: id={subscription_id}")

        return removed

    async def unsubscribe_pattern(self, pattern: str) -> int:
        """
        Unsubscribe all handlers matching pattern.

        Args:
            pattern: Event type pattern to unsubscribe

        Returns:
            Number of subscriptions removed
        """
        async with self._lock:
            original_count = len(self._subscriptions)
            self._subscriptions = [s for s in self._subscriptions if s.pattern != pattern]
            removed = original_count - len(self._subscriptions)
            self._stats["subscription_count"] = len(self._subscriptions)

        logger.debug(f"Unsubscribed pattern: pattern='{pattern}', removed={removed}")
        return removed

    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: Optional[str] = None,
        wait: bool = False
    ):
        """
        Publish event to all subscribers.

        Args:
            event_type: Event type (e.g., "task.started", "workflow.completed")
            data: Event data dictionary
            source: Optional source identifier
            wait: If True, wait for event to be processed (blocking)

        Example:
            await bus.publish("task.started", {
                "task_id": "123",
                "workflow_id": "wf_456",
                "task_name": "scan_dataset"
            })
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source
        )

        # Add to queue
        try:
            if wait:
                await self._queue.put(event)
            else:
                # Non-blocking put
                self._queue.put_nowait(event)

            self._stats["events_published"] += 1
            logger.debug(f"Published event: type='{event_type}', id={event.event_id}")

        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: type='{event_type}'")
            raise RuntimeError(f"Event queue full (max_size={self._queue.maxsize})")

        # If not started, process immediately
        if not self._running:
            await self._dispatch_event(event)

    async def _process_events(self):
        """Background task to process events from queue."""
        logger.info("Event processing loop started")

        while self._running:
            try:
                # Wait for event with timeout to allow checking _running flag
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                # No event available, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)

        # Process remaining events in queue
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch_event(event)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error processing remaining events: {e}", exc_info=True)

        logger.info("Event processing loop stopped")

    async def _dispatch_event(self, event: Event):
        """
        Dispatch event to all matching subscribers.

        Handlers are called in priority order. If a handler fails, the error is
        logged but other handlers continue to execute (error isolation).
        """
        # Add to history
        self._add_to_history(event)

        # Persist to disk
        if self._persist_events:
            self._persist_event(event)

        # Get matching subscriptions
        matching_subs = self._get_matching_subscriptions(event.event_type)

        if not matching_subs:
            logger.debug(f"No subscribers for event: type='{event.event_type}'")
            return

        logger.debug(f"Dispatching event to {len(matching_subs)} handlers: type='{event.event_type}'")

        # Execute handlers (already sorted by priority)
        for sub in matching_subs:
            try:
                if sub.is_async:
                    await sub.handler(event)
                else:
                    # Run sync handler in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, sub.handler, event)
            except Exception as e:
                # Error isolation: log but continue with other handlers
                self._stats["handler_errors"] += 1
                logger.error(
                    f"Handler error: pattern='{sub.pattern}', event='{event.event_type}', "
                    f"error={type(e).__name__}: {e}",
                    exc_info=True
                )

        self._stats["events_processed"] += 1

    def _get_matching_subscriptions(self, event_type: str) -> List[Subscription]:
        """Get all subscriptions matching event type (supports wildcards)."""
        matching = []
        for sub in self._subscriptions:
            if fnmatch.fnmatch(event_type, sub.pattern):
                matching.append(sub)
        return matching

    def _add_to_history(self, event: Event):
        """Add event to history with size limit."""
        self._history.append(event)

        # Trim history if needed
        if self._max_history > 0 and len(self._history) > self._max_history:
            # Remove oldest events
            excess = len(self._history) - self._max_history
            self._history = self._history[excess:]

    def _persist_event(self, event: Event):
        """Persist event to disk (JSONL format)."""
        try:
            with open(self._persist_path, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist event: {e}")

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[float] = None
    ) -> List[Event]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type (supports wildcards)
            limit: Maximum number of events to return (most recent)
            since: Only return events after this timestamp

        Returns:
            List of events (newest first)

        Example:
            # Get last 10 task events
            history = bus.get_history(event_type="task.*", limit=10)

            # Get all events in last 5 minutes
            history = bus.get_history(since=time.time() - 300)
        """
        # Filter by timestamp
        events = self._history
        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        # Filter by event type
        if event_type is not None:
            events = [e for e in events if fnmatch.fnmatch(e.event_type, event_type)]

        # Sort newest first
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if limit is not None:
            events = events[:limit]

        return events

    def clear_history(self):
        """Clear event history."""
        self._history.clear()
        logger.info("Event history cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Dictionary with statistics:
            - events_published: Total events published
            - events_processed: Total events processed
            - handler_errors: Total handler errors
            - subscription_count: Current number of subscriptions
            - queue_size: Current queue size
            - history_size: Current history size
        """
        if not self._enable_statistics:
            return {}

        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "history_size": len(self._history),
        }

    async def wait_for_event(
        self,
        event_type: str,
        timeout: Optional[float] = None,
        predicate: Optional[Callable[[Event], bool]] = None
    ) -> Optional[Event]:
        """
        Wait for specific event to occur.

        Args:
            event_type: Event type to wait for (supports wildcards)
            timeout: Maximum time to wait (None = wait forever)
            predicate: Optional function to filter events

        Returns:
            Event object if received, None if timeout

        Example:
            # Wait for task completion
            event = await bus.wait_for_event("task.completed", timeout=60.0)
            if event:
                print(f"Task completed: {event.data}")
        """
        future = asyncio.Future()

        async def waiter(event: Event):
            if predicate is None or predicate(event):
                if not future.done():
                    future.set_result(event)

        # Subscribe temporarily
        sub_id = await self.subscribe(event_type, waiter, priority=Priority.CRITICAL)

        try:
            # Wait for event
            if timeout is not None:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            return result
        except asyncio.TimeoutError:
            return None
        finally:
            # Unsubscribe
            await self.unsubscribe(sub_id)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"EventBus(subscriptions={stats.get('subscription_count', 0)}, "
            f"queue={stats.get('queue_size', 0)}, "
            f"history={stats.get('history_size', 0)})"
        )
