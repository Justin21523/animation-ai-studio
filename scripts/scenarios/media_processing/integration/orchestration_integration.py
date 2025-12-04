"""
Media Processing Orchestration Integration

EventBus adapter for media processing scenario.
Bridges MediaProcessor with the Orchestration Layer.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MediaProcessingEventAdapter:
    """
    Event adapter for media processing integration

    Features:
    - Bridges MediaProcessor with EventBus
    - Translates media processing events to orchestration events
    - Enables workflow orchestration and monitoring
    - Supports event filtering and routing

    Example:
        from scripts.automation.orchestration.event_bus import EventBus

        event_bus = EventBus()
        adapter = MediaProcessingEventAdapter(event_bus)

        # Register event handlers
        adapter.register_handlers()

        # Create MediaProcessor with EventBus
        processor = MediaProcessor(event_bus=event_bus)

        # Events are automatically emitted and handled
        result = processor.analyze_media(Path("video.mp4"))
    """

    # Event type mappings
    EVENT_MAPPINGS = {
        # Initialization
        "media_processor_initialized": "scenario.media_processing.initialized",

        # Analysis events
        "media_analysis_started": "scenario.media_processing.analysis.started",
        "media_analysis_completed": "scenario.media_processing.analysis.completed",
        "media_analysis_failed": "scenario.media_processing.analysis.failed",

        # Video events
        "video_transcode_started": "scenario.media_processing.transcode.started",
        "video_transcode_completed": "scenario.media_processing.transcode.completed",
        "video_transcode_failed": "scenario.media_processing.transcode.failed",

        # Frame extraction events
        "frame_extraction_started": "scenario.media_processing.frames.started",
        "frame_extraction_completed": "scenario.media_processing.frames.completed",
        "frame_extraction_failed": "scenario.media_processing.frames.failed",

        # Audio events
        "audio_extraction_started": "scenario.media_processing.audio.extract.started",
        "audio_extraction_completed": "scenario.media_processing.audio.extract.completed",
        "audio_extraction_failed": "scenario.media_processing.audio.extract.failed",
        "audio_normalization_started": "scenario.media_processing.audio.normalize.started",
        "audio_normalization_completed": "scenario.media_processing.audio.normalize.completed",
        "audio_normalization_failed": "scenario.media_processing.audio.normalize.failed",

        # Subtitle events
        "subtitle_extraction_started": "scenario.media_processing.subtitles.started",
        "subtitle_extraction_completed": "scenario.media_processing.subtitles.completed",
        "subtitle_extraction_failed": "scenario.media_processing.subtitles.failed",

        # Workflow events
        "media_workflow_started": "scenario.media_processing.workflow.started",
        "media_workflow_completed": "scenario.media_processing.workflow.completed",
        "media_workflow_failed": "scenario.media_processing.workflow.failed"
    }

    def __init__(self, event_bus: Any):
        """
        Initialize event adapter

        Args:
            event_bus: EventBus instance from orchestration layer
        """
        self.event_bus = event_bus
        self._handlers_registered = False

        logger.info("MediaProcessingEventAdapter initialized")

    def register_handlers(self):
        """Register all event handlers with EventBus"""
        if self._handlers_registered:
            logger.warning("Event handlers already registered")
            return

        # Register handlers for each event type
        for media_event, orchestration_event in self.EVENT_MAPPINGS.items():
            self.event_bus.on(
                media_event,
                lambda event, data, oe=orchestration_event: self._handle_event(oe, data)
            )

        self._handlers_registered = True
        logger.info(f"Registered {len(self.EVENT_MAPPINGS)} media processing event handlers")

    def _handle_event(self, orchestration_event: str, data: Dict[str, Any]):
        """
        Handle media processing event and emit orchestration event

        Args:
            orchestration_event: Orchestration event type
            data: Event data
        """
        try:
            # Enrich event data
            enriched_data = self._enrich_event_data(data)

            # Emit orchestration event
            self.event_bus.emit(orchestration_event, enriched_data)

            logger.debug(f"Translated event: {orchestration_event}")

        except Exception as e:
            logger.error(f"Failed to handle event {orchestration_event}: {e}")

    def _enrich_event_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich event data with additional context

        Args:
            data: Original event data

        Returns:
            Enriched event data
        """
        enriched = data.copy()

        # Add scenario identifier
        enriched["scenario"] = "media_processing"

        # Convert Path objects to strings
        for key, value in enriched.items():
            if isinstance(value, Path):
                enriched[key] = str(value)

        return enriched

    def create_media_processor(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        safety_manager: Optional[Any] = None
    ):
        """
        Create MediaProcessor with EventBus integration

        Args:
            ffmpeg_path: Path to ffmpeg
            ffprobe_path: Path to ffprobe
            safety_manager: Optional SafetyManager

        Returns:
            MediaProcessor instance with EventBus
        """
        from ..processor import MediaProcessor

        # Ensure handlers are registered
        if not self._handlers_registered:
            self.register_handlers()

        # Create processor with event bus
        processor = MediaProcessor(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            event_bus=self.event_bus,
            safety_manager=safety_manager
        )

        logger.info("Created MediaProcessor with EventBus integration")

        return processor

    def get_event_stats(self) -> Dict[str, int]:
        """
        Get event statistics

        Returns:
            Event counts by type
        """
        if not hasattr(self.event_bus, "get_event_stats"):
            logger.warning("EventBus does not support event statistics")
            return {}

        return self.event_bus.get_event_stats()
