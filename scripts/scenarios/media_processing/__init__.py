"""
Media Processing Automation Scenario

CPU-only media analysis and processing for Animation AI Studio.
Provides video, audio, and subtitle processing using ffmpeg/ffprobe.

This scenario provides:
- Metadata extraction (ffprobe-based)
- Scene detection (content-based, histogram)
- Quality assessment and recommendations
- Frame extraction with quality control
- Audio extraction and normalization
- Subtitle processing
- Batch processing workflows

Author: Animation AI Studio
Date: 2025-12-03
Version: 1.0.0
"""

from .common import (
    # Enums
    MediaType,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    ProcessingOperation,
    QualityLevel,
    ProcessingStatus,
    SceneDetectionMethod,

    # Dataclasses
    VideoStreamInfo,
    AudioStreamInfo,
    SubtitleStreamInfo,
    MediaMetadata,
    SceneInfo,
    ProcessingParameters,
    ProcessingTask,
    ProcessingResult,
    QualityMetrics,
    MediaAnalysisReport,
    BatchProcessingJob
)

# Phase 2: Analyzers
from .analyzers import MetadataExtractor, SceneDetector, QualityAnalyzer

# Phase 3: Processors
from .processors import VideoProcessor, AudioProcessor, SubtitleProcessor

# Phase 4: Main Orchestrator
from .processor import MediaProcessor

# Phase 4: Integration
from .integration import MediaProcessingEventAdapter, MediaProcessingSafetyAdapter

__version__ = "1.0.0"

__all__ = [
    # Enums
    "MediaType",
    "VideoCodec",
    "AudioCodec",
    "ContainerFormat",
    "ProcessingOperation",
    "QualityLevel",
    "ProcessingStatus",
    "SceneDetectionMethod",

    # Dataclasses
    "VideoStreamInfo",
    "AudioStreamInfo",
    "SubtitleStreamInfo",
    "MediaMetadata",
    "SceneInfo",
    "ProcessingParameters",
    "ProcessingTask",
    "ProcessingResult",
    "QualityMetrics",
    "MediaAnalysisReport",
    "BatchProcessingJob",

    # Analyzers
    "MetadataExtractor",
    "SceneDetector",
    "QualityAnalyzer",

    # Processors
    "VideoProcessor",
    "AudioProcessor",
    "SubtitleProcessor",

    # Main Orchestrator
    "MediaProcessor",

    # Integration
    "MediaProcessingEventAdapter",
    "MediaProcessingSafetyAdapter"
]
