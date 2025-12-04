"""
Media Processing Common Data Structures

Core enums and dataclasses for media processing automation scenario.
Designed for CPU-only processing using ffmpeg/ffprobe.

Author: Animation AI Studio
Date: 2025-12-03
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


# ============================================================================
# Enums
# ============================================================================

class MediaType(Enum):
    """Media file type classification"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    SUBTITLE = "subtitle"
    UNKNOWN = "unknown"


class VideoCodec(Enum):
    """Video codec types (CPU-decodable)"""
    H264 = "h264"          # AVC - most compatible
    H265 = "h265"          # HEVC - better compression
    VP9 = "vp9"            # WebM standard
    AV1 = "av1"            # Next-gen codec
    MPEG4 = "mpeg4"        # Legacy
    MPEG2 = "mpeg2"        # DVD standard
    THEORA = "theora"      # Ogg video
    PRORES = "prores"      # Production quality
    UNKNOWN = "unknown"


class AudioCodec(Enum):
    """Audio codec types"""
    AAC = "aac"            # Most common
    MP3 = "mp3"            # Legacy standard
    OPUS = "opus"          # Modern, efficient
    VORBIS = "vorbis"      # Ogg audio
    FLAC = "flac"          # Lossless
    PCM = "pcm"            # Raw audio
    AC3 = "ac3"            # Dolby Digital
    EAC3 = "eac3"          # Dolby Digital Plus
    UNKNOWN = "unknown"


class ContainerFormat(Enum):
    """Media container formats"""
    MP4 = "mp4"            # Most compatible
    MKV = "mkv"            # Matroska - flexible
    WEBM = "webm"          # Web standard
    MOV = "mov"            # QuickTime
    AVI = "avi"            # Legacy
    FLV = "flv"            # Flash video
    TS = "ts"              # MPEG transport stream
    M4A = "m4a"            # Audio-only MP4
    OGG = "ogg"            # Ogg container
    WAV = "wav"            # Raw audio
    UNKNOWN = "unknown"


class ProcessingOperation(Enum):
    """Media processing operations (CPU-only)"""
    EXTRACT_METADATA = "extract_metadata"
    DETECT_SCENES = "detect_scenes"
    EXTRACT_FRAMES = "extract_frames"
    EXTRACT_AUDIO = "extract_audio"
    TRANSCODE_VIDEO = "transcode_video"
    TRANSCODE_AUDIO = "transcode_audio"
    NORMALIZE_AUDIO = "normalize_audio"
    EXTRACT_SUBTITLES = "extract_subtitles"
    CONVERT_SUBTITLES = "convert_subtitles"
    ANALYZE_QUALITY = "analyze_quality"
    BATCH_PROCESS = "batch_process"


class QualityLevel(Enum):
    """Video quality presets (CPU encoding)"""
    LOW = "low"            # Fast encoding, lower quality
    MEDIUM = "medium"      # Balanced
    HIGH = "high"          # Slower, better quality
    ULTRA = "ultra"        # Slowest, best quality


class ProcessingStatus(Enum):
    """Processing task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SceneDetectionMethod(Enum):
    """Scene detection algorithms"""
    CONTENT_BASED = "content_based"      # ffmpeg select filter
    HISTOGRAM = "histogram"              # Histogram difference
    SHOT_BOUNDARY = "shot_boundary"      # Shot boundary detection
    HYBRID = "hybrid"                    # Combination of methods


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class VideoStreamInfo:
    """Video stream information from ffprobe"""
    index: int
    codec: VideoCodec
    width: int
    height: int
    fps: float
    bitrate: Optional[int] = None  # bits per second
    duration: Optional[float] = None  # seconds
    frame_count: Optional[int] = None
    pixel_format: Optional[str] = None
    color_space: Optional[str] = None

    @property
    def resolution(self) -> str:
        """Resolution string (e.g., '1920x1080')"""
        return f"{self.width}x{self.height}"

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width/height)"""
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class AudioStreamInfo:
    """Audio stream information from ffprobe"""
    index: int
    codec: AudioCodec
    sample_rate: int  # Hz
    channels: int
    bitrate: Optional[int] = None  # bits per second
    duration: Optional[float] = None  # seconds
    language: Optional[str] = None
    channel_layout: Optional[str] = None


@dataclass
class SubtitleStreamInfo:
    """Subtitle stream information from ffprobe"""
    index: int
    format: str  # srt, ass, vtt, etc.
    language: Optional[str] = None
    title: Optional[str] = None
    encoding: Optional[str] = None


@dataclass
class MediaMetadata:
    """Complete media file metadata"""
    path: Path
    media_type: MediaType
    container_format: ContainerFormat
    file_size_bytes: int
    duration: Optional[float] = None  # seconds
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None

    # Stream information
    video_streams: List[VideoStreamInfo] = field(default_factory=list)
    audio_streams: List[AudioStreamInfo] = field(default_factory=list)
    subtitle_streams: List[SubtitleStreamInfo] = field(default_factory=list)

    # Additional metadata
    metadata_tags: Dict[str, str] = field(default_factory=dict)

    @property
    def has_video(self) -> bool:
        return len(self.video_streams) > 0

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0

    @property
    def has_subtitles(self) -> bool:
        return len(self.subtitle_streams) > 0

    @property
    def primary_video_stream(self) -> Optional[VideoStreamInfo]:
        """Get primary (first) video stream"""
        return self.video_streams[0] if self.video_streams else None

    @property
    def primary_audio_stream(self) -> Optional[AudioStreamInfo]:
        """Get primary (first) audio stream"""
        return self.audio_streams[0] if self.audio_streams else None


@dataclass
class SceneInfo:
    """Detected scene information"""
    scene_id: int
    start_time: float  # seconds
    end_time: float    # seconds
    start_frame: int
    end_frame: int
    duration: float    # seconds
    frame_count: int
    confidence: float  # 0.0 to 1.0

    @property
    def timestamp_range(self) -> str:
        """Formatted timestamp range"""
        return f"{self.start_time:.2f}s - {self.end_time:.2f}s"


@dataclass
class ProcessingParameters:
    """Parameters for media processing operations"""
    operation: ProcessingOperation

    # Common parameters
    cpu_threads: int = 0  # 0 = auto-detect
    quality_level: QualityLevel = QualityLevel.MEDIUM

    # Video parameters (CPU-only encoding)
    video_codec: Optional[VideoCodec] = None
    video_bitrate: Optional[int] = None  # bits/sec (None = auto)
    video_crf: Optional[int] = None      # Constant rate factor (0-51)
    video_preset: str = "medium"         # x264/x265 preset
    target_resolution: Optional[Tuple[int, int]] = None  # (width, height)
    target_fps: Optional[float] = None

    # Audio parameters
    audio_codec: Optional[AudioCodec] = None
    audio_bitrate: Optional[int] = None  # bits/sec
    audio_sample_rate: Optional[int] = None  # Hz
    audio_channels: Optional[int] = None

    # Frame extraction parameters
    frame_interval: Optional[int] = None  # Extract every Nth frame
    frame_timestamps: Optional[List[float]] = None  # Specific timestamps
    frame_quality: int = 95  # JPEG quality (1-100)

    # Scene detection parameters
    scene_threshold: float = 0.3  # Sensitivity (0.0-1.0)
    scene_min_duration: float = 1.0  # Minimum scene duration (seconds)
    scene_detection_method: SceneDetectionMethod = SceneDetectionMethod.CONTENT_BASED

    # Audio normalization parameters
    normalize_loudness: bool = True
    target_loudness: float = -23.0  # LUFS

    # Subtitle parameters
    subtitle_format: str = "srt"
    subtitle_encoding: str = "utf-8"

    # General parameters
    overwrite: bool = False
    preserve_metadata: bool = True


@dataclass
class ProcessingTask:
    """Media processing task specification"""
    task_id: str
    operation: ProcessingOperation
    input_path: Path
    output_path: Optional[Path]
    parameters: ProcessingParameters
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Task execution duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ProcessingResult:
    """Result of a media processing operation"""
    task: ProcessingTask
    status: ProcessingStatus
    output_path: Optional[Path] = None
    output_metadata: Optional[MediaMetadata] = None

    # Processing metrics
    processing_time: Optional[float] = None  # seconds
    input_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Additional outputs (e.g., extracted frames, scenes)
    additional_outputs: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ProcessingStatus.COMPLETED and len(self.errors) == 0

    @property
    def size_reduction_percent(self) -> Optional[float]:
        """Calculate size reduction percentage"""
        if self.input_size_bytes and self.output_size_bytes:
            reduction = (1 - self.output_size_bytes / self.input_size_bytes) * 100
            return reduction
        return None


@dataclass
class QualityMetrics:
    """Media quality assessment metrics"""
    # Video quality
    resolution_score: float = 0.0  # 0-100
    bitrate_score: float = 0.0     # 0-100
    framerate_score: float = 0.0   # 0-100
    codec_efficiency: float = 0.0   # 0-100

    # Audio quality
    audio_bitrate_score: float = 0.0  # 0-100
    audio_sample_rate_score: float = 0.0  # 0-100

    # Overall
    overall_score: float = 0.0  # 0-100
    quality_rating: str = "unknown"  # low, medium, high, ultra

    # Issues
    detected_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MediaAnalysisReport:
    """Comprehensive media analysis report"""
    input_path: Path
    analyzed_at: datetime

    # Metadata
    metadata: MediaMetadata

    # Quality assessment
    quality_metrics: QualityMetrics

    # Scene information (if analyzed)
    scenes: List[SceneInfo] = field(default_factory=list)
    total_scenes: int = 0
    avg_scene_duration: float = 0.0

    # Content analysis
    content_tags: List[str] = field(default_factory=list)

    # Processing recommendations
    recommended_operations: List[ProcessingOperation] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProcessingJob:
    """Batch processing job specification"""
    job_id: str
    tasks: List[ProcessingTask]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    results: List[ProcessingResult] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        return len([r for r in self.results if r.success])

    @property
    def failed_tasks(self) -> int:
        return len([r for r in self.results if not r.success])

    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (len(self.results) / self.total_tasks) * 100

    @property
    def success_rate(self) -> float:
        if len(self.results) == 0:
            return 0.0
        return (self.completed_tasks / len(self.results)) * 100
