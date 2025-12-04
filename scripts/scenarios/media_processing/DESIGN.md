# Media Processing Automation - Architecture Design

**Version**: 1.0.0
**Date**: 2025-12-03
**Status**: Week 6 - Foundation Complete

---

## Overview

The **Media Processing Automation** scenario provides CPU-only media analysis and processing capabilities for the Animation AI Studio. It handles video, audio, and subtitle processing using ffmpeg/ffprobe without GPU acceleration.

### Key Features

- **100% CPU-only processing** - No GPU dependencies
- **Comprehensive metadata extraction** - ffprobe-based analysis
- **Scene detection** - Content-based shot boundary detection
- **Frame extraction** - Quality-controlled frame export
- **Audio processing** - Extraction, normalization, transcoding
- **Subtitle automation** - Extraction and format conversion
- **Quality assessment** - Automated media quality metrics
- **Batch processing** - Multi-file workflows with progress tracking

### Design Goals

1. **CPU-only operation** - Compatible with any system
2. **Production quality** - Suitable for film/animation workflows
3. **Safety first** - Dry-run mode, validation, error recovery
4. **Modular design** - Reusable analyzers and processors
5. **Integration ready** - Works with Orchestration Layer and Safety System

---

## Architecture

### Component Hierarchy

```
media_processing/
├── common.py                    # Core data structures (this file)
├── processor.py                 # Main orchestrator (Phase 4)
├── __main__.py                  # CLI entry point (Phase 5)
├── __init__.py                  # Package exports
├── DESIGN.md                    # Architecture documentation
├── IMPLEMENTATION_SUMMARY.md    # Implementation tracking
│
├── analyzers/                   # Analysis components (Phase 2)
│   ├── __init__.py
│   ├── metadata_extractor.py   # ffprobe-based metadata extraction
│   ├── scene_detector.py       # Scene/shot boundary detection
│   └── quality_analyzer.py     # Quality metrics and assessment
│
├── processors/                  # Processing components (Phase 3)
│   ├── __init__.py
│   ├── video_processor.py      # Video transcoding and frame extraction
│   ├── audio_processor.py      # Audio extraction and normalization
│   └── subtitle_processor.py   # Subtitle extraction and conversion
│
├── integration/                 # Integration components (Phase 4)
│   ├── __init__.py
│   ├── orchestration_integration.py  # WorkflowOrchestrator integration
│   └── safety_integration.py         # Safety System integration
│
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_metadata_extractor.py
    ├── test_scene_detector.py
    ├── test_quality_analyzer.py
    ├── test_video_processor.py
    ├── test_audio_processor.py
    └── run_all_tests.py
```

### Data Flow

```
Input Media File
    ↓
MetadataExtractor (ffprobe)
    ↓
MediaMetadata (streams, duration, codecs)
    ↓
┌────────────────┬────────────────┬────────────────┐
│                │                │                │
SceneDetector  QualityAnalyzer  VideoProcessor  AudioProcessor
│                │                │                │
SceneInfo[]    QualityMetrics   Frames/Video    Audio Files
│                │                │                │
└────────────────┴────────────────┴────────────────┘
    ↓
MediaAnalysisReport
    ↓
Processing Recommendations
    ↓
BatchProcessingJob (optional)
```

---

## Core Data Structures

### Enums

**MediaType**
- VIDEO, AUDIO, IMAGE, SUBTITLE, UNKNOWN
- Purpose: Top-level media classification

**VideoCodec** (CPU-decodable)
- H264, H265, VP9, AV1, MPEG4, MPEG2, THEORA, PRORES
- Purpose: Video codec identification
- Note: All codecs support CPU-only decoding/encoding

**AudioCodec**
- AAC, MP3, OPUS, VORBIS, FLAC, PCM, AC3, EAC3
- Purpose: Audio codec identification

**ContainerFormat**
- MP4, MKV, WEBM, MOV, AVI, FLV, TS, M4A, OGG, WAV
- Purpose: Container format detection

**ProcessingOperation**
- EXTRACT_METADATA, DETECT_SCENES, EXTRACT_FRAMES, EXTRACT_AUDIO
- TRANSCODE_VIDEO, TRANSCODE_AUDIO, NORMALIZE_AUDIO
- EXTRACT_SUBTITLES, CONVERT_SUBTITLES, ANALYZE_QUALITY, BATCH_PROCESS
- Purpose: Operation type specification

**QualityLevel** (CPU encoding presets)
- LOW (fast), MEDIUM (balanced), HIGH (quality), ULTRA (best)
- Purpose: Encoding quality/speed tradeoff

**ProcessingStatus**
- PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
- Purpose: Task status tracking

**SceneDetectionMethod**
- CONTENT_BASED, HISTOGRAM, SHOT_BOUNDARY, HYBRID
- Purpose: Scene detection algorithm selection

### Core Dataclasses

**MediaMetadata**
- Complete file metadata: path, type, format, size, duration, timestamps
- Stream lists: video_streams, audio_streams, subtitle_streams
- Convenience properties: has_video, has_audio, primary_video_stream, etc.
- Purpose: Central metadata container

**VideoStreamInfo**
- Video stream details: codec, resolution, fps, bitrate, pixel format
- Properties: resolution string, aspect_ratio
- Purpose: Per-stream video information

**AudioStreamInfo**
- Audio stream details: codec, sample_rate, channels, bitrate, language
- Purpose: Per-stream audio information

**SubtitleStreamInfo**
- Subtitle track details: format, language, title, encoding
- Purpose: Per-track subtitle information

**ProcessingParameters**
- Operation-specific parameters
- Video: codec, bitrate, CRF, preset, resolution, fps
- Audio: codec, bitrate, sample_rate, channels, normalization
- Frame extraction: interval, timestamps, quality
- Scene detection: threshold, min_duration, method
- Purpose: Comprehensive parameter specification

**ProcessingTask**
- Task specification: task_id, operation, paths, parameters, status
- Timestamps: created_at, started_at, completed_at
- Property: duration (execution time)
- Purpose: Single processing task definition

**ProcessingResult**
- Result data: task, status, output_path, metadata
- Metrics: processing_time, sizes, compression_ratio
- Errors/warnings: error and warning lists
- Additional outputs: extracted frames, scenes, etc.
- Properties: success, size_reduction_percent
- Purpose: Task execution result

**SceneInfo**
- Scene details: id, time range, frame range, duration, confidence
- Property: timestamp_range (formatted string)
- Purpose: Detected scene information

**QualityMetrics**
- Quality scores: resolution, bitrate, framerate, codec efficiency
- Overall: overall_score, quality_rating
- Issues: detected_issues, recommendations
- Purpose: Quality assessment results

**MediaAnalysisReport**
- Comprehensive analysis: metadata, quality, scenes
- Statistics: scene count, avg duration, content tags
- Recommendations: suggested operations, optimizations
- Purpose: Complete media analysis output

**BatchProcessingJob**
- Batch job: job_id, tasks list, timestamps
- Results: completed results list
- Properties: progress_percent, success_rate, task counts
- Purpose: Multi-file batch processing

---

## Component Design

### Phase 2: Analyzers

#### MetadataExtractor
**Purpose**: Extract comprehensive metadata using ffprobe

**Key Methods**:
- `extract_metadata(path: Path) -> MediaMetadata`
  - Run ffprobe JSON output
  - Parse streams (video/audio/subtitle)
  - Extract container-level metadata
  - Return structured MediaMetadata

- `detect_media_type(path: Path) -> MediaType`
  - Identify file type from streams
  - Fallback to extension if needed

- `parse_video_stream(stream_data: dict) -> VideoStreamInfo`
  - Parse ffprobe video stream JSON
  - Extract codec, resolution, fps, bitrate

- `parse_audio_stream(stream_data: dict) -> AudioStreamInfo`
  - Parse ffprobe audio stream JSON
  - Extract codec, sample rate, channels

**CPU-only**: Uses ffprobe (no GPU acceleration)

#### SceneDetector
**Purpose**: Detect scene boundaries for frame extraction

**Key Methods**:
- `detect_scenes(video_path: Path, method: SceneDetectionMethod, threshold: float) -> List[SceneInfo]`
  - Run ffmpeg select filter (content-based)
  - Or use histogram difference
  - Return list of detected scenes

- `_detect_content_based(video_path: Path, threshold: float) -> List[SceneInfo]`
  - Use ffmpeg select='gt(scene,threshold)'
  - Parse timestamps from output
  - Generate SceneInfo objects

- `_detect_histogram(video_path: Path, threshold: float) -> List[SceneInfo]`
  - Calculate frame histogram differences
  - Detect sudden changes above threshold
  - More sensitive than content-based

- `merge_short_scenes(scenes: List[SceneInfo], min_duration: float) -> List[SceneInfo]`
  - Merge scenes shorter than min_duration
  - Prevents over-segmentation

**CPU-only**: Uses ffmpeg select filter or histogram analysis

#### QualityAnalyzer
**Purpose**: Assess media quality and recommend improvements

**Key Methods**:
- `analyze_quality(metadata: MediaMetadata) -> QualityMetrics`
  - Score resolution (720p=70, 1080p=85, 4K=100)
  - Score bitrate vs resolution
  - Score framerate (24/30/60 fps standards)
  - Calculate overall score
  - Generate recommendations

- `_score_video_quality(video: VideoStreamInfo) -> Dict[str, float]`
  - Resolution score based on standards
  - Bitrate adequacy for resolution
  - Framerate appropriateness

- `_score_audio_quality(audio: AudioStreamInfo) -> Dict[str, float]`
  - Bitrate score (128k=80, 192k=90, 320k=100)
  - Sample rate score (44.1k=85, 48k=95, 96k=100)

- `detect_issues(metadata: MediaMetadata) -> List[str]`
  - Low bitrate warnings
  - Unusual resolution/framerate
  - Codec compatibility issues
  - Audio channel mismatches

**CPU-only**: Pure analysis, no processing

### Phase 3: Processors

#### VideoProcessor
**Purpose**: CPU-only video transcoding and frame extraction

**Key Methods**:
- `transcode_video(input_path: Path, output_path: Path, params: ProcessingParameters) -> ProcessingResult`
  - Use ffmpeg with CPU-only codecs (libx264, libx265, libvpx-vp9)
  - Apply CRF or bitrate settings
  - Use preset for speed/quality tradeoff
  - Stream copy when possible
  - Return result with metrics

- `extract_frames(video_path: Path, output_dir: Path, params: ProcessingParameters) -> ProcessingResult`
  - Extract by interval (every Nth frame)
  - Or by timestamps (specific times)
  - Or by scenes (SceneInfo list)
  - Export as JPEG/PNG with quality control
  - Return list of extracted frames

- `resize_video(input_path: Path, output_path: Path, target_resolution: Tuple[int, int]) -> ProcessingResult`
  - Scale filter with CPU
  - Maintain aspect ratio
  - Use lanczos scaling

**CPU-only encoding presets**:
- LOW: `veryfast` preset, CRF 28
- MEDIUM: `medium` preset, CRF 23
- HIGH: `slow` preset, CRF 18
- ULTRA: `veryslow` preset, CRF 15

**No GPU acceleration**: Avoid -hwaccel, -c:v h264_nvenc, etc.

#### AudioProcessor
**Purpose**: Audio extraction, normalization, transcoding

**Key Methods**:
- `extract_audio(video_path: Path, output_path: Path, params: ProcessingParameters) -> ProcessingResult`
  - Extract audio stream to separate file
  - Transcode to target codec (AAC, MP3, OPUS)
  - Apply bitrate settings
  - Return audio file metadata

- `normalize_audio(input_path: Path, output_path: Path, target_loudness: float) -> ProcessingResult`
  - Use ffmpeg loudnorm filter
  - Two-pass normalization for accuracy
  - Target -23 LUFS (broadcast standard)
  - Preserve dynamic range

- `transcode_audio(input_path: Path, output_path: Path, params: ProcessingParameters) -> ProcessingResult`
  - Convert codec/bitrate/sample rate
  - Channel mixing if needed
  - Quality-controlled encoding

**CPU-only**: All audio codecs support CPU encoding

#### SubtitleProcessor
**Purpose**: Subtitle extraction and format conversion

**Key Methods**:
- `extract_subtitles(video_path: Path, output_dir: Path) -> List[Path]`
  - Extract all subtitle tracks
  - Save to separate files
  - Preserve language tags

- `convert_subtitle_format(input_path: Path, output_path: Path, target_format: str) -> ProcessingResult`
  - Convert SRT ↔ ASS ↔ VTT
  - Preserve timing and formatting
  - Handle encoding (UTF-8)

**CPU-only**: Text processing only

### Phase 4: Main Orchestrator

#### MediaProcessor
**Purpose**: Main orchestrator following Dataset Quality Inspector pattern

**Workflow**:
1. **Analyze** → Extract metadata, detect scenes, assess quality
2. **Plan** → Generate ProcessingTask list based on goals
3. **Execute** → Run tasks with progress tracking
4. **Report** → Generate MediaAnalysisReport

**Key Methods**:
- `analyze(media_path: Path) -> MediaAnalysisReport`
  - Extract metadata
  - Detect scenes (if video)
  - Assess quality
  - Generate recommendations
  - Return comprehensive report

- `plan_processing(report: MediaAnalysisReport, goals: List[str]) -> BatchProcessingJob`
  - Based on goals (e.g., "extract_frames", "normalize_audio")
  - Create ProcessingTask list
  - Estimate time and resources
  - Return job specification

- `execute_batch(job: BatchProcessingJob, dry_run: bool = False) -> List[ProcessingResult]`
  - Execute tasks in order
  - Track progress
  - Handle errors
  - Return results

**Integration**:
- Emits events to EventBus (task_started, task_completed, etc.)
- Respects Safety System memory budgets
- Supports checkpoint/resume

---

## Integration Points

### Orchestration Layer (Week 1)

**EventBus Integration**:
- Event: `MediaProcessingStarted` → task_id, operation, input_path
- Event: `MediaProcessingProgress` → task_id, progress_percent
- Event: `MediaProcessingCompleted` → task_id, result
- Event: `MediaProcessingFailed` → task_id, error

**WorkflowOrchestrator**:
- Register as scenario "media_processing"
- Provide scenario interface (analyze, process)
- Support cancellation and pause

### Safety System (Week 2)

**Resource Monitoring**:
- CPU usage tracking (ffmpeg processes)
- Memory budget enforcement
- Disk space checks before processing

**Emergency Handling**:
- Graceful ffmpeg termination
- Partial result preservation
- Cleanup temporary files

**Checkpoint/Resume**:
- Save BatchProcessingJob state
- Resume from last completed task
- Handle interrupted encodings

### Agent Framework (Week 1) - Future

**AI-powered recommendations**:
- Suggest optimal encoding settings
- Recommend quality improvements
- Content-aware scene detection tuning

### RAG System (Week 1) - Future

**Best practices lookup**:
- Codec selection guidance
- Resolution/bitrate standards
- Platform-specific optimizations

---

## CPU-Only Processing Constraints

### Video Encoding

**Supported CPU-only codecs**:
- libx264 (H.264/AVC) - Most compatible, good speed
- libx265 (H.265/HEVC) - Better compression, slower
- libvpx-vp9 (VP9) - Modern, good for web
- libaom-av1 (AV1) - Best compression, very slow

**Encoding presets (x264/x265)**:
- ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
- Trade-off: speed vs compression efficiency
- Recommendation: medium for balanced, slow for quality

**CRF (Constant Rate Factor)**:
- Range: 0 (lossless) to 51 (worst)
- Recommended: 18-23 for high quality, 23-28 for medium
- Lower = better quality, larger file

**Avoid GPU acceleration**:
- ❌ -hwaccel cuda
- ❌ -c:v h264_nvenc, hevc_nvenc
- ❌ -c:v h264_qsv, hevc_qsv
- ✅ -c:v libx264, libx265 (CPU-only)

### Audio Encoding

**Supported codecs** (all CPU-only):
- libfdk_aac (AAC) - Best quality AAC
- aac (native) - Built-in AAC
- libmp3lame (MP3) - Universal compatibility
- libopus (Opus) - Modern, efficient
- libvorbis (Vorbis) - Open source

**Quality settings**:
- AAC: 128k (acceptable), 192k (good), 256k (excellent)
- MP3: 192k (good), 320k (best)
- Opus: 96k (speech), 128k (music)

### Performance Expectations

**Frame extraction** (1080p video):
- ~10-30 fps extraction speed on modern CPU
- Quality: JPEG q=95 recommended

**Video transcoding** (1080p, libx264 medium):
- ~1-5x realtime on 8-core CPU
- Slower presets reduce speed proportionally

**Scene detection**:
- Content-based: ~20-50 fps analysis
- Histogram: ~30-70 fps analysis

**Audio normalization**:
- Two-pass: ~2x file duration
- Fast enough for batch processing

---

## Output Contracts

### MetadataExtractor
```
Output: MediaMetadata object
- Complete stream information
- Container metadata
- Timestamps and file info
```

### SceneDetector
```
Output: List[SceneInfo]
- Scene boundaries (time + frame)
- Confidence scores
- Duration statistics
```

### QualityAnalyzer
```
Output: QualityMetrics
- Quality scores (0-100)
- Detected issues
- Recommendations
```

### VideoProcessor
```
Frame Extraction:
output_dir/
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── extraction_metadata.json

Transcoding:
output_path (video file)
+ ProcessingResult with metrics
```

### AudioProcessor
```
Output: Audio file (AAC/MP3/OPUS)
+ ProcessingResult
- Normalization stats if applied
```

### BatchProcessingJob
```
Output: List[ProcessingResult]
+ Job statistics (success rate, timing)
+ Error summaries
```

---

## Testing Strategy

### Unit Tests (Phase 2-3)
- Test metadata extraction with sample videos
- Test scene detection accuracy
- Test quality scoring algorithms
- Test encoding parameter generation
- Test error handling

### Integration Tests (Phase 4)
- End-to-end workflow tests
- Batch processing tests
- Error recovery tests
- Performance benchmarks

### Sample Test Files
- 1080p sample video (10s)
- Audio-only file (MP3, AAC)
- Multi-track video (multiple audio/subtitle streams)
- Unusual format (old codec, odd resolution)

### Performance Tests
- Large file handling (4K, 2+ hours)
- Batch job (100+ files)
- Memory usage profiling
- CPU usage monitoring

---

## Dependencies

### Required
- **ffmpeg** (4.4+) - Video/audio processing
- **ffprobe** (4.4+) - Metadata extraction
- Python 3.10+

### Optional
- **PySceneDetect** - Advanced scene detection (CPU-only)
- **mutagen** - Audio metadata tagging

### Python Libraries
- pathlib (stdlib) - Path operations
- subprocess (stdlib) - ffmpeg execution
- json (stdlib) - Metadata parsing
- datetime (stdlib) - Timestamp handling
- dataclasses (stdlib) - Data structures
- enum (stdlib) - Enumerations
- logging (stdlib) - Logging

**No GPU dependencies** - 100% CPU-only processing

---

## Future Enhancements

### Short-term
- [ ] Advanced scene detection (shot boundaries, histogram)
- [ ] Video quality metrics (VMAF, PSNR, SSIM)
- [ ] Intelligent bitrate calculation
- [ ] Content-aware encoding optimization

### Medium-term
- [ ] Parallel batch processing
- [ ] Web UI for job monitoring
- [ ] Preset library for common tasks
- [ ] Integration with cloud storage

### Long-term
- [ ] ML-based scene classification
- [ ] Automatic content tagging
- [ ] Multi-language subtitle generation
- [ ] Adaptive bitrate packaging

---

## Success Criteria

- [x] 100% CPU-only operation (no GPU dependencies)
- [ ] ffmpeg/ffprobe integration for all operations
- [ ] Comprehensive metadata extraction
- [ ] Scene detection with configurable methods
- [ ] Quality assessment with actionable recommendations
- [ ] Frame extraction with quality control
- [ ] Audio extraction and normalization
- [ ] Subtitle processing
- [ ] Batch processing with progress tracking
- [ ] Integration with Orchestration Layer
- [ ] Integration with Safety System
- [ ] Complete test coverage (unit + integration)
- [ ] CLI interface with dry-run mode
- [ ] Comprehensive documentation

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
**Status**: Foundation Complete, Ready for Phase 2
