"""
Media Processor

Main orchestrator for media processing operations.
Coordinates analyzers and processors, emits events, respects safety constraints.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .common import (
    ProcessingParameters,
    ProcessingResult,
    ProcessingTask,
    ProcessingStatus,
    ProcessingOperation,
    MediaMetadata,
    QualityMetrics,
    SceneInfo,
    MediaType
)
from .analyzers import MetadataExtractor, SceneDetector, QualityAnalyzer
from .processors import VideoProcessor, AudioProcessor, SubtitleProcessor

logger = logging.getLogger(__name__)


class MediaProcessor:
    """
    Main orchestrator for media processing operations

    Features:
    - Metadata extraction and quality analysis
    - Video transcoding and frame extraction
    - Audio extraction and normalization
    - Subtitle extraction and conversion
    - Scene detection for intelligent processing
    - Event emission for orchestration integration
    - Safety constraint enforcement
    - Comprehensive reporting

    Example:
        processor = MediaProcessor()

        # Analyze media
        analysis = processor.analyze_media(Path("input.mp4"))
        print(f"Quality: {analysis['quality'].quality_rating}")

        # Transcode video
        result = processor.transcode_video(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
            params=ProcessingParameters(
                operation=ProcessingOperation.TRANSCODE_VIDEO,
                video_codec=VideoCodec.H265,
                quality_level=QualityLevel.HIGH
            )
        )

        # Extract frames
        result = processor.extract_frames(
            video_path=Path("input.mp4"),
            output_dir=Path("frames/"),
            params=ProcessingParameters(
                operation=ProcessingOperation.EXTRACT_FRAMES,
                frame_interval=30
            )
        )

        # Complete workflow
        result = processor.process_media(
            input_path=Path("input.mp4"),
            output_dir=Path("output/"),
            workflow={
                "analyze": True,
                "transcode": {"codec": "h265", "quality": "high"},
                "extract_frames": {"interval": 30},
                "extract_audio": {"codec": "aac"}
            }
        )
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        event_bus: Optional[Any] = None,
        safety_manager: Optional[Any] = None
    ):
        """
        Initialize media processor

        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
            event_bus: Optional EventBus for orchestration integration
            safety_manager: Optional SafetyManager for constraint enforcement
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.event_bus = event_bus
        self.safety_manager = safety_manager

        # Initialize components
        self.metadata_extractor = MetadataExtractor(ffprobe_path=ffprobe_path)
        self.scene_detector = SceneDetector(ffmpeg_path=ffmpeg_path)
        self.quality_analyzer = QualityAnalyzer()
        self.video_processor = VideoProcessor(ffmpeg_path=ffmpeg_path)
        self.audio_processor = AudioProcessor(ffmpeg_path=ffmpeg_path)
        self.subtitle_processor = SubtitleProcessor(ffmpeg_path=ffmpeg_path)

        logger.info("MediaProcessor initialized")
        self._emit_event("media_processor_initialized")

    def analyze_media(
        self,
        input_path: Path,
        detect_scenes: bool = False,
        scene_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze media file (metadata + quality + optional scene detection)

        Args:
            input_path: Media file path
            detect_scenes: Whether to detect scene boundaries
            scene_threshold: Scene detection threshold (0.0-1.0)

        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing media: {input_path}")
        self._emit_event("media_analysis_started", {"path": str(input_path)})

        try:
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(input_path)

            # Analyze quality
            quality = self.quality_analyzer.analyze_quality(metadata)

            # Optional scene detection
            scenes = None
            if detect_scenes and metadata.has_video:
                scenes = self.scene_detector.detect_scenes(
                    video_path=input_path,
                    threshold=scene_threshold,
                    video_stream=metadata.primary_video_stream
                )

            analysis = {
                "path": input_path,
                "metadata": metadata,
                "quality": quality,
                "scenes": scenes,
                "analyzed_at": datetime.now()
            }

            self._emit_event("media_analysis_completed", {
                "path": str(input_path),
                "quality_rating": quality.quality_rating.value,
                "scene_count": len(scenes) if scenes else 0
            })

            logger.info(
                f"Analysis complete: {quality.quality_rating.value} quality, "
                f"{len(scenes) if scenes else 0} scenes"
            )

            return analysis

        except Exception as e:
            logger.error(f"Media analysis failed: {e}")
            self._emit_event("media_analysis_failed", {"path": str(input_path), "error": str(e)})
            raise

    def transcode_video(
        self,
        input_path: Path,
        output_path: Path,
        params: ProcessingParameters
    ) -> ProcessingResult:
        """
        Transcode video file

        Args:
            input_path: Input video path
            output_path: Output video path
            params: Processing parameters

        Returns:
            ProcessingResult
        """
        logger.info(f"Transcoding video: {input_path} -> {output_path}")
        self._emit_event("video_transcode_started", {
            "input": str(input_path),
            "output": str(output_path)
        })

        # Check safety constraints
        if self.safety_manager:
            self._check_safety_constraints(input_path)

        try:
            result = self.video_processor.transcode_video(input_path, output_path, params)

            self._emit_event("video_transcode_completed", {
                "input": str(input_path),
                "output": str(output_path),
                "processing_time": result.processing_time,
                "compression_ratio": result.compression_ratio
            })

            return result

        except Exception as e:
            logger.error(f"Video transcoding failed: {e}")
            self._emit_event("video_transcode_failed", {
                "input": str(input_path),
                "error": str(e)
            })
            raise

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        params: ProcessingParameters
    ) -> ProcessingResult:
        """
        Extract frames from video

        Args:
            video_path: Input video path
            output_dir: Output directory
            params: Processing parameters

        Returns:
            ProcessingResult with frame paths
        """
        logger.info(f"Extracting frames: {video_path} -> {output_dir}")
        self._emit_event("frame_extraction_started", {
            "video": str(video_path),
            "output_dir": str(output_dir)
        })

        # Check safety constraints
        if self.safety_manager:
            self._check_safety_constraints(video_path)

        try:
            result = self.video_processor.extract_frames(video_path, output_dir, params)

            frame_count = len(result.additional_outputs.get("frame_paths", []))

            self._emit_event("frame_extraction_completed", {
                "video": str(video_path),
                "frame_count": frame_count,
                "processing_time": result.processing_time
            })

            return result

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            self._emit_event("frame_extraction_failed", {
                "video": str(video_path),
                "error": str(e)
            })
            raise

    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        params: ProcessingParameters
    ) -> ProcessingResult:
        """
        Extract audio from video

        Args:
            video_path: Input video path
            output_path: Output audio path
            params: Processing parameters

        Returns:
            ProcessingResult
        """
        logger.info(f"Extracting audio: {video_path} -> {output_path}")
        self._emit_event("audio_extraction_started", {
            "video": str(video_path),
            "output": str(output_path)
        })

        # Check safety constraints
        if self.safety_manager:
            self._check_safety_constraints(video_path)

        try:
            result = self.audio_processor.extract_audio(video_path, output_path, params)

            self._emit_event("audio_extraction_completed", {
                "video": str(video_path),
                "output": str(output_path),
                "processing_time": result.processing_time
            })

            return result

        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            self._emit_event("audio_extraction_failed", {
                "video": str(video_path),
                "error": str(e)
            })
            raise

    def normalize_audio(
        self,
        input_path: Path,
        output_path: Path,
        target_loudness: float = -23.0
    ) -> ProcessingResult:
        """
        Normalize audio loudness

        Args:
            input_path: Input audio path
            output_path: Output audio path
            target_loudness: Target loudness in LUFS

        Returns:
            ProcessingResult
        """
        logger.info(f"Normalizing audio: {input_path} -> {output_path}")
        self._emit_event("audio_normalization_started", {
            "input": str(input_path),
            "target_loudness": target_loudness
        })

        try:
            result = self.audio_processor.normalize_audio(
                input_path, output_path, target_loudness
            )

            self._emit_event("audio_normalization_completed", {
                "input": str(input_path),
                "output": str(output_path),
                "processing_time": result.processing_time
            })

            return result

        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            self._emit_event("audio_normalization_failed", {
                "input": str(input_path),
                "error": str(e)
            })
            raise

    def extract_subtitles(
        self,
        video_path: Path,
        output_dir: Path
    ) -> ProcessingResult:
        """
        Extract all subtitle tracks from video

        Args:
            video_path: Input video path
            output_dir: Output directory

        Returns:
            ProcessingResult with subtitle paths
        """
        logger.info(f"Extracting subtitles: {video_path} -> {output_dir}")
        self._emit_event("subtitle_extraction_started", {
            "video": str(video_path),
            "output_dir": str(output_dir)
        })

        try:
            result = self.subtitle_processor.extract_subtitles(video_path, output_dir)

            subtitle_count = len(result.additional_outputs.get("subtitle_paths", []))

            self._emit_event("subtitle_extraction_completed", {
                "video": str(video_path),
                "subtitle_count": subtitle_count
            })

            return result

        except Exception as e:
            logger.error(f"Subtitle extraction failed: {e}")
            self._emit_event("subtitle_extraction_failed", {
                "video": str(video_path),
                "error": str(e)
            })
            raise

    def process_media(
        self,
        input_path: Path,
        output_dir: Path,
        workflow: Dict[str, Any],
        checkpoint_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Process media with complete workflow

        Args:
            input_path: Input media path
            output_dir: Output directory
            workflow: Workflow configuration dictionary
            checkpoint_enabled: Enable checkpoint/resume support

        Returns:
            Processing results dictionary

        Workflow format:
            {
                "analyze": True,  # Run analysis
                "detect_scenes": {"threshold": 0.3},  # Scene detection
                "transcode": {  # Video transcoding
                    "codec": "h265",
                    "quality": "high",
                    "resolution": (1920, 1080)
                },
                "extract_frames": {  # Frame extraction
                    "interval": 30,
                    "quality": 95
                },
                "extract_audio": {  # Audio extraction
                    "codec": "aac",
                    "bitrate": 192000
                },
                "normalize_audio": {  # Audio normalization
                    "target_loudness": -23.0
                },
                "extract_subtitles": True  # Subtitle extraction
            }
        """
        logger.info(f"Processing media: {input_path}")
        self._emit_event("media_workflow_started", {
            "input": str(input_path),
            "workflow": list(workflow.keys())
        })

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "input_path": input_path,
            "output_dir": output_dir,
            "workflow": workflow,
            "started_at": datetime.now(),
            "results": {}
        }

        try:
            # Analysis
            if workflow.get("analyze"):
                detect_scenes = "detect_scenes" in workflow
                scene_threshold = workflow.get("detect_scenes", {}).get("threshold", 0.3)
                analysis = self.analyze_media(
                    input_path,
                    detect_scenes=detect_scenes,
                    scene_threshold=scene_threshold
                )
                results["results"]["analysis"] = analysis

                # Save checkpoint
                if checkpoint_enabled and self.safety_manager:
                    self._save_checkpoint(results)

            # Video transcoding
            if "transcode" in workflow:
                transcode_config = workflow["transcode"]
                output_path = output_dir / f"{input_path.stem}_transcoded{input_path.suffix}"

                params = self._build_transcode_params(transcode_config)
                result = self.transcode_video(input_path, output_path, params)
                results["results"]["transcode"] = result

                if checkpoint_enabled and self.safety_manager:
                    self._save_checkpoint(results)

            # Frame extraction
            if "extract_frames" in workflow:
                frame_config = workflow["extract_frames"]
                frames_dir = output_dir / "frames"

                params = self._build_frame_extraction_params(frame_config)
                result = self.extract_frames(input_path, frames_dir, params)
                results["results"]["extract_frames"] = result

                if checkpoint_enabled and self.safety_manager:
                    self._save_checkpoint(results)

            # Audio extraction
            if "extract_audio" in workflow:
                audio_config = workflow["extract_audio"]
                audio_path = output_dir / f"{input_path.stem}.aac"

                params = self._build_audio_extraction_params(audio_config)
                result = self.extract_audio(input_path, audio_path, params)
                results["results"]["extract_audio"] = result

                if checkpoint_enabled and self.safety_manager:
                    self._save_checkpoint(results)

            # Audio normalization
            if "normalize_audio" in workflow:
                norm_config = workflow["normalize_audio"]

                # Use extracted audio or original file
                if "extract_audio" in results["results"]:
                    audio_input = results["results"]["extract_audio"].output_path
                else:
                    audio_input = input_path

                audio_normalized = output_dir / f"{input_path.stem}_normalized.aac"
                target_loudness = norm_config.get("target_loudness", -23.0)

                result = self.normalize_audio(audio_input, audio_normalized, target_loudness)
                results["results"]["normalize_audio"] = result

                if checkpoint_enabled and self.safety_manager:
                    self._save_checkpoint(results)

            # Subtitle extraction
            if workflow.get("extract_subtitles"):
                subtitles_dir = output_dir / "subtitles"
                result = self.extract_subtitles(input_path, subtitles_dir)
                results["results"]["extract_subtitles"] = result

                if checkpoint_enabled and self.safety_manager:
                    self._save_checkpoint(results)

            results["completed_at"] = datetime.now()
            results["status"] = "completed"

            self._emit_event("media_workflow_completed", {
                "input": str(input_path),
                "operations": len(results["results"])
            })

            logger.info(f"Media workflow completed: {len(results['results'])} operations")

            return results

        except Exception as e:
            logger.error(f"Media workflow failed: {e}")
            results["completed_at"] = datetime.now()
            results["status"] = "failed"
            results["error"] = str(e)

            self._emit_event("media_workflow_failed", {
                "input": str(input_path),
                "error": str(e)
            })

            raise

    def _build_transcode_params(self, config: Dict[str, Any]) -> ProcessingParameters:
        """Build ProcessingParameters for transcoding"""
        from .common import VideoCodec, QualityLevel

        # Parse codec
        codec_map = {
            "h264": VideoCodec.H264,
            "h265": VideoCodec.H265,
            "vp9": VideoCodec.VP9,
            "av1": VideoCodec.AV1
        }
        video_codec = codec_map.get(config.get("codec", "h264"))

        # Parse quality
        quality_map = {
            "low": QualityLevel.LOW,
            "medium": QualityLevel.MEDIUM,
            "high": QualityLevel.HIGH,
            "ultra": QualityLevel.ULTRA
        }
        quality_level = quality_map.get(config.get("quality", "medium"))

        return ProcessingParameters(
            operation=ProcessingOperation.TRANSCODE_VIDEO,
            video_codec=video_codec,
            quality_level=quality_level,
            video_crf=config.get("crf"),
            video_bitrate=config.get("bitrate"),
            target_resolution=config.get("resolution"),
            target_fps=config.get("fps"),
            overwrite=True
        )

    def _build_frame_extraction_params(self, config: Dict[str, Any]) -> ProcessingParameters:
        """Build ProcessingParameters for frame extraction"""
        return ProcessingParameters(
            operation=ProcessingOperation.EXTRACT_FRAMES,
            frame_interval=config.get("interval", 30),
            frame_quality=config.get("quality", 95),
            frame_timestamps=config.get("timestamps"),
            overwrite=True
        )

    def _build_audio_extraction_params(self, config: Dict[str, Any]) -> ProcessingParameters:
        """Build ProcessingParameters for audio extraction"""
        from .common import AudioCodec

        codec_map = {
            "aac": AudioCodec.AAC,
            "mp3": AudioCodec.MP3,
            "opus": AudioCodec.OPUS,
            "flac": AudioCodec.FLAC
        }
        audio_codec = codec_map.get(config.get("codec", "aac"))

        return ProcessingParameters(
            operation=ProcessingOperation.EXTRACT_AUDIO,
            audio_codec=audio_codec,
            audio_bitrate=config.get("bitrate", 192000),
            audio_sample_rate=config.get("sample_rate"),
            audio_channels=config.get("channels"),
            overwrite=True
        )

    def _emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Emit event to EventBus if available"""
        if self.event_bus:
            try:
                self.event_bus.emit(event_type, data or {})
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    def _check_safety_constraints(self, input_path: Path):
        """Check safety constraints before processing"""
        if not self.safety_manager:
            return

        try:
            # Check file size
            file_size = input_path.stat().st_size
            max_size = self.safety_manager.get_max_file_size()

            if max_size and file_size > max_size:
                raise RuntimeError(
                    f"File size {file_size} exceeds safety limit {max_size}"
                )

            # Check memory budget
            if hasattr(self.safety_manager, "check_memory_budget"):
                self.safety_manager.check_memory_budget()

        except Exception as e:
            logger.error(f"Safety constraint check failed: {e}")
            raise

    def _save_checkpoint(self, results: Dict[str, Any]):
        """Save processing checkpoint"""
        if not self.safety_manager:
            return

        try:
            if hasattr(self.safety_manager, "save_checkpoint"):
                self.safety_manager.save_checkpoint(results)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
