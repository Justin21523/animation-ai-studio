"""
Video Processor

CPU-only video transcoding and frame extraction.
Uses ffmpeg with software encoders (no GPU acceleration).

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..common import (
    ProcessingParameters,
    ProcessingResult,
    ProcessingTask,
    ProcessingStatus,
    ProcessingOperation,
    QualityLevel,
    VideoCodec,
    MediaMetadata
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    CPU-only video transcoding and frame extraction

    Features:
    - Video transcoding with CPU-only codecs (libx264, libx265, libvpx-vp9)
    - Quality presets (LOW/MEDIUM/HIGH/ULTRA)
    - CRF and bitrate control
    - Resolution scaling
    - Frame extraction (interval-based, timestamp-based, scene-based)
    - JPEG/PNG export with quality control
    - Processing metrics (time, compression ratio)
    - 100% CPU operation (no GPU)

    Example:
        processor = VideoProcessor(ffmpeg_path="ffmpeg")

        # Transcode video
        result = processor.transcode_video(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
            params=ProcessingParameters(
                operation=ProcessingOperation.TRANSCODE_VIDEO,
                video_codec=VideoCodec.H265,
                quality_level=QualityLevel.HIGH,
                video_crf=18
            )
        )

        # Extract frames
        result = processor.extract_frames(
            video_path=Path("input.mp4"),
            output_dir=Path("frames/"),
            params=ProcessingParameters(
                operation=ProcessingOperation.EXTRACT_FRAMES,
                frame_interval=30,
                frame_quality=95
            )
        )
    """

    # CPU encoding presets
    QUALITY_PRESETS = {
        QualityLevel.LOW: {
            "preset": "veryfast",
            "crf": 28
        },
        QualityLevel.MEDIUM: {
            "preset": "medium",
            "crf": 23
        },
        QualityLevel.HIGH: {
            "preset": "slow",
            "crf": 18
        },
        QualityLevel.ULTRA: {
            "preset": "veryslow",
            "crf": 15
        }
    }

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize video processor

        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path

        # Verify ffmpeg is available
        self._verify_ffmpeg()

        logger.info(f"VideoProcessor initialized with ffmpeg: {self.ffmpeg_path}")

    def _verify_ffmpeg(self):
        """Verify ffmpeg is available and working"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            logger.debug(f"ffmpeg version: {result.stdout.split()[2]}")

        except FileNotFoundError:
            raise RuntimeError(
                f"ffmpeg not found at: {self.ffmpeg_path}. "
                "Please install ffmpeg or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg verification timed out")

    def transcode_video(
        self,
        input_path: Path,
        output_path: Path,
        params: ProcessingParameters
    ) -> ProcessingResult:
        """
        Transcode video with CPU-only codecs

        Args:
            input_path: Input video path
            output_path: Output video path
            params: Processing parameters

        Returns:
            ProcessingResult with metrics

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If transcoding fails
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        logger.info(f"Transcoding: {input_path} -> {output_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build task
        task = ProcessingTask(
            task_id=f"transcode_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.TRANSCODE_VIDEO,
            input_path=input_path,
            output_path=output_path,
            parameters=params,
            started_at=datetime.now()
        )

        try:
            # Get input size
            input_size = input_path.stat().st_size

            # Build ffmpeg command
            cmd = self._build_transcode_command(input_path, output_path, params)

            # Run ffmpeg
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            processing_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            # Get output size
            output_size = output_path.stat().st_size

            # Calculate compression ratio
            compression_ratio = output_size / input_size if input_size > 0 else 0.0

            # Create result
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED

            processing_result = ProcessingResult(
                task=task,
                status=ProcessingStatus.COMPLETED,
                output_path=output_path,
                processing_time=processing_time,
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                compression_ratio=compression_ratio
            )

            logger.info(
                f"Transcoding complete: {processing_time:.1f}s, "
                f"compression: {compression_ratio:.2%}"
            )

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Transcoding timed out (exceeded 1 hour)"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

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
            output_dir: Output directory for frames
            params: Processing parameters

        Returns:
            ProcessingResult with frame paths

        Raises:
            FileNotFoundError: If video doesn't exist
            RuntimeError: If extraction fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Extracting frames from: {video_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build task
        task = ProcessingTask(
            task_id=f"extract_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.EXTRACT_FRAMES,
            input_path=video_path,
            output_path=output_dir,
            parameters=params,
            started_at=datetime.now()
        )

        try:
            # Build ffmpeg command based on extraction mode
            if params.frame_interval:
                cmd = self._build_interval_extraction_command(
                    video_path, output_dir, params
                )
            elif params.frame_timestamps:
                cmd = self._build_timestamp_extraction_command(
                    video_path, output_dir, params
                )
            else:
                raise ValueError("Must specify frame_interval or frame_timestamps")

            # Run ffmpeg
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            processing_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            # Count extracted frames
            frame_files = sorted(output_dir.glob("frame_*.jpg")) + sorted(output_dir.glob("frame_*.png"))
            frame_count = len(frame_files)

            # Create result
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED

            processing_result = ProcessingResult(
                task=task,
                status=ProcessingStatus.COMPLETED,
                output_path=output_dir,
                processing_time=processing_time,
                additional_outputs={"frame_paths": [str(f) for f in frame_files]}
            )

            logger.info(
                f"Frame extraction complete: {frame_count} frames in {processing_time:.1f}s"
            )

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Frame extraction timed out"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

    def resize_video(
        self,
        input_path: Path,
        output_path: Path,
        target_resolution: Tuple[int, int]
    ) -> ProcessingResult:
        """
        Resize video to target resolution

        Args:
            input_path: Input video path
            output_path: Output video path
            target_resolution: (width, height)

        Returns:
            ProcessingResult
        """
        params = ProcessingParameters(
            operation=ProcessingOperation.TRANSCODE_VIDEO,
            target_resolution=target_resolution,
            video_codec=VideoCodec.H264,
            quality_level=QualityLevel.MEDIUM
        )

        return self.transcode_video(input_path, output_path, params)

    def _build_transcode_command(
        self,
        input_path: Path,
        output_path: Path,
        params: ProcessingParameters
    ) -> List[str]:
        """Build ffmpeg command for transcoding"""
        cmd = [self.ffmpeg_path, "-i", str(input_path)]

        # Video codec selection (CPU-only)
        if params.video_codec == VideoCodec.H264:
            cmd.extend(["-c:v", "libx264"])
        elif params.video_codec == VideoCodec.H265:
            cmd.extend(["-c:v", "libx265"])
        elif params.video_codec == VideoCodec.VP9:
            cmd.extend(["-c:v", "libvpx-vp9"])
        elif params.video_codec == VideoCodec.AV1:
            cmd.extend(["-c:v", "libaom-av1"])
        else:
            cmd.extend(["-c:v", "libx264"])  # Default to H.264

        # Quality preset
        preset_config = self.QUALITY_PRESETS.get(
            params.quality_level,
            self.QUALITY_PRESETS[QualityLevel.MEDIUM]
        )

        # Preset (x264/x265 only)
        if params.video_codec in {VideoCodec.H264, VideoCodec.H265}:
            cmd.extend(["-preset", params.video_preset or preset_config["preset"]])

        # CRF or bitrate
        if params.video_crf is not None:
            cmd.extend(["-crf", str(params.video_crf)])
        elif params.video_bitrate:
            cmd.extend(["-b:v", f"{params.video_bitrate}"])
        else:
            # Use preset CRF
            cmd.extend(["-crf", str(preset_config["crf"])])

        # Resolution scaling
        if params.target_resolution:
            width, height = params.target_resolution
            cmd.extend(["-vf", f"scale={width}:{height}"])

        # FPS
        if params.target_fps:
            cmd.extend(["-r", str(params.target_fps)])

        # Audio handling
        if params.audio_codec:
            cmd.extend(["-c:a", "copy"])  # Copy audio by default
        else:
            cmd.extend(["-an"])  # No audio

        # CPU threads
        if params.cpu_threads > 0:
            cmd.extend(["-threads", str(params.cpu_threads)])

        # Overwrite
        if params.overwrite:
            cmd.append("-y")

        # Output
        cmd.append(str(output_path))

        return cmd

    def _build_interval_extraction_command(
        self,
        video_path: Path,
        output_dir: Path,
        params: ProcessingParameters
    ) -> List[str]:
        """Build command for interval-based frame extraction"""
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"select='not(mod(n\\,{params.frame_interval}))'",
            "-vsync", "vfr",
            "-q:v", str(100 - params.frame_quality),  # Quality to q scale
            str(output_dir / "frame_%05d.jpg")
        ]

        if params.overwrite:
            cmd.insert(1, "-y")

        return cmd

    def _build_timestamp_extraction_command(
        self,
        video_path: Path,
        output_dir: Path,
        params: ProcessingParameters
    ) -> List[str]:
        """Build command for timestamp-based frame extraction"""
        # Extract frames at specific timestamps
        # This would require multiple ffmpeg calls or complex filter
        # Simplified version: extract all frames at timestamps
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path)
        ]

        # For each timestamp, add an output
        for i, timestamp in enumerate(params.frame_timestamps):
            cmd.extend([
                "-ss", str(timestamp),
                "-vframes", "1",
                "-q:v", str(100 - params.frame_quality),
                str(output_dir / f"frame_{i:05d}.jpg")
            ])

        return cmd
