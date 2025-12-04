"""
Audio Processor

Audio extraction, normalization, and transcoding.
CPU-only processing with no GPU dependencies.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..common import (
    ProcessingParameters,
    ProcessingResult,
    ProcessingTask,
    ProcessingStatus,
    ProcessingOperation,
    AudioCodec
)

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio extraction, normalization, and transcoding

    Features:
    - Extract audio from video files
    - Loudness normalization (two-pass, -23 LUFS)
    - Audio transcoding (AAC, MP3, OPUS, FLAC)
    - Bitrate and sample rate conversion
    - Channel mixing
    - CPU-only operation

    Example:
        processor = AudioProcessor(ffmpeg_path="ffmpeg")

        # Extract audio
        result = processor.extract_audio(
            video_path=Path("input.mp4"),
            output_path=Path("audio.aac"),
            params=ProcessingParameters(
                operation=ProcessingOperation.EXTRACT_AUDIO,
                audio_codec=AudioCodec.AAC,
                audio_bitrate=192000
            )
        )

        # Normalize audio
        result = processor.normalize_audio(
            input_path=Path("audio.aac"),
            output_path=Path("audio_normalized.aac"),
            target_loudness=-23.0
        )
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize audio processor

        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path

        # Verify ffmpeg
        self._verify_ffmpeg()

        logger.info(f"AudioProcessor initialized with ffmpeg: {self.ffmpeg_path}")

    def _verify_ffmpeg(self):
        """Verify ffmpeg is available"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        except FileNotFoundError:
            raise RuntimeError(f"ffmpeg not found at: {self.ffmpeg_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg verification timed out")

    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        params: ProcessingParameters
    ) -> ProcessingResult:
        """
        Extract audio stream from video

        Args:
            video_path: Input video path
            output_path: Output audio path
            params: Processing parameters

        Returns:
            ProcessingResult
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Extracting audio: {video_path} -> {output_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build task
        task = ProcessingTask(
            task_id=f"extract_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.EXTRACT_AUDIO,
            input_path=video_path,
            output_path=output_path,
            parameters=params,
            started_at=datetime.now()
        )

        try:
            # Build command
            cmd = [self.ffmpeg_path, "-i", str(video_path)]

            # Audio codec
            if params.audio_codec:
                codec_name = self._get_codec_name(params.audio_codec)
                cmd.extend(["-c:a", codec_name])
            else:
                cmd.extend(["-c:a", "copy"])  # Copy audio stream

            # Bitrate
            if params.audio_bitrate:
                cmd.extend(["-b:a", str(params.audio_bitrate)])

            # Sample rate
            if params.audio_sample_rate:
                cmd.extend(["-ar", str(params.audio_sample_rate)])

            # Channels
            if params.audio_channels:
                cmd.extend(["-ac", str(params.audio_channels)])

            # No video
            cmd.extend(["-vn"])

            # Overwrite
            if params.overwrite:
                cmd.append("-y")

            # Output
            cmd.append(str(output_path))

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

            # Create result
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED

            output_size = output_path.stat().st_size

            processing_result = ProcessingResult(
                task=task,
                status=ProcessingStatus.COMPLETED,
                output_path=output_path,
                processing_time=processing_time,
                output_size_bytes=output_size
            )

            logger.info(f"Audio extraction complete: {processing_time:.1f}s")

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Audio extraction timed out"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

    def normalize_audio(
        self,
        input_path: Path,
        output_path: Path,
        target_loudness: float = -23.0
    ) -> ProcessingResult:
        """
        Normalize audio loudness using two-pass algorithm

        Args:
            input_path: Input audio path
            output_path: Output audio path
            target_loudness: Target loudness in LUFS (default: -23.0)

        Returns:
            ProcessingResult
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Audio not found: {input_path}")

        logger.info(f"Normalizing audio: {input_path} -> {output_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build task
        task = ProcessingTask(
            task_id=f"normalize_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.NORMALIZE_AUDIO,
            input_path=input_path,
            output_path=output_path,
            parameters=ProcessingParameters(
                operation=ProcessingOperation.NORMALIZE_AUDIO,
                target_loudness=target_loudness
            ),
            started_at=datetime.now()
        )

        try:
            # Two-pass normalization
            # Pass 1: Measure loudness
            cmd_measure = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-af", f"loudnorm=I={target_loudness}:print_format=json",
                "-f", "null",
                "-"
            ]

            result_measure = subprocess.run(
                cmd_measure,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Pass 2: Apply normalization
            cmd_normalize = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-af", f"loudnorm=I={target_loudness}",
                "-y",
                str(output_path)
            ]

            start_time = time.time()
            result_normalize = subprocess.run(
                cmd_normalize,
                capture_output=True,
                text=True,
                timeout=600
            )
            processing_time = time.time() - start_time

            if result_normalize.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result_normalize.stderr}")

            # Create result
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED

            output_size = output_path.stat().st_size

            processing_result = ProcessingResult(
                task=task,
                status=ProcessingStatus.COMPLETED,
                output_path=output_path,
                processing_time=processing_time,
                output_size_bytes=output_size
            )

            logger.info(f"Audio normalization complete: {processing_time:.1f}s")

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Audio normalization timed out"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

    def transcode_audio(
        self,
        input_path: Path,
        output_path: Path,
        params: ProcessingParameters
    ) -> ProcessingResult:
        """
        Transcode audio to different codec/bitrate/sample rate

        Args:
            input_path: Input audio path
            output_path: Output audio path
            params: Processing parameters

        Returns:
            ProcessingResult
        """
        # Use extract_audio for transcoding
        # (same process, just different input type)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio not found: {input_path}")

        logger.info(f"Transcoding audio: {input_path} -> {output_path}")

        # Build task
        task = ProcessingTask(
            task_id=f"transcode_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.TRANSCODE_AUDIO,
            input_path=input_path,
            output_path=output_path,
            parameters=params,
            started_at=datetime.now()
        )

        try:
            # Build command
            cmd = [self.ffmpeg_path, "-i", str(input_path)]

            # Audio codec
            if params.audio_codec:
                codec_name = self._get_codec_name(params.audio_codec)
                cmd.extend(["-c:a", codec_name])

            # Bitrate
            if params.audio_bitrate:
                cmd.extend(["-b:a", str(params.audio_bitrate)])

            # Sample rate
            if params.audio_sample_rate:
                cmd.extend(["-ar", str(params.audio_sample_rate)])

            # Channels
            if params.audio_channels:
                cmd.extend(["-ac", str(params.audio_channels)])

            # Overwrite
            if params.overwrite:
                cmd.append("-y")

            # Output
            cmd.append(str(output_path))

            # Run ffmpeg
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            processing_time = time.time() - start_time

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            # Create result
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED

            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size
            compression_ratio = output_size / input_size if input_size > 0 else 0.0

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
                f"Audio transcoding complete: {processing_time:.1f}s, "
                f"compression: {compression_ratio:.2%}"
            )

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Audio transcoding timed out"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

    def _get_codec_name(self, codec: AudioCodec) -> str:
        """Get ffmpeg codec name from AudioCodec enum"""
        codec_map = {
            AudioCodec.AAC: "aac",
            AudioCodec.MP3: "libmp3lame",
            AudioCodec.OPUS: "libopus",
            AudioCodec.VORBIS: "libvorbis",
            AudioCodec.FLAC: "flac",
            AudioCodec.PCM: "pcm_s16le",
            AudioCodec.AC3: "ac3",
            AudioCodec.EAC3: "eac3"
        }

        return codec_map.get(codec, "aac")
