"""
Subtitle Processor

Subtitle extraction and format conversion.
CPU-only text processing.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List

from ..common import (
    ProcessingParameters,
    ProcessingResult,
    ProcessingTask,
    ProcessingStatus,
    ProcessingOperation
)

logger = logging.getLogger(__name__)


class SubtitleProcessor:
    """
    Subtitle extraction and format conversion

    Features:
    - Extract subtitle tracks from video
    - Format conversion (SRT ↔ ASS ↔ VTT)
    - Encoding handling (UTF-8)
    - Language tag preservation
    - CPU-only operation

    Example:
        processor = SubtitleProcessor(ffmpeg_path="ffmpeg")

        # Extract subtitles
        result = processor.extract_subtitles(
            video_path=Path("input.mp4"),
            output_dir=Path("subtitles/")
        )

        # Convert subtitle format
        result = processor.convert_subtitle_format(
            input_path=Path("subtitles.srt"),
            output_path=Path("subtitles.vtt"),
            target_format="webvtt"
        )
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize subtitle processor

        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path

        # Verify ffmpeg
        self._verify_ffmpeg()

        logger.info(f"SubtitleProcessor initialized with ffmpeg: {self.ffmpeg_path}")

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

    def extract_subtitles(
        self,
        video_path: Path,
        output_dir: Path
    ) -> ProcessingResult:
        """
        Extract all subtitle tracks from video

        Args:
            video_path: Input video path
            output_dir: Output directory for subtitle files

        Returns:
            ProcessingResult with subtitle file paths
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Extracting subtitles from: {video_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build task
        task = ProcessingTask(
            task_id=f"extract_subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.EXTRACT_SUBTITLES,
            input_path=video_path,
            output_path=output_dir,
            parameters=ProcessingParameters(
                operation=ProcessingOperation.EXTRACT_SUBTITLES
            ),
            started_at=datetime.now()
        )

        try:
            # Get subtitle stream count
            subtitle_streams = self._get_subtitle_stream_count(video_path)

            if subtitle_streams == 0:
                logger.warning(f"No subtitle tracks found in: {video_path}")
                task.status = ProcessingStatus.COMPLETED
                return ProcessingResult(
                    task=task,
                    status=ProcessingStatus.COMPLETED,
                    warnings=["No subtitle tracks found"],
                    additional_outputs={"subtitle_paths": []}
                )

            # Extract each subtitle track
            subtitle_paths = []
            start_time = time.time()

            for i in range(subtitle_streams):
                output_path = output_dir / f"subtitle_{i}.srt"

                cmd = [
                    self.ffmpeg_path,
                    "-i", str(video_path),
                    "-map", f"0:s:{i}",
                    "-c:s", "srt",
                    "-y",
                    str(output_path)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    subtitle_paths.append(str(output_path))
                else:
                    logger.warning(f"Failed to extract subtitle track {i}: {result.stderr}")

            processing_time = time.time() - start_time

            # Create result
            task.completed_at = datetime.now()
            task.status = ProcessingStatus.COMPLETED

            processing_result = ProcessingResult(
                task=task,
                status=ProcessingStatus.COMPLETED,
                output_path=output_dir,
                processing_time=processing_time,
                additional_outputs={"subtitle_paths": subtitle_paths}
            )

            logger.info(f"Extracted {len(subtitle_paths)} subtitle tracks in {processing_time:.1f}s")

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Subtitle extraction timed out"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

    def convert_subtitle_format(
        self,
        input_path: Path,
        output_path: Path,
        target_format: str = "srt"
    ) -> ProcessingResult:
        """
        Convert subtitle format

        Args:
            input_path: Input subtitle file
            output_path: Output subtitle file
            target_format: Target format (srt, webvtt, ass)

        Returns:
            ProcessingResult
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {input_path}")

        logger.info(f"Converting subtitle: {input_path} -> {output_path} ({target_format})")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build task
        task = ProcessingTask(
            task_id=f"convert_subtitle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            operation=ProcessingOperation.CONVERT_SUBTITLES,
            input_path=input_path,
            output_path=output_path,
            parameters=ProcessingParameters(
                operation=ProcessingOperation.CONVERT_SUBTITLES,
                subtitle_format=target_format
            ),
            started_at=datetime.now()
        )

        try:
            # Build command
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-c:s", target_format,
                "-y",
                str(output_path)
            ]

            # Run ffmpeg
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
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

            logger.info(f"Subtitle conversion complete: {processing_time:.1f}s")

            return processing_result

        except subprocess.TimeoutExpired:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=["Subtitle conversion timed out"]
            )
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            return ProcessingResult(
                task=task,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )

    def _get_subtitle_stream_count(self, video_path: Path) -> int:
        """Get number of subtitle streams in video"""
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index",
            "-of", "csv=p=0"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Count lines (each line is a subtitle stream)
            count = len([line for line in result.stdout.strip().split('\n') if line])
            return count

        except Exception as e:
            logger.warning(f"Failed to count subtitle streams: {e}")
            return 0
