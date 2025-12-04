"""
Metadata Extractor

Extracts comprehensive metadata from media files using ffprobe.
CPU-only operation with no GPU dependencies.

Author: Animation AI Studio
Date: 2025-12-03
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..common import (
    MediaType,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    MediaMetadata,
    VideoStreamInfo,
    AudioStreamInfo,
    SubtitleStreamInfo
)

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract comprehensive metadata from media files using ffprobe

    Features:
    - Parse video/audio/subtitle streams
    - Detect media type and container format
    - Extract codec, resolution, bitrate, duration
    - Handle corrupted or incomplete files
    - CPU-only operation (no GPU)

    Example:
        extractor = MetadataExtractor(ffprobe_path="ffprobe")
        metadata = extractor.extract_metadata(Path("/path/to/video.mp4"))

        print(f"Duration: {metadata.duration}s")
        print(f"Resolution: {metadata.primary_video_stream.resolution}")
        print(f"Video codec: {metadata.primary_video_stream.codec}")
    """

    def __init__(self, ffprobe_path: str = "ffprobe"):
        """
        Initialize metadata extractor

        Args:
            ffprobe_path: Path to ffprobe executable (default: "ffprobe" in PATH)
        """
        self.ffprobe_path = ffprobe_path

        # Verify ffprobe is available
        self._verify_ffprobe()

        logger.info(f"MetadataExtractor initialized with ffprobe: {self.ffprobe_path}")

    def _verify_ffprobe(self):
        """Verify ffprobe is available and working"""
        try:
            result = subprocess.run(
                [self.ffprobe_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            logger.debug(f"ffprobe version: {result.stdout.split()[2]}")

        except FileNotFoundError:
            raise RuntimeError(
                f"ffprobe not found at: {self.ffprobe_path}. "
                "Please install ffmpeg or specify correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe verification timed out")

    def extract_metadata(self, path: Path) -> MediaMetadata:
        """
        Extract comprehensive metadata from media file

        Args:
            path: Path to media file

        Returns:
            MediaMetadata object with all extracted information

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If ffprobe fails or file is corrupted
        """
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        logger.info(f"Extracting metadata from: {path}")

        # Run ffprobe to get JSON output
        probe_data = self._run_ffprobe(path)

        # Parse streams
        video_streams = self._parse_video_streams(probe_data.get("streams", []))
        audio_streams = self._parse_audio_streams(probe_data.get("streams", []))
        subtitle_streams = self._parse_subtitle_streams(probe_data.get("streams", []))

        # Parse container metadata
        format_data = probe_data.get("format", {})

        # Detect media type
        media_type = self.detect_media_type(video_streams, audio_streams, subtitle_streams)

        # Detect container format
        container_format = self._detect_container_format(path, format_data)

        # Extract file metadata
        file_size_bytes = int(format_data.get("size", path.stat().st_size))
        duration = float(format_data.get("duration", 0.0)) if format_data.get("duration") else None

        # Extract timestamps
        stat = path.stat()
        created_time = datetime.fromtimestamp(stat.st_ctime)
        modified_time = datetime.fromtimestamp(stat.st_mtime)

        # Extract metadata tags
        metadata_tags = format_data.get("tags", {})

        # Create MediaMetadata object
        metadata = MediaMetadata(
            path=path,
            media_type=media_type,
            container_format=container_format,
            file_size_bytes=file_size_bytes,
            duration=duration,
            created_time=created_time,
            modified_time=modified_time,
            video_streams=video_streams,
            audio_streams=audio_streams,
            subtitle_streams=subtitle_streams,
            metadata_tags=metadata_tags
        )

        logger.info(
            f"Metadata extracted: {media_type.value}, "
            f"{len(video_streams)} video, {len(audio_streams)} audio, "
            f"{len(subtitle_streams)} subtitle streams"
        )

        return metadata

    def _run_ffprobe(self, path: Path) -> Dict[str, Any]:
        """
        Run ffprobe and return JSON output

        Args:
            path: Path to media file

        Returns:
            Parsed JSON data from ffprobe

        Raises:
            RuntimeError: If ffprobe fails
        """
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            probe_data = json.loads(result.stdout)
            return probe_data

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffprobe timed out processing: {path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")

    def detect_media_type(
        self,
        video_streams: List[VideoStreamInfo],
        audio_streams: List[AudioStreamInfo],
        subtitle_streams: List[SubtitleStreamInfo]
    ) -> MediaType:
        """
        Detect media type based on available streams

        Args:
            video_streams: List of video streams
            audio_streams: List of audio streams
            subtitle_streams: List of subtitle streams

        Returns:
            Detected MediaType
        """
        has_video = len(video_streams) > 0
        has_audio = len(audio_streams) > 0

        if has_video:
            return MediaType.VIDEO
        elif has_audio:
            return MediaType.AUDIO
        elif len(subtitle_streams) > 0:
            return MediaType.SUBTITLE
        else:
            return MediaType.UNKNOWN

    def _detect_container_format(self, path: Path, format_data: Dict[str, Any]) -> ContainerFormat:
        """
        Detect container format from ffprobe data and file extension

        Args:
            path: File path
            format_data: Format data from ffprobe

        Returns:
            Detected ContainerFormat
        """
        # Try format name from ffprobe first
        format_name = format_data.get("format_name", "").lower()

        format_map = {
            "mp4": ContainerFormat.MP4,
            "mov": ContainerFormat.MOV,
            "matroska": ContainerFormat.MKV,
            "webm": ContainerFormat.WEBM,
            "avi": ContainerFormat.AVI,
            "flv": ContainerFormat.FLV,
            "mpegts": ContainerFormat.TS,
            "ogg": ContainerFormat.OGG,
            "wav": ContainerFormat.WAV,
            "m4a": ContainerFormat.M4A
        }

        # Check format name
        for key, container in format_map.items():
            if key in format_name:
                return container

        # Fallback to extension
        ext_lower = path.suffix.lower().lstrip('.')

        ext_map = {
            "mp4": ContainerFormat.MP4,
            "mkv": ContainerFormat.MKV,
            "webm": ContainerFormat.WEBM,
            "mov": ContainerFormat.MOV,
            "avi": ContainerFormat.AVI,
            "flv": ContainerFormat.FLV,
            "ts": ContainerFormat.TS,
            "m4a": ContainerFormat.M4A,
            "ogg": ContainerFormat.OGG,
            "wav": ContainerFormat.WAV
        }

        return ext_map.get(ext_lower, ContainerFormat.UNKNOWN)

    def _parse_video_streams(self, streams: List[Dict[str, Any]]) -> List[VideoStreamInfo]:
        """
        Parse video streams from ffprobe data

        Args:
            streams: List of stream data from ffprobe

        Returns:
            List of VideoStreamInfo objects
        """
        video_streams = []

        for stream in streams:
            if stream.get("codec_type") != "video":
                continue

            try:
                # Parse codec
                codec_name = stream.get("codec_name", "unknown").lower()
                codec = self._parse_video_codec(codec_name)

                # Parse resolution
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))

                # Parse fps
                fps_str = stream.get("r_frame_rate", "0/1")
                fps = self._parse_fps(fps_str)

                # Parse bitrate (optional)
                bitrate = None
                if "bit_rate" in stream:
                    bitrate = int(stream["bit_rate"])

                # Parse duration (optional)
                duration = None
                if "duration" in stream:
                    duration = float(stream["duration"])

                # Parse frame count (optional)
                frame_count = None
                if "nb_frames" in stream:
                    frame_count = int(stream["nb_frames"])

                # Parse pixel format
                pixel_format = stream.get("pix_fmt")

                # Parse color space
                color_space = stream.get("color_space")

                video_info = VideoStreamInfo(
                    index=stream.get("index", 0),
                    codec=codec,
                    width=width,
                    height=height,
                    fps=fps,
                    bitrate=bitrate,
                    duration=duration,
                    frame_count=frame_count,
                    pixel_format=pixel_format,
                    color_space=color_space
                )

                video_streams.append(video_info)

                logger.debug(
                    f"Parsed video stream: {codec.value}, "
                    f"{width}x{height}, {fps:.2f} fps"
                )

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse video stream: {e}")
                continue

        return video_streams

    def _parse_audio_streams(self, streams: List[Dict[str, Any]]) -> List[AudioStreamInfo]:
        """
        Parse audio streams from ffprobe data

        Args:
            streams: List of stream data from ffprobe

        Returns:
            List of AudioStreamInfo objects
        """
        audio_streams = []

        for stream in streams:
            if stream.get("codec_type") != "audio":
                continue

            try:
                # Parse codec
                codec_name = stream.get("codec_name", "unknown").lower()
                codec = self._parse_audio_codec(codec_name)

                # Parse sample rate
                sample_rate = int(stream.get("sample_rate", 0))

                # Parse channels
                channels = int(stream.get("channels", 0))

                # Parse bitrate (optional)
                bitrate = None
                if "bit_rate" in stream:
                    bitrate = int(stream["bit_rate"])

                # Parse duration (optional)
                duration = None
                if "duration" in stream:
                    duration = float(stream["duration"])

                # Parse language (optional)
                language = None
                if "tags" in stream and "language" in stream["tags"]:
                    language = stream["tags"]["language"]

                # Parse channel layout
                channel_layout = stream.get("channel_layout")

                audio_info = AudioStreamInfo(
                    index=stream.get("index", 0),
                    codec=codec,
                    sample_rate=sample_rate,
                    channels=channels,
                    bitrate=bitrate,
                    duration=duration,
                    language=language,
                    channel_layout=channel_layout
                )

                audio_streams.append(audio_info)

                logger.debug(
                    f"Parsed audio stream: {codec.value}, "
                    f"{sample_rate}Hz, {channels}ch"
                )

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse audio stream: {e}")
                continue

        return audio_streams

    def _parse_subtitle_streams(self, streams: List[Dict[str, Any]]) -> List[SubtitleStreamInfo]:
        """
        Parse subtitle streams from ffprobe data

        Args:
            streams: List of stream data from ffprobe

        Returns:
            List of SubtitleStreamInfo objects
        """
        subtitle_streams = []

        for stream in streams:
            if stream.get("codec_type") != "subtitle":
                continue

            try:
                # Parse format
                codec_name = stream.get("codec_name", "unknown")

                # Parse language (optional)
                language = None
                if "tags" in stream and "language" in stream["tags"]:
                    language = stream["tags"]["language"]

                # Parse title (optional)
                title = None
                if "tags" in stream and "title" in stream["tags"]:
                    title = stream["tags"]["title"]

                # Encoding is usually in tags
                encoding = None
                if "tags" in stream and "encoding" in stream["tags"]:
                    encoding = stream["tags"]["encoding"]

                subtitle_info = SubtitleStreamInfo(
                    index=stream.get("index", 0),
                    format=codec_name,
                    language=language,
                    title=title,
                    encoding=encoding
                )

                subtitle_streams.append(subtitle_info)

                logger.debug(
                    f"Parsed subtitle stream: {codec_name}, "
                    f"language={language}"
                )

            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse subtitle stream: {e}")
                continue

        return subtitle_streams

    def _parse_video_codec(self, codec_name: str) -> VideoCodec:
        """
        Parse video codec from codec name

        Args:
            codec_name: Codec name from ffprobe

        Returns:
            VideoCodec enum value
        """
        codec_map = {
            "h264": VideoCodec.H264,
            "avc": VideoCodec.H264,
            "h265": VideoCodec.H265,
            "hevc": VideoCodec.H265,
            "vp9": VideoCodec.VP9,
            "av1": VideoCodec.AV1,
            "mpeg4": VideoCodec.MPEG4,
            "mpeg2video": VideoCodec.MPEG2,
            "theora": VideoCodec.THEORA,
            "prores": VideoCodec.PRORES
        }

        for key, codec in codec_map.items():
            if key in codec_name:
                return codec

        return VideoCodec.UNKNOWN

    def _parse_audio_codec(self, codec_name: str) -> AudioCodec:
        """
        Parse audio codec from codec name

        Args:
            codec_name: Codec name from ffprobe

        Returns:
            AudioCodec enum value
        """
        codec_map = {
            "aac": AudioCodec.AAC,
            "mp3": AudioCodec.MP3,
            "opus": AudioCodec.OPUS,
            "vorbis": AudioCodec.VORBIS,
            "flac": AudioCodec.FLAC,
            "pcm": AudioCodec.PCM,
            "ac3": AudioCodec.AC3,
            "eac3": AudioCodec.EAC3
        }

        for key, codec in codec_map.items():
            if key in codec_name:
                return codec

        return AudioCodec.UNKNOWN

    def _parse_fps(self, fps_str: str) -> float:
        """
        Parse FPS from fractional string (e.g., "30000/1001")

        Args:
            fps_str: FPS string in format "num/den"

        Returns:
            FPS as float
        """
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            else:
                return float(fps_str)
        except (ValueError, ZeroDivisionError):
            logger.warning(f"Failed to parse FPS: {fps_str}")
            return 0.0
