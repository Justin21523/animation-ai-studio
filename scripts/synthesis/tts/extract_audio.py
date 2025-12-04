#!/usr/bin/env python3
"""
Extract Audio from Video

Extracts audio track from video files for voice training.

Usage:
    python scripts/synthesis/tts/extract_audio.py \
        --input /path/to/video.mp4 \
        --output audio.wav

    # Extract for specific film
    python scripts/synthesis/tts/extract_audio.py \
        --film luca \
        --output data/films/luca/audio/luca_audio.wav
"""

import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract and process audio from video files"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.raw_videos_dir = Path("/mnt/c/raw_videos")

    def get_video_info(self, video_path: Path) -> dict:
        """
        Get video metadata using ffprobe

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)

            # Find audio stream
            audio_streams = [s for s in metadata.get('streams', []) if s['codec_type'] == 'audio']

            info = {
                'duration': float(metadata['format'].get('duration', 0)),
                'size': int(metadata['format'].get('size', 0)),
                'format_name': metadata['format'].get('format_name', 'unknown'),
                'audio_streams': len(audio_streams),
            }

            if audio_streams:
                audio_stream = audio_streams[0]
                info.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'bit_rate': int(audio_stream.get('bit_rate', 0)) if 'bit_rate' in audio_stream else None,
                })

            return info

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing video info: {e}")
            return {}

    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        sample_rate: int = 48000,
        channels: int = 2,
        audio_codec: str = 'pcm_s16le',
        format: str = 'wav'
    ) -> bool:
        """
        Extract audio from video using ffmpeg

        Args:
            video_path: Input video file
            output_path: Output audio file
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels (1=mono, 2=stereo)
            audio_codec: Audio codec (pcm_s16le for WAV)
            format: Output format

        Returns:
            True if successful
        """
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', audio_codec,
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-y',  # Overwrite output file
            str(output_path)
        ]

        logger.info(f"Extracting audio from: {video_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Settings: {sample_rate}Hz, {channels}ch, {audio_codec}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Audio extracted successfully: {size_mb:.2f} MB")
                return True
            else:
                logger.error("Audio file was not created")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return False

    def extract_audio_segment(
        self,
        video_path: Path,
        output_path: Path,
        start_time: float,
        duration: float,
        sample_rate: int = 48000
    ) -> bool:
        """
        Extract specific segment from video

        Args:
            video_path: Input video file
            output_path: Output audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            sample_rate: Audio sample rate

        Returns:
            True if successful
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '2',
            '-y',
            str(output_path)
        ]

        logger.info(f"Extracting segment: {start_time}s to {start_time + duration}s")

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if output_path.exists():
                logger.info(f"✓ Segment extracted: {output_path}")
                return True
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract segment: {e}")
            return False

    def convert_to_mono(self, input_path: Path, output_path: Path) -> bool:
        """
        Convert stereo audio to mono

        Args:
            input_path: Input audio file
            output_path: Output mono audio file

        Returns:
            True if successful
        """
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ac', '1',  # 1 channel = mono
            '-y',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"✓ Converted to mono: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert to mono: {e}")
            return False

    def normalize_audio(self, input_path: Path, output_path: Path, target_db: float = -20.0) -> bool:
        """
        Normalize audio volume

        Args:
            input_path: Input audio file
            output_path: Output normalized audio
            target_db: Target loudness in dB

        Returns:
            True if successful
        """
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-af', f'loudnorm=I={target_db}:TP=-1.5:LRA=11',
            '-y',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"✓ Audio normalized: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to normalize audio: {e}")
            return False

    def extract_for_film(self, film_name: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Extract audio for a specific film

        Args:
            film_name: Name of the film (e.g., 'luca')
            output_dir: Optional output directory

        Returns:
            Path to extracted audio file, or None if failed
        """
        # Find video file
        video_dir = self.raw_videos_dir / film_name
        if not video_dir.exists():
            logger.error(f"Video directory not found: {video_dir}")
            return None

        # Find video files
        video_files = list(video_dir.glob("*.mp4")) + \
                     list(video_dir.glob("*.mkv")) + \
                     list(video_dir.glob("*.ts")) + \
                     list(video_dir.glob("*.avi"))

        if not video_files:
            logger.error(f"No video files found in {video_dir}")
            return None

        video_path = video_files[0]
        logger.info(f"Found video: {video_path}")

        # Get video info
        info = self.get_video_info(video_path)
        if info:
            logger.info(f"Video info:")
            logger.info(f"  Duration: {info.get('duration', 0) / 60:.2f} minutes")
            logger.info(f"  Size: {info.get('size', 0) / (1024**3):.2f} GB")
            logger.info(f"  Audio codec: {info.get('audio_codec', 'unknown')}")
            logger.info(f"  Sample rate: {info.get('sample_rate', 0)} Hz")
            logger.info(f"  Channels: {info.get('channels', 0)}")

        # Set output path
        if output_dir is None:
            output_dir = self.project_root / "data" / "films" / film_name / "audio"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{film_name}_audio.wav"

        # Extract audio
        success = self.extract_audio(
            video_path=video_path,
            output_path=output_path,
            sample_rate=48000,  # High quality for voice training
            channels=2,  # Stereo (will convert to mono later if needed)
        )

        if success:
            return output_path
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract audio from video for voice training")
    parser.add_argument('--input', '-i', type=str, help="Input video file")
    parser.add_argument('--output', '-o', type=str, help="Output audio file")
    parser.add_argument('--film', '-f', type=str, help="Film name (e.g., luca)")
    parser.add_argument('--sample-rate', type=int, default=48000, help="Audio sample rate (default: 48000)")
    parser.add_argument('--mono', action='store_true', help="Convert to mono")
    parser.add_argument('--normalize', action='store_true', help="Normalize audio volume")
    parser.add_argument('--start', type=float, help="Start time in seconds (for segment extraction)")
    parser.add_argument('--duration', type=float, help="Duration in seconds (for segment extraction)")

    args = parser.parse_args()

    extractor = AudioExtractor()

    # Mode 1: Extract from specific film
    if args.film:
        output_path = extractor.extract_for_film(args.film)
        if output_path:
            print(f"\n{'='*60}")
            print(f"✓ Audio extraction complete!")
            print(f"{'='*60}")
            print(f"Film: {args.film}")
            print(f"Output: {output_path}")
            print(f"Size: {output_path.stat().st_size / (1024**2):.2f} MB")
            print(f"{'='*60}\n")

            # Optional post-processing
            if args.mono:
                mono_path = output_path.parent / f"{output_path.stem}_mono.wav"
                extractor.convert_to_mono(output_path, mono_path)
                print(f"✓ Mono version: {mono_path}")

            if args.normalize:
                norm_path = output_path.parent / f"{output_path.stem}_normalized.wav"
                extractor.normalize_audio(output_path, norm_path)
                print(f"✓ Normalized version: {norm_path}")

        else:
            print("✗ Audio extraction failed")
            return 1

    # Mode 2: Extract from specific file
    elif args.input and args.output:
        video_path = Path(args.input)
        output_path = Path(args.output)

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return 1

        # Segment extraction
        if args.start is not None and args.duration is not None:
            success = extractor.extract_audio_segment(
                video_path=video_path,
                output_path=output_path,
                start_time=args.start,
                duration=args.duration,
                sample_rate=args.sample_rate
            )
        else:
            # Full extraction
            success = extractor.extract_audio(
                video_path=video_path,
                output_path=output_path,
                sample_rate=args.sample_rate,
                channels=1 if args.mono else 2
            )

        if success:
            print(f"\n✓ Audio extracted: {output_path}")

            if args.normalize:
                norm_path = output_path.parent / f"{output_path.stem}_normalized.wav"
                extractor.normalize_audio(output_path, norm_path)
                print(f"✓ Normalized: {norm_path}")

        return 0 if success else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
