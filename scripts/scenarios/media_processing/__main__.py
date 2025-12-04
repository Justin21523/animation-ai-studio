#!/usr/bin/env python3
"""
Media Processing CLI

Command-line interface for media processing operations.
Supports analysis, transcoding, frame extraction, audio processing, and batch workflows.

Usage:
    python -m scripts.scenarios.media_processing analyze <input>
    python -m scripts.scenarios.media_processing transcode <input> <output> [options]
    python -m scripts.scenarios.media_processing extract-frames <input> <output-dir> [options]
    python -m scripts.scenarios.media_processing extract-audio <input> <output> [options]
    python -m scripts.scenarios.media_processing batch <workflow-file>

Author: Animation AI Studio
Date: 2025-12-03
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .processor import MediaProcessor
from .common import (
    ProcessingParameters,
    ProcessingOperation,
    VideoCodec,
    AudioCodec,
    QualityLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_analyze(args):
    """Analyze media file"""
    processor = MediaProcessor(
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe
    )

    result = processor.analyze_media(
        input_path=Path(args.input),
        detect_scenes=args.detect_scenes,
        scene_threshold=args.scene_threshold
    )

    # Print results
    metadata = result["metadata"]
    quality = result["quality"]

    print(f"\n=== Media Analysis: {args.input} ===")
    print(f"Type: {metadata.media_type.value}")
    print(f"Container: {metadata.container_format.value}")
    print(f"Duration: {metadata.duration:.2f}s" if metadata.duration else "Duration: N/A")
    print(f"File Size: {metadata.file_size_bytes / 1024**2:.2f} MB")

    if metadata.has_video:
        vs = metadata.primary_video_stream
        print(f"\nVideo: {vs.codec.value}, {vs.width}x{vs.height} @ {vs.fps:.2f} fps")
        print(f"Bitrate: {vs.bitrate / 1000:.0f} kbps" if vs.bitrate else "Bitrate: N/A")

    if metadata.has_audio:
        aus = metadata.primary_audio_stream
        print(f"\nAudio: {aus.codec.value}, {aus.channels} channels, {aus.sample_rate} Hz")
        print(f"Bitrate: {aus.bitrate / 1000:.0f} kbps" if aus.bitrate else "Bitrate: N/A")

    print(f"\nQuality: {quality.quality_rating.value} ({quality.overall_score:.1f}/100)")

    if quality.detected_issues:
        print("\nIssues:")
        for issue in quality.detected_issues:
            print(f"  - {issue}")

    if quality.recommendations:
        print("\nRecommendations:")
        for rec in quality.recommendations:
            print(f"  - {rec}")

    if result["scenes"]:
        print(f"\nScenes: {len(result['scenes'])} detected")

    # Save JSON if requested
    if args.output:
        output_data = {
            "metadata": {
                "type": metadata.media_type.value,
                "duration": metadata.duration,
                "file_size": metadata.file_size_bytes
            },
            "quality": {
                "rating": quality.quality_rating.value,
                "score": quality.overall_score,
                "issues": quality.detected_issues,
                "recommendations": quality.recommendations
            }
        }

        if result["scenes"]:
            output_data["scenes"] = [
                {
                    "start": s.start_time,
                    "end": s.end_time,
                    "duration": s.duration,
                    "frame_count": s.frame_count
                }
                for s in result["scenes"]
            ]

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nAnalysis saved to: {args.output}")


def cmd_transcode(args):
    """Transcode video"""
    # Build parameters
    codec_map = {
        "h264": VideoCodec.H264,
        "h265": VideoCodec.H265,
        "vp9": VideoCodec.VP9,
        "av1": VideoCodec.AV1
    }
    quality_map = {
        "low": QualityLevel.LOW,
        "medium": QualityLevel.MEDIUM,
        "high": QualityLevel.HIGH,
        "ultra": QualityLevel.ULTRA
    }

    params = ProcessingParameters(
        operation=ProcessingOperation.TRANSCODE_VIDEO,
        video_codec=codec_map.get(args.codec, VideoCodec.H264),
        quality_level=quality_map.get(args.quality, QualityLevel.MEDIUM),
        video_crf=args.crf,
        target_resolution=tuple(map(int, args.resolution.split('x'))) if args.resolution else None,
        target_fps=args.fps,
        cpu_threads=args.threads,
        overwrite=True
    )

    processor = MediaProcessor(
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe
    )

    result = processor.transcode_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        params=params
    )

    print(f"\n=== Transcode Complete ===")
    print(f"Output: {result.output_path}")
    print(f"Time: {result.processing_time:.1f}s")
    print(f"Input: {result.input_size_bytes / 1024**2:.2f} MB")
    print(f"Output: {result.output_size_bytes / 1024**2:.2f} MB")
    print(f"Compression: {result.compression_ratio:.2%}")


def cmd_extract_frames(args):
    """Extract frames from video"""
    params = ProcessingParameters(
        operation=ProcessingOperation.EXTRACT_FRAMES,
        frame_interval=args.interval,
        frame_quality=args.quality,
        overwrite=True
    )

    processor = MediaProcessor(
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe
    )

    result = processor.extract_frames(
        video_path=Path(args.input),
        output_dir=Path(args.output_dir),
        params=params
    )

    frame_count = len(result.additional_outputs.get("frame_paths", []))

    print(f"\n=== Frame Extraction Complete ===")
    print(f"Output: {result.output_path}")
    print(f"Frames: {frame_count}")
    print(f"Time: {result.processing_time:.1f}s")


def cmd_extract_audio(args):
    """Extract audio from video"""
    codec_map = {
        "aac": AudioCodec.AAC,
        "mp3": AudioCodec.MP3,
        "opus": AudioCodec.OPUS,
        "flac": AudioCodec.FLAC
    }

    params = ProcessingParameters(
        operation=ProcessingOperation.EXTRACT_AUDIO,
        audio_codec=codec_map.get(args.codec, AudioCodec.AAC),
        audio_bitrate=args.bitrate * 1000 if args.bitrate else None,
        audio_sample_rate=args.sample_rate,
        audio_channels=args.channels,
        overwrite=True
    )

    processor = MediaProcessor(
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe
    )

    result = processor.extract_audio(
        video_path=Path(args.input),
        output_path=Path(args.output),
        params=params
    )

    print(f"\n=== Audio Extraction Complete ===")
    print(f"Output: {result.output_path}")
    print(f"Size: {result.output_size_bytes / 1024**2:.2f} MB")
    print(f"Time: {result.processing_time:.1f}s")


def cmd_batch(args):
    """Run batch workflow from file"""
    # Load workflow
    with open(args.workflow_file, 'r') as f:
        workflow_config = json.load(f)

    input_path = Path(workflow_config["input"])
    output_dir = Path(workflow_config.get("output_dir", input_path.parent / "output"))
    workflow = workflow_config["workflow"]

    processor = MediaProcessor(
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe
    )

    result = processor.process_media(
        input_path=input_path,
        output_dir=output_dir,
        workflow=workflow,
        checkpoint_enabled=not args.no_checkpoint
    )

    print(f"\n=== Batch Processing Complete ===")
    print(f"Input: {result['input_path']}")
    print(f"Output: {result['output_dir']}")
    print(f"Operations: {len(result['results'])}")
    print(f"Status: {result['status']}")

    for op_name, op_result in result["results"].items():
        if hasattr(op_result, 'processing_time'):
            print(f"  - {op_name}: {op_result.processing_time:.1f}s")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Media Processing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze media
  python -m scripts.scenarios.media_processing analyze video.mp4

  # Transcode to H.265
  python -m scripts.scenarios.media_processing transcode input.mp4 output.mp4 --codec h265 --quality high

  # Extract frames every 30 frames
  python -m scripts.scenarios.media_processing extract-frames video.mp4 frames/ --interval 30

  # Extract audio as AAC
  python -m scripts.scenarios.media_processing extract-audio video.mp4 audio.aac --codec aac --bitrate 192

  # Run batch workflow
  python -m scripts.scenarios.media_processing batch workflow.json
        """
    )

    # Global options
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze media file")
    analyze_parser.add_argument("input", help="Input media file")
    analyze_parser.add_argument("--output", "-o", help="Output JSON file")
    analyze_parser.add_argument("--detect-scenes", action="store_true", help="Detect scene boundaries")
    analyze_parser.add_argument("--scene-threshold", type=float, default=0.3, help="Scene detection threshold")

    # Transcode command
    transcode_parser = subparsers.add_parser("transcode", help="Transcode video")
    transcode_parser.add_argument("input", help="Input video file")
    transcode_parser.add_argument("output", help="Output video file")
    transcode_parser.add_argument("--codec", choices=["h264", "h265", "vp9", "av1"], default="h264")
    transcode_parser.add_argument("--quality", choices=["low", "medium", "high", "ultra"], default="medium")
    transcode_parser.add_argument("--crf", type=int, help="CRF value (overrides quality preset)")
    transcode_parser.add_argument("--resolution", help="Target resolution (e.g., 1920x1080)")
    transcode_parser.add_argument("--fps", type=float, help="Target FPS")
    transcode_parser.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto)")

    # Extract frames command
    frames_parser = subparsers.add_parser("extract-frames", help="Extract frames from video")
    frames_parser.add_argument("input", help="Input video file")
    frames_parser.add_argument("output_dir", help="Output directory")
    frames_parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval")
    frames_parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")

    # Extract audio command
    audio_parser = subparsers.add_parser("extract-audio", help="Extract audio from video")
    audio_parser.add_argument("input", help="Input video file")
    audio_parser.add_argument("output", help="Output audio file")
    audio_parser.add_argument("--codec", choices=["aac", "mp3", "opus", "flac"], default="aac")
    audio_parser.add_argument("--bitrate", type=int, help="Audio bitrate in kbps")
    audio_parser.add_argument("--sample-rate", type=int, help="Sample rate in Hz")
    audio_parser.add_argument("--channels", type=int, help="Audio channels")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch workflow")
    batch_parser.add_argument("workflow_file", help="Workflow configuration JSON file")
    batch_parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoints")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Dispatch command
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "transcode":
        cmd_transcode(args)
    elif args.command == "extract-frames":
        cmd_extract_frames(args)
    elif args.command == "extract-audio":
        cmd_extract_audio(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
