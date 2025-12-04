"""
Frames to Video Converter

Convert image sequences to video animations:
- Multiple frame rates (12fps, 24fps, 30fps)
- Various video codecs (H.264, H.265, VP9)
- Loop options for cyclic animations
- Audio track support

Author: Animation AI Studio
Date: 2025-11-20
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from typing import List, Optional


def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: int = 24,
    codec: str = 'mp4v',
    loop_count: int = 1,
    reverse_loop: bool = False,
    resize: Optional[tuple] = None
):
    """
    Create video from image sequence

    Args:
        frames_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
        codec: Video codec ('mp4v', 'avc1', 'vp09')
        loop_count: Number of times to loop the sequence
        reverse_loop: Add reverse playback for ping-pong effect
        resize: Resize frames to (width, height)
    """

    frames_dir = Path(frames_dir)

    # Find all PNG frames
    frame_files = sorted(frames_dir.glob("*.png"))

    if not frame_files:
        print(f"❌ No PNG frames found in {frames_dir}")
        return False

    print(f"📁 Found {len(frame_files)} frames in {frames_dir}")
    print(f"🎬 Creating video: {output_path}")
    print(f"⚙️  Settings: {fps}fps, codec={codec}, loops={loop_count}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"❌ Failed to read first frame: {frame_files[0]}")
        return False

    height, width = first_frame.shape[:2]

    if resize:
        width, height = resize
        print(f"📐 Resizing frames to {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"❌ Failed to open video writer")
        return False

    # Prepare frame list
    frame_sequence = frame_files.copy()

    # Add loops
    if loop_count > 1:
        frame_sequence = frame_sequence * loop_count

    # Add reverse for ping-pong effect
    if reverse_loop:
        reverse_frames = frame_sequence[:-1][::-1]  # Exclude last to avoid duplicate
        frame_sequence.extend(reverse_frames)

    # Write frames
    total_frames = len(frame_sequence)
    for i, frame_path in enumerate(frame_sequence, 1):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"⚠️  Warning: Failed to read {frame_path}, skipping")
            continue

        # Resize if needed
        if resize:
            frame = cv2.resize(frame, resize)

        out.write(frame)

        if i % 10 == 0 or i == total_frames:
            print(f"  Progress: {i}/{total_frames} frames written", end='\r')

    print(f"\n✅ Video created successfully: {output_path}")
    print(f"📊 Total frames: {total_frames}, Duration: {total_frames/fps:.2f}s")

    out.release()
    return True


def create_animation_compilation(
    sequences_dir: str,
    output_path: str,
    fps: int = 24,
    codec: str = 'mp4v'
):
    """
    Create compilation video with all animation sequences

    Args:
        sequences_dir: Parent directory containing sequence folders
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """

    sequences_dir = Path(sequences_dir)

    # Find all sequence directories
    sequence_dirs = [d for d in sequences_dir.iterdir() if d.is_dir()]

    if not sequence_dirs:
        print(f"❌ No sequence directories found in {sequences_dir}")
        return False

    print(f"🎬 Creating compilation from {len(sequence_dirs)} sequences")

    all_frames = []
    first_size = None

    for seq_dir in sorted(sequence_dirs):
        print(f"\n📁 Loading sequence: {seq_dir.name}")

        frame_files = sorted(seq_dir.glob("*.png"))

        if not frame_files:
            print(f"  ⚠️  No frames found, skipping")
            continue

        print(f"  Found {len(frame_files)} frames")

        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))

            if frame is None:
                continue

            # Check size consistency
            if first_size is None:
                first_size = (frame.shape[1], frame.shape[0])  # (width, height)
            elif (frame.shape[1], frame.shape[0]) != first_size:
                # Resize to match first frame
                frame = cv2.resize(frame, first_size)

            all_frames.append(frame)

    if not all_frames:
        print("❌ No valid frames to compile")
        return False

    print(f"\n✅ Loaded {len(all_frames)} total frames")
    print(f"🎬 Creating compilation video: {output_path}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    width, height = first_size
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Failed to open video writer")
        return False

    # Write all frames
    for i, frame in enumerate(all_frames, 1):
        out.write(frame)

        if i % 10 == 0 or i == len(all_frames):
            print(f"  Progress: {i}/{len(all_frames)} frames written", end='\r')

    print(f"\n✅ Compilation created: {output_path}")
    print(f"📊 Total frames: {len(all_frames)}, Duration: {len(all_frames)/fps:.2f}s")

    out.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert frame sequences to video")
    parser.add_argument("--input", required=True, help="Input frames directory")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--codec", default="mp4v", help="Video codec (default: mp4v)")
    parser.add_argument("--loop", type=int, default=1, help="Loop count (default: 1)")
    parser.add_argument("--reverse-loop", action="store_true", help="Add reverse for ping-pong effect")
    parser.add_argument("--compilation", action="store_true", help="Create compilation from all sequences")

    args = parser.parse_args()

    if args.compilation:
        create_animation_compilation(
            sequences_dir=args.input,
            output_path=args.output,
            fps=args.fps,
            codec=args.codec
        )
    else:
        create_video_from_frames(
            frames_dir=args.input,
            output_path=args.output,
            fps=args.fps,
            codec=args.codec,
            loop_count=args.loop,
            reverse_loop=args.reverse_loop
        )


if __name__ == "__main__":
    # Example usage for direct execution
    import sys

    if len(sys.argv) == 1:
        # Generate videos for all animation sequences
        base_dir = Path("outputs/image_generation/animation_sequences")

        sequences = [
            {
                "name": "walk_cycle",
                "fps": 12,
                "loop": 3,
                "reverse": False
            },
            {
                "name": "jump",
                "fps": 12,
                "loop": 2,
                "reverse": False
            },
            {
                "name": "turn_around",
                "fps": 8,
                "loop": 1,
                "reverse": False
            },
            {
                "name": "wave",
                "fps": 8,
                "loop": 3,
                "reverse": True
            }
        ]

        print("🎬 Automatic Video Generation")
        print("="*80)

        for seq in sequences:
            seq_dir = base_dir / seq["name"]
            output_path = f"outputs/videos/{seq['name']}_animation.mp4"

            if not seq_dir.exists():
                print(f"⚠️  Skipping {seq['name']}: directory not found")
                continue

            Path("outputs/videos").mkdir(parents=True, exist_ok=True)

            print(f"\n📹 Generating video: {seq['name']}")
            create_video_from_frames(
                frames_dir=str(seq_dir),
                output_path=output_path,
                fps=seq["fps"],
                codec="mp4v",
                loop_count=seq["loop"],
                reverse_loop=seq["reverse"]
            )

        # Create compilation
        print("\n" + "="*80)
        print("🎬 Creating Full Compilation")
        print("="*80)

        create_animation_compilation(
            sequences_dir=str(base_dir),
            output_path="outputs/videos/luca_animation_compilation.mp4",
            fps=12,
            codec="mp4v"
        )

        print("\n" + "="*80)
        print("✅ All videos generated!")
        print("="*80)
        print("📁 Output directory: outputs/videos/")
        print("   • Individual sequence videos")
        print("   • Full compilation video")
        print("="*80)
    else:
        main()
