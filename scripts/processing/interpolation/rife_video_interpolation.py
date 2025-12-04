"""
RIFE Video Frame Interpolation

Increase frame rate of animation videos using RIFE (Real-Time Intermediate Flow Estimation).
Perfect for making choppy 12fps animations smooth and fluid at 24fps, 48fps, or even 60fps.

Features:
- 2x, 4x, 8x frame rate multiplication
- GPU-accelerated interpolation
- Batch processing support
- Quality preservation

Author: Animation AI Studio
Date: 2025-11-20
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import subprocess


class RIFEInterpolator:
    """RIFE-based video frame interpolation"""

    def __init__(
        self,
        rife_dir: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/video/frame-interpolation/Practical-RIFE",
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize RIFE interpolator

        Args:
            rife_dir: Path to Practical-RIFE repository
            model_path: Path to RIFE model weights
            device: Device to use (cuda/cpu)
        """
        self.rife_dir = Path(rife_dir)
        self.device = device

        if model_path is None:
            # Use default model path
            self.model_path = self.rife_dir / "train_log"
        else:
            self.model_path = Path(model_path)

        # Check if RIFE exists
        if not self.rife_dir.exists():
            raise FileNotFoundError(f"RIFE directory not found: {self.rife_dir}")

        print(f"RIFE Interpolator initialized")
        print(f"  RIFE dir: {self.rife_dir}")
        print(f"  Model path: {self.model_path}")
        print(f"  Device: {self.device}")

    def interpolate_video(
        self,
        input_video: str,
        output_video: str,
        multiplier: int = 2,
        fps: Optional[float] = None,
        scale: float = 1.0,
        use_fp16: bool = True
    ):
        """
        Interpolate video frames using RIFE

        Args:
            input_video: Input video path
            output_video: Output video path
            multiplier: Frame rate multiplier (2, 4, 8, etc.)
            fps: Output FPS (if None, auto-calculated from multiplier)
            scale: Downscale factor for processing (1.0 = original size)
            use_fp16: Use FP16 for faster processing
        """
        input_path = Path(input_video)
        output_path = Path(output_video)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get input video info
        cap = cv2.VideoCapture(str(input_path))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate output FPS
        if fps is None:
            output_fps = input_fps * multiplier
        else:
            output_fps = fps

        print(f"\n{'='*80}")
        print(f"RIFE Video Interpolation")
        print(f"{'='*80}")
        print(f"Input: {input_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {input_fps:.2f}")
        print(f"  Frames: {total_frames}")
        print(f"\nOutput: {output_path.name}")
        print(f"  FPS: {output_fps:.2f}")
        print(f"  Multiplier: {multiplier}x")
        print(f"  Expected frames: {total_frames * multiplier}")
        print(f"{'='*80}\n")

        # Build RIFE command
        cmd = [
            sys.executable,
            str(self.rife_dir / "inference_video.py"),
            "--video", str(input_path),
            "--output", str(output_path),
            "--multi", str(multiplier),
            "--fps", str(output_fps),
            "--scale", str(scale)
        ]

        if use_fp16:
            cmd.append("--fp16")

        # Run RIFE
        print("Starting RIFE interpolation...")
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ RIFE failed with error:")
            print(result.stderr)
            raise RuntimeError("RIFE interpolation failed")

        print(result.stdout)
        print(f"\n✅ Interpolation complete: {output_path}")

        # Verify output
        if output_path.exists():
            cap = cv2.VideoCapture(str(output_path))
            out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            print(f"\nOutput verification:")
            print(f"  Frames: {out_frames}")
            print(f"  FPS: {out_fps:.2f}")
            print(f"  Duration: {out_frames/out_fps:.2f}s")

        return output_path

    def batch_interpolate(
        self,
        input_dir: str,
        output_dir: str,
        multiplier: int = 2,
        pattern: str = "*.mp4"
    ):
        """
        Batch interpolate all videos in a directory

        Args:
            input_dir: Input directory with videos
            output_dir: Output directory
            multiplier: Frame rate multiplier
            pattern: File pattern to match
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(input_dir.glob(pattern))

        if not video_files:
            print(f"No videos found in {input_dir} matching {pattern}")
            return

        print(f"Found {len(video_files)} videos to interpolate")

        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*80}")
            print(f"Processing {i}/{len(video_files)}: {video_path.name}")
            print(f"{'='*80}")

            output_path = output_dir / f"{video_path.stem}_interpolated_{multiplier}x{video_path.suffix}"

            try:
                self.interpolate_video(
                    input_video=str(video_path),
                    output_video=str(output_path),
                    multiplier=multiplier
                )
            except Exception as e:
                print(f"❌ Failed to process {video_path.name}: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"✅ Batch interpolation complete!")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}")


def main():
    """Interpolate V3 animation videos"""

    parser = argparse.ArgumentParser(description="RIFE Video Frame Interpolation")
    parser.add_argument("--input", required=True, help="Input video or directory")
    parser.add_argument("--output", required=True, help="Output video or directory")
    parser.add_argument("--multiplier", type=int, default=2, help="Frame rate multiplier (default: 2)")
    parser.add_argument("--fps", type=float, help="Output FPS (overrides multiplier)")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    parser.add_argument("--pattern", default="*.mp4", help="File pattern for batch mode")

    args = parser.parse_args()

    # Initialize interpolator
    interpolator = RIFEInterpolator()

    if args.batch:
        # Batch processing
        interpolator.batch_interpolate(
            input_dir=args.input,
            output_dir=args.output,
            multiplier=args.multiplier,
            pattern=args.pattern
        )
    else:
        # Single file processing
        interpolator.interpolate_video(
            input_video=args.input,
            output_video=args.output,
            multiplier=args.multiplier,
            fps=args.fps
        )


if __name__ == "__main__":
    # Example usage for V3 animations
    if len(sys.argv) == 1:
        print("🎬 RIFE Frame Interpolation - V3 Animation Demo")
        print("="*80)

        interpolator = RIFEInterpolator()

        # Example: Interpolate walk cycle from 12fps to 24fps
        v3_videos = Path("outputs/videos")

        test_videos = [
            {"name": "walk_cycle_v3.mp4", "mult": 2, "desc": "12fps → 24fps"},
            {"name": "jump_sequence_v3.mp4", "mult": 2, "desc": "12fps → 24fps"},
            {"name": "turn_around_v3.mp4", "mult": 3, "desc": "8fps → 24fps"},
        ]

        for video_info in test_videos:
            input_path = v3_videos / video_info["name"]

            if not input_path.exists():
                print(f"⚠️  Skipping {video_info['name']}: not found")
                continue

            output_path = Path("outputs/videos/interpolated") / f"{input_path.stem}_smooth{input_path.suffix}"

            print(f"\n📹 {video_info['name']}")
            print(f"   {video_info['desc']}")
            print(f"   Multiplier: {video_info['mult']}x")

            try:
                interpolator.interpolate_video(
                    input_video=str(input_path),
                    output_video=str(output_path),
                    multiplier=video_info["mult"]
                )
            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n" + "="*80)
        print("✅ Demo complete! Check outputs/videos/interpolated/")
        print("="*80)
    else:
        main()
