"""
Test OpenPose Detector with V3 Generated Images

Test OpenPose on our V3 generated animation frames (high-quality, clear characters).
This should produce much better pose detection results.

Author: Animation AI Studio
Date: 2025-11-21
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

from controlnet_aux import OpenposeDetector
from PIL import Image
from pathlib import Path
import time

def test_openpose_v3():
    """Test OpenPose detector on V3 generated frames"""

    print("=" * 80)
    print("Testing OpenPose Detector on V3 Generated Frames")
    print("=" * 80)

    # Initialize detector
    print("\n1. Loading OpenPose detector...")
    start = time.time()
    detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    load_time = time.time() - start
    print(f"✅ Detector loaded in {load_time:.2f}s")

    # Test on V3 animation frames
    test_dirs = [
        ("walk", Path("outputs/image_generation/animation_v3/walk")),
        ("jump", Path("outputs/image_generation/animation_v3/jump")),
        ("turn", Path("outputs/image_generation/animation_v3/turn"))
    ]

    output_dir = Path("outputs/preprocessing/openpose_v3_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0

    for action, frames_dir in test_dirs:
        if not frames_dir.exists():
            print(f"\n⚠️  {action} directory not found: {frames_dir}")
            continue

        # Get all PNG frames
        frame_files = sorted(frames_dir.glob("*_frame_*.png"))

        if not frame_files:
            print(f"\n⚠️  No frames found in {frames_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {action.upper()} action ({len(frame_files)} frames)")
        print(f"{'='*80}")

        for i, frame_path in enumerate(frame_files, 1):
            print(f"\nFrame {i}/{len(frame_files)}: {frame_path.name}")

            # Load image
            image = Image.open(frame_path)
            print(f"   Image size: {image.size}")

            # Extract pose
            print(f"   Extracting pose...")
            start = time.time()

            pose_image = detector(
                image,
                hand_and_face=True,  # Include hands and face for better detail
                detect_resolution=768,
                image_resolution=1024
            )

            extract_time = time.time() - start

            # Save result
            output_path = output_dir / f"{action}_{frame_path.stem}_pose.png"
            pose_image.save(output_path)

            print(f"   ✅ Extracted in {extract_time:.2f}s → {output_path.name}")
            total_processed += 1

    print("\n" + "=" * 80)
    print("✅ OpenPose V3 Test Complete!")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Total frames processed: {total_processed}")
    print(f"   Actions: walk, jump, turn")
    print("\n💡 Next steps:")
    print("   1. Review pose quality (should show full skeleton)")
    print("   2. Use these poses for ControlNet generation")
    print("   3. Build action library from extracted poses")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_openpose_v3()
