"""
Test OpenPose Detector

Quick test to verify controlnet_aux OpenPose detection works correctly.
This will download the OpenPose model on first run (~1GB).

Author: Animation AI Studio
Date: 2025-11-21
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

from controlnet_aux import OpenposeDetector
from PIL import Image
from pathlib import Path
import time

def test_openpose_detector():
    """Test OpenPose detector on a sample Luca frame"""

    print("=" * 80)
    print("Testing OpenPose Detector")
    print("=" * 80)

    # Initialize detector (will download model on first run)
    print("\n1. Loading OpenPose detector...")
    print("   (First run will download ~1GB model from HuggingFace)")

    start = time.time()
    detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    load_time = time.time() - start

    print(f"✅ Detector loaded in {load_time:.2f}s")

    # Find a sample Luca frame
    frames_dir = Path("/mnt/data/ai_data/datasets/3d-anime/luca/frames")

    if not frames_dir.exists():
        print(f"\n❌ Error: Frames directory not found: {frames_dir}")
        print("   Please check if Luca film frames are available.")
        return

    # Get first frame
    sample_frames = sorted(frames_dir.glob("*.jpg"))[:5]

    if not sample_frames:
        sample_frames = sorted(frames_dir.glob("*.png"))[:5]

    if not sample_frames:
        print(f"\n❌ Error: No frames found in {frames_dir}")
        return

    print(f"\n2. Found {len(sample_frames)} sample frames")

    # Test on first frame
    test_frame = sample_frames[0]
    print(f"\n3. Testing on: {test_frame.name}")

    # Load image
    image = Image.open(test_frame)
    print(f"   Image size: {image.size}")

    # Extract pose
    print("\n4. Extracting pose...")
    start = time.time()

    pose_image = detector(
        image,
        hand_and_face=False,  # Start simple, no hand/face detection
        detect_resolution=512,
        image_resolution=1024
    )

    extract_time = time.time() - start
    print(f"✅ Pose extracted in {extract_time:.2f}s")

    # Save result
    output_dir = Path("outputs/preprocessing/openpose_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{test_frame.stem}_pose.png"
    pose_image.save(output_path)

    print(f"\n5. Saved pose image: {output_path}")
    print(f"   Size: {pose_image.size}")

    # Test multiple frames
    print("\n" + "=" * 80)
    print("Testing on multiple frames...")
    print("=" * 80)

    for i, frame_path in enumerate(sample_frames[1:4], 2):
        print(f"\nFrame {i}/{len(sample_frames[:4])}: {frame_path.name}")

        image = Image.open(frame_path)
        start = time.time()

        pose_image = detector(
            image,
            hand_and_face=False,
            detect_resolution=512,
            image_resolution=1024
        )

        extract_time = time.time() - start

        output_path = output_dir / f"{frame_path.stem}_pose.png"
        pose_image.save(output_path)

        print(f"✅ Extracted in {extract_time:.2f}s → {output_path.name}")

    print("\n" + "=" * 80)
    print("✅ OpenPose Test Complete!")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Check the pose skeleton images to verify quality")
    print("\n💡 Next steps:")
    print("   1. Review pose quality (skeleton should match character)")
    print("   2. Build pose_extractor.py module")
    print("   3. Extract poses from more frames")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_openpose_detector()
