"""
LTX-Video Image-to-Video Generation

Generate short video clips from single images using LTX-Video.
Perfect for extending animation frames into smooth 3-5 second clips.

Features:
- Image-to-video generation
- Text prompt control
- 768x512 @ 24fps output
- 16GB VRAM compatible (2B model)
- GPU memory optimization

Author: Animation AI Studio
Date: 2025-11-20
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

import torch
from pathlib import Path
from PIL import Image
from diffusers import LTXVideoTransformer3DModel, LTXVideoPipeline
from diffusers.utils import export_to_video
import argparse
from typing import Optional
import numpy as np


class LTXVideoGenerator:
    """LTX-Video image-to-video generation"""

    def __init__(
        self,
        model_path: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/video/ltx-video",
        model_variant: str = "2b-v0.9.5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_optimizations: bool = True
    ):
        """
        Initialize LTX-Video generator

        Args:
            model_path: Path to LTX-Video model
            model_variant: Model variant to use (2b or 13b)
            device: Device to use
            dtype: Data type (float16 for memory efficiency)
            enable_optimizations: Enable memory optimizations
        """
        self.model_path = Path(model_path)
        self.model_variant = model_variant
        self.device = device
        self.dtype = dtype
        self.pipeline = None

        print(f"LTX-Video Generator")
        print(f"  Model: {model_variant}")
        print(f"  Path: {self.model_path}")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")

    def load_pipeline(self):
        """Load LTX-Video pipeline"""

        print("\nLoading LTX-Video pipeline...")

        # Determine model file
        if self.model_variant == "2b-v0.9.5":
            model_file = "ltx-video-2b-v0.9.5.safetensors"
        elif self.model_variant == "2b-v0.9":
            model_file = "ltx-video-2b-v0.9.safetensors"
        else:
            model_file = f"ltx-video-{self.model_variant}.safetensors"

        transformer = LTXVideoTransformer3DModel.from_single_file(
            str(self.model_path / model_file),
            torch_dtype=self.dtype
        )

        self.pipeline = LTXVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            transformer=transformer,
            torch_dtype=self.dtype
        )

        # Memory optimizations
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()

        print("✅ Pipeline loaded successfully")

    def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        negative_prompt: str = "blurry, low quality, distorted",
        num_frames: int = 121,  # ~5 seconds at 24fps
        width: int = 768,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None
    ):
        """
        Generate video from single image

        Args:
            image_path: Input image path
            prompt: Text prompt describing the motion
            output_path: Output video path
            negative_prompt: Negative prompt
            num_frames: Number of frames to generate
            width: Video width
            height: Video height
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            seed: Random seed
        """

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        image_path = Path(image_path)
        output_path = Path(output_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        print(f"\n{'='*80}")
        print(f"Generating Video from Image")
        print(f"{'='*80}")
        print(f"Input image: {image_path.name}")
        print(f"  Size: {image.size}")
        print(f"\nPrompt: {prompt}")
        print(f"Negative: {negative_prompt}")
        print(f"\nOutput:")
        print(f"  Frames: {num_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: ~{num_frames/24:.1f}s @ 24fps")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"{'='*80}\n")

        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate video
        print("Generating video frames...")

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        video_frames = output.frames[0]

        # Export to video
        print(f"\nExporting to video: {output_path}")

        export_to_video(
            video_frames,
            output_path,
            fps=24
        )

        print(f"✅ Video saved: {output_path}")
        print(f"   Frames: {len(video_frames)}")
        print(f"   Duration: {len(video_frames)/24:.2f}s")

        return output_path

    def batch_generate(
        self,
        input_dir: str,
        output_dir: str,
        prompts: dict,
        pattern: str = "*.png"
    ):
        """
        Batch generate videos from images

        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            prompts: Dict mapping image names to prompts
            pattern: File pattern
        """

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(input_dir.glob(pattern))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        for i, image_path in enumerate(image_files, 1):
            print(f"\n{'='*80}")
            print(f"Processing {i}/{len(image_files)}: {image_path.name}")
            print(f"{'='*80}")

            # Get prompt
            prompt = prompts.get(image_path.stem, "animate this character")

            output_path = output_dir / f"{image_path.stem}_video.mp4"

            try:
                self.generate_video(
                    image_path=str(image_path),
                    prompt=prompt,
                    output_path=str(output_path)
                )
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue

        print(f"\n✅ Batch generation complete!")

    def unload_pipeline(self):
        """Unload pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            print("Pipeline unloaded")


def main():
    """Generate videos from animation frames"""

    parser = argparse.ArgumentParser(description="LTX-Video Image-to-Video Generation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Motion description prompt")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--negative", default="blurry, low quality", help="Negative prompt")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Initialize generator
    generator = LTXVideoGenerator()
    generator.load_pipeline()

    # Generate video
    generator.generate_video(
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        seed=args.seed
    )

    generator.unload_pipeline()


if __name__ == "__main__":
    # Example usage with V3 animation frames
    if len(sys.argv) == 1:
        print("🎬 LTX-Video Generator - V3 Animation Demo")
        print("="*80)

        generator = LTXVideoGenerator()
        generator.load_pipeline()

        # Example prompts for different frames
        test_cases = [
            {
                "image": "outputs/image_generation/animation_v3/walk/walk_frame_001.png",
                "prompt": "luca paguro walking forward naturally, smooth walking animation, maintaining consistent character, pixar 3d style",
                "output": "outputs/videos/ltx_generated/walk_001_video.mp4"
            },
            {
                "image": "outputs/image_generation/animation_v3/jump/jump_frame_004.png",
                "prompt": "luca paguro jumping at peak, floating in air, happy expression, pixar 3d animation",
                "output": "outputs/videos/ltx_generated/jump_004_video.mp4"
            }
        ]

        for test in test_cases:
            if not Path(test["image"]).exists():
                print(f"⚠️  Skipping: {test['image']} not found")
                continue

            print(f"\n📹 Generating video from: {Path(test['image']).name}")

            try:
                generator.generate_video(
                    image_path=test["image"],
                    prompt=test["prompt"],
                    output_path=test["output"],
                    num_frames=97,  # ~4 seconds
                    num_inference_steps=40
                )
            except Exception as e:
                print(f"❌ Error: {e}")

        generator.unload_pipeline()

        print("\n" + "="*80)
        print("✅ Demo complete! Check outputs/videos/ltx_generated/")
        print("="*80)
    else:
        main()
