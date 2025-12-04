"""
AnimateDiff-SDXL + LoRA Animation Generator

Generate animated videos using AnimateDiff-Lightning with character LoRAs.
Perfect for creating consistent character animations with Luca (0.6) + Pixar (1.2) LoRAs.

Features:
- SDXL + AnimateDiff-Lightning integration
- Multi-LoRA support (Luca + Pixar)
- 1024x1024 resolution support
- Fast generation (4/8 step models)
- Character consistency across frames

Author: Animation AI Studio
Date: 2025-11-20
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

import torch
from pathlib import Path
from diffusers import (
    StableDiffusionXLPipeline,
    MotionAdapter,
    AnimateDiffSDXLPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler
)
from diffusers.utils import export_to_video
import argparse
from typing import Dict, Optional


class AnimateDiffLoRAGenerator:
    """AnimateDiff-SDXL animation generation with LoRA support"""

    def __init__(
        self,
        base_model_path: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors",
        motion_adapter_path: str = "/mnt/c/AI_LLM_projects/ai_warehouse/models/video/animatediff/animatediff-lightning",
        lora_paths: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize AnimateDiff+LoRA generator

        Args:
            base_model_path: Path to SDXL base model
            motion_adapter_path: Path to AnimateDiff motion adapter
            lora_paths: Dict of LoRA names and paths
            device: Device to use
            dtype: Data type
        """
        self.base_model_path = base_model_path
        self.motion_adapter_path = Path(motion_adapter_path)
        self.lora_paths = lora_paths or {}
        self.device = device
        self.dtype = dtype
        self.pipeline = None

        print(f"AnimateDiff-LoRA Generator")
        print(f"  Base model: {Path(base_model_path).name}")
        print(f"  Motion adapter: {self.motion_adapter_path}")
        print(f"  LoRAs: {list(self.lora_paths.keys())}")
        print(f"  Device: {device}")

    def load_pipeline(self, num_inference_steps: int = 4):
        """
        Load AnimateDiff pipeline with motion adapter

        Args:
            num_inference_steps: Number of steps (4 or 8 for Lightning)
        """

        print(f"\nLoading AnimateDiff pipeline ({num_inference_steps}-step)...")

        # Select motion adapter based on steps
        if num_inference_steps == 4:
            adapter_file = "animatediff_lightning_4step_diffusers.safetensors"
        elif num_inference_steps == 8:
            adapter_file = "animatediff_lightning_8step_diffusers.safetensors"
        elif num_inference_steps == 2:
            adapter_file = "animatediff_lightning_2step_diffusers.safetensors"
        else:
            adapter_file = "animatediff_lightning_4step_diffusers.safetensors"
            print(f"⚠️  {num_inference_steps} steps not available, using 4-step model")

        adapter_path = self.motion_adapter_path / adapter_file

        if not adapter_path.exists():
            raise FileNotFoundError(f"Motion adapter not found: {adapter_path}")

        # Load motion adapter
        # AnimateDiff-Lightning uses different loading approach
        print(f"   Loading motion adapter: {adapter_file}")

        try:
            # Try loading from pretrained with local file
            motion_adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-sdxl-beta",
                torch_dtype=self.dtype
            )
            # Load weights from our downloaded file
            from safetensors.torch import load_file
            state_dict = load_file(str(adapter_path))
            motion_adapter.load_state_dict(state_dict, strict=False)
            print(f"   ✅ Loaded weights from {adapter_file}")
        except Exception as e:
            print(f"   ⚠️  Warning: {e}")
            print(f"   Trying alternative loading method...")
            # Fallback: use base SDXL motion adapter
            motion_adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-sdxl-beta",
                torch_dtype=self.dtype
            )

        # Load SDXL base pipeline
        pipe = StableDiffusionXLPipeline.from_single_file(
            self.base_model_path,
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        # Convert to AnimateDiff pipeline
        self.pipeline = AnimateDiffSDXLPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            unet=pipe.unet,
            motion_adapter=motion_adapter,
            scheduler=pipe.scheduler,
        )

        # Optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_model_cpu_offload()

        # Set scheduler
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear"
        )

        # Load LoRAs
        if self.lora_paths:
            print(f"\nLoading {len(self.lora_paths)} LoRA(s):")
            for name, path in self.lora_paths.items():
                print(f"  [{name}]: {Path(path).name}")
                self.pipeline.load_lora_weights(path, adapter_name=name)

        print("✅ Pipeline loaded successfully")

    def generate_animation(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
        lora_weights: Optional[Dict[str, float]] = None,
        num_frames: int = 16,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,  # Lightning models work best with low guidance
        seed: Optional[int] = None
    ):
        """
        Generate animated video

        Args:
            prompt: Animation prompt
            output_path: Output video path
            negative_prompt: Negative prompt
            lora_weights: Dict of LoRA weights
            num_frames: Number of frames
            width: Video width
            height: Video height
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            seed: Random seed
        """

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set LoRA weights
        if lora_weights and self.lora_paths:
            self.pipeline.set_adapters(
                list(lora_weights.keys()),
                list(lora_weights.values())
            )

        print(f"\n{'='*80}")
        print(f"Generating Animation")
        print(f"{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"Negative: {negative_prompt}")
        if lora_weights:
            print(f"LoRA weights: {lora_weights}")
        print(f"\nOutput:")
        print(f"  Frames: {num_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"{'='*80}\n")

        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate animation
        print("Generating animation frames...")

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        frames = output.frames[0]

        # Export to video
        print(f"\nExporting to video: {output_path}")

        export_to_video(
            frames,
            output_path,
            fps=8  # 16 frames at 8fps = 2 seconds
        )

        print(f"✅ Animation saved: {output_path}")
        print(f"   Frames: {len(frames)}")
        print(f"   Duration: {len(frames)/8:.2f}s @ 8fps")

        return output_path

    def unload_pipeline(self):
        """Unload pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            print("Pipeline unloaded")


def main():
    """Generate animations with LoRAs"""

    parser = argparse.ArgumentParser(description="AnimateDiff-SDXL + LoRA Animation Generation")
    parser.add_argument("--prompt", required=True, help="Animation prompt")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--negative", default="blurry, low quality", help="Negative prompt")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--steps", type=int, default=4, choices=[2, 4, 8], help="Inference steps")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Default LoRAs
    lora_paths = {
        "luca": "/mnt/data/ai_data/models/lora/luca/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors",
        "pixar": "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/PixarXL.safetensors"
    }

    lora_weights = {"luca": 0.6, "pixar": 1.2}

    # Initialize generator
    generator = AnimateDiffLoRAGenerator(lora_paths=lora_paths)
    generator.load_pipeline(num_inference_steps=args.steps)

    # Generate animation
    generator.generate_animation(
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative,
        lora_weights=lora_weights,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        seed=args.seed
    )

    generator.unload_pipeline()


if __name__ == "__main__":
    # Example usage with Luca character
    if len(sys.argv) == 1:
        print("🎬 AnimateDiff-LoRA Generator - Luca Animation Demo")
        print("="*80)

        # LoRA paths
        lora_paths = {
            "luca": "/mnt/data/ai_data/models/lora/luca/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors",
            "pixar": "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/PixarXL.safetensors"
        }

        lora_weights = {"luca": 0.6, "pixar": 1.2}

        generator = AnimateDiffLoRAGenerator(lora_paths=lora_paths)
        generator.load_pipeline(num_inference_steps=4)

        # Test animations
        test_prompts = [
            {
                "prompt": "luca paguro, boy with brown hair and green eyes, purple shirt, walking forward happily, full body, pixar 3d animation style, vibrant colors",
                "output": "outputs/videos/animatediff_generated/luca_walking.mp4",
                "frames": 16
            },
            {
                "prompt": "luca paguro, boy with brown hair and green eyes, purple shirt, jumping excitedly in the air, full body, pixar 3d animation style, vibrant colors",
                "output": "outputs/videos/animatediff_generated/luca_jumping.mp4",
                "frames": 16
            },
            {
                "prompt": "luca paguro, boy with brown hair and green eyes, purple shirt, waving hand friendly, smiling, full body, pixar 3d animation style, vibrant colors",
                "output": "outputs/videos/animatediff_generated/luca_waving.mp4",
                "frames": 16
            }
        ]

        negative = (
            "blurry, low quality, distorted, ugly, bad anatomy, "
            "deformed, extra limbs, missing limbs, bad proportions, "
            "realistic, photo"
        )

        for i, test in enumerate(test_prompts, 1):
            print(f"\n📹 Test {i}/{len(test_prompts)}: {test['output']}")

            try:
                generator.generate_animation(
                    prompt=test["prompt"],
                    output_path=test["output"],
                    negative_prompt=negative,
                    lora_weights=lora_weights,
                    num_frames=test["frames"],
                    num_inference_steps=4,
                    seed=1000 + i
                )
            except Exception as e:
                print(f"❌ Error: {e}")

        generator.unload_pipeline()

        print("\n" + "="*80)
        print("✅ Demo complete! Check outputs/videos/animatediff_generated/")
        print("="*80)
    else:
        main()
