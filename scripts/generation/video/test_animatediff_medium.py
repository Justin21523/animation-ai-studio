"""
Medium-Quality AnimateDiff SDXL + LoRA Test

Generate medium-quality 1024x1024 animation with balanced speed and quality.
Optimized for faster generation while maintaining good visual quality.

Author: Animation AI Studio
Date: 2025-11-21
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

import torch
from pathlib import Path
from diffusers import (
    StableDiffusionXLPipeline,
    MotionAdapter,
    AnimateDiffSDXLPipeline,
    DDIMScheduler
)
from diffusers.utils import export_to_video


def test_animatediff_medium():
    """Medium-quality AnimateDiff SDXL test - balanced speed/quality"""

    print("="*80)
    print("AnimateDiff SDXL + LoRA MEDIUM QUALITY Test")
    print("="*80)

    # Paths
    base_model = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
    luca_lora = "/mnt/data/ai_data/models/lora/luca/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors"
    pixar_lora = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/PixarXL.safetensors"
    adapter_path = "guoyww/animatediff-motion-adapter-sdxl-beta"

    print("\n1. Loading motion adapter...")
    motion_adapter = MotionAdapter.from_pretrained(
        adapter_path,
        torch_dtype=torch.float16
    )
    print("   ✅ Motion adapter loaded")

    print("\n2. Loading SDXL base model...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    print("   ✅ SDXL loaded")

    print("\n3. Creating AnimateDiff pipeline...")
    pipeline = AnimateDiffSDXLPipeline(
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
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    pipeline.enable_model_cpu_offload()

    # Set scheduler
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1
    )
    print("   ✅ Pipeline created")

    print("\n4. Loading LoRAs...")
    pipeline.load_lora_weights(luca_lora, adapter_name="luca")
    pipeline.load_lora_weights(pixar_lora, adapter_name="pixar")
    print("   ✅ LoRAs loaded (Luca + Pixar)")

    print("\n5. Setting LoRA weights...")
    pipeline.set_adapters(["luca", "pixar"], [0.6, 1.2])
    print("   Luca: 0.6, Pixar: 1.2")

    # Medium-quality test cases (balanced)
    test_cases = [
        {
            "name": "waving",
            "prompt": "luca paguro, young boy with brown hair and green eyes, purple shirt, waving hand happily at camera, bright smile, cheerful expression, pixar 3d animation style, vibrant colors, high quality",
            "frames": 16,
            "fps": 12
        },
        {
            "name": "jumping",
            "prompt": "luca paguro, young boy with brown hair and green eyes, purple shirt, jumping excitedly in the air, arms raised, joyful expression, pixar 3d animation style, vibrant colors, high quality",
            "frames": 16,
            "fps": 12
        },
        {
            "name": "turning",
            "prompt": "luca paguro, young boy with brown hair and green eyes, purple shirt, turning around happily, full body, pixar 3d animation style, vibrant colors, high quality",
            "frames": 16,
            "fps": 12
        }
    ]

    negative = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, deformed, mutated, realistic, photo"

    for i, test in enumerate(test_cases, 1):
        print("\n" + "="*80)
        print(f"Test {i}/{len(test_cases)}: {test['name'].upper()}")
        print("="*80)
        print(f"Prompt: {test['prompt']}")
        print(f"\nSettings:")
        print(f"  Frames: {test['frames']}")
        print(f"  Resolution: 1024x1024")
        print(f"  Steps: 20 (medium quality - faster)")
        print(f"  Guidance: 7.5")
        print(f"  FPS: {test['fps']}")
        print(f"  Duration: {test['frames']/test['fps']:.1f}s")
        print(f"  Estimated time: ~2-3 minutes")
        print("="*80 + "\n")

        print(f"Generating {test['name']}...")

        generator = torch.Generator(device="cuda").manual_seed(42 + i)

        output = pipeline(
            prompt=test["prompt"],
            negative_prompt=negative,
            num_frames=test["frames"],
            width=1024,
            height=1024,
            num_inference_steps=20,  # Reduced from 30 for speed
            guidance_scale=7.5,  # Slightly lower for stability
            generator=generator
        )

        frames = output.frames[0]

        # Export
        output_path = Path(f"outputs/videos/animatediff_medium/luca_{test['name']}_1024_medium.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting to: {output_path}")
        export_to_video(frames, output_path, fps=test['fps'])

        print(f"✅ {test['name'].capitalize()} complete!")
        print(f"   Frames: {len(frames)}")
        print(f"   Duration: {len(frames)/test['fps']:.2f}s @ {test['fps']}fps")
        print(f"   File: {output_path}")

    print("\n" + "="*80)
    print("✅ All Medium-Quality Tests Complete!")
    print("="*80)
    print(f"Output directory: outputs/videos/animatediff_medium/")
    print(f"\nGenerated 3 animations:")
    print(f"  - Waving (1024x1024, 16 frames)")
    print(f"  - Jumping (1024x1024, 16 frames)")
    print(f"  - Turning (1024x1024, 16 frames)")
    print(f"\n💡 Next steps:")
    print(f"   1. Use RIFE to interpolate to 24fps for smoother playback")
    print(f"   2. Combine with ControlNet for precise pose control")
    print(f"   3. Try different actions and expressions")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_animatediff_medium()
