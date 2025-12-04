"""
High-Quality AnimateDiff SDXL + LoRA Test

Generate high-quality 1024x1024 animation with more frames and steps.
Uses Luca LoRA (0.6) + Pixar LoRA (1.2) for character consistency.

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


def test_animatediff_hq():
    """High-quality AnimateDiff SDXL test"""

    print("="*80)
    print("AnimateDiff SDXL + LoRA HIGH QUALITY Test")
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

    # High-quality prompts
    test_cases = [
        {
            "name": "waving",
            "prompt": "luca paguro, young boy with brown hair and green eyes, purple shirt, waving hand happily at camera, bright smile, cheerful expression, pixar 3d animation style, vibrant colors, high quality, detailed",
            "frames": 24,
            "fps": 12
        },
        {
            "name": "jumping",
            "prompt": "luca paguro, young boy with brown hair and green eyes, purple shirt, jumping excitedly in the air, arms raised, joyful expression, pixar 3d animation style, vibrant colors, high quality, detailed",
            "frames": 24,
            "fps": 12
        }
    ]

    negative = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, deformed, mutated, realistic, photo, grainy, noise"

    for i, test in enumerate(test_cases, 1):
        print("\n" + "="*80)
        print(f"Test {i}/{len(test_cases)}: {test['name'].upper()}")
        print("="*80)
        print(f"Prompt: {test['prompt']}")
        print(f"\nSettings:")
        print(f"  Frames: {test['frames']}")
        print(f"  Resolution: 1024x1024 (HIGH QUALITY)")
        print(f"  Steps: 30 (high quality)")
        print(f"  Guidance: 8.0")
        print(f"  FPS: {test['fps']}")
        print(f"  Duration: {test['frames']/test['fps']:.1f}s")
        print("="*80 + "\n")

        print(f"Generating {test['name']}... (this will take 3-5 minutes)")

        generator = torch.Generator(device="cuda").manual_seed(42 + i)

        output = pipeline(
            prompt=test["prompt"],
            negative_prompt=negative,
            num_frames=test["frames"],
            width=1024,
            height=1024,
            num_inference_steps=30,  # Higher steps for quality
            guidance_scale=8.0,  # Higher guidance for better prompt following
            generator=generator
        )

        frames = output.frames[0]

        # Export
        output_path = Path(f"outputs/videos/animatediff_hq/luca_{test['name']}_1024_hq.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting to: {output_path}")
        export_to_video(frames, output_path, fps=test['fps'])

        print(f"✅ {test['name'].capitalize()} complete!")
        print(f"   Frames: {len(frames)}")
        print(f"   Duration: {len(frames)/test['fps']:.2f}s @ {test['fps']}fps")

    print("\n" + "="*80)
    print("✅ All High-Quality Tests Complete!")
    print("="*80)
    print(f"Output directory: outputs/videos/animatediff_hq/")
    print(f"\n💡 Next steps:")
    print(f"   1. Use RIFE to interpolate to 24fps for smoother playback")
    print(f"   2. Try different actions and expressions")
    print(f"   3. Combine with ControlNet for precise pose control")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_animatediff_hq()
