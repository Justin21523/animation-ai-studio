"""
Simple AnimateDiff + LoRA Test

Quick test with just 1 short animation to verify the pipeline works.

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
    EulerDiscreteScheduler
)
from diffusers.utils import export_to_video


def test_animatediff_simple():
    """Simple AnimateDiff test with Luca LoRA"""

    print("="*80)
    print("AnimateDiff + LoRA Simple Test")
    print("="*80)

    # Paths
    base_model = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
    luca_lora = "/mnt/data/ai_data/models/lora/luca/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors"
    pixar_lora = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/PixarXL.safetensors"
    adapter_file = "/mnt/c/AI_LLM_projects/ai_warehouse/models/video/animatediff/animatediff-lightning/animatediff_lightning_4step_diffusers.safetensors"

    print("\n1. Loading motion adapter...")

    # Load motion adapter (use base SDXL adapter config)
    motion_adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-sdxl-beta",
        torch_dtype=torch.float16
    )

    # Load Lightning weights
    print("   Loading AnimateDiff-Lightning weights...")
    from safetensors.torch import load_file
    state_dict = load_file(adapter_file)
    motion_adapter.load_state_dict(state_dict, strict=False)
    print("   ✅ Motion adapter loaded with Lightning weights")

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

    # Set scheduler for Lightning
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear"
    )
    print("   ✅ Pipeline created")

    print("\n4. Loading LoRAs...")
    pipeline.load_lora_weights(luca_lora, adapter_name="luca")
    pipeline.load_lora_weights(pixar_lora, adapter_name="pixar")
    print("   ✅ LoRAs loaded (Luca + Pixar)")

    print("\n5. Setting LoRA weights...")
    pipeline.set_adapters(["luca", "pixar"], [0.6, 1.2])
    print("   Luca: 0.6, Pixar: 1.2")

    # Generate
    prompt = "luca paguro, boy with brown hair and green eyes, purple shirt, waving hand happily, smiling, pixar 3d animation style, vibrant colors"
    negative = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, realistic, photo"

    print("\n" + "="*80)
    print("Generating Animation")
    print("="*80)
    print(f"Prompt: {prompt}")
    print(f"\nSettings:")
    print(f"  Frames: 16")
    print(f"  Resolution: 512x512 (reduced for speed)")
    print(f"  Steps: 4 (Lightning)")
    print(f"  Guidance: 1.0")
    print("="*80 + "\n")

    print("Generating... (this may take 2-3 minutes)")

    generator = torch.Generator(device="cuda").manual_seed(42)

    output = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        num_frames=16,
        width=512,  # Reduced for faster test
        height=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator
    )

    frames = output.frames[0]

    # Export
    output_path = Path("outputs/videos/animatediff_test/luca_waving_test.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to: {output_path}")
    export_to_video(frames, output_path, fps=8)

    print("\n" + "="*80)
    print("✅ Test Complete!")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Frames: {len(frames)}")
    print(f"Duration: {len(frames)/8:.2f}s @ 8fps")
    print("\n💡 If this works, you can:")
    print("   1. Run full demo with 3 animations (1024x1024)")
    print("   2. Adjust LoRA weights for different styles")
    print("   3. Try different prompts and actions")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_animatediff_simple()
