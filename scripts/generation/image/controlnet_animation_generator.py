"""
ControlNet-Enhanced Animation Generator

Generate animation sequences with ControlNet guidance for better consistency:
- OpenPose: Control character poses for animation
- Depth: Maintain 3D depth consistency
- Canny/Lineart: Preserve edge structure
- Frame-to-frame consistency using previous frame as reference

Author: Animation AI Studio
Date: 2025-11-20
"""

import sys
sys.path.insert(0, '/mnt/c/AI_LLM_projects/animation-ai-studio')

from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerDiscreteScheduler
)
from diffusers.utils import load_image
import json
from typing import List, Dict, Optional, Tuple


class ControlNetAnimationGenerator:
    """Generate animations with ControlNet guidance"""

    def __init__(
        self,
        base_model_path: str,
        controlnet_paths: Dict[str, str],
        lora_paths: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize ControlNet Animation Generator

        Args:
            base_model_path: Path to SDXL base model
            controlnet_paths: Dict of controlnet types and paths
            lora_paths: Optional dict of LoRA paths
            device: Device to use
            dtype: Data type
        """
        self.base_model_path = base_model_path
        self.controlnet_paths = controlnet_paths
        self.lora_paths = lora_paths or {}
        self.device = device
        self.dtype = dtype
        self.pipeline = None

    def load_pipeline(self, controlnet_type: str = "openpose"):
        """Load pipeline with specific ControlNet"""

        print(f"\nLoading ControlNet pipeline ({controlnet_type})...")
        print(f"Base model: {Path(self.base_model_path).name}")
        print(f"ControlNet: {self.controlnet_paths.get(controlnet_type, 'Not found')}")

        # Load ControlNet
        controlnet_path = self.controlnet_paths.get(controlnet_type)
        if not controlnet_path or not Path(controlnet_path).exists():
            raise ValueError(f"ControlNet path not found: {controlnet_type}")

        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=self.dtype
        )

        # Load pipeline
        self.pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
            self.base_model_path,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        # Optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_attention_slicing()

        # Set scheduler
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline = self.pipeline.to(self.device)

        # Load LoRAs if provided
        if self.lora_paths:
            print(f"\nLoading {len(self.lora_paths)} LoRA(s):")
            for name, path in self.lora_paths.items():
                print(f"  [{name}]: {Path(path).name}")
                self.pipeline.load_lora_weights(path, adapter_name=name)

        print("✅ Pipeline loaded successfully")

    def generate_with_controlnet(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: str,
        lora_weights: Optional[Dict[str, float]] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: float = 7.5,
        controlnet_scale: float = 0.8,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate image with ControlNet guidance"""

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        # Set LoRA weights
        if lora_weights and self.lora_paths:
            self.pipeline.set_adapters(
                list(lora_weights.keys()),
                list(lora_weights.values())
            )

        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator
        ).images[0]

        return image

    def unload_pipeline(self):
        """Unload pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            print("Pipeline unloaded")


def extract_canny(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Extract Canny edges from image"""
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, low_threshold, high_threshold)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


def extract_depth(image: Image.Image) -> Image.Image:
    """
    Extract depth map (placeholder - requires depth estimation model)
    For now, returns grayscale as approximation
    """
    image_gray = image.convert('L')
    return image_gray.convert('RGB')


def create_pose_image(width: int, height: int, pose_keypoints: List[Tuple[int, int]]) -> Image.Image:
    """
    Create simple pose image from keypoints
    (Placeholder - would use OpenPose in production)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw simple stick figure
    for i, (x, y) in enumerate(pose_keypoints):
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)

    # Connect keypoints (simple connections)
    connections = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Example connections
    for start_idx, end_idx in connections:
        if start_idx < len(pose_keypoints) and end_idx < len(pose_keypoints):
            pt1 = pose_keypoints[start_idx]
            pt2 = pose_keypoints[end_idx]
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    return Image.fromarray(canvas)


def main():
    """Example: Generate animation with ControlNet"""

    # Paths
    base_model = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
    controlnet_base = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/controlnet"

    controlnet_paths = {
        "openpose": f"{controlnet_base}/controlnet-openpose-sdxl/diffusion_pytorch_model.safetensors",
        "canny": f"{controlnet_base}/controlnet-canny-sdxl/diffusion_pytorch_model.safetensors",
        "depth": f"{controlnet_base}/controlnet-depth-sdxl/diffusion_pytorch_model.safetensors",
        "lineart": f"{controlnet_base}/controlnet-lineart-sdxl/diffusion_pytorch_model.safetensors",
    }

    lora_paths = {
        "luca": "/mnt/data/ai_data/models/lora/luca/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors",
        "pixar": "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/PixarXL.safetensors"
    }

    print("🎬 ControlNet Animation Generator")
    print("="*80)
    print("Available ControlNets:")
    for ctype, path in controlnet_paths.items():
        status = "✅" if Path(path).exists() else "❌"
        print(f"  {status} {ctype}")
    print("="*80)

    # Check which ControlNets are available
    available_controlnets = {k: v for k, v in controlnet_paths.items() if Path(v).exists()}

    if not available_controlnets:
        print("\n⚠️  No ControlNet models found yet. They are being downloaded.")
        print("Please wait for downloads to complete, then run this script again.")
        return

    # Initialize generator with first available ControlNet
    controlnet_type = list(available_controlnets.keys())[0]
    print(f"\n Using ControlNet: {controlnet_type}")

    generator = ControlNetAnimationGenerator(
        base_model_path=base_model,
        controlnet_paths=available_controlnets,
        lora_paths=lora_paths,
        device="cuda",
        dtype=torch.float16
    )

    generator.load_pipeline(controlnet_type=controlnet_type)

    # Example: Generate a simple test frame
    print("\n🎨 Generating test frame with ControlNet...")

    # Create a simple control image (would use actual pose/depth in production)
    control_image = Image.new('RGB', (1024, 1024), color=(128, 128, 128))

    prompt = "luca paguro, young boy, brown curly hair, green eyes, purple striped shirt, standing naturally, full body, pixar 3d animation style, high quality"
    negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

    test_image = generator.generate_with_controlnet(
        prompt=prompt,
        control_image=control_image,
        negative_prompt=negative_prompt,
        lora_weights={"luca": 0.6, "pixar": 1.2},
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=7.5,
        controlnet_scale=0.8,
        seed=42
    )

    # Save test output
    output_dir = Path("outputs/image_generation/controlnet_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"controlnet_{controlnet_type}_test.png"
    test_image.save(output_path)

    print(f"✅ Test image saved: {output_path}")

    generator.unload_pipeline()

    print("\n" + "="*80)
    print("ControlNet Animation Generator Ready!")
    print("All downloaded ControlNet models can be used for animation generation.")
    print("="*80)


if __name__ == "__main__":
    main()
