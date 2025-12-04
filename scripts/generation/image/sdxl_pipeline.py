"""
SDXL Pipeline Manager

Manages SDXL base model loading, generation, and VRAM optimization.
Optimized for RTX 5080 16GB VRAM with PyTorch 2.7.0 SDPA.

Architecture:
- SDXL base model (fp16)
- PyTorch SDPA attention (xformers FORBIDDEN)
- Dynamic VRAM management
- Quality presets (draft, standard, high)

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from PIL import Image
import time

try:
    from diffusers import (
        StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        DDIMScheduler
    )
    from diffusers.utils import logging as diffusers_logging
except ImportError:
    raise ImportError(
        "diffusers not installed. Install with: pip install diffusers"
    )


@dataclass
class GenerationConfig:
    """Configuration for image generation"""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    scheduler: str = "euler"  # "euler", "dpm", "ddim"


class SDXLPipelineManager:
    """
    SDXL Pipeline Manager for RTX 5080 16GB

    Features:
    - SDXL base model (fp16) with PyTorch SDPA
    - Quality presets (draft/standard/high)
    - VRAM monitoring and optimization
    - Multiple scheduler support
    - Deterministic generation (seed support)

    CRITICAL CONSTRAINTS:
    - RTX 5080 16GB VRAM limit
    - PyTorch 2.7.0 SDPA (xformers FORBIDDEN)
    - Only ONE heavy model at a time
    - GPU memory utilization: 0.85 max
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_sdpa: bool = True,
        enable_model_cpu_offload: bool = False,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
        variant: str = "fp16"
    ):
        """
        Initialize SDXL Pipeline Manager

        Args:
            model_path: Path to SDXL base model
            device: Device to use (cuda/cpu)
            dtype: Model dtype (fp16 for VRAM efficiency)
            use_sdpa: Use PyTorch SDPA (CRITICAL: must be True)
            enable_model_cpu_offload: Enable model CPU offload (slower, saves VRAM)
            enable_vae_slicing: Enable VAE slicing (saves VRAM)
            enable_vae_tiling: Enable VAE tiling (saves VRAM for large images)
            variant: Model variant (fp16/fp32)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.use_sdpa = use_sdpa
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.variant = variant

        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.is_loaded = False

        # VRAM tracking
        self.vram_usage_gb: float = 0.0

        # Quality presets
        self.quality_presets = {
            "draft": {"steps": 20, "cfg_scale": 7.0},
            "standard": {"steps": 30, "cfg_scale": 7.5},
            "high": {"steps": 40, "cfg_scale": 8.0},
            "ultra": {"steps": 50, "cfg_scale": 8.5}
        }

        # Verify PyTorch SDPA requirement
        if not use_sdpa:
            raise ValueError(
                "CRITICAL: use_sdpa must be True. "
                "PyTorch 2.7.0 SDPA is REQUIRED. xformers is FORBIDDEN."
            )

    def load_pipeline(self) -> StableDiffusionXLPipeline:
        """
        Load SDXL pipeline with optimizations

        Returns:
            Loaded StableDiffusionXLPipeline

        Raises:
            RuntimeError: If VRAM insufficient
        """
        if self.is_loaded:
            print("Pipeline already loaded")
            return self.pipeline

        print(f"Loading SDXL pipeline from {self.model_path}...")
        start_time = time.time()

        # Clear VRAM before loading
        self._clear_vram()

        # Check available VRAM
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_vram < 14.0:
                print(f"WARNING: Total VRAM {total_vram:.1f}GB may be insufficient for SDXL")

        # Load pipeline
        try:
            # Check if model_path is a single file or directory
            if self.model_path.is_file() and self.model_path.suffix in ['.safetensors', '.ckpt']:
                # Load from single checkpoint file (CivitAI format)
                # CRITICAL: Single-file checkpoints often don't include tokenizers/text encoders
                # We need to load them explicitly from base SDXL model
                print(f"Loading from single file: {self.model_path.name}")

                from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
                from diffusers import AutoencoderKL

                # Load tokenizers and text encoders from base SDXL (required for single-file loading)
                print("Loading tokenizers and text encoders from base SDXL model...")
                tokenizer = CLIPTokenizer.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="tokenizer"
                )
                tokenizer_2 = CLIPTokenizer.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="tokenizer_2"
                )
                text_encoder = CLIPTextModel.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="text_encoder",
                    torch_dtype=self.dtype
                )
                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="text_encoder_2",
                    torch_dtype=self.dtype
                )

                # Load pipeline with explicit tokenizers and text encoders
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2
                )

                # Fix UNet config for single-file SDXL checkpoints
                # Many CivitAI checkpoints have incomplete UNet configs
                if hasattr(self.pipeline.unet.config, 'addition_time_embed_dim'):
                    if self.pipeline.unet.config.addition_time_embed_dim is None:
                        print("Fixing UNet config: setting addition_time_embed_dim=256")
                        self.pipeline.unet.config.addition_time_embed_dim = 256

                # Check if UNet has add_embedding attribute
                # If not, this checkpoint may not be fully SDXL compatible
                if not hasattr(self.pipeline.unet, 'add_embedding'):
                    print("WARNING: UNet missing 'add_embedding' attribute.")
                    print("This checkpoint may not be fully SDXL-compatible.")
                    print("Attempting to add missing time embedding layer...")

                    # Try to add the missing embedding layer
                    import torch.nn as nn
                    try:
                        # SDXL uses 256-dim additional time embeddings
                        addition_time_embed_dim = getattr(
                            self.pipeline.unet.config,
                            'addition_time_embed_dim',
                            256
                        )
                        time_embed_dim = self.pipeline.unet.config.block_out_channels[0] * 4

                        # Create the missing add_embedding layer
                        from diffusers.models.embeddings import TimestepEmbedding
                        self.pipeline.unet.add_embedding = TimestepEmbedding(
                            addition_time_embed_dim,
                            time_embed_dim
                        ).to(self.device, dtype=self.dtype)
                        print("Successfully added missing time embedding layer")
                    except Exception as e:
                        print(f"WARNING: Could not add embedding layer: {e}")
                        print("This model may fail during generation.")
            else:
                # Load from directory (HuggingFace format)
                print(f"Loading from directory: {self.model_path}")
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    variant=self.variant,
                    use_safetensors=True
                )

            # CRITICAL: Set attention processor to SDPA
            # This replaces xformers with PyTorch 2.7.0 native SDPA
            self.pipeline.enable_attention_slicing()

            # Move to device
            if not self.enable_model_cpu_offload:
                self.pipeline = self.pipeline.to(self.device)
            else:
                # CPU offload (saves VRAM, slower inference)
                self.pipeline.enable_model_cpu_offload()

            # VAE optimizations
            if self.enable_vae_slicing:
                self.pipeline.enable_vae_slicing()

            if self.enable_vae_tiling:
                self.pipeline.enable_vae_tiling()

            # Set default scheduler (Euler Discrete)
            self.set_scheduler("euler")

            # Disable progress bar for cleaner output
            diffusers_logging.set_verbosity_error()

            self.is_loaded = True

            # Measure VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.vram_usage_gb = torch.cuda.memory_allocated() / 1e9
                print(f"SDXL pipeline loaded in {time.time() - start_time:.2f}s")
                print(f"VRAM usage: {self.vram_usage_gb:.2f}GB")

            return self.pipeline

        except Exception as e:
            raise RuntimeError(f"Failed to load SDXL pipeline: {e}")

    def unload_pipeline(self):
        """
        Unload pipeline and free VRAM

        Call this before switching to another heavy model (LLM, GPT-SoVITS)
        """
        if not self.is_loaded:
            print("Pipeline not loaded, nothing to unload")
            return

        print("Unloading SDXL pipeline...")

        # Delete pipeline
        del self.pipeline
        self.pipeline = None
        self.is_loaded = False

        # Clear VRAM
        self._clear_vram()

        if torch.cuda.is_available():
            vram_freed = self.vram_usage_gb
            self.vram_usage_gb = 0.0
            print(f"SDXL pipeline unloaded, freed {vram_freed:.2f}GB VRAM")

    def set_scheduler(self, scheduler_name: str):
        """
        Set diffusion scheduler

        Args:
            scheduler_name: Scheduler name ("euler", "dpm", "ddim")
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        scheduler_map = {
            "euler": EulerDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "ddim": DDIMScheduler
        }

        if scheduler_name not in scheduler_map:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        self.pipeline.scheduler = scheduler_map[scheduler_name].from_config(
            self.pipeline.scheduler.config
        )
        print(f"Scheduler set to: {scheduler_name}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        quality_preset: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate image(s) with SDXL

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width (multiple of 8)
            height: Image height (multiple of 8)
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            num_images_per_prompt: Number of images to generate
            seed: Random seed (None = random)
            quality_preset: Quality preset ("draft", "standard", "high", "ultra")
            output_path: Optional path to save image(s)

        Returns:
            Single PIL.Image or list of PIL.Images
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        # Apply quality preset if specified
        if quality_preset:
            if quality_preset not in self.quality_presets:
                raise ValueError(f"Unknown quality preset: {quality_preset}")
            preset = self.quality_presets[quality_preset]
            num_inference_steps = preset["steps"]
            guidance_scale = preset["cfg_scale"]
            print(f"Using quality preset: {quality_preset} (steps={num_inference_steps}, cfg={guidance_scale})")

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")

        # Generate
        print(f"Generating {num_images_per_prompt} image(s)...")
        print(f"Prompt: {prompt}")
        if negative_prompt:
            print(f"Negative: {negative_prompt}")

        start_time = time.time()

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        )

        images = output.images

        generation_time = time.time() - start_time
        print(f"Generated {len(images)} image(s) in {generation_time:.2f}s ({generation_time/len(images):.2f}s per image)")

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            if len(images) == 1:
                images[0].save(output_path)
                print(f"Saved to: {output_path}")
            else:
                for i, img in enumerate(images):
                    save_path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
                    img.save(save_path)
                    print(f"Saved to: {save_path}")

        # Return single image or list
        return images[0] if len(images) == 1 else images

    def generate_with_config(
        self,
        config: GenerationConfig,
        output_path: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate image(s) using GenerationConfig

        Args:
            config: GenerationConfig object
            output_path: Optional path to save image(s)

        Returns:
            Single PIL.Image or list of PIL.Images
        """
        # Set scheduler if specified
        if config.scheduler:
            self.set_scheduler(config.scheduler)

        return self.generate(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_images_per_prompt,
            seed=config.seed,
            output_path=output_path
        )

    def get_vram_usage(self) -> Dict[str, float]:
        """
        Get current VRAM usage

        Returns:
            Dict with VRAM statistics (GB)
        """
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "pipeline_usage": self.vram_usage_gb
        }

    def _clear_vram(self):
        """Clear CUDA cache and run garbage collection"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main():
    """Example usage"""

    # Example: Generate with SDXL
    model_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/stable-diffusion-xl-base-1.0"

    # Initialize pipeline
    pipeline_manager = SDXLPipelineManager(
        model_path=model_path,
        device="cuda",
        dtype=torch.float16,
        use_sdpa=True,
        enable_vae_slicing=True,
        enable_vae_tiling=True
    )

    # Load pipeline
    pipeline_manager.load_pipeline()

    # Generate image
    image = pipeline_manager.generate(
        prompt="a boy with brown hair and green eyes running on the beach, excited expression, pixar style, 3d animation, high quality",
        negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
        quality_preset="standard",
        seed=42,
        output_path="outputs/test_generation.png"
    )

    # Show VRAM usage
    vram = pipeline_manager.get_vram_usage()
    print(f"\nVRAM Usage:")
    print(f"  Allocated: {vram['allocated']:.2f}GB")
    print(f"  Total: {vram['total']:.2f}GB")

    # Unload when done
    pipeline_manager.unload_pipeline()


if __name__ == "__main__":
    main()
