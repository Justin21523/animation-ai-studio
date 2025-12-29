"""
ControlNet Pipeline for SDXL

Manages ControlNet-guided image generation with SDXL.
Supports pose, depth, canny edge, and other control types.

Architecture:
- SDXL + ControlNet models
- Multiple control types (pose, depth, canny, etc.)
- Dynamic ControlNet loading/switching
- PyTorch SDPA attention

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import time
import yaml

try:
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        EulerDiscreteScheduler
    )
    from diffusers.utils import load_image
    import cv2
except ImportError:
    raise ImportError(
        "Required packages not installed. "
        "Install with: pip install diffusers opencv-python"
    )


@dataclass
class ControlNetConfig:
    """ControlNet configuration"""
    control_type: str  # "pose", "depth", "canny", "seg", "normal"
    model_path: str
    conditioning_scale: float = 1.0
    guess_mode: bool = False


class ControlNetPipelineManager:
    """
    ControlNet Pipeline Manager for SDXL

    Features:
    - Multiple ControlNet types (pose, depth, canny, etc.)
    - Control image preprocessing
    - Adjustable conditioning scale
    - PyTorch SDPA attention (xformers FORBIDDEN)

    CRITICAL CONSTRAINTS:
    - RTX 5080 16GB VRAM limit
    - SDXL + ControlNet requires ~14-15GB VRAM
    - Only ONE heavy model at a time
    """

    DEFAULT_CONTROLNET_MODELS = {
        "pose": "diffusers/controlnet-openpose-sdxl-1.0",
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
        "seg": "diffusers/controlnet-seg-sdxl-1.0",
        "normal": "diffusers/controlnet-normal-sdxl-1.0"
    }

    def __init__(
        self,
        sdxl_model_path: str,
        base_repo_path: Optional[str] = None,
        control_type: str = "pose",
        controlnet_model_path: Optional[str] = None,
        controlnet_config_path: str = "configs/generation/controlnet_config.yaml",
        prefer_local_path: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_sdpa: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
        variant: str = "fp16"
    ):
        """
        Initialize ControlNet Pipeline Manager

        Args:
            sdxl_model_path: Path to SDXL base model
            base_repo_path: Optional local SDXL base repo directory for tokenizers/text encoders
            control_type: Control type ("pose", "canny", "depth", etc.)
            controlnet_model_path: Optional custom ControlNet model path
            device: Device to use (cuda/cpu)
            dtype: Model dtype (fp16 for VRAM efficiency)
            use_sdpa: Use PyTorch SDPA (CRITICAL: must be True)
            enable_vae_slicing: Enable VAE slicing
            enable_vae_tiling: Enable VAE tiling
            variant: Model variant (fp16/fp32)
        """
        self.sdxl_model_path = Path(sdxl_model_path)
        self.base_repo_path = Path(base_repo_path) if base_repo_path else None
        self.control_type = control_type
        self.device = device
        self.dtype = dtype
        self.use_sdpa = use_sdpa
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.variant = variant

        self.controlnet_config_path = Path(controlnet_config_path)
        self.prefer_local_path = prefer_local_path

        # Determine ControlNet model path (prefer config local_path to avoid downloads)
        self.controlnet_model_path = self._resolve_controlnet_model_path(
            control_type=control_type,
            override_path=controlnet_model_path,
        )

        self.pipeline: Optional[StableDiffusionXLControlNetPipeline] = None
        self.is_loaded = False
        self.vram_usage_gb: float = 0.0

        # Verify PyTorch SDPA requirement
        if not use_sdpa:
            raise ValueError(
                "CRITICAL: use_sdpa must be True. "
                "PyTorch 2.7.0 SDPA is REQUIRED. xformers is FORBIDDEN."
            )

    def _resolve_controlnet_model_path(self, control_type: str, override_path: Optional[str]) -> str:
        if override_path:
            return override_path

        # 1) Prefer local_path from config for offline/reproducible runs.
        if self.controlnet_config_path.exists():
            try:
                with open(self.controlnet_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

                models_cfg = cfg.get("controlnet_models") or {}
                entry = models_cfg.get(control_type) or {}
                local_path = entry.get("local_path")
                model_id = entry.get("model_id") or entry.get("model")

                if self.prefer_local_path and local_path:
                    local_path_p = Path(str(local_path))
                    if local_path_p.exists():
                        return str(local_path_p)

                if model_id:
                    return str(model_id)
            except Exception as e:
                print(f"WARNING: Failed to read ControlNet config {self.controlnet_config_path}: {e}")

        # 2) Fallback to hardcoded model IDs.
        if control_type not in self.DEFAULT_CONTROLNET_MODELS:
            raise ValueError(
                f"Unknown control_type: {control_type}. "
                f"Available: {list(self.DEFAULT_CONTROLNET_MODELS.keys())}"
            )
        return self.DEFAULT_CONTROLNET_MODELS[control_type]

    def _load_controlnet_model(self) -> ControlNetModel:
        model_path = str(self.controlnet_model_path)
        is_local_dir = False
        local_dir = Path(model_path)
        if local_dir.exists() and local_dir.is_dir():
            is_local_dir = True

        def has_variant_weights(model_dir: Path, variant: str) -> bool:
            if not variant:
                return False
            candidates = [
                model_dir / f"diffusion_pytorch_model.{variant}.safetensors",
                model_dir / f"diffusion_pytorch_model.{variant}.bin",
            ]
            return any(p.exists() for p in candidates)

        # Prefer not passing `variant` for local dirs unless the variant file exists,
        # because many local ControlNet dirs only have `diffusion_pytorch_model.safetensors`.
        variant = self.variant if (not is_local_dir or has_variant_weights(local_dir, self.variant)) else None

        try:
            return ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                variant=variant,
            )
        except Exception as e:
            if variant:
                print(f"WARNING: Failed to load ControlNet with variant='{variant}', retrying without variant: {e}")
                return ControlNetModel.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                )
            raise

    def load_pipeline(self) -> StableDiffusionXLControlNetPipeline:
        """
        Load SDXL + ControlNet pipeline

        Returns:
            Loaded StableDiffusionXLControlNetPipeline

        Raises:
            RuntimeError: If VRAM insufficient
        """
        if self.is_loaded:
            print("Pipeline already loaded")
            return self.pipeline

        print(f"Loading SDXL + ControlNet ({self.control_type}) pipeline...")
        start_time = time.time()

        # Clear VRAM
        self._clear_vram()

        # Check available VRAM
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_vram < 15.0:
                print(f"WARNING: Total VRAM {total_vram:.1f}GB may be insufficient for SDXL + ControlNet")

        try:
            # Load ControlNet model
            print(f"Loading ControlNet model: {self.controlnet_model_path}")
            controlnet = self._load_controlnet_model()

            # Load SDXL pipeline with ControlNet
            print(f"Loading SDXL base model: {self.sdxl_model_path}")
            if self.sdxl_model_path.is_file():
                # Single-file checkpoint (.safetensors/.ckpt)
                from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

                base_repo = (
                    str(self.base_repo_path)
                    if self.base_repo_path and self.base_repo_path.exists()
                    else "stabilityai/stable-diffusion-xl-base-1.0"
                )
                print(f"Loading tokenizers/text encoders from base SDXL repo: {base_repo}")

                tokenizer = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer")
                tokenizer_2 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer_2")
                text_encoder = CLIPTextModel.from_pretrained(
                    base_repo,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype,
                )
                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    base_repo,
                    subfolder="text_encoder_2",
                    torch_dtype=self.dtype,
                )

                self.pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                    str(self.sdxl_model_path),
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                )
            else:
                # HuggingFace diffusers directory
                self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    str(self.sdxl_model_path),
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    variant=self.variant,
                    use_safetensors=True
                )

            # CRITICAL: Enable attention optimizations
            self.pipeline.enable_attention_slicing()

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # VAE optimizations
            if self.enable_vae_slicing:
                self.pipeline.enable_vae_slicing()

            if self.enable_vae_tiling:
                self.pipeline.enable_vae_tiling()

            # Set scheduler
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )

            self.is_loaded = True

            # Measure VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.vram_usage_gb = torch.cuda.memory_allocated() / 1e9
                print(f"ControlNet pipeline loaded in {time.time() - start_time:.2f}s")
                print(f"VRAM usage: {self.vram_usage_gb:.2f}GB")

            return self.pipeline

        except Exception as e:
            raise RuntimeError(f"Failed to load ControlNet pipeline: {e}")

    def unload_pipeline(self):
        """Unload pipeline and free VRAM"""
        if not self.is_loaded:
            print("Pipeline not loaded, nothing to unload")
            return

        print("Unloading ControlNet pipeline...")

        del self.pipeline
        self.pipeline = None
        self.is_loaded = False

        self._clear_vram()

        if torch.cuda.is_available():
            vram_freed = self.vram_usage_gb
            self.vram_usage_gb = 0.0
            print(f"ControlNet pipeline unloaded, freed {vram_freed:.2f}GB VRAM")

    def preprocess_control_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        detect_resolution: int = 512,
        image_resolution: int = 1024
    ) -> Image.Image:
        """
        Preprocess control image based on control type

        Args:
            image: Input image (path, PIL.Image, or numpy array)
            detect_resolution: Detection resolution
            image_resolution: Output image resolution

        Returns:
            Preprocessed PIL.Image
        """
        # Load image
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize to detection resolution
        image = image.resize((detect_resolution, detect_resolution))

        # Convert to numpy
        image_np = np.array(image)

        # Preprocess based on control type
        if self.control_type == "canny":
            # Canny edge detection
            image_np = cv2.Canny(image_np, 100, 200)
            image_np = np.stack([image_np] * 3, axis=-1)

        elif self.control_type == "pose":
            # For pose, use external pose detector (not included here)
            # Placeholder: return original image
            print("WARNING: Pose detection not implemented. Use preprocessed pose image.")

        elif self.control_type == "depth":
            # For depth, use external depth estimator (not included here)
            # Placeholder: return original image
            print("WARNING: Depth estimation not implemented. Use preprocessed depth map.")

        # Convert back to PIL and resize to target resolution
        control_image = Image.fromarray(image_np).resize(
            (image_resolution, image_resolution)
        )

        return control_image

    def generate(
        self,
        prompt: str,
        control_image: Union[str, Image.Image, np.ndarray],
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        guess_mode: bool = False,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        preprocess_control: bool = True,
        output_path: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate image with ControlNet guidance

        Args:
            prompt: Text prompt
            control_image: Control image (pose, depth, canny, etc.)
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet influence (0.0-2.0)
            guess_mode: Enable guess mode (ControlNet generates without prompt)
            num_images_per_prompt: Number of images to generate
            seed: Random seed
            preprocess_control: Whether to preprocess control image
            output_path: Optional path to save image(s)

        Returns:
            Single PIL.Image or list of PIL.Images
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        # Preprocess control image if needed
        if preprocess_control:
            control_image = self.preprocess_control_image(
                control_image,
                image_resolution=width
            )
        elif isinstance(control_image, str):
            control_image = load_image(control_image)

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")

        print(f"Generating {num_images_per_prompt} image(s) with {self.control_type} control...")
        print(f"Prompt: {prompt}")
        print(f"ControlNet scale: {controlnet_conditioning_scale:.2f}")

        start_time = time.time()

        output = self.pipeline(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        )

        images = output.images

        generation_time = time.time() - start_time
        print(f"Generated {len(images)} image(s) in {generation_time:.2f}s")

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

        return images[0] if len(images) == 1 else images

    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage (GB)"""
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

    # Example: Generate with ControlNet (Canny)
    sdxl_model_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/stable-diffusion-xl-base-1.0"

    # Initialize ControlNet pipeline
    controlnet_manager = ControlNetPipelineManager(
        sdxl_model_path=sdxl_model_path,
        control_type="canny",
        device="cuda",
        dtype=torch.float16,
        use_sdpa=True
    )

    # Load pipeline
    controlnet_manager.load_pipeline()

    # Generate with canny edge control
    control_image_path = "path/to/reference_image.jpg"

    image = controlnet_manager.generate(
        prompt="a boy with brown hair running on the beach, pixar style, 3d animation, high quality",
        control_image=control_image_path,
        negative_prompt="blurry, low quality",
        controlnet_conditioning_scale=0.8,
        num_inference_steps=30,
        seed=42,
        preprocess_control=True,
        output_path="outputs/controlnet_generation.png"
    )

    # Unload when done
    controlnet_manager.unload_pipeline()


if __name__ == "__main__":
    main()
