"""
Character Generator

High-level wrapper for character image generation with SDXL + LoRA + ControlNet.
Provides simple interface for generating consistent character images.

Architecture:
- SDXL base model
- Character LoRA adapters
- Optional ControlNet (pose, depth, etc.)
- Quality-driven generation

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from PIL import Image
import yaml
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image.sdxl_pipeline import SDXLPipelineManager, GenerationConfig
from scripts.generation.image.lora_manager import LoRAManager, LoRARegistry
from scripts.generation.image.controlnet_pipeline import ControlNetPipelineManager


@dataclass
class CharacterGenerationConfig:
    """Configuration for character image generation"""
    character: str
    scene_description: str
    style: str = "pixar_3d"
    quality_preset: str = "standard"
    use_controlnet: bool = False
    control_type: Optional[str] = None
    control_image: Optional[str] = None
    controlnet_scale: float = 0.9
    additional_loras: Optional[List[Dict[str, Any]]] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    width: int = 1024
    height: int = 1024


class CharacterGenerator:
    """
    Character Generator

    High-level interface for generating character images with SDXL + LoRA.

    Features:
    - Automatic LoRA selection based on character
    - Style prompt integration
    - Optional ControlNet for pose consistency
    - Quality presets
    - Automatic prompt engineering

    Example:
        generator = CharacterGenerator()
        image = generator.generate_character(
            character="luca",
            scene_description="running on the beach, excited expression",
            quality_preset="high",
            seed=42
        )
    """

    def __init__(
        self,
        sdxl_config_path: str = "configs/generation/sdxl_config.yaml",
        lora_registry_path: str = "configs/generation/lora_registry.yaml",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize Character Generator

        Args:
            sdxl_config_path: Path to SDXL config YAML
            lora_registry_path: Path to LoRA registry YAML
            device: Device to use (cuda/cpu)
            dtype: Model dtype
        """
        self.device = device
        self.dtype = dtype

        # Load configurations
        self.sdxl_config = self._load_config(sdxl_config_path)
        self.lora_registry = LoRARegistry(lora_registry_path)

        # Initialize managers (lazy loading)
        self.sdxl_manager: Optional[SDXLPipelineManager] = None
        self.lora_manager: Optional[LoRAManager] = None
        self.controlnet_manager: Optional[ControlNetPipelineManager] = None

        self.current_mode: Optional[str] = None  # "sdxl" or "controlnet"

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_sdxl(self):
        """Initialize SDXL pipeline (lazy)"""
        if self.sdxl_manager is not None and self.sdxl_manager.is_loaded:
            return

        print("Initializing SDXL pipeline...")

        model_config = self.sdxl_config["model"]
        vram_config = self.sdxl_config["vram"]
        device_config = self.sdxl_config["device"]

        self.sdxl_manager = SDXLPipelineManager(
            model_path=model_config["base_model"],
            device=self.device,
            dtype=self.dtype,
            use_sdpa=True,
            enable_model_cpu_offload=vram_config["enable_model_cpu_offload"],
            enable_vae_slicing=vram_config["enable_vae_slicing"],
            enable_vae_tiling=vram_config["enable_vae_tiling"],
            variant=model_config["variant"]
        )

        self.sdxl_manager.load_pipeline()

        # Initialize LoRA manager
        self.lora_manager = LoRAManager(
            pipeline=self.sdxl_manager.pipeline,
            registry=self.lora_registry
        )

        self.current_mode = "sdxl"

    def _initialize_controlnet(self, control_type: str):
        """Initialize ControlNet pipeline (lazy)"""
        if (self.controlnet_manager is not None and
            self.controlnet_manager.is_loaded and
            self.controlnet_manager.control_type == control_type):
            return

        print(f"Initializing ControlNet pipeline ({control_type})...")

        # Unload SDXL if loaded
        if self.sdxl_manager is not None and self.sdxl_manager.is_loaded:
            self.sdxl_manager.unload_pipeline()

        model_config = self.sdxl_config["model"]
        vram_config = self.sdxl_config["vram"]

        self.controlnet_manager = ControlNetPipelineManager(
            sdxl_model_path=model_config["base_model"],
            control_type=control_type,
            device=self.device,
            dtype=self.dtype,
            use_sdpa=True,
            enable_vae_slicing=vram_config["enable_vae_slicing"],
            enable_vae_tiling=vram_config["enable_vae_tiling"],
            variant=model_config["variant"]
        )

        self.controlnet_manager.load_pipeline()

        # Note: LoRA not yet supported with ControlNet in this implementation
        # Future enhancement: integrate LoRA with ControlNet pipeline

        self.current_mode = "controlnet"

    def _build_prompt(
        self,
        character: str,
        scene_description: str,
        style: str,
        include_trigger_words: bool = True
    ) -> str:
        """
        Build complete prompt from components

        Args:
            character: Character name
            scene_description: Scene description
            style: Style key
            include_trigger_words: Whether to include LoRA trigger words

        Returns:
            Complete prompt string
        """
        # Get character trigger words if using LoRA
        character_prompt = ""
        if include_trigger_words:
            lora_config = self.lora_registry.get_character_lora(character)
            if lora_config and lora_config.trigger_words:
                character_prompt = ", ".join(lora_config.trigger_words)

        # Get style prompt
        style_prompts = self.sdxl_config.get("style_prompts", {})
        style_prompt = style_prompts.get(style, "high quality, detailed")

        # Combine components
        if character_prompt:
            full_prompt = f"{character_prompt}, {scene_description}, {style_prompt}"
        else:
            full_prompt = f"{scene_description}, {style_prompt}"

        return full_prompt

    def _get_negative_prompt(self, negative_prompt_type: str = "character") -> str:
        """Get negative prompt from config"""
        negative_prompts = self.sdxl_config.get("negative_prompts", {})
        return negative_prompts.get(negative_prompt_type, negative_prompts.get("default", ""))

    def generate_character(
        self,
        character: str,
        scene_description: str,
        style: str = "pixar_3d",
        quality_preset: str = "standard",
        use_controlnet: bool = False,
        control_type: Optional[str] = None,
        control_image: Optional[str] = None,
        controlnet_scale: float = 0.9,
        additional_loras: Optional[List[Dict[str, Any]]] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generate character image

        Args:
            character: Character name (e.g., "luca", "alberto")
            scene_description: Scene description
            style: Style key ("pixar_3d", "disney_3d", etc.)
            quality_preset: Quality preset ("draft", "standard", "high", "ultra")
            use_controlnet: Whether to use ControlNet
            control_type: ControlNet type (if use_controlnet=True)
            control_image: Path to control image (if use_controlnet=True)
            controlnet_scale: ControlNet conditioning scale
            additional_loras: Additional LoRAs to load (list of {"name": str, "weight": float})
            negative_prompt: Custom negative prompt (overrides default)
            seed: Random seed for reproducibility
            width: Image width
            height: Image height
            output_path: Optional path to save image

        Returns:
            Generated PIL.Image
        """
        # Build prompt
        prompt = self._build_prompt(
            character=character,
            scene_description=scene_description,
            style=style,
            include_trigger_words=not use_controlnet  # Don't use trigger words with ControlNet for now
        )

        # Get negative prompt
        if negative_prompt is None:
            negative_prompt = self._get_negative_prompt("character")

        # Generate with or without ControlNet
        if use_controlnet:
            if control_type is None or control_image is None:
                raise ValueError("control_type and control_image required when use_controlnet=True")

            # Initialize ControlNet pipeline
            self._initialize_controlnet(control_type)

            # Generate with ControlNet
            image = self.controlnet_manager.generate(
                prompt=prompt,
                control_image=control_image,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                controlnet_conditioning_scale=controlnet_scale,
                seed=seed,
                output_path=output_path
            )

        else:
            # Initialize SDXL pipeline
            self._initialize_sdxl()

            # Load character LoRA
            self.lora_manager.load_lora(character)

            # Load additional LoRAs if specified
            if additional_loras:
                for lora_config in additional_loras:
                    self.lora_manager.load_lora(
                        lora_name=lora_config["name"],
                        weight=lora_config.get("weight")
                    )

            # Generate with SDXL + LoRA
            image = self.sdxl_manager.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                quality_preset=quality_preset,
                seed=seed,
                output_path=output_path
            )

            # Unload LoRAs
            self.lora_manager.unload_all_loras()

        return image

    def generate_batch(
        self,
        configs: List[CharacterGenerationConfig],
        output_dir: str = "outputs/batch_generation"
    ) -> List[Image.Image]:
        """
        Generate multiple character images in batch

        Args:
            configs: List of CharacterGenerationConfig
            output_dir: Output directory for batch

        Returns:
            List of generated PIL.Images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = []

        for i, config in enumerate(configs):
            print(f"\n=== Generating image {i+1}/{len(configs)} ===")

            output_path = output_dir / f"{config.character}_{i:03d}.png"

            image = self.generate_character(
                character=config.character,
                scene_description=config.scene_description,
                style=config.style,
                quality_preset=config.quality_preset,
                use_controlnet=config.use_controlnet,
                control_type=config.control_type,
                control_image=config.control_image,
                controlnet_scale=config.controlnet_scale,
                additional_loras=config.additional_loras,
                negative_prompt=config.negative_prompt,
                seed=config.seed,
                width=config.width,
                height=config.height,
                output_path=str(output_path)
            )

            images.append(image)

        print(f"\n✓ Batch generation complete: {len(images)} images generated")
        return images

    def cleanup(self):
        """Cleanup and free VRAM"""
        if self.sdxl_manager is not None:
            self.sdxl_manager.unload_pipeline()

        if self.controlnet_manager is not None:
            self.controlnet_manager.unload_pipeline()

        print("✓ Character generator cleaned up")


def main():
    """Example usage"""

    # Initialize generator
    generator = CharacterGenerator()

    # Example 1: Simple character generation
    print("=== Example 1: Simple Generation ===")
    image1 = generator.generate_character(
        character="luca",
        scene_description="running on the beach, excited expression, summer day",
        quality_preset="standard",
        seed=42,
        output_path="outputs/luca_beach.png"
    )

    # Example 2: Multiple LoRAs
    print("\n=== Example 2: Multiple LoRAs ===")
    image2 = generator.generate_character(
        character="luca",
        scene_description="standing in Portorosso town square, smiling",
        additional_loras=[
            {"name": "portorosso_town", "weight": 0.7},
            {"name": "warm_summer_lighting", "weight": 0.5}
        ],
        quality_preset="high",
        seed=123,
        output_path="outputs/luca_town.png"
    )

    # Cleanup
    generator.cleanup()

    print("\n✓ All examples complete!")


if __name__ == "__main__":
    main()
