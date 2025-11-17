"""
LoRA Manager

Manages LoRA adapter loading, fusion, and switching for SDXL.
Supports character LoRAs, style LoRAs, and background LoRAs.

Architecture:
- Dynamic LoRA loading/unloading
- Multiple LoRA fusion (weighted)
- LoRA registry from config file
- Character-specific LoRA management

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import yaml

try:
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file
except ImportError:
    raise ImportError(
        "Required packages not installed. "
        "Install with: pip install diffusers safetensors"
    )


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""
    name: str
    path: str
    type: str  # "character", "style", "background", "pose"
    trigger_words: List[str]
    recommended_weight: float = 0.8
    description: str = ""
    metadata: Dict[str, Any] = None


class LoRARegistry:
    """
    LoRA Registry

    Manages LoRA metadata from configuration file.
    Maps character names to their LoRA adapters.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize LoRA Registry

        Args:
            registry_path: Path to lora_registry.yaml
        """
        self.registry_path = registry_path
        self.loras: Dict[str, LoRAConfig] = {}

        if registry_path:
            self.load_registry(registry_path)

    def load_registry(self, registry_path: str):
        """
        Load LoRA registry from YAML file

        Args:
            registry_path: Path to lora_registry.yaml
        """
        registry_path = Path(registry_path)
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {registry_path}")

        with open(registry_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse LoRAs
        for lora_data in data.get("loras", []):
            lora_config = LoRAConfig(
                name=lora_data["name"],
                path=lora_data["path"],
                type=lora_data["type"],
                trigger_words=lora_data.get("trigger_words", []),
                recommended_weight=lora_data.get("recommended_weight", 0.8),
                description=lora_data.get("description", ""),
                metadata=lora_data.get("metadata", {})
            )
            self.loras[lora_config.name] = lora_config

        print(f"Loaded {len(self.loras)} LoRAs from registry")

    def get_lora(self, name: str) -> Optional[LoRAConfig]:
        """Get LoRA config by name"""
        return self.loras.get(name)

    def get_loras_by_type(self, lora_type: str) -> List[LoRAConfig]:
        """Get all LoRAs of a specific type"""
        return [lora for lora in self.loras.values() if lora.type == lora_type]

    def get_character_lora(self, character_name: str) -> Optional[LoRAConfig]:
        """
        Get character LoRA by character name

        Args:
            character_name: Character name (e.g., "luca", "alberto")

        Returns:
            LoRAConfig if found, None otherwise
        """
        # Try exact match first
        lora = self.get_lora(character_name)
        if lora:
            return lora

        # Try case-insensitive match
        for name, lora in self.loras.items():
            if name.lower() == character_name.lower() and lora.type == "character":
                return lora

        return None

    def list_loras(self) -> List[str]:
        """List all LoRA names"""
        return list(self.loras.keys())


class LoRAManager:
    """
    LoRA Manager for SDXL Pipeline

    Features:
    - Load/unload LoRA adapters
    - Fuse multiple LoRAs with weights
    - Character-specific LoRA management
    - Trigger word integration

    Usage:
        lora_manager = LoRAManager(pipeline, registry_path="configs/generation/lora_registry.yaml")
        lora_manager.load_lora("luca", weight=0.8)
        prompt = lora_manager.add_trigger_words("a boy running on the beach", "luca")
    """

    def __init__(
        self,
        pipeline: StableDiffusionXLPipeline,
        registry: Optional[LoRARegistry] = None,
        registry_path: Optional[str] = None
    ):
        """
        Initialize LoRA Manager

        Args:
            pipeline: SDXL pipeline
            registry: Optional LoRARegistry instance
            registry_path: Optional path to lora_registry.yaml
        """
        self.pipeline = pipeline
        self.registry = registry or LoRARegistry(registry_path)
        self.loaded_loras: Dict[str, float] = {}  # name -> weight

    def load_lora(
        self,
        lora_name: str,
        weight: Optional[float] = None,
        adapter_name: Optional[str] = None
    ):
        """
        Load LoRA adapter

        Args:
            lora_name: LoRA name (from registry)
            weight: LoRA weight (0.0-1.0), defaults to recommended_weight
            adapter_name: Optional adapter name (defaults to lora_name)
        """
        # Get LoRA config
        lora_config = self.registry.get_lora(lora_name)
        if not lora_config:
            raise ValueError(f"LoRA '{lora_name}' not found in registry")

        # Use recommended weight if not specified
        if weight is None:
            weight = lora_config.recommended_weight

        # Adapter name defaults to lora_name
        adapter_name = adapter_name or lora_name

        # Check if LoRA file exists
        lora_path = Path(lora_config.path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        print(f"Loading LoRA: {lora_name} (weight={weight:.2f})")

        try:
            # Load LoRA weights
            self.pipeline.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name=adapter_name
            )

            # Set adapter weight
            self.pipeline.set_adapters([adapter_name], adapter_weights=[weight])

            # Track loaded LoRA
            self.loaded_loras[adapter_name] = weight

            print(f"✓ LoRA '{lora_name}' loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")

    def load_multiple_loras(
        self,
        lora_configs: List[Dict[str, Any]]
    ):
        """
        Load and fuse multiple LoRAs

        Args:
            lora_configs: List of dicts with "name" and optional "weight"

        Example:
            lora_manager.load_multiple_loras([
                {"name": "luca", "weight": 0.8},
                {"name": "pixar_style", "weight": 0.6}
            ])
        """
        adapter_names = []
        adapter_weights = []

        for config in lora_configs:
            lora_name = config["name"]
            weight = config.get("weight")

            # Load LoRA
            self.load_lora(lora_name, weight=weight, adapter_name=lora_name)

            adapter_names.append(lora_name)
            adapter_weights.append(self.loaded_loras[lora_name])

        # Fuse all loaded LoRAs
        print(f"Fusing {len(adapter_names)} LoRAs...")
        self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

    def unload_lora(self, adapter_name: str):
        """
        Unload specific LoRA adapter

        Args:
            adapter_name: Adapter name to unload
        """
        if adapter_name not in self.loaded_loras:
            print(f"LoRA '{adapter_name}' not loaded")
            return

        try:
            self.pipeline.delete_adapters(adapter_name)
            del self.loaded_loras[adapter_name]
            print(f"✓ LoRA '{adapter_name}' unloaded")
        except Exception as e:
            print(f"Warning: Failed to unload LoRA '{adapter_name}': {e}")

    def unload_all_loras(self):
        """Unload all LoRA adapters"""
        if not self.loaded_loras:
            print("No LoRAs loaded")
            return

        print(f"Unloading {len(self.loaded_loras)} LoRAs...")
        for adapter_name in list(self.loaded_loras.keys()):
            self.unload_lora(adapter_name)

    def set_lora_weight(self, adapter_name: str, weight: float):
        """
        Adjust weight of loaded LoRA

        Args:
            adapter_name: Adapter name
            weight: New weight (0.0-1.0)
        """
        if adapter_name not in self.loaded_loras:
            raise ValueError(f"LoRA '{adapter_name}' not loaded")

        self.loaded_loras[adapter_name] = weight

        # Update adapter weights
        adapter_names = list(self.loaded_loras.keys())
        adapter_weights = [self.loaded_loras[name] for name in adapter_names]
        self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

        print(f"✓ LoRA '{adapter_name}' weight set to {weight:.2f}")

    def add_trigger_words(
        self,
        prompt: str,
        lora_name: str,
        prepend: bool = True
    ) -> str:
        """
        Add LoRA trigger words to prompt

        Args:
            prompt: Original prompt
            lora_name: LoRA name
            prepend: If True, prepend trigger words; else append

        Returns:
            Prompt with trigger words
        """
        lora_config = self.registry.get_lora(lora_name)
        if not lora_config:
            print(f"Warning: LoRA '{lora_name}' not found in registry")
            return prompt

        if not lora_config.trigger_words:
            return prompt

        trigger_str = ", ".join(lora_config.trigger_words)

        if prepend:
            return f"{trigger_str}, {prompt}"
        else:
            return f"{prompt}, {trigger_str}"

    def get_character_prompt(
        self,
        character_name: str,
        scene_description: str,
        include_trigger_words: bool = True
    ) -> str:
        """
        Generate character-specific prompt with trigger words

        Args:
            character_name: Character name (e.g., "luca")
            scene_description: Scene description
            include_trigger_words: Whether to add trigger words

        Returns:
            Complete prompt string
        """
        if include_trigger_words:
            lora_config = self.registry.get_character_lora(character_name)
            if lora_config and lora_config.trigger_words:
                trigger_str = ", ".join(lora_config.trigger_words)
                return f"{trigger_str}, {scene_description}"

        return scene_description

    def get_loaded_loras(self) -> Dict[str, float]:
        """Get currently loaded LoRAs and their weights"""
        return self.loaded_loras.copy()


def main():
    """Example usage"""

    # Example: Create registry and load LoRA
    from scripts.generation.image.sdxl_pipeline import SDXLPipelineManager

    # Initialize SDXL pipeline
    model_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/stable-diffusion-xl-base-1.0"
    pipeline_manager = SDXLPipelineManager(model_path=model_path)
    pipeline_manager.load_pipeline()

    # Initialize LoRA manager
    lora_manager = LoRAManager(
        pipeline=pipeline_manager.pipeline,
        registry_path="configs/generation/lora_registry.yaml"
    )

    # Load character LoRA
    lora_manager.load_lora("luca", weight=0.8)

    # Generate prompt with trigger words
    prompt = lora_manager.add_trigger_words(
        "running on the beach, excited expression, pixar style, 3d animation",
        "luca"
    )

    # Generate image
    image = pipeline_manager.generate(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        quality_preset="standard",
        seed=42
    )

    # Unload LoRA
    lora_manager.unload_all_loras()

    # Unload pipeline
    pipeline_manager.unload_pipeline()


if __name__ == "__main__":
    main()
