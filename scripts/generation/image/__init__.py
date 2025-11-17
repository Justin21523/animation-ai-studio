"""
Image Generation Module

SDXL-based image generation with LoRA and ControlNet support.

Main Components:
- SDXLPipelineManager: Base SDXL pipeline
- LoRAManager: LoRA adapter management
- ControlNetPipelineManager: ControlNet-guided generation
- CharacterGenerator: High-level character generation interface

Author: Animation AI Studio
Date: 2025-11-17
"""

from .sdxl_pipeline import SDXLPipelineManager, GenerationConfig
from .lora_manager import LoRAManager, LoRARegistry, LoRAConfig
from .controlnet_pipeline import ControlNetPipelineManager
from .character_generator import CharacterGenerator, CharacterGenerationConfig

__all__ = [
    "SDXLPipelineManager",
    "GenerationConfig",
    "LoRAManager",
    "LoRARegistry",
    "LoRAConfig",
    "ControlNetPipelineManager",
    "CharacterGenerator",
    "CharacterGenerationConfig"
]
