"""
Image Generation Tools for Agent Framework

Thin wrappers that connect agent tool calls to the real SDXL/LoRA/ControlNet modules.

Notes:
- Uses ModelManager + ServiceController to stop/restart LLM when needed (VRAM constraint).
- Keeps imports lightweight at module import time; heavy deps are imported inside functions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


STYLE_ALIASES: Dict[str, str] = {
    "disney": "disney_3d",
    "dreamworks": "dreamworks_3d",
    # "italian_summer" isn't a first-class key in sdxl_config.yaml; map to pixar_3d default.
    "italian_summer": "pixar_3d",
}


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip())[:80]


def _default_output_dir() -> Path:
    return Path("outputs/agent/image_generation")


def _build_prompt(
    *,
    character: str,
    scene_description: str,
    style: str,
    sdxl_config: Dict[str, Any],
    lora_registry: Any,
) -> str:
    style_key = STYLE_ALIASES.get(style, style)
    style_prompt = (sdxl_config.get("style_prompts") or {}).get(style_key, "high quality, detailed")

    trigger_words: List[str] = []
    try:
        lora_config = lora_registry.get_character_lora(character)
        if lora_config and getattr(lora_config, "trigger_words", None):
            trigger_words = list(lora_config.trigger_words)
    except Exception:
        trigger_words = []

    base = f"{scene_description}, {style_prompt}"
    if trigger_words:
        return f"{', '.join(trigger_words)}, {base}"
    return base


async def generate_character_image(
    character: str,
    scene_description: str,
    style: str = "pixar_3d",
    quality_preset: str = "high",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate an image of a character using SDXL + LoRA (if available).

    Tool name: generate_character_image
    """
    from scripts.core.model_management.model_manager import ModelManager
    from scripts.generation.image.lora_manager import LoRARegistry, LoRAManager

    sdxl_config = _load_yaml("configs/generation/sdxl_config.yaml")
    lora_registry = LoRARegistry("configs/generation/lora_registry.yaml")

    negative_prompt = (sdxl_config.get("negative_prompts") or {}).get("character", "")
    gen_defaults = sdxl_config.get("generation") or {}
    width = int(gen_defaults.get("width", 1024))
    height = int(gen_defaults.get("height", 1024))

    prompt = _build_prompt(
        character=character,
        scene_description=scene_description,
        style=style,
        sdxl_config=sdxl_config,
        lora_registry=lora_registry,
    )

    output_dir = _default_output_dir() / _safe_name(character.lower())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(character)}_{_timestamp_ms()}.png"

    manager = ModelManager()
    llm_was_running = manager.service_controller.is_llm_running()

    try:
        if llm_was_running and not manager.service_controller.stop_llm(wait=True):
            raise RuntimeError("Failed to stop LLM service before SDXL generation")

        with manager.use_sdxl(auto_unload=True) as sdxl:
            lora_manager = LoRAManager(
                pipeline=sdxl.pipeline,
                registry=lora_registry,
            )

            lora_loaded = False
            try:
                lora_manager.load_lora(character)
                lora_loaded = True
            except FileNotFoundError as e:
                logger.warning(f"Character LoRA not found; falling back to base SDXL: {e}")
            except Exception as e:
                logger.warning(f"Failed to load character LoRA; falling back to base SDXL: {e}")

            sdxl.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                quality_preset=quality_preset,
                seed=seed,
                output_path=str(output_path),
            )

            if lora_loaded:
                lora_manager.unload_all_loras()

        return {
            "success": True,
            "output_path": str(output_path),
            "character": character,
            "style": style,
            "quality_preset": quality_preset,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
        }

    finally:
        if llm_was_running:
            # Best-effort restore for the agent to continue reasoning.
            manager.service_controller.start_llm(wait=True)


async def generate_scene_with_controlnet(
    character: str,
    scene_description: str,
    control_type: str,
    control_image_path: str,
    controlnet_scale: float = 0.9,
) -> Dict[str, Any]:
    """
    Generate image with pose/depth/etc control using ControlNet.

    Tool name: generate_scene_with_controlnet
    """
    from scripts.core.model_management.model_manager import ModelManager
    from scripts.generation.image.controlnet_pipeline import ControlNetPipelineManager
    from scripts.generation.image.lora_manager import LoRARegistry
    import torch

    sdxl_config = _load_yaml("configs/generation/sdxl_config.yaml")
    lora_registry = LoRARegistry("configs/generation/lora_registry.yaml")

    model_cfg = sdxl_config.get("model") or {}
    vram_cfg = sdxl_config.get("vram") or {}
    base_model = model_cfg.get("base_model")
    if not base_model:
        raise ValueError("Missing 'model.base_model' in configs/generation/sdxl_config.yaml")

    negative_prompt = (sdxl_config.get("negative_prompts") or {}).get("character", "")
    gen_defaults = sdxl_config.get("generation") or {}
    width = int(gen_defaults.get("width", 1024))
    height = int(gen_defaults.get("height", 1024))

    prompt = _build_prompt(
        character=character,
        scene_description=scene_description,
        style="pixar_3d",
        sdxl_config=sdxl_config,
        lora_registry=lora_registry,
    )

    output_dir = _default_output_dir() / "controlnet" / _safe_name(character.lower())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(character)}_{control_type}_{_timestamp_ms()}.png"

    manager = ModelManager()
    llm_was_running = manager.service_controller.is_llm_running()

    try:
        if llm_was_running and not manager.service_controller.stop_llm(wait=True):
            raise RuntimeError("Failed to stop LLM service before ControlNet generation")

        controlnet = ControlNetPipelineManager(
            sdxl_model_path=base_model,
            control_type=control_type,
            device="cuda",
            dtype=torch.float16,
            use_sdpa=True,
            enable_vae_slicing=bool(vram_cfg.get("enable_vae_slicing", True)),
            enable_vae_tiling=bool(vram_cfg.get("enable_vae_tiling", True)),
            variant=model_cfg.get("variant", "fp16"),
        )

        controlnet.load_pipeline()
        try:
            controlnet.generate(
                prompt=prompt,
                control_image=control_image_path,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                controlnet_conditioning_scale=controlnet_scale,
                output_path=str(output_path),
            )
        finally:
            controlnet.unload_pipeline()

        return {
            "success": True,
            "output_path": str(output_path),
            "character": character,
            "control_type": control_type,
            "control_image_path": control_image_path,
            "controlnet_scale": controlnet_scale,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
        }

    finally:
        if llm_was_running:
            manager.service_controller.start_llm(wait=True)


async def batch_generate_character_images(
    character: str,
    scene_description: str,
    num_images: int,
    consistency_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Generate multiple character images (optionally filter by face-consistency).

    Tool name: batch_generate_character_images
    """
    from scripts.core.model_management.model_manager import ModelManager
    from scripts.generation.image.lora_manager import LoRARegistry, LoRAManager

    sdxl_config = _load_yaml("configs/generation/sdxl_config.yaml")
    lora_registry = LoRARegistry("configs/generation/lora_registry.yaml")

    negative_prompt = (sdxl_config.get("negative_prompts") or {}).get("character", "")
    gen_defaults = sdxl_config.get("generation") or {}
    width = int(gen_defaults.get("width", 1024))
    height = int(gen_defaults.get("height", 1024))

    prompt = _build_prompt(
        character=character,
        scene_description=scene_description,
        style="pixar_3d",
        sdxl_config=sdxl_config,
        lora_registry=lora_registry,
    )

    batch_id = _timestamp_ms()
    output_dir = _default_output_dir() / "batch" / _safe_name(character.lower()) / str(batch_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = ModelManager()
    llm_was_running = manager.service_controller.is_llm_running()

    generated_paths: List[str] = []
    kept_paths: List[str] = []
    consistency: Optional[Dict[str, Any]] = None

    try:
        if llm_was_running and not manager.service_controller.stop_llm(wait=True):
            raise RuntimeError("Failed to stop LLM service before SDXL generation")

        with manager.use_sdxl(auto_unload=True) as sdxl:
            lora_manager = LoRAManager(
                pipeline=sdxl.pipeline,
                registry=lora_registry,
            )

            try:
                lora_manager.load_lora(character)
            except Exception as e:
                logger.warning(f"Failed to load character LoRA; generating with base SDXL: {e}")

            for i in range(int(num_images)):
                output_path = output_dir / f"{_safe_name(character)}_{i:03d}.png"
                sdxl.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    quality_preset="standard",
                    seed=None,
                    output_path=str(output_path),
                )
                generated_paths.append(str(output_path))

            lora_manager.unload_all_loras()

        # Optional face-consistency filtering (best-effort).
        try:
            if generated_paths:
                from scripts.generation.image.consistency_checker import CharacterConsistencyChecker

                checker = CharacterConsistencyChecker(device="cuda")
                results = checker.check_batch_consistency(
                    reference_image=generated_paths[0],
                    generated_images=generated_paths,
                    threshold=float(consistency_threshold),
                )
                kept_paths = [
                    p for p, r in zip(generated_paths, results)
                    if getattr(r, "is_consistent", False)
                ]
                consistency = {
                    "threshold": float(consistency_threshold),
                    "kept": len(kept_paths),
                    "total": len(generated_paths),
                }
        except Exception as e:
            logger.warning(f"Consistency filtering skipped/failed: {e}")

        if not kept_paths:
            kept_paths = list(generated_paths)

        return {
            "success": True,
            "output_dir": str(output_dir),
            "character": character,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generated_paths": generated_paths,
            "kept_paths": kept_paths,
            "consistency": consistency,
        }

    finally:
        if llm_was_running:
            manager.service_controller.start_llm(wait=True)
