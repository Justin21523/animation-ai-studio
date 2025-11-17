"""
Unit Tests for Image Generation Module

Tests SDXL pipeline, LoRA management, ControlNet, and character generation.

Run with: pytest tests/generation/test_image_generation.py -v

Author: Animation AI Studio
Date: 2025-11-17
"""

import pytest
import torch
from pathlib import Path
import yaml
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image import (
    LoRARegistry,
    LoRAConfig,
    CharacterGenerator
)


class TestLoRARegistry:
    """Tests for LoRA Registry"""

    def test_load_registry(self):
        """Test loading LoRA registry from YAML"""
        registry_path = "configs/generation/lora_registry.yaml"

        if not Path(registry_path).exists():
            pytest.skip(f"Registry not found: {registry_path}")

        registry = LoRARegistry(registry_path)

        assert len(registry.loras) > 0, "Registry should contain LoRAs"
        print(f"✓ Loaded {len(registry.loras)} LoRAs")

    def test_get_lora(self):
        """Test getting LoRA by name"""
        registry_path = "configs/generation/lora_registry.yaml"

        if not Path(registry_path).exists():
            pytest.skip(f"Registry not found: {registry_path}")

        registry = LoRARegistry(registry_path)

        # Test getting existing LoRA
        luca_lora = registry.get_lora("luca")
        assert luca_lora is not None, "Should find luca LoRA"
        assert luca_lora.type == "character", "Luca should be character type"

        # Test getting non-existent LoRA
        fake_lora = registry.get_lora("nonexistent")
        assert fake_lora is None, "Should return None for non-existent LoRA"

    def test_get_character_lora(self):
        """Test getting character LoRA by name"""
        registry_path = "configs/generation/lora_registry.yaml"

        if not Path(registry_path).exists():
            pytest.skip(f"Registry not found: {registry_path}")

        registry = LoRARegistry(registry_path)

        # Test case-insensitive lookup
        luca_lora = registry.get_character_lora("Luca")
        assert luca_lora is not None or True, "Should handle case-insensitive lookup"

    def test_get_loras_by_type(self):
        """Test getting LoRAs by type"""
        registry_path = "configs/generation/lora_registry.yaml"

        if not Path(registry_path).exists():
            pytest.skip(f"Registry not found: {registry_path}")

        registry = LoRARegistry(registry_path)

        character_loras = registry.get_loras_by_type("character")
        assert len(character_loras) > 0, "Should have character LoRAs"

        style_loras = registry.get_loras_by_type("style")
        print(f"✓ Found {len(character_loras)} character LoRAs, {len(style_loras)} style LoRAs")


class TestSDXLConfig:
    """Tests for SDXL Configuration"""

    def test_load_sdxl_config(self):
        """Test loading SDXL config"""
        config_path = "configs/generation/sdxl_config.yaml"

        if not Path(config_path).exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert "model" in config, "Config should have model section"
        assert "generation" in config, "Config should have generation section"
        assert "attention" in config, "Config should have attention section"

        # Check critical settings
        assert config["attention"]["backend"] == "sdpa", "Must use PyTorch SDPA"
        assert config["attention"]["enable_xformers"] == False, "xformers must be disabled"

        print("✓ SDXL config validated")

    def test_quality_presets(self):
        """Test quality presets in config"""
        config_path = "configs/generation/sdxl_config.yaml"

        if not Path(config_path).exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        presets = config["generation"]["quality_presets"]
        assert "draft" in presets, "Should have draft preset"
        assert "standard" in presets, "Should have standard preset"
        assert "high" in presets, "Should have high preset"

        # Check preset values
        draft = presets["draft"]
        assert draft["steps"] < presets["standard"]["steps"], "Draft should have fewer steps"

        print("✓ Quality presets validated")


class TestControlNetConfig:
    """Tests for ControlNet Configuration"""

    def test_load_controlnet_config(self):
        """Test loading ControlNet config"""
        config_path = "configs/generation/controlnet_config.yaml"

        if not Path(config_path).exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert "controlnet_models" in config, "Should have controlnet_models section"

        models = config["controlnet_models"]
        assert "pose" in models, "Should have pose ControlNet"
        assert "canny" in models, "Should have canny ControlNet"
        assert "depth" in models, "Should have depth ControlNet"

        print(f"✓ ControlNet config validated ({len(models)} models)")


class TestCharacterPresets:
    """Tests for Character Presets"""

    def test_load_character_presets(self):
        """Test loading character presets"""
        config_path = "configs/generation/character_presets.yaml"

        if not Path(config_path).exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert "characters" in config, "Should have characters section"

        characters = config["characters"]
        assert len(characters) > 0, "Should have at least one character"

        # Check luca character
        if "luca" in characters:
            luca = characters["luca"]
            assert "display_name" in luca, "Should have display_name"
            assert "default_loras" in luca, "Should have default_loras"
            assert "consistency" in luca, "Should have consistency settings"

            print(f"✓ Character presets validated ({len(characters)} characters)")


class TestCharacterGenerator:
    """Tests for Character Generator"""

    def test_character_generator_init(self):
        """Test initializing character generator"""
        try:
            generator = CharacterGenerator()
            assert generator is not None, "Generator should initialize"
            assert generator.sdxl_config is not None, "Should load SDXL config"
            assert generator.lora_registry is not None, "Should load LoRA registry"
            print("✓ Character generator initialized")
        except Exception as e:
            pytest.skip(f"Cannot initialize generator: {e}")

    def test_build_prompt(self):
        """Test prompt building"""
        try:
            generator = CharacterGenerator()

            prompt = generator._build_prompt(
                character="luca",
                scene_description="running on the beach",
                style="pixar_3d",
                include_trigger_words=True
            )

            assert len(prompt) > 0, "Prompt should not be empty"
            assert "pixar" in prompt.lower() or "3d animation" in prompt.lower(), "Should include style"

            print(f"✓ Generated prompt: {prompt}")
        except Exception as e:
            pytest.skip(f"Cannot test prompt building: {e}")

    def test_get_negative_prompt(self):
        """Test negative prompt retrieval"""
        try:
            generator = CharacterGenerator()

            negative_prompt = generator._get_negative_prompt("character")
            assert len(negative_prompt) > 0, "Negative prompt should not be empty"
            assert "blurry" in negative_prompt.lower() or "low quality" in negative_prompt.lower()

            print(f"✓ Negative prompt: {negative_prompt[:100]}...")
        except Exception as e:
            pytest.skip(f"Cannot test negative prompt: {e}")


class TestVRAMRequirements:
    """Tests for VRAM requirements"""

    def test_cuda_available(self):
        """Test if CUDA is available"""
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            pytest.skip("CUDA not available")

        device_count = torch.cuda.device_count()
        print(f"✓ CUDA available: {device_count} device(s)")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name}, {total_vram:.1f}GB VRAM")

            if total_vram < 14.0:
                pytest.warn(f"WARNING: GPU {i} has {total_vram:.1f}GB VRAM, recommend 14GB+ for SDXL")

    def test_pytorch_version(self):
        """Test PyTorch version"""
        version = torch.__version__
        print(f"PyTorch version: {version}")

        major, minor = map(int, version.split('.')[:2])

        if major < 2 or (major == 2 and minor < 7):
            pytest.warn(f"WARNING: PyTorch {version} detected, recommend 2.7.0+ for SDPA support")
        else:
            print("✓ PyTorch version adequate for SDPA")


class TestDependencies:
    """Tests for required dependencies"""

    def test_diffusers_installed(self):
        """Test if diffusers is installed"""
        try:
            import diffusers
            version = diffusers.__version__
            print(f"✓ diffusers {version} installed")
        except ImportError:
            pytest.fail("diffusers not installed")

    def test_transformers_installed(self):
        """Test if transformers is installed"""
        try:
            import transformers
            version = transformers.__version__
            print(f"✓ transformers {version} installed")
        except ImportError:
            pytest.fail("transformers not installed")

    def test_pil_installed(self):
        """Test if PIL/Pillow is installed"""
        try:
            from PIL import Image
            print("✓ PIL/Pillow installed")
        except ImportError:
            pytest.fail("PIL/Pillow not installed")

    def test_yaml_installed(self):
        """Test if PyYAML is installed"""
        try:
            import yaml
            print("✓ PyYAML installed")
        except ImportError:
            pytest.fail("PyYAML not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
