"""
Unit Tests for Voice Synthesis Module

Tests GPT-SoVITS wrapper, emotion control, dataset building, and batch synthesis.

Author: Animation AI Studio
Date: 2025-11-17
"""

import pytest
import torch
from pathlib import Path
import yaml

from scripts.synthesis.tts import (
    GPTSoVITSWrapper,
    VoiceModel,
    SynthesisResult,
    VoiceDatasetBuilder,
    VoiceSample,
    EmotionController,
    EmotionPreset,
    CharacterVoiceManager,
    CharacterVoiceConfig,
    VoiceModelTrainer,
    TrainingConfig,
    BatchSynthesisPipeline,
    BatchSynthesisConfig
)


# ============================================================================
# GPTSoVITSWrapper Tests
# ============================================================================

class TestGPTSoVITSWrapper:
    """Tests for GPT-SoVITS wrapper"""

    def test_wrapper_initialization(self):
        """Test wrapper initialization"""
        # This will fail if GPT-SoVITS repo doesn't exist
        # Skip test if repo not found
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        wrapper = GPTSoVITSWrapper(
            repo_path=repo_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert wrapper.repo_path.exists()
        assert wrapper.device in ["cuda", "cpu"]

    def test_emotion_presets(self):
        """Test emotion presets are defined"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        wrapper = GPTSoVITSWrapper(repo_path=repo_path)

        # Check emotion presets exist
        assert "neutral" in wrapper.EMOTION_PRESETS
        assert "happy" in wrapper.EMOTION_PRESETS
        assert "excited" in wrapper.EMOTION_PRESETS
        assert "sad" in wrapper.EMOTION_PRESETS

        # Check preset structure
        neutral = wrapper.EMOTION_PRESETS["neutral"]
        assert "temperature" in neutral
        assert "top_k" in neutral
        assert "top_p" in neutral

    def test_load_voice_model(self):
        """Test voice model loading (placeholder)"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        wrapper = GPTSoVITSWrapper(repo_path=repo_path)

        # This will fail without actual models, but tests the interface
        # In actual implementation, this would load real models
        result = wrapper.load_voice_model(
            character_name="test_char",
            gpt_model_path="/nonexistent/gpt.ckpt",
            sovits_model_path="/nonexistent/sovits.pth"
        )

        # Should return False since models don't exist
        assert result is False

    def test_get_available_voices(self):
        """Test getting available voices"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        wrapper = GPTSoVITSWrapper(repo_path=repo_path)

        voices = wrapper.get_available_voices()
        assert isinstance(voices, list)
        assert len(voices) == 0  # No voices loaded initially


# ============================================================================
# EmotionController Tests
# ============================================================================

class TestEmotionController:
    """Tests for emotion controller"""

    def test_emotion_presets(self):
        """Test emotion presets"""
        assert "neutral" in EmotionController.EMOTION_PRESETS
        assert "happy" in EmotionController.EMOTION_PRESETS

        happy = EmotionController.EMOTION_PRESETS["happy"]
        assert isinstance(happy, EmotionPreset)
        assert happy.temperature > 1.0  # Happy should be more varied
        assert happy.speed >= 1.0  # Happy might be slightly faster

    def test_get_available_emotions(self):
        """Test getting available emotions"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        wrapper = GPTSoVITSWrapper(repo_path=repo_path)
        controller = EmotionController(wrapper)

        emotions = controller.get_available_emotions()
        assert isinstance(emotions, list)
        assert len(emotions) > 0
        assert "neutral" in emotions
        assert "happy" in emotions

    def test_get_emotion_info(self):
        """Test getting emotion info"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        wrapper = GPTSoVITSWrapper(repo_path=repo_path)
        controller = EmotionController(wrapper)

        info = controller.get_emotion_info("happy")
        assert info is not None
        assert info.name == "happy"
        assert info.temperature > 0
        assert info.speed > 0


# ============================================================================
# VoiceDatasetBuilder Tests
# ============================================================================

class TestVoiceDatasetBuilder:
    """Tests for voice dataset builder"""

    def test_builder_initialization(self):
        """Test builder initialization"""
        builder = VoiceDatasetBuilder(
            whisper_model="tiny",  # Use small model for testing
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert builder.whisper_model_name == "tiny"
        assert builder.device in ["cuda", "cpu"]


# ============================================================================
# VoiceModelTrainer Tests
# ============================================================================

class TestVoiceModelTrainer:
    """Tests for voice model trainer"""

    def test_trainer_initialization(self):
        """Test trainer initialization"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        trainer = VoiceModelTrainer(
            gpt_sovits_repo=repo_path,
            output_dir="outputs/test_models"
        )

        assert trainer.repo_path.exists()
        assert trainer.output_dir.exists()


# ============================================================================
# CharacterVoiceManager Tests
# ============================================================================

class TestCharacterVoiceManager:
    """Tests for character voice manager"""

    def test_manager_without_config(self):
        """Test manager initialization without config"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        # Create manager with non-existent config
        manager = CharacterVoiceManager(
            config_path="/nonexistent/config.yaml",
            repo_path=repo_path
        )

        # Should initialize but have no characters
        assert len(manager.characters) == 0

    def test_get_available_characters(self):
        """Test getting available characters"""
        repo_path = "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
        config_path = "configs/generation/character_voices.yaml"

        if not Path(repo_path).exists():
            pytest.skip("GPT-SoVITS repository not found")

        if not Path(config_path).exists():
            pytest.skip("Character voices config not found")

        manager = CharacterVoiceManager(
            config_path=config_path,
            repo_path=repo_path
        )

        characters = manager.get_available_characters()
        assert isinstance(characters, list)


# ============================================================================
# BatchSynthesisPipeline Tests
# ============================================================================

class TestBatchSynthesisPipeline:
    """Tests for batch synthesis pipeline"""

    def test_batch_config_creation(self):
        """Test batch synthesis config"""
        config = BatchSynthesisConfig(
            character="test_char",
            texts=["Hello", "World"],
            emotions=["happy", "neutral"]
        )

        assert config.character == "test_char"
        assert len(config.texts) == 2
        assert len(config.emotions) == 2


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfigurations:
    """Test configuration files"""

    def test_tts_config_exists(self):
        """Test TTS config file exists"""
        config_path = Path("configs/generation/tts_config.yaml")
        assert config_path.exists(), f"TTS config not found: {config_path}"

    def test_tts_config_valid(self):
        """Test TTS config is valid YAML"""
        config_path = Path("configs/generation/tts_config.yaml")
        if not config_path.exists():
            pytest.skip("TTS config not found")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "gpt_sovits" in config
        assert "emotions" in config

    def test_character_voices_config_exists(self):
        """Test character voices config exists"""
        config_path = Path("configs/generation/character_voices.yaml")
        assert config_path.exists(), f"Character voices config not found: {config_path}"

    def test_character_voices_config_valid(self):
        """Test character voices config is valid"""
        config_path = Path("configs/generation/character_voices.yaml")
        if not config_path.exists():
            pytest.skip("Character voices config not found")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "characters" in config

        # Check character structure
        if config["characters"]:
            first_char = list(config["characters"].values())[0]
            assert "display_name" in first_char
            assert "gpt_model" in first_char
            assert "sovits_model" in first_char


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
