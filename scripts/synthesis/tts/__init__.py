"""
Text-to-Speech (TTS) Module

GPT-SoVITS-based character voice synthesis.

Components:
- GPTSoVITSWrapper: Main TTS inference wrapper
- VoiceDatasetBuilder: Extract voice samples from films
- EmotionController: Emotion-aware synthesis
- CharacterVoiceManager: Voice model registry and management

Author: Animation AI Studio
Date: 2025-11-17
"""

from .gpt_sovits_wrapper import (
    GPTSoVITSWrapper,
    VoiceModel,
    SynthesisResult
)
from .voice_dataset_builder import (
    VoiceDatasetBuilder,
    VoiceSample,
    DatasetStats
)
from .emotion_controller import (
    EmotionController,
    EmotionPreset
)
from .character_voice_manager import (
    CharacterVoiceManager,
    CharacterVoiceConfig
)

__all__ = [
    # Core wrapper
    "GPTSoVITSWrapper",
    "VoiceModel",
    "SynthesisResult",

    # Dataset building
    "VoiceDatasetBuilder",
    "VoiceSample",
    "DatasetStats",

    # Emotion control
    "EmotionController",
    "EmotionPreset",

    # Character management
    "CharacterVoiceManager",
    "CharacterVoiceConfig",
]
