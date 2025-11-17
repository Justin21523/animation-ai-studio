"""
Character Voice Manager

Manages character voice models registry and provides high-level synthesis interface.

Author: Animation AI Studio
Date: 2025-11-17
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import yaml

from .gpt_sovits_wrapper import GPTSoVITSWrapper, VoiceModel, SynthesisResult
from .emotion_controller import EmotionController


logger = logging.getLogger(__name__)


@dataclass
class CharacterVoiceConfig:
    """Character voice configuration"""
    character_name: str
    display_name: str
    gpt_model: str
    sovits_model: str
    reference_audio: Optional[str]
    reference_text: Optional[str]
    language: str
    default_emotion: str
    voice_description: str


class CharacterVoiceManager:
    """
    High-level manager for character voice synthesis

    Features:
    - Load character voices from registry
    - Simple synthesis interface
    - Emotion-aware synthesis
    - Batch generation
    - Voice model caching

    Example:
        manager = CharacterVoiceManager(
            config_path="configs/generation/character_voices.yaml",
            repo_path="/path/to/GPT-SoVITS"
        )

        # Synthesize with character
        audio = manager.synthesize(
            text="Silenzio, Bruno!",
            character="luca",
            emotion="excited"
        )
    """

    def __init__(
        self,
        config_path: str,
        repo_path: str,
        models_dir: str = None,
        device: str = "cuda"
    ):
        """
        Initialize character voice manager

        Args:
            config_path: Path to character_voices.yaml
            repo_path: Path to GPT-SoVITS repository
            models_dir: Directory containing voice models
            device: CUDA device
        """
        self.config_path = Path(config_path)
        self.repo_path = Path(repo_path)
        self.models_dir = Path(models_dir) if models_dir else None
        self.device = device

        # Character registry
        self.characters: Dict[str, CharacterVoiceConfig] = {}

        # GPT-SoVITS wrapper
        self.wrapper = GPTSoVITSWrapper(
            repo_path=str(self.repo_path),
            models_dir=str(self.models_dir) if self.models_dir else None,
            device=self.device
        )

        # Emotion controller
        self.emotion_controller = EmotionController(self.wrapper)

        # Load character registry
        if self.config_path.exists():
            self._load_character_registry()
        else:
            logger.warning(f"Character config not found: {self.config_path}")

        logger.info(f"CharacterVoiceManager initialized ({len(self.characters)} characters)")

    def _load_character_registry(self):
        """Load character voice registry from YAML"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if "characters" not in config:
            logger.warning("No 'characters' section in config")
            return

        for char_id, char_config in config["characters"].items():
            character = CharacterVoiceConfig(
                character_name=char_id,
                display_name=char_config.get("display_name", char_id),
                gpt_model=char_config["gpt_model"],
                sovits_model=char_config["sovits_model"],
                reference_audio=char_config.get("reference_audio"),
                reference_text=char_config.get("reference_text"),
                language=char_config.get("language", "en"),
                default_emotion=char_config.get("default_emotion", "neutral"),
                voice_description=char_config.get("description", "")
            )

            self.characters[char_id] = character

        logger.info(f"Loaded {len(self.characters)} characters from registry")

    def load_character(self, character_name: str) -> bool:
        """
        Load character voice model

        Args:
            character_name: Character identifier

        Returns:
            True if loaded successfully
        """
        if character_name not in self.characters:
            logger.error(f"Character '{character_name}' not in registry")
            return False

        char_config = self.characters[character_name]

        success = self.wrapper.load_voice_model(
            character_name=character_name,
            gpt_model_path=char_config.gpt_model,
            sovits_model_path=char_config.sovits_model,
            reference_audio_path=char_config.reference_audio,
            reference_text=char_config.reference_text,
            language=char_config.language
        )

        if success:
            logger.info(f"✓ Character '{char_config.display_name}' ready")

        return success

    def synthesize(
        self,
        text: str,
        character: str,
        emotion: str = None,
        intensity: float = 1.0,
        language: str = None,
        output_path: str = None
    ) -> SynthesisResult:
        """
        Synthesize speech for character

        Args:
            text: Text to synthesize
            character: Character name
            emotion: Emotion (uses character default if None)
            intensity: Emotion intensity
            language: Language override
            output_path: Output file path

        Returns:
            SynthesisResult
        """
        # Ensure character is loaded
        if character not in self.wrapper.get_available_voices():
            logger.info(f"Loading character '{character}'...")
            if not self.load_character(character):
                return SynthesisResult(
                    audio_path="",
                    duration_seconds=0.0,
                    sample_rate=44100,
                    text=text,
                    character=character,
                    emotion=emotion or "neutral",
                    success=False,
                    error_message=f"Failed to load character '{character}'"
                )

        # Get character config
        char_config = self.characters[character]

        # Use character default emotion if not specified
        if emotion is None:
            emotion = char_config.default_emotion

        # Synthesize with emotion
        result = self.emotion_controller.synthesize_with_emotion(
            text=text,
            character=character,
            emotion=emotion,
            intensity=intensity,
            language=language or char_config.language,
            output_path=output_path
        )

        return result

    def synthesize_batch(
        self,
        texts: List[str],
        character: str,
        emotions: Optional[List[str]] = None,
        output_dir: str = "outputs/tts/batch"
    ) -> List[SynthesisResult]:
        """
        Batch synthesis for character

        Args:
            texts: List of texts
            character: Character name
            emotions: List of emotions (optional)
            output_dir: Output directory

        Returns:
            List of SynthesisResult
        """
        # Ensure character is loaded
        if character not in self.wrapper.get_available_voices():
            self.load_character(character)

        return self.wrapper.synthesize_batch(
            texts=texts,
            character=character,
            emotions=emotions,
            output_dir=output_dir
        )

    def get_available_characters(self) -> List[str]:
        """Get list of available characters"""
        return list(self.characters.keys())

    def get_character_info(self, character: str) -> Optional[CharacterVoiceConfig]:
        """Get character configuration"""
        return self.characters.get(character)

    def unload_all(self):
        """Unload all voice models"""
        self.wrapper.unload_voice_model()

    def cleanup(self):
        """Cleanup resources"""
        self.wrapper.cleanup()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Character Voice Manager Example")
    print("=" * 60)

    manager = CharacterVoiceManager(
        config_path="configs/generation/character_voices.yaml",
        repo_path="/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS",
        models_dir="/mnt/c/AI_LLM_projects/ai_warehouse/models/tts"
    )

    print(f"\nAvailable characters: {manager.get_available_characters()}")

    # Synthesize speech
    result = manager.synthesize(
        text="Silenzio, Bruno!",
        character="luca",
        emotion="excited"
    )

    print(f"\n✓ Generated: {result.audio_path}")

    manager.cleanup()
