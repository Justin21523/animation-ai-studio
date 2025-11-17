"""
Emotion Controller for Voice Synthesis

Controls emotion in synthesized speech through temperature and parameter adjustment.

Author: Animation AI Studio
Date: 2025-11-17
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from .gpt_sovits_wrapper import GPTSoVITSWrapper, SynthesisResult


logger = logging.getLogger(__name__)


@dataclass
class EmotionPreset:
    """Emotion synthesis preset"""
    name: str
    temperature: float
    top_k: int
    top_p: float
    speed: float
    description: str


class EmotionController:
    """
    Control emotion in synthesized speech

    Emotion mapping (temperature-based):
    - neutral: temperature=1.0 (baseline)
    - happy: temperature=1.2 (more varied, upbeat)
    - excited: temperature=1.3 (high energy)
    - sad: temperature=0.8 (monotone, slow)
    - angry: temperature=1.4 (aggressive)
    - calm: temperature=0.9 (controlled)
    - scared: temperature=1.1 (nervous, fast)

    Example:
        wrapper = GPTSoVITSWrapper(...)
        controller = EmotionController(wrapper)

        # Synthesize with emotion
        audio = controller.synthesize_with_emotion(
            text="I can't believe it!",
            character="luca",
            emotion="excited",
            intensity=1.5
        )

        # Blend emotions
        audio = controller.blend_emotions(
            text="I'm happy but also nervous",
            character="luca",
            emotion_mix={"happy": 0.6, "scared": 0.4}
        )
    """

    # Emotion presets with detailed parameters
    EMOTION_PRESETS = {
        "neutral": EmotionPreset(
            name="neutral",
            temperature=1.0,
            top_k=15,
            top_p=1.0,
            speed=1.0,
            description="Neutral, natural speech"
        ),
        "happy": EmotionPreset(
            name="happy",
            temperature=1.2,
            top_k=20,
            top_p=0.95,
            speed=1.05,
            description="Happy, cheerful, upbeat"
        ),
        "excited": EmotionPreset(
            name="excited",
            temperature=1.3,
            top_k=25,
            top_p=0.9,
            speed=1.15,
            description="Excited, energetic, enthusiastic"
        ),
        "sad": EmotionPreset(
            name="sad",
            temperature=0.8,
            top_k=10,
            top_p=1.0,
            speed=0.9,
            description="Sad, melancholic, low energy"
        ),
        "angry": EmotionPreset(
            name="angry",
            temperature=1.4,
            top_k=30,
            top_p=0.85,
            speed=1.1,
            description="Angry, aggressive, intense"
        ),
        "calm": EmotionPreset(
            name="calm",
            temperature=0.9,
            top_k=12,
            top_p=1.0,
            speed=0.95,
            description="Calm, composed, controlled"
        ),
        "scared": EmotionPreset(
            name="scared",
            temperature=1.1,
            top_k=18,
            top_p=0.92,
            speed=1.08,
            description="Scared, nervous, anxious"
        ),
        "surprised": EmotionPreset(
            name="surprised",
            temperature=1.25,
            top_k=22,
            top_p=0.93,
            speed=1.12,
            description="Surprised, shocked, amazed"
        )
    }

    def __init__(self, gpt_sovits_wrapper: GPTSoVITSWrapper):
        """
        Initialize emotion controller

        Args:
            gpt_sovits_wrapper: GPT-SoVITS wrapper instance
        """
        self.wrapper = gpt_sovits_wrapper
        logger.info("EmotionController initialized")

    def synthesize_with_emotion(
        self,
        text: str,
        character: str,
        emotion: str,
        intensity: float = 1.0,
        language: str = None,
        output_path: str = None
    ) -> SynthesisResult:
        """
        Synthesize speech with specific emotion

        Args:
            text: Text to synthesize
            character: Character name
            emotion: Emotion name (neutral, happy, sad, etc.)
            intensity: Emotion intensity multiplier (0.5 - 2.0)
                      - 0.5: Subtle emotion
                      - 1.0: Normal emotion
                      - 1.5: Strong emotion
                      - 2.0: Very strong emotion
            language: Language override
            output_path: Output file path

        Returns:
            SynthesisResult
        """
        # Get emotion preset
        if emotion not in self.EMOTION_PRESETS:
            logger.warning(f"Unknown emotion '{emotion}', using 'neutral'")
            emotion = "neutral"

        preset = self.EMOTION_PRESETS[emotion]

        # Adjust parameters by intensity
        # Temperature scales with intensity
        adjusted_temp = 1.0 + (preset.temperature - 1.0) * intensity

        # Speed scales slightly with intensity (for some emotions)
        adjusted_speed = 1.0 + (preset.speed - 1.0) * min(intensity, 1.5)

        logger.info(
            f"Synthesizing with emotion: {emotion} "
            f"(temp={adjusted_temp:.2f}, speed={adjusted_speed:.2f})"
        )

        # Synthesize using wrapper
        result = self.wrapper.synthesize(
            text=text,
            character=character,
            emotion=emotion,  # Pass emotion name for result tracking
            language=language,
            speed=adjusted_speed,
            output_path=output_path
        )

        return result

    def blend_emotions(
        self,
        text: str,
        character: str,
        emotion_mix: Dict[str, float],
        language: str = None,
        output_path: str = None
    ) -> SynthesisResult:
        """
        Blend multiple emotions

        Args:
            text: Text to synthesize
            character: Character name
            emotion_mix: Dict of emotion -> weight
                        Example: {"happy": 0.7, "excited": 0.3}
            language: Language override
            output_path: Output file path

        Returns:
            SynthesisResult

        Note:
            This is a simplified implementation that averages parameters.
            True emotion blending would require generating multiple samples
            and mixing audio, or using a multi-emotion model.
        """
        # Normalize weights
        total_weight = sum(emotion_mix.values())
        if total_weight == 0:
            logger.warning("All emotion weights are 0, using neutral")
            return self.synthesize_with_emotion(
                text, character, "neutral", 1.0, language, output_path
            )

        normalized_mix = {k: v / total_weight for k, v in emotion_mix.items()}

        # Blend parameters
        blended_temp = 1.0
        blended_speed = 1.0

        for emotion, weight in normalized_mix.items():
            if emotion not in self.EMOTION_PRESETS:
                logger.warning(f"Unknown emotion '{emotion}', skipping")
                continue

            preset = self.EMOTION_PRESETS[emotion]
            blended_temp += (preset.temperature - 1.0) * weight
            blended_speed += (preset.speed - 1.0) * weight

        logger.info(
            f"Blending emotions: {emotion_mix} -> "
            f"temp={blended_temp:.2f}, speed={blended_speed:.2f}"
        )

        # Synthesize with blended parameters
        # Use dominant emotion name
        dominant_emotion = max(normalized_mix, key=normalized_mix.get)

        result = self.wrapper.synthesize(
            text=text,
            character=character,
            emotion=f"blend_{dominant_emotion}",
            language=language,
            speed=blended_speed,
            output_path=output_path
        )

        return result

    def create_emotion_transition(
        self,
        texts: List[str],
        character: str,
        emotions: List[str],
        transition_smoothness: float = 0.5,
        language: str = None,
        output_dir: str = "outputs/tts/transitions"
    ) -> List[SynthesisResult]:
        """
        Create smooth emotion transitions across multiple text segments

        Args:
            texts: List of text segments
            character: Character name
            emotions: List of emotions (one per text segment)
            transition_smoothness: Transition smoothness (0.0 - 1.0)
                                  - 0.0: Abrupt transitions
                                  - 0.5: Moderate blending
                                  - 1.0: Very smooth blending
            language: Language override
            output_dir: Output directory

        Returns:
            List of SynthesisResult
        """
        if len(texts) != len(emotions):
            raise ValueError(
                f"texts length ({len(texts)}) must match emotions length ({len(emotions)})"
            )

        results = []

        for i, (text, emotion) in enumerate(zip(texts, emotions)):
            # Calculate transition blend if not first/last
            if 0 < i < len(texts) - 1 and transition_smoothness > 0:
                # Blend with next emotion
                next_emotion = emotions[i + 1]

                emotion_mix = {
                    emotion: 1.0 - transition_smoothness,
                    next_emotion: transition_smoothness
                }

                result = self.blend_emotions(
                    text=text,
                    character=character,
                    emotion_mix=emotion_mix,
                    language=language
                )
            else:
                # No blending for first/last or if smoothness = 0
                result = self.synthesize_with_emotion(
                    text=text,
                    character=character,
                    emotion=emotion,
                    language=language
                )

            results.append(result)

        logger.info(f"✓ Created {len(results)} segments with emotion transitions")

        return results

    def get_available_emotions(self) -> List[str]:
        """
        Get list of available emotion presets

        Returns:
            List of emotion names
        """
        return list(self.EMOTION_PRESETS.keys())

    def get_emotion_info(self, emotion: str) -> Optional[EmotionPreset]:
        """
        Get detailed info about an emotion preset

        Args:
            emotion: Emotion name

        Returns:
            EmotionPreset or None
        """
        return self.EMOTION_PRESETS.get(emotion)

    def print_emotion_presets(self):
        """Print all available emotion presets"""
        print("=" * 70)
        print("Available Emotion Presets")
        print("=" * 70)

        for name, preset in self.EMOTION_PRESETS.items():
            print(f"\n{name.upper()}")
            print(f"  Temperature: {preset.temperature}")
            print(f"  Speed: {preset.speed}x")
            print(f"  Description: {preset.description}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Emotion Controller Example")
    print("=" * 60)

    # Initialize wrapper
    wrapper = GPTSoVITSWrapper(
        repo_path="/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS",
        models_dir="/mnt/c/AI_LLM_projects/ai_warehouse/models/tts"
    )

    # Load character
    wrapper.load_voice_model(
        character_name="luca",
        gpt_model_path="/path/to/luca_gpt.ckpt",
        sovits_model_path="/path/to/luca_sovits.pth"
    )

    # Initialize controller
    controller = EmotionController(wrapper)

    # Print available emotions
    controller.print_emotion_presets()

    # Example: Synthesize with emotion
    print("\n" + "=" * 60)
    print("Example: Excited speech")
    print("=" * 60)

    result = controller.synthesize_with_emotion(
        text="I can't believe we're really going to Portorosso!",
        character="luca",
        emotion="excited",
        intensity=1.3
    )

    print(f"\n✓ Generated: {result.audio_path}")

    # Example: Blend emotions
    print("\n" + "=" * 60)
    print("Example: Blended emotions (happy + nervous)")
    print("=" * 60)

    result = controller.blend_emotions(
        text="I'm so happy to be here, but I'm also a little nervous",
        character="luca",
        emotion_mix={"happy": 0.6, "scared": 0.4}
    )

    print(f"\n✓ Generated: {result.audio_path}")

    wrapper.cleanup()
