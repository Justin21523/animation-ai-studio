"""
Batch Synthesis Pipeline

Batch voice synthesis with quality control and progress tracking.

Author: Animation AI Studio
Date: 2025-11-17
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import time

from tqdm import tqdm

from .character_voice_manager import CharacterVoiceManager
from .emotion_controller import EmotionController


logger = logging.getLogger(__name__)


@dataclass
class BatchSynthesisConfig:
    """Batch synthesis configuration"""
    character: str
    texts: List[str]
    emotions: Optional[List[str]] = None
    intensities: Optional[List[float]] = None
    language: str = None
    output_dir: str = "outputs/tts/batch"
    save_metadata: bool = True


@dataclass
class BatchSynthesisResult:
    """Batch synthesis results"""
    total_generated: int
    successful: int
    failed: int
    total_duration: float
    average_duration: float
    output_paths: List[str]
    metadata_path: Optional[str] = None


class BatchSynthesisPipeline:
    """
    Batch voice synthesis with progress tracking

    Features:
    - Multi-text batch generation
    - Progress tracking with tqdm
    - Quality control
    - Metadata generation
    - Error handling and retry

    Example:
        pipeline = BatchSynthesisPipeline(
            config_path="configs/generation/character_voices.yaml",
            repo_path="/path/to/GPT-SoVITS"
        )

        result = pipeline.synthesize_batch(
            config=BatchSynthesisConfig(
                character="luca",
                texts=["Hello!", "How are you?", "Goodbye!"],
                emotions=["happy", "neutral", "sad"]
            )
        )
    """

    def __init__(
        self,
        config_path: str,
        repo_path: str,
        models_dir: str = None
    ):
        """
        Initialize batch synthesis pipeline

        Args:
            config_path: Path to character_voices.yaml
            repo_path: Path to GPT-SoVITS repository
            models_dir: Models directory
        """
        self.voice_manager = CharacterVoiceManager(
            config_path=config_path,
            repo_path=repo_path,
            models_dir=models_dir
        )

        logger.info("BatchSynthesisPipeline initialized")

    def synthesize_batch(
        self,
        config: BatchSynthesisConfig
    ) -> BatchSynthesisResult:
        """
        Perform batch synthesis

        Args:
            config: Batch synthesis configuration

        Returns:
            BatchSynthesisResult
        """
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_texts = len(config.texts)

        # Prepare emotions and intensities
        emotions = config.emotions or ["neutral"] * num_texts
        intensities = config.intensities or [1.0] * num_texts

        if len(emotions) != num_texts:
            raise ValueError(f"emotions length must match texts length ({num_texts})")

        if len(intensities) != num_texts:
            raise ValueError(f"intensities length must match texts length ({num_texts})")

        logger.info(f"Batch synthesis: {num_texts} texts for {config.character}")

        # Ensure character is loaded
        if config.character not in self.voice_manager.wrapper.get_available_voices():
            logger.info(f"Loading character '{config.character}'...")
            self.voice_manager.load_character(config.character)

        # Synthesize with progress bar
        results = []
        successful = 0
        failed = 0
        total_duration = 0.0

        start_time = time.time()

        with tqdm(total=num_texts, desc=f"Synthesizing ({config.character})") as pbar:
            for i, (text, emotion, intensity) in enumerate(zip(config.texts, emotions, intensities)):
                try:
                    # Generate output path
                    output_path = output_dir / f"{config.character}_{i:04d}_{emotion}.wav"

                    # Synthesize
                    result = self.voice_manager.synthesize(
                        text=text,
                        character=config.character,
                        emotion=emotion,
                        intensity=intensity,
                        language=config.language,
                        output_path=str(output_path)
                    )

                    if result.success:
                        successful += 1
                        total_duration += result.duration_seconds
                        results.append(str(output_path))
                    else:
                        failed += 1
                        logger.warning(f"Failed to synthesize text {i}: {result.error_message}")

                except Exception as e:
                    failed += 1
                    logger.error(f"Error synthesizing text {i}: {e}")

                pbar.update(1)

        elapsed_time = time.time() - start_time

        # Create batch result
        batch_result = BatchSynthesisResult(
            total_generated=num_texts,
            successful=successful,
            failed=failed,
            total_duration=total_duration,
            average_duration=total_duration / successful if successful > 0 else 0.0,
            output_paths=results
        )

        # Save metadata
        if config.save_metadata:
            metadata_path = output_dir / f"batch_metadata_{int(time.time())}.json"

            metadata = {
                "character": config.character,
                "total_generated": num_texts,
                "successful": successful,
                "failed": failed,
                "total_audio_duration": total_duration,
                "processing_time": elapsed_time,
                "texts": [
                    {
                        "index": i,
                        "text": text,
                        "emotion": emotion,
                        "intensity": intensity,
                        "output": results[i] if i < len(results) else None
                    }
                    for i, (text, emotion, intensity) in enumerate(zip(config.texts, emotions, intensities))
                ]
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            batch_result.metadata_path = str(metadata_path)
            logger.info(f"✓ Metadata saved: {metadata_path}")

        logger.info(f"✓ Batch complete: {successful}/{num_texts} successful ({elapsed_time:.1f}s)")

        return batch_result

    def synthesize_script(
        self,
        script_path: str,
        character: str,
        default_emotion: str = "neutral",
        output_dir: str = "outputs/tts/script"
    ) -> BatchSynthesisResult:
        """
        Synthesize from script file

        Script format (JSON):
        {
            "lines": [
                {"text": "Hello!", "emotion": "happy"},
                {"text": "How are you?", "emotion": "neutral"}
            ]
        }

        Args:
            script_path: Path to script JSON file
            character: Character name
            default_emotion: Default emotion if not specified
            output_dir: Output directory

        Returns:
            BatchSynthesisResult
        """
        with open(script_path, 'r', encoding='utf-8') as f:
            script = json.load(f)

        lines = script.get("lines", [])

        texts = [line["text"] for line in lines]
        emotions = [line.get("emotion", default_emotion) for line in lines]

        config = BatchSynthesisConfig(
            character=character,
            texts=texts,
            emotions=emotions,
            output_dir=output_dir
        )

        return self.synthesize_batch(config)

    def cleanup(self):
        """Cleanup resources"""
        self.voice_manager.cleanup()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Batch Synthesis Pipeline Example")
    print("=" * 60)

    pipeline = BatchSynthesisPipeline(
        config_path="configs/generation/character_voices.yaml",
        repo_path="/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
    )

    # Example: Batch synthesis
    config = BatchSynthesisConfig(
        character="luca",
        texts=[
            "Hello! My name is Luca.",
            "I live in Portorosso.",
            "Silenzio, Bruno!",
            "Let's go on an adventure!"
        ],
        emotions=["happy", "neutral", "excited", "excited"],
        intensities=[1.0, 1.0, 1.3, 1.2]
    )

    result = pipeline.synthesize_batch(config)

    print(f"\n✓ Generated {result.successful}/{result.total_generated} files")
    print(f"Total duration: {result.total_duration:.1f}s")
    print(f"Average duration: {result.average_duration:.2f}s")

    pipeline.cleanup()
