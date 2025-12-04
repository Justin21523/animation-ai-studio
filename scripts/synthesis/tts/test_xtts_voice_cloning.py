#!/usr/bin/env python3
"""
Test XTTS-v2 Zero-Shot Voice Cloning for Luca Character

This script tests Coqui TTS (XTTS-v2) zero-shot voice cloning capabilities
using voice samples from the Luca character.
"""

import os
import sys
from pathlib import Path

# IMPORTANT: Patch torch.load for PyTorch 2.6+ compatibility with TTS 0.22.0
# TTS 0.22.0 models require weights_only=False
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for TTS compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torchaudio
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_xtts_voice_cloning(
    character_name: str = "Luca",
    reference_audio: str = None,
    test_text: str = None,
    output_dir: str = None,
    language: str = "en"
):
    """
    Test XTTS-v2 zero-shot voice cloning

    Args:
        character_name: Character name (default: Luca)
        reference_audio: Path to reference audio file (if None, uses first sample)
        test_text: Text to synthesize (if None, uses default)
        output_dir: Output directory (if None, uses default)
        language: Language code (default: en)
    """

    # Setup paths
    project_root = Path("/mnt/c/AI_LLM_projects/animation-ai-studio")
    samples_dir = project_root / f"data/films/{character_name.lower()}/voice_samples_auto/by_character/{character_name}"

    if output_dir is None:
        output_dir = project_root / f"outputs/tts/xtts_tests/{character_name.lower()}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference audio
    if reference_audio is None:
        # Find a good reference sample (prefer longer ones)
        samples = sorted(samples_dir.glob("*.wav"))
        if not samples:
            raise FileNotFoundError(f"No voice samples found in {samples_dir}")

        # Get file sizes and pick a medium-length one (not too short, not too long)
        sample_sizes = [(s, s.stat().st_size) for s in samples[:20]]  # Check first 20
        sample_sizes.sort(key=lambda x: x[1])
        reference_audio = sample_sizes[len(sample_sizes)//2][0]  # Pick median

        logger.info(f"Selected reference audio: {reference_audio.name}")
    else:
        reference_audio = Path(reference_audio)

    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    # Default test text
    if test_text is None:
        test_text = (
            "Hello! My name is Luca. "
            "I love spending summer days by the sea with my friends. "
            "The water is so beautiful and the town is amazing!"
        )

    logger.info("="*60)
    logger.info("XTTS-v2 Zero-Shot Voice Cloning Test")
    logger.info("="*60)
    logger.info(f"Character: {character_name}")
    logger.info(f"Reference: {reference_audio.name}")
    logger.info(f"Text: {test_text}")
    logger.info(f"Language: {language}")
    logger.info("")

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    try:
        # Import TTS
        from TTS.api import TTS

        # Set environment variable to accept license
        os.environ["COQUI_TOS_AGREED"] = "1"

        logger.info("\n" + "="*60)
        logger.info("Loading XTTS-v2 model...")
        logger.info("="*60)

        # Initialize TTS with XTTS-v2
        # This will download the model if not already cached
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

        logger.info("✓ Model loaded successfully")

        # Generate speech
        logger.info("\n" + "="*60)
        logger.info("Generating speech with cloned voice...")
        logger.info("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{character_name.lower()}_xtts_test_{timestamp}.wav"

        # Perform zero-shot voice cloning
        tts.tts_to_file(
            text=test_text,
            speaker_wav=str(reference_audio),
            language=language,
            file_path=str(output_file)
        )

        logger.info(f"✓ Speech generated successfully")
        logger.info(f"Output saved to: {output_file}")

        # Get output file info
        if output_file.exists():
            size_kb = output_file.stat().st_size / 1024
            logger.info(f"File size: {size_kb:.2f} KB")

            # Load and get duration
            waveform, sample_rate = torchaudio.load(str(output_file))
            duration = waveform.shape[1] / sample_rate
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Sample rate: {sample_rate} Hz")

        logger.info("\n" + "="*60)
        logger.info("✅ Test completed successfully!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Listen to the generated audio to verify quality")
        logger.info("2. Try different reference samples if needed")
        logger.info("3. Test with various texts")
        logger.info("4. Proceed with RVC training for voice conversion enhancement")

        return output_file

    except Exception as e:
        logger.error(f"❌ Error during voice cloning: {e}", exc_info=True)
        raise

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test XTTS-v2 voice cloning")
    parser.add_argument("--character", default="Luca", help="Character name")
    parser.add_argument("--reference", help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--language", default="en", help="Language code")

    args = parser.parse_args()

    test_xtts_voice_cloning(
        character_name=args.character,
        reference_audio=args.reference,
        test_text=args.text,
        output_dir=args.output_dir,
        language=args.language
    )

if __name__ == "__main__":
    main()
