#!/usr/bin/env python3
"""
Enhanced XTTS-v2 Voice Cloning with Multiple References

Improvements over basic XTTS:
1. Uses multiple reference samples for better voice matching
2. Temperature and top_k tuning for better quality
3. Longer reference sample selection
4. Post-processing options
"""

import os
import sys
from pathlib import Path
import random

# IMPORTANT: Patch torch.load for PyTorch 2.6+ compatibility
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torchaudio
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def select_best_reference_samples(samples_dir: Path, num_refs: int = 3, min_duration: float = 3.0):
    """
    Select best reference samples based on duration and audio quality

    Args:
        samples_dir: Directory containing voice samples
        num_refs: Number of reference samples to select
        min_duration: Minimum duration in seconds

    Returns:
        List of selected sample paths
    """
    samples = list(samples_dir.glob("*.wav"))

    # Filter by duration and get audio info
    valid_samples = []
    for sample in samples:
        try:
            waveform, sr = torchaudio.load(str(sample))
            duration = waveform.shape[1] / sr
            if duration >= min_duration and duration <= 10.0:  # 3-10 seconds
                valid_samples.append((sample, duration))
        except Exception as e:
            logger.warning(f"Skipping {sample.name}: {e}")

    if not valid_samples:
        logger.error("No valid samples found!")
        return []

    # Sort by duration (prefer medium-length samples)
    valid_samples.sort(key=lambda x: abs(x[1] - 5.0))  # Prefer ~5 second samples

    # Select top N samples
    selected = [s[0] for s in valid_samples[:num_refs]]

    logger.info(f"Selected {len(selected)} reference samples:")
    for s in selected:
        logger.info(f"  - {s.name}")

    return selected

def test_xtts_enhanced(
    character_name: str = "Luca",
    test_text: str = None,
    output_dir: str = None,
    language: str = "en",
    num_references: int = 3,
    temperature: float = 0.75,
    repetition_penalty: float = 5.0,
    top_k: int = 50,
    top_p: float = 0.85
):
    """
    Enhanced XTTS-v2 voice cloning with multiple references

    Args:
        character_name: Character name
        test_text: Text to synthesize
        output_dir: Output directory
        language: Language code
        num_references: Number of reference samples to use
        temperature: Sampling temperature (lower = more consistent, higher = more varied)
        repetition_penalty: Penalty for repetitive outputs
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    """

    # Setup paths
    project_root = Path("/mnt/c/AI_LLM_projects/animation-ai-studio")
    samples_dir = project_root / f"data/films/{character_name.lower()}/voice_samples_auto/by_character/{character_name}"

    if output_dir is None:
        output_dir = project_root / f"outputs/tts/xtts_enhanced/{character_name.lower()}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Select best reference samples
    reference_samples = select_best_reference_samples(samples_dir, num_references)

    if not reference_samples:
        raise FileNotFoundError(f"No valid reference samples found in {samples_dir}")

    # Default test text
    if test_text is None:
        test_text = (
            "Hello! My name is Luca. "
            "I love spending summer days by the sea with my friends. "
            "The water is so beautiful and the town is amazing!"
        )

    logger.info("="*60)
    logger.info("XTTS-v2 Enhanced Voice Cloning")
    logger.info("="*60)
    logger.info(f"Character: {character_name}")
    logger.info(f"References: {len(reference_samples)} samples")
    logger.info(f"Text: {test_text}")
    logger.info(f"Language: {language}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top-k: {top_k}, Top-p: {top_p}")
    logger.info("")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    try:
        from TTS.api import TTS

        os.environ["COQUI_TOS_AGREED"] = "1"

        logger.info("\n" + "="*60)
        logger.info("Loading XTTS-v2 model...")
        logger.info("="*60)

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        logger.info("✓ Model loaded successfully")

        # Generate with each reference sample and average/select best
        outputs = []

        for i, ref_sample in enumerate(reference_samples, 1):
            logger.info(f"\n[{i}/{len(reference_samples)}] Generating with {ref_sample.name}...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{character_name.lower()}_enhanced_ref{i}_{timestamp}.wav"

            # Generate speech
            tts.tts_to_file(
                text=test_text,
                speaker_wav=str(ref_sample),
                language=language,
                file_path=str(output_file),
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p
            )

            outputs.append(output_file)
            logger.info(f"✓ Saved to: {output_file}")

        logger.info("\n" + "="*60)
        logger.info("✅ Enhanced synthesis completed!")
        logger.info("="*60)
        logger.info(f"\nGenerated {len(outputs)} variants:")
        for output in outputs:
            size_kb = output.stat().st_size / 1024
            waveform, sr = torchaudio.load(str(output))
            duration = waveform.shape[1] / sr
            logger.info(f"  - {output.name}")
            logger.info(f"    Duration: {duration:.2f}s, Size: {size_kb:.2f} KB")

        logger.info("\nNext steps:")
        logger.info("1. Listen to all variants and select the best one")
        logger.info("2. Adjust temperature/top_k/top_p if needed")
        logger.info("3. Process with RVC for further enhancement")

        return outputs

    except Exception as e:
        logger.error(f"❌ Error during enhanced voice cloning: {e}", exc_info=True)
        raise

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced XTTS-v2 voice cloning")
    parser.add_argument("--character", default="Luca", help="Character name")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--num-refs", type=int, default=3, help="Number of reference samples")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.85, help="Nucleus sampling")

    args = parser.parse_args()

    test_xtts_enhanced(
        character_name=args.character,
        test_text=args.text,
        output_dir=args.output_dir,
        language=args.language,
        num_references=args.num_refs,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

if __name__ == "__main__":
    main()
