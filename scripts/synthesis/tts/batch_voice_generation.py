#!/usr/bin/env python3
"""
Batch Voice Generation Script for Animation Production

Generates multiple voice samples from text files using Enhanced XTTS-v2.
Designed for production workflow integration.

Features:
- Batch processing from text files or CSV
- Multiple reference samples for quality
- Progress tracking and logging
- Automatic file naming and organization
- Quality metrics reporting

Usage:
    # Single text file
    python batch_voice_generation.py --input dialogue.txt --character Luca

    # CSV with multiple lines
    python batch_voice_generation.py --input script.csv --character Luca

    # Custom parameters
    python batch_voice_generation.py --input script.csv --character Luca \
        --num-refs 5 --temperature 0.65 --top-k 40 --top-p 0.90
"""

import os
import sys
from pathlib import Path
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

# IMPORTANT: Patch torch.load for PyTorch 2.6+ compatibility
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torchaudio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchVoiceGenerator:
    """Batch voice generation using Enhanced XTTS-v2"""

    def __init__(
        self,
        character_name: str,
        output_base_dir: str,
        num_references: int = 5,
        temperature: float = 0.65,
        top_k: int = 40,
        top_p: float = 0.90,
        language: str = "en"
    ):
        """
        Initialize batch voice generator

        Args:
            character_name: Character name for voice samples
            output_base_dir: Base output directory
            num_references: Number of reference samples
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            language: Language code
        """
        self.character_name = character_name
        self.output_base_dir = Path(output_base_dir)
        self.num_references = num_references
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.language = language

        # Setup paths
        self.project_root = Path("/mnt/c/AI_LLM_projects/animation-ai-studio")
        self.samples_dir = self.project_root / f"data/films/{character_name.lower()}/voice_samples_auto/by_character/{character_name}"

        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TTS model
        self._init_model()

        # Results tracking
        self.results = []

    def _init_model(self):
        """Initialize XTTS-v2 model"""
        from TTS.api import TTS

        os.environ["COQUI_TOS_AGREED"] = "1"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing XTTS-v2 model on {device}...")

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        logger.info("✓ Model loaded successfully")

        # Select reference samples
        self.reference_samples = self._select_reference_samples()
        logger.info(f"✓ Selected {len(self.reference_samples)} reference samples")

    def _select_reference_samples(self, min_duration: float = 3.0) -> List[Path]:
        """Select best reference samples"""
        samples = list(self.samples_dir.glob("*.wav"))

        valid_samples = []
        for sample in samples:
            try:
                waveform, sr = torchaudio.load(str(sample))
                duration = waveform.shape[1] / sr
                if duration >= min_duration and duration <= 10.0:
                    valid_samples.append((sample, duration))
            except Exception as e:
                logger.warning(f"Skipping {sample.name}: {e}")

        # Sort by duration (prefer medium-length samples)
        valid_samples.sort(key=lambda x: abs(x[1] - 5.0))

        # Select top N samples
        selected = [s[0] for s in valid_samples[:self.num_references]]

        return selected

    def generate_from_text_file(self, text_file: Path) -> List[Dict]:
        """
        Generate voice from plain text file

        Each line is treated as a separate generation task.
        Empty lines are skipped.
        Lines starting with # are treated as comments.

        Args:
            text_file: Path to text file

        Returns:
            List of generation results
        """
        logger.info(f"Reading text file: {text_file}")

        tasks = []
        with open(text_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                tasks.append({
                    'id': f"line_{line_num:03d}",
                    'text': line,
                    'metadata': {'source': str(text_file), 'line': line_num}
                })

        logger.info(f"Found {len(tasks)} generation tasks")
        return self._process_tasks(tasks)

    def generate_from_csv(self, csv_file: Path) -> List[Dict]:
        """
        Generate voice from CSV file

        CSV format:
            id,text,character,notes
            line_001,"Hello world",Luca,"Happy greeting"

        Args:
            csv_file: Path to CSV file

        Returns:
            List of generation results
        """
        logger.info(f"Reading CSV file: {csv_file}")

        tasks = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip if character doesn't match (for multi-character scripts)
                if 'character' in row and row['character'] != self.character_name:
                    continue

                tasks.append({
                    'id': row.get('id', f"row_{len(tasks)+1:03d}"),
                    'text': row['text'],
                    'metadata': row
                })

        logger.info(f"Found {len(tasks)} generation tasks for {self.character_name}")
        return self._process_tasks(tasks)

    def _process_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Process generation tasks"""
        results = []

        logger.info("=" * 70)
        logger.info(f"Starting batch generation: {len(tasks)} tasks")
        logger.info("=" * 70)

        for task in tqdm(tasks, desc="Generating voices"):
            task_id = task['id']
            text = task['text']

            logger.info(f"\nProcessing: {task_id}")
            logger.info(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")

            # Create task output directory
            task_output_dir = self.output_base_dir / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate with each reference
            variants = []
            for i, ref_sample in enumerate(self.reference_samples, 1):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = task_output_dir / f"{self.character_name.lower()}_variant{i}_{timestamp}.wav"

                    # Generate speech
                    start_time = datetime.now()
                    self.tts.tts_to_file(
                        text=text,
                        speaker_wav=str(ref_sample),
                        language=self.language,
                        file_path=str(output_file),
                        temperature=self.temperature,
                        repetition_penalty=5.0,
                        top_k=self.top_k,
                        top_p=self.top_p
                    )
                    generation_time = (datetime.now() - start_time).total_seconds()

                    # Get audio info
                    waveform, sr = torchaudio.load(str(output_file))
                    duration = waveform.shape[1] / sr
                    size_mb = output_file.stat().st_size / (1024 * 1024)

                    variant_info = {
                        'variant': i,
                        'file': str(output_file),
                        'duration': duration,
                        'size_mb': size_mb,
                        'generation_time': generation_time,
                        'real_time_factor': generation_time / duration if duration > 0 else 0,
                        'reference': ref_sample.name
                    }
                    variants.append(variant_info)

                except Exception as e:
                    logger.error(f"Failed to generate variant {i}: {e}")
                    variants.append({
                        'variant': i,
                        'error': str(e),
                        'reference': ref_sample.name
                    })

            # Save task result
            result = {
                'id': task_id,
                'text': text,
                'character': self.character_name,
                'timestamp': datetime.now().isoformat(),
                'variants': variants,
                'parameters': {
                    'num_references': self.num_references,
                    'temperature': self.temperature,
                    'top_k': self.top_k,
                    'top_p': self.top_p,
                    'language': self.language
                },
                'metadata': task.get('metadata', {})
            }
            results.append(result)

            # Save individual task result
            result_file = task_output_dir / "generation_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Generated {len(variants)} variants for {task_id}")

        # Save batch results
        self._save_batch_results(results)

        return results

    def _save_batch_results(self, results: List[Dict]):
        """Save batch generation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_base_dir / f"batch_results_{timestamp}.json"

        # Calculate statistics
        total_variants = sum(len(r['variants']) for r in results)
        successful_variants = sum(
            len([v for v in r['variants'] if 'error' not in v])
            for r in results
        )

        total_duration = sum(
            v['duration']
            for r in results
            for v in r['variants']
            if 'duration' in v
        )

        batch_summary = {
            'character': self.character_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_tasks': len(results),
                'total_variants': total_variants,
                'successful_variants': successful_variants,
                'failed_variants': total_variants - successful_variants,
                'total_audio_duration': total_duration,
                'success_rate': successful_variants / total_variants if total_variants > 0 else 0
            },
            'parameters': {
                'num_references': self.num_references,
                'temperature': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'language': self.language
            },
            'results': results
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)

        logger.info("\n" + "=" * 70)
        logger.info("Batch Generation Complete!")
        logger.info("=" * 70)
        logger.info(f"Total tasks: {len(results)}")
        logger.info(f"Total variants: {total_variants}")
        logger.info(f"Successful: {successful_variants}")
        logger.info(f"Failed: {total_variants - successful_variants}")
        logger.info(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        logger.info(f"Results saved to: {result_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch voice generation for animation production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from text file (one line per generation)
    python batch_voice_generation.py --input script.txt --character Luca

    # Generate from CSV file
    python batch_voice_generation.py --input dialogue.csv --character Luca

    # Custom quality parameters
    python batch_voice_generation.py --input script.txt --character Luca \\
        --num-refs 5 --temperature 0.65 --top-k 40 --top-p 0.90
        """
    )

    parser.add_argument("--input", required=True, help="Input file (txt or csv)")
    parser.add_argument("--character", required=True, help="Character name")
    parser.add_argument("--output-dir", help="Output directory (default: outputs/tts/batch/<timestamp>)")
    parser.add_argument("--num-refs", type=int, default=5, help="Number of reference samples (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.65, help="Sampling temperature (default: 0.65)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (default: 40)")
    parser.add_argument("--top-p", type=float, default=0.90, help="Nucleus sampling (default: 0.90)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/tts/batch/{args.character.lower()}_{timestamp}")

    # Initialize generator
    generator = BatchVoiceGenerator(
        character_name=args.character,
        output_base_dir=str(output_dir),
        num_references=args.num_refs,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        language=args.language
    )

    # Process input file
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    if input_file.suffix.lower() == '.csv':
        results = generator.generate_from_csv(input_file)
    else:
        results = generator.generate_from_text_file(input_file)

    logger.info(f"\n✅ Batch generation complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
