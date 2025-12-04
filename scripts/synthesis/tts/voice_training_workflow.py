#!/usr/bin/env python3
"""
Complete Voice Training Workflow

End-to-end pipeline for extracting and training character voices from films.

Usage:
    # Complete workflow for Luca
    python scripts/synthesis/tts/voice_training_workflow.py \
        --film luca \
        --characters Luca Alberto Giulia

    # Custom workflow
    python scripts/synthesis/tts/voice_training_workflow.py \
        --video /path/to/video.mp4 \
        --output ./voice_training \
        --characters Character1 Character2

Workflow Steps:
    1. Extract audio from video
    2. Transcribe with Whisper
    3. Diarize speakers with Pyannote
    4. Extract voice samples for each character
    5. Validate sample quality
    6. (Manual) Map speakers to characters
    7. Train GPT-SoVITS models
    8. Test synthesis

Author: Animation AI Studio
Date: 2025-11-19
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceTrainingWorkflow:
    """Complete voice training workflow"""

    def __init__(
        self,
        film_name: Optional[str] = None,
        video_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize workflow

        Args:
            film_name: Name of film (e.g., 'luca')
            video_path: Path to video file (alternative to film_name)
            output_dir: Output directory
        """
        self.film_name = film_name
        self.video_path = video_path
        self.project_root = Path(__file__).parent.parent.parent.parent

        if film_name:
            self.raw_videos_dir = Path("/mnt/c/raw_videos")
            self.film_data_dir = self.project_root / "data" / "films" / film_name
            self.output_dir = output_dir or self.film_data_dir
        else:
            self.output_dir = output_dir or Path("./voice_training")

        self.audio_file = None
        self.voice_samples_dir = None
        self.speaker_mapping = {}

    def step1_extract_audio(self) -> bool:
        """
        Step 1: Extract audio from video

        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Extract Audio from Video")
        logger.info("="*60)

        from scripts.synthesis.tts.extract_audio import AudioExtractor

        extractor = AudioExtractor()

        if self.film_name:
            # Extract from film
            self.audio_file = extractor.extract_for_film(
                film_name=self.film_name,
                output_dir=self.output_dir / "audio"
            )
        elif self.video_path:
            # Extract from specific video
            self.audio_file = self.output_dir / "audio" / "audio.wav"
            success = extractor.extract_audio(
                video_path=self.video_path,
                output_path=self.audio_file,
                sample_rate=48000,
                channels=2
            )
            if not success:
                self.audio_file = None

        if self.audio_file and self.audio_file.exists():
            logger.info(f"✓ Audio extracted: {self.audio_file}")
            logger.info(f"  Size: {self.audio_file.stat().st_size / (1024**2):.2f} MB")
            return True
        else:
            logger.error("✗ Audio extraction failed")
            return False

    def step2_extract_voice_samples(
        self,
        num_speakers: Optional[int] = None,
        language: str = "en"
    ) -> bool:
        """
        Step 2: Extract voice samples with Whisper + Pyannote

        Args:
            num_speakers: Expected number of speakers
            language: Audio language

        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Extract Voice Samples (Whisper + Pyannote)")
        logger.info("="*60)

        if not self.audio_file or not self.audio_file.exists():
            logger.error("Audio file not available. Run step 1 first.")
            return False

        from scripts.synthesis.tts.extract_voice_samples import VoiceSampleExtractor

        self.voice_samples_dir = self.output_dir / "voice_samples"

        extractor = VoiceSampleExtractor(
            whisper_model="medium",
            device="cuda"
        )

        try:
            speaker_segments = extractor.process_film(
                audio_path=self.audio_file,
                output_dir=self.voice_samples_dir,
                num_speakers=num_speakers,
                language=language
            )

            logger.info(f"✓ Voice samples extracted: {self.voice_samples_dir}")
            return True

        except Exception as e:
            logger.error(f"✗ Voice sample extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step3_interactive_speaker_mapping(self, character_names: List[str]) -> bool:
        """
        Step 3: Interactive speaker to character mapping

        Args:
            character_names: List of character names

        Returns:
            True if mapping complete
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Map Speakers to Characters")
        logger.info("="*60)

        if not self.voice_samples_dir or not self.voice_samples_dir.exists():
            logger.error("Voice samples directory not available. Run step 2 first.")
            return False

        # Load metadata
        metadata_file = self.voice_samples_dir / "segments_metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        speakers = list(metadata.keys())

        logger.info(f"Found {len(speakers)} speakers: {speakers}")
        logger.info(f"Target characters: {character_names}")

        # Interactive mapping
        print("\n" + "="*60)
        print("Speaker to Character Mapping")
        print("="*60)

        for speaker in speakers:
            num_samples = len(metadata[speaker])
            total_duration = sum(seg['end'] - seg['start'] for seg in metadata[speaker])

            print(f"\nSpeaker: {speaker}")
            print(f"  Samples: {num_samples}")
            print(f"  Total duration: {total_duration:.1f}s")

            # Show sample texts
            sample_texts = [seg['text'] for seg in metadata[speaker][:3]]
            print(f"  Sample texts:")
            for i, text in enumerate(sample_texts, 1):
                print(f"    {i}. {text}")

            # Ask for character name
            print(f"\nAvailable characters: {', '.join(character_names)}")
            character = input(f"Map '{speaker}' to character (or 'skip'): ").strip()

            if character.lower() != 'skip' and character in character_names:
                self.speaker_mapping[speaker] = character
                logger.info(f"  Mapped: {speaker} → {character}")
            else:
                logger.info(f"  Skipped: {speaker}")

        # Save mapping
        mapping_file = self.voice_samples_dir / "speaker_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(self.speaker_mapping, f, indent=2)

        logger.info(f"\n✓ Speaker mapping saved: {mapping_file}")
        logger.info(f"  Mappings: {self.speaker_mapping}")

        return len(self.speaker_mapping) > 0

    def step4_organize_samples_by_character(self) -> bool:
        """
        Step 4: Reorganize samples by character name

        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Organize Samples by Character")
        logger.info("="*60)

        if not self.speaker_mapping:
            # Try to load existing mapping
            mapping_file = self.voice_samples_dir / "speaker_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    self.speaker_mapping = json.load(f)
            else:
                logger.error("No speaker mapping available. Run step 3 first.")
                return False

        # Create character directories
        for speaker, character in self.speaker_mapping.items():
            speaker_dir = self.voice_samples_dir / speaker
            character_dir = self.voice_samples_dir / "by_character" / character

            if not speaker_dir.exists():
                logger.warning(f"Speaker directory not found: {speaker_dir}")
                continue

            character_dir.mkdir(parents=True, exist_ok=True)

            # Copy audio files
            import shutil
            audio_files = list(speaker_dir.glob("*.wav"))

            for audio_file in audio_files:
                dest = character_dir / audio_file.name
                if not dest.exists():
                    shutil.copy2(audio_file, dest)

            logger.info(f"✓ {character}: {len(audio_files)} samples → {character_dir}")

        return True

    def step5_generate_training_dataset(self) -> bool:
        """
        Step 5: Generate training dataset format for GPT-SoVITS

        Returns:
            True if successful
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Generate Training Dataset")
        logger.info("="*60)

        by_character_dir = self.voice_samples_dir / "by_character"
        if not by_character_dir.exists():
            logger.error("Character samples not organized. Run step 4 first.")
            return False

        # Load segments metadata
        metadata_file = self.voice_samples_dir / "segments_metadata.json"
        with open(metadata_file, 'r') as f:
            segments_metadata = json.load(f)

        # For each character, create training filelist
        for character_dir in by_character_dir.iterdir():
            if not character_dir.is_dir():
                continue

            character = character_dir.name
            logger.info(f"\nProcessing character: {character}")

            # Get audio files
            audio_files = sorted(character_dir.glob("*.wav"))

            # Create filelist with transcriptions
            filelist = []
            for audio_file in audio_files:
                # Extract speaker and segment info from filename
                # Format: SPEAKER_XXXX_YYYYs.wav
                parts = audio_file.stem.split('_')
                if len(parts) >= 3:
                    speaker = parts[0]
                    seg_idx = int(parts[1])

                    # Find transcription
                    if speaker in segments_metadata and seg_idx < len(segments_metadata[speaker]):
                        text = segments_metadata[speaker][seg_idx]['text']
                        filelist.append({
                            'audio_path': str(audio_file.relative_to(self.voice_samples_dir)),
                            'text': text,
                            'speaker': character
                        })

            # Save filelist
            filelist_path = character_dir / "training_filelist.json"
            with open(filelist_path, 'w', encoding='utf-8') as f:
                json.dump(filelist, f, indent=2, ensure_ascii=False)

            logger.info(f"  ✓ Training filelist: {filelist_path}")
            logger.info(f"    Samples: {len(filelist)}")

        return True

    def generate_summary_report(self) -> str:
        """
        Generate workflow summary report

        Returns:
            Report string
        """
        lines = []
        lines.append("\n" + "="*60)
        lines.append("Voice Training Workflow Summary")
        lines.append("="*60)
        lines.append(f"Film: {self.film_name or 'Custom'}")
        lines.append(f"Output directory: {self.output_dir}")
        lines.append("")

        if self.audio_file:
            lines.append(f"Audio file: {self.audio_file}")
            if self.audio_file.exists():
                size_mb = self.audio_file.stat().st_size / (1024**2)
                lines.append(f"  Size: {size_mb:.2f} MB")

        if self.voice_samples_dir:
            lines.append(f"\nVoice samples: {self.voice_samples_dir}")

            by_character_dir = self.voice_samples_dir / "by_character"
            if by_character_dir.exists():
                lines.append("\nCharacter samples:")
                for char_dir in sorted(by_character_dir.iterdir()):
                    if char_dir.is_dir():
                        samples = list(char_dir.glob("*.wav"))
                        total_duration = 0  # Would need to calculate from audio files
                        lines.append(f"  {char_dir.name}: {len(samples)} samples")

        if self.speaker_mapping:
            lines.append("\nSpeaker mapping:")
            for speaker, character in self.speaker_mapping.items():
                lines.append(f"  {speaker} → {character}")

        lines.append("\nNext steps:")
        lines.append("1. Review extracted voice samples")
        lines.append("2. Train GPT-SoVITS models for each character")
        lines.append("3. Test voice synthesis quality")
        lines.append("="*60)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Complete voice training workflow")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--film', '-f', type=str, help="Film name (e.g., luca)")
    input_group.add_argument('--video', '-v', type=str, help="Path to video file")

    # Workflow options
    parser.add_argument('--output', '-o', type=str, help="Output directory")
    parser.add_argument('--characters', '-c', nargs='+', help="Character names to extract")
    parser.add_argument('--num-speakers', type=int, help="Expected number of speakers")
    parser.add_argument('--language', default='en', help="Audio language (en, it, etc.)")

    # Step control
    parser.add_argument('--start-step', type=int, default=1, help="Start from step N")
    parser.add_argument('--end-step', type=int, default=5, help="End at step N")
    parser.add_argument('--skip-interactive', action='store_true', help="Skip interactive speaker mapping")

    args = parser.parse_args()

    # Initialize workflow
    workflow = VoiceTrainingWorkflow(
        film_name=args.film,
        video_path=Path(args.video) if args.video else None,
        output_dir=Path(args.output) if args.output else None
    )

    # Run workflow steps
    success = True

    if args.start_step <= 1 <= args.end_step:
        if not workflow.step1_extract_audio():
            logger.error("Step 1 failed")
            success = False

    if success and args.start_step <= 2 <= args.end_step:
        if not workflow.step2_extract_voice_samples(
            num_speakers=args.num_speakers,
            language=args.language
        ):
            logger.error("Step 2 failed")
            success = False

    if success and args.start_step <= 3 <= args.end_step:
        if not args.skip_interactive:
            if args.characters:
                if not workflow.step3_interactive_speaker_mapping(args.characters):
                    logger.error("Step 3 failed")
                    success = False
            else:
                logger.warning("Skipping step 3: No character names provided")
        else:
            logger.info("Skipping interactive mapping (--skip-interactive)")

    if success and args.start_step <= 4 <= args.end_step:
        if not workflow.step4_organize_samples_by_character():
            logger.error("Step 4 failed")
            success = False

    if success and args.start_step <= 5 <= args.end_step:
        if not workflow.step5_generate_training_dataset():
            logger.error("Step 5 failed")
            success = False

    # Generate summary
    print(workflow.generate_summary_report())

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
