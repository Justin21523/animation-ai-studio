#!/usr/bin/env python3
"""
Extract Voice Samples from Film Audio

Uses Whisper for transcription and Pyannote for speaker diarization
to extract clean voice samples for each character.

Usage:
    # Extract samples for Luca
    python scripts/synthesis/tts/extract_voice_samples.py \
        --audio data/films/luca/audio/luca_audio.wav \
        --output data/films/luca/voice_samples \
        --character Luca

    # Extract all characters
    python scripts/synthesis/tts/extract_voice_samples.py \
        --audio data/films/luca/audio/luca_audio.wav \
        --output data/films/luca/voice_samples \
        --all-characters

Requirements:
    pip install openai-whisper pyannote.audio torch torchaudio
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VoiceSegment:
    """Represents a voice segment"""
    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker: str  # Speaker ID
    text: str  # Transcribed text
    confidence: float  # Confidence score
    audio_path: Optional[str] = None  # Path to extracted audio file


class VoiceSampleExtractor:
    """Extract voice samples using Whisper + Pyannote"""

    def __init__(
        self,
        whisper_model: str = "medium",
        device: str = "cuda"
    ):
        """
        Initialize extractor

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu)
        """
        self.whisper_model_name = whisper_model
        self.device = device
        self.whisper_model = None
        self.diarization_pipeline = None

        logger.info(f"Initializing VoiceSampleExtractor")
        logger.info(f"  Whisper model: {whisper_model}")
        logger.info(f"  Device: {device}")

    def _load_whisper(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.whisper_model_name}...")
                # Use AI Warehouse for model storage
                whisper_cache = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/whisper")
                self.whisper_model = whisper.load_model(
                    self.whisper_model_name,
                    device=self.device,
                    download_root=str(whisper_cache)
                )
                logger.info("✓ Whisper model loaded")
            except ImportError:
                logger.error("openai-whisper not installed. Install with: pip install openai-whisper")
                raise
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                raise

    def _load_diarization(self):
        """Lazy load Pyannote diarization pipeline"""
        if self.diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                logger.info("Loading Pyannote diarization pipeline...")

                # Note: Requires HuggingFace token for pyannote models
                # Set: export HF_TOKEN=your_token
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True
                )

                if self.device == "cuda":
                    import torch
                    self.diarization_pipeline.to(torch.device("cuda"))

                logger.info("✓ Diarization pipeline loaded")
            except ImportError:
                logger.error("pyannote.audio not installed. Install with: pip install pyannote.audio")
                raise
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                logger.error("Make sure you have HuggingFace token set: export HF_TOKEN=your_token")
                logger.error("Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
                raise

    def transcribe_audio(self, audio_path: Path, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio using Whisper

        Args:
            audio_path: Path to audio file
            language: Language code (en, it, etc.)

        Returns:
            Transcription result with segments
        """
        self._load_whisper()

        logger.info(f"Transcribing audio: {audio_path}")
        logger.info(f"  Language: {language}")

        result = self.whisper_model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,  # Get word-level timestamps
            verbose=False
        )

        logger.info(f"✓ Transcription complete")
        logger.info(f"  Detected language: {result.get('language', 'unknown')}")
        logger.info(f"  Segments: {len(result.get('segments', []))}")

        return result

    def diarize_audio(self, audio_path: Path, num_speakers: Optional[int] = None) -> Any:
        """
        Perform speaker diarization using Pyannote

        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers (optional)

        Returns:
            Diarization result
        """
        self._load_diarization()

        logger.info(f"Performing speaker diarization: {audio_path}")
        if num_speakers:
            logger.info(f"  Expected speakers: {num_speakers}")

        # Run diarization
        diarization = self.diarization_pipeline(
            str(audio_path),
            num_speakers=num_speakers
        )

        # Count unique speakers
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)

        logger.info(f"✓ Diarization complete")
        logger.info(f"  Found {len(speakers)} speakers: {sorted(speakers)}")

        return diarization

    def align_transcription_with_speakers(
        self,
        transcription: Dict[str, Any],
        diarization: Any
    ) -> List[VoiceSegment]:
        """
        Align Whisper transcription with Pyannote speaker labels

        Args:
            transcription: Whisper transcription result
            diarization: Pyannote diarization result

        Returns:
            List of voice segments with speaker labels
        """
        logger.info("Aligning transcription with speakers...")

        segments = []

        for segment in transcription.get('segments', []):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()

            # Find speaker at this time
            # Use middle of segment for speaker detection
            mid_time = (start + end) / 2

            speaker = None
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    speaker = spk
                    break

            if speaker:
                voice_segment = VoiceSegment(
                    start=start,
                    end=end,
                    speaker=speaker,
                    text=text,
                    confidence=segment.get('no_speech_prob', 0.0)
                )
                segments.append(voice_segment)

        logger.info(f"✓ Aligned {len(segments)} segments")

        # Group by speaker
        speaker_counts = {}
        for seg in segments:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1

        for speaker, count in sorted(speaker_counts.items()):
            logger.info(f"  {speaker}: {count} segments")

        return segments

    def extract_audio_segments(
        self,
        audio_path: Path,
        segments: List[VoiceSegment],
        output_dir: Path,
        min_duration: float = 1.0,
        max_duration: float = 10.0
    ) -> List[VoiceSegment]:
        """
        Extract audio segments to individual files

        Args:
            audio_path: Source audio file
            segments: Voice segments to extract
            output_dir: Output directory
            min_duration: Minimum segment duration (seconds)
            max_duration: Maximum segment duration (seconds)

        Returns:
            Updated segments with audio_path filled
        """
        import subprocess

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting audio segments to: {output_dir}")

        extracted_segments = []

        for i, segment in enumerate(segments):
            duration = segment.end - segment.start

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                continue

            # Create filename
            speaker_dir = output_dir / segment.speaker
            speaker_dir.mkdir(exist_ok=True)

            filename = f"{segment.speaker}_{i:04d}_{segment.start:.2f}s.wav"
            output_file = speaker_dir / filename

            # Extract using ffmpeg
            cmd = [
                'ffmpeg',
                '-ss', str(segment.start),
                '-t', str(duration),
                '-i', str(audio_path),
                '-acodec', 'pcm_s16le',
                '-ar', '48000',
                '-ac', '1',  # Mono
                '-y',
                str(output_file)
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True)

                # Update segment
                segment.audio_path = str(output_file)
                extracted_segments.append(segment)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Extracted {i + 1}/{len(segments)} segments...")

            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to extract segment {i}: {e}")

        logger.info(f"✓ Extracted {len(extracted_segments)} audio segments")
        return extracted_segments

    def filter_by_quality(
        self,
        segments: List[VoiceSegment],
        min_confidence: float = 0.1,  # Lower is better for no_speech_prob
        min_words: int = 3
    ) -> List[VoiceSegment]:
        """
        Filter segments by quality metrics

        Args:
            segments: Voice segments
            min_confidence: Minimum confidence (max no_speech_prob)
            min_words: Minimum number of words

        Returns:
            Filtered segments
        """
        filtered = []

        for segment in segments:
            # Check word count
            word_count = len(segment.text.split())
            if word_count < min_words:
                continue

            # Check confidence (no_speech_prob should be low)
            if segment.confidence > min_confidence:
                continue

            filtered.append(segment)

        logger.info(f"Quality filtering: {len(segments)} → {len(filtered)} segments")
        return filtered

    def process_film(
        self,
        audio_path: Path,
        output_dir: Path,
        num_speakers: Optional[int] = None,
        language: str = "en"
    ) -> Dict[str, List[VoiceSegment]]:
        """
        Complete processing pipeline for a film

        Args:
            audio_path: Path to film audio
            output_dir: Output directory for voice samples
            num_speakers: Expected number of speakers
            language: Audio language

        Returns:
            Dictionary mapping speaker to segments
        """
        logger.info("="*60)
        logger.info("Voice Sample Extraction Pipeline")
        logger.info("="*60)

        # Step 1: Transcribe with Whisper
        logger.info("\n[1/5] Transcribing audio with Whisper...")
        transcription = self.transcribe_audio(audio_path, language=language)

        # Save transcription
        transcript_file = output_dir / "full_transcription.json"
        transcript_file.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Transcription saved: {transcript_file}")

        # Step 2: Speaker diarization with Pyannote
        logger.info("\n[2/5] Performing speaker diarization...")
        diarization = self.diarize_audio(audio_path, num_speakers=num_speakers)

        # Step 3: Align transcription with speakers
        logger.info("\n[3/5] Aligning transcription with speakers...")
        segments = self.align_transcription_with_speakers(transcription, diarization)

        # Step 4: Filter by quality
        logger.info("\n[4/5] Filtering by quality...")
        segments = self.filter_by_quality(segments)

        # Step 5: Extract audio segments
        logger.info("\n[5/5] Extracting audio segments...")
        segments = self.extract_audio_segments(
            audio_path=audio_path,
            segments=segments,
            output_dir=output_dir,
            min_duration=1.0,
            max_duration=10.0
        )

        # Group by speaker
        speaker_segments = {}
        for segment in segments:
            if segment.speaker not in speaker_segments:
                speaker_segments[segment.speaker] = []
            speaker_segments[segment.speaker].append(segment)

        # Save metadata
        metadata_file = output_dir / "segments_metadata.json"
        metadata = {
            speaker: [asdict(seg) for seg in segs]
            for speaker, segs in speaker_segments.items()
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Metadata saved: {metadata_file}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Extraction Complete!")
        logger.info("="*60)
        for speaker, segs in sorted(speaker_segments.items()):
            total_duration = sum(seg.end - seg.start for seg in segs)
            logger.info(f"{speaker}: {len(segs)} segments, {total_duration:.1f}s total")
        logger.info("="*60)

        return speaker_segments


def main():
    parser = argparse.ArgumentParser(description="Extract voice samples from film audio")
    parser.add_argument('--audio', '-a', required=True, help="Input audio file")
    parser.add_argument('--output', '-o', required=True, help="Output directory")
    parser.add_argument('--whisper-model', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help="Whisper model size")
    parser.add_argument('--language', default='en', help="Audio language (en, it, etc.)")
    parser.add_argument('--num-speakers', type=int, help="Expected number of speakers")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="Device to use")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.output)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    # Create extractor
    extractor = VoiceSampleExtractor(
        whisper_model=args.whisper_model,
        device=args.device
    )

    # Process
    try:
        speaker_segments = extractor.process_film(
            audio_path=audio_path,
            output_dir=output_dir,
            num_speakers=args.num_speakers,
            language=args.language
        )

        print(f"\n✓ Voice samples extracted successfully!")
        print(f"Output directory: {output_dir}")
        print(f"\nNext steps:")
        print(f"1. Review extracted samples in {output_dir}")
        print(f"2. Map speaker IDs to character names")
        print(f"3. Select best samples for training (1-5 minutes per character)")
        print(f"4. Run voice model training")

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
