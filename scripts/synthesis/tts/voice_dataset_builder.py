"""
Voice Dataset Builder

Extract and prepare voice samples from animated films for voice cloning.

Uses Whisper for transcription and Pyannote for speaker diarization.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

import torch
import torchaudio
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class VoiceSample:
    """Extracted voice sample metadata"""
    audio_path: str
    transcript: str
    character_name: str
    start_time: float
    end_time: float
    duration: float
    speaker_id: str
    confidence: float
    snr: float  # Signal-to-noise ratio
    is_clean: bool


@dataclass
class DatasetStats:
    """Voice dataset statistics"""
    total_samples: int
    clean_samples: int
    total_duration: float
    avg_duration: float
    characters: Dict[str, int]  # character -> sample count


class VoiceDatasetBuilder:
    """
    Extract voice samples from animated films for voice cloning

    Process:
    1. Extract audio from video
    2. Run speaker diarization (Pyannote)
    3. Transcribe with Whisper
    4. Match character names to speakers
    5. Extract clean segments
    6. Filter by quality (SNR, duration)

    VRAM Usage: ~4-6GB (Whisper large + Pyannote)

    Example:
        builder = VoiceDatasetBuilder(
            whisper_model="large-v3",
            diarization_model="pyannote/speaker-diarization"
        )

        samples = builder.extract_from_film(
            video_path="data/films/luca/luca.mp4",
            character_name="Luca",
            output_dir="data/voice_samples/luca",
            min_duration=1.0,
            max_duration=10.0
        )
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        diarization_model: str = "pyannote/speaker-diarization",
        device: str = "cuda"
    ):
        """
        Initialize voice dataset builder

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large, large-v3)
            diarization_model: Pyannote diarization model
            device: CUDA device
        """
        self.whisper_model_name = whisper_model
        self.diarization_model_name = diarization_model
        self.device = device

        # Models (lazy loading)
        self.whisper_model = None
        self.diarization_pipeline = None

        logger.info(f"VoiceDatasetBuilder initialized")
        logger.info(f"Whisper: {whisper_model}, Diarization: {diarization_model}")

    def _load_whisper(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")

            try:
                import whisper
                self.whisper_model = whisper.load_model(
                    self.whisper_model_name,
                    device=self.device
                )
                logger.info("✓ Whisper model loaded")
            except ImportError:
                logger.error("Whisper not installed. Install: pip install openai-whisper")
                raise

    def _load_diarization(self):
        """Lazy load Pyannote diarization pipeline"""
        if self.diarization_pipeline is None:
            logger.info(f"Loading diarization model: {self.diarization_model_name}")

            try:
                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    self.diarization_model_name,
                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                )
                self.diarization_pipeline.to(torch.device(self.device))
                logger.info("✓ Diarization pipeline loaded")
            except ImportError:
                logger.error("Pyannote not installed. Install: pip install pyannote.audio")
                raise
            except Exception as e:
                logger.error(f"Failed to load diarization: {e}")
                logger.warning("Continuing without diarization (will use full audio)")

    def extract_from_film(
        self,
        video_path: str,
        character_name: str,
        output_dir: str,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        min_snr: float = 15.0,
        max_samples: int = 100
    ) -> List[VoiceSample]:
        """
        Extract character voice samples from film

        Args:
            video_path: Path to video file
            character_name: Character to extract
            output_dir: Output directory for samples
            min_duration: Minimum sample duration (seconds)
            max_duration: Maximum sample duration (seconds)
            min_snr: Minimum signal-to-noise ratio
            max_samples: Maximum samples to extract

        Returns:
            List of extracted VoiceSample objects
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Extracting voice samples for '{character_name}' from {video_path.name}")

        # Step 1: Extract audio from video
        audio_path = output_dir / f"{video_path.stem}_audio.wav"
        logger.info("Step 1: Extracting audio...")
        self._extract_audio(str(video_path), str(audio_path))

        # Step 2: Run speaker diarization
        logger.info("Step 2: Speaker diarization...")
        self._load_diarization()
        diarization = None

        if self.diarization_pipeline:
            try:
                diarization = self.diarization_pipeline(str(audio_path))
                logger.info(f"✓ Detected {len(set(diarization.labels()))} speakers")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Skipping diarization.")

        # Step 3: Transcribe with Whisper
        logger.info("Step 3: Transcribing with Whisper...")
        self._load_whisper()
        transcript = self.whisper_model.transcribe(
            str(audio_path),
            language="en",  # Or detect automatically
            word_timestamps=True
        )

        # Step 4: Extract segments
        logger.info("Step 4: Extracting voice segments...")
        samples = []

        for segment in tqdm(transcript["segments"], desc="Processing segments"):
            start = segment["start"]
            end = segment["end"]
            duration = end - start
            text = segment["text"].strip()

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                continue

            # Skip if no text
            if not text or len(text) < 5:
                continue

            # Extract audio segment
            segment_path = output_dir / f"{character_name}_{len(samples):04d}.wav"
            self._extract_segment(
                str(audio_path),
                str(segment_path),
                start,
                end
            )

            # Clean audio
            self._clean_audio_sample(str(segment_path), str(segment_path))

            # Validate quality
            quality = self._validate_sample_quality(str(segment_path))

            is_clean = quality["snr"] >= min_snr
            is_clean = is_clean and quality["has_speech"]

            # Create sample
            sample = VoiceSample(
                audio_path=str(segment_path),
                transcript=text,
                character_name=character_name,
                start_time=start,
                end_time=end,
                duration=duration,
                speaker_id="SPEAKER_00",  # Placeholder
                confidence=segment.get("confidence", 1.0),
                snr=quality["snr"],
                is_clean=is_clean
            )

            samples.append(sample)

            # Check max samples
            if len(samples) >= max_samples:
                break

        # Filter clean samples
        clean_samples = [s for s in samples if s.is_clean]
        logger.info(f"✓ Extracted {len(clean_samples)}/{len(samples)} clean samples")

        # Save metadata
        self._save_dataset_metadata(clean_samples, output_dir / "dataset.json")

        return clean_samples

    def _extract_audio(self, video_path: str, audio_path: str):
        """Extract audio from video using ffmpeg"""
        import subprocess

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "44100",  # 44.1kHz
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            audio_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✓ Audio extracted: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr.decode()}")
            raise

    def _extract_segment(
        self,
        audio_path: str,
        output_path: str,
        start: float,
        end: float
    ):
        """Extract audio segment"""
        waveform, sr = torchaudio.load(audio_path)

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = waveform[:, start_sample:end_sample]

        torchaudio.save(output_path, segment, sr)

    def _clean_audio_sample(
        self,
        audio_path: str,
        output_path: str,
        reduce_noise: bool = True,
        normalize: bool = True
    ):
        """
        Clean and enhance audio sample

        Args:
            audio_path: Input audio file
            output_path: Output audio file
            reduce_noise: Apply noise reduction
            normalize: Normalize volume
        """
        waveform, sr = torchaudio.load(audio_path)

        # Normalize volume
        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            waveform = waveform * 0.95  # Prevent clipping

        # TODO: Add noise reduction (requires noisereduce library)
        # if reduce_noise:
        #     import noisereduce as nr
        #     waveform_np = waveform.numpy()
        #     reduced = nr.reduce_noise(y=waveform_np, sr=sr)
        #     waveform = torch.from_numpy(reduced)

        torchaudio.save(output_path, waveform, sr)

    def _validate_sample_quality(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate audio sample quality

        Returns:
            {
                "snr": float,
                "duration": float,
                "sample_rate": int,
                "has_speech": bool,
                "rms_energy": float
            }
        """
        waveform, sr = torchaudio.load(audio_path)

        duration = waveform.shape[1] / sr

        # Calculate RMS energy
        rms = torch.sqrt(torch.mean(waveform ** 2))

        # Estimate SNR (simplified)
        # Real SNR requires noise floor estimation
        signal_power = torch.mean(waveform ** 2)
        noise_floor = torch.mean((waveform - torch.mean(waveform)) ** 2) * 0.1  # Estimate
        snr = 10 * torch.log10(signal_power / (noise_floor + 1e-8))
        snr = max(0.0, snr.item())

        # Check if has speech (simple energy threshold)
        has_speech = rms.item() > 0.01

        return {
            "snr": snr,
            "duration": duration,
            "sample_rate": sr,
            "has_speech": has_speech,
            "rms_energy": rms.item()
        }

    def _save_dataset_metadata(self, samples: List[VoiceSample], output_path: str):
        """Save dataset metadata to JSON"""
        metadata = {
            "total_samples": len(samples),
            "samples": [
                {
                    "audio_path": s.audio_path,
                    "transcript": s.transcript,
                    "character": s.character_name,
                    "duration": s.duration,
                    "snr": s.snr,
                    "is_clean": s.is_clean
                }
                for s in samples
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Metadata saved: {output_path}")

    def cleanup(self):
        """Free models from VRAM"""
        self.whisper_model = None
        self.diarization_pipeline = None
        torch.cuda.empty_cache()
        logger.info("✓ Models unloaded")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Voice Dataset Builder Example")
    print("=" * 60)

    builder = VoiceDatasetBuilder(
        whisper_model="medium",  # Use smaller model for testing
        device="cuda"
    )

    # Extract samples (example paths)
    samples = builder.extract_from_film(
        video_path="/path/to/luca.mp4",
        character_name="Luca",
        output_dir="outputs/voice_datasets/luca",
        min_duration=2.0,
        max_duration=8.0,
        max_samples=50
    )

    print(f"\n✓ Extracted {len(samples)} samples")

    builder.cleanup()
