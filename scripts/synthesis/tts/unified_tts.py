"""
Unified TTS adapter used by ModelManager.

Goal: provide a stable `synthesize(...)` API for the agent/tooling layer without
hard-coupling to a single TTS backend.

Backend selection (best-effort):
1) Coqui TTS XTTS-v2 (if `TTS` package is available)
2) Fallback: generate silent WAV (always available)
"""

from __future__ import annotations

import os
import wave
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SimpleSynthesisResult:
    audio_path: str
    sample_rate: int
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


class UnifiedTTS:
    """
    Unified TTS adapter with a small, stable surface area:

    - synthesize(text, character, emotion, intensity, output_path) -> SimpleSynthesisResult
    """

    def __init__(
        self,
        language: str = "en",
        prefer_device: Optional[str] = None,
    ):
        self.language = language
        self.prefer_device = prefer_device

        self._backend: Optional[str] = None
        self._xtts = None
        self._xtts_device: Optional[str] = None

    def cleanup(self):
        self._xtts = None

    def synthesize(
        self,
        *,
        text: str,
        character: str,
        emotion: str = "neutral",
        intensity: float = 1.0,
        output_path: Optional[str] = None,
    ) -> SimpleSynthesisResult:
        if not text:
            return SimpleSynthesisResult(
                audio_path="",
                sample_rate=22050,
                duration_seconds=0.0,
                success=False,
                error_message="Empty text",
            )

        output_path = output_path or self._default_output_path(character, emotion)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer XTTS if installed, else fallback to silence WAV.
        if self._backend is None:
            self._backend = "xtts" if self._can_use_xtts() else "silence"

        try:
            if self._backend == "xtts":
                return self._synthesize_xtts(text=text, character=character, output_path=out_path)
        except Exception as e:
            logger.warning(f"XTTS synthesis failed; falling back to silent audio: {e}")
            self._backend = "silence"

        return self._synthesize_silence(text=text, output_path=out_path)

    def _default_output_path(self, character: str, emotion: str) -> str:
        safe_char = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in character)[:60]
        safe_emotion = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in emotion)[:30]
        return str(Path("outputs/tts") / f"{safe_char}_{safe_emotion}.wav")

    def _can_use_xtts(self) -> bool:
        try:
            import TTS  # noqa: F401
            return True
        except Exception:
            return False

    def _ensure_xtts_loaded(self):
        if self._xtts is not None:
            return

        import torch
        from TTS.api import TTS

        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        device = self.prefer_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading XTTS-v2 (device={device})")
        self._xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        self._xtts_device = device

    def _find_reference_audio(self, character: str) -> Path:
        """
        Find a reference WAV from the repo's extracted voice samples.

        Expected layout:
        - data/films/*/voice_samples_auto/by_character/<Character>/*.wav
        """
        films_root = Path("data/films")
        if not films_root.exists():
            raise FileNotFoundError("Missing data/films (no voice samples available)")

        target = character.strip().lower()

        by_character_dirs = []
        for film_dir in films_root.iterdir():
            candidate = film_dir / "voice_samples_auto" / "by_character"
            if not candidate.exists():
                continue
            for child in candidate.iterdir():
                if child.is_dir() and child.name.lower() == target:
                    by_character_dirs.append(child)

        if not by_character_dirs:
            raise FileNotFoundError(
                f"No reference voice samples found for '{character}'. "
                "Expected under data/films/*/voice_samples_auto/by_character/<Character>/"
            )

        # Prefer first match; pick a medium-length sample by file size.
        samples = sorted(by_character_dirs[0].glob("*.wav"))
        if not samples:
            raise FileNotFoundError(f"No .wav files found in: {by_character_dirs[0]}")

        sample_sizes = [(p, p.stat().st_size) for p in samples[:30]]
        sample_sizes.sort(key=lambda x: x[1])
        return sample_sizes[len(sample_sizes) // 2][0]

    def _synthesize_xtts(self, *, text: str, character: str, output_path: Path) -> SimpleSynthesisResult:
        self._ensure_xtts_loaded()
        reference_wav = self._find_reference_audio(character)

        self._xtts.tts_to_file(
            text=text,
            speaker_wav=str(reference_wav),
            language=self.language,
            file_path=str(output_path),
        )

        # Best-effort duration estimate: read PCM frame count.
        duration = 0.0
        sample_rate = 22050
        try:
            with wave.open(str(output_path), "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.getnframes()
                duration = frames / float(sample_rate)
        except Exception:
            pass

        return SimpleSynthesisResult(
            audio_path=str(output_path),
            sample_rate=sample_rate,
            duration_seconds=duration,
            success=True,
        )

    def _synthesize_silence(self, *, text: str, output_path: Path) -> SimpleSynthesisResult:
        # Deterministic duration heuristic.
        sample_rate = 22050
        duration = max(0.6, min(20.0, len(text) * 0.06))
        num_samples = int(duration * sample_rate)

        audio = np.zeros(num_samples, dtype=np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        return SimpleSynthesisResult(
            audio_path=str(output_path),
            sample_rate=sample_rate,
            duration_seconds=duration,
            success=True,
        )

