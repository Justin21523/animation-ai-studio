"""
Voice Synthesis Tools for Agent Framework

Thin wrappers that connect agent tool calls to the real TTS modules.

Notes:
- Uses ModelManager + ServiceController to stop/restart LLM when needed (VRAM constraint).
- Keeps imports lightweight at module import time; heavy deps are imported inside functions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip())[:80]


def _default_output_dir() -> Path:
    return Path("outputs/agent/tts")


async def synthesize_character_voice(
    character: str,
    text: str,
    emotion: str = "neutral",
    intensity: float = 0.8,
) -> Dict[str, Any]:
    """
    Synthesize speech in character's voice.

    Tool name: synthesize_character_voice
    """
    from scripts.core.model_management.model_manager import ModelManager

    output_dir = _default_output_dir() / _safe_name(character.lower())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(character)}_{emotion}_{_timestamp_ms()}.wav"

    manager = ModelManager()
    llm_was_running = manager.service_controller.is_llm_running()

    try:
        if llm_was_running and not manager.service_controller.stop_llm(wait=True):
            raise RuntimeError("Failed to stop LLM service before TTS synthesis")

        with manager.use_tts(auto_unload_heavy=True) as tts:
            result = tts.synthesize(
                text=text,
                character=character,
                emotion=emotion,
                intensity=float(intensity),
                output_path=str(output_path),
            )

        audio_path = getattr(result, "audio_path", None) or str(output_path)
        return {
            "success": True,
            "audio_path": audio_path,
            "character": character,
            "emotion": emotion,
            "intensity": float(intensity),
            "text": text,
        }

    finally:
        if llm_was_running:
            manager.service_controller.start_llm(wait=True)


async def batch_synthesize_script(
    character: str,
    script_lines: List[str],
    default_emotion: str = "neutral",
) -> Dict[str, Any]:
    """
    Synthesize multiple lines from a script.

    Tool name: batch_synthesize_script
    """
    from scripts.core.model_management.model_manager import ModelManager

    output_dir = _default_output_dir() / "batch" / _safe_name(character.lower()) / str(_timestamp_ms())
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = ModelManager()
    llm_was_running = manager.service_controller.is_llm_running()

    results: List[Dict[str, Any]] = []

    try:
        if llm_was_running and not manager.service_controller.stop_llm(wait=True):
            raise RuntimeError("Failed to stop LLM service before TTS synthesis")

        with manager.use_tts(auto_unload_heavy=True) as tts:
            for i, line in enumerate(script_lines):
                line = (line or "").strip()
                if not line:
                    continue

                output_path = output_dir / f"{i:03d}_{default_emotion}.wav"
                synth_result = tts.synthesize(
                    text=line,
                    character=character,
                    emotion=default_emotion,
                    intensity=1.0,
                    output_path=str(output_path),
                )
                audio_path = getattr(synth_result, "audio_path", None) or str(output_path)
                results.append(
                    {
                        "line_index": i,
                        "text": line,
                        "emotion": default_emotion,
                        "audio_path": audio_path,
                        "success": True,
                    }
                )

        return {
            "success": True,
            "character": character,
            "output_dir": str(output_dir),
            "results": results,
        }

    finally:
        if llm_was_running:
            manager.service_controller.start_llm(wait=True)

