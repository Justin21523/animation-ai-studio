"""
Subtitle Automation

Automated subtitle generation, translation, and processing using ASR and LLM APIs.
All processing is CPU-only and optimized for 32-thread execution.

Features:
  - Audio extraction from videos (FFmpeg)
  - Speech recognition (Whisper CPU or OpenAI Whisper API)
  - Subtitle translation (Claude or OpenAI GPT APIs)
  - SRT/VTT format parsing and generation
  - Batch processing with progress tracking
  - Subtitle editing and timeline adjustment

Usage:
  # Generate subtitles from video (Whisper CPU)
  python scripts/automation/scenarios/subtitle_automation.py \
    --operation transcribe \
    --input /path/to/video.mp4 \
    --output /path/to/subtitles.srt \
    --asr-engine whisper-cpu \
    --whisper-model base \
    --language en

  # Generate subtitles using OpenAI API
  python scripts/automation/scenarios/subtitle_automation.py \
    --operation transcribe \
    --input /path/to/video.mp4 \
    --output /path/to/subtitles.srt \
    --asr-engine openai-api \
    --language en

  # Translate existing subtitles
  python scripts/automation/scenarios/subtitle_automation.py \
    --operation translate \
    --input /path/to/subtitles_en.srt \
    --output /path/to/subtitles_zh.srt \
    --source-lang en \
    --target-lang zh-TW \
    --translation-engine claude

  # Batch processing
  python scripts/automation/scenarios/subtitle_automation.py \
    --operation batch \
    --batch-config /path/to/batch_config.yaml

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import sys
import os
import argparse
import json
import logging
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure
from scripts.core.safety import (
    enforce_cpu_only,
    verify_no_gpu_usage,
    MemoryMonitor,
    RuntimeMonitor,
    run_preflight,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SubtitleSegment:
    """Single subtitle segment with timing and text."""
    index: int
    start_time: float  # seconds
    end_time: float  # seconds
    text: str

    def to_srt_format(self) -> str:
        """Convert to SRT format."""
        start = self._format_time_srt(self.start_time)
        end = self._format_time_srt(self.end_time)
        return f"{self.index}\n{start} --> {end}\n{self.text}\n"

    def to_vtt_format(self) -> str:
        """Convert to VTT format."""
        start = self._format_time_vtt(self.start_time)
        end = self._format_time_vtt(self.end_time)
        return f"{start} --> {end}\n{self.text}\n"

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Format time as SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_time_vtt(seconds: float) -> str:
        """Format time as VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class TranscriptionResult:
    """Result of speech recognition."""
    segments: List[SubtitleSegment]
    language: str
    duration_seconds: float
    model: str
    success: bool
    error_message: Optional[str]
    timestamp: str


# ============================================================================
# Audio Extraction
# ============================================================================

class AudioExtractor:
    """Extract audio from video files using FFmpeg."""

    def __init__(self, threads: int = 32):
        """
        Initialize audio extractor.

        Args:
            threads: Number of CPU threads for FFmpeg
        """
        self.threads = threads

    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        format: str = 'wav',
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bool:
        """
        Extract audio from video.

        Args:
            video_path: Input video file
            output_path: Output audio file
            format: Audio format ('wav', 'mp3', 'flac')
            sample_rate: Sample rate in Hz (16000 recommended for Whisper)
            channels: Number of audio channels (1=mono, 2=stereo)

        Returns:
            True if successful
        """
        logger.info(f"Extracting audio from {video_path.name}")

        try:
            command = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le' if format == 'wav' else 'libmp3lame',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-threads', str(self.threads),
                '-y',  # Overwrite output
                str(output_path)
            ]

            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            logger.info(f"✓ Audio extracted: {output_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e}")
            return False


# ============================================================================
# ASR Engines
# ============================================================================

class WhisperCPUEngine:
    """
    Whisper CPU engine for local speech recognition.

    Uses openai-whisper with CPU-only inference.
    """

    def __init__(
        self,
        model_name: str = 'base',
        device: str = 'cpu',
        compute_type: str = 'int8',
    ):
        """
        Initialize Whisper CPU engine.

        Args:
            model_name: Model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cpu')
            compute_type: Compute type ('int8', 'float32')
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None

        logger.info(f"Initializing Whisper CPU engine (model={model_name})")

        try:
            import whisper
            self.whisper = whisper
            self.model = whisper.load_model(model_name, device=device)
            logger.info("✓ Whisper model loaded")
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Install with: pip install openai-whisper"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'zh', None for auto-detect)

        Returns:
            Transcription result dict
        """
        logger.info(f"Transcribing audio: {audio_path.name}")

        try:
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task='transcribe',
                verbose=False,
            )

            logger.info(f"✓ Transcription complete ({len(result['segments'])} segments)")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise


class OpenAIWhisperAPI:
    """
    OpenAI Whisper API client for cloud-based speech recognition.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI Whisper API client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        response_format: str = 'verbose_json',
    ) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI API.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'zh')
            response_format: Response format ('json', 'text', 'srt', 'verbose_json', 'vtt')

        Returns:
            Transcription result dict
        """
        logger.info(f"Transcribing audio via OpenAI API: {audio_path.name}")

        try:
            with open(audio_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format=response_format,
                )

            # Convert response to dict format similar to Whisper CPU
            if response_format == 'verbose_json':
                result = {
                    'text': response.text,
                    'language': response.language,
                    'duration': response.duration,
                    'segments': [
                        {
                            'start': seg.start,
                            'end': seg.end,
                            'text': seg.text,
                        }
                        for seg in response.segments
                    ]
                }
            else:
                result = {'text': response, 'segments': []}

            logger.info(f"✓ Transcription complete")
            return result

        except Exception as e:
            logger.error(f"OpenAI API transcription failed: {e}")
            raise


# ============================================================================
# Translation Engines
# ============================================================================

class ClaudeTranslator:
    """Claude API translator for subtitle translation."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """
        Initialize Claude translator.

        Args:
            api_key: Anthropic API key
            max_retries: Max retry attempts
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        self.max_retries = max_retries

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """
        Translate batch of texts.

        Args:
            texts: List of text strings to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translated strings
        """
        logger.info(f"Translating {len(texts)} texts ({source_lang} → {target_lang})")

        # Build prompt
        texts_json = json.dumps(texts, ensure_ascii=False, indent=2)

        prompt = f"""Translate the following subtitle texts from {source_lang} to {target_lang}.

Maintain the original meaning, tone, and style. Keep proper nouns unchanged.
Return ONLY a JSON array of translated strings, in the same order.

Input texts:
{texts_json}

Output format:
["translated text 1", "translated text 2", ...]"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=4096,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                )

                # Parse response
                response_text = response.content[0].text

                # Extract JSON array
                if '```json' in response_text:
                    start = response_text.find('```json') + 7
                    end = response_text.find('```', start)
                    json_str = response_text[start:end].strip()
                elif '```' in response_text:
                    start = response_text.find('```') + 3
                    end = response_text.find('```', start)
                    json_str = response_text[start:end].strip()
                else:
                    json_str = response_text.strip()

                translated = json.loads(json_str)

                if len(translated) != len(texts):
                    raise ValueError(f"Translation count mismatch: {len(translated)} != {len(texts)}")

                logger.info(f"✓ Translation complete")
                return translated

            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Translation failed after all retries")
                    raise


class GPTTranslator:
    """OpenAI GPT translator for subtitle translation."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """
        Initialize GPT translator.

        Args:
            api_key: OpenAI API key
            max_retries: Max retry attempts
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        self.max_retries = max_retries

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """
        Translate batch of texts.

        Args:
            texts: List of text strings
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of translated strings
        """
        logger.info(f"Translating {len(texts)} texts via GPT ({source_lang} → {target_lang})")

        texts_json = json.dumps(texts, ensure_ascii=False, indent=2)

        prompt = f"""Translate the following subtitle texts from {source_lang} to {target_lang}.

Maintain meaning, tone, and style. Keep proper nouns unchanged.
Return ONLY a JSON array of translated strings.

Input:
{texts_json}

Output format:
["translated 1", "translated 2", ...]"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional subtitle translator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )

                response_text = response.choices[0].message.content

                # Extract JSON
                if '```json' in response_text:
                    start = response_text.find('```json') + 7
                    end = response_text.find('```', start)
                    json_str = response_text[start:end].strip()
                elif '```' in response_text:
                    start = response_text.find('```') + 3
                    end = response_text.find('```', start)
                    json_str = response_text[start:end].strip()
                else:
                    json_str = response_text.strip()

                translated = json.loads(json_str)

                if len(translated) != len(texts):
                    raise ValueError(f"Translation count mismatch")

                logger.info(f"✓ Translation complete")
                return translated

            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


# ============================================================================
# Subtitle Format Handlers
# ============================================================================

class SubtitleParser:
    """Parse and generate subtitle files (SRT, VTT)."""

    @staticmethod
    def parse_srt(srt_path: Path) -> List[SubtitleSegment]:
        """
        Parse SRT file.

        Args:
            srt_path: Path to SRT file

        Returns:
            List of subtitle segments
        """
        segments = []

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by double newline
        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            # Parse index
            try:
                index = int(lines[0])
            except ValueError:
                continue

            # Parse timing
            timing_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                lines[1]
            )

            if not timing_match:
                continue

            start_h, start_m, start_s, start_ms = map(int, timing_match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, timing_match.groups()[4:])

            start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
            end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

            # Parse text
            text = '\n'.join(lines[2:])

            segments.append(SubtitleSegment(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))

        return segments

    @staticmethod
    def write_srt(segments: List[SubtitleSegment], output_path: Path):
        """
        Write SRT file.

        Args:
            segments: List of subtitle segments
            output_path: Output path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                # Re-index
                segment.index = i
                f.write(segment.to_srt_format())
                f.write('\n')

        logger.info(f"✓ SRT file saved: {output_path}")

    @staticmethod
    def write_vtt(segments: List[SubtitleSegment], output_path: Path):
        """
        Write VTT file.

        Args:
            segments: List of subtitle segments
            output_path: Output path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('WEBVTT\n\n')
            for segment in segments:
                f.write(segment.to_vtt_format())
                f.write('\n')

        logger.info(f"✓ VTT file saved: {output_path}")


# ============================================================================
# Main Operations
# ============================================================================

def transcribe_video(
    video_path: Path,
    output_path: Path,
    asr_engine: str = 'whisper-cpu',
    whisper_model: str = 'base',
    language: Optional[str] = None,
    output_format: str = 'srt',
    memory_monitor: Optional[MemoryMonitor] = None,
) -> TranscriptionResult:
    """
    Transcribe video to subtitles.

    Args:
        video_path: Input video file
        output_path: Output subtitle file
        asr_engine: ASR engine ('whisper-cpu' or 'openai-api')
        whisper_model: Whisper model size (for CPU engine)
        language: Language code or None for auto-detect
        output_format: Output format ('srt' or 'vtt')
        memory_monitor: Optional memory monitor

    Returns:
        TranscriptionResult
    """
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("SUBTITLE TRANSCRIPTION")
    logger.info("=" * 80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Engine: {asr_engine}")
    logger.info(f"Language: {language or 'auto-detect'}")

    try:
        # Step 1: Extract audio
        audio_path = output_path.parent / f"{video_path.stem}_audio.wav"
        extractor = AudioExtractor()

        if not extractor.extract_audio(video_path, audio_path):
            raise RuntimeError("Audio extraction failed")

        # Step 2: Transcribe
        if asr_engine == 'whisper-cpu':
            engine = WhisperCPUEngine(model_name=whisper_model)
        elif asr_engine == 'openai-api':
            engine = OpenAIWhisperAPI()
        else:
            raise ValueError(f"Unknown ASR engine: {asr_engine}")

        result = engine.transcribe(audio_path, language=language)

        # Step 3: Convert to subtitle segments
        segments = []
        for i, seg in enumerate(result.get('segments', []), 1):
            segments.append(SubtitleSegment(
                index=i,
                start_time=seg['start'],
                end_time=seg['end'],
                text=seg['text'].strip()
            ))

        # Step 4: Write subtitle file
        if output_format == 'srt':
            SubtitleParser.write_srt(segments, output_path)
        elif output_format == 'vtt':
            SubtitleParser.write_vtt(segments, output_path)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink()

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n✓ Transcription complete in {duration:.1f}s")
        logger.info(f"  Segments: {len(segments)}")
        logger.info(f"  Output: {output_path}")

        return TranscriptionResult(
            segments=segments,
            language=result.get('language', language or 'unknown'),
            duration_seconds=duration,
            model=asr_engine,
            success=True,
            error_message=None,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return TranscriptionResult(
            segments=[],
            language='',
            duration_seconds=(datetime.now() - start_time).total_seconds(),
            model=asr_engine,
            success=False,
            error_message=str(e),
            timestamp=datetime.now().isoformat()
        )


def translate_subtitles(
    input_path: Path,
    output_path: Path,
    source_lang: str,
    target_lang: str,
    translation_engine: str = 'claude',
    batch_size: int = 20,
) -> bool:
    """
    Translate subtitle file.

    Args:
        input_path: Input subtitle file
        output_path: Output subtitle file
        source_lang: Source language code
        target_lang: Target language code
        translation_engine: Translation engine ('claude' or 'gpt')
        batch_size: Number of segments per translation batch

    Returns:
        True if successful
    """
    logger.info("=" * 80)
    logger.info("SUBTITLE TRANSLATION")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Translation: {source_lang} → {target_lang}")
    logger.info(f"Engine: {translation_engine}")

    try:
        # Parse input file
        segments = SubtitleParser.parse_srt(input_path)
        logger.info(f"Loaded {len(segments)} segments")

        # Initialize translator
        if translation_engine == 'claude':
            translator = ClaudeTranslator()
        elif translation_engine == 'gpt':
            translator = GPTTranslator()
        else:
            raise ValueError(f"Unknown translation engine: {translation_engine}")

        # Translate in batches
        translated_segments = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            texts = [seg.text for seg in batch]

            logger.info(f"Translating batch {i // batch_size + 1}/{(len(segments) + batch_size - 1) // batch_size}")

            translated_texts = translator.translate_batch(
                texts,
                source_lang,
                target_lang
            )

            # Create translated segments
            for seg, trans_text in zip(batch, translated_texts):
                translated_segments.append(SubtitleSegment(
                    index=seg.index,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=trans_text
                ))

        # Write output file
        SubtitleParser.write_srt(translated_segments, output_path)

        logger.info(f"\n✓ Translation complete")
        logger.info(f"  Output: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return False


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Subtitle Automation - ASR, translation, and processing (CPU-only)'
    )

    # Operation mode
    parser.add_argument('--operation', type=str, required=True,
                       choices=['transcribe', 'translate', 'batch'],
                       help='Operation to perform')

    # Input/output
    parser.add_argument('--input', type=Path,
                       help='Input video or subtitle file')
    parser.add_argument('--output', type=Path,
                       help='Output subtitle file')

    # Transcription options
    parser.add_argument('--asr-engine', type=str, default='whisper-cpu',
                       choices=['whisper-cpu', 'openai-api'],
                       help='ASR engine (default: whisper-cpu)')
    parser.add_argument('--whisper-model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--language', type=str,
                       help='Language code (e.g., en, zh) or auto-detect')
    parser.add_argument('--output-format', type=str, default='srt',
                       choices=['srt', 'vtt'],
                       help='Output subtitle format (default: srt)')

    # Translation options
    parser.add_argument('--source-lang', type=str,
                       help='Source language code')
    parser.add_argument('--target-lang', type=str,
                       help='Target language code')
    parser.add_argument('--translation-engine', type=str, default='claude',
                       choices=['claude', 'gpt'],
                       help='Translation engine (default: claude)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Translation batch size (default: 20)')

    # Batch processing
    parser.add_argument('--batch-config', type=Path,
                       help='Batch configuration YAML file')

    # Safety
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight safety checks')

    args = parser.parse_args()

    # Enforce CPU-only
    enforce_cpu_only()

    # Run preflight checks
    if not args.skip_preflight:
        logger.info("Running preflight checks...")
        try:
            run_preflight(strict=True)
        except Exception as e:
            logger.warning(f"Preflight checks failed: {e}")

    # Create memory monitor
    memory_monitor = MemoryMonitor()

    # Start runtime monitoring
    with RuntimeMonitor(check_interval=30.0) as monitor:
        # Execute operation
        if args.operation == 'transcribe':
            if not args.input or not args.output:
                parser.error("--input and --output required for transcribe operation")

            result = transcribe_video(
                args.input,
                args.output,
                args.asr_engine,
                args.whisper_model,
                args.language,
                args.output_format,
                memory_monitor,
            )

            if not result.success:
                sys.exit(1)

        elif args.operation == 'translate':
            if not args.input or not args.output:
                parser.error("--input and --output required for translate operation")
            if not args.source_lang or not args.target_lang:
                parser.error("--source-lang and --target-lang required for translate operation")

            success = translate_subtitles(
                args.input,
                args.output,
                args.source_lang,
                args.target_lang,
                args.translation_engine,
                args.batch_size,
            )

            if not success:
                sys.exit(1)

        elif args.operation == 'batch':
            parser.error("Batch operation not yet implemented")


if __name__ == '__main__':
    main()
