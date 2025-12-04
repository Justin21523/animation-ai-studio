# Phase 2: Video Processing Automation - Subtitle Automation Guide

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2025-12-02

---

## Overview

The Subtitle Automation system provides comprehensive speech recognition, translation, and subtitle generation capabilities for automation workflows. All operations are CPU-only and optimized for 32-thread processing to run alongside GPU training.

### Key Features

- ✅ **Dual ASR Engines:** Whisper CPU (local) and OpenAI Whisper API (cloud)
- ✅ **Translation Support:** Claude API and GPT API for high-quality subtitle translation
- ✅ **Format Support:** SRT and VTT subtitle parsing and generation
- ✅ **Batch Processing:** Process multiple videos from config file
- ✅ **CPU-Only Processing:** Never interferes with GPU training
- ✅ **32-Thread Optimization:** Fully utilizes your CPU resources
- ✅ **Memory Safety:** Integrated with Phase 1 safety infrastructure
- ✅ **Model Size Options:** tiny (73MB), base (142MB), small (466MB), medium (1.5GB), large (2.9GB)
- ✅ **Language Detection:** Automatic language detection or manual specification
- ✅ **Rate Limiting:** Built-in API request throttling for translation services

---

## Operations

### 1. Transcribe Video to Subtitles

Generate subtitles from video audio using speech recognition.

**Use Cases:**
- Create subtitles for videos without captions
- Extract dialogue for analysis or training data
- Generate multilingual subtitle bases for translation
- Automate caption creation for large video libraries

**Example (Whisper CPU - Local):**
```bash
# Basic transcription with tiny model (fast, lower quality)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation transcribe \
  --input /path/to/video.mp4 \
  --output /path/to/subtitles.srt \
  --asr-engine whisper-cpu \
  --whisper-model tiny \
  --language en

# High-quality transcription with large model (slower, best quality)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation transcribe \
  --input /path/to/video.mp4 \
  --output /path/to/subtitles.srt \
  --asr-engine whisper-cpu \
  --whisper-model large \
  --language en

# Auto-detect language
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation transcribe \
  --input /path/to/video.mp4 \
  --output /path/to/subtitles.srt \
  --asr-engine whisper-cpu \
  --whisper-model medium
```

**Example (OpenAI API - Cloud):**
```bash
# Requires OPENAI_API_KEY environment variable
export OPENAI_API_KEY="sk-..."

PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation transcribe \
  --input /path/to/video.mp4 \
  --output /path/to/subtitles.srt \
  --asr-engine openai-api \
  --language en
```

**Whisper Model Comparison:**

| Model | Size | Speed (relative) | Quality | Use Case |
|-------|------|------------------|---------|----------|
| `tiny` | 73MB | 10x | Fair | Quick drafts, testing |
| `base` | 142MB | 7x | Good | Balanced speed/quality |
| `small` | 466MB | 4x | Better | General use |
| `medium` | 1.5GB | 2x | Very Good | High quality, reasonable speed |
| `large` | 2.9GB | 1x | Best | Maximum accuracy, production |

**Language Codes:**
- `en` - English
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `es` - Spanish
- `fr` - French
- `de` - German
- ... (all ISO 639-1 codes supported)

**Output Formats:**
- `.srt` - SubRip format (default, most compatible)
- `.vtt` - WebVTT format (web-optimized)

---

### 2. Translate Subtitles

Translate existing subtitles to another language using AI translation.

**Use Cases:**
- Create multilingual subtitle versions
- Localize content for different regions
- Generate training data in multiple languages
- Automate subtitle translation workflows

**Example (Claude API):**
```bash
# Requires ANTHROPIC_API_KEY environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation translate \
  --input /path/to/english.srt \
  --output /path/to/chinese.srt \
  --translator claude \
  --source-lang en \
  --target-lang zh

# Batch translation (processes 20 segments at a time)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation translate \
  --input /path/to/english.srt \
  --output /path/to/japanese.srt \
  --translator claude \
  --source-lang en \
  --target-lang ja \
  --batch-size 20
```

**Example (GPT API):**
```bash
# Requires OPENAI_API_KEY environment variable
export OPENAI_API_KEY="sk-..."

PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation translate \
  --input /path/to/english.srt \
  --output /path/to/spanish.srt \
  --translator gpt \
  --source-lang en \
  --target-lang es
```

**Translation Features:**
- **Batch processing:** Translates 20 segments at a time for efficiency
- **Context preservation:** Maintains dialogue context across segments
- **Format preservation:** Keeps timing and formatting intact
- **Rate limiting:** Automatic throttling to prevent API rate limits
- **Character consistency:** Preserves character names and terminology

---

### 3. Sync Subtitles to Video

Adjust subtitle timing to match video (future feature).

**Use Cases:**
- Fix subtitle timing drift
- Resync after video editing
- Adjust timing for different frame rates
- Align subtitles with audio cues

**Example (Planned):**
```bash
# Coming in future update
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation sync \
  --input /path/to/subtitles.srt \
  --video /path/to/video.mp4 \
  --output /path/to/synced.srt \
  --offset 2.5  # Shift by 2.5 seconds
```

---

### 4. Batch Processing

Process multiple operations from a YAML configuration file.

**Use Cases:**
- Automated overnight subtitle workflows
- Consistent multi-step processing pipelines
- Transcribe + translate in one workflow
- Process entire video libraries

**Example:**
```bash
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation batch \
  --batch-config configs/automation/subtitle_automation_example.yaml
```

**Batch Config Format** (`subtitle_automation_example.yaml`):
```yaml
# ASR Configuration
asr:
  engine: whisper-cpu  # or openai-api
  model: large         # tiny/base/small/medium/large (for whisper-cpu)
  language: en         # or auto-detect

# Translation Configuration
translation:
  enabled: true
  translator: claude   # or gpt
  source_lang: en
  target_languages:
    - zh  # Chinese
    - ja  # Japanese
    - ko  # Korean

# Batch operations
operations:
  # Operation 1: Transcribe video 1
  - operation: transcribe
    input: /path/to/video1.mp4
    output: /path/to/video1_en.srt
    asr_engine: whisper-cpu
    whisper_model: large
    language: en

  # Operation 2: Translate to Chinese
  - operation: translate
    input: /path/to/video1_en.srt
    output: /path/to/video1_zh.srt
    translator: claude
    source_lang: en
    target_lang: zh

  # Operation 3: Translate to Japanese
  - operation: translate
    input: /path/to/video1_en.srt
    output: /path/to/video1_ja.srt
    translator: claude
    source_lang: en
    target_lang: ja
```

---

## Safety Infrastructure Integration

The Subtitle Automation system is fully integrated with Phase 1 safety infrastructure:

### GPU Isolation
- **CPU-only operations** - No GPU usage for Whisper or translation
- **Environment enforcement** - `CUDA_VISIBLE_DEVICES=""` inherited
- **Process affinity** - CPU cores 0-31

### Memory Monitoring
- **Runtime checks** during long transcriptions
- **Memory thresholds** enforced (70% warning, 85% emergency)
- **Graceful abort** if memory exceeds limits

### Resource Management
- **32-thread optimization** - Full CPU utilization for Whisper
- **Nice priority 10** - Lower than training processes
- **OOM protection** - Killable before training (score +500)

---

## Performance Optimization

### Whisper CPU Performance

Based on 32-thread CPU processing:

| Model | 1-hour video | 10-minute video | Real-time factor |
|-------|--------------|-----------------|------------------|
| tiny | ~6 minutes | ~36 seconds | ~10x |
| base | ~9 minutes | ~54 seconds | ~7x |
| small | ~15 minutes | ~90 seconds | ~4x |
| medium | ~30 minutes | ~3 minutes | ~2x |
| large | ~60 minutes | ~6 minutes | ~1x |

**Note:** Times vary based on audio complexity and language.

### Translation Performance

| Service | Speed | Quality | Cost |
|---------|-------|---------|------|
| Claude API | ~5s per 20 segments | Excellent | Moderate |
| GPT API | ~3s per 20 segments | Excellent | Moderate |

### Best Practices

1. **Model Selection:**
   - Use `tiny` for quick drafts and testing
   - Use `medium` for balanced quality/speed
   - Use `large` for production and critical content
   - Use OpenAI API for cloud processing (requires API key)

2. **Batch Processing:**
   - Process multiple videos overnight using batch configs
   - Transcribe first, then translate (reuse transcriptions)
   - Use tmux/screen for long-running jobs

3. **Translation:**
   - Batch size 20 is optimal for most cases
   - Enable rate limiting for API stability
   - Test with small samples before large batches

---

## Common Workflows

### Workflow 1: English Video → Chinese Subtitles

```bash
# Step 1: Transcribe to English (large model for best quality)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation transcribe \
  --input /path/to/video.mp4 \
  --output /tmp/english.srt \
  --asr-engine whisper-cpu \
  --whisper-model large \
  --language en

# Step 2: Translate English → Chinese
export ANTHROPIC_API_KEY="sk-ant-..."
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation translate \
  --input /tmp/english.srt \
  --output /path/to/chinese.srt \
  --translator claude \
  --source-lang en \
  --target-lang zh
```

### Workflow 2: Multilingual Subtitle Generation

```bash
# Step 1: Transcribe to English
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation transcribe \
  --input /path/to/video.mp4 \
  --output /tmp/english.srt \
  --asr-engine whisper-cpu \
  --whisper-model large \
  --language en

# Step 2: Translate to multiple languages
for lang in zh ja ko es fr; do
  PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
  conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
    --operation translate \
    --input /tmp/english.srt \
    --output /path/to/subtitles_${lang}.srt \
    --translator claude \
    --source-lang en \
    --target-lang $lang
done
```

### Workflow 3: Batch Processing with Config

```bash
# Create batch config
cat > /tmp/subtitle_batch.yaml <<EOF
asr:
  engine: whisper-cpu
  model: large
  language: en

translation:
  enabled: true
  translator: claude
  source_lang: en
  target_languages:
    - zh
    - ja

operations:
  - operation: transcribe
    input: /path/to/video1.mp4
    output: /tmp/video1_en.srt
    asr_engine: whisper-cpu
    whisper_model: large
    language: en

  - operation: translate
    input: /tmp/video1_en.srt
    output: /path/to/video1_zh.srt
    translator: claude
    source_lang: en
    target_lang: zh

  - operation: translate
    input: /tmp/video1_en.srt
    output: /path/to/video1_ja.srt
    translator: claude
    source_lang: en
    target_lang: ja
EOF

# Run batch processing
export ANTHROPIC_API_KEY="sk-ant-..."
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/subtitle_automation.py \
  --operation batch \
  --batch-config /tmp/subtitle_batch.yaml
```

---

## Troubleshooting

### Whisper Not Installed

**Symptom:** "Whisper not installed" error

**Solution:**
```bash
# Install openai-whisper
conda activate ai_env
pip install openai-whisper

# Verify installation
python -c "import whisper; print(whisper.__version__)"
```

### Model Download Issues

**Symptom:** Model download fails or times out

**Solution:**
- Models are automatically downloaded to `/mnt/c/ai_models/video/whisper/`
- Large model is 2.9GB, ensure sufficient disk space
- Check internet connection
- Retry download (models are cached after first download)

### Poor Transcription Quality

**Symptom:** Incorrect transcriptions, missing words, wrong names

**Solution:**
- **Use larger model:** tiny → base → small → medium → large
- **Specify language:** Use `--language en` instead of auto-detect
- **Check audio quality:** Whisper performs best on clear audio
- **Test segment:** Try 30-second test before full video

**Example comparison:**
- Tiny model: "believe we see monsters" ❌
- Large model: "believe in sea monsters" ✅

### Translation API Errors

**Symptom:** Rate limit errors, API key errors

**Solution:**
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY  # Should show your key

# For rate limits, reduce batch size
--batch-size 10  # Instead of default 20

# Add delays between batches (automatic, but can be adjusted in code)
```

### Memory Warnings During Transcription

**Symptom:** "Memory level critical" warnings

**Solution:**
- Use smaller Whisper model (large → medium → small)
- Process shorter video segments
- Close other applications
- Monitor with `htop` or `watch -n 2 free -h`

### Subtitle Timing Issues

**Symptom:** Subtitles appear too early or too late

**Solution:**
- Whisper timing is generally accurate for the source video
- If timing drift occurs, use sync operation (planned feature)
- For now, use subtitle editing tools (Aegisub, Subtitle Edit)

---

## Model Storage

All Whisper models are stored according to `data_model_structure.md`:

```
/mnt/c/ai_models/video/whisper/
├── tiny.pt         (73MB)
├── base.pt         (142MB)
├── small.pt        (466MB)
├── medium.pt       (1.5GB)
└── large-v3.pt     (2.9GB)
```

**Environment Variables:**
```bash
# Already configured in Phase 1
export HF_HOME=/mnt/c/ai_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface
export TORCH_HOME=/mnt/c/ai_cache/torch
export XDG_CACHE_HOME=/mnt/c/ai_cache
```

Models are automatically managed and stored in the correct location. See `configs/automation/model_storage_spec.yaml` for full specification.

---

## Dependencies

### Core Requirements

Already installed in `ai_env`:
- **openai-whisper** (version 20250625)
- **Python 3.10+**
- **PyYAML** (for batch configs)
- **psutil** (for safety monitoring)
- **ffmpeg** (for audio extraction)

### API Keys (Optional)

For cloud services:
```bash
# OpenAI Whisper API (for transcription)
export OPENAI_API_KEY="sk-..."

# Anthropic Claude API (for translation)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Installation (If Needed)

```bash
# Activate environment
conda activate ai_env

# Install openai-whisper
pip install openai-whisper

# Verify installation
python -c "import whisper; print('Whisper installed successfully')"

# Install ffmpeg (if not already installed)
sudo apt update
sudo apt install ffmpeg
```

---

## Future Enhancements (Phase 2 Expansion)

### Planned Features

1. **Subtitle Synchronization**
   - Automatic timing adjustment
   - Audio alignment detection
   - Frame-rate conversion
   - Offset correction

2. **Advanced Translation**
   - Custom terminology dictionaries
   - Character name consistency
   - Style preservation (formal/casual)
   - Context-aware translation

3. **Subtitle Editing**
   - Merge/split segments
   - Timing adjustment
   - Text formatting
   - Quality validation

4. **Format Conversion**
   - SRT ↔ VTT ↔ ASS/SSA
   - Styling preservation
   - Positioning support
   - Multi-track subtitles

5. **Quality Metrics**
   - WER (Word Error Rate) calculation
   - Translation quality scores
   - Timing accuracy metrics
   - Automated QA checks

---

## Support

### Documentation

- **Phase 1 Guide:** `docs/automation/PHASE1_GUIDE.md`
- **Video Processor Guide:** `docs/automation/PHASE2_VIDEO_PROCESSOR.md`
- **Safety Infrastructure:** `docs/automation/SAFETY_INFRASTRUCTURE.md`
- **Model Storage Spec:** `configs/automation/model_storage_spec.yaml`

### Reporting Issues

1. Check Whisper installation: `python -c "import whisper; print(whisper.__version__)"`
2. Verify model location: `ls -lh /mnt/c/ai_models/video/whisper/`
3. Test with tiny model on short video first
4. Check logs for error messages
5. Verify API keys are set (for cloud services)

---

## Changelog

### v1.0.0 (2025-12-02)

- ✅ Initial Subtitle Automation implementation
- ✅ Whisper CPU engine (tiny/base/small/medium/large models)
- ✅ OpenAI Whisper API engine
- ✅ Claude API translator
- ✅ GPT API translator
- ✅ SRT/VTT format support
- ✅ Batch processing from YAML config
- ✅ Language detection and specification
- ✅ 32-thread CPU optimization
- ✅ Phase 1 safety infrastructure integration
- ✅ Memory monitoring and graceful abort
- ✅ Model storage compliance (data_model_structure.md)
- ✅ Comprehensive documentation and examples

---

**End of Subtitle Automation Guide**
