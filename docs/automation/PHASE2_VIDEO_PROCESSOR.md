# Phase 2: Video Processing Automation - Video Processor Guide

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2025-12-02

---

## Overview

The Video Processor provides FFmpeg-based video processing capabilities for automation workflows. All operations are CPU-only and optimized for 32-thread processing to run alongside GPU training.

### Key Features

- ✅ **CPU-Only Processing:** Never interferes with GPU training
- ✅ **32-Thread Optimization:** Fully utilizes your CPU resources
- ✅ **Precise Frame-Accurate Cuts:** Extract exact video segments
- ✅ **Seamless Concatenation:** Join videos with optional transitions
- ✅ **Format Conversion:** Optimize codecs and quality settings
- ✅ **Video Effects:** Apply fades, color adjustments, and filters
- ✅ **Batch Processing:** Process multiple operations from config file
- ✅ **Memory Safety:** Integrated with Phase 1 safety infrastructure

---

## Operations

### 1. Cut Video Segments

Extract precise segments from videos with frame-accurate timing.

**Use Cases:**
- Extract specific scenes from movies
- Remove unwanted sections
- Create clips for training data
- Prepare video segments for concatenation

**Example:**
```bash
# Cut 30-second segment starting at 1:30
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation cut \
  --input /path/to/video.mp4 \
  --output /path/to/segment.mp4 \
  --start-time "00:01:30" \
  --duration "30"

# Cut segment between two timestamps
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation cut \
  --input /path/to/video.mp4 \
  --output /path/to/segment.mp4 \
  --start-time "00:01:30" \
  --end-time "00:05:45"
```

**Time Formats:**
- `HH:MM:SS` - Hours:Minutes:Seconds (e.g., "00:01:30")
- Seconds as string (e.g., "90")

---

### 2. Concatenate Videos

Join multiple videos seamlessly with optional transition effects.

**Use Cases:**
- Combine multiple scene extracts
- Create compilation videos
- Join video segments with transitions
- Build multi-clip sequences

**Example:**
```bash
# Simple concatenation (no transitions)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation concat \
  --inputs video1.mp4 video2.mp4 video3.mp4 \
  --output concatenated.mp4

# Concatenation with fade transitions
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation concat \
  --input-list videos_to_concat.txt \
  --output concatenated_with_fades.mp4 \
  --transition fade \
  --transition-duration 1.0
```

**Input List Format** (`videos_to_concat.txt`):
```
/path/to/video1.mp4
/path/to/video2.mp4
/path/to/video3.mp4
```

**Transition Types:**
- `none` - Direct cut (default, fastest)
- `fade` - Fade transition between clips
- `wipe` - Wipe transition (future)

---

### 3. Convert Format and Optimize

Convert video formats and optimize encoding settings for quality and file size.

**Use Cases:**
- Convert between formats (MOV → MP4, AVI → MP4, etc.)
- Optimize codec settings for quality/size balance
- Standardize video encoding for consistency
- Reduce file sizes for storage/distribution

**Example:**
```bash
# Convert with balanced quality (CRF 23, medium preset)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation convert \
  --input /path/to/input.mov \
  --output /path/to/output.mp4 \
  --codec h264 \
  --crf 23 \
  --preset medium \
  --audio-codec aac \
  --audio-bitrate 192k

# High-quality conversion (CRF 18, slow preset)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation convert \
  --input /path/to/input.avi \
  --output /path/to/output_hq.mp4 \
  --codec h264 \
  --crf 18 \
  --preset slow \
  --audio-codec aac \
  --audio-bitrate 256k
```

**Codec Options:**
- `h264` - Best compatibility, good quality (recommended)
- `h265` - Better compression, slower encoding
- `vp9` - WebM format, good for web

**CRF (Constant Rate Factor):**
- **0-51 range** (lower = better quality, larger file)
- **18** - Visually lossless quality
- **23** - Default, good quality (balanced)
- **28** - Lower quality, smaller file

**Encoding Presets** (speed vs compression):
- `ultrafast` - Fastest, largest file
- `fast` - Quick encoding, larger file
- `medium` - Balanced (default)
- `slow` - Better compression, takes longer
- `veryslow` - Best compression, slowest

**Audio Settings:**
- Codec: `aac` (recommended), `mp3`, `opus`
- Bitrate: `192k` (standard), `256k` (high quality), `128k` (lower quality)

---

### 4. Apply Video Effects

Apply visual effects including fades, color adjustments, and filters.

**Use Cases:**
- Add professional fade in/out effects
- Adjust brightness/contrast for consistency
- Color correction and grading
- Enhance visual quality

**Example:**
```bash
# Apply fade in/out with color adjustments
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation effects \
  --input /path/to/video.mp4 \
  --output /path/to/video_effects.mp4 \
  --fade-in 2.0 \
  --fade-out 2.0 \
  --brightness 1.1 \
  --contrast 1.05 \
  --saturation 1.05

# Just fade effects (no color adjustments)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation effects \
  --input /path/to/video.mp4 \
  --output /path/to/video_faded.mp4 \
  --fade-in 1.5 \
  --fade-out 1.5
```

**Effect Parameters:**
- `--fade-in` - Fade in duration in seconds (0 = no fade)
- `--fade-out` - Fade out duration in seconds (0 = no fade)
- `--brightness` - Brightness multiplier (1.0 = no change, >1.0 = brighter, <1.0 = darker)
- `--contrast` - Contrast multiplier (1.0 = no change, >1.0 = more contrast)
- `--saturation` - Saturation multiplier (1.0 = no change, >1.0 = more saturated)

---

### 5. Extract Video Metadata

Extract detailed metadata from video files (duration, resolution, codecs, etc.).

**Example:**
```bash
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation metadata \
  --input /path/to/video.mp4
```

**Output:**
```json
{
  "duration_seconds": 300.5,
  "width": 1920,
  "height": 1080,
  "fps": 24.0,
  "codec": "h264",
  "bitrate": 5000000,
  "audio_codec": "aac",
  "audio_bitrate": 192000,
  "file_size_bytes": 187500000
}
```

---

### 6. Batch Processing

Process multiple operations from a YAML configuration file.

**Use Cases:**
- Automated overnight processing
- Consistent multi-step workflows
- Repeatable video processing pipelines
- Complex editing sequences

**Example:**
```bash
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/video_processor.py \
  --operation batch \
  --batch-config configs/automation/video_processor_batch_example.yaml
```

**Batch Config Format** (`video_processor_batch_example.yaml`):
```yaml
# CPU threads for FFmpeg
threads: 32

# Batch operations
operations:
  # Operation 1: Cut segment
  - operation: cut
    input: /path/to/video.mp4
    output: /path/to/segment1.mp4
    start_time: "00:01:30"
    duration: "30"

  # Operation 2: Apply effects
  - operation: effects
    input: /path/to/segment1.mp4
    output: /path/to/segment1_effects.mp4
    fade_in: 1.0
    fade_out: 1.0
    brightness: 1.05

  # Operation 3: Convert format
  - operation: convert
    input: /path/to/segment1_effects.mp4
    output: /path/to/segment1_final.mp4
    codec: h264
    crf: 23
    preset: medium

  # Operation 4: Concatenate multiple segments
  - operation: concat
    inputs:
      - /path/to/segment1_final.mp4
      - /path/to/segment2.mp4
    output: /path/to/concatenated.mp4
```

**Output:**
- Batch report JSON with operation results
- Success/failure status for each operation
- Processing time and metadata

---

## Safety Infrastructure Integration

The Video Processor is fully integrated with Phase 1 safety infrastructure:

### GPU Isolation
- **CPU-only FFmpeg operations** - No GPU usage
- **Environment enforcement** - `CUDA_VISIBLE_DEVICES=""` inherited
- **Process affinity** - CPU cores 0-31

### Memory Monitoring
- **Runtime checks** during long operations
- **Memory thresholds** enforced (70% warning, 85% emergency)
- **Graceful abort** if memory exceeds limits

### Resource Management
- **32-thread optimization** - Full CPU utilization
- **Nice priority 10** - Lower than training processes
- **OOM protection** - Killable before training (score +500)

---

## Performance Optimization

### CPU Thread Utilization

The Video Processor automatically uses all 32 CPU threads for FFmpeg operations:

```bash
# FFmpeg automatically uses all threads
ffmpeg -i input.mp4 -threads 32 output.mp4
```

### Processing Speed Estimates

Based on 32-thread CPU processing:

| Operation | Speed Factor | Example (1 hour video) |
|-----------|--------------|------------------------|
| Cut (copy codec) | ~50x realtime | ~1.2 minutes |
| Concat (copy codec) | ~50x realtime | ~1.2 minutes |
| Convert (h264, medium) | ~2-5x realtime | 12-30 minutes |
| Convert (h264, slow) | ~1-2x realtime | 30-60 minutes |
| Effects (fade only) | ~10-20x realtime | 3-6 minutes |
| Effects (full filters) | ~5-10x realtime | 6-12 minutes |

**Note:** Actual speeds depend on input/output formats, resolution, and codec settings.

### Batch Processing Best Practices

1. **Use copy codec for cuts** - Fastest, no re-encoding
2. **Group similar operations** - Reduce intermediate files
3. **Convert once at the end** - Avoid multiple re-encodes
4. **Run overnight for large batches** - Use tmux/screen for persistence

---

## Common Workflows

### Workflow 1: Extract and Optimize Scene

```bash
# Step 1: Extract scene
python scripts/automation/scenarios/video_processor.py \
  --operation cut \
  --input /path/to/movie.mp4 \
  --output /tmp/scene_raw.mp4 \
  --start-time "00:15:30" \
  --duration "120"

# Step 2: Apply effects
python scripts/automation/scenarios/video_processor.py \
  --operation effects \
  --input /tmp/scene_raw.mp4 \
  --output /tmp/scene_effects.mp4 \
  --fade-in 1.0 \
  --fade-out 1.0 \
  --brightness 1.05

# Step 3: Optimize format
python scripts/automation/scenarios/video_processor.py \
  --operation convert \
  --input /tmp/scene_effects.mp4 \
  --output /path/to/scene_final.mp4 \
  --codec h264 \
  --crf 23 \
  --preset medium
```

### Workflow 2: Create Compilation Video

```bash
# Step 1: Extract multiple scenes
for start in "00:01:30" "00:05:00" "00:10:15"; do
  python scripts/automation/scenarios/video_processor.py \
    --operation cut \
    --input /path/to/movie.mp4 \
    --output /tmp/scene_${start//:/_}.mp4 \
    --start-time "$start" \
    --duration "30"
done

# Step 2: Concatenate with transitions
python scripts/automation/scenarios/video_processor.py \
  --operation concat \
  --inputs /tmp/scene_*.mp4 \
  --output /path/to/compilation.mp4 \
  --transition fade \
  --transition-duration 1.0
```

### Workflow 3: Batch Processing with Config

```bash
# Create batch config file with all operations
cat > /tmp/batch_config.yaml <<EOF
threads: 32
operations:
  - operation: cut
    input: /path/to/movie.mp4
    output: /tmp/scene1.mp4
    start_time: "00:01:30"
    duration: "30"
  - operation: effects
    input: /tmp/scene1.mp4
    output: /tmp/scene1_effects.mp4
    fade_in: 1.0
    fade_out: 1.0
  - operation: convert
    input: /tmp/scene1_effects.mp4
    output: /path/to/scene1_final.mp4
    codec: h264
    crf: 23
    preset: medium
EOF

# Run batch processing
python scripts/automation/scenarios/video_processor.py \
  --operation batch \
  --batch-config /tmp/batch_config.yaml
```

---

## Troubleshooting

### FFmpeg Not Installed

**Symptom:** "FFmpeg not installed" error

**Solution:**
```bash
# Install FFmpeg
sudo apt update
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

### Cut Operation Not Frame-Accurate

**Symptom:** Cut segments start/end at wrong positions

**Solution:**
- Use `--start-time` with exact timestamps
- For frame-accurate cuts, consider re-encoding (remove `-c copy`)
- Verify input video has keyframes at desired positions

### Concatenation Fails

**Symptom:** "Could not find codec parameters" or mismatched formats

**Solution:**
- Ensure all input videos have same resolution, codec, and frame rate
- Convert all videos to same format first using `convert` operation
- Check video list file has correct paths (one per line)

### Memory Warnings During Processing

**Symptom:** "Memory level critical" warnings

**Solution:**
- Process videos in smaller segments
- Use lower CRF values (higher compression)
- Reduce batch size in config
- Monitor with `htop` or `watch -n 2 free -h`

### Slow Processing Speed

**Symptom:** Processing takes longer than expected

**Solution:**
- Use faster encoding presets (`fast`, `ultrafast`)
- Increase CRF value (lower quality, faster encoding)
- Use copy codec for cuts/concat when possible
- Verify 32 threads are being used (check with `htop`)

---

## Dependencies

### Core Requirements

Already installed in `ai_env`:
- **FFmpeg** (system package)
- **Python 3.10+**
- **PyYAML** (for batch configs)
- **psutil** (for safety monitoring)

### Installation (If Needed)

```bash
# Install FFmpeg (system)
sudo apt update
sudo apt install ffmpeg

# Verify FFmpeg installation
ffmpeg -version
ffprobe -version
```

---

## Future Enhancements (Phase 2 Expansion)

### Planned Features

1. **Advanced Transitions**
   - Wipe transitions
   - Dissolve effects
   - Custom transition templates

2. **Audio Processing**
   - Audio extraction
   - Audio mixing (multiple tracks)
   - Volume normalization
   - Audio fade in/out

3. **Resolution and Frame Rate**
   - Upscaling/downscaling
   - Frame rate conversion
   - Aspect ratio adjustment
   - Crop and padding

4. **Advanced Filters**
   - Stabilization
   - Denoising
   - Sharpening
   - Color grading presets

5. **Thumbnail Generation**
   - Extract keyframes
   - Create contact sheets
   - Generate GIF previews

---

## Support

### Documentation

- **Phase 1 Guide:** `docs/automation/PHASE1_GUIDE.md`
- **Safety Infrastructure:** `docs/automation/SAFETY_INFRASTRUCTURE.md`
- **API Reference:** `docs/automation/API_REFERENCE.md`

### Reporting Issues

1. Check FFmpeg installation: `ffmpeg -version`
2. Verify CPU-only environment: `echo $CUDA_VISIBLE_DEVICES`
3. Check logs for error messages
4. Test with simple operation first (metadata or short cut)

---

## Changelog

### v1.0.0 (2025-12-02)

- ✅ Initial Video Processor implementation
- ✅ Cut operation (frame-accurate segments)
- ✅ Concatenate operation (with transition support)
- ✅ Convert operation (format and codec optimization)
- ✅ Effects operation (fade, color adjustments)
- ✅ Batch processing from YAML config
- ✅ Metadata extraction with ffprobe
- ✅ 32-thread CPU optimization
- ✅ Phase 1 safety infrastructure integration
- ✅ Memory monitoring and graceful abort
- ✅ Comprehensive documentation and examples

---

**End of Video Processor Guide**
