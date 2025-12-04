# Animation AI Studio - Quick Start Guide

**Version**: 1.0
**Date**: 2025-12-04
**Target Users**: Researchers, Animation Studios, AI Engineers

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [First Run - Complete Pipeline](#first-run---complete-pipeline)
5. [Individual Pipeline Execution](#individual-pipeline-execution)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tips](#performance-tips)

---

## Overview

Animation AI Studio provides **fully automated AI pipelines** for processing animation content:

- **CPU Pipeline**: Extract frames/audio → Analyze scenes → Build knowledge base
- **GPU Pipeline**: Segment characters (SAM2) → Generate images (SDXL) → LLM analysis → Voice training

**One-command execution**: From raw video to trained models in 3-6 hours.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 8 cores | AMD Ryzen 9 9950X (16 cores) |
| **RAM** | 16GB | 32GB+ |
| **GPU** | NVIDIA 16GB VRAM | RTX 5080 16GB |
| **Storage** | 100GB free | 500GB+ SSD |

### Software Requirements

- **OS**: Ubuntu 20.04+ / WSL2
- **Python**: 3.10+
- **CUDA**: 11.8+
- **Conda**: Miniconda or Anaconda

### Dependencies

```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    parallel \
    jq \
    nvidia-utils-545

# Python packages (installed via conda)
# See Installation section
```

---

## Installation

### Step 1: Clone Repository

```bash
cd /path/to/your/projects
git clone https://github.com/your-org/animation-ai-studio.git
cd animation-ai-studio
```

### Step 2: Create Conda Environment

```bash
# Create environment
conda create -n ai_env python=3.10 -y
conda activate ai_env

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements/core.txt
pip install -r requirements/video.txt
pip install -r requirements/audio.txt
pip install -r requirements/generation.txt
```

### Step 3: Verify Installation

```bash
# Check Python
python --version  # Should show 3.10+

# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi
```

Expected output:
```
PyTorch: 2.7.1+cu118
CUDA: True

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.xx       Driver Version: 545.xx       CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        ...          | ... |      Memory-Usage | ...               |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 5080 | ... |      234MiB / 16384MiB | ...         |
+-------------------------------+----------------------+----------------------+
```

### Step 4: Setup Data Directories

```bash
# Create output directories
mkdir -p /mnt/data/ai_data/datasets/3d-anime
mkdir -p /mnt/data/ai_data/outputs

# Create AI Warehouse (shared models)
mkdir -p /mnt/c/AI_LLM_projects/ai_warehouse/models
```

---

## First Run - Complete Pipeline

### Prepare Your Video

Place your animation video file(s) in an input directory:

```bash
mkdir -p ~/videos/luca_clips
cp /path/to/luca_clip.mp4 ~/videos/luca_clips/
```

### Execute Complete Pipeline

**Single command** to run everything (CPU + GPU):

```bash
bash scripts/batch/run_all_tasks_complete.sh \
    luca \
    ~/videos/luca_clips \
    /mnt/data/ai_data/datasets/3d-anime/luca \
    --enable-gpu \
    --parallel-jobs 16
```

**What happens**:
1. ✅ Validates prerequisites (dependencies, disk space, GPU)
2. ✅ **CPU Pipeline** (~30-60 min):
   - Stage 1: Extracts frames and audio
   - Stage 2: Analyzes scenes, composition, camera movement
   - Stage 3: Builds RAG knowledge base
3. ✅ **GPU Pipeline** (~2-4 hours):
   - Task 1: SAM2 character segmentation
   - Task 2: SDXL image generation
   - Task 3: LLM video analysis
   - Task 4: Voice training (optional, prompts for confirmation)
4. ✅ Generates comprehensive summary report

### Monitor Progress

**In the same terminal**, you'll see real-time progress:

```
[STAGE] Validating Prerequisites
[✓] CPU Pipeline Script found
[✓] GPU Pipeline Script found
[✓] All dependencies found
...

[STAGE] Executing CPU Pipeline (Stage 1-3)
[INFO] Stage 1: Data Preparation (Frame + Audio Extraction)
[✓] Processing video 1/1: luca_clip.mp4
...

[STAGE] Executing GPU Pipeline (Task 1-4)
[GPU] VRAM: 1200MB / 16384MB (7.3%) | Util: 45% | Temp: 62°C
[INFO] Task 1: SAM2 Character Segmentation
...

[STAGE] Generating Summary
════════════════════════════════════════════════════════════════
             PIPELINE EXECUTION SUMMARY
════════════════════════════════════════════════════════════════
Status: COMPLETED
Film: luca
Total Time: 2h 15m 32s
...
```

### Check Results

```bash
# View output directory
ls /mnt/data/ai_data/datasets/3d-anime/luca/

# Expected structure:
# frames/                    # Extracted frames
# audio/                     # Extracted audio
# analysis/                  # Scene analysis results
# rag/                       # Knowledge base
# segmented/                 # SAM2 segmentation masks
# generated_images/          # SDXL generated images
# llm_analysis/              # LLM analysis results
# voices/                    # Trained voice models (if enabled)
# logs/                      # Execution logs
# complete_pipeline_summary.json  # Summary report
```

---

## Individual Pipeline Execution

### CPU Pipeline Only

For machines without GPU or for preprocessing only:

```bash
bash scripts/batch/run_cpu_tasks_all.sh \
    luca \
    ~/videos/luca_clips \
    /mnt/data/ai_data/datasets/3d-anime/luca \
    --parallel-jobs 16
```

**Stages**:
- Stage 1: Frame + Audio Extraction (~10-20 min)
- Stage 2: Video Analysis (~20-30 min)
- Stage 3: RAG Preparation (~5-10 min)

### GPU Pipeline Only

If CPU preprocessing is already done:

```bash
bash scripts/batch/run_gpu_tasks_all.sh \
    /mnt/data/ai_data/datasets/3d-anime/luca \
    --enable-voice \
    --sam2-model sam2_hiera_base \
    --llm-model qwen-vl-7b
```

**Tasks**:
- Task 1: SAM2 Segmentation (~30-60 min)
- Task 2: SDXL Generation (~5-10 min)
- Task 3: LLM Analysis (~10-30 min)
- Task 4: Voice Training (~2-4 hours, optional)

---

## Common Use Cases

### Use Case 1: Quick Character Extraction

**Goal**: Extract and segment all character instances from a film.

```bash
# 1. Extract frames (scene-based)
python scripts/processing/extraction/universal_frame_extractor.py \
    --input ~/videos/luca.mp4 \
    --output /tmp/luca_frames \
    --mode scene \
    --scene-threshold 27.0

# 2. Segment characters with SAM2
bash scripts/batch/gpu_task1_segmentation.sh \
    /tmp/luca_frames \
    /tmp/luca_segmented \
    --model sam2_hiera_base \
    --device cuda
```

**Result**: All character instances segmented and tracked in `/tmp/luca_segmented/`.

### Use Case 2: Generate Character Images with LoRA

**Goal**: Generate high-quality character images in animation style.

```bash
# 1. Prepare generation config
cat > /tmp/luca_gen_config.json <<EOF
{
  "prompt": "luca, boy, brown hair, green eyes, smiling, pixar style, 3d animation",
  "negative_prompt": "blurry, low quality, distorted, text, watermark",
  "lora_path": "/path/to/luca_character_lora.safetensors",
  "lora_weight": 0.8,
  "seed": 42
}
EOF

# 2. Generate images
bash scripts/batch/gpu_task2_image_generation.sh \
    /tmp/luca_gen_config.json \
    /tmp/luca_generated \
    --num-images 20 \
    --quality high
```

**Result**: 20 high-quality images in `/tmp/luca_generated/`.

### Use Case 3: Train Character Voice Model

**Goal**: Clone a character's voice from film audio.

```bash
# 1. Extract character voice samples (manual or automated)
mkdir -p ~/voice_samples/luca

# 2. Train voice model
bash scripts/batch/gpu_task4_voice_training.sh \
    luca \
    ~/voice_samples/luca \
    ~/voice_models/luca \
    --epochs 100 \
    --device cuda
```

**Result**: Trained voice model at `~/voice_models/luca/luca_voice.pth`.

### Use Case 4: Complete Film Analysis

**Goal**: Extract all insights from a full-length film.

```bash
bash scripts/batch/run_all_tasks_complete.sh \
    coco \
    ~/films/coco_2017.mp4 \
    /mnt/data/ai_data/films/coco \
    --enable-gpu \
    --parallel-jobs 16
```

**Result**: Complete dataset with frames, audio, analysis, segmentation, and knowledge base.

---

## Troubleshooting

### Problem: GPU Out of Memory (OOM)

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. **Use smaller models**:
   ```bash
   # SAM2: base → small
   --sam2-model sam2_hiera_small

   # LLM: qwen-14b → qwen-vl-7b
   --llm-model qwen-vl-7b
   ```

2. **Reduce batch sizes** (edit scripts):
   ```python
   # In gpu_task2_image_generation.sh
   num_images_per_prompt=2  # Instead of 4
   ```

3. **Ensure model cleanup**:
   ```bash
   # Manually clear GPU
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### Problem: Disk Space Full

**Symptom**:
```
[✗] Insufficient disk space (need 100GB, have 45GB)
```

**Solutions**:
1. **Cleanup intermediate results**:
   ```bash
   # Remove frame boundaries (if extracted)
   rm -rf /mnt/data/ai_data/datasets/*/frames/boundaries

   # Reduce JPEG quality (edit cpu_task1 script)
   JPEG_QUALITY=85  # Default: 95
   ```

2. **Use external drive** for outputs:
   ```bash
   # Mount external SSD
   sudo mount /dev/sdX1 /mnt/external

   # Use as output directory
   bash scripts/batch/run_all_tasks_complete.sh \
       luca ~/videos /mnt/external/luca
   ```

### Problem: Parallel Jobs Overloading CPU

**Symptom**:
System becomes unresponsive during CPU pipeline.

**Solution**:
Reduce parallel workers:
```bash
bash scripts/batch/run_cpu_tasks_all.sh \
    luca ~/videos ~/output \
    --parallel-jobs 8  # Instead of default 16
```

### Problem: Voice Training Fails

**Symptom**:
```
[✗] Insufficient voice samples (need at least 5, found 2)
```

**Solution**:
1. **Extract more voice samples**:
   ```bash
   python scripts/synthesis/tts/extract_voice_samples.py \
       --video ~/videos/luca.mp4 \
       --transcript data/films/luca/transcript.json \
       --character "Luca" \
       --output data/films/luca/voice_samples
   ```

2. **Requirements**: 20-50 samples, 5-15 minutes total duration.

### Problem: Prerequisite Validation Fails

**Symptom**:
```
[✗] Missing dependencies: parallel ffmpeg
```

**Solution**:
Install missing packages:
```bash
# Ubuntu/Debian
sudo apt-get install -y parallel ffmpeg jq

# Or use conda
conda install -c conda-forge parallel ffmpeg
```

---

## Performance Tips

### 1. Use SSD for Working Directory

**Impact**: 2-3x faster frame extraction and analysis

```bash
# Use /tmp (usually on SSD)
bash scripts/batch/run_all_tasks_complete.sh \
    luca ~/videos /tmp/luca_output
```

### 2. Optimize Parallel Jobs

**Guideline**: `cores - 2` for best performance

```bash
# For 16-core CPU
--parallel-jobs 14

# For 8-core CPU
--parallel-jobs 6
```

### 3. Skip Already Processed Files

All scripts support **checkpoint/resume**:

```bash
# If script crashes, just re-run
bash scripts/batch/run_all_tasks_complete.sh luca ~/videos ~/output

# Already processed files are automatically skipped
[INFO] Checkpoint found, skipping already processed files
[INFO] Resuming from frame 1523/2000
```

### 4. Use Quality Presets Wisely

**SDXL quality presets**:

| Preset | Steps | Speed | Quality |
|--------|-------|-------|---------|
| draft | 20 | Fast (~5s/image) | Good for testing |
| standard | 30 | Medium (~8s/image) | **Recommended** |
| high | 50 | Slow (~12s/image) | High quality |
| ultra | 75 | Very slow (~18s/image) | Max quality |

**Example**:
```bash
bash scripts/batch/gpu_task2_image_generation.sh \
    config.json output/ \
    --quality standard  # Best balance
```

### 5. Monitor Resources

**Run monitoring in separate terminal**:

```bash
watch -n 1 nvidia-smi
```

Or use the built-in monitor (GPU pipeline):
```bash
# Automatically starts when running GPU pipeline
# Check: tail -f /tmp/gpu_monitor.log
```

### 6. Use tmux for Long Jobs

**Recommended** for overnight processing:

```bash
# Start tmux session
tmux new -s animation_pipeline

# Run pipeline
bash scripts/batch/run_all_tasks_complete.sh luca ~/videos ~/output

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t animation_pipeline
```

---

## Advanced Configuration

### Custom Model Paths

Edit `configs/global/paths.yaml`:

```yaml
ai_warehouse: /custom/path/to/models
datasets: /custom/path/to/datasets
outputs: /custom/path/to/outputs
```

### Custom Pipeline Workflow

For advanced users, see:
- `scripts/scenarios/data_pipeline_automation/examples/luca_full_pipeline.yaml`
- Documentation: `docs/WEEK_3_4_IMPLEMENTATION_PLAN.md`

---

## Next Steps

After completing your first run:

1. **Explore Results**:
   - View extracted frames
   - Check segmentation quality
   - Review LLM analysis

2. **Fine-tune Configuration**:
   - Adjust scene detection threshold
   - Modify generation prompts
   - Optimize quality vs. speed

3. **Read Advanced Guides**:
   - [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
   - [Voice Synthesis Guide](guides/voice_training.md)
   - [Dual LoRA Generation Guide](guides/dual_lora_generation_guide.md)

---

## Support

**Issues**: https://github.com/your-org/animation-ai-studio/issues
**Documentation**: [Project README](../README.md)
**Progress Report**: [PROGRESS_REPORT.md](../PROGRESS_REPORT.md)

---

**Version**: 1.0
**Last Updated**: 2025-12-04
**Maintainer**: Animation AI Studio Team
