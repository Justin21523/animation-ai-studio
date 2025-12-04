# CPU Batch Processing Scripts

**Complete automation pipeline for CPU-only video processing tasks**

This directory contains production-ready batch processing scripts for the Animation AI Studio project. All scripts are designed for **100% CPU-only execution** with comprehensive resource management.

---

## 📋 Overview

The batch processing system automates three sequential stages of video processing:

1. **Stage 1: Data Preparation** - Frame and audio extraction
2. **Stage 2: Video Analysis** - Scene detection, composition, camera tracking
3. **Stage 3: RAG Preparation** - Knowledge base and document processing

All scripts support:
- ✅ Parallel processing (GNU parallel)
- ✅ Checkpoint/resume capability
- ✅ Resource monitoring (CPU, RAM, Disk)
- ✅ Error handling and rollback
- ✅ Progress tracking and logging

---

## 🚀 Quick Start

### Complete Pipeline (All 3 Stages)

```bash
# Process a complete film with monitoring
bash scripts/batch/run_cpu_tasks_all.sh \
  luca \
  /mnt/c/raw_videos/luca \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --workers 8 \
  --monitor
```

### Individual Stages

```bash
# Stage 1 only: Frame and audio extraction
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /mnt/c/raw_videos/luca \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --workers 8 \
  --resume

# Stage 2 only: Video analysis
bash scripts/batch/cpu_tasks_stage2_analysis.sh \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --workers 8

# Stage 3 only: RAG preparation
bash scripts/batch/cpu_tasks_stage3_rag_prep.sh \
  luca \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

---

## 📁 Script Details

### 1. run_cpu_tasks_all.sh

**Master orchestration script** - Executes all 3 stages sequentially with integrated monitoring.

**Features**:
- Automatic stage dependency handling
- Integrated resource monitoring daemon
- Comprehensive error handling and rollback
- Execution time tracking and reporting
- Final summary with output statistics

**Usage**:
```bash
bash scripts/batch/run_cpu_tasks_all.sh \
  FILM_NAME \
  INPUT_VIDEO_DIR \
  OUTPUT_BASE_DIR \
  [OPTIONS]
```

**Arguments**:
- `FILM_NAME`: Name of the film (e.g., luca, coco)
- `INPUT_VIDEO_DIR`: Directory containing raw video files
- `OUTPUT_BASE_DIR`: Base output directory for all results

**Options**:
- `--workers N`: Number of parallel workers (default: 8)
- `--monitor`: Enable resource monitoring daemon
- `--resume`: Resume from checkpoints (skip completed stages)

**Example**:
```bash
bash scripts/batch/run_cpu_tasks_all.sh luca \
  /mnt/c/raw_videos/luca \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --workers 8 \
  --monitor \
  --resume
```

**Output Structure**:
```
OUTPUT_DIR/
├── frames/              # Stage 1: Extracted frames
├── audio/               # Stage 1: Extracted audio
├── analysis/            # Stage 2: Video analysis
│   ├── scenes/
│   ├── composition/
│   └── camera/
├── rag/                 # Stage 3: RAG preparation
│   ├── documents/
│   └── knowledge_base/
├── monitoring/          # Resource monitoring logs
├── logs/                # Stage execution logs
└── execution_metadata.json
```

---

### 2. cpu_tasks_stage1_data_prep.sh

**Data preparation stage** - Parallel frame and audio extraction from videos.

**Features**:
- Parallel frame extraction using universal_frame_extractor.py
- Parallel audio extraction using FFmpeg
- Scene-based frame extraction (PySceneDetect)
- Checkpoint support for resume
- Memory and disk space monitoring

**Usage**:
```bash
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  INPUT_DIR \
  OUTPUT_DIR \
  [OPTIONS]
```

**Options**:
- `--workers N`: Number of parallel workers (default: 8, max: 16)
- `--resume`: Resume from checkpoint (skip already processed)

**Requirements**:
- CPU: 8+ cores recommended
- RAM: 16GB+ recommended
- Disk: 50-100GB free space per film
- NO GPU required

**Example**:
```bash
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /mnt/c/raw_videos/luca \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --workers 8 \
  --resume
```

**Processing Details**:
- Frame extraction mode: `scene` (adaptive, based on shot changes)
- Scene threshold: 27.0 (default PySceneDetect)
- Frames per scene: 3 (beginning, middle, end)
- JPEG quality: 95 (high quality)
- Audio format: 16kHz, mono, PCM WAV

**Output**:
- `frames/{video_name}/`: Extracted frames with timestamps
- `audio/{video_name}_audio.wav`: Extracted audio tracks
- `dataset_index.json`: Dataset metadata
- `checkpoints/*.txt`: Checkpoint files for resume

---

### 3. cpu_tasks_stage2_analysis.sh

**Video analysis stage** - Parallel scene detection, composition analysis, and camera tracking.

**Features**:
- Parallel scene detection (PySceneDetect)
- Parallel composition analysis (OpenCV)
- Parallel camera movement tracking (Optical Flow)
- Results aggregation and summary generation

**Usage**:
```bash
bash scripts/batch/cpu_tasks_stage2_analysis.sh \
  BASE_DIR \
  [OPTIONS]
```

**Arguments**:
- `BASE_DIR`: Output directory from Stage 1

**Options**:
- `--workers N`: Number of parallel workers (default: 8)
- `--resume`: Resume from checkpoint

**Requirements**:
- CPU: 8+ cores
- RAM: 16GB+ (analysis can be memory-intensive)
- NO GPU required

**Example**:
```bash
bash scripts/batch/cpu_tasks_stage2_analysis.sh \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --workers 8
```

**Analysis Tasks**:
1. **Scene Detection**: PySceneDetect with content-based threshold
2. **Composition Analysis**: Rule of thirds, symmetry, balance
3. **Camera Tracking**: Optical flow-based motion estimation

**Output**:
- `analysis/scenes/*.json`: Scene detection results
- `analysis/composition/*.json`: Composition metrics
- `analysis/camera/*.json`: Camera movement data
- `analysis_summary.json`: Aggregated results

---

### 4. cpu_tasks_stage3_rag_prep.sh

**RAG preparation stage** - Document processing and knowledge base creation with **CPU-only embeddings**.

**Features**:
- Process character description documents
- Process film metadata
- CPU-only embedding models (sentence-transformers)
- FAISS index creation (CPU)

**Usage**:
```bash
bash scripts/batch/cpu_tasks_stage3_rag_prep.sh \
  FILM_NAME \
  OUTPUT_DIR \
  [OPTIONS]
```

**Options**:
- `--embedding-model MODEL`: CPU-only embedding model (default: sentence-transformers/all-MiniLM-L6-v2)

**Requirements**:
- CPU: 4+ cores
- RAM: 8GB+ (embedding models need ~2GB)
- Disk: Minimal (~1GB for index)
- **NO GPU** (forced CPU-only via `CUDA_VISIBLE_DEVICES=""`)

**Example**:
```bash
bash scripts/batch/cpu_tasks_stage3_rag_prep.sh \
  luca \
  /mnt/data/ai_data/datasets/3d-anime/luca \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

**CPU-Only Embedding Models** (Recommended):
- `sentence-transformers/all-MiniLM-L6-v2` (fastest, 80MB)
- `sentence-transformers/all-mpnet-base-v2` (better quality, 420MB)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

**Output**:
- `rag/documents/`: Processed documents
- `rag/knowledge_base/`: FAISS index (CPU)
- `rag/rag_summary.json`: RAG metadata

---

### 5. monitor_resources.sh

**Real-time resource monitoring** - Track CPU, RAM, GPU, disk usage during processing.

**Features**:
- Continuous monitoring with configurable interval
- CSV log output for analysis
- Threshold-based warnings (CPU, RAM, VRAM, Disk, GPU temp)
- Background daemon mode
- Graceful shutdown (SIGTERM/SIGINT)

**Usage**:
```bash
# Foreground mode (print to console)
bash scripts/batch/monitor_resources.sh --interval 10

# Background daemon mode (log to file)
bash scripts/batch/monitor_resources.sh \
  --daemon \
  --log-dir /tmp/monitoring \
  --interval 30

# Stop daemon
pkill -f monitor_resources.sh
```

**Options**:
- `--interval N`: Monitoring interval in seconds (default: 10)
- `--daemon`: Run in background daemon mode
- `--log-dir DIR`: Directory for log files (default: /tmp/resource_monitoring)
- `--once`: Run once and exit (useful for scripts)

**Thresholds** (Warning/Critical):
- CPU: 85% / 95%
- RAM: 80% / 90%
- GPU VRAM: 80% / 90%
- Disk: 90% / 95%
- GPU Temperature: 80°C / 85°C

**Output**:
- Console: Real-time status with color-coded warnings
- CSV: Timestamped metrics (CPU, RAM, VRAM, GPU util, temp, disk)
- Log file: Daemon execution log

---

## 🔧 Configuration

### Resource Thresholds

All scripts have built-in safety thresholds:

```bash
# Memory safety (Stage 1)
MEMORY_THRESHOLD_PCT=90  # Stop if RAM > 90%
DISK_THRESHOLD_GB=10     # Stop if free disk < 10GB

# Worker limits (All stages)
DEFAULT_WORKERS=8
MAX_WORKERS=16
```

### CPU-Only Enforcement

All scripts enforce CPU-only execution:

```bash
# Stage 3 (RAG) explicitly disables GPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU-only embeddings
```

### Checkpoint Files

Checkpoint files are stored in `{output_dir}/checkpoints/`:

```
checkpoints/
├── frames_processed.txt      # Stage 1: Frame extraction
├── audio_processed.txt       # Stage 1: Audio extraction
├── scenes_processed.txt      # Stage 2: Scene detection
├── composition_processed.txt # Stage 2: Composition
└── camera_processed.txt      # Stage 2: Camera tracking
```

To resume from a checkpoint, use `--resume` flag.

---

## 📊 Performance Expectations

### Processing Times (Luca - 95 minutes film)

**Hardware**: AMD Ryzen 9 9950X (16 cores), 32GB RAM

| Stage | Task | Time (8 workers) | Time (16 workers) |
|-------|------|------------------|-------------------|
| Stage 1 | Frame extraction | 30-45 min | 20-30 min |
| Stage 1 | Audio extraction | 5-10 min | 3-5 min |
| Stage 2 | Scene detection | 15-20 min | 10-15 min |
| Stage 2 | Composition analysis | 20-30 min | 15-20 min |
| Stage 2 | Camera tracking | 25-35 min | 18-25 min |
| Stage 3 | Document processing | 1-2 min | 1-2 min |
| Stage 3 | Knowledge ingestion | 5-15 min | 5-15 min |
| **Total** | **All stages** | **~2 hours** | **~1.5 hours** |

*Note: Times vary based on video complexity, frame count, and CPU performance.*

### Resource Usage

- **CPU**: 80-95% utilization during parallel tasks
- **RAM**: Peak ~8-12GB (frame extraction), steady ~4-6GB
- **Disk I/O**: Sequential writes, ~5-10 MB/s average
- **Disk Space**: ~50-100GB per full-length film

---

## 🛡️ Safety Features

### Memory Management

All scripts monitor memory usage and stop if threshold exceeded:

```bash
check_memory_usage() {
    free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}'
}

if [ "$mem_pct" -gt "$MEMORY_THRESHOLD_PCT" ]; then
    log_error "Memory usage too high (${mem_pct}%)"
    exit 1
fi
```

### Disk Space Monitoring

Stage 1 checks disk space before processing:

```bash
check_disk_space() {
    df -BG "$output_dir" | awk 'NR==2 {print $4}' | sed 's/G//'
}

if [ "$disk_gb" -lt "$DISK_THRESHOLD_GB" ]; then
    log_error "Disk space too low (${disk_gb}GB)"
    exit 1
fi
```

### Error Handling

All scripts have comprehensive error handling:

```bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

trap cleanup_on_error ERR SIGTERM SIGINT

cleanup_on_error() {
    log_error "Pipeline interrupted or failed"
    stop_monitoring
    exit 1
}
```

### Checkpoint/Resume

Every processed item is checkpointed:

```bash
is_processed() {
    grep -Fxq "$item" "$checkpoint_file" && return 0 || return 1
}

save_checkpoint() {
    echo "$processed_item" >> "$checkpoint_file"
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```
[ERROR] Memory usage too high (92% > 90%)
```

**Solution**:
- Reduce number of workers: `--workers 4` (instead of 8)
- Close other applications
- Increase system swap space

**2. Disk Space Full**

```
[ERROR] Disk space too low (8GB < 10GB)
```

**Solution**:
- Free up disk space
- Use different output directory on larger disk
- Reduce JPEG quality in frame extractor (edit script)

**3. GNU Parallel Not Found**

```
[WARNING] GNU parallel not found, using sequential processing
```

**Solution**:
```bash
# Install GNU parallel
sudo apt-get install parallel

# Or use sequential mode (slower)
# Scripts will auto-fallback
```

**4. Video Files Not Found**

```
[ERROR] No video files found in: /path/to/dir
```

**Solution**:
- Verify input directory path
- Check video file extensions (mp4, mkv, avi, mov supported)
- Ensure proper permissions

**5. Checkpoint File Corruption**

```
# Delete checkpoint to restart stage
rm /path/to/output/checkpoints/frames_processed.txt
```

---

## 📝 Best Practices

### 1. Use tmux for Long-Running Jobs

```bash
# Start tmux session
tmux new -s batch_processing

# Run batch script
bash scripts/batch/run_cpu_tasks_all.sh luca /input /output

# Detach: Ctrl+B, D
# Reattach: tmux attach -t batch_processing
```

### 2. Enable Resource Monitoring

Always use `--monitor` flag for production runs:

```bash
bash scripts/batch/run_cpu_tasks_all.sh luca /input /output --monitor
```

### 3. Use Resume for Recovery

If a stage fails, use `--resume` to skip completed work:

```bash
bash scripts/batch/run_cpu_tasks_all.sh luca /input /output --resume
```

### 4. Test with Small Dataset First

Before processing full film, test with a small subset:

```bash
# Create test directory with 1-2 video clips
mkdir /tmp/test_videos
cp /path/to/film_clip.mp4 /tmp/test_videos/

# Test Stage 1
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /tmp/test_videos \
  /tmp/test_output \
  --workers 2
```

### 5. Monitor Logs in Real-Time

```bash
# Monitor master script log
tail -f /path/to/output/logs/*.log

# Monitor resource log
tail -f /path/to/output/monitoring/resources_*.csv
```

---

## 🔗 Integration with Week 9 Pipeline

These batch scripts complement the Week 9 Data Pipeline Automation:

**Week 9 Python Pipeline**:
```bash
# Execute full 4-stage pipeline (Python)
python -m scripts.scenarios.data_pipeline_automation run \
  --config examples/luca_full_pipeline.yaml
```

**Batch Scripts (Bash)**:
```bash
# Execute CPU stages (Bash)
bash scripts/batch/run_cpu_tasks_all.sh luca /input /output
```

**When to use which**:
- **Week 9 Pipeline**: For fine-grained control, DAG execution, programmatic access
- **Batch Scripts**: For production automation, parallel processing, simple CLI usage

Both can be combined:
```bash
# Use batch scripts for Stage 1-3 (CPU)
bash scripts/batch/run_cpu_tasks_all.sh luca /input /output

# Use Week 9 pipeline for Stage 4 (GPU segmentation + clustering)
python -m scripts.scenarios.data_pipeline_automation run \
  --config luca_segmentation_only.yaml
```

---

## 📚 Related Documentation

- **Week 9 Completion**: `scripts/scenarios/data_pipeline_automation/README.md`
- **Project Progress**: `PROGRESS_REPORT.md`
- **Frame Extraction**: `scripts/processing/extraction/README.md`
- **Video Analysis**: `scripts/analysis/video/README.md`
- **RAG System**: `scripts/rag/README.md`

---

## 🎯 Summary

This batch processing system provides:

✅ **Complete CPU-only automation** (no GPU required)
✅ **Parallel processing** (GNU parallel, configurable workers)
✅ **Production-ready** (error handling, checkpoints, monitoring)
✅ **Safe and robust** (memory monitoring, disk checks, rollback)
✅ **Easy to use** (simple CLI, comprehensive logging)

**One-line complete execution**:
```bash
bash scripts/batch/run_cpu_tasks_all.sh luca /input /output --workers 8 --monitor
```

---

**Version**: 1.0
**Created**: 2025-12-04
**Author**: Animation AI Studio Team
