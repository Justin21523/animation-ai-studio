# Testing and Usage Guide

**Animation AI Studio - Week 9 Pipeline & CPU Automation Scripts**

This guide provides comprehensive testing procedures and usage examples for both the Week 9 Data Pipeline Automation and the CPU Automation Batch Scripts.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Week 9 Pipeline Testing](#week-9-pipeline-testing)
3. [CPU Batch Scripts Testing](#cpu-batch-scripts-testing)
4. [Integration Testing](#integration-testing)
5. [Performance Benchmarking](#performance-benchmarking)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements

**Hardware:**
- CPU: 8+ cores (16 cores recommended for parallel processing)
- RAM: 16GB+ (32GB recommended)
- Disk: 50-100GB free space per film
- GPU: Optional (only for segmentation stage in Week 9 pipeline)

**Software:**
- Python 3.10+
- Conda environment: `ai_env`
- GNU parallel (for batch scripts)
- FFmpeg (for audio extraction)
- PySceneDetect (for scene detection)

### Installation

```bash
# Activate conda environment
conda activate ai_env

# Install GNU parallel (if not installed)
sudo apt-get install parallel

# Verify tools
which python3
which ffmpeg
which scenedetect
which parallel

# Verify project path
cd /mnt/c/ai_projects/animation-ai-studio
```

### Test Data Preparation

```bash
# Create test directory
mkdir -p /tmp/animation_ai_test
mkdir -p /tmp/animation_ai_test/videos

# Copy a small video clip for testing (1-2 minutes)
# Example: Extract first 2 minutes from a film
ffmpeg -i /mnt/c/raw_videos/luca/luca_part1.mp4 \
  -t 120 \
  -c copy \
  /tmp/animation_ai_test/videos/luca_test_clip.mp4

# Verify test file
ls -lh /tmp/animation_ai_test/videos/
```

---

## 🧪 Week 9 Pipeline Testing

### Test 1: Pipeline Validation (Dry-Run)

**Purpose**: Validate YAML configuration without execution

```bash
# Test with the included Luca example
python -m scripts.scenarios.data_pipeline_automation validate \
  scripts/scenarios/data_pipeline_automation/examples/luca_full_pipeline.yaml
```

**Expected Output**:
```
✓ Pipeline validation successful
✓ 4 stages defined
✓ No circular dependencies
✓ All dependencies valid
✓ All required config keys present
```

**If validation fails**:
- Check YAML syntax (indentation, quotes)
- Verify stage IDs are unique
- Verify dependency IDs exist
- Check required config keys

### Test 2: Dry-Run Execution

**Purpose**: Test pipeline execution plan without running scripts

```bash
# Create a minimal test pipeline
cat > /tmp/test_pipeline.yaml <<'EOF'
name: "Test Frame Extraction"
stages:
  - id: extract_frames
    type: frame_extraction
    config:
      input_dir: "/tmp/animation_ai_test/videos"
      output_dir: "/tmp/animation_ai_test/frames"
      mode: "scene"
      scene_threshold: 27.0
      frames_per_scene: 3
      jpeg_quality: 95
      workers: 2
EOF

# Dry-run test
python -m scripts.scenarios.data_pipeline_automation run \
  --config /tmp/test_pipeline.yaml \
  --dry-run

# Check execution plan
cat /tmp/animation_ai_test/execution_plan.txt
```

**Expected Output**:
```
[DRY-RUN] Pipeline: Test Frame Extraction
[DRY-RUN] Execution plan:
  Stage 1: extract_frames (frame_extraction)
[DRY-RUN] Would execute: python scripts/processing/extraction/universal_frame_extractor.py ...
[DRY-RUN] Dry-run complete (no files created)
```

### Test 3: Single Stage Execution

**Purpose**: Test frame extraction stage in isolation

```bash
# Execute frame extraction only
python -m scripts.scenarios.data_pipeline_automation run \
  --config /tmp/test_pipeline.yaml \
  --log-level DEBUG

# Verify outputs
ls -la /tmp/animation_ai_test/frames/
cat /tmp/animation_ai_test/frames/extraction_metadata.json
```

**Success Criteria**:
- ✅ Frames extracted to output directory
- ✅ `extraction_metadata.json` created
- ✅ Logs show no errors
- ✅ Frame count matches expected scenes

### Test 4: Multi-Stage Pipeline

**Purpose**: Test dependency resolution and data flow

```bash
# Create 2-stage pipeline
cat > /tmp/test_multi_stage.yaml <<'EOF'
name: "Test Multi-Stage"
stages:
  - id: extract_frames
    type: frame_extraction
    config:
      input_dir: "/tmp/animation_ai_test/videos"
      output_dir: "/tmp/animation_ai_test/frames"
      mode: "scene"
      scene_threshold: 27.0

  - id: cluster_characters
    type: clustering
    depends_on: [extract_frames]
    config:
      instances_dir: "{extract_frames.output_dir}/instances"
      output_dir: "/tmp/animation_ai_test/clustered"
      method: "kmeans"
      k_range: [5, 10]
EOF

# Execute pipeline
python -m scripts.scenarios.data_pipeline_automation run \
  --config /tmp/test_multi_stage.yaml \
  --checkpoint-dir /tmp/animation_ai_test/checkpoints
```

**Success Criteria**:
- ✅ Stage 1 completes before Stage 2
- ✅ Template variable `{extract_frames.output_dir}` resolved
- ✅ Stage 2 receives correct input from Stage 1
- ✅ Checkpoints created for resume

### Test 5: Checkpoint/Resume

**Purpose**: Test resume from failure

```bash
# Simulate failure by killing process mid-execution
python -m scripts.scenarios.data_pipeline_automation run \
  --config /tmp/test_multi_stage.yaml \
  --checkpoint-dir /tmp/animation_ai_test/checkpoints &

PID=$!
sleep 30  # Let Stage 1 run for 30 seconds
kill $PID  # Simulate failure

# Resume from checkpoint
python -m scripts.scenarios.data_pipeline_automation run \
  --config /tmp/test_multi_stage.yaml \
  --checkpoint-dir /tmp/animation_ai_test/checkpoints \
  --resume

# Verify checkpoint files
ls -la /tmp/animation_ai_test/checkpoints/
```

**Success Criteria**:
- ✅ Pipeline detects existing checkpoint
- ✅ Skips completed stages
- ✅ Resumes from last incomplete stage
- ✅ Final output matches full execution

---

## 🖥️ CPU Batch Scripts Testing

### Test 1: Stage 1 - Data Preparation

**Purpose**: Test parallel frame and audio extraction

```bash
# Run Stage 1 with 4 workers (small test)
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /tmp/animation_ai_test/videos \
  /tmp/animation_ai_test/batch_output \
  --workers 4

# Verify outputs
ls -la /tmp/animation_ai_test/batch_output/frames/
ls -la /tmp/animation_ai_test/batch_output/audio/
cat /tmp/animation_ai_test/batch_output/dataset_index.json
```

**Success Criteria**:
- ✅ Frames extracted (JPEG files)
- ✅ Audio extracted (WAV files)
- ✅ `dataset_index.json` created
- ✅ Logs in `logs/` directory
- ✅ Checkpoint files in `checkpoints/`

**Expected Metrics**:
```json
{
  "stage": "cpu_stage1_data_prep",
  "total_videos": 1,
  "frames_extracted": 120,  // ~60 frames/minute for 2-min clip
  "audio_files": 1,
  "workers_used": 4,
  "completed": true
}
```

### Test 2: Stage 1 - Resume Test

**Purpose**: Test checkpoint/resume functionality

```bash
# Create a larger test set (3-4 video clips)
# ... copy multiple clips to /tmp/animation_ai_test/videos

# Start processing
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /tmp/animation_ai_test/videos \
  /tmp/animation_ai_test/batch_output \
  --workers 4 &

PID=$!
sleep 60  # Let it process 1-2 videos
kill $PID  # Simulate failure

# Resume with --resume flag
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /tmp/animation_ai_test/videos \
  /tmp/animation_ai_test/batch_output \
  --workers 4 \
  --resume
```

**Success Criteria**:
- ✅ Skips already processed videos
- ✅ Continues with remaining videos
- ✅ Log shows "Skipping (already processed): ..."
- ✅ Final output complete

### Test 3: Stage 2 - Video Analysis

**Purpose**: Test scene detection, composition, camera tracking

**Prerequisites**: Complete Stage 1 first

```bash
# Run Stage 2 analysis
bash scripts/batch/cpu_tasks_stage2_analysis.sh \
  /tmp/animation_ai_test/batch_output \
  --workers 4

# Verify outputs
ls -la /tmp/animation_ai_test/batch_output/analysis/scenes/
ls -la /tmp/animation_ai_test/batch_output/analysis/composition/
ls -la /tmp/animation_ai_test/batch_output/analysis/camera/
cat /tmp/animation_ai_test/batch_output/analysis/analysis_summary.json
```

**Success Criteria**:
- ✅ Scene detection JSON files created
- ✅ Composition analysis JSON files created
- ✅ Camera tracking JSON files created
- ✅ `analysis_summary.json` created

**Expected Scene Detection Output**:
```json
{
  "video": "luca_test_clip",
  "scenes": [
    {"scene_id": 0, "start_frame": 0, "end_frame": 45},
    {"scene_id": 1, "start_frame": 46, "end_frame": 98}
  ],
  "total_scenes": 2
}
```

### Test 4: Stage 3 - RAG Preparation

**Purpose**: Test CPU-only embeddings and knowledge base creation

**Prerequisites**: Create test film data

```bash
# Create test film data
mkdir -p /tmp/animation_ai_test/film_data/luca/characters
cat > /tmp/animation_ai_test/film_data/luca/README.md <<'EOF'
# Luca

An animated film about a sea monster boy.
EOF

cat > /tmp/animation_ai_test/film_data/luca/characters/luca.md <<'EOF'
# Luca Paguro

- Species: Sea monster
- Age: 13 years old
- Appearance: Brown hair, green eyes, curious expression
- Personality: Adventurous, curious, kind
EOF

# Run Stage 3 (CPU-only embeddings)
bash scripts/batch/cpu_tasks_stage3_rag_prep.sh \
  luca \
  /tmp/animation_ai_test/batch_output \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Verify outputs
ls -la /tmp/animation_ai_test/batch_output/rag/documents/
ls -la /tmp/animation_ai_test/batch_output/rag/knowledge_base/
cat /tmp/animation_ai_test/batch_output/rag/rag_summary.json
```

**Success Criteria**:
- ✅ Documents processed to `rag/documents/`
- ✅ FAISS index created (CPU)
- ✅ `index_metadata.json` shows `device: cpu`
- ✅ No GPU usage (verify with `nvidia-smi`)

**Verify CPU-Only**:
```bash
# Monitor during Stage 3 execution
watch -n 1 nvidia-smi

# GPU memory should be 0% used during embedding generation
```

### Test 5: Resource Monitoring

**Purpose**: Test real-time resource monitoring

```bash
# Start monitoring in foreground (10-second intervals)
bash scripts/batch/monitor_resources.sh --interval 10
```

**Expected Output (Live)**:
```
=========================================
Resource Monitor - 2025-12-04 10:30:00
=========================================
CPU:
  Cores: 16
  Overall: 45.3%
  Per-core: Core0:42.1% Core1:48.5% ...
RAM:
  Used: 12.5GB / 32.0GB (39.1%)
GPU:
  VRAM: 0MB / 16384MB (0.0%)
  Utilization: 0%
  Temperature: 35°C
Disk:
  /     : 1.2TB / 2.0TB (60%)
  /mnt  : 450GB / 1.0TB (45%)
Top CPU Consumers:
  python(8.5%) ffmpeg(12.3%) bash(2.1%)
=========================================
```

**Test Daemon Mode**:
```bash
# Start in background daemon mode
bash scripts/batch/monitor_resources.sh \
  --daemon \
  --log-dir /tmp/monitoring_test \
  --interval 30

# Check daemon running
ps aux | grep monitor_resources

# View CSV log (real-time)
tail -f /tmp/monitoring_test/resources_*.csv

# Stop daemon
pkill -f monitor_resources.sh
```

### Test 6: Master Script (All 3 Stages)

**Purpose**: Test complete pipeline orchestration

```bash
# Execute all 3 stages with monitoring
bash scripts/batch/run_cpu_tasks_all.sh \
  luca \
  /tmp/animation_ai_test/videos \
  /tmp/animation_ai_test/complete_output \
  --workers 4 \
  --monitor

# Monitor execution
tail -f /tmp/animation_ai_test/complete_output/logs/*.log

# After completion, verify all outputs
tree /tmp/animation_ai_test/complete_output
```

**Expected Output Structure**:
```
/tmp/animation_ai_test/complete_output/
├── frames/
│   └── luca_test_clip/
├── audio/
│   └── luca_test_clip_audio.wav
├── analysis/
│   ├── scenes/
│   ├── composition/
│   └── camera/
├── rag/
│   ├── documents/
│   └── knowledge_base/
├── monitoring/
│   ├── resources_20251204_103000.csv
│   └── monitor_daemon.log
├── logs/
├── checkpoints/
├── dataset_index.json
├── analysis_summary.json
├── rag_summary.json
└── execution_metadata.json
```

**Success Criteria**:
- ✅ All 3 stages complete successfully
- ✅ Total execution time logged
- ✅ Final summary shows all statistics
- ✅ Monitoring logs created
- ✅ No errors in logs

---

## 🔗 Integration Testing

### Test 1: Week 9 Pipeline → CPU Batch Scripts

**Purpose**: Use Week 9 pipeline for GPU stages, then CPU scripts for analysis

```bash
# Step 1: Use Week 9 for frame extraction
python -m scripts.scenarios.data_pipeline_automation run \
  --config /tmp/test_pipeline.yaml

# Step 2: Use CPU batch for analysis (Stage 2)
bash scripts/batch/cpu_tasks_stage2_analysis.sh \
  /tmp/animation_ai_test/frames \
  --workers 8

# Step 3: Use CPU batch for RAG (Stage 3)
bash scripts/batch/cpu_tasks_stage3_rag_prep.sh \
  luca \
  /tmp/animation_ai_test/frames
```

### Test 2: CPU Batch Scripts → Week 9 Pipeline

**Purpose**: Use CPU batch for data prep, then Week 9 for segmentation

```bash
# Step 1: CPU batch for Stage 1
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /tmp/animation_ai_test/videos \
  /tmp/animation_ai_test/output \
  --workers 8

# Step 2: Week 9 pipeline for segmentation + clustering
python -m scripts.scenarios.data_pipeline_automation run \
  --config segmentation_clustering_pipeline.yaml
```

---

## 📊 Performance Benchmarking

### Benchmark Test Setup

```bash
# Prepare full-length test video (Luca - 95 minutes)
FILM_PATH="/mnt/c/raw_videos/luca/luca.mp4"
OUTPUT_BASE="/tmp/benchmark_test"

# Clear cache and drop caches (Linux)
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Start system monitoring
bash scripts/batch/monitor_resources.sh \
  --daemon \
  --log-dir "${OUTPUT_BASE}/monitoring" \
  --interval 5 &
MONITOR_PID=$!
```

### Benchmark 1: Stage 1 (Frame Extraction)

```bash
# Test with different worker counts
for WORKERS in 4 8 12 16; do
  echo "Testing with ${WORKERS} workers..."

  time bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
    "${FILM_PATH}" \
    "${OUTPUT_BASE}/workers_${WORKERS}" \
    --workers "${WORKERS}" \
    2>&1 | tee "${OUTPUT_BASE}/benchmark_stage1_w${WORKERS}.log"

  # Extract timing
  grep "Total execution time" "${OUTPUT_BASE}/benchmark_stage1_w${WORKERS}.log"
done

# Stop monitoring
kill ${MONITOR_PID}

# Analyze results
python -c "
import json
import glob

for log in glob.glob('${OUTPUT_BASE}/**/dataset_index.json', recursive=True):
    with open(log) as f:
        data = json.load(f)
        print(f\"{log}: {data['frames_extracted']} frames, {data['workers_used']} workers\")
"
```

**Expected Performance (AMD Ryzen 9 9950X, 16 cores)**:

| Workers | Time | Frames/sec | CPU Util |
|---------|------|------------|----------|
| 4 | 45 min | 2.2 | ~60% |
| 8 | 30 min | 3.3 | ~80% |
| 12 | 22 min | 4.5 | ~90% |
| 16 | 20 min | 5.0 | ~95% |

### Benchmark 2: End-to-End Pipeline

```bash
# Full pipeline benchmark
time bash scripts/batch/run_cpu_tasks_all.sh \
  luca \
  /mnt/c/raw_videos/luca \
  /tmp/full_benchmark \
  --workers 16 \
  --monitor \
  2>&1 | tee /tmp/full_pipeline_benchmark.log

# Extract metrics
grep "Total execution time" /tmp/full_pipeline_benchmark.log
grep "Frames extracted" /tmp/full_pipeline_benchmark.log
grep "Scene detection" /tmp/full_pipeline_benchmark.log
```

**Expected Total Time**: 1.5-2 hours for 95-minute film (16 workers)

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms**:
```
[ERROR] Memory usage too high (92% > 90%)
```

**Solutions**:
1. Reduce worker count:
   ```bash
   --workers 4  # Instead of 8 or 16
   ```

2. Clear checkpoints and retry with fewer workers:
   ```bash
   rm -rf /path/to/output/checkpoints/*
   bash scripts/batch/cpu_tasks_stage1_data_prep.sh ... --workers 4 --resume
   ```

3. Close other applications:
   ```bash
   # Check memory hogs
   ps aux --sort=-%mem | head -n 10
   ```

### Issue 2: Disk Space Full

**Symptoms**:
```
[ERROR] Disk space too low (8GB < 10GB)
```

**Solutions**:
1. Free up disk space:
   ```bash
   # Remove old outputs
   rm -rf /tmp/animation_ai_test/old_outputs

   # Clear system cache
   sudo apt-get clean
   ```

2. Use different output directory (larger disk):
   ```bash
   bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
     /input/dir \
     /mnt/large_disk/output \  # Different disk
     --workers 8
   ```

3. Reduce JPEG quality (in script):
   ```bash
   # Edit cpu_tasks_stage1_data_prep.sh
   # Change: --jpeg-quality 95
   # To:     --jpeg-quality 85  # Smaller files
   ```

### Issue 3: GNU Parallel Not Found

**Symptoms**:
```
[WARNING] GNU parallel not found, using sequential processing
```

**Solution**:
```bash
# Install GNU parallel
sudo apt-get update
sudo apt-get install parallel

# Verify installation
which parallel
parallel --version
```

### Issue 4: GPU Usage in CPU Scripts

**Symptoms**:
```
nvidia-smi shows GPU memory usage during Stage 3
```

**Solutions**:
1. Verify CUDA_VISIBLE_DEVICES:
   ```bash
   # Check in script
   grep "CUDA_VISIBLE_DEVICES" scripts/batch/cpu_tasks_stage3_rag_prep.sh

   # Should show: export CUDA_VISIBLE_DEVICES=""
   ```

2. Manually force CPU:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   bash scripts/batch/cpu_tasks_stage3_rag_prep.sh ...
   ```

3. Verify model device:
   ```bash
   # Check embedding model logs
   grep "device" /path/to/output/rag/logs/knowledge_ingestion.log

   # Should show: device='cpu'
   ```

### Issue 5: Checkpoint File Corruption

**Symptoms**:
```
Resume doesn't work, processes same files again
```

**Solution**:
```bash
# Delete corrupted checkpoints
rm /path/to/output/checkpoints/*.txt

# Re-run without --resume (fresh start)
bash scripts/batch/cpu_tasks_stage1_data_prep.sh ... # No --resume flag
```

### Issue 6: SceneDetect Command Not Found

**Symptoms**:
```
scenedetect: command not found
```

**Solution**:
```bash
# Install PySceneDetect
pip install scenedetect[opencv]

# Verify installation
scenedetect version

# Alternative: Use Python script fallback
# (Script will auto-fallback to scripts/analysis/video/scene_detection.py)
```

---

## ✅ Test Completion Checklist

### Week 9 Pipeline Tests
- [ ] Pipeline validation (dry-run) successful
- [ ] Single stage execution working
- [ ] Multi-stage dependency resolution working
- [ ] Template variable substitution working
- [ ] Checkpoint/resume functionality verified
- [ ] All 4 stage executors tested individually

### CPU Batch Scripts Tests
- [ ] Stage 1 (data prep) working with parallel processing
- [ ] Stage 2 (analysis) working with all 3 analyzers
- [ ] Stage 3 (RAG) working with CPU-only embeddings
- [ ] Resource monitoring working (foreground and daemon)
- [ ] Master script executing all 3 stages successfully
- [ ] Checkpoint/resume working across all stages
- [ ] No GPU usage in CPU scripts (verified with nvidia-smi)

### Integration Tests
- [ ] Week 9 → CPU batch integration working
- [ ] CPU batch → Week 9 integration working
- [ ] Outputs compatible between systems

### Performance Tests
- [ ] Benchmark completed for different worker counts
- [ ] End-to-end pipeline benchmark completed
- [ ] Resource monitoring logs analyzed
- [ ] Performance meets expectations (1.5-2 hours for full film)

---

## 📚 Additional Resources

- **Week 9 Pipeline Documentation**: `scripts/scenarios/data_pipeline_automation/DESIGN.md`
- **CPU Batch Scripts Documentation**: `scripts/batch/README.md`
- **Project Progress Report**: `PROGRESS_REPORT.md`
- **Overall Project README**: `README.md`

---

## 🎯 Quick Test Command Summary

```bash
# Week 9 Pipeline - Quick Test
python -m scripts.scenarios.data_pipeline_automation validate \
  scripts/scenarios/data_pipeline_automation/examples/luca_full_pipeline.yaml

# CPU Batch - Quick Test (Stage 1 only)
bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
  /tmp/test/videos \
  /tmp/test/output \
  --workers 4

# CPU Batch - Complete Pipeline
bash scripts/batch/run_cpu_tasks_all.sh \
  test_film \
  /tmp/test/videos \
  /tmp/test/output \
  --workers 4 \
  --monitor

# Resource Monitoring
bash scripts/batch/monitor_resources.sh --interval 10
```

---

**Document Version:** 1.0
**Created:** 2025-12-04
**Author:** Animation AI Studio Team
