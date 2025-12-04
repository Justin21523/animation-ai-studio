# Phase 1: CPU-Only Automation - User Guide

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2025-12-02

---

## Overview

Phase 1 provides CPU-only automation tools for content analysis, categorization, and knowledge base building. These tools run independently on CPU while GPU training continues uninterrupted.

### Key Features

- ✅ **Absolute GPU Isolation:** Never interferes with GPU training
- ✅ **Memory Safety:** Automatic batch size adjustment and emergency checkpointing
- ✅ **32-Core Utilization:** Fully leverages your 32-thread CPU
- ✅ **Remote LLM Integration:** Claude 3.5 Sonnet for vision analysis
- ✅ **Vector DB Support:** FAISS and ChromaDB for RAG preparation
- ✅ **Production Ready:** Comprehensive safety infrastructure and testing

---

## Phase 1 Scenarios

### 1. Media Asset Analyzer

Analyzes video files to extract metadata, detect scenes, and assess frame quality.

**Capabilities:**
- Video metadata extraction (resolution, FPS, duration, codec)
- Scene detection using PySceneDetect (CPU-only, ContentDetector)
- Frame quality assessment (blur detection, brightness, contrast)
- Representative frame extraction with quality filtering
- JSON report generation

**Use Cases:**
- Pre-processing video content before frame extraction
- Identifying high-quality scenes for training data
- Video content analysis and cataloging

**Example:**
```bash
# Analyze full movie for scene detection
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/media_asset_analyzer.py \
  --input /path/to/video.mp4 \
  --output /path/to/analysis_report.json \
  --extract-frames \
  --frames-dir /path/to/frames/ \
  --frames-per-scene 5 \
  --scene-threshold 27.0
```

**Output:**
```json
{
  "metadata": {
    "duration_seconds": 5400.0,
    "resolution": "1920x1080",
    "fps": 24.0,
    "codec": "h264"
  },
  "scenes": [
    {
      "scene_id": 0,
      "start_frame": 0,
      "end_frame": 120,
      "duration_seconds": 5.0
    }
  ],
  "extracted_frames": [...]
}
```

---

### 2. Auto Categorizer

Uses Claude 3.5 Sonnet to automatically categorize and tag images with semantic analysis.

**Capabilities:**
- Image analysis using Claude API with vision capabilities
- Automatic category suggestion based on content
- Tag extraction (objects, scenes, moods, styles)
- Content description generation
- Batch processing with rate limiting (50 requests/min)
- Checkpoint saving for interruption recovery

**Use Cases:**
- Organizing extracted frames by content type
- Building training datasets with semantic labels
- Content moderation and filtering

**Example:**
```bash
# Categorize extracted frames
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/auto_categorizer.py \
  --input-dir /path/to/frames/ \
  --output /path/to/categorization.json \
  --categories "character,scene,action,dialogue,background" \
  --batch-size 10
```

**Environment Variable Required:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Output:**
```json
{
  "results": [
    {
      "image_path": "/path/to/frame001.jpg",
      "primary_category": "character",
      "sub_categories": ["closeup", "dialogue"],
      "tags": ["character_portrait", "indoor_lighting", "warm_colors"],
      "description": "A close-up shot of a character in warm indoor lighting.",
      "confidence": 0.95
    }
  ],
  "category_distribution": {
    "character": 120,
    "scene": 45,
    "action": 30
  }
}
```

---

### 3. Knowledge Base Builder

Builds vector databases from documents for RAG (Retrieval Augmented Generation).

**Capabilities:**
- Document ingestion (text, markdown, PDF, images with OCR)
- Intelligent text chunking with context-preserving overlap
- CPU-based embeddings using sentence-transformers
- Vector database creation (FAISS, ChromaDB)
- Metadata extraction and indexing
- Incremental updates support

**Use Cases:**
- Building RAG knowledge bases from project documentation
- Indexing training logs and experiment notes
- Creating searchable archives of analysis reports

**Example:**
```bash
# Build knowledge base from documentation
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/knowledge_base_builder.py \
  --input-dir /mnt/c/ai_projects/animation-ai-studio/docs \
  --output-dir /mnt/data/ai_data/knowledge_bases/studio_docs \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --chunk-size 512 \
  --chunk-overlap 128 \
  --vector-db faiss \
  --batch-size 32
```

**Output:**
```
knowledge_base/
├── faiss.index              # FAISS vector index
├── metadata.json            # Chunk metadata store
├── manifest.json            # Build manifest
└── chromadb/                # (if using ChromaDB)
```

---

## Safety Infrastructure

### GPU Isolation

**4-Layer Protection Strategy:**

1. **Environment Variables:** `CUDA_VISIBLE_DEVICES=""`
2. **Python Import Guards:** `verify_no_gpu_usage()`
3. **Process Affinity:** CPU cores 0-31
4. **Subprocess Sandboxing:** Inherited CPU-only environment

**Verification:**
```bash
# Source CPU-only environment
source configs/automation/cpu_only_env.sh

# Verify configuration
verify_cpu_only
```

### Memory Monitoring

**Tiered Thresholds:**

| Level | RAM Usage | Action |
|-------|-----------|--------|
| Normal | < 70% | Full speed processing |
| Warning | 70-80% | Reduce batch size by 50% |
| Critical | 80-85% | Minimal batch, streaming mode |
| Emergency | > 85% | Save checkpoint, exit gracefully |

**Memory Budget Calculation:**
```
Available Budget = Total RAM - 18GB (system reserve)
                 = 128GB - 18GB = 110GB (aggressive mode)
```

### OOM Protection

**Process Priority Configuration:**

- Training processes: OOM score **-300** (protected)
- Automation processes: OOM score **+500** (killable first)

This ensures training is never killed by OOM, even if automation consumes excessive memory.

### Runtime Monitoring

**Background Thread Monitoring:**
- Check interval: 30 seconds
- GPU violation detection
- Memory threshold enforcement
- Automatic batch size adjustment
- Emergency checkpoint saving

**Example:**
```python
from scripts.core.safety import RuntimeMonitor

with RuntimeMonitor(check_interval=30.0) as monitor:
    # Your automation code here
    process_data()
```

---

## Configuration

### Global Resource Limits

**File:** `configs/automation/resource_limits.yaml`

```yaml
memory:
  max_process_ram_gb: 24.0        # Aggressive mode
  reserve_system_ram_gb: 18.0     # System + training reserve
  thresholds:
    warning_percent: 70.0
    critical_percent: 80.0
    emergency_percent: 85.0

cpu:
  max_threads: 32                 # Utilize all 32 cores
  nice_priority: 10               # Lower priority than training

process:
  max_concurrent_workflows: 2
  oom_score_adj: 500              # Killable before training
```

### CPU-Only Environment

**File:** `configs/automation/cpu_only_env.sh`

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
export TORCH_DEVICE="cpu"
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

# Source before running workflows
source configs/automation/cpu_only_env.sh
```

### Example Workflow

**File:** `configs/automation/example_workflow.yaml`

Complete workflow configuration demonstrating all three Phase 1 scenarios. See file for detailed settings.

---

## Dependencies

### Core Dependencies (Already Installed)

```bash
# Safety infrastructure
psutil>=5.9.0
pyyaml>=6.0

# Media analysis
scenedetect[opencv]>=0.6.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0

# LLM integration
anthropic>=0.21.0

# Knowledge base
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
chromadb>=0.4.0 (optional)

# Document processing
PyPDF2>=3.0.0 (optional)
pytesseract>=0.3.10 (optional, for OCR)
```

### Installation (If Needed)

```bash
# All Phase 1 dependencies
conda run -n ai_env pip install \
  psutil pyyaml \
  scenedetect[opencv] opencv-python pillow numpy scipy \
  anthropic \
  sentence-transformers faiss-cpu chromadb \
  PyPDF2 pytesseract
```

---

## Testing

### Comprehensive Safety Test

```bash
# Run full safety infrastructure test
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python tests/safety/test_comprehensive_safety.py
```

**Expected Output:**
```
✓ GPU Isolation (Subprocess) - PASSED
✓ Memory Monitoring - PASSED
✓ Runtime Monitoring - PASSED
✓ Preflight Checks - PASSED
✓ Configuration Loading - PASSED

TOTAL: 5/5 tests passed (100%)
```

### Scenario Testing

**Media Asset Analyzer:**
```bash
# Test on sample video (Luca film)
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/media_asset_analyzer.py \
  --input /mnt/data/datasets/general/luca/raw_videos/luca_film.ts \
  --output /tmp/test_analysis.json \
  --skip-preflight
```

**Auto Categorizer:**
```bash
# Test on sample images
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/auto_categorizer.py \
  --input-dir /tmp/test_frames \
  --output /tmp/test_categorization.json \
  --categories "test,sample" \
  --batch-size 5
```

**Knowledge Base Builder:**
```bash
# Test on documentation
PYTHONPATH=/mnt/c/ai_projects/animation-ai-studio \
conda run -n ai_env python scripts/automation/scenarios/knowledge_base_builder.py \
  --input-dir /mnt/c/ai_projects/animation-ai-studio/docs \
  --output-dir /tmp/test_kb \
  --vector-db faiss \
  --batch-size 16
```

---

## Performance Optimization

### CPU Utilization (32 Threads)

All scenarios are configured to utilize all 32 CPU threads:

- **OMP_NUM_THREADS=32** (NumPy, scikit-learn)
- **MKL_NUM_THREADS=32** (Intel Math Kernel Library)
- **OPENBLAS_NUM_THREADS=32** (OpenBLAS)
- **NUMEXPR_NUM_THREADS=32** (NumPy expression evaluation)

### Memory Budget (Aggressive Mode)

With 128GB total RAM:
- System reserve: 18GB
- GPU training: ~16GB VRAM (not RAM)
- **Available for automation: ~110GB** (aggressive)
- Typical usage: 10-24GB per workflow

### Batch Size Guidelines

**Media Analysis:**
- Scene detection: CPU-bound, no batch size needed
- Frame extraction: 10 frames per checkpoint

**Auto Categorization:**
- API rate limit: 50 requests/min (Anthropic)
- Batch size: 10 images per checkpoint
- Processing time: ~1-2 seconds per image

**Knowledge Base Building:**
- Embedding batch: 32 texts (default)
- Memory per batch: ~500MB-1GB
- Increase to 64-128 for aggressive mode

---

## Troubleshooting

### GPU Violations

**Symptom:** Safety checks detect CUDA libraries loaded

**Solution:**
```bash
# Ensure CPU-only environment is sourced
source configs/automation/cpu_only_env.sh

# Verify environment
echo $CUDA_VISIBLE_DEVICES  # Should be empty
echo $TORCH_DEVICE          # Should be "cpu"

# Re-run preflight checks
python -m scripts.core.safety.preflight_checks
```

### Memory Warnings

**Symptom:** Process exits with "Memory level critical"

**Solution:**
```yaml
# Reduce batch sizes in configs/automation/resource_limits.yaml
memory:
  thresholds:
    warning_percent: 65.0    # More conservative
    critical_percent: 75.0
```

### API Rate Limiting

**Symptom:** "Rate limit exceeded" errors from Claude API

**Solution:**
```python
# Auto Categorizer automatically handles rate limiting (1.2s between requests)
# If still hitting limits, reduce batch size:
--batch-size 5  # Process fewer images before checkpoint
```

### Scene Detection Slow

**Symptom:** Video analysis taking too long

**Solution:**
```bash
# Increase scene threshold (less sensitive = fewer scenes = faster)
--scene-threshold 35.0  # Default is 27.0

# Or disable frame extraction for initial analysis
# (remove --extract-frames flag)
```

---

## Best Practices

### 1. Always Source CPU-Only Environment

```bash
# Before running ANY automation workflow
source configs/automation/cpu_only_env.sh
```

### 2. Use Checkpointing for Long Jobs

All scenarios support automatic checkpointing:
- Media analysis: Every 10 frames
- Categorization: Every 10 images
- Knowledge base: Every 32 text chunks

Resume from checkpoint with same command.

### 3. Monitor Memory Usage

```bash
# Real-time monitoring
watch -n 2 'free -h; echo ""; ps aux | grep python | head -5'

# Or use htop
htop -u $(whoami)
```

### 4. Run Overnight for Large Datasets

```bash
# Use tmux or screen for persistence
tmux new -s automation

# Run workflow in tmux
source configs/automation/cpu_only_env.sh
python scripts/automation/scenarios/media_asset_analyzer.py ...

# Detach: Ctrl+B then D
# Reattach: tmux attach -t automation
```

### 5. Validate Output Before Proceeding

```bash
# Check JSON reports
jq '.' /path/to/report.json | head -50

# Verify frame extraction
ls -lh /path/to/frames/ | head -20

# Check knowledge base
ls -lh /path/to/kb/
```

---

## Next Steps

### Phase 2: Video Editing Automation (Week 3-4)

- **FFmpeg wrapper** for video processing (cut, trim, concat, effects)
- **Subtitle/caption automation** (ASR, translation, SRT generation)
- **Batch video editing** with templates
- **Quality enhancement** (upscaling, denoising, color correction)

### Phase 3: Multi-Modal + GPU Scheduling (Week 5-6)

- **Audio processing** (speech recognition, music separation)
- **Smart GPU scheduling** (pause training during idle, resume automatically)
- **Multi-modal analysis** (video + audio + text correlation)

### Phase 4: Production Polish (Week 7-8)

- **Web UI** for workflow management
- **CLI tool** with subcommands
- **Monitoring dashboard** (metrics, logs, resource usage)
- **Documentation finalization**

---

## Support

### Documentation

- **Safety Infrastructure:** `docs/automation/SAFETY_INFRASTRUCTURE.md`
- **API Reference:** `docs/automation/API_REFERENCE.md`
- **Configuration Guide:** `configs/automation/README.md`

### Reporting Issues

1. Check logs: `/mnt/data/ai_data/logs/automation/`
2. Run safety tests: `tests/safety/test_comprehensive_safety.py`
3. Verify environment: `source configs/automation/cpu_only_env.sh && verify_cpu_only`
4. Collect diagnostics:
   ```bash
   # System info
   python -c "import platform; print(platform.platform())"
   python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / 1e9:.1f}GB')"

   # Environment
   env | grep -E "(CUDA|TORCH|OMP|MKL)"

   # Dependencies
   conda run -n ai_env pip list | grep -E "(psutil|anthropic|sentence|faiss)"
   ```

---

## Changelog

### v1.0.0 (2025-12-02)

- ✅ Phase 1 core safety infrastructure complete
- ✅ Media asset analyzer implemented
- ✅ Auto categorizer with Claude 3.5 Sonnet implemented
- ✅ Knowledge base builder with FAISS/ChromaDB implemented
- ✅ Comprehensive safety testing (5/5 tests passed)
- ✅ 32-thread CPU optimization configured
- ✅ Production-ready configuration and documentation

---

**End of Phase 1 Guide**
