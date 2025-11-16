# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Animation AI Studio** is an advanced multimodal AI analysis platform for animated content. This project focuses on **extracting deep insights from animation videos** through computer vision, audio analysis, and applying various AI techniques for creative applications.

**Key Distinction from 3d-animation-lora-pipeline:**
- **LoRA Pipeline**: Trains LoRA adapters for character/background/pose generation
- **AI Studio**: Analyzes, processes, and transforms existing animation content using SOTA AI models

This project **shares data sources** with the LoRA pipeline project (same film datasets, character metadata) but serves different purposes.

## Code Language Convention

- All code and comments: **English only**
- User-facing summaries and explanations: **Traditional Chinese** (when user requests)

## Architecture

### Core Capabilities
```
1. Video Analysis
   - Scene detection and segmentation
   - Shot composition analysis
   - Camera movement tracking
   - Temporal consistency analysis

2. Audio Analysis
   - Speech recognition and transcription
   - Speaker diarization
   - Music/SFX separation
   - Emotion detection from voice

3. Image Analysis
   - Style transfer and artistic effects
   - Super-resolution and enhancement
   - Color grading and tone mapping
   - Composition analysis

4. Advanced Processing
   - Frame interpolation (RIFE, IFRNet)
   - Video synthesis and editing
   - Multi-modal fusion (audio + visual)
   - Temporal coherence enhancement
```

### Directory Structure
```
scripts/
├── core/              # Shared utilities (config, logging, model loading)
│   ├── utils/         # config_loader.py, logger.py, path_utils.py, etc.
│   ├── models/        # model_loader.py, model_paths.py
│   ├── face_matching/ # ArcFace and identity tools
│   └── diversity/     # Diversity metrics
├── analysis/          # Analysis tools
│   ├── video/         # Scene detection, shot analysis, composition
│   ├── audio/         # Transcription, diarization, emotion
│   ├── image/         # Style analysis, composition metrics
│   └── style/         # Style classification and extraction
├── processing/        # Processing pipelines
│   ├── extraction/    # Frame/audio extraction
│   ├── enhancement/   # Super-resolution, denoising, color grading
│   └── synthesis/     # Interpolation, video generation
└── applications/      # End-user applications
    ├── style_transfer/ # Neural style transfer
    ├── interpolation/  # Frame rate conversion
    └── effects/        # Various AI effects

configs/               # Configuration files
data/
├── films/             # Film metadata (shared with LoRA pipeline)
│   ├── luca/          # Character info, scene descriptions
│   ├── coco/
│   └── ...
└── prompts/           # Analysis prompts and templates
outputs/               # Analysis results and processed media
requirements/          # Modular dependencies
```

### Key Components

**Configuration System** (`scripts/core/utils/config_loader.py`):
- Shared with LoRA pipeline project
- Uses OmegaConf to load YAML configs
- Automatically converts relative paths to absolute

**Shared Data Sources**:
- **Film datasets**: `/mnt/data/ai_data/datasets/3d-anime/` (frames, audio, metadata)
- **Character metadata**: `data/films/{film}/characters/` (copied from LoRA pipeline)
- **AI Warehouse**: `/mnt/c/AI_LLM_projects/ai_warehouse/models/` (shared models)

**Analysis Pipelines**:
- **Video**: Shot detection, composition analysis, camera movement
- **Audio**: Transcription (Whisper), diarization, emotion analysis
- **Image**: Style transfer, enhancement, color analysis

## Common Workflows

### 1. Scene Analysis Workflow
```bash
# Extract frames with scene detection
python scripts/processing/extraction/universal_frame_extractor.py \
  --input /path/to/video.mp4 \
  --output outputs/analysis/film_name/frames \
  --mode scene \
  --scene-threshold 0.3

# Analyze shot composition
python scripts/analysis/video/shot_composition_analyzer.py \
  --input outputs/analysis/film_name/frames \
  --output outputs/analysis/film_name/composition_analysis.json

# Extract camera movement
python scripts/analysis/video/camera_movement_tracker.py \
  --input /path/to/video.mp4 \
  --output outputs/analysis/film_name/camera_analysis.json
```

### 2. Audio Analysis Workflow
```bash
# Extract audio
python scripts/processing/extraction/audio_extractor.py \
  --input /path/to/video.mp4 \
  --output outputs/analysis/film_name/audio.wav

# Transcribe with speaker diarization
python scripts/analysis/audio/transcribe_with_speakers.py \
  --input outputs/analysis/film_name/audio.wav \
  --output outputs/analysis/film_name/transcript.json \
  --model whisper-large-v3 \
  --diarize

# Analyze emotion from speech
python scripts/analysis/audio/emotion_analyzer.py \
  --input outputs/analysis/film_name/audio.wav \
  --transcript outputs/analysis/film_name/transcript.json \
  --output outputs/analysis/film_name/emotion_timeline.json
```

### 3. Style Transfer Workflow
```bash
# Apply neural style transfer
python scripts/applications/style_transfer/neural_style_transfer.py \
  --content outputs/analysis/film_name/frames \
  --style /path/to/style_image.jpg \
  --output outputs/style_transfer/film_name_styled \
  --method adain \
  --preserve-temporal

# Video-to-video style transfer with temporal consistency
python scripts/applications/style_transfer/video_style_transfer.py \
  --input /path/to/video.mp4 \
  --style /path/to/style_image.jpg \
  --output outputs/style_transfer/video_styled.mp4 \
  --method ebsynth \
  --keyframe-interval 10
```

### 4. Enhancement Workflow
```bash
# Upscale frames with super-resolution
python scripts/processing/enhancement/super_resolution.py \
  --input outputs/analysis/film_name/frames \
  --output outputs/enhanced/film_name_4k \
  --model realesrgan-x4plus \
  --scale 4

# Color grading
python scripts/processing/enhancement/color_grading.py \
  --input outputs/analysis/film_name/frames \
  --output outputs/graded/film_name \
  --lut /path/to/lut.cube \
  --strength 0.8
```

## Data Sources

### Shared with LoRA Pipeline
- **Film frames**: `/mnt/data/ai_data/datasets/3d-anime/{film}/frames`
- **Character metadata**: `data/films/{film}/characters/`
- **Film information**: `data/films/{film}/README.md`

### AI Warehouse (Shared Models)
- **Computer Vision**: `/mnt/c/AI_LLM_projects/ai_warehouse/models/cv/`
- **Audio Models**: `/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/`
- **Enhancement**: `/mnt/c/AI_LLM_projects/ai_warehouse/models/enhancement/`
- **Inpainting**: `/mnt/c/AI_LLM_projects/ai_warehouse/models/inpainting/`

## Key Technologies

### Video Analysis
- **PySceneDetect**: Shot and scene detection
- **OpenCV**: Basic video processing
- **Optical Flow (RAFT)**: Motion analysis and tracking

### Audio Processing
- **Whisper**: Speech transcription
- **Pyannote**: Speaker diarization
- **Wav2Vec2**: Speech emotion recognition
- **Demucs**: Audio source separation

### Enhancement
- **Real-ESRGAN**: Super-resolution upscaling
- **CodeFormer**: Face enhancement
- **GFPGAN**: Face restoration
- **ColorMNet**: Color grading

### Style Transfer
- **AdaIN**: Adaptive instance normalization style transfer
- **EbSynth**: Video stylization with temporal consistency
- **Neural Style Transfer**: Classic Gatys et al. method

### Frame Interpolation
- **RIFE**: Real-time intermediate flow estimation
- **IFRNet**: Intermediate frame synthesis

## Development Principles

1. **Code & comments in English.** Summaries to user in Chinese when asked.
2. **Modular design:** Each analysis/processing tool is standalone CLI script
3. **Determinism:** Seed RNGs, write artifacts to timestamped output directories
4. **Shared resources:** Leverage existing film data and character metadata
5. **GPU efficiency:** Batch processing, memory management, multi-GPU support
6. **Quality first:** Validate outputs, provide quality metrics

## Output Contracts

**Analysis outputs** should include:
- JSON metadata files with structured results
- Visualization outputs (plots, annotated frames)
- Summary reports (markdown or HTML)
- Timestamped directories for reproducibility

**Processing outputs** should include:
- Processed media files (images, videos, audio)
- Processing logs and parameters
- Quality metrics (PSNR, SSIM, etc.)
- Side-by-side comparisons

## Integration with LoRA Pipeline

While these are separate projects, they share:
1. **Film datasets**: Both access same source frames and metadata
2. **Character information**: Both use same character descriptions
3. **Core utilities**: Logging, config loading, path utilities
4. **AI Warehouse**: Shared model weights

**Workflow example:**
```
LoRA Pipeline: Film → Frames → Segmentation → Clustering → LoRA Training
AI Studio:     Film → Frames → Analysis → Enhancement → Style Transfer
                      ↓
                 (Shared frame source)
```

## Environment

- Python 3.10
- CUDA GPU (required for most models)
- PyTorch 2.7.1 + CUDA 12.8
- Conda environment: `ai_env` (shared with LoRA pipeline)
- All paths configurable via configs

## Important Notes

1. **Data paths:** All production data under `/mnt/data/ai_data/` warehouse
2. **GPU memory:** Most models require 16GB+ VRAM
3. **Dependencies:** Modular requirements (core, video, audio, enhancement)
4. **Long processes:** Use nohup/tmux for overnight jobs
5. **Logging:** Stream logs to `logs/` directory

## Future Directions

- Multi-modal emotion analysis (visual + audio)
- Automated scene understanding and tagging
- Character interaction analysis
- Music-driven visual effects
- Real-time style transfer pipeline
- 3D reconstruction from 2D animation

## Version

v0.1.0 - Initial setup (2025-11-16)
