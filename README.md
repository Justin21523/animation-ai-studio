# Animation AI Studio

Advanced multimodal AI analysis platform for animated content.

## Overview

**Animation AI Studio** extracts deep insights from animation videos through computer vision, audio analysis, and applies various AI techniques for creative applications.

### Key Features

- **Video Analysis**: Scene detection, shot composition, camera movement tracking
- **Audio Processing**: Transcription, speaker diarization, emotion detection
- **Image Enhancement**: Super-resolution, face restoration, color grading
- **Style Transfer**: Neural style transfer with temporal consistency
- **Frame Interpolation**: High-quality frame rate conversion

### Relationship to 3D Animation LoRA Pipeline

This project **shares data sources** with the [3d-animation-lora-pipeline](../3d-animation-lora-pipeline) project but serves different purposes:

- **LoRA Pipeline**: Trains LoRA adapters for character/background/pose generation
- **AI Studio**: Analyzes, processes, and transforms existing animation content

Both projects access the same:
- Film datasets (`/mnt/data/ai_data/datasets/3d-anime/`)
- Character metadata
- AI Warehouse models
- Core utilities

## Quick Start

### Installation

```bash
# Create conda environment (or use shared ai_env)
conda create -n ai_studio python=3.10
conda activate ai_studio

# Install dependencies
pip install -r requirements/all.txt
```

### Basic Usage

```bash
# Scene analysis
python scripts/analysis/video/scene_analyzer.py \
  --input /path/to/video.mp4 \
  --output outputs/analysis/film_name

# Audio transcription
python scripts/analysis/audio/transcribe.py \
  --input /path/to/video.mp4 \
  --output outputs/analysis/film_name/transcript.json

# Style transfer
python scripts/applications/style_transfer/video_style_transfer.py \
  --input /path/to/video.mp4 \
  --style /path/to/style_image.jpg \
  --output outputs/style_transfer/result.mp4
```

## Project Structure

```
scripts/
├── core/              # Shared utilities
├── analysis/          # Analysis tools (video, audio, image, style)
├── processing/        # Processing pipelines (extraction, enhancement, synthesis)
└── applications/      # End-user applications (style_transfer, interpolation, effects)

configs/               # Configuration files
data/
├── films/             # Film metadata (shared with LoRA pipeline)
└── prompts/           # Analysis prompts
outputs/               # Analysis results and processed media
requirements/          # Modular dependencies
```

## Data Sources

### Shared Film Datasets
- Location: `/mnt/data/ai_data/datasets/3d-anime/`
- Films: luca, coco, elio, onward, orion, turning-red, up
- Content: frames, audio, metadata

### AI Warehouse
- Location: `/mnt/c/AI_LLM_projects/ai_warehouse/models/`
- Shared models: CV, audio, enhancement, inpainting

## Documentation

See `CLAUDE.md` for detailed project instructions and development guidelines.

## Version

v0.1.0 - Initial setup (2025-11-16)

## License

Internal research project.
