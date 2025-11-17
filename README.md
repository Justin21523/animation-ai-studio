# Animation AI Studio

**Advanced LLM-Driven AI Platform for 3D Animation Creation**

[![Status](https://img.shields.io/badge/Module-Image%20Generation%20In%20Progress-yellow)](docs/modules/module-progress.md)
[![Completion](https://img.shields.io/badge/Overall-20%25%20Complete-blue)](docs/modules/module-progress.md)

---

## 🎯 Overview

**Animation AI Studio** is an advanced multimodal AI platform that uses **open-source LLM agents** as the core decision-making engine to create, analyze, and transform 3D animated content (Pixar/Disney-style).

### Core Architecture: LLM + RAG + Agent (缺一不可)

```
Creative Studio (大壓軸) - AI 自主創作影片
    ↓
Agent Framework + RAG - LLM 理解意圖 + RAG 檢索資料 + Agent 決策
    ↓
Generation Tools - SDXL + LoRA + ControlNet + GPT-SoVITS (IN PROGRESS)
    ↓
LLM Backend - vLLM + FastAPI + Redis + Docker (COMPLETE ✅)
```

### Key Features

- **LLM Decision Engine**: Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B (self-hosted)
- **Image Generation**: SDXL + LoRA + ControlNet (character, pose, style)
- **Voice Synthesis**: GPT-SoVITS (voice cloning, emotion control)
- **Agent Framework**: LangGraph + RAG (autonomous creative decisions)
- **Video Editing**: AI-powered parody generation and effects

---

## 🚀 Quick Start

### For New LLMProvider Tooling Sessions

**English:** See [docs/guides/llm-provider-tooling-onboarding.md](docs/guides/llm-provider-tooling-onboarding.md)

**繁體中文：** 見 [docs/guides/llm-provider-tooling-onboarding.md](docs/guides/llm-provider-tooling-onboarding.md)

### For Project Context

1. **[docs/modules/module-progress.md](docs/modules/module-progress.md)** - Current implementation progress
2. **[docs/architecture/project-architecture.md](docs/architecture/project-architecture.md)** - Overall architecture
3. **[LLM_PROVIDER.md](LLM_PROVIDER.md)** - Complete project instructions

---

## 📊 Module Status

**Overall Completion:** 50% (4.5 of 9 modules complete)

| Module | Status | Completion | VRAM | Dependencies |
|--------|--------|------------|------|--------------|
| **LLM Backend** | ✅ Complete | 100% | 12-14GB | None |
| **Image Generation** | ✅ Complete | 100% | 13-15GB | LLM Backend, Model Manager |
| **Model Manager** | ✅ Complete | 100% | - | LLM Backend |
| **Voice Synthesis** | ✅ Complete | 100% | 3-4GB | LLM Backend, Model Manager |
| **RAG System** | 📋 Planned | 0% | Minimal | LLM Backend |
| **Agent Framework** | 📋 Planned | 0% | Uses LLM | RAG, Image Gen, Voice |
| **Video Analysis** | 📋 Planned | 0% | Varies | None |
| **Video Editing** | 📋 Planned | 0% | Varies | Agent Framework |
| **Creative Studio** | 📋 Planned | 0% | - | All modules |

**Status Legend:** ✅ Complete | 🔄 In Progress | 📋 Planned

**Details:** See [docs/modules/module-progress.md](docs/modules/module-progress.md)

---

## 🖥️ Hardware Configuration

**CRITICAL:** RTX 5080 16GB VRAM (single GPU)

```yaml
CPU: AMD Ryzen 9 9950X (16 cores, 32 threads)
RAM: 64GB DDR5
GPU: NVIDIA RTX 5080 16GB VRAM
PyTorch: 2.7.0 + CUDA 12.8 (IMMUTABLE)
Environment: conda ai_env
```

**Constraints:**
- Only ONE heavy model at a time (LLM OR SDXL)
- Dynamic model switching supported (20-35s)
- PyTorch SDPA only (xformers FORBIDDEN)

**See:** [docs/reference/hardware-optimization.md](docs/reference/hardware-optimization.md)

---

## 🗂️ Project Structure

```
animation-ai-studio/
├── docs/                       # 📚 All documentation
│   ├── architecture/           # Module architecture and design
│   ├── guides/                 # User guides and onboarding
│   ├── modules/                # Module implementation status
│   └── reference/              # Technical reference
├── llm_backend/                # ✅ LLM Backend (Complete)
│   ├── gateway/                # FastAPI Gateway
│   ├── services/               # vLLM configurations
│   ├── docker/                 # Docker orchestration
│   └── scripts/                # Management scripts
├── scripts/
│   ├── core/                   # Shared utilities
│   │   ├── llm_client/         # ✅ LLM client (Complete)
│   │   └── generation/         # 🔄 Model manager (Planned)
│   ├── generation/             # 🔄 Image generation (In Progress)
│   ├── synthesis/              # 📋 Voice synthesis (Planned)
│   ├── ai_editing/             # 📋 Agent framework (Planned)
│   ├── analysis/               # Video, audio, image analysis
│   └── applications/           # End-user applications
├── configs/
│   ├── generation/             # 🔄 Generation configs (In Progress)
│   └── agent/                  # 📋 Agent configs (Planned)
├── data/films/                 # Character metadata (shared with LoRA pipeline)
├── outputs/                    # Generated content
├── requirements/               # Modular dependencies
├── LLM_PROVIDER.md                   # Complete project instructions
├── OPEN_SOURCE_MODELS.md       # Models and tools reference
└── README.md                   # This file
```

---

## 📂 Data Sources

### Shared Film Datasets

**Location:** `/mnt/data/ai_data/datasets/3d-anime/`

- Films: luca, coco, elio, onward, orion, turning-red, up
- Content: frames, audio, metadata
- Shared with 3D Animation LoRA Pipeline

### AI Warehouse

**Location:** `/mnt/c/AI_LLM_projects/ai_warehouse/`

```
models/
├── llm/           # LLM models (Qwen2.5)
├── diffusion/     # SDXL, ControlNet
├── tts/           # GPT-SoVITS models
└── cv/            # Computer vision models

cache/
├── huggingface/
├── vllm/
└── diffusers/
```

---

## 🎬 Usage Examples

### LLM Backend (✅ Ready)

```bash
# Start LLM services (interactive model selection)
bash llm_backend/scripts/start_all.sh

# Check health
bash llm_backend/scripts/health_check.sh

# Python client usage
python -c "
from scripts.core.llm_client import LLMClient
import asyncio

async def main():
    async with LLMClient() as client:
        response = await client.chat(
            model='qwen-14b',
            messages=[{'role': 'user', 'content': 'Explain AI'}]
        )
        print(response)

asyncio.run(main())
"
```

### Image Generation (✅ Ready)

```python
# Character generation with LoRA
from scripts.generation.image import CharacterGenerator

generator = CharacterGenerator()
result = await generator.generate_character(
    character="luca",
    scene="running on the beach, excited expression",
    quality="high"
)
```

### Voice Synthesis (✅ Ready)

```python
# Character voice synthesis
from scripts.synthesis.tts import GPTSoVITSWrapper

synthesizer = GPTSoVITSWrapper()
audio = await synthesizer.synthesize(
    text="Silenzio, Bruno!",
    character="luca",
    emotion="excited"
)
```

---

## 📚 Documentation

### Essential Reading

1. **[LLM_PROVIDER.md](LLM_PROVIDER.md)** - Complete project instructions for LLMProvider Tooling
2. **[docs/modules/module-progress.md](docs/modules/module-progress.md)** - Current implementation status
3. **[docs/guides/llm-provider-tooling-onboarding.md](docs/guides/llm-provider-tooling-onboarding.md)** - Quick start guide
4. **[docs/reference/hardware-optimization.md](docs/reference/hardware-optimization.md)** - VRAM management

### Module Documentation

- **[docs/modules/image-generation.md](docs/modules/image-generation.md)** - Image generation architecture
- **[docs/modules/voice-synthesis.md](docs/modules/voice-synthesis.md)** - Voice synthesis architecture
- **[docs/modules/llm-backend-completion.md](docs/modules/llm-backend-completion.md)** - LLM backend completion report

### Architecture Docs

- **[docs/architecture/project-architecture.md](docs/architecture/project-architecture.md)** - Overall architecture
- **[docs/architecture/llm-backend.md](docs/architecture/llm-backend.md)** - LLM backend architecture

---

## 🔗 Related Projects

### 3D Animation LoRA Pipeline

**Location:** `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline`

**Purpose:** Train LoRA adapters for character/background/pose generation

**Current Status:**
- Luca SAM2 segmentation: 14.8% (~43h remaining)
- Next: LaMa inpainting → Batch process 6 films

**Integration:**
- Trained LoRAs will be loaded via `configs/generation/lora_registry.yaml`
- Character metadata shared via `data/films/`

---

## ⚠️ Critical Requirements

### MUST Use (Open-Source Only)

- ✅ Qwen2.5-VL, Qwen2.5-14B, Qwen2.5-Coder (LLM)
- ✅ vLLM (self-hosted backend)
- ✅ SDXL + LoRA (image generation)
- ✅ GPT-SoVITS (voice synthesis)
- ✅ LangGraph (agent framework)
- ✅ PyTorch 2.7.0 + CUDA 12.8

### MUST NOT Use

- ❌ Ollama (we use vLLM)
- ❌ GPT-4, LLMProvider, Gemini (closed-source)
- ❌ Any paid APIs
- ❌ xformers (breaks PyTorch compatibility)
- ❌ Modify PyTorch or CUDA versions

---

## 🎓 Key Concepts

### LLM as Creative Brain

Not just a tool - LLM makes creative decisions:
- Understands artistic intent
- Plans execution steps
- Selects appropriate tools
- Evaluates quality
- Iterates until perfect

### RAG for Context

Retrieves relevant information:
- Character descriptions
- Style guides
- Past generations
- Film analysis

### Agent for Automation

Autonomous workflow execution:
- Tool calling
- Multi-step planning
- Quality-driven iteration
- Self-improvement

---

## 📞 Getting Help

**For New Sessions:** [docs/guides/llm-provider-tooling-onboarding.md](docs/guides/llm-provider-tooling-onboarding.md)

**For Architecture:** [docs/architecture/project-architecture.md](docs/architecture/project-architecture.md)

**For Current Status:** [docs/modules/module-progress.md](docs/modules/module-progress.md)

**For Hardware:** [docs/reference/hardware-optimization.md](docs/reference/hardware-optimization.md)

**For Models:** [OPEN_SOURCE_MODELS.md](OPEN_SOURCE_MODELS.md)

---

## 📊 Progress

**Version:** v0.7.0
**Last Updated:** 2025-11-17
**Current Focus:** RAG System (Next module)
**Overall Completion:** 50% (4.5 of 9 modules complete)

**Module Status:**
- ✅ LLM Backend (100%)
- ✅ Image Generation (100%)
- ✅ Model Manager (100%)
- ✅ Voice Synthesis (100%)
- 📋 RAG, Agent Framework, Video Analysis, Video Editing, Creative Studio (0%)

**See [docs/modules/module-progress.md](docs/modules/module-progress.md) for detailed progress tracking.**

---

## 📄 License

Internal research project.
