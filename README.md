# Animation AI Studio

**Advanced LLM-Driven AI Platform for 3D Animation Creation**

[![Status](https://img.shields.io/badge/Status-Week%203--4%20In%20Progress-yellow)](docs/reports/project-milestones.md)
[![Phase](https://img.shields.io/badge/Phase-25%25%20Complete-blue)](docs/reports/project-milestones.md)

---

## 🎯 Overview

**Animation AI Studio** is an advanced multimodal AI platform that uses **open-source LLM agents** as the core decision-making engine to create, analyze, and transform 3D animated content (Pixar/Disney-style).

### Core Architecture: LLM + RAG + Agent (缺一不可)

```
Week 7-8: AI Video Editing (大壓軸) - AI 自主創作影片
    ↓
Week 5-6: LangGraph Agent + RAG - LLM 理解意圖 + RAG 檢索資料 + Agent 決策
    ↓
Week 3-4: 3D Character Tools - SDXL + LoRA + ControlNet + GPT-SoVITS (IN PROGRESS)
    ↓
Week 1-2: LLM Backend - vLLM + FastAPI + Redis + Docker (COMPLETE ✅)
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

1. **[docs/architecture/project-architecture.md](docs/architecture/project-architecture.md)** - Overall architecture and implementation plan
2. **[LLM_PROVIDER.md](LLM_PROVIDER.md)** - Complete project instructions
3. **[docs/reports/project-milestones.md](docs/reports/project-milestones.md)** - Current progress

---

## 📊 Current Status

**Phase:** Week 3-4 - 3D Character Generation Tools (IN PROGRESS)

**Progress:** 25% Complete (Week 1-2 of 8)

| Week | Goal | Status |
|------|------|--------|
| 1-2 | LLM Backend Foundation | ✅ COMPLETE |
| 3-4 | 3D Character Tools | 🔄 IN PROGRESS |
| 5-6 | Agent Framework | 📋 PENDING |
| 7-8 | Integration (大壓軸) | 📋 PENDING |

**Details:** See [docs/reports/project-milestones.md](docs/reports/project-milestones.md)

---

## 🖥️ Hardware Configuration

**CRITICAL:** RTX 5080 16GB VRAM (single GPU)

```yaml
CPU: AMD Ryzen 9 9950X (16 cores)
RAM: 64GB DDR5
GPU: NVIDIA RTX 5080 16GB VRAM
PyTorch: 2.7.0 + CUDA 12.8 (IMMUTABLE)
Environment: conda ai_env
```

**Constraints:**
- Only ONE heavy model at a time (LLM OR SDXL)
- Dynamic model switching supported (20-35s)
- PyTorch SDPA only (xformers FORBIDDEN)

---

## 🗂️ Project Structure

```
animation-ai-studio/
├── docs/                       # 📚 All documentation
│   ├── architecture/           # Project architecture and design
│   ├── guides/                 # User guides and onboarding
│   ├── reports/                # Weekly completion reports
│   └── reference/              # Technical reference
├── llm_backend/                # ✅ Week 1-2: LLM services
│   ├── gateway/                # FastAPI Gateway
│   ├── services/               # vLLM configurations
│   ├── docker/                 # Docker orchestration
│   └── scripts/                # Management scripts
├── scripts/
│   ├── core/                   # Shared utilities
│   │   ├── llm_client/         # ✅ LLM client
│   │   └── generation/         # 🔄 Model manager (Week 3-4)
│   ├── generation/             # 🔄 Image generation (Week 3-4)
│   ├── synthesis/              # 🔄 Voice synthesis (Week 3-4)
│   ├── ai_editing/             # 📋 Agent framework (Week 5-8)
│   ├── analysis/               # Video, audio, image analysis
│   └── applications/           # End-user applications
├── configs/
│   ├── generation/             # 🔄 Generation configs (Week 3-4)
│   └── agent/                  # 📋 Agent configs (Week 5-6)
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
├── llm/           # LLM models (Week 1-2)
├── diffusion/     # SDXL, ControlNet (Week 3-4)
├── tts/           # GPT-SoVITS models (Week 3-4)
└── cv/            # Computer vision models

cache/
├── huggingface/
├── vllm/
└── diffusers/
```

---

## 🎬 Usage Examples

### Week 1-2: LLM Backend (READY ✅)

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

### Week 3-4: Character Generation (IN PROGRESS 🔄)

```python
# Image generation (coming soon)
from scripts.generation.image import CharacterGenerator

generator = CharacterGenerator()
result = await generator.generate_character(
    character="luca",
    scene="running on the beach, excited expression",
    quality="high"
)

# Voice synthesis (coming soon)
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
2. **[docs/architecture/project-architecture.md](docs/architecture/project-architecture.md)** - Overall architecture
3. **[docs/guides/llm-provider-tooling-onboarding.md](docs/guides/llm-provider-tooling-onboarding.md)** - Quick start guide
4. **[OPEN_SOURCE_MODELS.md](OPEN_SOURCE_MODELS.md)** - Models reference

### Implementation Guides

- **[docs/reports/week-1-2-completion.md](docs/reports/week-1-2-completion.md)** - Week 1-2 completion report
- **[docs/reports/week-3-4-plan.md](docs/reports/week-3-4-plan.md)** - Week 3-4 implementation plan
- **[docs/reports/project-milestones.md](docs/reports/project-milestones.md)** - Progress tracking

### Technical Reference

- **[docs/architecture/llm-backend.md](docs/architecture/llm-backend.md)** - LLM backend architecture
- **[llm_backend/README.md](llm_backend/README.md)** - LLM backend usage guide
- **[llm_backend/HARDWARE_SPECS.md](llm_backend/HARDWARE_SPECS.md)** - Hardware specifications

---

## 🔗 Related Projects

### 3D Animation LoRA Pipeline

**Location:** `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline`

**Purpose:** Train LoRA adapters for character/background/pose generation

**Current Status:**
- Luca SAM2 segmentation: 14.8% (約 43h remaining)
- Next: LaMa inpainting → Batch process 6 films

**Integration:**
- Trained LoRAs will be loaded via `configs/generation/lora_registry.yaml`
- Character metadata shared via `data/films/`

---

## ⚠️ Critical Requirements

### MUST Use (Open-Source Only)

- ✅ Qwen2.5-VL, Qwen2.5-14B (LLM)
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

**For Current Status:** [docs/reports/project-milestones.md](docs/reports/project-milestones.md)

**For Models:** [OPEN_SOURCE_MODELS.md](OPEN_SOURCE_MODELS.md)

---

## 📊 Progress

**Version:** v0.2.0
**Last Updated:** 2025-11-16
**Current Phase:** Week 3-4 (3D Character Tools)
**Completion:** 25% (Week 1-2 of 8)

**See [docs/reports/project-milestones.md](docs/reports/project-milestones.md) for detailed progress tracking.**

---

## 📄 License

Internal research project.
