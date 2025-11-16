# Animation AI Studio - Project Architecture

**Last Updated:** 2025-11-16
**Current Phase:** Week 3-4 - 3D Character Generation Tools
**Version:** v0.2.0

> **Consolidated Documentation**
> This document integrates PROJECT_STATUS.md + IMPLEMENTATION_ROADMAP.md to provide a unified view of project architecture, status, and implementation plan.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Design](#architecture-design)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Current Status](#current-status)
5. [Technical Stack](#technical-stack)
6. [Hardware Configuration](#hardware-configuration)
7. [Key Workflows](#key-workflows)
8. [Related Projects](#related-projects)

---

## 🎯 Project Overview

**Animation AI Studio** is an advanced multimodal AI platform designed for creating, analyzing, and transforming animated content using **open-source LLM agents** as the core decision-making engine.

### Core Philosophy

**LLM + RAG + Agent: 缺一不可**
- **LLM**: 理解創意意圖、規劃執行步驟、評估品質
- **RAG**: 檢索動畫資料、角色資訊、風格指南、過往作品
- **Agent**: 自主決策使用哪些工具、如何組合、迭代優化

**3D Animation Focus:**
- Optimized for Pixar/Disney-style 3D animation
- Character-centric workflows
- Maintains consistency across generations

### Project Distinction

**Animation AI Studio vs. 3D Animation LoRA Pipeline:**
- **LoRA Pipeline**: Trains LoRA adapters for character/background/pose generation
- **AI Studio**: Analyzes, processes, and transforms existing animation content using SOTA AI models
- **Shared Resources**: Film datasets, character metadata, AI Warehouse

---

## 🧠 Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Week 7-8: AI Video Editing (大壓軸)              │
│         🎬 AI 自主創作影片，整合所有前期組件                   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│            Week 5-6: LangGraph Agent + RAG (核心)            │
│    🤖 LLM 理解意圖 + RAG 檢索資料 + Agent 自主決策           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│              Week 3-4: 3D Character Tools (工具庫)           │
│    🎨 SDXL + LoRA + ControlNet + GPT-SoVITS                 │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│              Week 1-2: LLM Backend (基礎設施) ✅              │
│    🖥️ vLLM + FastAPI + Redis + Docker                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**1. LLM Decision Engine (Creative Brain)**
- **Qwen2.5-VL-7B**: Multimodal understanding (vision + chat)
- **Qwen2.5-14B**: Reasoning and complex decision making
- **Qwen2.5-Coder-7B**: Code generation and tool orchestration

**2. Agent Framework (Week 5-6)**
- **LangGraph** (Primary): ReAct reasoning, tool calling, multi-agent
- **AutoGen** (Secondary): Multi-agent collaboration

**3. Tool Categories**
- **Image Generation**: SDXL, ControlNet, LoRA
- **Voice Synthesis**: GPT-SoVITS, Coqui TTS
- **Video Editing**: SAM2, MoviePy, FFmpeg
- **Multimodal Analysis**: MediaPipe, InsightFace
- **Parody & Effects**: Expression exaggeration, speed ramping

### Directory Structure

```
animation-ai-studio/
├── scripts/
│   ├── core/              # Shared utilities
│   │   ├── utils/         # Config, logging, paths
│   │   ├── models/        # Model loading
│   │   ├── llm_client/    # LLM client (Week 1-2 ✅)
│   │   └── generation/    # Model manager (Week 3-4)
│   ├── analysis/          # Video, audio, image, style analysis
│   ├── processing/        # Extraction, enhancement, synthesis
│   ├── generation/        # AI content generation (Week 3-4)
│   │   ├── image/         # SDXL + LoRA
│   │   ├── video/         # AnimateDiff
│   │   └── audio/         # Music, SFX
│   ├── synthesis/         # Voice and speech (Week 3-4)
│   │   ├── tts/           # GPT-SoVITS
│   │   ├── voice_cloning/
│   │   └── lip_sync/      # Wav2Lip
│   ├── ai_editing/        # LLM-powered editing (Week 5-8)
│   │   ├── decision_engine/  # LLM + RAG + Agent
│   │   ├── video_editor/     # Automated editing
│   │   └── style_remix/      # Parody generation
│   └── applications/      # End-user apps
├── configs/
│   ├── global.yaml
│   └── generation/        # Week 3-4 configs
├── data/films/            # Shared with LoRA pipeline
├── llm_backend/           # Week 1-2 LLM infrastructure ✅
├── docs/                  # Consolidated documentation
│   ├── architecture/      # This file
│   ├── guides/
│   ├── reports/
│   ├── reference/
│   └── theory/
├── outputs/               # Generated content
└── requirements/          # Modular dependencies
```

---

## 🚀 Implementation Roadmap

### Week 1-2: LLM Backend Foundation ✅ **COMPLETED**

**Goal:** Build self-hosted LLM inference backend

**Status:** Production ready (2025-11-16)

**Deliverables (34 files, ~5,900 lines of code):**
- ✅ vLLM service configurations (Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B)
- ✅ FastAPI Gateway with OpenAI-compatible API
- ✅ Redis caching layer
- ✅ Docker orchestration (single RTX 5080 16GB GPU)
- ✅ PyTorch 2.7.0 native SDPA (xformers FORBIDDEN)
- ✅ Application-layer LLM client
- ✅ Management scripts (start, stop, switch, health, logs)
- ✅ Monitoring (Prometheus, Grafana)

**Performance Metrics:**
- Qwen2.5-VL-7B: ~40 tok/s, 13.8GB VRAM
- Qwen2.5-14B: ~45 tok/s, 11.5GB VRAM
- Model switching: 20-35 seconds

**Details:** See [`docs/reports/week-1-2-completion.md`](../reports/week-1-2-completion.md)

---

### Week 3-4: 3D Character Generation Tools ⬅️ **CURRENT**

**Goal:** Integrate image/voice generation for 3D characters

**Deliverables:**
- [ ] SDXL + LoRA integration
- [ ] GPT-SoVITS voice cloning setup
- [ ] ControlNet guided generation (OpenPose, Depth, Canny)
- [ ] Character consistency pipeline (InstantID, ArcFace)
- [ ] VRAM dynamic management (LLM ↔ SDXL switching)
- [ ] Agent-Ready tool design (為 Week 5-6 準備)

**Key Files:**
```
scripts/generation/image/
├── sdxl_pipeline.py
├── lora_loader.py
├── controlnet_generator.py
├── character_generator.py
├── consistency_checker.py
└── batch_generator.py

scripts/synthesis/tts/
├── gpt_sovits_wrapper.py
├── voice_model_trainer.py
├── emotion_controller.py
└── voice_dataset_builder.py

scripts/core/generation/
├── model_manager.py          # VRAM 動態管理
└── generation_cache.py

configs/generation/
├── sdxl_config.yaml
├── lora_registry.yaml
├── controlnet_config.yaml
├── character_presets.yaml
└── tts_config.yaml
```

**Details:** See [`docs/reports/week-3-4-plan.md`](../reports/week-3-4-plan.md)

---

### Week 5-6: LangGraph Agent Decision Engine

**Goal:** Build autonomous creative decision system

**核心：LLM + RAG + Agent**
- LangGraph state machine (Agent framework)
- ReAct reasoning loop (決策循環)
- RAG integration (檢索動畫資料、角色資訊、過往作品)
- Tool registration and calling (呼叫 Week 3-4 工具)
- Quality evaluation system (品質評估與迭代)

**Deliverables:**
- [ ] LangGraph agent framework
- [ ] RAG system (vector store, embeddings, retrieval)
- [ ] Tool registry (standardized tool interface)
- [ ] Quality evaluator (LLM-based)
- [ ] Iteration logic (自主優化)

**Key Files:**
```
scripts/ai_editing/decision_engine/
├── agent_graph.py
├── react_agent.py
├── rag_system.py
├── tool_registry.py
└── quality_evaluator.py
```

---

### Week 7-8: End-to-End Integration (大壓軸)

**Goal:** AI 自主創作影片

**完整流程：**
```
User Request
     ↓
LLM 理解創意意圖
     ↓
RAG 檢索相關資料 (角色、場景、過往作品)
     ↓
Agent 規劃執行步驟
     ↓
調用工具 (圖像生成、語音合成、影片剪輯)
     ↓
LLM 評估品質
     ↓
Agent 決定是否迭代
     ↓
輸出最終作品
```

**Deliverables:**
- [ ] Parody video generator (搞笑影片自動生成)
- [ ] Multimodal analysis pipeline (多模態分析)
- [ ] End-to-end creative workflows
- [ ] User interface

**Key Files:**
```
scripts/ai_editing/style_remix/
├── parody_generator.py
└── effect_composer.py

scripts/applications/
└── creative_studio_app.py
```

---

## 📊 Current Status

### ✅ Completed

**Week 1-2: LLM Backend** (2025-11-16)
- Self-hosted vLLM inference backend
- FastAPI Gateway with Redis caching
- Docker orchestration
- Management scripts and monitoring
- Complete documentation (34 files, 5,900 LOC)

### 🔄 In Progress

**Week 3-4: 3D Character Tools** (Current)
- Documentation consolidation
- Environment setup
- SDXL pipeline implementation

**LoRA Pipeline Project** (Background)
- Luca SAM2 segmentation: 14.8% (2,129/14,411 frames)
- Smart batch launcher monitoring
- LaMa inpainting pending

### 📋 Pending

- Week 5-6: Agent Framework
- Week 7-8: Integration

---

## 🔧 Technical Stack

### Hardware (Actual Configuration)

**CRITICAL:** RTX 5080 16GB VRAM 限制

```yaml
CPU: AMD Ryzen 9 9950X (16 cores, 32 threads)
RAM: 64GB DDR5
GPU: NVIDIA RTX 5080 16GB VRAM (single card)
PyTorch: 2.7.0 (IMMUTABLE)
CUDA: 12.8 (IMMUTABLE)
Environment: conda ai_env
```

**VRAM Constraints:**
- LLM (7B/14B): 12-14GB
- SDXL + LoRA: 13-15GB
- **只能同時運行一個重型模型**
- 動態切換需要 20-35 秒

### LLM Models (Optimized for 16GB)

**Deployed Models:**

| Model | Purpose | VRAM | Port | Speed |
|-------|---------|------|------|-------|
| Qwen2.5-VL-7B | Multimodal (vision + chat) | ~14GB | 8000 | ~40 tok/s |
| Qwen2.5-14B | Reasoning | ~12GB | 8001 | ~45 tok/s |
| Qwen2.5-Coder-7B | Code generation | ~14GB | 8002 | ~42 tok/s |

**Note:** Only ONE model can run at a time. Dynamic switching supported via management scripts.

### Image Generation Stack (Week 3-4)

```yaml
SDXL Base:
  Model: stabilityai/stable-diffusion-xl-base-1.0
  VRAM: ~10-11GB
  Resolution: 1024x1024
  Attention: PyTorch SDPA (NOT xformers)

LoRA:
  Characters: Luca, Alberto, Giulia
  Backgrounds: Portorosso town
  Style: Pixar 3D animation
  Weight: 0.6-0.85

ControlNet:
  OpenPose: Pose control
  Depth: Composition guidance
  Canny: Edge structure
  VRAM: +1-2GB per ControlNet

Character Consistency:
  Method: InstantID, ArcFace embeddings
  Threshold: 0.60-0.65
```

### Voice Synthesis Stack (Week 3-4)

```yaml
GPT-SoVITS:
  Repo: RVC-Boss/GPT-SoVITS
  VRAM: ~3-4GB
  Languages: EN, IT
  Training: 1-5 min voice samples

Coqui TTS:
  Method: XTTS-v2
  Languages: 17 languages
  Zero-shot: Yes
```

### Agent Framework (Week 5-6)

```yaml
LangGraph:
  Purpose: Primary agent framework
  Features: ReAct, tool calling, multi-agent, state management

RAG System:
  Vector Store: Chroma / FAISS
  Embeddings: HuggingFace embeddings
  Collections: character_info, past_generations, style_guide
```

---

## 🖥️ Hardware Configuration

### Actual Setup (RTX 5080 16GB)

**Capabilities:**
- ✅ Single 7B/14B model (full precision)
- ✅ SDXL + LoRA (with LLM stopped)
- ✅ GPT-SoVITS (with SDXL unloaded)
- ❌ Multiple models simultaneously (OOM)
- ❌ 72B+ models (requires INT4 quantization)

**VRAM Management Strategy:**
```python
# Workflow 1: Image Generation
1. Stop LLM service (free 12-14GB)
2. Load SDXL pipeline (use 13-15GB)
3. Generate images
4. Unload SDXL
5. Restart LLM service

# Workflow 2: Voice Synthesis
1. Can run with LLM stopped
2. GPT-SoVITS uses ~4GB
3. Or run with SDXL unloaded

# Workflow 3: LLM Analysis
1. Stop all generation tools
2. Run LLM for analysis/planning
```

### Path Management (Unified)

**All projects share AI Warehouse:**

```yaml
Models: /mnt/c/AI_LLM_projects/ai_warehouse/models/
  ├── llm/           # LLM models (Week 1-2)
  ├── diffusion/     # SDXL, ControlNet (Week 3-4)
  ├── tts/           # GPT-SoVITS models (Week 3-4)
  └── cv/            # Computer vision models

Cache: /mnt/c/AI_LLM_projects/ai_warehouse/cache/
  ├── huggingface/
  ├── vllm/
  └── diffusers/

Data: /mnt/data/ai_data/
  └── datasets/3d-anime/  # Shared with LoRA pipeline
```

**Benefits:**
- 防止重複下載模型
- 節省儲存空間
- 跨專案資源共享

---

## 🎬 Key Workflows

### Workflow 1: LLM-Driven Character Image Generation

```
User: "Generate Luca running on the beach, excited expression"

┌─────────────────────────────────────────────────┐
│ 1. LLM Analysis (Qwen-14B)                      │
│    - Character: Luca                            │
│    - Action: Running                            │
│    - Emotion: Excited                           │
│    - Location: Beach                            │
│    - Decision: Use ControlNet (OpenPose)        │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 2. Prompt Engineering (Qwen-Coder-7B)           │
│    - Positive: "luca, boy, running pose..."     │
│    - Negative: "2d, anime, flat..."             │
│    - Find running pose reference                │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 3. Model Switching                              │
│    - Stop LLM service (free VRAM)               │
│    - Load SDXL + Luca LoRA + ControlNet         │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 4. Image Generation                             │
│    - Steps: 35, CFG: 7.5                        │
│    - ControlNet pose conditioning                │
│    - Time: ~15-20 seconds                       │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 5. Quality Evaluation (Qwen-VL-7B)              │
│    - Restart LLM service                        │
│    - Check character likeness: 9/10             │
│    - Check pose accuracy: 9.5/10                │
│    - Decision: Approve                          │
└─────────────────────────────────────────────────┘
                   ↓
             Output: luca_beach_running.png
```

### Workflow 2: Character Voice Synthesis

```
User: "Generate Luca saying 'Silenzio, Bruno!' with determination"

┌─────────────────────────────────────────────────┐
│ 1. Voice Model Selection                        │
│    - Character: Luca                            │
│    - Emotion: Determined                        │
│    - Tool: GPT-SoVITS                           │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 2. Model Loading                                │
│    - Stop SDXL (if running)                     │
│    - Load GPT-SoVITS (4GB)                      │
│    - Load Luca voice model                      │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 3. Synthesis                                    │
│    - Text: "Silenzio, Bruno!"                   │
│    - Emotion control: 0.8 (strong)              │
│    - Language: EN with Italian accent           │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 4. Quality Check                                │
│    - Voice similarity: 92%                      │
│    - Emotion accuracy: 88%                      │
│    - Approve                                    │
└─────────────────────────────────────────────────┘
                   ↓
             Output: luca_silenzio_bruno.wav
```

### Workflow 3: AI 自主創作影片 (Week 7-8 大壓軸)

```
User: "創作 Luca 第一次看到海的搞笑短片"

┌─────────────────────────────────────────────────┐
│ 1. 意圖理解 (LLM)                                │
│    - 類型: 搞笑短片                              │
│    - 角色: Luca                                  │
│    - 場景: 第一次看到海                          │
│    - 風格: 誇張、慢動作、戲劇化                  │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 2. RAG 檢索                                      │
│    - 角色資料: Luca 個性、表情參考               │
│    - 場景資料: 海邊場景、Portorosso 風格         │
│    - 過往作品: 類似搞笑短片案例                  │
│    - 影片素材: Luca 電影中的海邊片段             │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 3. Agent 規劃                                    │
│    Shot 1: Luca 背影                            │
│      Tool: generate_character_image              │
│    Shot 2: 誇張表情                              │
│      Tool: expression_exaggeration (2.5x)        │
│    Shot 3: 慢動作 + 戲劇音樂                     │
│      Tool: apply_slow_motion + add_music         │
│    Shot 4: 語音 "Wow!"                           │
│      Tool: synthesize_voice + lip_sync           │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 4. 自動執行所有工具                              │
│    - 動態切換模型 (LLM ↔ SDXL ↔ GPT-SoVITS)      │
│    - 生成所有素材                                │
│    - 自動剪輯組合                                │
└─────────────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 5. LLM 評估品質                                  │
│    - 搞笑程度: 8.5/10                            │
│    - Agent 決定: 增加變焦特寫                    │
│    - 重新生成 → 9.2/10 ✓                        │
└─────────────────────────────────────────────────┘
                   ↓
             Output: luca_first_ocean_parody.mp4
```

---

## 🔗 Related Projects

### 3D Animation LoRA Pipeline

**Location:** `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline`

**Purpose:** Train LoRA adapters for character/background/pose generation

**Shared Resources:**
- Film datasets: `/mnt/data/ai_data/datasets/3d-anime/`
- Character metadata: `data/films/`
- AI Warehouse: `/mnt/c/AI_LLM_projects/ai_warehouse/`

**Current Status:**
- Luca SAM2 segmentation: 14.8% (約 43h 剩餘)
- Smart batch launcher: Monitoring GPU
- Next: LaMa inpainting → Batch process 6 other films

**Integration:**
- LoRA 訓練完成後會整合到 Animation AI Studio
- 使用 `configs/generation/lora_registry.yaml` 註冊

---

## 💡 Core Requirements

### CRITICAL: All Open-Source

**✅ MUST USE:**
- Qwen2.5-VL, Qwen2.5-14B for LLM
- GPT-SoVITS for voice
- SDXL + LoRA for images
- LangGraph for agents

**❌ DO NOT USE:**
- GPT-4, LLMProvider, Gemini (closed-source)
- Any paid APIs for core functionality

### CRITICAL: PyTorch Compatibility

**IMMUTABLE:**
- PyTorch 2.7.0 + CUDA 12.8
- **絕對禁止修改**

**Attention Backend:**
- vLLM: TORCH_SDPA (NO xformers)
- SDXL: TORCH_SDPA (保持一致性)
- Environment: `XFORMERS_DISABLED=1`

### CRITICAL: Extensible Design

**All code must be designed with:**
- 充足的參數（可擴展）
- 標準化介面（Agent-Ready）
- 配置檔案（易調整）
- 元資料記錄（供 RAG）

---

## 📚 Documentation

### Primary Documentation

- **This File:** Project architecture and implementation plan
- **[LLM_PROVIDER.md](../../LLM_PROVIDER.md):** Complete project instructions for LLMProvider Tooling
- **[OPEN_SOURCE_MODELS.md](../../OPEN_SOURCE_MODELS.md):** Complete model reference

### Reports

- **[Week 1-2 Completion](../reports/week-1-2-completion.md):** LLM Backend completion report
- **[Week 3-4 Plan](../reports/week-3-4-plan.md):** 3D Character Tools detailed plan
- **[Project Milestones](../reports/project-milestones.md):** Overall progress tracking

### Guides

- **[LLMProvider Tooling Onboarding](../guides/llm-provider-tooling-onboarding.md):** Quick start for new sessions
- **[Week-by-Week Guide](../guides/week-by-week-guide.md):** Consolidated 8-week view
- **[Image Generation Guide](../guides/image-generation-guide.md):** SDXL + LoRA usage (Week 3-4)
- **[Voice Synthesis Guide](../guides/voice-synthesis-guide.md):** GPT-SoVITS usage (Week 3-4)

### Technical References

- **[LLM Backend](llm-backend.md):** LLM backend architecture design
- **[Hardware Requirements](hardware-requirements.md):** Hardware specs and VRAM management

---

## 🔄 Version History

- **v0.2.0** (2025-11-16): Week 1-2 Complete, Week 3-4 In Progress
  - LLM Backend foundation complete (34 files, 5,900 LOC)
  - Documentation consolidation initiated
  - Week 3-4 planning complete

- **v0.1.0** (2025-11-16): Initial Setup
  - Project structure created
  - Research completed
  - Documentation written

---

**Next Steps:** See [Week-by-Week Guide](../guides/week-by-week-guide.md) for detailed implementation timeline.

**For Questions:** Refer to [LLM_PROVIDER.md](../../LLM_PROVIDER.md) or [reports/](../reports/) for specific topics.
