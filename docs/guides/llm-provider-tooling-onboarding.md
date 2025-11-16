# LLMProvider Tooling Onboarding Guide

**Purpose:** Quick-start guide for new LLMProvider Tooling sessions
**Last Updated:** 2025-11-16
**Languages:** English (primary), Traditional Chinese (marked sections)

> **Consolidated Documentation**
> This document integrates HOW_TO_START.md + TELL_LLM_PROVIDER_TOOLING.md to provide bilingual onboarding instructions for LLMProvider Tooling.

---

## 📋 Quick Start for New LLMProvider Tooling Session

### Opening Message Template (English)

```
I'm working on Animation AI Studio, an LLM-driven AI platform for
3D animation content creation.

CRITICAL Requirements:
1. ✅ All models OPEN-SOURCE only (Qwen2.5-VL, Qwen2.5-14B, etc.)
2. ✅ Self-hosted vLLM backend (NO Ollama)
3. ✅ LLM as creative decision engine (brain, not just tool)
4. ✅ Optimized for 3D animation (Pixar-style)
5. ✅ LangGraph for agent framework

Hardware: RTX 5080 16GB VRAM (single GPU)
PyTorch: 2.7.0 + CUDA 12.8 (IMMUTABLE)

Please read in order:
1. docs/architecture/project-architecture.md - Overall architecture
2. LLM_PROVIDER.md - Complete project instructions
3. OPEN_SOURCE_MODELS.md - All models and tools
4. docs/reports/week-1-2-completion.md - Week 1-2 status

Current working directory: /mnt/c/AI_LLM_projects/animation-ai-studio

My task: [describe what you want to work on]
```

---

## 📋 開場訊息範本 (繁體中文)

```
我正在開發 Animation AI Studio 專案，這是一個使用開源LLM驅動的AI動畫創作平台。

核心要求：
1. ✅ 只能使用開源模型 (Qwen2.5-VL, Qwen2.5-14B, GPT-SoVITS等)
2. ✅ 自建LLM服務後端 (vLLM) - 絕對不使用Ollama
3. ✅ LLM作為創意決策引擎 (大腦，不只是工具)
4. ✅ 針對3D動畫角色優化 (Pixar風格)
5. ✅ LangGraph作為Agent框架

硬體: RTX 5080 16GB VRAM (單一GPU)
PyTorch: 2.7.0 + CUDA 12.8 (絕對不可修改)

請先閱讀這些文檔 (按順序):
1. docs/architecture/project-architecture.md - 專案整體架構
2. LLM_PROVIDER.md - 完整專案指南
3. OPEN_SOURCE_MODELS.md - 所有工具清單
4. docs/reports/week-1-2-completion.md - Week 1-2 完成狀態

當前工作目錄: /mnt/c/AI_LLM_projects/animation-ai-studio

我想要做的是: [描述您的具體任務]
```

---

## 🎯 Core Concepts / 核心概念

### Project Purpose

**English:**
- Create, analyze, and transform 3D animated content
- LLM agents make creative decisions autonomously
- Coordinate 50+ specialized AI tools
- Optimize for Pixar-style character consistency

**繁體中文：**
- 創建、分析、轉換 3D 動畫內容
- LLM Agent 自主做創意決策
- 協調 50+ 專業 AI 工具
- 優化 Pixar 風格角色一致性

### Architecture: LLM + RAG + Agent (缺一不可)

```
Week 7-8: AI Video Editing (大壓軸)
           ↓ 整合所有組件
Week 5-6: LangGraph Agent + RAG (核心決策)
           ↓ 調用工具
Week 3-4: 3D Character Tools (工具庫)
           ↓ 使用推理服務
Week 1-2: LLM Backend (基礎設施) ✅
```

**English:**
- **LLM**: Understand intent, plan execution, evaluate quality
- **RAG**: Retrieve character info, style guides, past work
- **Agent**: Autonomous tool selection, composition, iteration

**繁體中文：**
- **LLM**: 理解意圖、規劃執行、評估品質
- **RAG**: 檢索角色資訊、風格指南、過往作品
- **Agent**: 自主選擇工具、組合、迭代優化

---

## 📚 Documentation Reading Order / 文檔閱讀順序

### For Quick Context / 快速了解

1. **This File** - Quick onboarding
2. **[docs/architecture/project-architecture.md](../architecture/project-architecture.md)** - Overall architecture
3. **[LLM_PROVIDER.md](../../LLM_PROVIDER.md)** - Complete project guide

### For Implementation / 實作時

4. **[docs/reports/week-1-2-completion.md](../reports/week-1-2-completion.md)** - Week 1-2 status
5. **[docs/reports/week-3-4-plan.md](../reports/week-3-4-plan.md)** - Week 3-4 plan
6. **[OPEN_SOURCE_MODELS.md](../../OPEN_SOURCE_MODELS.md)** - Models reference

### For Technical Details / 技術細節

7. **[docs/architecture/llm-backend.md](../architecture/llm-backend.md)** - LLM backend design
8. **[llm_backend/README.md](../../llm_backend/README.md)** - LLM backend usage
9. **[llm_backend/HARDWARE_SPECS.md](../../llm_backend/HARDWARE_SPECS.md)** - Hardware specs

---

## 🖥️ Hardware Configuration / 硬體配置

### Actual Hardware (CRITICAL) / 實際硬體 (關鍵)

```yaml
CPU: AMD Ryzen 9 9950X (16 cores, 32 threads)
RAM: 64GB DDR5
GPU: NVIDIA RTX 5080 16GB VRAM (single card)
PyTorch: 2.7.0 (CANNOT MODIFY / 絕對不可修改)
CUDA: 12.8 (CANNOT MODIFY / 絕對不可修改)
Environment: conda ai_env
```

### VRAM Constraints / VRAM 限制

**English:**
- LLM (7B/14B): 12-14GB
- SDXL + LoRA: 13-15GB
- **Can only run ONE heavy model at a time**
- Dynamic switching takes 20-35 seconds

**繁體中文：**
- LLM (7B/14B): 12-14GB
- SDXL + LoRA: 13-15GB
- **一次只能運行一個重型模型**
- 動態切換需要 20-35 秒

---

## ⚠️ CRITICAL: What NOT to Use / 絕對禁止使用

### ❌ Forbidden

**English:**
- Ollama (we use self-hosted vLLM)
- GPT-4, LLMProvider 3, Gemini (closed-source)
- Any paid APIs
- xformers (breaks PyTorch 2.7.0 compatibility)
- Modifying PyTorch version
- Modifying CUDA version

**繁體中文：**
- Ollama (我們使用自建 vLLM)
- GPT-4, LLMProvider 3, Gemini (閉源)
- 任何付費 API
- xformers (破壞 PyTorch 2.7.0 相容性)
- 修改 PyTorch 版本
- 修改 CUDA 版本

### ✅ Required

**English:**
- vLLM (LLM inference)
- Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B
- LangGraph (agent framework)
- FastAPI (gateway)
- Redis (caching)
- PyTorch SDPA (attention backend)

**繁體中文：**
- vLLM (LLM 推理)
- Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B
- LangGraph (Agent 框架)
- FastAPI (閘道)
- Redis (快取)
- PyTorch SDPA (注意力後端)

---

## 🎯 Task-Specific Instructions / 針對任務的具體說明

### Task 1: Week 1-2 LLM Backend ✅ COMPLETE

**English:**
```
Status: Production ready (2025-11-16)

Completed:
- vLLM services deployed (Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B)
- FastAPI Gateway operational
- Redis caching functional
- Docker orchestration working
- Management scripts available

Usage:
bash llm_backend/scripts/start_all.sh  # Interactive model selection
bash llm_backend/scripts/health_check.sh  # Check status

Details: See docs/reports/week-1-2-completion.md
```

**繁體中文：**
```
狀態：已投產 (2025-11-16)

已完成：
- vLLM 服務部署 (Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B)
- FastAPI Gateway 運作中
- Redis 快取功能正常
- Docker 編排運作中
- 管理腳本可用

使用：
bash llm_backend/scripts/start_all.sh  # 互動式模型選擇
bash llm_backend/scripts/health_check.sh  # 檢查狀態

詳情：見 docs/reports/week-1-2-completion.md
```

---

### Task 2: Week 3-4 3D Character Tools ⬅️ CURRENT

**English:**
```
Goal: Integrate image/voice generation for 3D characters

Tasks:
1. SDXL + LoRA integration
   - Load SDXL base model
   - Integrate LoRA adapters (character, background, style)
   - Implement dynamic model switching (LLM ↔ SDXL)

2. ControlNet guided generation
   - OpenPose for pose control
   - Depth for composition
   - Canny for edge structure

3. Character consistency
   - InstantID / ArcFace embeddings
   - Similarity threshold: 0.60-0.65

4. GPT-SoVITS voice synthesis
   - Voice cloning from film audio
   - Emotion control
   - Multi-language support (EN, IT)

Reference: docs/reports/week-3-4-plan.md
```

**繁體中文：**
```
目標：整合 3D 角色的圖像/語音生成

任務：
1. SDXL + LoRA 整合
   - 載入 SDXL 基礎模型
   - 整合 LoRA 適配器 (角色、背景、風格)
   - 實作動態模型切換 (LLM ↔ SDXL)

2. ControlNet 引導生成
   - OpenPose 姿態控制
   - Depth 構圖引導
   - Canny 邊緣結構

3. 角色一致性
   - InstantID / ArcFace embeddings
   - 相似度門檻: 0.60-0.65

4. GPT-SoVITS 語音合成
   - 從影片音訊克隆語音
   - 情緒控制
   - 多語言支援 (EN, IT)

參考：docs/reports/week-3-4-plan.md
```

---

### Task 3: Week 5-6 Agent Framework (FUTURE)

**English:**
```
Goal: Build autonomous creative decision system

Core: LLM + RAG + Agent
- LangGraph state machine
- ReAct reasoning loop
- RAG for retrieving character info, past work
- Tool registry for calling Week 3-4 tools
- Quality evaluation and iteration

Reference: docs/architecture/project-architecture.md
```

**繁體中文：**
```
目標：建立自主創意決策系統

核心：LLM + RAG + Agent
- LangGraph 狀態機
- ReAct 推理循環
- RAG 檢索角色資訊、過往作品
- 工具註冊系統調用 Week 3-4 工具
- 品質評估與迭代

參考：docs/architecture/project-architecture.md
```

---

## 📂 File Paths / 檔案路徑

### Shared Resources / 共用資源

**All data paths use `/mnt/data/ai_data/` base:**

```yaml
Film Data / 影片資料:
  /mnt/data/ai_data/datasets/3d-anime/luca/
  /mnt/data/ai_data/datasets/3d-anime/coco/

AI Warehouse / AI 倉庫:
  Models: /mnt/c/AI_LLM_projects/ai_warehouse/models/
    ├── llm/         # LLM models
    ├── diffusion/   # SDXL, ControlNet
    ├── tts/         # GPT-SoVITS
    └── cv/          # Computer vision

  Cache: /mnt/c/AI_LLM_projects/ai_warehouse/cache/
    ├── huggingface/
    ├── vllm/
    └── diffusers/

Character Metadata / 角色元資料:
  data/films/luca/characters/
  data/films/coco/characters/
```

---

## 🎨 3D Animation Specific Settings / 3D 動畫特定設定

### CRITICAL Parameters (DO NOT CHANGE) / 關鍵參數 (不可更改)

```yaml
Segmentation:
  alpha_threshold: 0.15    # Soft anti-aliased edges / 柔和邊緣
  blur_threshold: 80       # Tolerate DoF blur / 允許景深模糊

Clustering:
  min_cluster_size: 10-15  # Smaller than 2D / 比 2D 動畫小
  min_samples: 2           # Tighter identity / 更緊密的身份

Training:
  dataset_size: 200-500    # Fewer than 2D / 比 2D 動畫少
  color_jitter: false      # Breaks PBR / 破壞 PBR 材質
  horizontal_flip: false   # Breaks asymmetry / 破壞非對稱性

LoRA:
  learning_rate: 1e-4 to 2e-4
  network_rank: 32-64
  epochs: 10-20
```

### Prompt Engineering

```yaml
Positive Keywords:
  - "pixar style"
  - "3d animation"
  - "smooth shading"
  - "pbr materials"
  - "cinematic lighting"
  - "rendered"

Negative Keywords:
  - "2d"
  - "flat"
  - "anime"
  - "sketchy"
  - "low quality"
```

---

## 🚀 Quick Commands / 快速指令

### Environment Setup / 環境設定

```bash
# Navigate to project / 導航至專案
cd /mnt/c/AI_LLM_projects/animation-ai-studio

# Activate conda environment / 啟動 conda 環境
conda activate ai_env

# Check GPU / 檢查 GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### LLM Backend Management / LLM 後端管理

```bash
# Start services (interactive) / 啟動服務 (互動式)
bash llm_backend/scripts/start_all.sh

# Check health / 檢查健康狀態
bash llm_backend/scripts/health_check.sh

# Switch model / 切換模型
bash llm_backend/scripts/switch_model.sh

# View logs / 查看日誌
bash llm_backend/scripts/logs.sh gateway
bash llm_backend/scripts/logs.sh qwen-vl

# Stop services / 停止服務
bash llm_backend/scripts/stop_all.sh
```

---

## 💡 Common Questions / 常見問題

### Q: Why not use Ollama? / 為什麼不用 Ollama？

**English:** We need full control and optimization of LLM services. Ollama has limited functionality for our needs.

**繁體中文：** 我們需要完全控制和優化 LLM 服務。Ollama 的功能對我們的需求有限。

---

### Q: Why vLLM? / 為什麼用 vLLM？

**English:** PagedAttention + Continuous Batching = 24x higher throughput, 2-4x memory efficiency.

**繁體中文：** PagedAttention + Continuous Batching = 吞吐量高 24 倍，記憶體效率高 2-4 倍。

---

### Q: What is LLM's role? / LLM 的角色是什麼？

**English:** LLM is the creative brain - understands intent, plans strategy, selects tools, evaluates quality, iterates autonomously.

**繁體中文：** LLM 是創意大腦 - 理解意圖、規劃策略、選擇工具、評估品質、自主迭代。

---

### Q: 3D vs 2D animation differences? / 3D 和 2D 動畫有什麼不同？

**English:** 3D needs soft edges (alpha 0.15), smaller datasets (200-500), no color jitter.

**繁體中文：** 3D 需要柔和邊緣 (alpha 0.15)、較小數據集 (200-500 張)、不能用色彩抖動。

---

### Q: Where is data shared? / 資料在哪裡共用？

**English:**
- Film data: `/mnt/data/ai_data/datasets/3d-anime/`
- AI Warehouse: `/mnt/c/AI_LLM_projects/ai_warehouse/`
- Character info: `data/films/`

**繁體中文：**
- 影片資料: `/mnt/data/ai_data/datasets/3d-anime/`
- AI 倉庫: `/mnt/c/AI_LLM_projects/ai_warehouse/`
- 角色資訊: `data/films/`

---

## ✅ Onboarding Checklist / 入職檢查清單

Before starting work, ensure LLMProvider Tooling understands:

在開始工作前，確認 LLMProvider Tooling 理解：

- [ ] ONLY open-source models / 只用開源模型
- [ ] Self-hosted vLLM backend (NOT Ollama) / 自建 vLLM 後端 (不用 Ollama)
- [ ] LLM as decision engine (not just tool) / LLM 是決策引擎 (不只是工具)
- [ ] Optimized for 3D animation / 針對 3D 動畫優化
- [ ] LangGraph for agents / LangGraph 作為 Agent 框架
- [ ] Week 1-2 COMPLETE, Week 3-4 CURRENT / Week 1-2 完成，Week 3-4 進行中
- [ ] Shared resources with LoRA pipeline / 與 LoRA Pipeline 共享資源
- [ ] Hardware: RTX 5080 16GB (single GPU) / 硬體: RTX 5080 16GB (單一 GPU)
- [ ] PyTorch 2.7.0 + CUDA 12.8 IMMUTABLE / PyTorch 2.7.0 + CUDA 12.8 不可變
- [ ] Data paths: `/mnt/data/ai_data/...` / 資料路徑: `/mnt/data/ai_data/...`

---

## 📖 Additional Resources / 其他資源

### External Documentation / 外部文檔

**English:**
- LangGraph: https://langchain-ai.github.io/langgraph/
- Qwen2.5: https://github.com/QwenLM/Qwen2.5
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS

**繁體中文：**
- LangGraph: https://langchain-ai.github.io/langgraph/
- Qwen2.5: https://github.com/QwenLM/Qwen2.5
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS

---

**Ready to start? / 準備好開始了嗎？**

**English:** Ask LLMProvider Tooling to begin with the current task. Refer to [docs/architecture/project-architecture.md](../architecture/project-architecture.md) for overall context.

**繁體中文：** 讓 LLMProvider Tooling 開始當前任務。參考 [docs/architecture/project-architecture.md](../architecture/project-architecture.md) 了解整體內容。

---

**Last Updated:** 2025-11-16
**Maintained By:** Animation AI Studio Team
