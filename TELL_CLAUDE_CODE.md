# 🎯 如何告訴新的 Claude Code 會話

**當您在另一個 WSL 視窗開始新的 Claude Code 會話時，複製貼上以下內容：**

---

## 📋 開場訊息 (複製這段給 Claude Code)

```
我正在開發 Animation AI Studio 專案，這是一個使用開源LLM驅動的AI動畫創作平台。

核心要求：
1. ✅ 只能使用開源模型 (Qwen2.5-VL, DeepSeek-V3, GPT-SoVITS等)
2. ✅ LLM作為創意決策引擎 (不是工具，是大腦)
3. ✅ 自建LLM服務後端 (vLLM) - 不使用Ollama
4. ✅ 針對3D動畫角色優化 (Pixar風格)
5. ✅ LangGraph作為Agent框架

請先閱讀這些文檔 (按順序):
1. PROJECT_STATUS.md - 專案現狀和架構
2. LLM_BACKEND_ARCHITECTURE.md - 自建LLM後端設計
3. CLAUDE.md - 完整專案指南
4. OPEN_SOURCE_MODELS.md - 所有工具清單
5. HOW_TO_START.md - 快速開始指南

當前工作目錄: /mnt/c/AI_LLM_projects/animation-ai-studio

我想要做的是: [描述您的具體任務]
```

---

## 🎯 根據任務類型的具體說明

### 如果您要實現 LLM 後端：

```
任務：實現自建 LLM 服務後端

具體需求：
1. 使用 vLLM 部署 Qwen2.5-VL 72B 和 DeepSeek-V3 671B
2. 創建 FastAPI Gateway 進行請求路由
3. 添加 Redis 快取層
4. 實現負載均衡

參考文檔：
- LLM_BACKEND_ARCHITECTURE.md (完整架構)
- 第一步：實現 FastAPI Gateway (llm_backend/gateway/main.py)
```

### 如果您要實現 Agent 決策系統：

```
任務：實現 LangGraph Agent 決策引擎

具體需求：
1. 創建 LangGraph 狀態圖
2. 實現 ReAct 推理循環
3. 工具註冊和調用系統
4. 品質評估和自動迭代

參考文檔：
- CLAUDE.md (Agent workflow範例)
- PROJECT_STATUS.md (架構說明)
- 使用 Qwen2.5-VL 和 DeepSeek-V3 作為決策模型
```

### 如果您要實現具體工具：

```
任務：實現 [具體工具名稱]

例如：
- 3D角色圖像生成工具 (SDXL + LoRA)
- 角色聲音合成工具 (GPT-SoVITS)
- 影片惡搞生成工具 (SAM2 + 表情誇張)

參考文檔：
- OPEN_SOURCE_MODELS.md (查找需要的模型)
- CLAUDE.md (工具workflow範例)
```

---

## ⚠️ 重要提醒事項

Claude Code 必須知道的關鍵點：

### 1. 絕對不使用的東西
```
❌ Ollama (我們自建後端)
❌ GPT-4, Claude 3, Gemini (閉源)
❌ 任何付費API
```

### 2. 必須使用的技術
```
✅ vLLM (LLM推理)
✅ Qwen2.5-VL, DeepSeek-V3 (決策模型)
✅ LangGraph (Agent框架)
✅ FastAPI (Gateway)
✅ Redis (快取)
```

### 3. 3D 動畫特定參數 (不可隨意更改)
```yaml
alpha_threshold: 0.15    # 柔和邊緣
blur_threshold: 80       # 允許景深模糊
min_cluster_size: 10-15  # 3D角色聚類
dataset_size: 200-500    # 訓練數據量
color_jitter: false      # 保留PBR材質
horizontal_flip: false   # 保留非對稱細節
```

### 4. 硬體配置
```yaml
最低需求:
  - Qwen2.5-VL 72B: 2x RTX 4090 或 1x A6000 (48GB)
  - DeepSeek-V3 671B: 1x A100 80GB (FP8量化)
  - Qwen2.5-Coder 32B: 1x RTX 4090

推薦配置:
  - 2x A100 80GB 或 4x RTX 4090
```

---

## 📚 關鍵文檔快速索引

### 架構和設計
- `PROJECT_STATUS.md` - 專案現狀、架構、進度
- `LLM_BACKEND_ARCHITECTURE.md` - LLM後端完整設計
- `CLAUDE.md` - 完整專案指南和工作流程

### 技術參考
- `OPEN_SOURCE_MODELS.md` - 50+ 開源模型清單
- `HOW_TO_START.md` - 快速開始和優先順序

### 實現參考
- `LLM_BACKEND_ARCHITECTURE.md` 中的代碼範例
- `CLAUDE.md` 中的 workflow 範例

---

## 🚀 實現優先順序

告訴 Claude Code 按這個順序工作：

```
Week 1-2: LLM 後端基礎
- [ ] vLLM 服務部署 (Qwen2.5-VL, DeepSeek-V3)
- [ ] FastAPI Gateway
- [ ] Redis 快取
- [ ] 基礎測試

Week 3-4: 3D 角色工具
- [ ] SDXL + LoRA 生成
- [ ] GPT-SoVITS 語音合成
- [ ] ControlNet 整合

Week 5-6: Agent 決策引擎
- [ ] LangGraph Agent 實現
- [ ] ReAct 推理循環
- [ ] 品質評估系統

Week 7-8: 應用整合
- [ ] 惡搞影片生成器
- [ ] 多模態分析
- [ ] 端到端測試
```

---

## 💡 常見問題快速回答

**Q: 為什麼不用 Ollama？**
A: 我們需要完全控制和優化的LLM服務，Ollama功能有限。

**Q: 為什麼用 vLLM？**
A: PagedAttention + Continuous Batching，吞吐量高24x，記憶體效率高2-4x。

**Q: LLM 的角色是什麼？**
A: LLM是創意大腦，負責理解意圖、規劃策略、選擇工具、評估品質、自主迭代。

**Q: 3D動畫和2D動畫有什麼不同？**
A: 3D需要柔和邊緣(alpha 0.15)、較小數據集(200-500張)、不能用色彩抖動。

**Q: 資料在哪裡？**
A:
- 電影數據: `/mnt/data/ai_data/datasets/3d-anime/`
- AI倉庫: `/mnt/c/AI_LLM_projects/ai_warehouse/`
- 角色資訊: `data/films/`

---

## ✅ 檢查清單

在開始工作前，確認 Claude Code 理解：

- [ ] 只用開源模型
- [ ] 自建vLLM後端 (不用Ollama)
- [ ] LLM是決策引擎 (不只是工具)
- [ ] 針對3D動畫優化
- [ ] LangGraph作為Agent框架
- [ ] 當前狀態：文檔完成，準備實現
- [ ] 實現優先順序 (Week 1-8)
- [ ] 與LoRA Pipeline共享資源

---

## 🎬 開始工作的建議

告訴 Claude Code：

```
現在請開始 Week 1 的任務：

1. 創建 llm_backend/gateway/main.py (FastAPI Gateway)
2. 實現基礎路由和健康檢查
3. 添加 Redis 快取邏輯

參考 LLM_BACKEND_ARCHITECTURE.md 中的完整範例代碼。
```

---

**準備好了嗎？在新的 WSL 視窗打開 Claude Code，複製上面的開場訊息開始吧！** 🚀
