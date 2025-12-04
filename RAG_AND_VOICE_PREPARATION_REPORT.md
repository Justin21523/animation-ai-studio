# RAG 知識庫與語音訓練準備報告

**日期：** 2025-11-19
**任務：** 準備 RAG 知識庫數據和語音訓練數據
**狀態：** RAG 數據準備完成 ✅ | 語音訓練待視頻檔案 ⏳

---

## 📊 執行摘要

### ✅ 已完成項目

1. **電影數據結構檢查** - 完成
   - 確認 `data/films/luca/` 目錄結構完整
   - 發現 6 個詳細的角色描述文檔
   - 發現完整的電影元數據和風格指南

2. **RAG 導入腳本開發** - 完成
   - 創建 `scripts/rag/ingest_film_knowledge.py` (380+ 行)
   - 創建 `scripts/rag/test_rag_retrieval.py` (350+ 行)
   - 支持導入角色、元數據、風格指南、提示詞庫

### ⏳ 待處理項目

1. **RAG 系統實際導入和測試** - 需要解決依賴問題
2. **語音訓練數據提取** - 需要找到視頻檔案

---

## 🎯 RAG 知識庫數據狀態

### ✅ 數據已就緒

#### 1. 角色描述文檔 (6個)

位置：`data/films/luca/characters/`

```
✓ character_luca.md        (20,900 bytes) - 主角完整檔案
✓ character_alberto.md     (9,211 bytes)  - 第二主角
✓ character_giulia.md      (11,788 bytes) - 女主角
✓ character_massimo.md     (13,406 bytes) - 支援角色
✓ character_ercole.md      (13,659 bytes) - 反派
✓ character_ciccio_guido.md (11,200 bytes) - 配角
```

**每個角色檔案包含：**
- 完整人物背景
- 外貌詳細描述（人類形態和海怪形態）
- 性格特徵和演變
- 人際關係網絡
- 關鍵劇情時刻
- LoRA 訓練專用描述
- 提示詞模板
- 場景上下文

**示例：Luca 檔案內容**
```markdown
# 基本信息
- 年齡：13歲
- 物種：海怪（乾燥時變人類）
- 配音：Jacob Tremblay

# 外貌描述
人類形態：
- 苗條青少年體型
- 深棕色波浪髮
- 大型棕色眼睛
- 青綠條紋襯衫
- 藍色短褲

海怪形態：
- 藍綠色鱗片
- 藍色鰭狀"頭髮"
- 黃色鞏膜
- 長尾巴帶圓潤尾鰭

# AI 生成用提示詞
"a 3d animated character, luca paguro from pixar luca (2021),
teenage boy with wavy dark brown hair and large curious brown eyes,
wearing teal striped shirt, pixar style, smooth shading,
italian coastal setting"
```

#### 2. 電影元數據

位置：`data/films/luca/film_metadata.json`

```json
{
  "film": {
    "title": "Luca",
    "studio": "Disney / Pixar",
    "director": "Enrico Casarosa",
    "runtime": "95 minutes"
  },
  "setting": {
    "location": "Italian Riviera (Liguria)",
    "town": "Portorosso (fictional)",
    "time_period": "circa 1959"
  },
  "color_palette": {
    "primary": ["#87CEEB", "#FFD700", "#FF6B4A", "#2E8B57"],
    "mood": "Warm Mediterranean summer"
  },
  "themes": [
    "Friendship and identity",
    "Acceptance and openness",
    "Childhood freedom"
  ]
}
```

#### 3. 視覺風格指南

位置：`data/films/luca/style_guide.md`

包含：
- 光照設定（地中海、水下、室內）
- PBR 材質和著色器規格
- 相機和攝影技術
- 色彩分級和氛圍
- 字幕風格指南
- 負面提示詞建議

#### 4. 提示詞庫

位置：`data/films/luca/prompt_descriptions/`

包含各種場景和角色的生成提示詞模板。

---

## 🛠️ 創建的工具

### 1. RAG 知識庫導入工具

**檔案：** `scripts/rag/ingest_film_knowledge.py`

**功能：**
```python
class FilmKnowledgeIngester:
    - ingest_all()              # 導入所有知識
    - _ingest_characters()      # 導入角色描述
    - _ingest_metadata()        # 導入電影元數據
    - _ingest_style_guide()     # 導入風格指南
    - _ingest_prompts()         # 導入提示詞庫
```

**用法：**
```bash
# 導入 Luca 的所有知識
python scripts/rag/ingest_film_knowledge.py --film luca

# 重建知識庫（清除現有數據）
python scripts/rag/ingest_film_knowledge.py --film luca --rebuild
```

**導入流程：**
1. 讀取角色 Markdown 檔案
2. 解析電影元數據 JSON
3. 處理風格指南
4. 轉換提示詞庫
5. 創建 Document 對象（附帶元數據）
6. 生成 embeddings
7. 存入向量數據庫（FAISS/ChromaDB）

### 2. RAG 檢索測試工具

**檔案：** `scripts/rag/test_rag_retrieval.py`

**功能：**
- 自動化測試（9個預定義查詢）
- 交互式查詢模式
- 覆蓋率評估
- LLM Q&A 生成

**測試查詢示例：**
```
✓ "Tell me about Luca's personality and character traits"
✓ "What does Alberto look like in human form?"
✓ "Describe the visual style of Portorosso"
✓ "What color palette should I use for Luca-style images?"
✓ "What is the relationship between Luca and Alberto?"
✓ "Give me a good prompt for generating an image of Luca"
```

**用法：**
```bash
# 運行自動化測試
python scripts/rag/test_rag_retrieval.py

# 交互式查詢模式
python scripts/rag/test_rag_retrieval.py --interactive
```

---

## ❌ 遇到的問題

### 問題 1: Python 模塊依賴錯誤

**錯誤：**
```
ModuleNotFoundError: No module named 'loguru'
NameError: name 'Dict' is not defined
ImportError: cannot import name 'create_message'
```

**原因：**
- 缺少部分 Python 依賴包
- 類型註解導入不完整
- LLM Client 工具函數缺失

**已嘗試的修復：**
1. ✅ 安裝 loguru, aiohttp, omegaconf
2. ✅ 安裝 faiss-cpu, chromadb, sentence-transformers
3. ✅ 修復 utils.py 的類型導入
4. ❌ 仍有部分函數缺失需要補充

**建議解決方案：**
```bash
# 選項 1: 安裝完整依賴
pip install -r requirements/rag.txt
pip install -r requirements/core.txt

# 選項 2: 創建簡化版導入腳本（不依賴 LLM Backend）
# 使用 sentence-transformers 直接生成 embeddings
```

### 問題 2: 未找到視頻檔案

**搜索結果：**
- ❌ `/mnt/data/ai_data/datasets/3d-anime/luca/` - 無 .mp4/.mkv/.avi
- ✅ 有大量幀圖片（frames 目錄）
- ✅ 有 SAM2 分割數據

**影響：**
- 無法提取音軌進行語音訓練
- 無法運行 Whisper 轉錄
- 無法運行 Pyannote 語音分離

**可能的解決方案：**
1. 從其他位置獲取 Luca 完整視頻檔案
2. 從幀圖片重建視頻（需要音軌）
3. 使用其他電影的視頻進行語音訓練測試

---

## 📋 下一步行動計劃

### 優先級 1: 完成 RAG 系統導入和測試

#### 選項 A: 修復現有依賴問題
```bash
# 1. 創建虛擬環境並安裝所有依賴
cd /mnt/c/AI_LLM_projects/animation-ai-studio
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 補充缺失的工具函數
# 編輯 scripts/core/llm_client/utils.py

# 3. 運行導入
PYTHONPATH=. python scripts/rag/ingest_film_knowledge.py --film luca

# 4. 測試檢索
PYTHONPATH=. python scripts/rag/test_rag_retrieval.py
```

#### 選項 B: 創建獨立的簡化版本
```bash
# 創建不依賴 LLM Backend 的獨立導入工具
# 使用 sentence-transformers 生成 embeddings
# 直接操作 FAISS/ChromaDB

# 優點：
- 立即可用
- 無需啟動 LLM Backend
- 簡單直接

# 缺點：
- 與現有架構脫節
- 需要額外維護
```

**建議：選擇 A，修復依賴並使用完整架構**

### 優先級 2: 尋找或準備視頻檔案

#### 步驟：
1. 檢查其他可能的視頻檔案位置
2. 如果有 DVD/Blu-ray，進行擷取
3. 或者先用其他已有視頻的電影測試流程
4. 準備音軌提取腳本

### 優先級 3: 語音訓練流程（待視頻）

#### 完整流程：
```bash
# 1. 提取音軌
ffmpeg -i luca.mp4 -vn -acodec pcm_s16le luca_audio.wav

# 2. Whisper 轉錄
python scripts/synthesis/voice_dataset_builder.py \
    --video luca.mp4 \
    --output data/films/luca/voice_samples

# 3. Pyannote 語音分離
# 已集成在 voice_dataset_builder.py 中

# 4. 訓練語音模型
python scripts/synthesis/voice_model_trainer.py \
    --samples data/films/luca/voice_samples \
    --character Luca \
    --output models/voices/luca_voice.pth

# 5. 測試合成
python scripts/synthesis/tts/gpt_sovits_wrapper.py \
    --text "Silenzio, Bruno!" \
    --character luca \
    --output test_voice.wav
```

---

## 📊 數據統計

### RAG 知識庫數據

| 類型 | 數量 | 總大小 | 預估 Chunks |
|------|------|--------|-------------|
| 角色描述 | 6 檔案 | ~80KB | ~150-200 |
| 電影元數據 | 1 檔案 | ~6KB | ~10-15 |
| 風格指南 | 1 檔案 | ~15KB | ~20-30 |
| 提示詞庫 | 未知 | 未知 | ~50-100 |
| **總計** | **8-10 檔案** | **~100KB+** | **~230-345** |

### 語音訓練數據（待準備）

| 階段 | 需求 | 預估時間 |
|------|------|----------|
| 視頻獲取 | Luca 完整視頻 (1.5-3GB) | - |
| 音軌提取 | 95 分鐘音頻 (~150MB WAV) | 1-2 分鐘 |
| Whisper 轉錄 | 全片轉錄 + 時間戳 | 10-15 分鐘 |
| 語音分離 | Pyannote diarization | 20-30 分鐘 |
| 樣本篩選 | 每角色 1-5 分鐘乾淨語音 | 人工審查 |
| 模型訓練 | GPT-SoVITS 訓練 | 2-4 小時 |

---

## 💡 建議

### 立即可做的工作

1. **修復 RAG 依賴並完成導入**
   - 時間估計：1-2 小時
   - 價值：高 - 啟用知識檢索功能
   - 難度：中等

2. **創建簡單的查詢測試**
   - 測試角色查詢
   - 測試風格查詢
   - 測試關係查詢

3. **準備語音提取腳本**
   - 編寫音軌提取工具
   - 集成 Whisper + Pyannote
   - 待視頻檔案到位即可運行

### 待視頻檔案的工作

1. **語音訓練完整流程**
2. **視頻分析測試**
3. **端到端創意工作流測試**

---

## ✅ 完成標準

### RAG 系統就緒
- [ ] 成功導入所有 Luca 知識
- [ ] 檢索測試通過率 > 80%
- [ ] 可以回答角色/場景/風格查詢
- [ ] 可以生成提示詞建議

### 語音訓練就緒
- [ ] 獲得視頻檔案
- [ ] 成功提取音軌
- [ ] Whisper 轉錄完成
- [ ] 語音分離完成
- [ ] 至少訓練 1 個角色語音模型
- [ ] 合成測試成功

---

## 📝 總結

### ✅ 成就

1. **數據準備完成率：100%**
   - RAG 知識庫數據齊全且結構化
   - 6 個詳細角色檔案
   - 完整電影元數據
   - 風格指南和提示詞庫

2. **工具開發完成率：100%**
   - RAG 導入腳本 (380+ 行)
   - RAG 測試腳本 (350+ 行)
   - 支持多種文檔類型
   - 自動化測試框架

### ⏳ 待完成

1. **RAG 系統運行：需要修復依賴**
   - 估計工作量：1-2 小時
   - 阻礙：Python 模塊依賴問題

2. **語音訓練：需要視頻檔案**
   - 估計工作量：視頻到位後 4-6 小時
   - 阻礙：未找到完整視頻檔案

### 🎯 建議行動

**立即進行：**
1. 修復 RAG 依賴問題
2. 完成知識庫導入
3. 運行檢索測試

**並行準備：**
1. 尋找 Luca 視頻檔案
2. 或使用其他電影測試語音流程

**最終目標：**
完整的 RAG 知識系統 + 角色語音模型 = 可以開始測試端到端的自動化內容生成！

---

**報告日期：** 2025-11-19
**下次更新：** 完成 RAG 導入後
