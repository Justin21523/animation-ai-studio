# Voice Training Guide - GPT-SoVITS & RVC

**創建日期**: 2025-11-20
**狀態**: 實現完成
**目的**: 完整的角色語音訓練指南

---

## 系統架構

### 訓練流程概述

```
語音樣本提取 (已完成) → GPT-SoVITS 訓練 → RVC 訓練 → 語音合成
     ↓                           ↓              ↓           ↓
  Whisper+Pyannote          兩階段訓練      聲音轉換    最終輸出
```

### 環境要求

- **ai_env**: 語音樣本提取 (Whisper, Pyannote)
- **voice_env**: GPT-SoVITS & RVC 訓練與推理
- **GPU**: RTX 5080 16GB
- **CUDA**: 12.8
- **PyTorch**: 2.7.1+cu128

---

## 第一部分：GPT-SoVITS 訓練

### 1.1 快速開始

訓練 Luca 角色語音模型：

```bash
# 啟動 voice_env 環境
conda activate voice_env

# 完整訓練流程 (自動執行兩個階段)
python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --mode full \
  --s1-epochs 15 \
  --s2-epochs 10 \
  --device cuda
```

**預計訓練時間**: 2-4 小時（取決於樣本數量和 GPU）

### 1.2 訓練階段說明

#### 階段 1: GPT 模型訓練

**目的**: 學習文本到語義 token 的映射

```bash
python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --mode s1 \
  --s1-epochs 15 \
  --s1-batch-size 8 \
  --device cuda
```

**訓練參數**:
- `--s1-epochs`: 訓練輪數 (默認: 15)
- `--s1-batch-size`: 批次大小 (默認: 8)
- 學習率: 0.01 (with warmup)
- 精度: 16-mixed (half precision)

**輸出**:
- `logs/Luca/s1_ckpt/`: GPT checkpoint 文件
- `logs/Luca/s1_config.yaml`: 訓練配置

#### 階段 2: SoVITS 模型訓練

**目的**: 學習語義 token 到音頻波形的映射

```bash
python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --mode s2 \
  --s2-epochs 10 \
  --s2-batch-size 8 \
  --device cuda
```

**訓練參數**:
- `--s2-epochs`: 訓練輪數 (默認: 10)
- `--s2-batch-size`: 批次大小 (默認: 8)
- 學習率: 0.0001
- 採樣率: 32000 Hz

**輸出**:
- `logs/Luca/s2_ckpt/`: SoVITS model 文件
- `logs/Luca/s2_config.json`: 訓練配置

### 1.3 僅準備資料 (不訓練)

```bash
python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --mode prepare
```

這會：
1. 轉換 `training_filelist.json` 為 GPT-SoVITS 格式
2. 創建 `train.list` 和 `val.list`
3. 複製音頻文件到 GPT-SoVITS 目錄
4. 90/10 train/val 分割

### 1.4 進階參數

```bash
python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Alberto \
  --samples data/films/luca/voice_samples_auto/by_character/Alberto \
  --output models/voices/alberto/gpt_sovits \
  --gpt-sovits-root /mnt/c/AI_LLM_projects/GPT-SoVITS \
  --pretrained-gpt /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt \
  --pretrained-sovits /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s2G488k.pth \
  --language en \
  --s1-epochs 20 \
  --s2-epochs 15 \
  --s1-batch-size 4 \
  --s2-batch-size 4 \
  --log-level DEBUG
```

**所有參數說明**:

| 參數 | 描述 | 默認值 |
|------|------|--------|
| `--character` | 角色名稱 | (必需) |
| `--samples` | 語音樣本目錄 | (必需) |
| `--output` | 輸出目錄 | (必需) |
| `--gpt-sovits-root` | GPT-SoVITS 專案根目錄 | `/mnt/c/AI_LLM_projects/GPT-SoVITS` |
| `--pretrained-gpt` | 預訓練 GPT 模型路徑 | `ai_warehouse/.../s1bert25hz...ckpt` |
| `--pretrained-sovits` | 預訓練 SoVITS 模型路徑 | `ai_warehouse/.../s2G488k.pth` |
| `--s1-epochs` | GPT 訓練輪數 | 15 |
| `--s2-epochs` | SoVITS 訓練輪數 | 10 |
| `--s1-batch-size` | GPT 批次大小 | 8 |
| `--s2-batch-size` | SoVITS 批次大小 | 8 |
| `--language` | 語言代碼 | `en` |
| `--device` | 設備 | `cuda` |
| `--mode` | 訓練模式 | `full` |
| `--log-level` | 日誌級別 | `INFO` |

### 1.5 輸出結構

訓練完成後的目錄結構：

```
models/voices/luca/gpt_sovits/
├── data/                           # 處理後的數據
│   └── ...
├── training_metadata.json          # 訓練元數據
└── (trained models in GPT-SoVITS logs/)

/mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/
├── 0-audio/                        # 訓練音頻
│   ├── Luca_0000.wav
│   ├── Luca_0001.wav
│   └── ...
├── train.list                      # 訓練列表
├── val.list                        # 驗證列表
├── s1_config.yaml                  # GPT 配置
├── s2_config.json                  # SoVITS 配置
├── s1_ckpt/                        # GPT checkpoints
│   ├── Luca-e15.ckpt              # 最終 GPT 模型
│   └── ...
└── s2_ckpt/                        # SoVITS checkpoints
    ├── Luca-e10.pth               # 最終 SoVITS 模型
    └── ...
```

---

## 第二部分：RVC 訓練

### 2.1 快速開始

*(待實現)*

```bash
# 訓練 RVC 聲音轉換模型
python scripts/synthesis/tts/rvc_trainer.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/rvc \
  --device cuda
```

### 2.2 RVC 用途

**RVC (Retrieval-based Voice Conversion)** 用於：
- 實時語音轉換 (任意聲音 → 角色聲音)
- 聲音微調和增強
- 音高和音色控制
- 低延遲推理 (<100ms)

**與 GPT-SoVITS 的區別**:
- **GPT-SoVITS**: 文本 → 語音 (TTS)
- **RVC**: 語音 → 語音 (Voice Conversion)

---

## 第三部分：語音合成

### 3.1 基本 TTS

使用訓練好的 GPT-SoVITS 模型生成語音：

```bash
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "Ciao! My name is Luca." \
  --output test_luca_speech.wav \
  --gpt-model /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/s1_ckpt/Luca-e15.ckpt \
  --sovits-model /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/s2_ckpt/Luca-e10.pth \
  --language en \
  --device cuda
```

### 3.2 帶情緒控制的 TTS

```bash
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "Silenzio, Bruno!" \
  --emotion excited \
  --emotion-intensity 0.8 \
  --output test_luca_excited.wav
```

### 3.3 語音轉換 (RVC)

將任意語音轉換為 Luca 的聲音：

```bash
python scripts/synthesis/tts/voice_convert.py \
  --input /path/to/any_voice.wav \
  --target-character Luca \
  --rvc-model models/voices/luca/rvc/luca_rvc.pth \
  --output converted_to_luca.wav
```

---

## 第四部分：訓練所有角色

### 4.1 批量訓練腳本

創建一個批量訓練腳本來訓練所有角色：

```bash
#!/bin/bash
# train_all_characters.sh

CHARACTERS=("Luca" "Alberto" "Giulia" "Daniela" "Massimo" "Lorenzo" "Ercole")

for CHAR in "${CHARACTERS[@]}"; do
  echo "========================================="
  echo "Training $CHAR"
  echo "========================================="

  python scripts/synthesis/tts/gpt_sovits_trainer.py \
    --character "$CHAR" \
    --samples "data/films/luca/voice_samples_auto/by_character/$CHAR" \
    --output "models/voices/${CHAR,,}/gpt_sovits" \
    --mode full \
    --s1-epochs 15 \
    --s2-epochs 10 \
    --device cuda

  if [ $? -ne 0 ]; then
    echo "❌ Training failed for $CHAR"
    exit 1
  fi

  echo "✅ $CHAR training complete!"
  echo ""
done

echo "🎉 All characters trained successfully!"
```

運行：
```bash
chmod +x train_all_characters.sh
./train_all_characters.sh
```

### 4.2 並行訓練

如果有多個 GPU：

```bash
# GPU 0: Luca
CUDA_VISIBLE_DEVICES=0 python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Luca --samples ... --device cuda &

# GPU 1: Alberto
CUDA_VISIBLE_DEVICES=1 python scripts/synthesis/tts/gpt_sovits_trainer.py \
  --character Alberto --samples ... --device cuda &

wait
echo "All training complete!"
```

---

## 第五部分：故障排除

### 5.1 常見問題

#### 問題 1: CUDA Out of Memory

**症狀**: `RuntimeError: CUDA out of memory`

**解決方案**:
```bash
# 減少 batch size
--s1-batch-size 4  # 從 8 降到 4
--s2-batch-size 4

# 或使用 CPU (非常慢)
--device cpu
```

#### 問題 2: 訓練資料格式錯誤

**症狀**: `FileNotFoundError` 或 `KeyError`

**解決方案**:
```bash
# 確認 training_filelist.json 存在
ls data/films/luca/voice_samples_auto/by_character/Luca/training_filelist.json

# 檢查格式
head -20 data/films/luca/voice_samples_auto/by_character/Luca/training_filelist.json
```

#### 問題 3: 預訓練模型找不到

**症狀**: `pretrained model not found`

**解決方案**:
```bash
# 驗證預訓練模型存在
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/rvc/pretrained/

# 應該看到:
# - s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt (~148 MB)
# - s2G488k.pth (~102 MB)
```

### 5.2 質量評估

訓練完成後，評估模型質量：

```bash
# 生成測試語音
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "This is a test of the trained voice model." \
  --output quality_test.wav

# 聽一聽並評估:
# 1. 聲音相似度: 聽起來像 Luca 嗎？
# 2. 自然度: 語音是否自然流暢？
# 3. 清晰度: 是否清楚易懂？
# 4. 韻律: 語調和節奏是否正確？
```

**目標指標**:
- 聲音相似度: > 85%
- 自然度: > 90%
- MOS (Mean Opinion Score): > 4.0/5.0

### 5.3 重新訓練

如果質量不理想，嘗試：

1. **增加訓練輪數**:
   ```bash
   --s1-epochs 20  # 增加到 20
   --s2-epochs 15
   ```

2. **使用更多語音樣本**: 確保至少有 3-5 分鐘的清晰語音

3. **調整學習率**: 在 `create_s1_config()` 或 `create_s2_config()` 中修改

4. **使用更大的模型**: 嘗試 `s1big.yaml` 配置

---

## 第六部分：生產環境部署

### 6.1 模型導出

訓練完成後，導出最終模型：

```bash
# 複製到統一位置
mkdir -p models/voices/production/luca

cp /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/s1_ckpt/Luca-e15.ckpt \
   models/voices/production/luca/gpt.ckpt

cp /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/s2_ckpt/Luca-e10.pth \
   models/voices/production/luca/sovits.pth

# 創建元數據
cat > models/voices/production/luca/metadata.json <<EOF
{
  "character": "Luca",
  "language": "en",
  "gpt_model": "gpt.ckpt",
  "sovits_model": "sovits.pth",
  "training_date": "$(date -I)",
  "training_samples": $(jq 'length' data/films/luca/voice_samples_auto/by_character/Luca/training_filelist.json),
  "model_version": "1.0"
}
EOF
```

### 6.2 API 服務

創建 FastAPI 服務用於生產環境：

```python
# api/tts_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TTSRequest(BaseModel):
    character: str
    text: str
    language: str = "en"
    emotion: str = "neutral"

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    # Load model and generate speech
    # Return audio file
    pass
```

啟動服務：
```bash
uvicorn api.tts_service:app --host 0.0.0.0 --port 8000
```

---

## 附錄

### A. 目錄結構總覽

```
/mnt/c/AI_LLM_projects/
├── GPT-SoVITS/                     # GPT-SoVITS 專案
│   └── logs/                       # 訓練輸出
│       ├── Luca/
│       ├── Alberto/
│       └── ...
├── RVC/                            # RVC 專案
├── ai_warehouse/                   # 共享模型倉庫
│   └── models/audio/
│       ├── gpt_sovits/pretrained/
│       └── rvc/pretrained/
└── animation-ai-studio/
    ├── scripts/synthesis/tts/
    │   ├── gpt_sovits_trainer.py   # ✅ 已實現
    │   ├── rvc_trainer.py          # ⏳ 待實現
    │   ├── generate_speech.py      # ⏳ 待實現
    │   └── voice_convert.py        # ⏳ 待實現
    ├── models/voices/
    │   ├── luca/
    │   ├── alberto/
    │   └── production/
    └── data/films/luca/
        └── voice_samples_auto/
            └── by_character/
                ├── Luca/
                │   ├── training_filelist.json
                │   └── *.wav
                ├── Alberto/
                └── ...
```

### B. 參考資源

- **GPT-SoVITS GitHub**: https://github.com/RVC-Boss/GPT-SoVITS
- **RVC GitHub**: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **文檔**:
  - `docs/voice_system_architecture.md`
  - `docs/VOICE_SYSTEM_SETUP.md`

### C. 訓練時間估算

| 角色 | 樣本數量 | 總時長 | GPU | 訓練時間 (估計) |
|------|----------|--------|-----|----------------|
| Luca | ~200 | ~5 min | RTX 5080 | 2-3 小時 |
| Alberto | ~180 | ~4 min | RTX 5080 | 2-3 小時 |
| Giulia | ~150 | ~3 min | RTX 5080 | 1.5-2 小時 |

總計 (7 個角色): **約 12-15 小時**

---

**文檔版本**: v1.0
**最後更新**: 2025-11-20
**狀態**: ✅ GPT-SoVITS 訓練器已完成，RVC 訓練器待實現
