# Voice Synthesis Quick Start Guide

快速開始指南 - 使用 Enhanced XTTS-v2 進行角色語音合成

---

## 快速開始 (5 分鐘)

### 1. 單行文本測試

最簡單的使用方式 - 生成一句話:

```bash
export PATH="/home/b0979/.conda/envs/voice_env/bin:$PATH"
export COQUI_TOS_AGREED=1

python scripts/synthesis/tts/test_xtts_enhanced.py \
  --character Luca \
  --text "Hello! My name is Luca." \
  --num-refs 5
```

**輸出**: `outputs/tts/xtts_enhanced/luca/` 目錄下 5 個音頻變體

---

## 生產級批量生成

### 2. 準備對話腳本

創建文本文件 `my_dialogue.txt`:

```text
# Scene 1
Ciao! My name is Luca. What's your name?

# Scene 2
This place is amazing! I've never seen anything like it before.

# Scene 3
Silenzio Bruno! We can do this. Let's go!
```

### 3. 批量生成

```bash
export PATH="/home/b0979/.conda/envs/voice_env/bin:$PATH"
export COQUI_TOS_AGREED=1

python scripts/synthesis/tts/batch_voice_generation.py \
  --input my_dialogue.txt \
  --character Luca \
  --num-refs 5 \
  --temperature 0.65 \
  --top-k 40 \
  --top-p 0.90
```

**輸出結構**:
```
outputs/tts/batch/luca_20251120_162300/
├── line_001/
│   ├── luca_variant1_20251120_162301.wav
│   ├── luca_variant2_20251120_162302.wav
│   ├── ...
│   └── generation_result.json
├── line_002/
│   └── ...
└── batch_results_20251120_162500.json  # 批量統計
```

---

## 參數調整指南

### 品質參數

| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `--num-refs` | 5 | 參考樣本數量 | 3-5,越多品質越好但速度稍慢 |
| `--temperature` | 0.65 | 採樣溫度 | 0.6-0.7,越低越一致 |
| `--top-k` | 40 | Top-k 採樣 | 30-50,越低越穩定 |
| `--top-p` | 0.90 | 核採樣 | 0.85-0.95,越低越保守 |

### 常見場景設定

**1. 最高品質 (慢但最好)**
```bash
--num-refs 5 --temperature 0.65 --top-k 40 --top-p 0.90
```

**2. 平衡設定 (推薦)**
```bash
--num-refs 3 --temperature 0.70 --top-k 50 --top-p 0.85
```

**3. 快速預覽 (快但品質略低)**
```bash
--num-refs 1 --temperature 0.75 --top-k 50 --top-p 0.80
```

---

## 長篇語音最佳實踐

### 為什麼要生成長篇 (1分鐘+)?

- ✅ **韻律連貫性更好** - 語調更自然流暢
- ✅ **情感表達更豐富** - 有足夠空間展現情緒
- ✅ **語音穩定性更高** - 聲音特徵更一致
- ✅ **自然停頓** - 句子間的停頓更像真人

### 長篇文本範例

```python
long_text = """
Summer in Portorosso is the most magical time of the year.
The sun shines bright over the colorful buildings, and the sea
sparkles like a thousand diamonds. I love racing down the hills
on my Vespa, feeling the wind in my hair and the freedom in my heart.
Alberto and I spend every day exploring new places, discovering hidden
treasures, and dreaming about our adventures.
"""

# 生成約 50-70 秒的音頻
```

---

## CSV 格式批量生成

### CSV 文件格式

創建 `dialogue.csv`:

```csv
id,text,character,notes
scene01_line01,"Ciao! My name is Luca.",Luca,"Happy greeting"
scene01_line02,"What's your name?",Luca,"Curious"
scene02_line01,"This place is amazing!",Luca,"Excited"
```

### 生成命令

```bash
python scripts/synthesis/tts/batch_voice_generation.py \
  --input dialogue.csv \
  --character Luca
```

**優勢**:
- 支持多角色 (自動過濾)
- 包含元數據 (場景、情緒等)
- 易於管理大型項目

---

## 整合到動畫工作流程

### 1. 對話配音管線

```bash
# Step 1: 準備腳本
vim animation_script.txt

# Step 2: 批量生成
python scripts/synthesis/tts/batch_voice_generation.py \
  --input animation_script.txt \
  --character Luca

# Step 3: 挑選最佳變體
# 聆聽所有變體,選擇最自然的

# Step 4: 導入動畫軟件
# 將選中的 WAV 文件導入 Blender/Maya/等
```

### 2. 自動化工作流程

```bash
#!/bin/bash
# auto_voice_generation.sh

CHARACTER="Luca"
SCRIPT_FILE="$1"

export PATH="/home/b0979/.conda/envs/voice_env/bin:$PATH"
export COQUI_TOS_AGREED=1

echo "Generating voices for $CHARACTER..."

python scripts/synthesis/tts/batch_voice_generation.py \
  --input "$SCRIPT_FILE" \
  --character "$CHARACTER" \
  --num-refs 5 \
  --temperature 0.65 \
  --top-k 40 \
  --top-p 0.90

echo "✅ Generation complete! Check outputs/tts/batch/"
```

使用:
```bash
bash auto_voice_generation.sh my_script.txt
```

---

## 性能指標

### 生成速度

| 音頻長度 | 生成時間 | 實時因子 |
|----------|----------|----------|
| 10 秒 | ~6 秒 | 0.60x |
| 30 秒 | ~16 秒 | 0.53x |
| 60 秒 | ~36 秒 | 0.60x |

**實時因子**: 0.52-0.62 (約 1.6-1.9 倍快於實時)

### 硬件需求

- **GPU**: RTX 5080 (或類似,8GB+ VRAM)
- **VRAM 使用**: ~8GB (生成時)
- **CPU**: 任意現代 CPU
- **存儲**: 每小時音頻約 150-200 MB

---

## 常見問題

### Q: 生成的音質不夠好?

**A**: 調整參數:
1. 增加參考樣本: `--num-refs 5`
2. 降低溫度: `--temperature 0.60`
3. 使用更長的文本 (50+ 秒)

### Q: 聲音不夠穩定?

**A**:
- 檢查參考樣本品質
- 使用 `--top-k 30 --top-p 0.85` (更保守的採樣)
- 確保文本有足夠長度

### Q: 生成速度太慢?

**A**:
- 減少參考樣本: `--num-refs 1`
- 批量生成時會自動優化 (模型只加載一次)

### Q: 想要不同情緒的語音?

**A**:
- XTTS-v2 會自動從文本推斷情緒
- 使用感嘆句、問句等自然文本
- 參考樣本會影響情緒風格

---

## 文件位置速查

### 腳本
- 基礎測試: `scripts/synthesis/tts/test_xtts_voice_cloning.py`
- 增強版: `scripts/synthesis/tts/test_xtts_enhanced.py`
- 批量生成: `scripts/synthesis/tts/batch_voice_generation.py`

### 範例
- 對話腳本: `data/prompts/example_dialogue_script.txt`

### 輸出
- 單次測試: `outputs/tts/xtts_enhanced/luca/`
- 批量生成: `outputs/tts/batch/luca_<timestamp>/`

### 文檔
- 完整配置: `docs/voice_synthesis_setup.md`
- 快速指南: `docs/voice_synthesis_quick_start.md` (本文件)

---

## 下一步

1. **測試系統**: 運行單行文本測試確保一切正常
2. **準備腳本**: 編寫您的動畫對話腳本
3. **批量生成**: 使用批量工具生成所有語音
4. **挑選變體**: 聆聽並選擇最佳音頻
5. **整合動畫**: 將音頻導入您的動畫項目

---

**系統狀態**: ✅ 生產就緒
**最後更新**: 2025-11-20
**版本**: v1.0 - Enhanced XTTS-v2

有問題? 查看完整文檔: `docs/voice_synthesis_setup.md`
