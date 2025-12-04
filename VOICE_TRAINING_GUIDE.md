# 語音訓練完整指南

**創建日期：** 2025-11-19
**狀態：** 準備完成 ✅
**目的：** 從電影視頻提取角色語音並訓練 GPT-SoVITS 語音模型

---

## 📋 概述

本指南提供完整的端到端流程，從電影視頻中提取角色語音樣本，並訓練高質量的語音合成模型。

### 已創建的腳本

| 腳本 | 功能 | 狀態 |
|------|------|------|
| `extract_audio.py` | 從視頻提取音軌 | ✅ 完成 |
| `extract_voice_samples.py` | Whisper + Pyannote 提取語音樣本 | ✅ 完成 |
| `voice_training_workflow.py` | 完整端到端工作流 | ✅ 完成 |
| `voice_dataset_builder.py` | 數據集構建工具（已存在） | ✅ 完成 |
| `voice_model_trainer.py` | GPT-SoVITS 訓練（已存在） | ✅ 完成 |

---

## 🎯 工作流程

### 完整流程（5個步驟）

```
視頻檔案 (luca_film.ts)
    ↓
[步驟 1] 提取音軌
    ↓
音頻檔案 (luca_audio.wav, 95分鐘, 48kHz)
    ↓
[步驟 2] Whisper 轉錄 + Pyannote 說話者分離
    ↓
語音片段 (按說話者分組, 帶轉錄文字)
    ↓
[步驟 3] 手動映射: 說話者 → 角色名稱
    ↓
[步驟 4] 按角色整理樣本
    ↓
[步驟 5] 生成訓練數據集
    ↓
GPT-SoVITS 訓練 → 角色語音模型
```

---

## 🚀 快速開始

### 環境準備

```bash
# 確保使用 conda ai_env 環境
export PATH="/home/b0979/.conda/envs/ai_env/bin:/usr/bin:/bin:$PATH"

# 安裝必要依賴
pip install openai-whisper pyannote.audio torch torchaudio
pip install ffmpeg-python soundfile librosa noisereduce

# 設置 HuggingFace token（Pyannote 需要）
export HF_TOKEN=your_huggingface_token

# 接受 Pyannote 模型使用條款
# 訪問：https://huggingface.co/pyannote/speaker-diarization-3.1
# 點擊 "Agree and access repository"
```

### 方法 1：一鍵完整流程（推薦）

```bash
# 對 Luca 電影完整流程
python scripts/synthesis/tts/voice_training_workflow.py \
    --film luca \
    --characters Luca Alberto Giulia \
    --num-speakers 3 \
    --language en

# 流程會自動執行：
# 1. 提取音軌
# 2. Whisper 轉錄
# 3. Pyannote 說話者分離
# 4. 互動式映射說話者到角色
# 5. 整理樣本並生成訓練數據集
```

### 方法 2：分步驟執行

#### 步驟 1：提取音軌

```bash
# 從電影提取音軌
python scripts/synthesis/tts/extract_audio.py --film luca

# 或從自定義視頻
python scripts/synthesis/tts/extract_audio.py \
    --input /path/to/video.mp4 \
    --output audio.wav \
    --sample-rate 48000 \
    --mono  # 可選：轉為單聲道
```

**輸出：**
```
data/films/luca/audio/luca_audio.wav
Size: ~150 MB (95 minutes, 48kHz stereo)
```

#### 步驟 2：提取語音樣本

```bash
# 使用 Whisper + Pyannote 提取語音片段
python scripts/synthesis/tts/extract_voice_samples.py \
    --audio data/films/luca/audio/luca_audio.wav \
    --output data/films/luca/voice_samples \
    --whisper-model medium \
    --language en \
    --num-speakers 3 \
    --device cuda

# 處理時間：約 20-40 分鐘（95分鐘音頻）
```

**輸出結構：**
```
data/films/luca/voice_samples/
├── full_transcription.json      # 完整轉錄
├── segments_metadata.json       # 所有語音片段元數據
├── SPEAKER_00/                  # 說話者 0 的所有片段
│   ├── SPEAKER_00_0001_12.34s.wav
│   ├── SPEAKER_00_0002_25.67s.wav
│   └── ...
├── SPEAKER_01/                  # 說話者 1 的所有片段
└── SPEAKER_02/                  # 說話者 2 的所有片段
```

#### 步驟 3：映射說話者到角色

```bash
# 手動聽語音樣本，判斷每個 SPEAKER_XX 對應哪個角色
# 創建映射文件：speaker_mapping.json

# 示例映射：
{
  "SPEAKER_00": "Luca",
  "SPEAKER_01": "Alberto",
  "SPEAKER_02": "Giulia"
}

# 或使用互動式工作流自動提示
```

#### 步驟 4：整理樣本

```bash
# 按角色整理語音樣本
python scripts/synthesis/tts/voice_training_workflow.py \
    --film luca \
    --start-step 4 \
    --end-step 5 \
    --skip-interactive
```

**輸出結構：**
```
data/films/luca/voice_samples/by_character/
├── Luca/
│   ├── SPEAKER_00_0001_12.34s.wav
│   ├── SPEAKER_00_0002_25.67s.wav
│   ├── ...
│   └── training_filelist.json  # 訓練用文件列表
├── Alberto/
│   └── ...
└── Giulia/
    └── ...
```

#### 步驟 5：訓練語音模型

```bash
# 使用 GPT-SoVITS 訓練 Luca 的語音模型
python scripts/synthesis/tts/voice_model_trainer.py \
    --character Luca \
    --samples data/films/luca/voice_samples/by_character/Luca \
    --output models/voices/luca \
    --epochs 100 \
    --batch-size 4 \
    --device cuda

# 訓練時間：約 2-4 小時 (RTX 5080)
```

#### 步驟 6：測試合成

```bash
# 使用訓練好的模型合成語音
python scripts/synthesis/tts/gpt_sovits_wrapper.py \
    --character Luca \
    --text "Silenzio, Bruno!" \
    --emotion excited \
    --output test_voice.wav
```

---

## 📊 預期結果

### 數據量估算

對於 95 分鐘的 Luca 電影：

| 項目 | 數量/大小 |
|------|-----------|
| 原始音軌 | ~150 MB WAV (48kHz) |
| 總語音片段 | ~500-800 個 |
| 每個主角片段 | ~150-250 個 |
| 每個主角總時長 | ~10-20 分鐘 |
| 可用訓練樣本 | ~100-150 個/角色 |
| 訓練數據大小 | ~50-100 MB/角色 |

### 質量指標

**好的語音樣本特徵：**
- 時長：1-10 秒
- 內容：清晰的完整句子
- SNR：> 15 dB
- 背景音：最小化（無音樂/音效）
- 單一說話者：無重疊對話

**訓練目標：**
- 相似度：> 85%
- 自然度：> 4.0/5.0 MOS
- 可理解度：> 95%

---

## 🛠️ 進階用法

### 自定義音頻片段提取

```bash
# 提取特定時間段
python scripts/synthesis/tts/extract_audio.py \
    --input video.mp4 \
    --output segment.wav \
    --start 120.5 \
    --duration 30

# 音頻標準化
python scripts/synthesis/tts/extract_audio.py \
    --input audio.wav \
    --output normalized.wav \
    --normalize

# 轉為單聲道（推薦用於訓練）
python scripts/synthesis/tts/extract_audio.py \
    --input audio.wav \
    --output mono.wav \
    --mono
```

### 調整 Whisper 參數

```bash
# 使用更大的模型（更準確但更慢）
python scripts/synthesis/tts/extract_voice_samples.py \
    --audio audio.wav \
    --output voice_samples \
    --whisper-model large \
    --device cuda

# 支持的模型大小：
# - tiny:   最快，最不準確
# - base:   快速，基本準確
# - small:  平衡
# - medium: 推薦（預設）
# - large:  最準確但最慢
```

### 調整說話者數量

```bash
# 如果自動檢測的說話者不正確，可以手動指定
python scripts/synthesis/tts/extract_voice_samples.py \
    --audio audio.wav \
    --output voice_samples \
    --num-speakers 5  # 強制識別 5 個說話者
```

---

## ⚠️ 常見問題

### 問題 1：Pyannote 認證失敗

**錯誤：**
```
OSError: You are trying to access a gated repo.
```

**解決：**
1. 在 HuggingFace 創建帳號
2. 訪問 https://huggingface.co/pyannote/speaker-diarization-3.1
3. 點擊 "Agree and access repository"
4. 生成 token: https://huggingface.co/settings/tokens
5. 設置環境變量：`export HF_TOKEN=your_token`

### 問題 2：Whisper OOM（記憶體不足）

**解決：**
```bash
# 使用更小的模型
--whisper-model small  # 或 base, tiny

# 或增加批次處理
# （腳本已自動處理，無需手動調整）
```

### 問題 3：語音樣本質量差

**原因：**
- 背景音樂/音效太大
- 多個角色同時說話
- 說話聲音太小

**解決：**
1. 手動篩選：刪除質量差的樣本
2. 調整過濾參數：
   ```python
   # 在 extract_voice_samples.py 中修改
   min_duration=2.0,  # 只要 >2秒 的片段
   max_duration=8.0,  # 只要 <8秒 的片段
   ```
3. 音頻增強：使用 `noisereduce` 降噪

### 問題 4：說話者識別錯誤

**症狀：**
- Luca 的語音被分配到多個說話者
- 多個角色被歸為同一說話者

**解決：**
1. 調整 `--num-speakers` 參數
2. 手動重新分組：
   ```bash
   # 合併兩個說話者
   mv voice_samples/SPEAKER_01/* voice_samples/SPEAKER_00/
   ```
3. 使用更長的音頻片段（說話者識別需要足夠的語音特徵）

---

## 📂 完整文件結構

```
animation-ai-studio/
├── data/films/luca/
│   ├── audio/
│   │   ├── luca_audio.wav              # 步驟 1 輸出
│   │   ├── luca_audio_mono.wav         # （可選）單聲道版本
│   │   └── luca_audio_normalized.wav   # （可選）標準化版本
│   │
│   └── voice_samples/                   # 步驟 2 輸出
│       ├── full_transcription.json     # Whisper 完整轉錄
│       ├── segments_metadata.json      # 所有片段元數據
│       ├── speaker_mapping.json        # 步驟 3 創建
│       │
│       ├── SPEAKER_00/                 # 原始說話者分組
│       ├── SPEAKER_01/
│       ├── SPEAKER_02/
│       │
│       └── by_character/               # 步驟 4 輸出
│           ├── Luca/
│           │   ├── *.wav
│           │   └── training_filelist.json
│           ├── Alberto/
│           └── Giulia/
│
├── models/voices/                       # 步驟 5 輸出
│   ├── luca/
│   │   ├── luca_gpt.ckpt
│   │   ├── luca_sovits.pth
│   │   └── luca_reference.wav
│   ├── alberto/
│   └── giulia/
│
└── scripts/synthesis/tts/
    ├── extract_audio.py                # 音軌提取
    ├── extract_voice_samples.py        # 語音樣本提取
    ├── voice_training_workflow.py      # 完整工作流
    ├── voice_dataset_builder.py        # 數據集構建
    ├── voice_model_trainer.py          # GPT-SoVITS 訓練
    ├── gpt_sovits_wrapper.py           # 語音合成包裝器
    └── emotion_controller.py           # 情緒控制
```

---

## 🎯 下一步行動

### 立即可做

1. **測試音軌提取：**
   ```bash
   python scripts/synthesis/tts/extract_audio.py --film luca
   ```
   預計時間：1-2 分鐘

2. **安裝 Whisper 和 Pyannote：**
   ```bash
   pip install openai-whisper pyannote.audio
   export HF_TOKEN=your_token
   ```

### 完整流程（建議在 tmux/screen 中運行）

```bash
# 創建 tmux 會話
tmux new -s voice_training

# 運行完整工作流
python scripts/synthesis/tts/voice_training_workflow.py \
    --film luca \
    --characters Luca Alberto Giulia \
    --num-speakers 3 \
    --language en

# 預計總時間：
# - 音軌提取：    1-2 分鐘
# - Whisper:      20-30 分鐘
# - Pyannote:     10-20 分鐘
# - 手動映射：    5 分鐘
# - 整理樣本：    1 分鐘
# 總計：          約 40-60 分鐘
```

---

## 📝 檢查清單

語音訓練準備完成度：

- [x] 視頻檔案確認（`/mnt/c/raw_videos/luca/luca_film.ts`）
- [x] 音軌提取腳本創建
- [x] 語音樣本提取腳本創建
- [x] 完整工作流腳本創建
- [x] 使用文檔完成
- [ ] 環境依賴安裝（Whisper, Pyannote）
- [ ] HuggingFace Token 設置
- [ ] 運行音軌提取測試
- [ ] 運行語音樣本提取
- [ ] 訓練第一個語音模型
- [ ] 測試語音合成

---

## 🔗 相關資源

### 文檔
- `scripts/synthesis/tts/README.md` - TTS 模塊文檔
- `docs/modules/voice-synthesis.md` - 語音合成架構文檔

### 外部資源
- [Whisper GitHub](https://github.com/openai/whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [HuggingFace Pyannote Models](https://huggingface.co/pyannote)

---

**最後更新：** 2025-11-19
**狀態：** 準備完成，等待測試 ✅
