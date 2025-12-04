# 完整語音系統設置指南

**創建日期**: 2025-11-20
**狀態**: 設置中
**目的**: 建立完整的語音分析與合成系統

---

## 環境架構

### 環境 1: `ai_env` (分析環境)
**用途**:
- 視頻/音頻分析
- Whisper 轉錄
- Pyannote 說話者分離
- 語音特徵提取

**PyTorch版本**: 2.7.1+cu128
**PyTorch Lightning**: 1.9.0 (pyannote 要求)

**主要套件**:
- openai-whisper
- pyannote.audio==3.4.0
- librosa
- soundfile

### 環境 2: `voice_env` (訓練/合成環境)
**用途**:
- GPT-SoVITS 訓練與推理
- RVC 訓練與推理
- 情緒識別
- 語境分析

**PyTorch版本**: 2.7.1+cu128 (與 ai_env 一致)
**PyTorch Lightning**: 2.4+ (GPT-SoVITS 要求)

**主要套件**:
- GPT-SoVITS
- RVC
- transformers
- gradio

---

## 環境設置步驟

### Step 1: 創建 voice_env 環境

```bash
# 創建環境
conda create -n voice_env python=3.10 -y

# 啟動環境
conda activate voice_env

# 安裝 PyTorch 2.7.1+cu128
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: 安裝 GPT-SoVITS 依賴

```bash
# 核心依賴
pip install 'numpy<2.0' scipy tensorboard librosa==0.10.2 numba \
  gradio ffmpeg-python onnxruntime-gpu tqdm transformers peft \
  sentencepiece chardet PyYAML psutil

# PyTorch Lightning (較新版本)
pip install 'pytorch-lightning>=2.4'

# GPT-SoVITS 特定依賴
pip install funasr==1.0.27 cn2an pypinyin pyopenjtalk g2p_en \
  modelscope==1.10.0 jieba split-lang fast_langdetect rotary_embedding_torch \
  x_transformers torchmetrics 'pydantic<=2.10.6' 'ctranslate2>=4.0,<5' \
  'huggingface_hub>=0.13' 'tokenizers>=0.13,<1' 'av>=11'

# 中文語言支持（可選）
pip install ToJyutping g2pk2 ko_pron opencc jieba_fast wordsegment

# Web API
pip install 'fastapi[standard]>=0.115.2'
```

### Step 3: 克隆 GPT-SoVITS 和 RVC

```bash
cd /mnt/c/AI_LLM_projects/

# GPT-SoVITS
git clone https://github.com/RVC-Boss/GPT-SoVITS.git

# RVC
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git RVC
```

### Step 4: 下載預訓練模型

```bash
# 創建模型目錄
mkdir -p /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained
mkdir -p /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/rvc/pretrained

# GPT-SoVITS 預訓練模型
cd /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained

# 下載 GPT 模型 (~1.5 GB)
wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/pretrained_models/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt \
  -O GPT_SoVITS-e15.ckpt

# 下載 SoVITS 模型 (~500 MB)
wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/pretrained_models/s2G488k.pth

# RVC 預訓練模型
cd /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/rvc/pretrained

# HuBERT Base (~200 MB)
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

# RMVPEv2 F0 predictor (~50 MB)
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
```

### Step 5: 驗證環境

```bash
# 啟動 voice_env
conda activate voice_env

# 驗證 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 預期輸出:
# PyTorch: 2.7.1+cu128
# CUDA: True

# 驗證其他套件
python -c "import librosa, transformers, gradio; print('All packages OK')"
```

---

## 環境切換

### 使用 ai_env (分析)
```bash
conda activate ai_env

# 提取語音樣本
python scripts/synthesis/tts/extract_voice_samples.py \
  --audio data/films/luca/audio/luca_audio.wav \
  --output data/films/luca/voice_samples_auto \
  --whisper-model medium \
  --language en \
  --device cuda
```

### 使用 voice_env (訓練/合成)
```bash
conda activate voice_env

# 訓練 GPT-SoVITS 模型
python scripts/synthesis/tts/train_gpt_sovits.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca \
  --device cuda

# 語音合成
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "Ciao! My name is Luca." \
  --output test_luca.wav
```

---

## 目錄結構

```
/mnt/c/AI_LLM_projects/
├── GPT-SoVITS/              # GPT-SoVITS 專案（獨立）
├── RVC/                      # RVC 專案（獨立）
├── ai_warehouse/             # 共享模型倉庫
│   └── models/audio/
│       ├── gpt_sovits/pretrained/
│       ├── rvc/pretrained/
│       └── emotion/
│
└── animation-ai-studio/
    │
├── scripts/synthesis/tts/
│   ├── train_gpt_sovits.py         # GPT-SoVITS 訓練
│   ├── train_rvc.py                 # RVC 訓練
│   ├── generate_speech.py          # 語音合成
│   ├── voice_convert.py            # 語音轉換
│   ├── emotion_recognition.py      # 情緒識別
│   └── context_aware_tts.py        # 語境分析
│
└── models/voices/              # 訓練好的模型
    ├── luca/
    │   ├── gpt_sovits/
    │   │   ├── luca_gpt.ckpt
    │   │   └── luca_sovits.pth
    │   ├── rvc/
    │   │   └── luca_rvc.pth
    │   └── metadata.json
    ├── alberto/
    └── giulia/

/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/
├── gpt_sovits/
│   └── pretrained/
│       ├── GPT_SoVITS-e15.ckpt      # 1.5 GB
│       └── s2G488k.pth               # 500 MB
├── rvc/
│   └── pretrained/
│       ├── hubert_base.pt            # 200 MB
│       └── rmvpe.pt                  # 50 MB
└── emotion/
    └── wav2vec2-emotion/             # 400 MB (待下載)
```

---

## 使用範例

### 1. 訓練角色語音模型 (Luca)

```bash
# 切換到 voice_env
conda activate voice_env

# 訓練 GPT-SoVITS
python scripts/synthesis/tts/train_gpt_sovits.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --base-gpt /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/GPT_SoVITS-e15.ckpt \
  --base-sovits /mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/s2G488k.pth \
  --output models/voices/luca/gpt_sovits \
  --epochs 100 \
  --batch-size 4 \
  --device cuda

# 預計時間: 2-4 小時
```

### 2. 生成語音

```bash
# 基本 TTS
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "Silenzio, Bruno!" \
  --output test_luca_speech.wav

# 帶情緒控制
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "Silenzio, Bruno!" \
  --emotion excited \
  --emotion-intensity 0.8 \
  --output test_luca_excited.wav
```

### 3. 語音轉換 (RVC)

```bash
# 將任意語音轉換為 Luca 的聲音
python scripts/synthesis/tts/voice_convert.py \
  --input /path/to/any_voice.wav \
  --target-character Luca \
  --output converted_to_luca.wav
```

### 4. 情緒識別

```bash
# 從語音中檢測情緒
python scripts/synthesis/tts/emotion_recognition.py \
  --audio data/films/luca/voice_samples_auto/by_character/Luca/sample_001.wav \
  --output emotion_analysis.json

# 輸出範例:
# {
#   "dominant_emotion": "happy",
#   "confidence": 0.87,
#   "all_emotions": {
#     "happy": 0.87,
#     "excited": 0.45,
#     "neutral": 0.12,
#     ...
#   }
# }
```

---

## 故障排除

### 問題 1: PyTorch Lightning 版本衝突

**症狀**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
This behaviour is the source of the following dependency conflicts.
pyannote-audio requires pytorch-lightning<1.10,>=1.5.4
```

**解決**: 不用擔心，ai_env 和 voice_env 是分離的環境

### 問題 2: CUDA Out of Memory

**症狀**: RuntimeError: CUDA out of memory

**解決**:
```bash
# 減少 batch size
--batch-size 2  # 從 4 降到 2

# 使用 gradient accumulation
--gradient-accumulation-steps 2
```

### 問題 3: FFmpeg 找不到

**症狀**: FileNotFoundError: ffmpeg not found

**解決**:
```bash
# Ubuntu/WSL
sudo apt update && sudo apt install ffmpeg

# 驗證
ffmpeg -version
```

---

## 進階配置

### 多 GPU 訓練

```bash
# 使用所有可用 GPU
python scripts/synthesis/tts/train_gpt_sovits.py \
  --character Luca \
  --devices 0,1 \  # 使用 GPU 0 和 1
  --strategy ddp \  # 分散式訓練
  ...

# 僅使用 RTX 5080
python scripts/synthesis/tts/train_gpt_sovits.py \
  --device cuda:0 \
  ...
```

### 優化推理速度

```bash
# 使用 FP16
python scripts/synthesis/tts/generate_speech.py \
  --character Luca \
  --text "Hello" \
  --precision fp16 \
  --output fast_speech.wav

# 批量合成
python scripts/synthesis/tts/batch_generate.py \
  --character Luca \
  --texts-file dialogue_list.txt \
  --output-dir outputs/batch_speech/ \
  --num-workers 4
```

---

## 下一步

1. ✓ 環境創建 (voice_env)
2. ✓ PyTorch 2.7.1+cu128 安裝
3. 🔄 安裝所有依賴
4. ⏳ 下載預訓練模型
5. ⏳ 實現訓練腳本
6. ⏳ 訓練 Luca 模型
7. ⏳ 測試語音合成
8. ⏳ 整合情緒控制
9. ⏳ 整合 RVC
10. ⏳ 完整測試

---

**狀態**: 正在安裝依賴
**預計完成時間**: 2-3 小時（安裝 + 首次訓練）
