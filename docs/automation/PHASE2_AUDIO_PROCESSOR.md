# Phase 2: 音訊處理器 (Audio Processor)

**專案 (Project)**: Animation AI Studio - CPU-Only Automation Infrastructure
**元件 (Component)**: Phase 2.3 - Audio Processor (音訊處理器)
**狀態 (Status)**: ✅ 完成 (Complete)
**完成日期 (Completion Date)**: 2025-12-02
**作者 (Author)**: Animation AI Studio Team

---

## 目錄 (Table of Contents)

1. [概述 (Overview)](#概述-overview)
2. [功能特色 (Features)](#功能特色-features)
3. [安裝需求 (Requirements)](#安裝需求-requirements)
4. [快速開始 (Quick Start)](#快速開始-quick-start)
5. [操作模式 (Operations)](#操作模式-operations)
6. [批次處理 (Batch Processing)](#批次處理-batch-processing)
7. [音訊格式指南 (Audio Format Guide)](#音訊格式指南-audio-format-guide)
8. [工作流程範例 (Workflow Examples)](#工作流程範例-workflow-examples)
9. [參數詳解 (Parameter Details)](#參數詳解-parameter-details)
10. [效能與最佳化 (Performance & Optimization)](#效能與最佳化-performance--optimization)
11. [疑難排解 (Troubleshooting)](#疑難排解-troubleshooting)
12. [API 參考 (API Reference)](#api-參考-api-reference)

---

## 概述 (Overview)

**Audio Processor** 是一個基於 FFmpeg 的 CPU 專用音訊處理工具，提供完整的音訊處理功能，包括提取、轉換、切割、拼接、音量正規化和靜音處理。

### 關鍵特性 (Key Features)

- **CPU 專用**: 完全不使用 GPU 資源，與訓練任務並行運行
- **32 執行緒最佳化**: 充分利用 32 核心 CPU 的計算能力
- **記憶體安全**: 整合 Phase 1 安全基礎設施，自動監控記憶體使用
- **格式支援**: 支援所有常見音訊格式 (WAV, MP3, FLAC, AAC, OGG)
- **批次處理**: 支援 YAML 配置檔案進行大規模批次操作
- **中英雙語**: 完整的中英文雙語文件和日誌輸出

### 使用場景 (Use Cases)

1. **影片音訊提取**: 從動畫影片中提取高品質音訊軌道
2. **格式轉換**: 在不同音訊格式間轉換（無損/有損）
3. **音訊編輯**: 切割、拼接音訊片段
4. **音量處理**: 正規化音量至標準響度
5. **靜音處理**: 檢測並移除音訊中的靜音片段
6. **批次工作流程**: 自動化處理大量音訊檔案

---

## 功能特色 (Features)

### 1. 音訊提取 (Audio Extraction)

從影片檔案中提取音訊軌道，支援各種影片格式。

**支援的影片格式**:
- MP4, MKV, AVI, MOV, TS, M4V, WebM

**輸出格式**:
- WAV (無損)
- MP3 (有損)
- FLAC (無損)
- AAC (有損)
- OGG (有損)

**特色**:
- 32 執行緒加速處理
- 保留原始音訊品質
- 自動檢測影片屬性
- 支援多聲道音訊

### 2. 格式轉換 (Format Conversion)

在不同音訊格式間進行轉換，支援自訂取樣率、聲道和位元率。

**轉換選項**:
- 取樣率 (Sample Rate): 44100 Hz, 48000 Hz, 96000 Hz
- 聲道 (Channels): 1 (單聲道), 2 (立體聲)
- 位元率 (Bitrate): 128k, 192k, 320k

**常見轉換**:
- WAV → MP3 (檔案壓縮)
- MP3 → WAV (後製編輯)
- WAV → FLAC (無損歸檔)
- 立體聲 → 單聲道 (語音處理)

### 3. 音訊切割 (Audio Cutting)

精確切割音訊片段，支援毫秒級精度。

**切割模式**:
- 指定起始時間 + 時長 (Start time + duration)
- 指定起始時間 + 結束時間 (Start time + end time)
- 批次切割多個片段

**應用場景**:
- 提取特定對話片段
- 移除不需要的部分
- 建立音訊樣本庫

### 4. 音訊拼接 (Audio Concatenation)

無縫拼接多個音訊檔案。

**拼接選項**:
- 支援不同格式的音訊檔案
- 自動格式統一
- 保持音訊品質

**應用場景**:
- 合併分段音訊
- 建立長音訊檔案
- 組合不同來源的音訊

### 5. 音量正規化 (Volume Normalization)

將音訊音量正規化至目標響度等級。

**正規化標準**:
- -16 dB: 標準響度目標（推薦）
- -12 dB: 較大聲（適合音樂）
- -20 dB: 較小聲（適合語音）

**應用場景**:
- 統一多個音訊的音量
- 符合廣播/串流標準
- 避免削波失真

### 6. 靜音檢測 (Silence Detection)

自動檢測音訊中的靜音片段。

**檢測參數**:
- 噪音閾值 (Noise Threshold): -30 dB 至 -50 dB
- 最小靜音時長 (Minimum Duration): 0.1 秒至 2.0 秒

**輸出資訊**:
- 靜音片段的起始時間
- 靜音片段的結束時間
- 靜音片段的時長

### 7. 靜音移除 (Silence Removal)

自動移除音訊中的靜音片段。

**移除選項**:
- 保留適當間隔
- 平滑過渡
- 避免突兀切換

**應用場景**:
- 語音錄音後製
- Podcast 編輯
- 縮短音訊長度

### 8. Metadata 提取 (Metadata Extraction)

提取音訊檔案的詳細 metadata。

**提取資訊**:
- 時長 (Duration)
- 取樣率 (Sample Rate)
- 聲道數 (Channels)
- 編碼格式 (Codec)
- 位元率 (Bitrate)
- 檔案大小 (File Size)
- 格式 (Format)

---

## 安裝需求 (Requirements)

### 系統需求 (System Requirements)

- **作業系統**: Linux (Ubuntu 20.04+, WSL2)
- **CPU**: 32 核心處理器（建議）
- **記憶體**: 16GB RAM（最低），32GB RAM（建議）
- **磁碟空間**: 視音訊檔案大小而定

### 軟體依賴 (Software Dependencies)

#### FFmpeg（必須）

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# 驗證安裝
ffmpeg -version
```

#### Python 套件（必須）

```bash
# 安裝到 ai_env 環境
conda activate ai_env
pip install pyyaml
```

#### 可選套件

```bash
# pydub（進階音訊處理）
pip install pydub

# librosa（音訊分析）
pip install librosa soundfile
```

### 檔案結構 (File Structure)

```
animation-ai-studio/
├── scripts/
│   └── automation/
│       └── scenarios/
│           └── audio_processor.py          # 主程式
├── configs/
│   └── automation/
│       └── audio_processor_example.yaml    # 配置範例
└── docs/
    └── automation/
        └── PHASE2_AUDIO_PROCESSOR.md       # 本文件
```

---

## 快速開始 (Quick Start)

### 1. 基本用法 - 從影片提取音訊

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input /path/to/video.mp4 \
  --output /path/to/audio.wav \
  --format wav
```

### 2. 格式轉換 - WAV 轉 MP3

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input /path/to/audio.wav \
  --output /path/to/audio.mp3 \
  --output-format mp3 \
  --bitrate 192k
```

### 3. 切割音訊片段

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation cut \
  --input /path/to/audio.wav \
  --output /path/to/segment.wav \
  --start-time 10.0 \
  --duration 30.0
```

### 4. 拼接多個音訊

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation concat \
  --inputs audio1.wav audio2.wav audio3.wav \
  --output merged.wav
```

### 5. 音量正規化

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input /path/to/audio.wav \
  --output /path/to/normalized.wav \
  --target-level -16dB
```

### 6. 檢測靜音

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input /path/to/audio.wav \
  --noise-threshold=-40 \
  --min-silence-duration 0.5
```

### 7. 移除靜音

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation remove_silence \
  --input /path/to/audio.wav \
  --output /path/to/no_silence.wav \
  --noise-threshold=-40 \
  --min-silence-duration 0.5
```

### 8. 提取 Metadata

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation metadata \
  --input /path/to/audio.wav
```

---

## 操作模式 (Operations)

### Extract (音訊提取)

從影片檔案中提取音訊軌道。

**必要參數**:
- `--input`: 輸入影片檔案路徑
- `--output`: 輸出音訊檔案路徑
- `--format`: 輸出格式 (wav/mp3/flac/aac/ogg)

**可選參數**:
- `--sample-rate`: 取樣率 (預設: 原始值)
- `--channels`: 聲道數 (預設: 原始值)
- `--bitrate`: 位元率（有損格式）

**範例**:

```bash
# 提取為高品質 WAV
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input movie.mp4 \
  --output audio.wav \
  --format wav \
  --sample-rate 48000 \
  --channels 2

# 提取為壓縮 MP3
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input movie.mp4 \
  --output audio.mp3 \
  --format mp3 \
  --bitrate 192k
```

### Convert (格式轉換)

在不同音訊格式間轉換。

**必要參數**:
- `--input`: 輸入音訊檔案路徑
- `--output`: 輸出音訊檔案路徑
- `--output-format`: 輸出格式

**可選參數**:
- `--sample-rate`: 目標取樣率
- `--channels`: 目標聲道數
- `--bitrate`: 目標位元率

**範例**:

```bash
# WAV 轉 MP3
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input audio.wav \
  --output audio.mp3 \
  --output-format mp3 \
  --bitrate 192k

# 立體聲轉單聲道
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input stereo.wav \
  --output mono.wav \
  --output-format wav \
  --channels 1

# 轉換為 FLAC 無損格式
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input audio.wav \
  --output audio.flac \
  --output-format flac
```

### Cut (音訊切割)

切割音訊片段。

**必要參數**:
- `--input`: 輸入音訊檔案路徑
- `--output`: 輸出音訊檔案路徑
- `--start-time`: 起始時間（秒）

**可選參數**:
- `--duration`: 片段時長（秒）
- `--end-time`: 結束時間（秒，與 duration 二選一）

**範例**:

```bash
# 使用起始時間 + 時長
python scripts/automation/scenarios/audio_processor.py \
  --operation cut \
  --input audio.wav \
  --output segment.wav \
  --start-time 10.0 \
  --duration 30.0

# 使用起始時間 + 結束時間
python scripts/automation/scenarios/audio_processor.py \
  --operation cut \
  --input audio.wav \
  --output segment.wav \
  --start-time 10.0 \
  --end-time 40.0

# 從開頭切割
python scripts/automation/scenarios/audio_processor.py \
  --operation cut \
  --input audio.wav \
  --output beginning.wav \
  --start-time 0.0 \
  --duration 5.0
```

### Concat (音訊拼接)

拼接多個音訊檔案。

**必要參數**:
- `--inputs`: 輸入音訊檔案列表（空格分隔）
- `--output`: 輸出音訊檔案路徑

**或使用**:
- `--input-list`: 包含輸入檔案列表的文字檔案路徑（每行一個檔案）

**範例**:

```bash
# 直接指定檔案
python scripts/automation/scenarios/audio_processor.py \
  --operation concat \
  --inputs segment1.wav segment2.wav segment3.wav \
  --output merged.wav

# 使用檔案列表
echo "segment1.wav" > filelist.txt
echo "segment2.wav" >> filelist.txt
echo "segment3.wav" >> filelist.txt

python scripts/automation/scenarios/audio_processor.py \
  --operation concat \
  --input-list filelist.txt \
  --output merged.wav
```

### Normalize (音量正規化)

正規化音訊音量至目標等級。

**必要參數**:
- `--input`: 輸入音訊檔案路徑
- `--output`: 輸出音訊檔案路徑

**可選參數**:
- `--target-level`: 目標音量等級（預設: -16dB）

**範例**:

```bash
# 使用預設目標等級
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input audio.wav \
  --output normalized.wav

# 自訂目標等級（音樂）
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input music.wav \
  --output normalized_music.wav \
  --target-level -12dB

# 自訂目標等級（語音）
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input speech.wav \
  --output normalized_speech.wav \
  --target-level -20dB
```

### Detect Silence (靜音檢測)

檢測音訊中的靜音片段。

**必要參數**:
- `--input`: 輸入音訊檔案路徑

**可選參數**:
- `--noise-threshold`: 噪音閾值（預設: -40）
- `--min-silence-duration`: 最小靜音時長（預設: 0.5 秒）

**範例**:

```bash
# 使用預設參數
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input audio.wav

# 自訂參數（更敏感）
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input audio.wav \
  --noise-threshold=-50 \
  --min-silence-duration 0.3

# 自訂參數（較不敏感）
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input audio.wav \
  --noise-threshold=-30 \
  --min-silence-duration 1.0
```

### Remove Silence (靜音移除)

移除音訊中的靜音片段。

**必要參數**:
- `--input`: 輸入音訊檔案路徑
- `--output`: 輸出音訊檔案路徑

**可選參數**:
- `--noise-threshold`: 噪音閾值（預設: -40）
- `--min-silence-duration`: 最小靜音時長（預設: 0.5 秒）

**範例**:

```bash
# 使用預設參數
python scripts/automation/scenarios/audio_processor.py \
  --operation remove_silence \
  --input audio.wav \
  --output no_silence.wav

# 自訂參數
python scripts/automation/scenarios/audio_processor.py \
  --operation remove_silence \
  --input podcast.wav \
  --output podcast_trimmed.wav \
  --noise-threshold=-45 \
  --min-silence-duration 0.8
```

### Metadata (Metadata 提取)

提取音訊檔案的 metadata。

**必要參數**:
- `--input`: 輸入音訊檔案路徑

**範例**:

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation metadata \
  --input audio.wav
```

**輸出範例**:

```
Audio Metadata:
  Duration: 120.50s
  Sample Rate: 48000 Hz
  Channels: 2
  Codec: pcm_s16le
  Bitrate: 1536000 bps
  File Size: 23.05 MB
  Format: wav
```

---

## 批次處理 (Batch Processing)

使用 YAML 配置檔案進行大規模批次處理。

### 建立配置檔案

參考 `configs/automation/audio_processor_example.yaml`：

```yaml
# 全域配置
threads: 32

audio:
  sample_rate: 48000
  channels: 2
  bitrate: 192k

silence:
  noise_threshold: -40
  min_duration: 0.5

# 批次操作
operations:
  # 提取音訊
  - operation: extract
    input: /path/to/video.mp4
    output: /path/to/audio.wav
    format: wav

  # 轉換格式
  - operation: convert
    input: /path/to/audio.wav
    output: /path/to/audio.mp3
    output_format: mp3
    bitrate: 192k

  # 切割片段
  - operation: cut
    input: /path/to/audio.wav
    output: /path/to/segment.wav
    start_time: 10.0
    duration: 30.0

  # 音量正規化
  - operation: normalize
    input: /path/to/audio.wav
    output: /path/to/normalized.wav
    target_level: -16dB

  # 移除靜音
  - operation: remove_silence
    input: /path/to/audio.wav
    output: /path/to/no_silence.wav
```

### 執行批次處理

```bash
python scripts/automation/scenarios/audio_processor.py \
  --operation batch \
  --batch-config configs/automation/my_audio_batch.yaml
```

### 批次處理工作流程範例

#### 範例 1: 影片音訊提取與格式轉換

```yaml
operations:
  # 1. 提取高品質 WAV
  - operation: extract
    input: /path/to/movie.mp4
    output: /tmp/audio/movie.wav
    format: wav
    sample_rate: 48000
    channels: 2

  # 2. 轉換為 MP3（網頁播放）
  - operation: convert
    input: /tmp/audio/movie.wav
    output: /path/to/output/movie.mp3
    output_format: mp3
    bitrate: 192k

  # 3. 轉換為 FLAC（歸檔）
  - operation: convert
    input: /tmp/audio/movie.wav
    output: /path/to/archive/movie.flac
    output_format: flac
```

#### 範例 2: 音訊切割與拼接

```yaml
operations:
  # 1. 切割多個片段
  - operation: cut
    input: /path/to/long_audio.wav
    output: /tmp/segments/segment_01.wav
    start_time: 0.0
    duration: 60.0

  - operation: cut
    input: /path/to/long_audio.wav
    output: /tmp/segments/segment_02.wav
    start_time: 60.0
    duration: 60.0

  - operation: cut
    input: /path/to/long_audio.wav
    output: /tmp/segments/segment_03.wav
    start_time: 120.0
    duration: 60.0

  # 2. 處理每個片段（正規化）
  - operation: normalize
    input: /tmp/segments/segment_01.wav
    output: /tmp/processed/segment_01.wav

  - operation: normalize
    input: /tmp/segments/segment_02.wav
    output: /tmp/processed/segment_02.wav

  - operation: normalize
    input: /tmp/segments/segment_03.wav
    output: /tmp/processed/segment_03.wav

  # 3. 重新拼接
  - operation: concat
    inputs:
      - /tmp/processed/segment_01.wav
      - /tmp/processed/segment_02.wav
      - /tmp/processed/segment_03.wav
    output: /path/to/output/processed_audio.wav
```

#### 範例 3: Podcast 後製流程

```yaml
operations:
  # 1. 提取音訊
  - operation: extract
    input: /path/to/podcast_recording.mp4
    output: /tmp/podcast/raw.wav
    format: wav

  # 2. 移除靜音
  - operation: remove_silence
    input: /tmp/podcast/raw.wav
    output: /tmp/podcast/trimmed.wav
    noise_threshold: -45
    min_silence_duration: 0.8

  # 3. 音量正規化
  - operation: normalize
    input: /tmp/podcast/trimmed.wav
    output: /tmp/podcast/normalized.wav
    target_level: -16dB

  # 4. 轉換為 MP3
  - operation: convert
    input: /tmp/podcast/normalized.wav
    output: /path/to/output/podcast_final.mp3
    output_format: mp3
    bitrate: 192k
```

---

## 音訊格式指南 (Audio Format Guide)

### WAV (Waveform Audio File Format)

**特性**:
- 無損格式 (Lossless)
- 編碼: pcm_s16le
- 品質: 最高
- 檔案大小: 大

**優點**:
- 無品質損失
- 廣泛兼容
- 適合編輯

**缺點**:
- 檔案體積大
- 不適合網路傳輸

**適用場景**:
- 專業音訊編輯
- 後製處理
- 音訊歸檔

**建議設定**:
```yaml
format: wav
sample_rate: 48000
channels: 2
```

### MP3 (MPEG Audio Layer III)

**特性**:
- 有損格式 (Lossy)
- 編碼: libmp3lame
- 品質: 視位元率而定
- 檔案大小: 小

**優點**:
- 檔案小巧
- 通用性強
- 串流友好

**缺點**:
- 有品質損失
- 不適合多次編輯

**適用場景**:
- 網頁播放
- 音樂串流
- 一般分發

**建議設定**:
```yaml
output_format: mp3
bitrate: 192k  # 或 128k (較小), 320k (較高品質)
```

**位元率選擇**:
- 128k: 低品質，檔案最小
- 192k: 平衡品質與大小（推薦）
- 320k: 高品質，接近 CD

### FLAC (Free Lossless Audio Codec)

**特性**:
- 無損格式 (Lossless)
- 編碼: flac
- 品質: 最高
- 檔案大小: 中等（比 WAV 小 30-50%）

**優點**:
- 無品質損失
- 比 WAV 小
- 支援 metadata

**缺點**:
- 不如 WAV 通用
- 編碼/解碼需更多 CPU

**適用場景**:
- 高品質歸檔
- 音樂收藏
- 無損分發

**建議設定**:
```yaml
output_format: flac
```

### AAC (Advanced Audio Coding)

**特性**:
- 有損格式 (Lossy)
- 編碼: aac
- 品質: 比 MP3 更好
- 檔案大小: 小

**優點**:
- 比 MP3 品質好
- 檔案較小
- 適合影片

**缺點**:
- 有品質損失
- 較少播放器支援

**適用場景**:
- 影片音訊
- 行動裝置
- Apple 生態系統

**建議設定**:
```yaml
output_format: aac
bitrate: 192k
```

### OGG Vorbis

**特性**:
- 有損格式 (Lossy)
- 編碼: libvorbis
- 品質: 好
- 檔案大小: 小

**優點**:
- 開源免費
- 品質好
- 適合遊戲

**缺點**:
- 較少裝置支援
- 不如 MP3 通用

**適用場景**:
- 遊戲音效
- 開源專案
- Linux 系統

**建議設定**:
```yaml
output_format: ogg
bitrate: 192k
```

### 格式選擇建議

| 場景 | 格式 | 原因 |
|------|------|------|
| 專業編輯 | WAV | 無損、通用 |
| 後製處理 | WAV | 無損、易編輯 |
| 高品質歸檔 | FLAC | 無損、檔案較小 |
| 網頁播放 | MP3 192k | 平衡品質與大小 |
| 音樂串流 | MP3 320k | 高品質有損 |
| 影片音訊 | AAC 192k | 適合影片容器 |
| 語音錄音 | MP3 128k | 檔案小、足夠清晰 |
| Podcast | MP3 192k | 平衡品質與大小 |
| 遊戲音效 | OGG | 開源、品質好 |

---

## 工作流程範例 (Workflow Examples)

### 工作流程 1: 完整音訊提取與轉換

**目標**: 從影片提取音訊，產生多種格式供不同用途使用

```bash
#!/bin/bash
# complete_audio_extraction.sh

VIDEO_INPUT="/path/to/movie.mp4"
OUTPUT_DIR="/path/to/output"

# 1. 提取高品質 WAV（主要版本）
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input "$VIDEO_INPUT" \
  --output "$OUTPUT_DIR/audio_master.wav" \
  --format wav \
  --sample-rate 48000 \
  --channels 2

# 2. 轉換為 MP3（網頁播放）
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input "$OUTPUT_DIR/audio_master.wav" \
  --output "$OUTPUT_DIR/audio_web.mp3" \
  --output-format mp3 \
  --bitrate 192k

# 3. 轉換為 FLAC（無損歸檔）
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input "$OUTPUT_DIR/audio_master.wav" \
  --output "$OUTPUT_DIR/audio_archive.flac" \
  --output-format flac

# 4. 轉換為單聲道 MP3（語音分析）
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input "$OUTPUT_DIR/audio_master.wav" \
  --output "$OUTPUT_DIR/audio_speech.mp3" \
  --output-format mp3 \
  --channels 1 \
  --bitrate 128k

echo "✅ 完成！產生了 4 種格式的音訊檔案"
```

### 工作流程 2: Podcast 自動化後製

**目標**: 自動化 Podcast 錄音的後製流程

```bash
#!/bin/bash
# podcast_postproduction.sh

INPUT_VIDEO="/path/to/recording.mp4"
OUTPUT_DIR="/path/to/output"
TEMP_DIR="/tmp/podcast_temp"

mkdir -p "$TEMP_DIR"

echo "🎙️ 開始 Podcast 後製流程..."

# 1. 提取音訊
echo "[1/5] 提取音訊..."
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input "$INPUT_VIDEO" \
  --output "$TEMP_DIR/raw.wav" \
  --format wav

# 2. 檢測靜音片段
echo "[2/5] 檢測靜音片段..."
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input "$TEMP_DIR/raw.wav" \
  --noise-threshold=-45 \
  --min-silence-duration 1.0

# 3. 移除靜音
echo "[3/5] 移除靜音..."
python scripts/automation/scenarios/audio_processor.py \
  --operation remove_silence \
  --input "$TEMP_DIR/raw.wav" \
  --output "$TEMP_DIR/trimmed.wav" \
  --noise-threshold=-45 \
  --min-silence-duration 1.0

# 4. 音量正規化
echo "[4/5] 正規化音量..."
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input "$TEMP_DIR/trimmed.wav" \
  --output "$TEMP_DIR/normalized.wav" \
  --target-level -16dB

# 5. 轉換為最終 MP3
echo "[5/5] 轉換為 MP3..."
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input "$TEMP_DIR/normalized.wav" \
  --output "$OUTPUT_DIR/podcast_final.mp3" \
  --output-format mp3 \
  --bitrate 192k

# 清理暫存檔案
rm -rf "$TEMP_DIR"

echo "✅ Podcast 後製完成！"
echo "📁 輸出檔案: $OUTPUT_DIR/podcast_final.mp3"
```

### 工作流程 3: 批次影片音訊提取

**目標**: 從多部影片中批次提取音訊

```bash
#!/bin/bash
# batch_audio_extraction.sh

VIDEO_DIR="/path/to/videos"
OUTPUT_DIR="/path/to/audio_output"

mkdir -p "$OUTPUT_DIR"

echo "🎬 開始批次音訊提取..."

# 遍歷所有影片檔案
for video in "$VIDEO_DIR"/*.mp4; do
    # 取得檔案名稱（不含路徑和副檔名）
    basename=$(basename "$video" .mp4)

    echo "處理: $basename"

    # 提取 WAV
    python scripts/automation/scenarios/audio_processor.py \
      --operation extract \
      --input "$video" \
      --output "$OUTPUT_DIR/${basename}.wav" \
      --format wav \
      --sample-rate 48000 \
      --channels 2

    # 轉換為 MP3
    python scripts/automation/scenarios/audio_processor.py \
      --operation convert \
      --input "$OUTPUT_DIR/${basename}.wav" \
      --output "$OUTPUT_DIR/${basename}.mp3" \
      --output-format mp3 \
      --bitrate 192k

    echo "✅ $basename 完成"
done

echo "🎉 批次處理完成！"
```

### 工作流程 4: 音訊切割與品質分級

**目標**: 將長音訊切割成片段並產生不同品質版本

```bash
#!/bin/bash
# audio_segmentation_quality.sh

INPUT_AUDIO="/path/to/long_audio.wav"
OUTPUT_DIR="/path/to/output"
SEGMENT_DURATION=60  # 每個片段 60 秒

mkdir -p "$OUTPUT_DIR/segments/high"
mkdir -p "$OUTPUT_DIR/segments/medium"
mkdir -p "$OUTPUT_DIR/segments/low"

echo "✂️ 開始音訊切割與品質分級..."

# 1. 取得音訊總時長
duration=$(python scripts/automation/scenarios/audio_processor.py \
  --operation metadata \
  --input "$INPUT_AUDIO" \
  | grep "Duration" | awk '{print $2}' | sed 's/s//')

# 計算片段數量
num_segments=$(echo "($duration + $SEGMENT_DURATION - 1) / $SEGMENT_DURATION" | bc)

echo "📊 音訊總時長: ${duration}s"
echo "📊 將切割為 $num_segments 個片段"

# 2. 切割並產生多品質版本
for i in $(seq 0 $((num_segments - 1))); do
    start_time=$(echo "$i * $SEGMENT_DURATION" | bc)
    segment_num=$(printf "%03d" $i)

    echo "[Segment $segment_num] 起始: ${start_time}s"

    # 切割 WAV 片段
    python scripts/automation/scenarios/audio_processor.py \
      --operation cut \
      --input "$INPUT_AUDIO" \
      --output "$OUTPUT_DIR/segments/segment_${segment_num}.wav" \
      --start-time $start_time \
      --duration $SEGMENT_DURATION

    # 產生高品質 MP3 (320k)
    python scripts/automation/scenarios/audio_processor.py \
      --operation convert \
      --input "$OUTPUT_DIR/segments/segment_${segment_num}.wav" \
      --output "$OUTPUT_DIR/segments/high/segment_${segment_num}.mp3" \
      --output-format mp3 \
      --bitrate 320k

    # 產生中等品質 MP3 (192k)
    python scripts/automation/scenarios/audio_processor.py \
      --operation convert \
      --input "$OUTPUT_DIR/segments/segment_${segment_num}.wav" \
      --output "$OUTPUT_DIR/segments/medium/segment_${segment_num}.mp3" \
      --output-format mp3 \
      --bitrate 192k

    # 產生低品質 MP3 (128k)
    python scripts/automation/scenarios/audio_processor.py \
      --operation convert \
      --input "$OUTPUT_DIR/segments/segment_${segment_num}.wav" \
      --output "$OUTPUT_DIR/segments/low/segment_${segment_num}.mp3" \
      --output-format mp3 \
      --bitrate 128k
done

echo "✅ 切割與品質分級完成！"
echo "📁 WAV 片段: $OUTPUT_DIR/segments/"
echo "📁 高品質 (320k): $OUTPUT_DIR/segments/high/"
echo "📁 中等品質 (192k): $OUTPUT_DIR/segments/medium/"
echo "📁 低品質 (128k): $OUTPUT_DIR/segments/low/"
```

---

## 參數詳解 (Parameter Details)

### 音訊品質參數

#### Sample Rate (取樣率)

**定義**: 每秒採樣的次數，單位為 Hz

**常用值**:
- **44100 Hz**: CD 音質標準，適合音樂
- **48000 Hz**: 專業音訊/影片標準（推薦）
- **96000 Hz**: 高解析度音訊，用於專業製作

**選擇建議**:
- 一般用途: 48000 Hz
- 音樂製作: 48000 Hz 或更高
- 語音: 44100 Hz 或 48000 Hz
- 網頁音訊: 44100 Hz

#### Channels (聲道數)

**定義**: 音訊聲道數量

**選項**:
- **1 (Mono)**: 單聲道
  - 檔案大小小一半
  - 適合語音、podcast
  - 無空間感

- **2 (Stereo)**: 立體聲
  - 有左右聲道
  - 適合音樂、影片
  - 有空間感

**選擇建議**:
- 語音錄音: 單聲道
- 音樂: 立體聲
- 影片音訊: 立體聲
- Podcast: 視內容而定

#### Bitrate (位元率)

**定義**: 每秒傳輸的資料量，單位為 kbps

**常用值**（MP3）:
- **128k**: 低品質
  - 檔案最小
  - 適合語音
  - 音樂品質不佳

- **192k**: 中等品質（推薦）
  - 平衡品質與大小
  - 適合大多數用途
  - 一般聽眾難以分辨

- **320k**: 高品質
  - 接近 CD 品質
  - 適合音樂收藏
  - 檔案較大

**選擇建議**:
- 語音/Podcast: 128k
- 一般音樂: 192k
- 高品質音樂: 320k
- 網頁串流: 192k

### 靜音檢測參數

#### Noise Threshold (噪音閾值)

**定義**: 判定為靜音的音量閾值，單位為 dB

**常用值**:
- **-30 dB**: 較不敏感
  - 只檢測非常安靜的部分
  - 可能遺漏某些靜音
  - 保留背景音

- **-40 dB**: 標準（推薦）
  - 平衡敏感度
  - 適合大多數場景
  - 不會過度敏感

- **-50 dB**: 較敏感
  - 檢測更多靜音區域
  - 可能誤判背景音為靜音
  - 適合非常乾淨的錄音

**選擇建議**:
- 乾淨錄音室錄音: -50 dB
- 一般錄音: -40 dB
- 有背景音的錄音: -30 dB

#### Minimum Silence Duration (最小靜音時長)

**定義**: 判定為靜音片段的最小時長，單位為秒

**常用值**:
- **0.3 秒**: 較短
  - 檢測更多靜音
  - 可能過度切割
  - 適合語音暫停

- **0.5 秒**: 標準（推薦）
  - 平衡檢測
  - 適合大多數場景
  - 避免過度切割

- **1.0 秒**: 較長
  - 只檢測明顯靜音
  - 保留短暫停頓
  - 適合 Podcast

**選擇建議**:
- 語音轉錄: 0.3 秒
- Podcast: 0.5-1.0 秒
- 音樂: 1.0 秒或更長

### 音量正規化參數

#### Target Level (目標音量等級)

**定義**: 正規化後的目標響度，單位為 dB

**常用值**:
- **-20 dB**: 較小聲
  - 適合語音
  - 避免削波
  - 保留動態範圍

- **-16 dB**: 標準（推薦）
  - 符合廣播標準
  - 適合大多數內容
  - 平衡響度與品質

- **-12 dB**: 較大聲
  - 適合音樂
  - 更具衝擊力
  - 可能損失動態

**選擇建議**:
- 語音/Podcast: -18 dB 至 -16 dB
- 音樂: -14 dB 至 -12 dB
- 廣播: -16 dB
- 影片音訊: -16 dB

---

## 效能與最佳化 (Performance & Optimization)

### CPU 使用最佳化

**32 執行緒設定**:

```bash
# 預設使用 32 執行緒
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input video.mp4 \
  --output audio.wav \
  --format wav
  # 內部會設定: --threads 32
```

**自訂執行緒數量**:

```bash
# 使用 16 執行緒（如果系統有其他任務）
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input video.mp4 \
  --output audio.wav \
  --format wav \
  --threads 16
```

### 記憶體使用最佳化

**監控記憶體**:

Audio Processor 整合了 Phase 1 記憶體監控：

- **70% 記憶體**: 警告 (Warning)
- **80% 記憶體**: 嚴重 (Critical)
- **85% 記憶體**: 緊急 (Emergency) - 暫停處理

**降低記憶體使用**:

1. **批次處理時減少並行操作**:
   - 逐個處理檔案而非同時處理多個
   - 在批次配置中分階段執行

2. **處理大檔案時使用串流**:
   - FFmpeg 自動使用串流處理
   - 不需一次載入整個檔案

3. **清理暫存檔案**:
   ```bash
   # 處理完成後刪除暫存 WAV
   python scripts/automation/scenarios/audio_processor.py \
     --operation convert \
     --input audio.wav \
     --output audio.mp3 \
     --output-format mp3

   rm audio.wav  # 清理暫存檔案
   ```

### 效能基準測試 (Performance Benchmarks)

**測試環境**:
- CPU: 32-core processor
- RAM: 64GB
- 儲存: SSD

**測試結果**:

| 操作 | 檔案大小 | 時長 | 處理時間 | 速度比 |
|------|---------|------|---------|--------|
| 提取音訊 (WAV) | 3.9MB (影片) | 10s | 0.12s | 83x |
| 轉換 WAV→MP3 | 1.9MB (WAV) | 10s | 0.15s | 67x |
| 切割音訊 | 1.9MB (WAV) | 10s → 5s | 0.04s | 250x |
| 拼接音訊 (3個檔案) | 5.7MB (total) | 30s | 0.08s | 375x |
| 正規化音量 | 1.9MB (WAV) | 10s | 2.1s | 4.8x |
| 檢測靜音 | 1.9MB (WAV) | 10s | 0.09s | 111x |
| 移除靜音 | 1.9MB (WAV) | 10s | 0.11s | 91x |

**說明**:
- 大多數操作都達到實時速度的數十倍至數百倍
- 音量正規化較慢因為需要分析整個音訊並重新編碼
- 使用 codec copy 的操作（如切割）最快

### 最佳化建議

#### 1. 使用適當的格式

**快速操作**:
- 切割、拼接: 使用 WAV（可用 codec copy）
- 格式轉換: 目標格式視用途而定

**品質優先**:
- 後製編輯: 使用 WAV
- 最終輸出: 使用 MP3/AAC

#### 2. 批次處理策略

**逐個處理 vs. 並行處理**:

```yaml
# 推薦：逐個處理（記憶體安全）
operations:
  - operation: extract
    input: video1.mp4
    output: audio1.wav
  - operation: extract
    input: video2.mp4
    output: audio2.wav

# 避免：同時處理多個大檔案
# （可能導致記憶體不足）
```

#### 3. 儲存空間管理

**清理策略**:

```bash
# 處理流程：video → WAV → MP3
# 保留 MP3，刪除中間 WAV

python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input video.mp4 \
  --output /tmp/temp.wav \
  --format wav

python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input /tmp/temp.wav \
  --output final.mp3 \
  --output-format mp3

rm /tmp/temp.wav  # 刪除暫存檔案
```

#### 4. 長時間批次處理

**使用 tmux/screen**:

```bash
# 啟動 tmux session
tmux new -s audio_batch

# 執行批次處理
python scripts/automation/scenarios/audio_processor.py \
  --operation batch \
  --batch-config large_batch.yaml

# Detach: Ctrl+B, 然後 D
# Reattach: tmux attach -t audio_batch
```

---

## 疑難排解 (Troubleshooting)

### 常見問題與解決方案

#### 問題 1: FFmpeg 未安裝

**錯誤訊息**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**解決方案**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# 驗證安裝
ffmpeg -version
which ffmpeg
```

#### 問題 2: 音訊品質不佳

**症狀**: 轉換後的音訊有明顯失真或雜音

**可能原因**:
1. 位元率太低
2. 多次有損轉換
3. 來源音訊品質不佳

**解決方案**:
```bash
# 提高位元率
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input audio.wav \
  --output audio.mp3 \
  --output-format mp3 \
  --bitrate 320k  # 使用最高品質

# 避免多次有損轉換
# 不好: WAV → MP3 → AAC (兩次有損)
# 好: WAV → MP3 (一次有損)
#     WAV → AAC (一次有損，分別進行)
```

#### 問題 3: 檔案太大

**症狀**: 產生的音訊檔案佔用過多空間

**解決方案**:

```bash
# 使用有損格式
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input large_audio.wav \
  --output compressed.mp3 \
  --output-format mp3 \
  --bitrate 192k

# 降低取樣率（如果可接受）
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input audio.wav \
  --output audio_low_sr.mp3 \
  --output-format mp3 \
  --sample-rate 44100 \
  --bitrate 192k

# 轉換為單聲道（語音）
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input stereo.wav \
  --output mono.mp3 \
  --output-format mp3 \
  --channels 1 \
  --bitrate 128k
```

#### 問題 4: 靜音檢測不準確

**症狀 A**: 檢測到太多靜音（誤判背景音為靜音）

**解決方案**:
```bash
# 降低敏感度（提高閾值）
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input audio.wav \
  --noise-threshold=-30 \  # 改為 -30（原本 -40）
  --min-silence-duration 1.0  # 增加最小時長
```

**症狀 B**: 檢測不到靜音（遺漏明顯的靜音片段）

**解決方案**:
```bash
# 提高敏感度（降低閾值）
python scripts/automation/scenarios/audio_processor.py \
  --operation detect_silence \
  --input audio.wav \
  --noise-threshold=-50 \  # 改為 -50（原本 -40）
  --min-silence-duration 0.3  # 減少最小時長
```

#### 問題 5: 記憶體警告

**錯誤訊息**:
```
WARNING - Memory usage high: 75.3%
CRITICAL - Memory usage critical: 82.1%
```

**解決方案**:

1. **等待其他程序完成**:
   ```bash
   # 檢查記憶體使用
   free -h

   # 檢查佔用記憶體的程序
   top -o %MEM
   ```

2. **關閉不必要的程序**:
   ```bash
   # 關閉佔用記憶體的應用程式
   # 或暫停其他批次處理任務
   ```

3. **分批處理**:
   ```yaml
   # 將大批次拆分為多個小批次
   # batch_part1.yaml
   operations:
     - operation: extract
       input: video1.mp4
       output: audio1.wav

   # batch_part2.yaml
   operations:
     - operation: extract
       input: video2.mp4
       output: audio2.wav
   ```

#### 問題 6: 音訊切割位置不精確

**症狀**: 切割的音訊不是從預期的位置開始/結束

**可能原因**: 影片/音訊檔案的 keyframe 問題

**解決方案**:
```bash
# 使用精確切割（重新編碼）
python scripts/automation/scenarios/audio_processor.py \
  --operation cut \
  --input audio.wav \
  --output segment.wav \
  --start-time 10.5 \
  --duration 5.0
  # WAV 格式天然支援精確切割

# 如果是 MP3/AAC，考慮先轉換為 WAV
python scripts/automation/scenarios/audio_processor.py \
  --operation convert \
  --input audio.mp3 \
  --output audio.wav \
  --output-format wav

# 然後進行精確切割
python scripts/automation/scenarios/audio_processor.py \
  --operation cut \
  --input audio.wav \
  --output segment.wav \
  --start-time 10.5 \
  --duration 5.0
```

#### 問題 7: 音量正規化後有削波

**症狀**: 正規化後音訊有破音或失真

**可能原因**: 目標音量過高

**解決方案**:
```bash
# 降低目標音量
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input audio.wav \
  --output normalized.wav \
  --target-level -18dB  # 改為 -18 dB（原本 -16 dB）

# 或使用更保守的 -20 dB
python scripts/automation/scenarios/audio_processor.py \
  --operation normalize \
  --input audio.wav \
  --output normalized.wav \
  --target-level -20dB
```

#### 問題 8: 批次處理中斷

**症狀**: 批次處理執行到一半停止

**可能原因**:
1. 某個檔案損壞
2. 記憶體不足
3. 磁碟空間不足

**解決方案**:

1. **檢查日誌**:
   ```bash
   # 查看最後的錯誤訊息
   tail -50 /path/to/logfile.log
   ```

2. **逐個測試檔案**:
   ```bash
   # 測試可疑的檔案
   python scripts/automation/scenarios/audio_processor.py \
     --operation metadata \
     --input suspicious_file.mp4
   ```

3. **跳過問題檔案**:
   ```yaml
   # 在批次配置中移除或註解掉問題檔案
   operations:
     - operation: extract
       input: working_file.mp4
       output: audio1.wav
     # - operation: extract
     #   input: problematic_file.mp4  # 暫時跳過
     #   output: audio2.wav
   ```

#### 問題 9: 權限錯誤

**錯誤訊息**:
```
PermissionError: [Errno 13] Permission denied: '/path/to/output.wav'
```

**解決方案**:
```bash
# 檢查輸出目錄權限
ls -ld /path/to/output_dir

# 建立輸出目錄（如果不存在）
mkdir -p /path/to/output_dir

# 確保有寫入權限
chmod u+w /path/to/output_dir
```

### 除錯技巧

#### 1. 啟用詳細日誌

```bash
# Audio Processor 預設會輸出詳細日誌
python scripts/automation/scenarios/audio_processor.py \
  --operation extract \
  --input video.mp4 \
  --output audio.wav \
  --format wav \
  2>&1 | tee audio_processing.log
```

#### 2. 測試單個操作

```bash
# 先測試單個檔案
python scripts/automation/scenarios/audio_processor.py \
  --operation metadata \
  --input test_audio.wav

# 確認可行後再進行批次處理
```

#### 3. 驗證 FFmpeg 命令

```bash
# Audio Processor 會輸出實際執行的 FFmpeg 命令
# 你可以複製該命令直接執行來測試

# 範例輸出:
# Running audio extraction: ffmpeg -i input.mp4 -vn -threads 32 -acodec pcm_s16le -y output.wav

# 直接執行測試:
ffmpeg -i input.mp4 -vn -threads 32 -acodec pcm_s16le -y output.wav
```

#### 4. 檢查系統資源

```bash
# 檢查 CPU 使用率
top

# 檢查記憶體使用
free -h

# 檢查磁碟空間
df -h

# 檢查 I/O 使用
iostat -x 1
```

---

## API 參考 (API Reference)

### AudioProcessor 類別

```python
from scripts.automation.scenarios.audio_processor import AudioProcessor
from scripts.core.safety import MemoryMonitor

# 初始化
memory_monitor = MemoryMonitor(
    warning_threshold=0.70,
    critical_threshold=0.80,
    emergency_threshold=0.85
)

processor = AudioProcessor(
    threads=32,
    memory_monitor=memory_monitor
)
```

### 方法 (Methods)

#### extract_audio()

從影片提取音訊。

```python
def extract_audio(
    self,
    video_path: str,
    output_path: str,
    format: str = 'wav',
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    bitrate: Optional[str] = None
) -> bool:
    """
    Extract audio from video file.
    從影片檔案提取音訊。

    Args:
        video_path: 輸入影片檔案路徑
        output_path: 輸出音訊檔案路徑
        format: 輸出格式 (wav/mp3/flac/aac/ogg)
        sample_rate: 取樣率 (Hz)
        channels: 聲道數 (1=mono, 2=stereo)
        bitrate: 位元率 (如 '192k')

    Returns:
        bool: 成功回傳 True，失敗回傳 False
    """
```

**範例**:
```python
processor.extract_audio(
    video_path="/path/to/video.mp4",
    output_path="/path/to/audio.wav",
    format='wav',
    sample_rate=48000,
    channels=2
)
```

#### convert_format()

轉換音訊格式。

```python
def convert_format(
    self,
    input_path: str,
    output_path: str,
    output_format: str,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    bitrate: Optional[str] = None
) -> bool:
    """
    Convert audio format.
    轉換音訊格式。

    Args:
        input_path: 輸入音訊檔案路徑
        output_path: 輸出音訊檔案路徑
        output_format: 輸出格式 (wav/mp3/flac/aac/ogg)
        sample_rate: 目標取樣率 (Hz)
        channels: 目標聲道數
        bitrate: 目標位元率 (如 '192k')

    Returns:
        bool: 成功回傳 True，失敗回傳 False
    """
```

**範例**:
```python
processor.convert_format(
    input_path="/path/to/audio.wav",
    output_path="/path/to/audio.mp3",
    output_format='mp3',
    bitrate='192k'
)
```

#### cut_audio()

切割音訊片段。

```python
def cut_audio(
    self,
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: Optional[float] = None,
    duration: Optional[float] = None
) -> bool:
    """
    Cut audio segment.
    切割音訊片段。

    Args:
        input_path: 輸入音訊檔案路徑
        output_path: 輸出音訊檔案路徑
        start_time: 起始時間（秒）
        end_time: 結束時間（秒，與 duration 二選一）
        duration: 片段時長（秒，與 end_time 二選一）

    Returns:
        bool: 成功回傳 True，失敗回傳 False
    """
```

**範例**:
```python
# 使用 start_time + duration
processor.cut_audio(
    input_path="/path/to/audio.wav",
    output_path="/path/to/segment.wav",
    start_time=10.0,
    duration=30.0
)

# 使用 start_time + end_time
processor.cut_audio(
    input_path="/path/to/audio.wav",
    output_path="/path/to/segment.wav",
    start_time=10.0,
    end_time=40.0
)
```

#### concatenate_audio()

拼接多個音訊檔案。

```python
def concatenate_audio(
    self,
    input_paths: List[str],
    output_path: str
) -> bool:
    """
    Concatenate multiple audio files.
    拼接多個音訊檔案。

    Args:
        input_paths: 輸入音訊檔案列表
        output_path: 輸出音訊檔案路徑

    Returns:
        bool: 成功回傳 True，失敗回傳 False
    """
```

**範例**:
```python
processor.concatenate_audio(
    input_paths=[
        "/path/to/segment1.wav",
        "/path/to/segment2.wav",
        "/path/to/segment3.wav"
    ],
    output_path="/path/to/merged.wav"
)
```

#### normalize_volume()

正規化音訊音量。

```python
def normalize_volume(
    self,
    input_path: str,
    output_path: str,
    target_level: str = '-16dB'
) -> bool:
    """
    Normalize audio volume.
    正規化音訊音量。

    Args:
        input_path: 輸入音訊檔案路徑
        output_path: 輸出音訊檔案路徑
        target_level: 目標音量等級（如 '-16dB'）

    Returns:
        bool: 成功回傳 True，失敗回傳 False
    """
```

**範例**:
```python
processor.normalize_volume(
    input_path="/path/to/audio.wav",
    output_path="/path/to/normalized.wav",
    target_level='-16dB'
)
```

#### detect_silence()

檢測靜音片段。

```python
def detect_silence(
    self,
    input_path: str,
    noise_threshold: int = -40,
    min_silence_duration: float = 0.5
) -> List[SilenceSegment]:
    """
    Detect silence segments in audio.
    檢測音訊中的靜音片段。

    Args:
        input_path: 輸入音訊檔案路徑
        noise_threshold: 噪音閾值（dB）
        min_silence_duration: 最小靜音時長（秒）

    Returns:
        List[SilenceSegment]: 靜音片段列表
    """
```

**範例**:
```python
silence_segments = processor.detect_silence(
    input_path="/path/to/audio.wav",
    noise_threshold=-40,
    min_silence_duration=0.5
)

for seg in silence_segments:
    print(f"Silence: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")
```

#### remove_silence()

移除靜音片段。

```python
def remove_silence(
    self,
    input_path: str,
    output_path: str,
    noise_threshold: int = -40,
    min_silence_duration: float = 0.5
) -> bool:
    """
    Remove silence from audio.
    從音訊移除靜音。

    Args:
        input_path: 輸入音訊檔案路徑
        output_path: 輸出音訊檔案路徑
        noise_threshold: 噪音閾值（dB）
        min_silence_duration: 最小靜音時長（秒）

    Returns:
        bool: 成功回傳 True，失敗回傳 False
    """
```

**範例**:
```python
processor.remove_silence(
    input_path="/path/to/audio.wav",
    output_path="/path/to/no_silence.wav",
    noise_threshold=-40,
    min_silence_duration=0.5
)
```

#### extract_metadata()

提取音訊 metadata。

```python
def extract_metadata(
    self,
    input_path: str
) -> Optional[AudioMetadata]:
    """
    Extract audio metadata.
    提取音訊 metadata。

    Args:
        input_path: 輸入音訊檔案路徑

    Returns:
        Optional[AudioMetadata]: Metadata 物件，失敗回傳 None
    """
```

**範例**:
```python
metadata = processor.extract_metadata("/path/to/audio.wav")

if metadata:
    print(f"Duration: {metadata.duration_seconds}s")
    print(f"Sample Rate: {metadata.sample_rate} Hz")
    print(f"Channels: {metadata.channels}")
    print(f"Codec: {metadata.codec}")
    print(f"Bitrate: {metadata.bitrate} bps")
    print(f"File Size: {metadata.file_size_bytes} bytes")
    print(f"Format: {metadata.format}")
```

### 資料類別 (Data Classes)

#### AudioMetadata

```python
@dataclass
class AudioMetadata:
    """Audio file metadata (音訊檔案 Metadata)"""
    duration_seconds: float      # 時長（秒）
    sample_rate: int             # 取樣率（Hz）
    channels: int                # 聲道數
    codec: str                   # 編碼格式
    bitrate: int                 # 位元率（bps）
    file_size_bytes: int         # 檔案大小（bytes）
    format: str                  # 格式
```

#### SilenceSegment

```python
@dataclass
class SilenceSegment:
    """Silence segment information (靜音片段資訊)"""
    start_time: float            # 起始時間（秒）
    end_time: float              # 結束時間（秒）
    duration: float              # 時長（秒）
```

---

## 附錄 (Appendix)

### 支援的音訊格式

| 格式 | 副檔名 | 類型 | FFmpeg 編碼器 |
|------|--------|------|--------------|
| WAV | .wav | 無損 | pcm_s16le |
| MP3 | .mp3 | 有損 | libmp3lame |
| FLAC | .flac | 無損 | flac |
| AAC | .aac, .m4a | 有損 | aac |
| OGG | .ogg | 有損 | libvorbis |

### 常用 FFmpeg 命令參考

```bash
# 提取音訊（WAV）
ffmpeg -i input.mp4 -vn -acodec pcm_s16le output.wav

# 轉換格式（MP3）
ffmpeg -i input.wav -acodec libmp3lame -b:a 192k output.mp3

# 切割音訊
ffmpeg -i input.wav -ss 10.0 -t 30.0 -acodec copy output.wav

# 拼接音訊
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.wav

# 正規化音量
ffmpeg -i input.wav -af loudnorm=I=-16 output.wav

# 檢測靜音
ffmpeg -i input.wav -af silencedetect=n=-40dB:d=0.5 -f null -

# 移除靜音
ffmpeg -i input.wav -af silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB output.wav
```

### 相關資源

**文件**:
- [PHASE1_GUIDE.md](./PHASE1_GUIDE.md) - Phase 1 核心基礎設施
- [PHASE2_VIDEO_PROCESSOR.md](./PHASE2_VIDEO_PROCESSOR.md) - 影片處理器
- [PHASE2_SUBTITLE_AUTOMATION.md](./PHASE2_SUBTITLE_AUTOMATION.md) - 字幕自動化
- [SAFETY_INFRASTRUCTURE.md](./SAFETY_INFRASTRUCTURE.md) - 安全基礎設施

**外部資源**:
- [FFmpeg 官方文件](https://ffmpeg.org/documentation.html)
- [FFmpeg Audio Filters](https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters)
- [LAME MP3 Encoder](https://lame.sourceforge.io/)

---

## 版本歷史 (Version History)

### v1.0.0 (2025-12-02)

**首次發布**:
- ✅ 完整的音訊處理功能
- ✅ 支援 8 種操作模式
- ✅ 批次處理支援
- ✅ 32 執行緒最佳化
- ✅ 記憶體安全整合
- ✅ 中英雙語文件

**功能**:
- 音訊提取 (Extract)
- 格式轉換 (Convert)
- 音訊切割 (Cut)
- 音訊拼接 (Concat)
- 音量正規化 (Normalize)
- 靜音檢測 (Detect Silence)
- 靜音移除 (Remove Silence)
- Metadata 提取 (Metadata)

**效能**:
- 提取音訊: 83x 實時速度
- 格式轉換: 67x 實時速度
- 音訊切割: 250x 實時速度

---

## 授權與貢獻 (License & Contributing)

**授權 (License)**: MIT License

**貢獻 (Contributing)**:
歡迎提交 Issue 和 Pull Request！

**聯絡 (Contact)**:
Animation AI Studio Team

---

**文件版本**: v1.0.0
**最後更新**: 2025-12-02 19:15
**下次審查**: 2025-12-09
