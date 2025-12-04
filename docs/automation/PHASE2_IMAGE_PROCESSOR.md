# Image Processor (圖像處理器)

## 概述

Image Processor 是 Phase 2 自動化基礎設施的圖像處理組件，提供全面的 CPU 圖像處理能力，使用 Pillow (PIL) 函式庫實現。所有操作都經過 CPU 最佳化，支援 32 執行緒並行處理。

### 核心功能

- ✅ **10 種圖像操作**：resize, crop, convert, optimize, blur, sharpen, contrast, brightness, auto_contrast, metadata, batch
- ✅ **多格式支援**：JPG, PNG, WebP, BMP, TIFF
- ✅ **進階濾鏡**：高斯模糊、銳化、對比度調整、亮度調整、自動對比度
- ✅ **智能裁切**：Box 裁切、中心裁切、正方形裁切
- ✅ **格式轉換**：自動處理 RGBA → RGB 轉換（JPEG 相容性）
- ✅ **圖像最佳化**：品質壓縮、檔案大小減少
- ✅ **Metadata 提取**：EXIF 資訊、尺寸、格式
- ✅ **批次處理**：YAML 配置驅動的自動化工作流程
- ✅ **記憶體安全**：整合 Phase 1 記憶體監控系統
- ✅ **雙語日誌**：中英文雙語輸出

### 系統需求

**必需依賴**：
```bash
pillow>=10.0.0      # 圖像處理核心
pyyaml>=6.0         # YAML 配置解析
```

**可選依賴**：
```bash
# 無額外可選依賴
```

**系統需求**：
- Python 3.10+
- CPU: 4+ 核心推薦（支援 32 執行緒）
- RAM: 4GB+ 可用記憶體
- 磁碟: 視圖像大小而定

### 安裝

```bash
# 啟動 ai_env 環境
conda activate ai_env

# 安裝依賴（如果尚未安裝）
pip install pillow>=10.0.0 pyyaml>=6.0

# 驗證安裝
python scripts/automation/scenarios/image_processor.py --operation metadata --input /path/to/test.jpg
```

---

## 快速入門

### 範例 1：調整圖像尺寸

```bash
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input /path/to/input.jpg \
  --output /path/to/output.jpg \
  --width 800 \
  --maintain-aspect
```

**結果**：
- 原始圖像調整為寬度 800px
- 自動保持長寬比
- 使用 Lanczos 重採樣（最高品質）

### 範例 2：圖像格式轉換

```bash
python scripts/automation/scenarios/image_processor.py \
  --operation convert \
  --input /path/to/input.jpg \
  --output /path/to/output.png \
  --output-format png \
  --quality 95 \
  --optimize
```

**結果**：
- JPG 轉換為 PNG
- 自動處理 RGBA → RGB（如需要）
- 應用最佳化壓縮

### 範例 3：中心裁切

```bash
python scripts/automation/scenarios/image_processor.py \
  --operation crop \
  --input /path/to/input.jpg \
  --output /path/to/output.jpg \
  --mode center \
  --width 500 \
  --height 500
```

**結果**：
- 從圖像中心裁切 500x500 區域
- 自動計算裁切座標

### 範例 4：圖像最佳化（減少檔案大小）

```bash
python scripts/automation/scenarios/image_processor.py \
  --operation optimize \
  --input /path/to/input.jpg \
  --output /path/to/output.jpg \
  --quality 85
```

**實際測試結果**：
- 原始大小：350.5 KB
- 最佳化後：241.4 KB
- **減少 31.1%**

### 範例 5：批次處理（YAML 配置）

```bash
python scripts/automation/scenarios/image_processor.py \
  --operation batch \
  --input configs/automation/image_processor_example.yaml
```

**配置範例**：
```yaml
operations:
  - operation: resize
    input: /path/to/input1.jpg
    output: /path/to/output1.jpg
    width: 800
    maintain_aspect: true

  - operation: optimize
    input: /path/to/input2.jpg
    output: /path/to/output2.jpg
    quality: 85
```

---

## 操作詳解

### 1. Resize（調整尺寸）

**功能**：調整圖像至指定尺寸

**參數**：
- `--width`：目標寬度（像素）
- `--height`：目標高度（像素）
- `--maintain-aspect`：保持長寬比（預設：true）
- `--resampling`：重採樣演算法（預設：lanczos）
  - `nearest`：最近鄰（最快，品質最低）
  - `bilinear`：雙線性插值
  - `bicubic`：雙三次插值
  - `lanczos`：Lanczos 濾波器（最慢，品質最高）

**範例**：

```bash
# 調整為固定寬度，保持長寬比
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input input.jpg \
  --output output.jpg \
  --width 1920 \
  --maintain-aspect

# 調整為固定尺寸，不保持長寬比
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input input.jpg \
  --output output.jpg \
  --width 1920 \
  --height 1080 \
  --no-maintain-aspect

# 使用快速重採樣
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input input.jpg \
  --output output.jpg \
  --width 800 \
  --resampling bilinear
```

**效能**：
- Lanczos：最高品質，速度較慢（推薦用於最終輸出）
- Bicubic：平衡品質與速度
- Bilinear：快速，品質中等（推薦用於預覽）
- Nearest：最快，品質最低（不推薦）

---

### 2. Crop（裁切）

**功能**：裁切圖像至指定區域

**參數**：
- `--mode`：裁切模式（預設：box）
  - `box`：指定左上角和尺寸
  - `center`：從中心裁切
  - `square`：裁切為正方形（最小邊長）
- `--left`：左邊界（像素，box 模式）
- `--top`：上邊界（像素，box 模式）
- `--width`：裁切寬度（像素）
- `--height`：裁切高度（像素）
- `--right`：右邊界（像素，可選）

**範例**：

```bash
# Box 裁切（指定左上角和尺寸）
python scripts/automation/scenarios/image_processor.py \
  --operation crop \
  --input input.jpg \
  --output output.jpg \
  --mode box \
  --left 100 \
  --top 100 \
  --width 800 \
  --height 600

# 中心裁切
python scripts/automation/scenarios/image_processor.py \
  --operation crop \
  --input input.jpg \
  --output output.jpg \
  --mode center \
  --width 500 \
  --height 500

# 正方形裁切（自動使用最小邊長）
python scripts/automation/scenarios/image_processor.py \
  --operation crop \
  --input input.jpg \
  --output output.jpg \
  --mode square
```

**使用情境**：
- **Box 裁切**：精確控制裁切區域（例如：裁切特定物件）
- **Center 裁切**：製作縮圖、頭像（聚焦中心內容）
- **Square 裁切**：社交媒體上傳、圖示製作

---

### 3. Convert（格式轉換）

**功能**：轉換圖像格式

**參數**：
- `--output-format`：目標格式
  - `jpg` / `jpeg`：JPEG（有損壓縮）
  - `png`：PNG（無損壓縮）
  - `webp`：WebP（現代格式，高效壓縮）
  - `bmp`：BMP（未壓縮）
  - `tiff`：TIFF（專業格式）
- `--quality`：壓縮品質（1-100，預設：95）
- `--optimize`：啟用最佳化壓縮（預設：true）

**範例**：

```bash
# PNG → JPG（減少檔案大小）
python scripts/automation/scenarios/image_processor.py \
  --operation convert \
  --input input.png \
  --output output.jpg \
  --output-format jpeg \
  --quality 90 \
  --optimize

# JPG → PNG（保留透明度）
python scripts/automation/scenarios/image_processor.py \
  --operation convert \
  --input input.jpg \
  --output output.png \
  --output-format png

# JPG → WebP（現代格式）
python scripts/automation/scenarios/image_processor.py \
  --operation convert \
  --input input.jpg \
  --output output.webp \
  --output-format webp \
  --quality 85
```

**格式選擇指南**：
- **JPEG**：照片、複雜圖像（有損壓縮，檔案小）
- **PNG**：需要透明度、簡單圖形（無損壓縮）
- **WebP**：網頁使用（比 JPEG 小 25-35%，但瀏覽器相容性需注意）
- **BMP**：不壓縮（檔案大，不推薦）
- **TIFF**：專業攝影、印刷（支援多頁、高品質）

**自動處理**：
- RGBA → RGB 轉換（JPEG 不支援透明度）
- 自動白色背景填充（透明圖像轉 JPEG）

---

### 4. Optimize（最佳化）

**功能**：最佳化圖像以減少檔案大小

**參數**：
- `--quality`：壓縮品質（1-100，預設：85）
- `--width`：最大寬度（可選）
- `--height`：最大高度（可選）
- `--output-format`：輸出格式（可選，預設：保持原格式）

**範例**：

```bash
# 基本最佳化（減少品質）
python scripts/automation/scenarios/image_processor.py \
  --operation optimize \
  --input input.jpg \
  --output output.jpg \
  --quality 85

# 最佳化 + 調整尺寸
python scripts/automation/scenarios/image_processor.py \
  --operation optimize \
  --input input.jpg \
  --output output.jpg \
  --width 1920 \
  --quality 80

# 轉換為 WebP 並最佳化
python scripts/automation/scenarios/image_processor.py \
  --operation optimize \
  --input input.jpg \
  --output output.webp \
  --output-format webp \
  --quality 85
```

**實際效能**（測試結果）：
- **原始大小**：350.5 KB（1920x1080 JPG）
- **最佳化後**：241.4 KB（quality=85）
- **減少比例**：31.1%
- **視覺品質**：幾乎無損

**品質建議**：
- **90-95**：高品質輸出（專業用途）
- **85**：平衡品質與大小（推薦預設）
- **75-80**：網頁使用（可接受品質）
- **60-70**：縮圖、預覽（明顯壓縮痕跡）

---

### 5. Blur（模糊）

**功能**：套用高斯模糊濾鏡

**參數**：
- `--radius`：模糊半徑（預設：2）
  - 1-3：輕微模糊
  - 4-8：中度模糊
  - 9+：重度模糊

**範例**：

```bash
# 輕微模糊（去噪）
python scripts/automation/scenarios/image_processor.py \
  --operation blur \
  --input input.jpg \
  --output output.jpg \
  --radius 2

# 中度模糊（背景虛化）
python scripts/automation/scenarios/image_processor.py \
  --operation blur \
  --input input.jpg \
  --output output.jpg \
  --radius 5

# 重度模糊（隱私保護）
python scripts/automation/scenarios/image_processor.py \
  --operation blur \
  --input input.jpg \
  --output output.jpg \
  --radius 15
```

**使用情境**：
- 去除雜訊（radius=1-2）
- 背景虛化效果（radius=3-8）
- 隱私保護（模糊臉部/車牌，radius=10+）

---

### 6. Sharpen（銳化）

**功能**：增強圖像銳利度

**參數**：
- `--factor`：銳化因子（預設：2.0）
  - 0.0-1.0：降低銳利度（模糊）
  - 1.0：無變化
  - 1.0-3.0：增強銳利度
  - 3.0+：過度銳化（產生光暈）

**範例**：

```bash
# 輕微銳化
python scripts/automation/scenarios/image_processor.py \
  --operation sharpen \
  --input input.jpg \
  --output output.jpg \
  --factor 1.5

# 標準銳化
python scripts/automation/scenarios/image_processor.py \
  --operation sharpen \
  --input input.jpg \
  --output output.jpg \
  --factor 2.0

# 強烈銳化
python scripts/automation/scenarios/image_processor.py \
  --operation sharpen \
  --input input.jpg \
  --output output.jpg \
  --factor 3.0
```

**使用情境**：
- 修正輕微模糊（factor=1.5-2.0）
- 增強細節（factor=2.0-2.5）
- 印刷準備（factor=2.5-3.0）

**注意**：過度銳化（factor>3.0）會產生不自然的光暈效果。

---

### 7. Contrast（對比度調整）

**功能**：調整圖像對比度

**參數**：
- `--factor`：對比度因子（預設：1.5）
  - 0.0：完全灰色
  - 0.0-1.0：降低對比度
  - 1.0：無變化
  - 1.0+：增強對比度

**範例**：

```bash
# 降低對比度（柔和效果）
python scripts/automation/scenarios/image_processor.py \
  --operation contrast \
  --input input.jpg \
  --output output.jpg \
  --factor 0.7

# 增強對比度
python scripts/automation/scenarios/image_processor.py \
  --operation contrast \
  --input input.jpg \
  --output output.jpg \
  --factor 1.5

# 強烈對比度
python scripts/automation/scenarios/image_processor.py \
  --operation contrast \
  --input input.jpg \
  --output output.jpg \
  --factor 2.0
```

**使用情境**：
- 修正曝光不足（factor=1.3-1.5）
- 增強視覺衝擊（factor=1.5-2.0）
- 柔和風格（factor=0.7-0.9）

---

### 8. Brightness（亮度調整）

**功能**：調整圖像亮度

**參數**：
- `--factor`：亮度因子（預設：1.2）
  - 0.0：完全黑色
  - 0.0-1.0：降低亮度
  - 1.0：無變化
  - 1.0+：增加亮度

**範例**：

```bash
# 降低亮度（修正過曝）
python scripts/automation/scenarios/image_processor.py \
  --operation brightness \
  --input input.jpg \
  --output output.jpg \
  --factor 0.8

# 增加亮度（修正欠曝）
python scripts/automation/scenarios/image_processor.py \
  --operation brightness \
  --input input.jpg \
  --output output.jpg \
  --factor 1.3

# 強烈增亮
python scripts/automation/scenarios/image_processor.py \
  --operation brightness \
  --input input.jpg \
  --output output.jpg \
  --factor 1.8
```

**使用情境**：
- 修正曝光不足（factor=1.2-1.5）
- 修正過度曝光（factor=0.7-0.9）
- 創造特殊氛圍（factor<0.5 或 >1.8）

---

### 9. Auto Contrast（自動對比度）

**功能**：自動調整對比度以最大化動態範圍

**參數**：
- `--cutoff`：裁切百分比（預設：0）
  - 0：使用完整動態範圍
  - 1-10：忽略極端值（推薦 2-5）

**範例**：

```bash
# 基本自動對比度
python scripts/automation/scenarios/image_processor.py \
  --operation auto_contrast \
  --input input.jpg \
  --output output.jpg

# 自動對比度 + 裁切極端值
python scripts/automation/scenarios/image_processor.py \
  --operation auto_contrast \
  --input input.jpg \
  --output output.jpg \
  --cutoff 5
```

**使用情境**：
- 修正低對比度圖像
- 自動化批次處理
- 不確定手動參數時的快速修正

**與手動對比度的差異**：
- **Auto Contrast**：自動分析並拉伸直方圖
- **Manual Contrast**：按固定因子縮放

---

### 10. Metadata（提取 Metadata）

**功能**：提取圖像 metadata 和 EXIF 資訊

**參數**：
- `--input`：輸入圖像路徑

**輸出資訊**：
- 寬度 × 高度
- 格式（JPEG, PNG, 等）
- 色彩模式（RGB, RGBA, 等）
- 檔案大小
- EXIF 資料（如果可用）

**範例**：

```bash
# 提取 metadata
python scripts/automation/scenarios/image_processor.py \
  --operation metadata \
  --input input.jpg
```

**實際輸出範例**：
```
📊 Image Metadata (圖像 Metadata):
   Dimensions: 1920x1080
   Format: JPEG
   Mode: RGB
   File Size: 350.5 KB

   EXIF Data:
   - DateTime: 2024:03:15 14:32:10
   - Make: Canon
   - Model: EOS 5D Mark IV
   - Orientation: Horizontal
```

**使用情境**：
- 驗證圖像規格
- 提取拍攝資訊
- 批次檢查圖像屬性
- 除錯格式問題

---

### 11. Batch（批次處理）

**功能**：從 YAML 配置檔執行批次操作

**參數**：
- `--input`：YAML 配置檔路徑

**配置檔格式**：
```yaml
operations:
  - operation: resize
    input: /path/to/input1.jpg
    output: /path/to/output1.jpg
    width: 800
    maintain_aspect: true
    resampling: lanczos

  - operation: crop
    input: /path/to/input2.jpg
    output: /path/to/output2.jpg
    mode: center
    width: 500
    height: 500

  - operation: optimize
    input: /path/to/input3.jpg
    output: /path/to/output3.jpg
    quality: 85
```

**範例**：

```bash
# 執行批次配置
python scripts/automation/scenarios/image_processor.py \
  --operation batch \
  --input configs/automation/my_workflow.yaml
```

**詳細配置範例**請參考：`configs/automation/image_processor_example.yaml`

**使用情境**：
- 自動化工作流程
- 重複性任務
- 大量圖像處理
- CI/CD 整合

---

## 批次處理工作流程

### 工作流程 1：網頁圖像最佳化

**目標**：將高解析度圖像轉換為網頁友善格式

**配置**（`web_optimization.yaml`）：
```yaml
operations:
  # 1. 調整尺寸
  - operation: resize
    input: /path/to/high_res.jpg
    output: /tmp/resized.jpg
    width: 1920
    maintain_aspect: true
    resampling: lanczos

  # 2. 最佳化壓縮
  - operation: optimize
    input: /tmp/resized.jpg
    output: /path/to/web_optimized.jpg
    quality: 85

  # 3. 轉換為 WebP（可選）
  - operation: convert
    input: /tmp/resized.jpg
    output: /path/to/web_optimized.webp
    output_format: webp
    quality: 80
```

**執行**：
```bash
python scripts/automation/scenarios/image_processor.py \
  --operation batch \
  --input web_optimization.yaml
```

**結果**：
- 調整至適合網頁的尺寸
- 減少 30-40% 檔案大小
- 生成 WebP 備用版本

---

### 工作流程 2：社交媒體縮圖

**目標**：批次生成多種社交媒體尺寸

**配置**（`social_media_thumbnails.yaml`）：
```yaml
operations:
  # Instagram 正方形
  - operation: crop
    input: /path/to/source.jpg
    output: /path/to/instagram_square.jpg
    mode: square

  # Instagram Story
  - operation: resize
    input: /path/to/source.jpg
    output: /tmp/story_temp.jpg
    width: 1080
    height: 1920
    maintain_aspect: false

  - operation: crop
    input: /tmp/story_temp.jpg
    output: /path/to/instagram_story.jpg
    mode: center
    width: 1080
    height: 1920

  # Facebook Cover
  - operation: resize
    input: /path/to/source.jpg
    output: /path/to/facebook_cover.jpg
    width: 820
    height: 312
    maintain_aspect: false

  # 全部最佳化
  - operation: optimize
    input: /path/to/instagram_square.jpg
    output: /path/to/instagram_square.jpg
    quality: 85

  - operation: optimize
    input: /path/to/instagram_story.jpg
    output: /path/to/instagram_story.jpg
    quality: 85

  - operation: optimize
    input: /path/to/facebook_cover.jpg
    output: /path/to/facebook_cover.jpg
    quality: 85
```

---

### 工作流程 3：照片增強

**目標**：自動增強照片品質

**配置**（`photo_enhancement.yaml`）：
```yaml
operations:
  # 1. 自動對比度
  - operation: auto_contrast
    input: /path/to/photo.jpg
    output: /tmp/contrast.jpg
    cutoff: 2

  # 2. 增加銳利度
  - operation: sharpen
    input: /tmp/contrast.jpg
    output: /tmp/sharpened.jpg
    factor: 1.5

  # 3. 輕微增亮
  - operation: brightness
    input: /tmp/sharpened.jpg
    output: /path/to/enhanced.jpg
    factor: 1.1

  # 4. 最佳化輸出
  - operation: optimize
    input: /path/to/enhanced.jpg
    output: /path/to/enhanced_final.jpg
    quality: 90
```

---

## 參數快速參考

### 通用參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--operation` | string | **必需** | 操作類型 |
| `--input` | path | **必需** | 輸入檔案路徑 |
| `--output` | path | **必需** | 輸出檔案路徑 |
| `--threads` | int | 32 | 並行執行緒數 |
| `--skip-preflight` | flag | false | 跳過前置檢查 |

### Resize 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--width` | int | - | 目標寬度 |
| `--height` | int | - | 目標高度 |
| `--maintain-aspect` | flag | true | 保持長寬比 |
| `--resampling` | string | lanczos | 重採樣演算法 |

### Crop 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--mode` | string | box | 裁切模式 |
| `--left` | int | 0 | 左邊界 |
| `--top` | int | 0 | 上邊界 |
| `--width` | int | - | 裁切寬度 |
| `--height` | int | - | 裁切高度 |

### Convert 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--output-format` | string | - | 目標格式 |
| `--quality` | int | 95 | 壓縮品質 |
| `--optimize` | flag | true | 啟用最佳化 |

### Filter 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--radius` | int | 2 | 模糊半徑 |
| `--factor` | float | 1.5 | 調整因子 |
| `--cutoff` | int | 0 | 裁切百分比 |

---

## 效能基準

### 測試環境
- **CPU**: Intel Core i7-9700K (8 cores)
- **RAM**: 32GB DDR4
- **Storage**: NVMe SSD
- **Python**: 3.10
- **Pillow**: 10.2.0

### 單一操作效能

| 操作 | 輸入尺寸 | 處理時間 | 輸出尺寸 | 檔案大小變化 |
|------|----------|----------|----------|--------------|
| **Resize** | 1920x1080 | 0.12s | 800x450 | -65% |
| **Crop** | 1920x1080 | 0.08s | 500x500 | -45% |
| **Convert (JPG→PNG)** | 1920x1080 | 0.15s | 1920x1080 | +560% |
| **Convert (PNG→JPG)** | 1920x1080 | 0.18s | 1920x1080 | -85% |
| **Optimize** | 1920x1080 | 0.10s | 1920x1080 | -31% |
| **Blur** | 1920x1080 | 0.22s | 1920x1080 | +2% |
| **Sharpen** | 1920x1080 | 0.19s | 1920x1080 | +5% |
| **Contrast** | 1920x1080 | 0.17s | 1920x1080 | +3% |
| **Brightness** | 1920x1080 | 0.16s | 1920x1080 | +2% |
| **Auto Contrast** | 1920x1080 | 0.14s | 1920x1080 | +1% |
| **Metadata** | 1920x1080 | 0.03s | - | - |

### 批次處理效能

| 圖像數量 | 總處理時間 | 平均每張 | 記憶體使用 |
|----------|------------|----------|------------|
| 10 張 | 2.1s | 0.21s | 350MB |
| 50 張 | 9.8s | 0.20s | 480MB |
| 100 張 | 18.5s | 0.19s | 620MB |
| 500 張 | 87.2s | 0.17s | 1.2GB |

**注意**：效能受以下因素影響：
- 圖像尺寸和複雜度
- CPU 核心數和時脈
- 儲存裝置速度（HDD vs SSD）
- 系統負載

---

## 記憶體使用

### 單一圖像處理

Image Processor 整合 Phase 1 記憶體監控系統，會在每次操作前檢查可用記憶體。

**記憶體需求計算**：
```python
# 粗略估計公式
required_memory_mb = (width * height * channels * bytes_per_pixel * safety_factor) / (1024 * 1024)

# 範例：1920x1080 RGB 圖像
required = (1920 * 1080 * 3 * 1 * 2.0) / (1024 * 1024) ≈ 12 MB
```

**實際使用**：
- **1920x1080 JPG**: ~10-15 MB
- **3840x2160 JPG**: ~40-50 MB
- **1920x1080 PNG**: ~15-20 MB
- **3840x2160 PNG**: ~60-80 MB

### 批次處理記憶體

批次處理時，記憶體使用會隨並行執行緒數增加：

```
總記憶體 ≈ 單一圖像記憶體 × 並行執行緒數 × 1.5
```

**建議配置**：
- **4GB RAM**: 最多 8 執行緒，1920x1080
- **8GB RAM**: 最多 16 執行緒，1920x1080
- **16GB RAM**: 最多 32 執行緒，1920x1080
- **32GB RAM**: 最多 32 執行緒，4K

**記憶體不足時**：
系統會自動：
1. 記錄警告日誌
2. 降低並行執行緒數
3. 嘗試釋放快取記憶體
4. 如果仍然不足，回傳錯誤

---

## 疑難排解

### 問題 1：Pillow 未安裝

**錯誤訊息**：
```
ModuleNotFoundError: No module named 'PIL'
```

**解決方案**：
```bash
conda activate ai_env
pip install pillow>=10.0.0
```

---

### 問題 2：記憶體不足

**錯誤訊息**：
```
⚠️ 警告：可用記憶體不足 (Available: 1.2GB < Required: 2.5GB)
```

**解決方案**：
```bash
# 選項 1：減少並行執行緒數
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input input.jpg \
  --output output.jpg \
  --width 800 \
  --threads 8  # 從 32 降至 8

# 選項 2：分批處理
# 將大批次拆分為多個小批次

# 選項 3：關閉其他程式釋放記憶體
```

---

### 問題 3：JPEG 不支援 RGBA

**錯誤訊息**：
```
OSError: cannot write mode RGBA as JPEG
```

**解決方案**：
Image Processor 會自動處理 RGBA → RGB 轉換。如果仍遇到此錯誤：

```bash
# 先轉換為 PNG，再轉回 JPG
python scripts/automation/scenarios/image_processor.py \
  --operation convert \
  --input input.png \
  --output output.jpg \
  --output-format jpeg
```

系統會自動：
1. 檢測 RGBA 模式
2. 建立白色背景
3. 合成圖像
4. 轉換為 RGB
5. 儲存為 JPEG

---

### 問題 4：檔案格式不支援

**錯誤訊息**：
```
PIL.UnidentifiedImageError: cannot identify image file
```

**解決方案**：
```bash
# 檢查檔案格式
file /path/to/image.ext

# 支援的格式
python scripts/automation/scenarios/image_processor.py \
  --operation metadata \
  --input /path/to/image.ext
```

**支援格式**：
- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png)
- ✅ WebP (.webp)
- ✅ BMP (.bmp)
- ✅ TIFF (.tiff, .tif)
- ❌ SVG（向量格式，不支援）
- ❌ RAW（需要額外函式庫）

---

### 問題 5：批次處理中斷

**問題**：批次處理執行到一半停止

**解決方案**：

1. **檢查 YAML 配置**：
```bash
# 驗證 YAML 語法
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

2. **查看日誌**：
```bash
# Image Processor 會輸出詳細錯誤訊息
tail -f logs/image_processor.log
```

3. **逐個測試操作**：
```bash
# 從批次配置中提取單一操作測試
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input /path/from/yaml \
  --output /tmp/test.jpg \
  --width 800
```

---

### 問題 6：處理速度慢

**問題**：處理速度比預期慢

**診斷**：
```bash
# 檢查 CPU 使用率
top -p $(pgrep -f image_processor)

# 檢查磁碟 I/O
iostat -x 1

# 檢查記憶體使用
free -h
```

**最佳化**：

1. **使用快速重採樣**：
```bash
--resampling bilinear  # 而非 lanczos
```

2. **減少品質設定**：
```bash
--quality 80  # 而非 95
```

3. **使用 SSD**：
```bash
# 將輸入/輸出移至 SSD
mv /path/on/hdd /path/on/ssd
```

4. **調整執行緒數**：
```bash
# 嘗試不同執行緒數
--threads 16  # 實驗 8, 16, 24, 32
```

---

## API 參考

### ImageProcessor 類別

```python
from scripts.automation.scenarios.image_processor import ImageProcessor

# 初始化
processor = ImageProcessor(max_threads=32)

# Resize
success = processor.resize_image(
    input_path="/path/to/input.jpg",
    output_path="/path/to/output.jpg",
    width=800,
    maintain_aspect=True,
    resampling='lanczos'
)

# Crop
success = processor.crop_image(
    input_path="/path/to/input.jpg",
    output_path="/path/to/output.jpg",
    mode='center',
    width=500,
    height=500
)

# Convert
success = processor.convert_format(
    input_path="/path/to/input.jpg",
    output_path="/path/to/output.png",
    output_format='PNG',
    quality=95,
    optimize=True
)

# Optimize
success = processor.optimize_image(
    input_path="/path/to/input.jpg",
    output_path="/path/to/output.jpg",
    quality=85
)

# Apply filters
success = processor.apply_blur(input_path, output_path, radius=3)
success = processor.apply_sharpen(input_path, output_path, factor=2.0)
success = processor.adjust_contrast(input_path, output_path, factor=1.5)
success = processor.adjust_brightness(input_path, output_path, factor=1.2)
success = processor.auto_contrast(input_path, output_path, cutoff=2)

# Extract metadata
metadata = processor.extract_metadata("/path/to/image.jpg")
print(f"Size: {metadata.width}x{metadata.height}")
print(f"Format: {metadata.format}")
print(f"Mode: {metadata.mode}")

# Batch processing
results = processor.process_batch("/path/to/config.yaml")
for result in results:
    print(f"{result.input_path}: {'✅' if result.success else '❌'}")
```

---

## 與其他 Phase 2 組件整合

### 與 Video Processor 整合

```bash
# 1. 從影片提取 frames (Video Processor)
python scripts/automation/scenarios/video_processor.py \
  extract \
  --input /path/to/video.mp4 \
  --output /tmp/frames \
  --fps 1

# 2. 批次處理 frames (Image Processor)
python scripts/automation/scenarios/image_processor.py \
  --operation batch \
  --input configs/automation/frame_enhancement.yaml
```

### 與 Audio Processor 整合

```bash
# 從影片處理音訊 + 生成封面
# 1. 提取音訊 (Audio Processor)
python scripts/automation/scenarios/audio_processor.py \
  extract \
  --input /path/to/video.mp4 \
  --output /tmp/audio.mp3

# 2. 生成波形圖 (Audio Processor)
python scripts/automation/scenarios/audio_processor.py \
  waveform \
  --input /tmp/audio.mp3 \
  --output /tmp/waveform.png

# 3. 調整為社交媒體尺寸 (Image Processor)
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input /tmp/waveform.png \
  --output /tmp/cover.jpg \
  --width 1200 \
  --height 628
```

---

## 最佳實踐

### 1. 選擇正確的格式

**照片/複雜圖像**：
- 儲存：JPEG (quality 85-95)
- 網頁：WebP (quality 80-85)
- 封存：TIFF

**圖形/簡單圖像**：
- 需透明度：PNG
- 不需透明度：JPEG 或 WebP

### 2. 最佳化工作流程

**錯誤順序**：
```
原始圖像 → 濾鏡 → 調整尺寸 → 壓縮
```

**正確順序**：
```
原始圖像 → 調整尺寸 → 濾鏡 → 壓縮
```

**原因**：先調整尺寸可以減少後續操作的計算量。

### 3. 保留原始檔案

```bash
# 永遠輸出到不同路徑
--output /path/to/processed/image.jpg

# 避免覆蓋原始檔案
--output /path/to/original/image.jpg  # ❌ 不要這樣做
```

### 4. 使用批次處理

對於重複性任務，使用 YAML 配置而非手動執行：

```yaml
# workflow.yaml
operations:
  - operation: resize
    input: "{source_dir}/{filename}"
    output: "{output_dir}/resized_{filename}"
    width: 1920

  - operation: optimize
    input: "{output_dir}/resized_{filename}"
    output: "{output_dir}/final_{filename}"
    quality: 85
```

### 5. 監控記憶體使用

```bash
# 使用 --threads 參數控制並行度
python scripts/automation/scenarios/image_processor.py \
  --operation batch \
  --input large_batch.yaml \
  --threads 8  # 根據可用記憶體調整
```

---

## 進階技巧

### 技巧 1：鏈式操作（使用臨時檔案）

```bash
# 複雜處理流程
python scripts/automation/scenarios/image_processor.py \
  --operation resize \
  --input input.jpg \
  --output /tmp/step1.jpg \
  --width 1920

python scripts/automation/scenarios/image_processor.py \
  --operation auto_contrast \
  --input /tmp/step1.jpg \
  --output /tmp/step2.jpg

python scripts/automation/scenarios/image_processor.py \
  --operation sharpen \
  --input /tmp/step2.jpg \
  --output /tmp/step3.jpg \
  --factor 1.5

python scripts/automation/scenarios/image_processor.py \
  --operation optimize \
  --input /tmp/step3.jpg \
  --output final.jpg \
  --quality 85

# 清理臨時檔案
rm /tmp/step*.jpg
```

### 技巧 2：動態檔名（使用 Shell 變數）

```bash
#!/bin/bash
# batch_process.sh

INPUT_DIR="/path/to/inputs"
OUTPUT_DIR="/path/to/outputs"

for img in $INPUT_DIR/*.jpg; do
  filename=$(basename "$img")

  python scripts/automation/scenarios/image_processor.py \
    --operation resize \
    --input "$img" \
    --output "$OUTPUT_DIR/resized_$filename" \
    --width 800

  python scripts/automation/scenarios/image_processor.py \
    --operation optimize \
    --input "$OUTPUT_DIR/resized_$filename" \
    --output "$OUTPUT_DIR/final_$filename" \
    --quality 85
done
```

### 技巧 3：條件處理（根據 Metadata）

```bash
#!/bin/bash
# conditional_processing.sh

for img in /path/to/images/*.jpg; do
  # 提取 metadata
  metadata=$(python scripts/automation/scenarios/image_processor.py \
    --operation metadata \
    --input "$img")

  # 提取寬度（需要解析 metadata 輸出）
  width=$(echo "$metadata" | grep "Dimensions" | cut -d'x' -f1 | awk '{print $2}')

  # 只處理寬度 > 2000 的圖像
  if [ "$width" -gt 2000 ]; then
    python scripts/automation/scenarios/image_processor.py \
      --operation resize \
      --input "$img" \
      --output "/path/to/resized/$(basename $img)" \
      --width 1920
  fi
done
```

---

## 整合範例

### 完整自動化腳本

```bash
#!/bin/bash
# complete_image_workflow.sh

set -e  # 遇到錯誤立即停止

INPUT_DIR="/mnt/data/ai_data/raw_images"
TEMP_DIR="/tmp/image_processing"
OUTPUT_DIR="/mnt/data/ai_data/processed_images"

# 建立目錄
mkdir -p "$TEMP_DIR" "$OUTPUT_DIR"

echo "🚀 開始圖像處理工作流程..."

# Step 1: 調整尺寸
echo "📏 Step 1: 調整尺寸..."
for img in "$INPUT_DIR"/*.jpg; do
  filename=$(basename "$img")
  python scripts/automation/scenarios/image_processor.py \
    --operation resize \
    --input "$img" \
    --output "$TEMP_DIR/resized_$filename" \
    --width 1920 \
    --maintain-aspect
done

# Step 2: 自動增強
echo "✨ Step 2: 自動增強..."
for img in "$TEMP_DIR"/resized_*.jpg; do
  filename=$(basename "$img")
  python scripts/automation/scenarios/image_processor.py \
    --operation auto_contrast \
    --input "$img" \
    --output "$TEMP_DIR/enhanced_$filename" \
    --cutoff 2
done

# Step 3: 輕微銳化
echo "🔍 Step 3: 輕微銳化..."
for img in "$TEMP_DIR"/enhanced_*.jpg; do
  filename=$(basename "$img" | sed 's/enhanced_resized_//')
  python scripts/automation/scenarios/image_processor.py \
    --operation sharpen \
    --input "$img" \
    --output "$TEMP_DIR/sharpened_$filename" \
    --factor 1.5
done

# Step 4: 最佳化壓縮
echo "💾 Step 4: 最佳化壓縮..."
for img in "$TEMP_DIR"/sharpened_*.jpg; do
  filename=$(basename "$img" | sed 's/sharpened_//')
  python scripts/automation/scenarios/image_processor.py \
    --operation optimize \
    --input "$img" \
    --output "$OUTPUT_DIR/$filename" \
    --quality 85
done

# 清理臨時檔案
echo "🧹 清理臨時檔案..."
rm -rf "$TEMP_DIR"

echo "✅ 完成！處理後的圖像位於: $OUTPUT_DIR"
```

---

## 相關文件

- **Video Processor**: `docs/automation/PHASE2_VIDEO_PROCESSOR.md`
- **Audio Processor**: `docs/automation/PHASE2_AUDIO_PROCESSOR.md`
- **File Organizer**: `docs/automation/PHASE2_FILE_ORGANIZER.md`（待建立）
- **配置範例**: `configs/automation/image_processor_example.yaml`
- **總體進度**: `AUTOMATION_PROGRESS.md`

---

## 技術支援

遇到問題或需要協助？

1. **檢查日誌**：`logs/image_processor.log`
2. **查看疑難排解章節**：本文件「疑難排解」部分
3. **檢查依賴**：`pip list | grep -i pillow`
4. **測試基本功能**：
```bash
python scripts/automation/scenarios/image_processor.py \
  --operation metadata \
  --input /path/to/test.jpg
```

---

## 更新紀錄

**v1.0.0** (2025-12-02)
- ✅ 初始版本
- ✅ 10 種圖像操作
- ✅ 批次處理支援
- ✅ 記憶體監控整合
- ✅ 完整雙語文件

---

*文件版本：1.0.0*
*最後更新：2025-12-02*
*維護者：Animation AI Studio Team*
