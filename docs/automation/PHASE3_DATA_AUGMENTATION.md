# Phase 3: Data Processing Automation - 資料增強管道 (Data Augmentation Pipeline)

## 概覽 (Overview)

資料增強管道（Data Augmentation Pipeline）是 Phase 3 的第三個核心元件，提供全面的影像增強功能，透過各種轉換技術人工擴展訓練資料集。這個工具支援多種增強類型，從簡單的幾何變換到進階的影像處理效果，並提供三個預設等級（輕量、中度、強力）以及完整的自訂配置能力。

### 核心功能

- **5 大類增強類型**：幾何變換、色彩調整、噪點添加、模糊效果、進階處理
- **預設配置系統**：Light、Medium、Strong 三種預設等級
- **批次處理**：支援大量影像的並行處理
- **決定性增強**：使用隨機種子確保可重現的結果
- **機率式應用**：每種增強都有可調整的應用機率
- **統計報告**：產生詳細的處理統計資料
- **品質控制**：影像驗證和錯誤處理機制
- **CPU 運算**：基於 PIL/Pillow 的 CPU 處理，無需 GPU

### 適用場景

1. **機器學習訓練**：擴展訓練資料集以提高模型泛化能力
2. **資料稀缺處理**：在資料量有限時增加樣本多樣性
3. **類別平衡**：為少數類別產生更多訓練樣本
4. **模型正則化**：透過增強減少過擬合
5. **域適應**：調整資料集以適應不同的視覺條件

## 快速開始 (Quick Start)

### 基本使用

**單張影像增強（使用預設）**：
```bash
cd /mnt/c/ai_projects/animation-ai-studio

python scripts/automation/scenarios/data_augmentation.py single \
  --input /path/to/image.jpg \
  --output /path/to/augmented.jpg \
  --preset medium \
  --seed 42
```

**批次影像增強**：
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir /path/to/images \
  --output-dir /path/to/augmented \
  --preset medium \
  --num-per-image 3
```

**使用自訂配置**：
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir /path/to/images \
  --output-dir /path/to/augmented \
  --config configs/automation/augmentation/custom_augmentation_config.yaml \
  --num-per-image 5 \
  --seed 42
```

## 增強類型 (Augmentation Types)

### 1. 幾何變換 (Geometric Transformations)

幾何變換改變影像的空間配置，包括旋轉、翻轉、裁剪和縮放。

#### 可用的幾何變換

**水平翻轉 (Horizontal Flip)**
```python
# 配置範例
- name: "horizontal_flip"
  type: "geometric"
  probability: 0.5
  parameters: {}
```
- 用途：鏡像翻轉影像
- 適用：大多數自然影像，除非方向性重要

**垂直翻轉 (Vertical Flip)**
```python
- name: "vertical_flip"
  type: "geometric"
  probability: 0.2
  parameters: {}
```
- 用途：上下翻轉影像
- 適用：某些特定領域（醫學影像、衛星影像）

**旋轉 (Rotation)**
```python
- name: "rotate"
  type: "geometric"
  probability: 0.3
  parameters:
    angle_range: [-15, 15]  # 旋轉角度範圍（度）
    expand: true  # 擴展畫布以適應旋轉後的影像
```
- 用途：模擬不同的相機角度
- 參數：
  - `angle_range`: 隨機旋轉角度範圍（負數為逆時針）
  - `expand`: 是否擴展畫布（false 會裁切邊緣）

**隨機裁剪 (Random Crop)**
```python
- name: "random_crop"
  type: "geometric"
  probability: 0.3
  parameters:
    crop_fraction_range: [0.7, 0.9]  # 保留 70-90% 的影像
```
- 用途：隨機裁剪影像區域
- 參數：`crop_fraction_range` - 保留比例範圍

**中心裁剪 (Center Crop)**
```python
- name: "center_crop"
  type: "geometric"
  probability: 0.2
  parameters:
    crop_fraction: 0.9  # 保留中心 90%
```
- 用途：裁剪影像中心區域
- 參數：`crop_fraction` - 固定的保留比例

**縮放 (Scale/Zoom)**
```python
- name: "random_scale"
  type: "geometric"
  probability: 0.3
  parameters:
    scale_range: [0.8, 1.2]  # 縮放因子範圍
```
- 用途：縮放影像大小
- 參數：`scale_range` - 縮放因子範圍（1.0 = 原始大小）

### 2. 色彩調整 (Color Adjustments)

色彩調整修改影像的顏色屬性，包括亮度、對比度、飽和度和色調。

#### 可用的色彩調整

**亮度調整 (Brightness)**
```python
- name: "random_brightness"
  type: "color"
  probability: 0.7
  parameters:
    range: [0.8, 1.2]  # 亮度因子範圍
```
- 用途：模擬不同的光照條件
- 參數：
  - `range`: 亮度因子範圍
  - 1.0 = 無變化，< 1.0 = 變暗，> 1.0 = 變亮

**對比度調整 (Contrast)**
```python
- name: "random_contrast"
  type: "color"
  probability: 0.7
  parameters:
    range: [0.8, 1.2]  # 對比度因子範圍
```
- 用途：調整影像對比度
- 參數：
  - `range`: 對比度因子範圍
  - 1.0 = 無變化，< 1.0 = 降低對比，> 1.0 = 增加對比

**飽和度調整 (Saturation)**
```python
- name: "random_saturation"
  type: "color"
  probability: 0.5
  parameters:
    range: [0.8, 1.2]  # 飽和度因子範圍
```
- 用途：調整顏色飽和度
- 參數：
  - `range`: 飽和度因子範圍
  - 1.0 = 無變化，0.0 = 灰階，> 1.0 = 更飽和

**色調偏移 (Hue Shift)**
```python
- name: "random_hue_shift"
  type: "color"
  probability: 0.3
  parameters:
    delta_range: [-30, 30]  # 色調偏移範圍（度）
```
- 用途：旋轉色相環
- 參數：`delta_range` - 色調偏移範圍（-180 到 180 度）

### 3. 噪點添加 (Noise Augmentations)

噪點增強為影像添加隨機噪點，模擬感測器噪點或低光條件。

#### 可用的噪點類型

**高斯噪點 (Gaussian Noise)**
```python
- name: "gaussian_noise"
  type: "noise"
  probability: 0.2
  parameters:
    mean: 0
    std_range: [5, 15]  # 標準差範圍
```
- 用途：模擬相機感測器噪點
- 參數：
  - `mean`: 噪點平均值（通常為 0）
  - `std_range`: 標準差範圍（值越大噪點越明顯）

**椒鹽噪點 (Salt and Pepper)**
```python
- name: "salt_and_pepper"
  type: "noise"
  probability: 0.1
  parameters:
    amount: 0.02  # 受影響的像素比例
```
- 用途：模擬影像傳輸錯誤
- 參數：`amount` - 添加噪點的像素比例（0.0 到 1.0）

### 4. 模糊效果 (Blur Augmentations)

模糊增強應用不同類型的模糊效果，模擬失焦或運動模糊。

#### 可用的模糊類型

**隨機模糊 (Random Blur)**
```python
- name: "random_blur"
  type: "blur"
  probability: 0.2
  parameters:
    blur_type: "gaussian"  # 選項：gaussian, box, motion, random
    radius_range: [0.5, 2.0]  # 模糊半徑範圍
```
- 用途：模擬失焦或運動效果
- 參數：
  - `blur_type`: 模糊類型
    - `gaussian`: 高斯模糊（自然外觀）
    - `box`: 盒型模糊（快速但較不自然）
    - `motion`: 運動模糊（方向性模糊）
    - `random`: 隨機選擇類型
  - `radius_range`: 模糊強度範圍

### 5. 進階處理 (Advanced Augmentations)

進階增強包括更複雜的影像處理技術。

#### 可用的進階處理

**隨機遮罩 (Cutout)**
```python
- name: "cutout"
  type: "advanced"
  probability: 0.2
  parameters:
    num_holes: 1
    hole_size_range: [0.05, 0.15]  # 遮罩大小範圍（相對於影像大小）
```
- 用途：強制模型學習更魯棒的特徵
- 參數：
  - `num_holes`: 遮罩數量
  - `hole_size_range`: 遮罩大小範圍（影像大小的比例）

**色彩量化 (Posterize)**
```python
- name: "posterize"
  type: "advanced"
  probability: 0.1
  parameters:
    bits_range: [4, 6]  # 每個色彩通道的位元數
```
- 用途：減少色彩深度
- 參數：`bits_range` - 位元數範圍（1-8，值越小色彩越少）

**曝光反轉 (Solarize)**
```python
- name: "solarize"
  type: "advanced"
  probability: 0.1
  parameters:
    threshold: 128  # 閾值（0-255）
```
- 用途：反轉超過閾值的像素
- 參數：`threshold` - 反轉閾值（0-255）

**直方圖均衡 (Equalize)**
```python
- name: "equalize"
  type: "advanced"
  probability: 0.1
  parameters: {}
```
- 用途：改善影像對比度分佈

**自動對比 (Autocontrast)**
```python
- name: "autocontrast"
  type: "advanced"
  probability: 0.1
  parameters:
    cutoff: 0  # 忽略最亮/最暗像素的百分比
```
- 用途：自動調整對比度

## 預設配置 (Preset Configurations)

### Light Preset（輕量預設）

**特性**：
- 最小的影像修改
- 保持高影像品質（95% JPEG 品質）
- 適用於高品質資料集

**包含的增強**：
- 隨機亮度調整（±10%，50% 機率）
- 隨機對比度調整（±10%，50% 機率）
- 水平翻轉（50% 機率）

**使用時機**：
- 原始資料已經高品質且多樣化
- 需要保持最大保真度
- 專業攝影資料集

**範例**：
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./high_quality_images \
  --output-dir ./augmented_light \
  --preset light \
  --num-per-image 2
```

### Medium Preset（中度預設）

**特性**：
- 平衡多樣性和品質
- 多種增強類型
- 適用於大多數使用案例

**包含的增強**：
- 隨機亮度調整（±20%，70% 機率）
- 隨機對比度調整（±20%，70% 機率）
- 隨機飽和度調整（±20%，50% 機率）
- 水平翻轉（50% 機率）
- 垂直翻轉（20% 機率）
- 旋轉（±15°，30% 機率）
- 隨機模糊（20% 機率）

**使用時機**：
- 標準機器學習任務
- 需要平衡的增強策略
- 最常見的使用情境

**範例**：
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./standard_images \
  --output-dir ./augmented_medium \
  --preset medium \
  --num-per-image 3
```

### Strong Preset（強力預設）

**特性**：
- 最大的變化多樣性
- 積極的增強策略
- 適用於資料稀缺場景

**包含的增強**：
- 隨機亮度調整（±30%，80% 機率）
- 隨機對比度調整（±30%，80% 機率）
- 隨機飽和度調整（±30%，70% 機率）
- 隨機色調偏移（±30°，30% 機率）
- 水平翻轉（50% 機率）
- 垂直翻轉（30% 機率）
- 旋轉（±30°，50% 機率）
- 隨機裁剪（70-90% 保留，30% 機率）
- 隨機縮放（80-120%，30% 機率）
- 隨機模糊（30% 機率）
- 高斯噪點（20% 機率）
- 隨機遮罩（20% 機率）
- 色彩量化（10% 機率）

**使用時機**：
- 訓練資料極少（< 100 張/類別）
- 需要最大多樣性
- 少樣本學習場景

**範例**：
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./limited_images \
  --output-dir ./augmented_strong \
  --preset strong \
  --num-per-image 5
```

## 自訂管道 (Custom Pipelines)

### 建立自訂配置

1. **複製範本**：
```bash
cp configs/automation/augmentation/custom_augmentation_config.yaml \
   configs/automation/augmentation/my_custom_config.yaml
```

2. **編輯配置檔案**：
```yaml
metadata:
  name: "My Custom Augmentation"
  description: "Tailored for my specific dataset"
  version: "1.0.0"
  author: "Your Name"

augmentations:
  # 取消註解並調整需要的增強
  - name: "random_brightness"
    type: "color"
    probability: 0.6
    parameters:
      range: [0.85, 1.15]

  - name: "horizontal_flip"
    type: "geometric"
    probability: 0.5
    parameters: {}

  # 添加更多增強...
```

3. **測試配置**：
```bash
# 先用小批次測試
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./test_images \
  --output-dir ./test_output \
  --config configs/automation/augmentation/my_custom_config.yaml \
  --num-per-image 2
```

4. **檢查結果並調整**：
- 檢查輸出影像
- 根據需要調整機率和參數
- 重複測試直到滿意

### 配置範例：醫學影像增強

```yaml
metadata:
  name: "Medical Image Augmentation"
  description: "Optimized for medical imaging"
  version: "1.0.0"

augmentations:
  # 保守的亮度調整
  - name: "random_brightness"
    type: "color"
    probability: 0.4
    parameters:
      range: [0.95, 1.05]  # 僅 ±5%

  # 保守的對比度調整
  - name: "random_contrast"
    type: "color"
    probability: 0.4
    parameters:
      range: [0.95, 1.05]

  # 旋轉（小角度）
  - name: "rotate"
    type: "geometric"
    probability: 0.3
    parameters:
      angle_range: [-5, 5]  # 僅 ±5°

  # 水平翻轉（如果對稱性不重要）
  - name: "horizontal_flip"
    type: "geometric"
    probability: 0.5
    parameters: {}

  # 高斯噪點（模擬感測器噪點）
  - name: "gaussian_noise"
    type: "noise"
    probability: 0.2
    parameters:
      mean: 0
      std_range: [3, 8]  # 輕微噪點

output:
  format: "png"  # 無損格式
  quality: 100
  preserve_metadata: true

processing:
  seed: 42  # 可重現
  num_workers: 4
  batch_size: 8
```

### 配置範例：衛星影像增強

```yaml
metadata:
  name: "Satellite Image Augmentation"
  description: "Optimized for satellite imagery"
  version: "1.0.0"

augmentations:
  # 亮度調整（模擬不同時間）
  - name: "random_brightness"
    type: "color"
    probability: 0.6
    parameters:
      range: [0.7, 1.3]

  # 對比度調整
  - name: "random_contrast"
    type: "color"
    probability: 0.6
    parameters:
      range: [0.8, 1.2]

  # 旋轉（任意角度）
  - name: "rotate"
    type: "geometric"
    probability: 0.5
    parameters:
      angle_range: [-180, 180]  # 完整旋轉

  # 翻轉（所有方向）
  - name: "horizontal_flip"
    type: "geometric"
    probability: 0.5
    parameters: {}

  - name: "vertical_flip"
    type: "geometric"
    probability: 0.5
    parameters: {}

  # 隨機裁剪
  - name: "random_crop"
    type: "geometric"
    probability: 0.4
    parameters:
      crop_fraction_range: [0.8, 0.95]

  # 模糊（模擬大氣條件）
  - name: "random_blur"
    type: "blur"
    probability: 0.3
    parameters:
      blur_type: "gaussian"
      radius_range: [0.5, 3.0]

output:
  format: "jpg"
  quality: 95
  preserve_metadata: false

processing:
  seed: null  # 隨機
  num_workers: 8
  batch_size: 16
```

## 命令行參考 (CLI Reference)

### 通用參數

所有命令共用的參數：

```bash
--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
    日誌等級（預設：INFO）

--skip-preflight
    跳過預檢查（環境驗證）
```

### single 命令

增強單張影像。

**語法**：
```bash
python scripts/automation/scenarios/data_augmentation.py single \
  --input INPUT_PATH \
  --output OUTPUT_PATH \
  [OPTIONS]
```

**必要參數**：
- `--input PATH`: 輸入影像路徑
- `--output PATH`: 輸出影像路徑

**可選參數**：
- `--preset {light,medium,strong}`: 使用預設配置
- `--config PATH`: 使用自訂 YAML 配置檔案
- `--seed INT`: 隨機種子（用於可重現性）

**範例**：
```bash
# 使用 medium 預設
python scripts/automation/scenarios/data_augmentation.py single \
  --input photo.jpg \
  --output photo_augmented.jpg \
  --preset medium \
  --seed 42

# 使用自訂配置
python scripts/automation/scenarios/data_augmentation.py single \
  --input photo.jpg \
  --output photo_augmented.jpg \
  --config my_config.yaml
```

### batch 命令

批次增強多張影像。

**語法**：
```bash
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir INPUT_DIR \
  --output-dir OUTPUT_DIR \
  [OPTIONS]
```

**必要參數**：
- `--input-dir PATH`: 輸入影像目錄
- `--output-dir PATH`: 輸出影像目錄

**可選參數**：
- `--preset {light,medium,strong}`: 使用預設配置
- `--config PATH`: 使用自訂 YAML 配置檔案
- `--num-per-image INT`: 每張影像產生的增強版本數量（預設：1）
- `--seed INT`: 隨機種子

**範例**：
```bash
# 使用 medium 預設，每張影像產生 3 個版本
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented \
  --preset medium \
  --num-per-image 3

# 使用自訂配置，每張影像產生 5 個版本
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented \
  --config my_config.yaml \
  --num-per-image 5 \
  --seed 42
```

## 配置檔案 (Configuration Files)

### 配置結構

所有配置檔案遵循以下結構：

```yaml
# 元資料
metadata:
  name: "配置名稱"
  description: "配置說明"
  version: "1.0.0"
  author: "作者名稱"

# 增強管道配置
augmentations:
  - name: "增強名稱"
    type: "增強類型"  # color, geometric, noise, blur, advanced
    probability: 0.5  # 0.0 到 1.0
    parameters:
      # 類型特定的參數

# 輸出設定
output:
  format: "jpg"  # jpg, png, webp
  quality: 95  # 1-100
  preserve_metadata: true

# 處理設定
processing:
  seed: null  # null = 隨機，或指定整數
  num_workers: 4
  batch_size: 16

# 驗證設定
validation:
  min_image_size: [64, 64]
  max_image_size: [4096, 4096]
  allowed_formats: ["jpg", "jpeg", "png", "webp"]
```

### 參數說明

#### 機率（Probability）

控制增強被應用的機會：

- `0.0`: 從不應用
- `0.3`: 應用於 30% 的影像
- `0.5`: 應用於 50% 的影像（平衡）
- `0.7`: 應用於 70% 的影像
- `1.0`: 總是應用

#### 色彩因子範圍

色彩調整的強度：

- `[0.9, 1.1]`: 輕微（±10%）
- `[0.8, 1.2]`: 中度（±20%）
- `[0.7, 1.3]`: 強力（±30%）
- `1.0` = 無變化

#### 旋轉角度

旋轉的角度範圍：

- `[-5, 5]`: 非常輕微
- `[-15, 15]`: 中度
- `[-30, 30]`: 強力
- `[-45, 45]`: 非常強力

#### 裁剪比例

裁剪時保留的影像比例：

- `0.95`: 保留 95%（最小裁剪）
- `0.8-0.9`: 中度裁剪
- `0.7-0.8`: 強力裁剪

## 最佳實踐 (Best Practices)

### 1. 選擇適當的預設

**決策流程**：

```
是否有充足的資料（> 500 張/類別）？
├─ 是 → 使用 Light 預設
└─ 否
    └─ 資料品質如何？
        ├─ 高品質 → 使用 Medium 預設
        └─ 需要最大多樣性 → 使用 Strong 預設
```

### 2. 漸進式測試

不要直接在完整資料集上運行：

```bash
# 步驟 1：小批次測試
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./sample_10_images \
  --output-dir ./test_output \
  --preset medium \
  --num-per-image 2

# 步驟 2：檢查結果
ls -lh ./test_output
# 視覺檢查輸出影像

# 步驟 3：中批次測試
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./sample_100_images \
  --output-dir ./test_output_medium \
  --preset medium \
  --num-per-image 3

# 步驟 4：完整處理
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./all_images \
  --output-dir ./augmented_final \
  --preset medium \
  --num-per-image 3
```

### 3. 使用隨機種子進行除錯

在開發和測試時使用固定種子：

```bash
# 開發階段：使用種子
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented_test \
  --preset medium \
  --seed 42

# 生產階段：移除種子（隨機）
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented_prod \
  --preset medium
```

### 4. 調整每張影像的增強數量

根據資料集大小選擇適當的倍增因子：

| 原始資料大小 | 建議倍增 | num-per-image |
|------------|---------|---------------|
| < 100 張/類別 | 5-10x | 5-10 |
| 100-500 張/類別 | 3-5x | 3-5 |
| 500-1000 張/類別 | 2-3x | 2-3 |
| > 1000 張/類別 | 1-2x | 1-2 |

### 5. 領域特定考量

**自然影像（攝影）**：
- 使用 Medium 或 Strong 預設
- 啟用所有幾何變換
- 適度的色彩調整

**醫學影像**：
- 使用 Light 或自訂配置
- 小角度旋轉（±5-10°）
- 保守的色彩調整
- 使用無損格式（PNG）
- 避免強烈的色調偏移

**衛星影像**：
- 使用 Medium 或 Strong 預設
- 完整旋轉範圍（0-360°）
- 所有方向的翻轉
- 適度的模糊效果

**人臉辨識**：
- 使用 Light 或 Medium 預設
- 小角度旋轉（±15°）
- 僅水平翻轉
- 避免垂直翻轉
- 保守的色彩調整

### 6. 品質控制

**檢查點**：

1. **視覺檢查**：隨機抽樣檢查輸出影像
2. **統計驗證**：檢查 `augmentation_stats.json`
3. **檔案大小**：確保輸出檔案大小合理
4. **錯誤日誌**：檢查是否有失敗的影像

**範例檢查腳本**：
```bash
# 檢查輸出統計
cat ./augmented/augmentation_stats.json

# 隨機視覺檢查
ls ./augmented/*.jpg | shuf -n 10

# 檢查平均檔案大小
du -sh ./augmented
ls ./augmented/*.jpg | wc -l
```

### 7. 避免過度增強

**警告信號**：
- 輸出影像看起來不自然
- 影像品質明顯下降
- 過多的模糊或噪點
- 顏色失真嚴重

**解決方案**：
- 降低增強的機率值
- 使用更輕的預設
- 減少每張影像的增強數量
- 調整參數範圍

### 8. 平衡類別分佈

如果資料集類別不平衡，對少數類別使用更強的增強：

```bash
# 多數類別：輕量增強
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./majority_class \
  --output-dir ./augmented/majority_class \
  --preset light \
  --num-per-image 2

# 少數類別：強力增強
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./minority_class \
  --output-dir ./augmented/minority_class \
  --preset strong \
  --num-per-image 8
```

## 故障排除 (Troubleshooting)

### 常見問題

#### 1. 增強後的影像看起來太不同

**症狀**：輸出影像與原始影像差異過大

**原因**：增強太強或機率太高

**解決方案**：
```yaml
# 調整配置
augmentations:
  - name: "random_brightness"
    type: "color"
    probability: 0.5  # 從 0.8 降低到 0.5
    parameters:
      range: [0.9, 1.1]  # 從 [0.7, 1.3] 收窄到 [0.9, 1.1]
```

#### 2. 增強效果不明顯

**症狀**：輸出影像與原始影像幾乎相同

**原因**：增強太弱或機率太低

**解決方案**：
```yaml
# 調整配置
augmentations:
  - name: "random_brightness"
    type: "color"
    probability: 0.8  # 從 0.3 提高到 0.8
    parameters:
      range: [0.7, 1.3]  # 從 [0.95, 1.05] 擴大到 [0.7, 1.3]
```

#### 3. 輸出影像有明顯偽影

**症狀**：影像出現不自然的偽影或瑕疵

**原因**：模糊或噪點過強，或 JPEG 品質太低

**解決方案**：
```yaml
# 調整配置
augmentations:
  # 減少模糊強度
  - name: "random_blur"
    type: "blur"
    probability: 0.2
    parameters:
      radius_range: [0.5, 1.0]  # 從 [1.0, 3.0] 降低

  # 減少噪點
  - name: "gaussian_noise"
    type: "noise"
    probability: 0.1
    parameters:
      std_range: [5, 10]  # 從 [10, 20] 降低

output:
  quality: 95  # 從 85 提高到 95
```

#### 4. 處理速度太慢

**症狀**：批次處理花費過多時間

**原因**：批次大小太小，工作進程太少，或增強太複雜

**解決方案**：
```yaml
processing:
  num_workers: 8  # 從 4 增加到 8
  batch_size: 32  # 從 8 增加到 32

# 或減少複雜的增強
augmentations:
  # 移除或降低機率
  - name: "cutout"
    type: "advanced"
    probability: 0.0  # 從 0.3 降到 0.0（停用）
```

#### 5. 記憶體不足錯誤

**症狀**：處理大影像時記憶體不足

**原因**：批次大小太大，影像太大，或工作進程太多

**解決方案**：
```yaml
processing:
  num_workers: 2  # 從 8 降低到 2
  batch_size: 4  # 從 32 降低到 4

validation:
  max_image_size: [2048, 2048]  # 限制最大尺寸
```

或者在命令行中調整：
```bash
# 處理前調整影像大小
mogrify -resize 1024x1024 ./images/*.jpg

# 然後運行增強
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir ./images \
  --output-dir ./augmented \
  --preset medium
```

#### 6. 某些影像被跳過

**症狀**：統計顯示一些影像失敗

**原因**：影像格式不支援，檔案損壞，或尺寸不符

**解決方案**：
```bash
# 檢查日誌
grep "WARNING\|ERROR" logs/data_augmentation_*.log

# 驗證輸入影像
python -c "
from PIL import Image
from pathlib import Path

for img_path in Path('./images').glob('*.jpg'):
    try:
        img = Image.open(img_path)
        img.verify()
        print(f'✓ {img_path.name}: OK')
    except Exception as e:
        print(f'✗ {img_path.name}: {e}')
"
```

### 錯誤訊息

#### "Image too small"

**原因**：影像尺寸小於 `min_image_size`

**解決方案**：
```yaml
validation:
  min_image_size: [32, 32]  # 降低最小尺寸要求
```

#### "Image too large"

**原因**：影像尺寸大於 `max_image_size`

**解決方案**：
```yaml
validation:
  max_image_size: [8192, 8192]  # 提高最大尺寸限制
```

或預先調整影像大小：
```bash
mogrify -resize 4096x4096\> ./images/*.jpg
```

#### "Unsupported format"

**原因**：影像格式不在 `allowed_formats` 中

**解決方案**：
```yaml
validation:
  allowed_formats: ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
```

或轉換格式：
```bash
mogrify -format jpg ./images/*.bmp
```

## 輸出格式 (Output Format)

### 批次處理輸出結構

```
output_dir/
├── original_name_aug00.jpg
├── original_name_aug01.jpg
├── original_name_aug02.jpg
├── another_image_aug00.jpg
├── another_image_aug01.jpg
└── augmentation_stats.json
```

### 統計檔案格式

`augmentation_stats.json` 包含以下資訊：

```json
{
  "total_images": 100,
  "total_augmentations": 300,
  "successful": 298,
  "failed": 2,
  "augmentations_per_image": 3,
  "output_dir": "/path/to/output",
  "processing_time": 45.23,
  "timestamp": "2025-12-02T10:30:00"
}
```

**欄位說明**：
- `total_images`: 處理的原始影像總數
- `total_augmentations`: 嘗試產生的增強版本總數
- `successful`: 成功產生的增強版本數
- `failed`: 失敗的增強版本數
- `augmentations_per_image`: 每張影像的增強數量
- `output_dir`: 輸出目錄路徑
- `processing_time`: 處理時間（秒）
- `timestamp`: 處理時間戳記

## API 參考 (API Reference)

如果你想在 Python 程式碼中使用資料增強功能：

### 基本使用

```python
from pathlib import Path
from scripts.automation.scenarios.data_augmentation import (
    DataAugmentationTool,
    get_preset_augmentations
)

# 建立工具實例
tool = DataAugmentationTool()

# 取得預設配置
augmentations = get_preset_augmentations('medium')

# 增強單張影像
result = tool.augment_single(
    input_path=Path('/path/to/input.jpg'),
    output_path=Path('/path/to/output.jpg'),
    augmentations=augmentations,
    seed=42
)

if result.success:
    print(f"Success! Applied: {result.augmentations_applied}")
else:
    print(f"Failed: {result.error}")
```

### 批次處理

```python
from pathlib import Path
from scripts.automation.scenarios.data_augmentation import (
    DataAugmentationTool,
    get_preset_augmentations
)

# 建立工具實例
tool = DataAugmentationTool()

# 取得預設配置
augmentations = get_preset_augmentations('strong')

# 批次增強
stats = tool.augment_batch(
    input_dir=Path('/path/to/images'),
    output_dir=Path('/path/to/augmented'),
    augmentations=augmentations,
    num_per_image=5,
    seed=42
)

print(f"Processed: {stats['total_images']} images")
print(f"Generated: {stats['total_augmentations']} augmentations")
print(f"Success rate: {stats['successful']/stats['total_augmentations']*100:.1f}%")
```

### 自訂增強管道

```python
from scripts.automation.scenarios.data_augmentation import (
    AugmentationConfig,
    AugmentationType,
    AugmentationPipeline
)
from PIL import Image

# 定義自訂增強
custom_augmentations = [
    AugmentationConfig(
        name='random_brightness',
        type=AugmentationType.COLOR,
        probability=0.7,
        parameters={'range': [0.8, 1.2]}
    ),
    AugmentationConfig(
        name='horizontal_flip',
        type=AugmentationType.GEOMETRIC,
        probability=0.5,
        parameters={}
    ),
    AugmentationConfig(
        name='rotate',
        type=AugmentationType.GEOMETRIC,
        probability=0.4,
        parameters={'angle_range': [-20, 20], 'expand': True}
    ),
]

# 建立管道
pipeline = AugmentationPipeline(custom_augmentations, seed=42)

# 應用到影像
image = Image.open('/path/to/image.jpg')
augmented_image, applied = pipeline.apply(image)

print(f"Applied augmentations: {applied}")
augmented_image.save('/path/to/output.jpg', 'JPEG', quality=95)
```

## 與其他 Phase 3 元件整合

### 與 Dataset Builder 整合

在建構資料集之前增強影像：

```bash
# 步驟 1：增強影像
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir /path/to/original_images \
  --output-dir /path/to/augmented_images \
  --preset medium \
  --num-per-image 3

# 步驟 2：使用 Dataset Builder 組織
python scripts/automation/scenarios/dataset_builder.py organize \
  --source-dirs /path/to/original_images /path/to/augmented_images \
  --output-dir /path/to/dataset \
  --strategy by_class

# 步驟 3：劃分資料集
python scripts/automation/scenarios/dataset_builder.py split \
  --input-dir /path/to/dataset \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 與 Annotation Tool Integration 整合

增強已標註的影像並保持標註：

```bash
# 步驟 1：匯出標註
python scripts/automation/scenarios/annotation_tool.py export \
  --input /path/to/annotations.json \
  --format coco \
  --output /path/to/coco_annotations.json

# 步驟 2：增強影像（注意：需要相應調整標註）
python scripts/automation/scenarios/data_augmentation.py batch \
  --input-dir /path/to/images \
  --output-dir /path/to/augmented \
  --preset light \
  --num-per-image 2

# 注意：幾何變換（如旋轉、翻轉）需要相應調整邊界框座標
# 建議：對於需要保持標註的任務，使用僅包含色彩調整的自訂配置
```

**自訂配置（保持標註）**：
```yaml
# annotation_safe_augmentation.yaml
metadata:
  name: "Annotation-Safe Augmentation"
  description: "Only color adjustments to preserve annotations"

augmentations:
  # 僅色彩調整，不改變空間配置
  - name: "random_brightness"
    type: "color"
    probability: 0.7
    parameters:
      range: [0.8, 1.2]

  - name: "random_contrast"
    type: "color"
    probability: 0.7
    parameters:
      range: [0.8, 1.2]

  - name: "random_saturation"
    type: "color"
    probability: 0.5
    parameters:
      range: [0.8, 1.2]

  # 不包含任何幾何變換
```

## 效能考量 (Performance Considerations)

### 處理速度

影響處理速度的因素：

1. **影像大小**：更大的影像需要更多時間
2. **增強複雜度**：複雜的增強（如 cutout）比簡單的（如翻轉）慢
3. **批次大小**：更大的批次更有效率
4. **工作進程數**：更多工作進程可以並行處理

### 記憶體使用

記憶體使用由以下因素決定：

1. **影像大小**：更大的影像需要更多記憶體
2. **批次大小**：更大的批次需要更多記憶體
3. **工作進程數**：更多工作進程需要更多記憶體

### 最佳化建議

**小影像（< 512x512）**：
```yaml
processing:
  num_workers: 8
  batch_size: 32
```

**中等影像（512x512 到 1024x1024）**：
```yaml
processing:
  num_workers: 4
  batch_size: 16
```

**大影像（> 1024x1024）**：
```yaml
processing:
  num_workers: 2
  batch_size: 8
```

**超大影像（> 2048x2048）**：
```yaml
processing:
  num_workers: 1
  batch_size: 4
```

## 限制與注意事項 (Limitations and Notes)

### 當前限制

1. **CPU 運算**：所有處理在 CPU 上進行（無 GPU 加速）
2. **影像格式**：僅支援常見的影像格式（JPG, PNG, WebP等）
3. **無批次標註更新**：幾何變換不會自動更新對應的標註
4. **無影像品質評估**：不會自動評估增強後的影像品質

### 計劃中的改進

1. **GPU 加速**：支援 GPU 加速的增強（使用 Kornia 或 Albumentations）
2. **更多增強類型**：
   - Elastic deformation（彈性變形）
   - Grid distortion（網格扭曲）
   - Optical distortion（光學扭曲）
   - Perspective transform（透視變換）
3. **標註保持**：自動調整幾何變換後的標註座標
4. **品質評估**：自動評估增強後的影像品質
5. **增強策略學習**：使用 AutoAugment 或 RandAugment 自動學習最佳增強策略

## 參考資源 (References)

### 內部文件

- [Dataset Builder Documentation](./PHASE3_DATASET_BUILDER.md)
- [Annotation Tool Integration Documentation](./PHASE3_ANNOTATION_TOOL.md)
- [Configuration Files](../../configs/automation/augmentation/README.md)

### 工具實作

- [Data Augmentation Tool](../../scripts/automation/scenarios/data_augmentation.py)
- [Test Suite](/tmp/test_data_augmentation.py)

### 外部參考

1. **研究論文**：
   - "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
   - "Data Augmentation Generative Adversarial Networks" (DAGAN)
   - "AutoAugment: Learning Augmentation Strategies from Data"
   - "RandAugment: Practical automated data augmentation"

2. **Python 函式庫**：
   - [PIL/Pillow](https://pillow.readthedocs.io/) - 影像處理函式庫
   - [Albumentations](https://albumentations.ai/) - 進階增強函式庫
   - [imgaug](https://imgaug.readthedocs.io/) - 影像增強函式庫
   - [Kornia](https://kornia.readthedocs.io/) - GPU 加速的增強函式庫

3. **最佳實踐指南**：
   - [Data Augmentation Best Practices](https://neptune.ai/blog/data-augmentation-nlp)
   - [Image Augmentation for Deep Learning](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

## 支援與回饋 (Support and Feedback)

如有問題或建議，請：

1. 查看本文件的故障排除章節
2. 檢查配置範例和最佳實踐
3. 查看工具實作的程式碼註解
4. 提交 issue 或 pull request

## 更新日誌 (Changelog)

### v1.0.0 (2025-12-02)

**初始版本**：
- 實作 5 大類增強類型（幾何、色彩、噪點、模糊、進階）
- 提供 3 個預設等級（light, medium, strong）
- 支援單張和批次處理
- 完整的 YAML 配置系統
- 統計報告和錯誤處理
- 決定性增強（種子支援）
- 全面的測試套件（7/7 測試通過）
- 完整的中英文文件

---

**Phase 3 Component 3: Data Augmentation Pipeline - 完成 ✅**

這個元件提供了強大且靈活的資料增強功能，支援從輕量到強力的各種增強策略，並提供完整的配置系統和詳細的文件。與 Dataset Builder 和 Annotation Tool Integration 一起，完成了 Phase 3: Data Processing Automation 的所有核心功能。
