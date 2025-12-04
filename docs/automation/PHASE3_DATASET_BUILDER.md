# Dataset Builder 使用指南

## 目錄

1. [概覽](#概覽)
2. [快速開始](#快速開始)
3. [功能詳解](#功能詳解)
4. [命令行參考](#命令行參考)
5. [配置檔案](#配置檔案)
6. [工作流程範例](#工作流程範例)
7. [最佳實踐](#最佳實踐)
8. [故障排除](#故障排除)
9. [API 參考](#api-參考)

---

## 概覽

### 什麼是 Dataset Builder？

Dataset Builder（資料集建構器）是一個全面的資料集管理工具，專為機器學習和深度學習工作流程設計。它提供了創建、驗證、轉換和管理影像資料集的完整功能。

### 主要功能

#### 1. 資料集創建
- **從目錄創建**：自動掃描目錄結構並組織成資料集
- **ImageFolder 格式**：支援 PyTorch ImageFolder 標準格式
- **扁平結構**：支援無類別標籤的扁平檔案結構
- **遞迴掃描**：可選擇性遞迴搜索子目錄
- **格式支援**：支援 JPEG, PNG, BMP, TIFF, WebP, GIF 等常見格式

#### 2. 資料集分割
- **隨機分割**：將資料集隨機分為 train/val/test
- **分層分割**：保持各類別比例的平衡分割
- **K-fold 交叉驗證**：自動生成 K-fold 分割
- **自訂比例**：完全可配置的分割比例
- **可重現性**：支援隨機種子確保結果可重現

#### 3. 元數據生成
- **統計資訊**：自動計算影像尺寸、格式、檔案大小統計
- **類別分佈**：詳細的類別計數和分佈資訊
- **分割統計**：各分割的影像數量和比例
- **JSON 輸出**：結構化的 JSON 格式元數據
- **人類可讀摘要**：生成易讀的文字摘要檔案

#### 4. 資料集驗證
- **影像完整性**：檢測損壞或無法讀取的影像
- **格式驗證**：驗證影像格式和屬性
- **品質檢查**：可選的解析度和檔案大小檢查
- **詳細報告**：生成包含所有問題的驗證報告
- **批次處理**：高效的批次驗證處理

#### 5. 資料集操作
- **合併資料集**：合併多個資料集為一個
- **提取子集**：根據類別、分割或數量提取子集
- **重新分割**：調整現有資料集的分割比例
- **類別過濾**：選擇性保留或排除特定類別

### 技術特點

- **純 CPU 處理**：無需 GPU，適合任何環境
- **記憶體效率**：針對大型資料集優化的記憶體使用
- **平行處理**：支援多執行緒加速處理
- **安全機制**：整合 Phase 1 安全基礎設施
- **跨平台**：支援 Windows, Linux, macOS
- **可擴展性**：模組化設計，易於擴展功能

### 使用場景

1. **機器學習專案準備**
   - 組織訓練資料
   - 創建驗證和測試集
   - 生成資料集文件

2. **資料品質控制**
   - 驗證影像完整性
   - 檢測損壞檔案
   - 確保格式一致性

3. **實驗管理**
   - 快速創建不同的資料子集
   - 測試不同的資料分割策略
   - 複製和版本控制資料集

4. **資料集維護**
   - 合併新收集的資料
   - 更新資料集統計資訊
   - 重組資料集結構

---

## 快速開始

### 安裝需求

```bash
# 確保已安裝必要的 Python 套件
pip install Pillow numpy pyyaml

# 或者使用專案的環境
conda activate ai_env
```

### 基本使用

#### 1. 從目錄創建資料集

```bash
# 基本用法：從 ImageFolder 格式創建資料集
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /path/to/images \
  --output-dir /path/to/dataset \
  --skip-preflight

# 輸入目錄結構 (ImageFolder 格式):
# /path/to/images/
#   ├── class_a/
#   │   ├── img1.jpg
#   │   ├── img2.jpg
#   │   └── ...
#   ├── class_b/
#   │   ├── img1.jpg
#   │   └── ...
#   └── class_c/
#       └── ...

# 輸出結構：
# /path/to/dataset/
#   ├── train/
#   │   ├── class_a/
#   │   ├── class_b/
#   │   └── class_c/
#   ├── val/
#   │   ├── class_a/
#   │   ├── class_b/
#   │   └── class_c/
#   ├── metadata.json
#   ├── statistics.json
#   └── dataset_info.txt
```

#### 2. 自訂分割比例

```bash
# 70% 訓練集, 20% 驗證集, 10% 測試集（分層分割）
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /path/to/images \
  --output-dir /path/to/dataset \
  --split-ratio 0.7 0.2 0.1 \
  --stratify \
  --skip-preflight
```

#### 3. 驗證資料集

```bash
# 驗證資料集的完整性
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /path/to/dataset \
  --check-images \
  --skip-preflight

# 輸出：
# ✓ Validation complete
#   Valid: 1000/1000
#   Corrupted: 0
#   Warnings: 0
```

#### 4. 提取子集

```bash
# 提取特定類別的子集
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /path/to/dataset \
  --output-dir /path/to/subset \
  --classes cat dog bird \
  --skip-preflight
```

#### 5. 合併資料集

```bash
# 合併多個資料集
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
  --output-dir /path/to/merged_dataset \
  --skip-preflight
```

### 第一個完整範例

```bash
# 步驟 1: 創建基礎資料集
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir ~/data/animals \
  --output-dir ~/datasets/animals_v1 \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight

# 步驟 2: 驗證資料集
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir ~/datasets/animals_v1 \
  --check-images \
  --skip-preflight

# 步驟 3: 提取訓練用子集（只保留貓和狗）
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir ~/datasets/animals_v1 \
  --output-dir ~/datasets/cats_dogs_only \
  --classes cat dog \
  --skip-preflight

# 完成！現在你有：
# - 完整的動物資料集 (animals_v1)
# - 只有貓狗的子集 (cats_dogs_only)
# - 兩者都有完整的驗證報告
```

---

## 功能詳解

### 1. 資料集創建

#### 1.1 從目錄創建 (create-from-dir)

**功能描述**

從檔案系統目錄創建結構化的機器學習資料集。支援 ImageFolder 格式（資料夾名稱作為類別標籤）和扁平結構（無類別標籤）。

**參數說明**

| 參數 | 類型 | 必需 | 預設值 | 說明 |
|------|------|------|--------|------|
| `--input-dir` | string | 是 | - | 輸入影像目錄路徑 |
| `--output-dir` | string | 是 | - | 輸出資料集目錄路徑 |
| `--format` | string | 否 | imagefolder | 資料集格式 (imagefolder/flat) |
| `--split-ratio` | float[] | 否 | None | 分割比例 [train, val, test] |
| `--stratify` | flag | 否 | False | 是否使用分層分割 |
| `--min-images-per-class` | int | 否 | 1 | 每個類別的最小影像數量 |
| `--recursive` | flag | 否 | True | 是否遞迴掃描子目錄 |
| `--seed` | int | 否 | 42 | 隨機種子（可重現性） |

**使用範例**

```bash
# 範例 1: 基本創建（不分割）
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir /data/datasets/my_dataset \
  --skip-preflight

# 範例 2: 創建並分割（80/10/10，分層）
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir /data/datasets/my_dataset \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight

# 範例 3: 過濾小類別（每類至少 10 張影像）
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir /data/datasets/my_dataset \
  --min-images-per-class 10 \
  --skip-preflight

# 範例 4: 扁平結構（無類別標籤）
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/unlabeled_images \
  --output-dir /data/datasets/unlabeled_dataset \
  --format flat \
  --skip-preflight

# 範例 5: 自訂隨機種子
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir /data/datasets/my_dataset \
  --split-ratio 0.7 0.15 0.15 \
  --stratify \
  --seed 12345 \
  --skip-preflight
```

**輸出結構**

```
output_dir/
├── train/                      # 訓練集（如果分割）
│   ├── class_a/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── class_b/
│   │   └── ...
│   └── ...
├── val/                        # 驗證集（如果分割）
│   ├── class_a/
│   │   └── ...
│   └── ...
├── test/                       # 測試集（如果分割）
│   ├── class_a/
│   │   └── ...
│   └── ...
├── metadata.json               # 資料集元數據
├── statistics.json             # 統計資訊
└── dataset_info.txt            # 人類可讀摘要
```

**元數據檔案範例 (metadata.json)**

```json
{
  "created_at": "2025-12-02T10:30:00",
  "format": "imagefolder",
  "total_images": 1000,
  "num_classes": 5,
  "classes": ["cat", "dog", "bird", "fish", "hamster"],
  "class_to_idx": {
    "cat": 0,
    "dog": 1,
    "bird": 2,
    "fish": 3,
    "hamster": 4
  },
  "splits": {
    "train": 700,
    "val": 200,
    "test": 100
  },
  "split_ratio": [0.7, 0.2, 0.1],
  "stratified": true,
  "seed": 42
}
```

**統計資訊範例 (statistics.json)**

```json
{
  "class_distribution": {
    "cat": 200,
    "dog": 200,
    "bird": 200,
    "fish": 200,
    "hamster": 200
  },
  "split_distribution": {
    "train": 700,
    "val": 200,
    "test": 100
  },
  "format_distribution": {
    "JPEG": 850,
    "PNG": 150
  },
  "size_statistics": {
    "width": {
      "mean": 512.5,
      "std": 128.3,
      "min": 256,
      "max": 1024
    },
    "height": {
      "mean": 512.5,
      "std": 128.3,
      "min": 256,
      "max": 1024
    }
  },
  "file_size_statistics": {
    "mean_bytes": 245678,
    "std_bytes": 98234,
    "min_bytes": 50000,
    "max_bytes": 1500000
  }
}
```

#### 1.2 分層分割策略

**什麼是分層分割？**

分層分割（Stratified Split）確保在分割資料集時，每個分割（train/val/test）中各類別的比例與原始資料集保持一致。這對於類別不平衡的資料集特別重要。

**範例說明**

假設有以下資料集：
- Cat: 100 張影像
- Dog: 200 張影像
- Bird: 50 張影像

**非分層分割（隨機）**：可能導致不均衡
```
Train (70%): Cat: 65, Dog: 150, Bird: 30  (比例不同)
Val (20%):   Cat: 22, Dog: 35,  Bird: 13  (比例不同)
Test (10%):  Cat: 13, Dog: 15,  Bird: 7   (比例不同)
```

**分層分割**：保持比例一致
```
Train (70%): Cat: 70, Dog: 140, Bird: 35  (28.6% / 57.1% / 14.3%)
Val (20%):   Cat: 20, Dog: 40,  Bird: 10  (28.6% / 57.1% / 14.3%)
Test (10%):  Cat: 10, Dog: 20,  Bird: 5   (28.6% / 57.1% / 14.3%)
```

**何時使用分層分割？**

✅ **應該使用**：
- 類別不平衡的資料集
- 小型資料集（每類樣本數較少）
- 需要確保驗證/測試集代表性的情況
- 分類任務（尤其是多類別分類）

❌ **可以不使用**：
- 類別完全平衡的資料集
- 大型資料集（每類樣本數很多）
- 迴歸任務或無監督學習
- 時間序列資料（需按時間順序分割）

#### 1.3 K-fold 交叉驗證

**功能描述**

自動生成 K-fold 交叉驗證的資料集分割，適合模型評估和超參數調優。

**使用範例**

```bash
# 創建 5-fold 交叉驗證資料集
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir /data/datasets/kfold_dataset \
  --kfold 5 \
  --stratify \
  --skip-preflight
```

**輸出結構**

```
output_dir/
├── fold_1/
│   ├── train/
│   │   ├── class_a/
│   │   └── ...
│   └── val/
│       ├── class_a/
│       └── ...
├── fold_2/
│   ├── train/
│   └── val/
├── fold_3/
│   ├── train/
│   └── val/
├── fold_4/
│   ├── train/
│   └── val/
├── fold_5/
│   ├── train/
│   └── val/
├── kfold_metadata.json
└── kfold_statistics.json
```

### 2. 資料集驗證

#### 2.1 完整性驗證 (validate)

**功能描述**

驗證資料集中的影像完整性，檢測損壞、無法讀取或格式錯誤的檔案。

**參數說明**

| 參數 | 類型 | 必需 | 預設值 | 說明 |
|------|------|------|--------|------|
| `--dataset-dir` | string | 是 | - | 資料集目錄路徑 |
| `--check-images` | flag | 否 | False | 是否深度檢查每張影像 |
| `--min-width` | int | 否 | None | 最小寬度要求 |
| `--min-height` | int | 否 | None | 最小高度要求 |
| `--max-width` | int | 否 | None | 最大寬度限制 |
| `--max-height` | int | 否 | None | 最大高度限制 |
| `--allowed-formats` | string[] | 否 | All | 允許的影像格式 |

**使用範例**

```bash
# 範例 1: 基本驗證（只檢查檔案存在）
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /data/datasets/my_dataset \
  --skip-preflight

# 範例 2: 深度驗證（檢查影像可讀性）
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /data/datasets/my_dataset \
  --check-images \
  --skip-preflight

# 範例 3: 驗證解析度要求（至少 256x256）
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /data/datasets/my_dataset \
  --check-images \
  --min-width 256 \
  --min-height 256 \
  --skip-preflight

# 範例 4: 限制最大解析度（不超過 1024x1024）
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /data/datasets/my_dataset \
  --check-images \
  --max-width 1024 \
  --max-height 1024 \
  --skip-preflight

# 範例 5: 限制影像格式（只允許 JPEG 和 PNG）
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /data/datasets/my_dataset \
  --check-images \
  --allowed-formats jpg jpeg png \
  --skip-preflight
```

**驗證報告範例**

```
✓ Validation complete
  Valid: 980/1000
  Corrupted: 15
  Warnings: 5
  Errors: 0

Corrupted Images:
  - train/cat/img_045.jpg: Cannot identify image file
  - train/dog/img_123.jpg: Truncated image
  - val/bird/img_067.jpg: Image file is truncated
  ... (12 more)

Warnings:
  - test/cat/img_234.jpg: Image size too small (128x128 < 256x256)
  - test/dog/img_456.jpg: Unusual aspect ratio (4:1)
  ... (3 more)
```

**驗證報告檔案 (validation_report.json)**

```json
{
  "validated_at": "2025-12-02T11:00:00",
  "total_images": 1000,
  "valid_images": 980,
  "corrupted_images": 15,
  "warnings": 5,
  "errors": 0,
  "corrupted_files": [
    {
      "path": "train/cat/img_045.jpg",
      "reason": "Cannot identify image file",
      "error_type": "PIL.UnidentifiedImageError"
    },
    {
      "path": "train/dog/img_123.jpg",
      "reason": "Truncated image",
      "error_type": "PIL.Image.TruncatedFile"
    }
  ],
  "warnings": [
    {
      "path": "test/cat/img_234.jpg",
      "reason": "Image size too small (128x128 < 256x256)",
      "severity": "warning"
    }
  ]
}
```

#### 2.2 常見驗證問題

**問題 1: 損壞的 JPEG 檔案**

**症狀**：
```
PIL.Image.TruncatedFile: broken data stream when reading image file
```

**原因**：
- 檔案傳輸中斷
- 儲存空間不足導致寫入不完整
- 檔案系統錯誤

**解決方法**：
1. 重新下載或複製檔案
2. 使用影像修復工具（如 `jpegoptim`）
3. 從備份中恢復
4. 從資料集中移除損壞的檔案

**問題 2: 無法識別的影像格式**

**症狀**：
```
PIL.UnidentifiedImageError: cannot identify image file
```

**原因**：
- 檔案副檔名與實際格式不符
- 非標準或損壞的檔案頭
- 實際上不是影像檔案

**解決方法**：
1. 檢查檔案的真實格式：`file image.jpg`
2. 重新命名副檔名以匹配實際格式
3. 使用影像轉換工具重新儲存
4. 從資料集中移除無效檔案

**問題 3: 解析度不符合要求**

**症狀**：
```
Warning: Image size too small (128x128 < 256x256)
```

**原因**：
- 影像解析度低於訓練要求
- 縮圖誤被包含在資料集中

**解決方法**：
1. 使用超解析度工具放大影像
2. 從資料集中移除過小的影像
3. 調整最小解析度要求
4. 使用 padding 補齊影像

### 3. 資料集操作

#### 3.1 合併資料集 (merge)

**功能描述**

將多個資料集合併為一個統一的資料集，自動處理類別衝突和檔案重複。

**參數說明**

| 參數 | 類型 | 必需 | 預設值 | 說明 |
|------|------|------|--------|------|
| `--dataset-dirs` | string[] | 是 | - | 要合併的資料集目錄列表 |
| `--output-dir` | string | 是 | - | 輸出合併資料集目錄 |
| `--handle-duplicates` | string | 否 | skip | 重複處理策略 (skip/rename/overwrite) |
| `--merge-splits` | flag | 否 | False | 是否保留原始分割 |
| `--resplit` | flag | 否 | False | 合併後重新分割 |
| `--split-ratio` | float[] | 否 | None | 重新分割的比例 |

**使用範例**

```bash
# 範例 1: 基本合併（跳過重複檔案）
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /data/dataset1 /data/dataset2 /data/dataset3 \
  --output-dir /data/merged_dataset \
  --skip-preflight

# 範例 2: 合併並重命名重複檔案
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /data/dataset1 /data/dataset2 \
  --output-dir /data/merged_dataset \
  --handle-duplicates rename \
  --skip-preflight

# 範例 3: 合併並重新分割
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /data/dataset1 /data/dataset2 \
  --output-dir /data/merged_dataset \
  --resplit \
  --split-ratio 0.8 0.1 0.1 \
  --skip-preflight

# 範例 4: 保留原始分割結構
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /data/dataset1 /data/dataset2 \
  --output-dir /data/merged_dataset \
  --merge-splits \
  --skip-preflight
```

**合併策略說明**

**1. Skip（跳過）**
- 如果檔案名稱重複，保留第一個遇到的檔案
- 適合：確信各資料集無重複內容時
- 風險：可能遺失部分資料

**2. Rename（重命名）**
- 為重複的檔案名稱添加後綴（如 `img_001_2.jpg`）
- 適合：各資料集可能有同名但不同內容的檔案
- 風險：檔案名稱可能變得很長

**3. Overwrite（覆寫）**
- 後續資料集的檔案覆蓋先前的同名檔案
- 適合：後續資料集的版本更新或品質更好
- 風險：可能遺失原始資料

#### 3.2 提取子集 (extract-subset)

**功能描述**

從現有資料集中提取子集，支援按類別、分割或樣本數量篩選。

**參數說明**

| 參數 | 類型 | 必需 | 預設值 | 說明 |
|------|------|------|--------|------|
| `--dataset-dir` | string | 是 | - | 原始資料集目錄 |
| `--output-dir` | string | 是 | - | 輸出子集目錄 |
| `--classes` | string[] | 否 | None | 要提取的類別列表 |
| `--splits` | string[] | 否 | None | 要提取的分割 (train/val/test) |
| `--max-samples` | int | 否 | None | 每個類別的最大樣本數 |
| `--min-samples` | int | 否 | None | 每個類別的最小樣本數 |
| `--sample-ratio` | float | 否 | None | 採樣比例 (0.0-1.0) |
| `--stratify` | flag | 否 | False | 是否分層採樣 |

**使用範例**

```bash
# 範例 1: 提取特定類別
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/full_dataset \
  --output-dir /data/subset_animals \
  --classes cat dog bird \
  --skip-preflight

# 範例 2: 只提取訓練集
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/full_dataset \
  --output-dir /data/trainset_only \
  --splits train \
  --skip-preflight

# 範例 3: 限制每類樣本數（最多 100 張）
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/full_dataset \
  --output-dir /data/small_subset \
  --max-samples 100 \
  --stratify \
  --skip-preflight

# 範例 4: 採樣 20% 的資料
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/full_dataset \
  --output-dir /data/subset_20pct \
  --sample-ratio 0.2 \
  --stratify \
  --skip-preflight

# 範例 5: 組合條件（特定類別 + 特定分割 + 限制數量）
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/full_dataset \
  --output-dir /data/custom_subset \
  --classes cat dog \
  --splits train val \
  --max-samples 50 \
  --skip-preflight
```

**使用場景範例**

**場景 1: 快速原型驗證**
```bash
# 創建小型子集用於快速測試
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/imagenet \
  --output-dir /data/imagenet_mini \
  --sample-ratio 0.01 \
  --stratify \
  --skip-preflight

# 結果：從 ImageNet 提取 1% 的樣本，保持類別平衡
```

**場景 2: 特定任務資料集**
```bash
# 為寵物分類任務提取相關類別
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/animals_full \
  --output-dir /data/pets_only \
  --classes cat dog hamster rabbit guinea_pig \
  --skip-preflight
```

**場景 3: 平衡不平衡資料集**
```bash
# 限制每類最大樣本數以平衡資料集
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir /data/imbalanced_dataset \
  --output-dir /data/balanced_dataset \
  --max-samples 500 \
  --stratify \
  --skip-preflight

# 結果：每類最多 500 張影像，達到類別平衡
```

---

## 命令行參考

### 全域參數

所有命令都支援以下全域參數：

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--skip-preflight` | 跳過系統預檢查（測試模式） | False |
| `--log-level` | 日誌等級 (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--log-file` | 日誌檔案路徑 | None |
| `--quiet` | 安靜模式（只輸出錯誤） | False |
| `--verbose` | 詳細模式（輸出所有資訊） | False |

### 命令 1: create-from-dir

**用途**：從目錄創建資料集

**語法**：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir INPUT_DIR \
  --output-dir OUTPUT_DIR \
  [OPTIONS]
```

**必需參數**：
- `--input-dir PATH`：輸入影像目錄
- `--output-dir PATH`：輸出資料集目錄

**可選參數**：
- `--format {imagefolder,flat}`：資料集格式（預設：imagefolder）
- `--split-ratio FLOAT FLOAT FLOAT`：分割比例（如 0.7 0.2 0.1）
- `--stratify`：使用分層分割
- `--kfold INT`：K-fold 交叉驗證的 fold 數
- `--min-images-per-class INT`：每類最小影像數（預設：1）
- `--recursive`：遞迴掃描子目錄（預設：True）
- `--seed INT`：隨機種子（預設：42）

**範例**：
```bash
# 基本創建
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir ~/data/images \
  --output-dir ~/datasets/my_dataset \
  --skip-preflight

# 創建並分層分割
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir ~/data/images \
  --output-dir ~/datasets/my_dataset \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight

# 5-fold 交叉驗證
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir ~/data/images \
  --output-dir ~/datasets/my_dataset \
  --kfold 5 \
  --stratify \
  --skip-preflight
```

### 命令 2: validate

**用途**：驗證資料集完整性

**語法**：
```bash
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir DATASET_DIR \
  [OPTIONS]
```

**必需參數**：
- `--dataset-dir PATH`：資料集目錄路徑

**可選參數**：
- `--check-images`：深度檢查每張影像
- `--min-width INT`：最小寬度要求
- `--min-height INT`：最小高度要求
- `--max-width INT`：最大寬度限制
- `--max-height INT`：最大高度限制
- `--allowed-formats STR [STR ...]`：允許的格式（如 jpg png）
- `--report-file PATH`：驗證報告輸出路徑

**範例**：
```bash
# 基本驗證
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir ~/datasets/my_dataset \
  --skip-preflight

# 深度驗證（檢查影像可讀性）
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir ~/datasets/my_dataset \
  --check-images \
  --skip-preflight

# 驗證解析度和格式
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir ~/datasets/my_dataset \
  --check-images \
  --min-width 256 \
  --min-height 256 \
  --allowed-formats jpg jpeg png \
  --skip-preflight
```

### 命令 3: merge

**用途**：合併多個資料集

**語法**：
```bash
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs DIR1 DIR2 [DIR3 ...] \
  --output-dir OUTPUT_DIR \
  [OPTIONS]
```

**必需參數**：
- `--dataset-dirs PATH [PATH ...]`：要合併的資料集目錄列表
- `--output-dir PATH`：輸出合併資料集目錄

**可選參數**：
- `--handle-duplicates {skip,rename,overwrite}`：重複處理策略（預設：skip）
- `--merge-splits`：保留原始分割結構
- `--resplit`：合併後重新分割
- `--split-ratio FLOAT FLOAT FLOAT`：重新分割比例
- `--stratify`：使用分層分割

**範例**：
```bash
# 基本合併
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs ~/datasets/set1 ~/datasets/set2 ~/datasets/set3 \
  --output-dir ~/datasets/merged \
  --skip-preflight

# 合併並重新分割
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs ~/datasets/set1 ~/datasets/set2 \
  --output-dir ~/datasets/merged \
  --resplit \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight
```

### 命令 4: extract-subset

**用途**：提取資料集子集

**語法**：
```bash
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir DATASET_DIR \
  --output-dir OUTPUT_DIR \
  [OPTIONS]
```

**必需參數**：
- `--dataset-dir PATH`：原始資料集目錄
- `--output-dir PATH`：輸出子集目錄

**可選參數**：
- `--classes STR [STR ...]`：要提取的類別列表
- `--splits {train,val,test} [...]`：要提取的分割
- `--max-samples INT`：每類最大樣本數
- `--min-samples INT`：每類最小樣本數
- `--sample-ratio FLOAT`：採樣比例（0.0-1.0）
- `--stratify`：使用分層採樣
- `--seed INT`：隨機種子（預設：42）

**範例**：
```bash
# 提取特定類別
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir ~/datasets/full \
  --output-dir ~/datasets/subset \
  --classes cat dog bird \
  --skip-preflight

# 提取 20% 樣本
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir ~/datasets/full \
  --output-dir ~/datasets/subset \
  --sample-ratio 0.2 \
  --stratify \
  --skip-preflight

# 限制每類樣本數
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir ~/datasets/full \
  --output-dir ~/datasets/subset \
  --max-samples 100 \
  --stratify \
  --skip-preflight
```

---

## 配置檔案

### YAML 配置格式

Dataset Builder 支援使用 YAML 配置檔案來簡化命令行參數。

#### 基本配置範例

**檔案：`configs/dataset/basic_config.yaml`**

```yaml
# Dataset Builder 基本配置

# 資料集創建設定
creation:
  input_dir: /data/raw_images
  output_dir: /data/datasets/my_dataset
  format: imagefolder
  recursive: true
  min_images_per_class: 5

# 分割設定
split:
  enabled: true
  ratio: [0.7, 0.2, 0.1]  # train, val, test
  stratify: true
  seed: 42

# 驗證設定
validation:
  check_images: true
  min_width: 256
  min_height: 256
  max_width: 2048
  max_height: 2048
  allowed_formats:
    - jpg
    - jpeg
    - png

# 處理設定
processing:
  batch_size: 32
  num_workers: 4
  memory_limit_gb: 8
```

**使用方法**：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/dataset/basic_config.yaml \
  --skip-preflight
```

#### 高級配置範例

**檔案：`configs/dataset/advanced_config.yaml`**

```yaml
# Dataset Builder 高級配置

# 專案資訊
project:
  name: "Animal Classification Dataset v2.0"
  description: "Multi-class animal classification dataset"
  version: "2.0.0"
  author: "ML Team"
  created_date: "2025-12-02"

# 資料源設定
sources:
  - path: /data/raw_images/batch1
    weight: 1.0
    enabled: true
  - path: /data/raw_images/batch2
    weight: 1.0
    enabled: true
  - path: /data/raw_images/batch3
    weight: 0.5
    enabled: false  # 暫時停用

# 輸出設定
output:
  base_dir: /data/datasets
  dataset_name: animals_v2
  create_timestamp_dir: false

# 類別過濾
class_filter:
  mode: whitelist  # whitelist 或 blacklist
  classes:
    - cat
    - dog
    - bird
    - fish
    - hamster
  min_samples_per_class: 10
  max_samples_per_class: 1000

# 分割策略
split_strategy:
  method: stratified  # stratified, random, kfold
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  kfold: null  # 如果使用 kfold，設定 fold 數
  seed: 42
  preserve_class_ratio: true

# 影像處理
image_processing:
  resize:
    enabled: false
    width: 512
    height: 512
    keep_aspect_ratio: true

  quality_check:
    enabled: true
    min_resolution: [256, 256]
    max_resolution: [2048, 2048]
    min_file_size_kb: 10
    max_file_size_mb: 10
    allowed_formats: [jpg, jpeg, png, webp]
    check_corruption: true

  augmentation:
    enabled: false  # Dataset Builder 不做增強，交給後續管道

# 元數據設定
metadata:
  generate_statistics: true
  generate_summary: true
  include_sample_images: true
  sample_images_count: 10
  compute_mean_std: true  # 計算資料集 mean 和 std（用於正規化）

# 驗證設定
validation:
  auto_validate: true  # 創建後自動驗證
  deep_check: true
  generate_report: true
  report_format: json  # json, yaml, txt

# 性能設定
performance:
  batch_size: 64
  num_workers: 8
  prefetch_factor: 2
  memory_limit_gb: 16
  use_multiprocessing: true

# 日誌設定
logging:
  level: INFO
  file: logs/dataset_builder.log
  console: true
  rotation: "10 MB"
  retention: "30 days"
```

**使用方法**：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/dataset/advanced_config.yaml \
  --skip-preflight
```

#### K-fold 配置範例

**檔案：`configs/dataset/kfold_config.yaml`**

```yaml
# K-fold 交叉驗證配置

project:
  name: "Medical Images Dataset - 5-Fold CV"
  version: "1.0.0"

creation:
  input_dir: /data/medical_images
  output_dir: /data/datasets/medical_5fold
  format: imagefolder
  recursive: true

split:
  enabled: true
  method: kfold
  kfold: 5
  stratify: true
  seed: 12345

validation:
  check_images: true
  min_width: 512
  min_height: 512
  allowed_formats: [png, tiff]

metadata:
  generate_statistics: true
  compute_mean_std: true
```

#### 子集提取配置範例

**檔案：`configs/dataset/subset_config.yaml`**

```yaml
# 子集提取配置

subset:
  source_dataset: /data/datasets/full_imagenet
  output_dir: /data/datasets/imagenet_subset

  filters:
    classes:
      - n01440764  # tench
      - n01443537  # goldfish
      - n01484850  # great_white_shark
      - n01491361  # tiger_shark
      - n01494475  # hammerhead

    splits:
      - train
      - val

    sampling:
      method: stratified
      max_samples_per_class: 100
      min_samples_per_class: 10
      ratio: null  # 使用 max_samples 而非 ratio
      seed: 42

validation:
  auto_validate: true
  check_images: true

metadata:
  generate_statistics: true
```

### 配置檔案最佳實踐

#### 1. 使用版本控制

```yaml
project:
  name: "My Dataset"
  version: "2.1.0"  # 主要.次要.修補
  changelog:
    - "2.1.0: Added fish and hamster classes"
    - "2.0.0: Reorganized directory structure"
    - "1.0.0: Initial release"
```

#### 2. 環境變數支援

```yaml
creation:
  input_dir: ${DATA_ROOT}/raw_images  # 使用環境變數
  output_dir: ${DATA_ROOT}/datasets/${DATASET_NAME}
```

使用方法：
```bash
export DATA_ROOT=/mnt/data/ai_data
export DATASET_NAME=animals_v2
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/dataset/config.yaml \
  --skip-preflight
```

#### 3. 配置繼承

**基礎配置：`configs/dataset/base.yaml`**

```yaml
# 基礎配置（所有資料集共用）
split:
  seed: 42
  stratify: true

validation:
  check_images: true
  allowed_formats: [jpg, jpeg, png]

performance:
  batch_size: 64
  num_workers: 8
```

**專案配置：`configs/dataset/my_project.yaml`**

```yaml
# 繼承基礎配置並覆寫特定參數
extends: configs/dataset/base.yaml

project:
  name: "My Project Dataset"
  version: "1.0.0"

creation:
  input_dir: /data/my_project/images
  output_dir: /data/datasets/my_project

split:
  ratio: [0.8, 0.1, 0.1]  # 覆寫預設比例
```

---

## 工作流程範例

### 範例 1: 標準分類資料集工作流程

**場景**：為影像分類任務準備資料集

```bash
#!/bin/bash
# prepare_classification_dataset.sh

# 步驟 1: 創建資料集（80/10/10 分割，分層）
echo "Step 1: Creating dataset..."
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images/animals \
  --output-dir /data/datasets/animals_classification \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --seed 42 \
  --skip-preflight

# 步驟 2: 驗證資料集
echo "Step 2: Validating dataset..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /data/datasets/animals_classification \
  --check-images \
  --min-width 224 \
  --min-height 224 \
  --skip-preflight

# 步驟 3: 生成統計報告
echo "Step 3: Generating statistics..."
python scripts/automation/scenarios/dataset_builder.py stats \
  --dataset-dir /data/datasets/animals_classification \
  --output-file /data/datasets/animals_classification/detailed_stats.json \
  --skip-preflight

echo "Dataset preparation complete!"
echo "Location: /data/datasets/animals_classification"
```

### 範例 2: 增量資料集更新工作流程

**場景**：定期添加新資料到現有資料集

```bash
#!/bin/bash
# update_dataset_incremental.sh

DATE=$(date +%Y%m%d)
NEW_DATA_DIR="/data/incoming_images/${DATE}"
MAIN_DATASET="/data/datasets/production_dataset"
TEMP_DATASET="/tmp/new_batch_${DATE}"

# 步驟 1: 處理新批次資料
echo "Processing new batch: ${DATE}"
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir "${NEW_DATA_DIR}" \
  --output-dir "${TEMP_DATASET}" \
  --format imagefolder \
  --skip-preflight

# 步驟 2: 驗證新資料
echo "Validating new data..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${TEMP_DATASET}" \
  --check-images \
  --skip-preflight

# 步驟 3: 合併到主資料集
echo "Merging with main dataset..."
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs "${MAIN_DATASET}" "${TEMP_DATASET}" \
  --output-dir "${MAIN_DATASET}_updated" \
  --handle-duplicates skip \
  --resplit \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight

# 步驟 4: 備份舊版本
echo "Backing up old version..."
mv "${MAIN_DATASET}" "${MAIN_DATASET}_backup_${DATE}"

# 步驟 5: 啟用新版本
mv "${MAIN_DATASET}_updated" "${MAIN_DATASET}"

# 清理臨時檔案
rm -rf "${TEMP_DATASET}"

echo "Dataset update complete!"
```

### 範例 3: K-fold 交叉驗證實驗工作流程

**場景**：使用 K-fold 交叉驗證進行模型選擇

```bash
#!/bin/bash
# kfold_experiment.sh

DATASET_DIR="/data/datasets/kfold_experiment"
K_FOLDS=5

# 步驟 1: 創建 K-fold 資料集
echo "Creating ${K_FOLDS}-fold dataset..."
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir "${DATASET_DIR}" \
  --kfold ${K_FOLDS} \
  --stratify \
  --seed 42 \
  --skip-preflight

# 步驟 2: 驗證每個 fold
echo "Validating all folds..."
for fold in $(seq 1 ${K_FOLDS}); do
    echo "Validating fold ${fold}..."
    python scripts/automation/scenarios/dataset_builder.py validate \
      --dataset-dir "${DATASET_DIR}/fold_${fold}" \
      --check-images \
      --skip-preflight
done

# 步驟 3: 訓練模型（範例 - 需要自行實作訓練腳本）
echo "Training models on each fold..."
for fold in $(seq 1 ${K_FOLDS}); do
    echo "Training on fold ${fold}..."
    # python train.py --train-dir "${DATASET_DIR}/fold_${fold}/train" \
    #                 --val-dir "${DATASET_DIR}/fold_${fold}/val" \
    #                 --output-dir "models/fold_${fold}"
done

echo "K-fold experiment complete!"
```

### 範例 4: 多來源資料集整合工作流程

**場景**：從多個來源整合資料集

```bash
#!/bin/bash
# integrate_multi_source_dataset.sh

OUTPUT_DIR="/data/datasets/integrated_dataset"

# 來源 1: 公開資料集
SOURCE1="/data/public_datasets/imagenet_subset"

# 來源 2: 內部收集的資料
SOURCE2="/data/internal_data/collected_images"

# 來源 3: 網路爬取的資料
SOURCE3="/data/scraped_data/validated_images"

# 步驟 1: 預處理各來源（確保格式一致）
echo "Preprocessing source 1..."
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir "${SOURCE1}" \
  --output-dir /tmp/source1_processed \
  --format imagefolder \
  --skip-preflight

echo "Preprocessing source 2..."
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir "${SOURCE2}" \
  --output-dir /tmp/source2_processed \
  --format imagefolder \
  --skip-preflight

echo "Preprocessing source 3..."
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir "${SOURCE3}" \
  --output-dir /tmp/source3_processed \
  --format imagefolder \
  --skip-preflight

# 步驟 2: 合併所有來源
echo "Merging all sources..."
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /tmp/source1_processed /tmp/source2_processed /tmp/source3_processed \
  --output-dir "${OUTPUT_DIR}" \
  --handle-duplicates rename \
  --resplit \
  --split-ratio 0.7 0.2 0.1 \
  --stratify \
  --skip-preflight

# 步驟 3: 品質檢查
echo "Quality checking..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${OUTPUT_DIR}" \
  --check-images \
  --min-width 256 \
  --min-height 256 \
  --skip-preflight

# 步驟 4: 生成詳細報告
echo "Generating report..."
python scripts/automation/scenarios/dataset_builder.py stats \
  --dataset-dir "${OUTPUT_DIR}" \
  --detailed \
  --output-file "${OUTPUT_DIR}/integration_report.json" \
  --skip-preflight

# 清理臨時檔案
rm -rf /tmp/source*_processed

echo "Multi-source integration complete!"
echo "Final dataset: ${OUTPUT_DIR}"
```

### 範例 5: 快速原型實驗工作流程

**場景**：快速創建小型子集用於演算法原型測試

```bash
#!/bin/bash
# quick_prototype_workflow.sh

FULL_DATASET="/data/datasets/production_full"
PROTOTYPE_DIR="/data/experiments/prototype_$(date +%Y%m%d_%H%M%S)"

# 步驟 1: 提取小型子集（每類 50 張）
echo "Creating prototype subset..."
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --dataset-dir "${FULL_DATASET}" \
  --output-dir "${PROTOTYPE_DIR}/dataset" \
  --max-samples 50 \
  --stratify \
  --seed 42 \
  --skip-preflight

# 步驟 2: 驗證子集
echo "Validating subset..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${PROTOTYPE_DIR}/dataset" \
  --check-images \
  --skip-preflight

# 步驟 3: 生成摘要
echo "Generating summary..."
python scripts/automation/scenarios/dataset_builder.py stats \
  --dataset-dir "${PROTOTYPE_DIR}/dataset" \
  --output-file "${PROTOTYPE_DIR}/dataset_summary.txt" \
  --skip-preflight

# 步驟 4: 創建實驗配置
cat > "${PROTOTYPE_DIR}/experiment_config.yaml" <<EOF
experiment:
  name: "Quick Prototype Experiment"
  date: "$(date +%Y-%m-%d)"
  dataset: "${PROTOTYPE_DIR}/dataset"

hyperparameters:
  learning_rate: 0.001
  batch_size: 32
  epochs: 10

notes: |
  Fast prototyping experiment with reduced dataset.
  Using stratified sampling, 50 samples per class.
EOF

echo "Prototype setup complete!"
echo "Experiment directory: ${PROTOTYPE_DIR}"
```

### 範例 6: 資料集版本管理工作流程

**場景**：管理資料集的多個版本

```bash
#!/bin/bash
# dataset_version_management.sh

DATASET_NAME="animals_classification"
VERSION="v2.0.0"
BASE_DIR="/data/datasets/${DATASET_NAME}"
VERSION_DIR="${BASE_DIR}/${VERSION}"

# 步驟 1: 創建新版本資料集
echo "Creating ${DATASET_NAME} ${VERSION}..."
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images/batch_new \
  --output-dir "${VERSION_DIR}" \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --seed 42 \
  --skip-preflight

# 步驟 2: 驗證新版本
echo "Validating ${VERSION}..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${VERSION_DIR}" \
  --check-images \
  --skip-preflight

# 步驟 3: 生成版本資訊
cat > "${VERSION_DIR}/VERSION_INFO.txt" <<EOF
Dataset: ${DATASET_NAME}
Version: ${VERSION}
Created: $(date +%Y-%m-%d\ %H:%M:%S)
Creator: $(whoami)

Changes in this version:
- Added 500 new images
- Improved class balance
- Fixed corrupted images from v1.x

Statistics:
$(python scripts/automation/scenarios/dataset_builder.py stats \
  --dataset-dir "${VERSION_DIR}" \
  --brief \
  --skip-preflight)
EOF

# 步驟 4: 創建符號連結到最新版本
ln -sfn "${VERSION}" "${BASE_DIR}/latest"

# 步驟 5: 生成變更日誌
echo "${VERSION} - $(date +%Y-%m-%d)" >> "${BASE_DIR}/CHANGELOG.txt"
echo "  - Added 500 new images" >> "${BASE_DIR}/CHANGELOG.txt"
echo "  - Improved class balance" >> "${BASE_DIR}/CHANGELOG.txt"
echo "" >> "${BASE_DIR}/CHANGELOG.txt"

echo "Version ${VERSION} created successfully!"
echo "Location: ${VERSION_DIR}"
echo "Latest link: ${BASE_DIR}/latest -> ${VERSION}"
```

### 範例 7: 自動化品質控制工作流程

**場景**：定期執行資料集品質檢查

```bash
#!/bin/bash
# automated_quality_check.sh

DATASET_DIR="/data/datasets/production_dataset"
REPORT_DIR="/data/reports/quality_checks"
DATE=$(date +%Y%m%d)
REPORT_FILE="${REPORT_DIR}/quality_report_${DATE}.json"

mkdir -p "${REPORT_DIR}"

# 步驟 1: 執行完整驗證
echo "Running quality check for ${DATE}..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --check-images \
  --min-width 224 \
  --min-height 224 \
  --max-width 2048 \
  --max-height 2048 \
  --report-file "${REPORT_FILE}" \
  --skip-preflight

# 步驟 2: 解析報告並檢查問題
CORRUPTED=$(jq '.corrupted_images' "${REPORT_FILE}")
WARNINGS=$(jq '.warnings' "${REPORT_FILE}")

# 步驟 3: 如果發現問題，發送警報
if [ ${CORRUPTED} -gt 0 ] || [ ${WARNINGS} -gt 5 ]; then
    echo "⚠️  Quality issues detected!"
    echo "Corrupted images: ${CORRUPTED}"
    echo "Warnings: ${WARNINGS}"

    # 發送通知（範例 - 需要配置通知系統）
    # python scripts/utils/send_notification.py \
    #   --type warning \
    #   --message "Dataset quality check failed. Corrupted: ${CORRUPTED}, Warnings: ${WARNINGS}" \
    #   --report "${REPORT_FILE}"
else
    echo "✓ Quality check passed!"
fi

# 步驟 4: 生成趨勢報告（與歷史資料比較）
# python scripts/utils/generate_trend_report.py \
#   --reports-dir "${REPORT_DIR}" \
#   --output "${REPORT_DIR}/trend_report.html"

echo "Quality check complete. Report: ${REPORT_FILE}"
```

---

## 最佳實踐

### 1. 資料集組織

#### 1.1 目錄結構規範

**推薦結構**：

```
project_root/
├── data/
│   ├── raw/                    # 原始未處理資料
│   │   ├── source1/
│   │   ├── source2/
│   │   └── ...
│   ├── processed/              # 預處理後的資料
│   │   ├── cleaned/
│   │   ├── filtered/
│   │   └── ...
│   └── datasets/               # 最終資料集
│       ├── project_v1.0/
│       │   ├── train/
│       │   ├── val/
│       │   ├── test/
│       │   ├── metadata.json
│       │   └── statistics.json
│       ├── project_v2.0/
│       │   └── ...
│       └── latest -> project_v2.0  # 符號連結
├── models/                     # 訓練的模型
├── experiments/                # 實驗記錄
└── reports/                    # 報告和分析
```

**命名規範**：
- 使用小寫字母和底線：`animal_classification_v2`
- 包含版本號：`dataset_v1.0.0`
- 包含日期（如適用）：`snapshot_20251202`
- 使用描述性名稱：`cats_dogs_balanced` 而非 `dataset1`

#### 1.2 類別命名規範

**推薦做法**：
```
classes/
├── cat/                        # 使用小寫
├── dog/
└── bird/
```

**避免**：
```
classes/
├── Cat/                        # 混合大小寫
├── DOG/                        # 全大寫
└── bird!/                      # 特殊字符
```

**特殊情況處理**：
- 多單詞類別：`golden_retriever` 而非 `GoldenRetriever` 或 `golden retriever`
- 數字編碼：`class_001`, `class_002` 等（適用於無語義類別）
- 層次結構：`animal_cat`, `animal_dog`（如需保留層次資訊）

### 2. 資料分割策略

#### 2.1 標準分割比例

| 資料集大小 | Train | Val | Test | 說明 |
|-----------|-------|-----|------|------|
| < 1,000 | 60% | 20% | 20% | 小型資料集，需要較大測試集 |
| 1K - 10K | 70% | 15% | 15% | 中型資料集，標準分割 |
| 10K - 100K | 80% | 10% | 10% | 大型資料集，可減少測試集比例 |
| > 100K | 85% | 10% | 5% | 超大型資料集，少量測試集即可 |

#### 2.2 何時使用分層分割

**必須使用分層分割的情況**：
- 類別不平衡（最大類 / 最小類 > 2）
- 小型資料集（總樣本數 < 10,000）
- 關鍵應用（醫療、安全等）

**可選擇性使用的情況**：
- 類別完全平衡
- 大型資料集（> 100,000 樣本）
- 初步實驗階段

#### 2.3 K-fold 交叉驗證策略

**何時使用 K-fold**：
- 小型資料集（< 5,000 樣本）
- 模型選擇和超參數調優
- 需要更可靠的性能評估

**K 值選擇**：
- **K=5**：標準選擇，平衡計算成本和評估可靠性
- **K=10**：更可靠的評估，但計算成本更高
- **K=資料集大小**（Leave-One-Out）：只適用於極小資料集（< 100 樣本）

### 3. 資料品質控制

#### 3.1 驗證檢查清單

**基本檢查**：
- [ ] 影像檔案可讀取
- [ ] 副檔名與實際格式匹配
- [ ] 無損壞或截斷的檔案

**進階檢查**：
- [ ] 解析度符合要求（如 min 256x256）
- [ ] 檔案大小合理（如 10KB - 10MB）
- [ ] 色彩空間正確（RGB, 非 CMYK 或灰階）
- [ ] 無重複影像（使用 pHash 或 perceptual hash）

**語義檢查**（可選，需人工或模型輔助）：
- [ ] 影像內容與類別標籤匹配
- [ ] 無明顯錯誤標註
- [ ] 影像品質足夠（清晰度、光線等）

#### 3.2 自動化品質控制腳本

```bash
#!/bin/bash
# quality_control.sh

DATASET_DIR=$1

echo "Starting quality control for: ${DATASET_DIR}"

# 檢查 1: 基本驗證
echo "1. Basic validation..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --check-images \
  --skip-preflight

# 檢查 2: 解析度檢查
echo "2. Resolution check..."
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --check-images \
  --min-width 224 \
  --min-height 224 \
  --skip-preflight

# 檢查 3: 重複檢測（需要自行實作）
echo "3. Duplicate detection..."
# python scripts/utils/detect_duplicates.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --method phash \
#   --threshold 5

# 檢查 4: 統計分析
echo "4. Statistical analysis..."
python scripts/automation/scenarios/dataset_builder.py stats \
  --dataset-dir "${DATASET_DIR}" \
  --detailed \
  --skip-preflight

echo "Quality control complete!"
```

### 4. 性能優化

#### 4.1 大型資料集處理

**問題**：處理數百萬張影像時速度慢、記憶體不足

**解決方案**：

```bash
# 1. 增加批次大小和工作執行緒
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/large_dataset \
  --output-dir /data/output \
  --batch-size 128 \
  --num-workers 16 \
  --skip-preflight

# 2. 分批處理
# 將大型資料集分成多個批次處理
for batch in /data/raw_images/batch_*; do
    python scripts/automation/scenarios/dataset_builder.py create-from-dir \
      --input-dir "${batch}" \
      --output-dir "/tmp/processed_$(basename ${batch})" \
      --skip-preflight
done

# 合併所有批次
python scripts/automation/scenarios/dataset_builder.py merge \
  --dataset-dirs /tmp/processed_batch_* \
  --output-dir /data/final_dataset \
  --skip-preflight

# 3. 使用快取（如果多次處理相同資料）
export DATASET_BUILDER_CACHE=/tmp/dataset_cache
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --use-cache \
  --skip-preflight
```

#### 4.2 記憶體使用優化

**配置記憶體限制**：

```yaml
# config.yaml
performance:
  memory_limit_gb: 8          # 限制記憶體使用
  batch_size: 32              # 減少批次大小
  prefetch_factor: 2          # 減少預取數量
  use_streaming: true         # 使用串流處理（不將所有資料載入記憶體）
```

**監控記憶體使用**：

```bash
# 使用系統監控工具
watch -n 1 'ps aux | grep dataset_builder | grep -v grep'

# 或在腳本中設定記憶體限制（Linux）
ulimit -v 8388608  # 限制 8GB 虛擬記憶體
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --skip-preflight
```

### 5. 版本控制與追蹤

#### 5.1 資料集版本化

**使用語義化版本**：

```
v主要版本.次要版本.修補版本

範例：
- v1.0.0: 初始版本
- v1.1.0: 新增類別
- v1.1.1: 修復損壞影像
- v2.0.0: 重大重構（不兼容 v1.x）
```

**版本控制腳本**：

```bash
#!/bin/bash
# version_control.sh

DATASET_NAME="animals_classification"
NEW_VERSION=$1  # 例如：v2.0.0

BASE_DIR="/data/datasets/${DATASET_NAME}"
VERSION_DIR="${BASE_DIR}/${NEW_VERSION}"

# 創建新版本
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/raw_images \
  --output-dir "${VERSION_DIR}" \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight

# 生成版本標籤
cat > "${VERSION_DIR}/.dataset_version" <<EOF
name: ${DATASET_NAME}
version: ${NEW_VERSION}
created: $(date -Iseconds)
creator: $(whoami)
commit_hash: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
EOF

# 更新 latest 連結
ln -sfn "${NEW_VERSION}" "${BASE_DIR}/latest"

echo "Created ${DATASET_NAME} ${NEW_VERSION}"
```

#### 5.2 元數據追蹤

**擴展元數據**：

```json
{
  "dataset": {
    "name": "animals_classification",
    "version": "v2.0.0",
    "created_at": "2025-12-02T10:30:00Z",
    "created_by": "data_team",
    "description": "Multi-class animal classification dataset"
  },
  "sources": [
    {
      "name": "ImageNet subset",
      "url": "https://...",
      "date_acquired": "2025-11-01",
      "license": "CC BY-SA 4.0"
    },
    {
      "name": "Internal collection",
      "date_acquired": "2025-11-15",
      "count": 5000
    }
  ],
  "processing": {
    "steps": [
      "Deduplication using pHash",
      "Resolution filtering (min 256x256)",
      "Corruption detection",
      "Stratified split (80/10/10)"
    ],
    "tools": [
      "dataset_builder v1.0.0"
    ]
  },
  "statistics": {
    "total_images": 50000,
    "num_classes": 10,
    "splits": {
      "train": 40000,
      "val": 5000,
      "test": 5000
    }
  },
  "changelog": [
    {
      "version": "v2.0.0",
      "date": "2025-12-02",
      "changes": [
        "Added hamster and guinea_pig classes",
        "Increased training set by 10,000 images",
        "Fixed mislabeled samples in v1.1.0"
      ]
    },
    {
      "version": "v1.1.0",
      "date": "2025-11-15",
      "changes": [
        "Added bird class",
        "Improved class balance"
      ]
    },
    {
      "version": "v1.0.0",
      "date": "2025-11-01",
      "changes": [
        "Initial release"
      ]
    }
  ]
}
```

### 6. 錯誤處理與恢復

#### 6.1 中斷恢復

**啟用檢查點**：

```bash
# 創建資料集時啟用檢查點
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/large_dataset \
  --output-dir /data/output \
  --enable-checkpoint \
  --checkpoint-dir /tmp/checkpoints \
  --skip-preflight

# 如果中斷，從檢查點恢復
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/large_dataset \
  --output-dir /data/output \
  --resume-from-checkpoint /tmp/checkpoints/latest \
  --skip-preflight
```

#### 6.2 錯誤日誌分析

**配置詳細日誌**：

```bash
# 啟用 DEBUG 等級日誌
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --log-level DEBUG \
  --log-file /tmp/dataset_builder.log \
  --skip-preflight

# 分析日誌
grep "ERROR" /tmp/dataset_builder.log
grep "WARNING" /tmp/dataset_builder.log
```

---

## 故障排除

### 常見問題

#### 問題 1: 記憶體不足 (Out of Memory)

**症狀**：
```
MemoryError: Unable to allocate array with shape (10000, 512, 512, 3)
```

**原因**：
- 批次大小過大
- 同時載入太多影像到記憶體
- 系統記憶體不足

**解決方法**：

1. 減少批次大小
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --batch-size 16 \  # 減少批次大小
  --skip-preflight
```

2. 減少工作執行緒
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --num-workers 2 \  # 減少並行處理數
  --skip-preflight
```

3. 啟用串流模式（不將所有資料載入記憶體）
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --streaming-mode \
  --skip-preflight
```

#### 問題 2: 處理速度慢

**症狀**：
- 處理大型資料集需要數小時
- CPU 使用率低（< 50%）

**原因**：
- 未充分利用多核心 CPU
- I/O 瓶頸（磁碟讀寫速度）
- 未使用快取

**解決方法**：

1. 增加工作執行緒
```bash
# 使用所有 CPU 核心
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --num-workers $(nproc) \  # 使用所有核心
  --skip-preflight
```

2. 使用 SSD 或 RAM disk
```bash
# 將資料複製到更快的儲存裝置
cp -r /slow/disk/images /tmp/images
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /tmp/images \
  --output-dir /data/output \
  --skip-preflight
```

3. 啟用平行處理
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --parallel \
  --num-workers 16 \
  --skip-preflight
```

#### 問題 3: 類別不平衡導致分割失敗

**症狀**：
```
Error: Cannot perform stratified split - class 'rare_class' has only 2 samples but needs at least 3 for train/val/test split
```

**原因**：
- 某些類別樣本數太少
- 分割比例要求每個分割至少有 1 個樣本

**解決方法**：

1. 調整最小樣本數過濾
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --min-images-per-class 10 \  # 過濾掉樣本數 < 10 的類別
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight
```

2. 調整分割比例（只分成 train/val）
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --split-ratio 0.9 0.1 \  # 只分成兩份
  --stratify \
  --skip-preflight
```

3. 不使用分層分割
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --split-ratio 0.8 0.1 0.1 \
  # 不加 --stratify 標誌
  --skip-preflight
```

#### 問題 4: 副檔名與實際格式不符

**症狀**：
```
PIL.UnidentifiedImageError: cannot identify image file 'image.jpg'
```

**原因**：
- 檔案被錯誤重命名
- 檔案實際上是其他格式

**解決方法**：

1. 檢查檔案的真實格式
```bash
file suspicious_image.jpg
# 輸出: suspicious_image.jpg: PNG image data, 512 x 512, 8-bit/color RGB
```

2. 批次修正副檔名
```bash
#!/bin/bash
# fix_extensions.sh

for file in /data/images/**/*; do
    if [ -f "$file" ]; then
        # 獲取真實格式
        real_format=$(file -b --mime-type "$file" | cut -d'/' -f2)
        current_ext="${file##*.}"

        # 如果格式不符，重新命名
        if [ "$real_format" != "$current_ext" ]; then
            new_name="${file%.*}.${real_format}"
            mv "$file" "$new_name"
            echo "Renamed: $file -> $new_name"
        fi
    fi
done
```

3. 在 Dataset Builder 中啟用自動修復
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --auto-fix-extensions \
  --skip-preflight
```

#### 問題 5: 磁碟空間不足

**症狀**：
```
OSError: [Errno 28] No space left on device
```

**原因**：
- 輸出目錄所在磁碟空間不足
- 創建符號連結失敗（需要實際複製檔案）

**解決方法**：

1. 檢查磁碟空間
```bash
df -h /data/output
```

2. 使用符號連結（而非複製檔案）
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --use-symlinks \  # 使用符號連結
  --skip-preflight
```

3. 輸出到更大的磁碟
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /mnt/large_disk/output \
  --skip-preflight
```

4. 清理臨時檔案
```bash
# 清理 Dataset Builder 快取
rm -rf /tmp/dataset_builder_cache/*

# 清理系統臨時檔案
sudo apt-get clean  # Ubuntu/Debian
# 或
sudo yum clean all  # CentOS/RHEL
```

### 調試技巧

#### 1. 啟用詳細日誌

```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /data/output \
  --log-level DEBUG \
  --log-file /tmp/debug.log \
  --verbose \
  --skip-preflight
```

#### 2. 測試小樣本

在處理大型資料集前，先用小樣本測試：

```bash
# 創建測試子集
mkdir -p /tmp/test_subset
cp -r /data/images/class_a /tmp/test_subset/
cp -r /data/images/class_b /tmp/test_subset/

# 在小樣本上測試
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /tmp/test_subset \
  --output-dir /tmp/test_output \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight

# 檢查結果
tree /tmp/test_output
```

#### 3. 逐步執行

分步驟執行，每步驗證結果：

```bash
# 步驟 1: 只掃描，不分割
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /data/images \
  --output-dir /tmp/step1_scan \
  --skip-preflight

# 步驟 2: 驗證掃描結果
python scripts/automation/scenarios/dataset_builder.py validate \
  --dataset-dir /tmp/step1_scan \
  --check-images \
  --skip-preflight

# 步驟 3: 執行分割
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --input-dir /tmp/step1_scan \
  --output-dir /tmp/step2_split \
  --split-ratio 0.8 0.1 0.1 \
  --stratify \
  --skip-preflight
```

---

## API 參考

### Python API 使用

Dataset Builder 也可以作為 Python 模組使用。

#### 基本使用

```python
import sys
sys.path.insert(0, '/mnt/c/ai_projects/animation-ai-studio')

from scripts.automation.scenarios.dataset_builder import DatasetBuilder

# 創建 Dataset Builder 實例
builder = DatasetBuilder(skip_preflight=True)

# 創建資料集
result = builder.create_from_directory(
    input_dir='/data/raw_images',
    output_dir='/data/datasets/my_dataset',
    format='imagefolder',
    split_ratio=[0.8, 0.1, 0.1],
    stratify=True,
    seed=42
)

print(f"Created dataset with {result['total_images']} images")
print(f"Classes: {result['num_classes']}")
print(f"Splits: {result['splits']}")
```

#### 驗證資料集

```python
from scripts.automation.scenarios.dataset_builder import DatasetBuilder

builder = DatasetBuilder(skip_preflight=True)

# 驗證資料集
report = builder.validate_dataset(
    dataset_dir='/data/datasets/my_dataset',
    check_images=True,
    min_width=256,
    min_height=256
)

print(f"Valid images: {report['valid_images']}/{report['total_images']}")
print(f"Corrupted: {report['corrupted_images']}")
print(f"Warnings: {report['warnings']}")

# 處理損壞的檔案
if report['corrupted_files']:
    for corrupted in report['corrupted_files']:
        print(f"  {corrupted['path']}: {corrupted['reason']}")
```

#### 合併資料集

```python
from scripts.automation.scenarios.dataset_builder import DatasetBuilder

builder = DatasetBuilder(skip_preflight=True)

# 合併多個資料集
result = builder.merge_datasets(
    dataset_dirs=[
        '/data/datasets/set1',
        '/data/datasets/set2',
        '/data/datasets/set3'
    ],
    output_dir='/data/datasets/merged',
    handle_duplicates='rename',
    resplit=True,
    split_ratio=[0.8, 0.1, 0.1],
    stratify=True
)

print(f"Merged dataset: {result['total_images']} images")
```

#### 提取子集

```python
from scripts.automation.scenarios.dataset_builder import DatasetBuilder

builder = DatasetBuilder(skip_preflight=True)

# 提取特定類別的子集
result = builder.extract_subset(
    dataset_dir='/data/datasets/full',
    output_dir='/data/datasets/subset',
    classes=['cat', 'dog', 'bird'],
    max_samples=100,
    stratify=True,
    seed=42
)

print(f"Extracted subset: {result['total_images']} images")
print(f"Classes: {result['classes']}")
```

### 類別和方法參考

#### DatasetBuilder 類別

**初始化**：
```python
DatasetBuilder(skip_preflight=False)
```

**參數**：
- `skip_preflight` (bool): 是否跳過系統預檢查，預設 False

**方法**：

##### create_from_directory()

創建資料集從目錄。

```python
def create_from_directory(
    self,
    input_dir: str,
    output_dir: str,
    format: str = 'imagefolder',
    split_ratio: Optional[List[float]] = None,
    stratify: bool = True,
    min_images_per_class: int = 1,
    recursive: bool = True,
    seed: int = 42
) -> Dict
```

**參數**：
- `input_dir` (str): 輸入影像目錄
- `output_dir` (str): 輸出資料集目錄
- `format` (str): 資料集格式 ('imagefolder' 或 'flat')
- `split_ratio` (List[float], 可選): 分割比例 [train, val, test]
- `stratify` (bool): 是否使用分層分割
- `min_images_per_class` (int): 每類最小影像數
- `recursive` (bool): 是否遞迴掃描
- `seed` (int): 隨機種子

**返回**：
- Dict: 包含資料集資訊的字典

##### validate_dataset()

驗證資料集完整性。

```python
def validate_dataset(
    self,
    dataset_dir: str,
    check_images: bool = False,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    allowed_formats: Optional[List[str]] = None
) -> Dict
```

**參數**：
- `dataset_dir` (str): 資料集目錄
- `check_images` (bool): 是否深度檢查影像
- `min_width` (int, 可選): 最小寬度
- `min_height` (int, 可選): 最小高度
- `max_width` (int, 可選): 最大寬度
- `max_height` (int, 可選): 最大高度
- `allowed_formats` (List[str], 可選): 允許的格式列表

**返回**：
- Dict: 驗證報告

##### merge_datasets()

合併多個資料集。

```python
def merge_datasets(
    self,
    dataset_dirs: List[str],
    output_dir: str,
    handle_duplicates: str = 'skip',
    resplit: bool = False,
    split_ratio: Optional[List[float]] = None,
    stratify: bool = False
) -> Dict
```

**參數**：
- `dataset_dirs` (List[str]): 要合併的資料集目錄列表
- `output_dir` (str): 輸出目錄
- `handle_duplicates` (str): 重複處理策略 ('skip', 'rename', 'overwrite')
- `resplit` (bool): 是否重新分割
- `split_ratio` (List[float], 可選): 分割比例
- `stratify` (bool): 是否分層分割

**返回**：
- Dict: 合併結果資訊

##### extract_subset()

提取資料集子集。

```python
def extract_subset(
    self,
    dataset_dir: str,
    output_dir: str,
    classes: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    stratify: bool = False,
    seed: int = 42
) -> Dict
```

**參數**：
- `dataset_dir` (str): 原始資料集目錄
- `output_dir` (str): 輸出目錄
- `classes` (List[str], 可選): 要提取的類別
- `splits` (List[str], 可選): 要提取的分割
- `max_samples` (int, 可選): 每類最大樣本數
- `sample_ratio` (float, 可選): 採樣比例
- `stratify` (bool): 是否分層採樣
- `seed` (int): 隨機種子

**返回**：
- Dict: 子集資訊

### 資料類別

#### ImageInfo

影像資訊資料類別。

```python
@dataclass
class ImageInfo:
    path: str                       # 影像路徑
    filename: str                   # 檔案名稱
    class_name: Optional[str]       # 類別名稱
    class_id: Optional[int]         # 類別 ID
    split: Optional[str]            # 分割 (train/val/test)
    width: Optional[int]            # 寬度
    height: Optional[int]           # 高度
    format: Optional[str]           # 格式 (JPEG/PNG/etc)
    size_bytes: Optional[int]       # 檔案大小（位元組）
    is_corrupted: bool = False      # 是否損壞
```

#### DatasetStatistics

資料集統計資料類別。

```python
@dataclass
class DatasetStatistics:
    class_distribution: Dict[str, int]      # 類別分佈
    split_distribution: Dict[str, int]      # 分割分佈
    format_distribution: Dict[str, int]     # 格式分佈
    size_statistics: Dict                   # 尺寸統計
    file_size_statistics: Dict              # 檔案大小統計
```

#### ValidationReport

驗證報告資料類別。

```python
@dataclass
class ValidationReport:
    validated_at: str                       # 驗證時間
    total_images: int                       # 總影像數
    valid_images: int                       # 有效影像數
    corrupted_images: int                   # 損壞影像數
    warnings: int                           # 警告數
    errors: int                             # 錯誤數
    corrupted_files: List[Dict]             # 損壞檔案列表
    warning_files: List[Dict]               # 警告檔案列表
```

---

## 附錄

### A. 支援的影像格式

| 格式 | 副檔名 | 支援讀取 | 支援寫入 | 備註 |
|------|--------|----------|----------|------|
| JPEG | .jpg, .jpeg | ✓ | ✓ | 最常用，有損壓縮 |
| PNG | .png | ✓ | ✓ | 無損壓縮，支援透明度 |
| BMP | .bmp | ✓ | ✓ | 未壓縮，檔案較大 |
| TIFF | .tiff, .tif | ✓ | ✓ | 高品質，支援多層 |
| WebP | .webp | ✓ | ✓ | Google 格式，壓縮率高 |
| GIF | .gif | ✓ | ✓ | 支援動畫，256 色限制 |

### B. 錯誤代碼參考

| 錯誤代碼 | 說明 | 解決方法 |
|----------|------|----------|
| E001 | 輸入目錄不存在 | 檢查路徑是否正確 |
| E002 | 輸出目錄已存在且非空 | 使用 --force 或選擇其他目錄 |
| E003 | 無法讀取影像檔案 | 檢查檔案權限和格式 |
| E004 | 類別樣本數不足 | 調整 min_images_per_class |
| E005 | 分割比例無效 | 確保比例和為 1.0 |
| E006 | 記憶體不足 | 減少 batch_size 或增加系統記憶體 |
| E007 | 磁碟空間不足 | 清理磁碟或使用其他儲存位置 |
| E008 | 權限不足 | 使用 sudo 或調整檔案權限 |

### C. 性能基準測試

**測試環境**：
- CPU: Intel i7-10700K (8 cores, 16 threads)
- RAM: 32GB DDR4
- Storage: NVMe SSD

**測試結果**：

| 資料集大小 | 影像數量 | 處理時間 | 記憶體使用 | 備註 |
|-----------|----------|----------|------------|------|
| 小型 | 1,000 | 5 秒 | 500 MB | 快速處理 |
| 中型 | 10,000 | 45 秒 | 1.2 GB | 標準配置 |
| 大型 | 100,000 | 8 分鐘 | 3.5 GB | 需要優化 |
| 超大型 | 1,000,000 | 90 分鐘 | 8 GB | 建議分批處理 |

### D. 相關資源

**官方文檔**：
- [PyTorch ImageFolder 文檔](https://pytorch.org/vision/stable/datasets.html#imagefolder)
- [scikit-learn train_test_split 文檔](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Pillow 文檔](https://pillow.readthedocs.io/)

**社群資源**：
- [Dataset Builder GitHub Issues](https://github.com/your-org/animation-ai-studio/issues)
- [討論區](https://github.com/your-org/animation-ai-studio/discussions)

**相關工具**：
- [imgaug](https://github.com/aleju/imgaug): 影像增強工具
- [Albumentations](https://github.com/albumentations-team/albumentations): 快速影像增強庫
- [fiftyone](https://github.com/voxel51/fiftyone): 資料集視覺化和管理工具

---

## 版本歷史

### v1.0.0 (2025-12-02)

**初始版本**

- ✓ 資料集創建（從目錄、ImageFolder 格式、扁平結構）
- ✓ 資料集分割（隨機、分層、K-fold）
- ✓ 元數據生成（JSON 格式、統計資訊）
- ✓ 資料集驗證（影像完整性、解析度檢查）
- ✓ 資料集操作（合併、子集提取）
- ✓ 命令行介面（4 個主要命令）
- ✓ Python API
- ✓ 配置檔案支援（YAML）
- ✓ 完整測試覆蓋

**已知限制**：
- 暫不支援影像增強（交由後續管道處理）
- 暫不支援線上更新（需要重新創建資料集）
- 暫不支援分散式處理

**未來計劃**：
- v1.1.0: 添加進階過濾選項
- v1.2.0: 添加資料集視覺化工具
- v2.0.0: 整合資料增強管道

---

## 結論

Dataset Builder 是一個強大且靈活的資料集管理工具，提供了機器學習工作流程所需的所有基本功能。通過遵循本指南中的最佳實踐和工作流程範例，你可以高效地組織、驗證和管理你的影像資料集。

如有任何問題或建議，請參閱：
- [故障排除](#故障排除) 部分
- [GitHub Issues](https://github.com/your-org/animation-ai-studio/issues)
- [討論區](https://github.com/your-org/animation-ai-studio/discussions)

祝你使用愉快！
