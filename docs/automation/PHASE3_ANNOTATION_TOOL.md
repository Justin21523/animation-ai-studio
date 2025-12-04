# Phase 3: Annotation Tool Integration - 標註工具整合

**版本**: 1.0.0
**狀態**: ✅ 已完成
**最後更新**: 2025-12-02

## 目錄

- [概覽](#概覽)
- [核心功能](#核心功能)
- [快速開始](#快速開始)
- [格式轉換](#格式轉換)
- [驗證功能](#驗證功能)
- [統計分析](#統計分析)
- [命令行參考](#命令行參考)
- [配置檔案](#配置檔案)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)
- [API 參考](#api-參考)

---

## 概覽

### 什麼是 Annotation Tool Integration？

Annotation Tool Integration 是一個強大的標註格式轉換和驗證工具，支援主流的物件偵測標註格式之間的相互轉換，並提供完整的驗證和統計分析功能。

### 主要特點

- **多格式支援**：支援 COCO、YOLO、Pascal VOC 三種主流格式
- **雙向轉換**：任意兩種格式之間的相互轉換
- **完整驗證**：邊界框有效性、影像存在性、類別一致性檢查
- **統計分析**：類別分布、影像統計、標註品質分析
- **CPU 安全**：完全基於 CPU 處理，無需 GPU
- **批次處理**：支援大規模資料集處理

### 使用場景

1. **訓練資料準備**：轉換標註格式以適配不同的訓練框架
2. **資料集整合**：統一多來源資料的標註格式
3. **品質保證**：驗證標註品質，發現和修正錯誤
4. **資料分析**：分析資料集統計特性，優化資料分布

### 系統需求

- **Python**: 3.8+
- **依賴套件**: Pillow, numpy
- **記憶體**: 最低 2GB（取決於資料集大小）
- **儲存空間**: 視資料集大小而定

---

## 核心功能

### 1. 格式轉換 (Convert)

在 COCO、YOLO、Pascal VOC 三種格式之間進行轉換。

**支援的轉換路徑**：
- COCO ↔ YOLO
- COCO ↔ Pascal VOC
- YOLO ↔ Pascal VOC（透過 COCO 作為中間格式）

**轉換特性**：
- 自動處理座標系統差異
- 保留完整的標註資訊
- 生成標準格式輸出
- 支援批次轉換

### 2. 標註驗證 (Validate)

檢查標註檔案的有效性和完整性。

**驗證項目**：
- **邊界框驗證**：檢查座標有效性、尺寸合理性
- **影像驗證**：確認影像檔案存在且可讀
- **類別驗證**：檢查類別一致性和完整性
- **重複檢測**：發現重複或衝突的標註

**輸出結果**：
- 驗證報告（JSON 格式）
- 錯誤和警告列表
- 修復建議

### 3. 統計分析 (Analyze)

分析資料集的統計特性。

**分析內容**：
- **影像統計**：總數、尺寸分布、格式統計
- **標註統計**：總數、每張影像平均標註數
- **類別分布**：各類別樣本數、比例
- **邊界框分析**：尺寸分布、位置分布

**輸出格式**：
- JSON 統計報告
- 控制台摘要輸出

---

## 快速開始

### 基本使用範例

#### 1. COCO 轉 YOLO

```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/coco/annotations.json \
  --output /data/yolo \
  --input-format coco \
  --output-format yolo \
  --images-dir /data/images \
  --output-classes-file /data/yolo/classes.txt
```

#### 2. YOLO 轉 COCO

```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/yolo/labels \
  --output /data/coco/annotations.json \
  --input-format yolo \
  --output-format coco \
  --images-dir /data/images \
  --labels-dir /data/yolo/labels \
  --classes-file /data/yolo/classes.txt
```

#### 3. 驗證標註

```bash
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/annotations.json \
  --format coco \
  --report-file /data/validation_report.json
```

#### 4. 分析統計

```bash
python scripts/automation/scenarios/annotation_tool.py analyze \
  --input /data/annotations.json \
  --format coco \
  --output /data/statistics.json
```

### 完整工作流程

```bash
# 步驟 1: 驗證原始標註
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/original/annotations.json \
  --format coco \
  --report-file /data/original_validation.json

# 步驟 2: 轉換格式
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/original/annotations.json \
  --output /data/yolo \
  --input-format coco \
  --output-format yolo \
  --images-dir /data/images \
  --output-classes-file /data/yolo/classes.txt

# 步驟 3: 驗證轉換結果
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/yolo/labels \
  --format yolo \
  --classes-file /data/yolo/classes.txt \
  --images-dir /data/images \
  --report-file /data/yolo_validation.json

# 步驟 4: 分析轉換後的資料集
python scripts/automation/scenarios/annotation_tool.py analyze \
  --input /data/yolo/labels \
  --format yolo \
  --classes-file /data/yolo/classes.txt \
  --output /data/yolo_statistics.json
```

---

## 格式轉換

### COCO 格式

#### 格式說明

COCO (Common Objects in Context) 是一個廣泛使用的物件偵測標註格式。

**檔案結構**：
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],
      "area": 30000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "cat",
      "supercategory": "animal"
    }
  ]
}
```

**座標格式**：`[x, y, width, height]`
- `x, y`: 邊界框左上角座標
- `width, height`: 邊界框寬度和高度
- 所有值為絕對像素座標

#### COCO → YOLO 轉換

```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/coco/annotations.json \
  --output /data/yolo \
  --input-format coco \
  --output-format yolo \
  --images-dir /data/images \
  --output-classes-file /data/yolo/classes.txt
```

**轉換過程**：
1. 讀取 COCO JSON 檔案
2. 提取影像資訊和標註
3. 轉換座標格式：`[x, y, w, h]` → `[x_center, y_center, w, h]`
4. 標準化座標：除以影像寬高
5. 為每張影像生成對應的 .txt 檔案
6. 生成 classes.txt 檔案

**輸出結構**：
```
yolo/
├── labels/
│   ├── image_001.txt
│   ├── image_002.txt
│   └── ...
└── classes.txt
```

#### COCO → VOC 轉換

```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/coco/annotations.json \
  --output /data/voc \
  --input-format coco \
  --output-format voc
```

**轉換過程**：
1. 讀取 COCO JSON 檔案
2. 轉換座標格式：`[x, y, w, h]` → `[xmin, ymin, xmax, ymax]`
3. 為每張影像生成對應的 XML 檔案
4. 創建 Pascal VOC 目錄結構

**輸出結構**：
```
voc/
├── Annotations/
│   ├── image_001.xml
│   ├── image_002.xml
│   └── ...
└── ImageSets/
    └── Main/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

### YOLO 格式

#### 格式說明

YOLO 格式使用簡單的文字檔案，每張影像對應一個 .txt 檔案。

**檔案格式**：
```
# image_001.txt
0 0.5 0.5 0.3 0.3
1 0.2 0.3 0.15 0.2
```

**每行格式**：`class_id x_center y_center width height`
- `class_id`: 類別索引（從 0 開始）
- `x_center, y_center`: 邊界框中心點（標準化到 0-1）
- `width, height`: 邊界框寬高（標準化到 0-1）

**classes.txt 格式**：
```
cat
dog
bird
```
每行一個類別名稱，行號對應類別 ID。

#### YOLO → COCO 轉換

```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/yolo/labels \
  --output /data/coco/annotations.json \
  --input-format yolo \
  --output-format coco \
  --images-dir /data/images \
  --labels-dir /data/yolo/labels \
  --classes-file /data/yolo/classes.txt
```

**轉換過程**：
1. 讀取 classes.txt 獲取類別資訊
2. 掃描 labels 目錄中的所有 .txt 檔案
3. 對每個標註檔案：
   - 查找對應的影像檔案
   - 讀取影像尺寸
   - 反標準化座標：乘以影像寬高
   - 轉換座標格式：`[x_center, y_center, w, h]` → `[x, y, w, h]`
4. 組裝成 COCO JSON 格式
5. 生成輸出檔案

### Pascal VOC 格式

#### 格式說明

Pascal VOC 使用 XML 格式儲存標註資訊。

**XML 結構**：
```xml
<annotation>
  <folder>VOC2012</folder>
  <filename>image_001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>cat</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin>
      <ymin>100</ymin>
      <xmax>300</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

**座標格式**：`[xmin, ymin, xmax, ymax]`
- `xmin, ymin`: 邊界框左上角座標
- `xmax, ymax`: 邊界框右下角座標
- 所有值為絕對像素座標

#### VOC → COCO 轉換

```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/voc/Annotations \
  --output /data/coco/annotations.json \
  --input-format voc \
  --output-format coco \
  --images-dir /data/voc/JPEGImages
```

**轉換過程**：
1. 掃描 Annotations 目錄中的所有 XML 檔案
2. 解析每個 XML 檔案
3. 轉換座標格式：`[xmin, ymin, xmax, ymax]` → `[x, y, w, h]`
4. 組裝成 COCO JSON 格式
5. 生成輸出檔案

---

## 驗證功能

### 驗證類型

#### 1. 邊界框驗證

**檢查項目**：
- 座標是否在影像範圍內
- 寬度和高度是否為正數
- 面積是否合理
- 寬高比是否正常

**範例**：
```bash
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/annotations.json \
  --format coco \
  --report-file /data/validation_report.json
```

**錯誤範例**：
```json
{
  "error_type": "invalid_bbox",
  "image_id": "image_001.jpg",
  "bbox": [650, 490, 50, 50],
  "image_size": [640, 480],
  "reason": "Bounding box extends beyond image boundaries"
}
```

#### 2. 影像驗證

**檢查項目**：
- 影像檔案是否存在
- 檔案是否可讀
- 影像格式是否支援
- 影像尺寸是否與標註一致

**範例**：
```bash
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/annotations.json \
  --format coco \
  --images-dir /data/images \
  --report-file /data/validation_report.json
```

#### 3. 類別驗證

**檢查項目**：
- 類別 ID 是否存在於類別列表中
- 類別名稱是否一致
- 是否有未使用的類別
- 是否有缺失的類別定義

### 驗證報告

驗證完成後會生成 JSON 格式的報告：

```json
{
  "total_images": 100,
  "total_annotations": 450,
  "valid_annotations": 445,
  "invalid_annotations": 5,
  "errors": [
    {
      "type": "bbox_out_of_bounds",
      "image": "image_023.jpg",
      "annotation_id": 67,
      "details": "Bounding box exceeds image boundaries"
    }
  ],
  "warnings": [
    {
      "type": "small_bbox",
      "image": "image_045.jpg",
      "annotation_id": 123,
      "details": "Bounding box area less than 100 pixels"
    }
  ],
  "statistics": {
    "images_with_no_annotations": 5,
    "avg_annotations_per_image": 4.5,
    "bbox_size_stats": {
      "min_area": 50,
      "max_area": 150000,
      "avg_area": 12500
    }
  }
}
```

### 自訂驗證規則

可以透過配置檔案自訂驗證規則（參見 `configs/automation/annotation/validation_config.yaml`）。

---

## 統計分析

### 基本統計

```bash
python scripts/automation/scenarios/annotation_tool.py analyze \
  --input /data/annotations.json \
  --format coco \
  --output /data/statistics.json
```

### 統計報告內容

```json
{
  "dataset_info": {
    "format": "coco",
    "total_images": 1000,
    "total_annotations": 5432,
    "total_categories": 10
  },
  "image_statistics": {
    "avg_width": 640,
    "avg_height": 480,
    "size_distribution": {
      "640x480": 800,
      "1280x720": 150,
      "1920x1080": 50
    }
  },
  "annotation_statistics": {
    "avg_annotations_per_image": 5.43,
    "min_annotations_per_image": 1,
    "max_annotations_per_image": 25,
    "images_with_no_annotations": 10
  },
  "category_distribution": {
    "cat": 1234,
    "dog": 987,
    "bird": 765,
    "...": "..."
  },
  "bbox_statistics": {
    "avg_area": 12500,
    "min_area": 100,
    "max_area": 200000,
    "avg_width": 125,
    "avg_height": 100
  }
}
```

---

## 命令行參考

### convert 命令

轉換標註格式。

**語法**：
```bash
python scripts/automation/scenarios/annotation_tool.py convert \
  --input INPUT \
  --output OUTPUT \
  --input-format {coco,yolo,voc} \
  --output-format {coco,yolo,voc} \
  [OPTIONS]
```

**必要參數**：
- `--input`: 輸入檔案或目錄
- `--output`: 輸出檔案或目錄
- `--input-format`: 輸入格式 (coco, yolo, voc)
- `--output-format`: 輸出格式 (coco, yolo, voc)

**可選參數**：
- `--images-dir`: 影像目錄
- `--labels-dir`: 標籤目錄（YOLO 格式需要）
- `--classes-file`: 類別檔案（YOLO 格式需要）
- `--output-classes-file`: 輸出類別檔案（YOLO 輸出時）

### validate 命令

驗證標註有效性。

**語法**：
```bash
python scripts/automation/scenarios/annotation_tool.py validate \
  --input INPUT \
  --format {coco,yolo,voc} \
  [OPTIONS]
```

**必要參數**：
- `--input`: 輸入檔案或目錄
- `--format`: 標註格式 (coco, yolo, voc)

**可選參數**：
- `--images-dir`: 影像目錄（用於驗證影像存在性）
- `--report-file`: 驗證報告輸出檔案
- `--classes-file`: 類別檔案（YOLO 格式需要）

### analyze 命令

分析標註統計資訊。

**語法**：
```bash
python scripts/automation/scenarios/annotation_tool.py analyze \
  --input INPUT \
  --format {coco,yolo,voc} \
  [OPTIONS]
```

**必要參數**：
- `--input`: 輸入檔案或目錄
- `--format`: 標註格式 (coco, yolo, voc)

**可選參數**：
- `--output`: 統計報告輸出檔案
- `--classes-file`: 類別檔案（YOLO 格式需要）

### 全域選項

- `--skip-preflight`: 跳過預檢查（開發用）
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: 設定日誌級別（預設：INFO）

---

## 配置檔案

配置檔案位於 `configs/automation/annotation/` 目錄。

### 可用配置

1. **coco_to_yolo_config.yaml** - COCO 到 YOLO 轉換
2. **yolo_to_coco_config.yaml** - YOLO 到 COCO 轉換
3. **coco_to_voc_config.yaml** - COCO 到 Pascal VOC 轉換
4. **validation_config.yaml** - 標註驗證規則

詳細說明請參閱 `configs/automation/annotation/README.md`。

---

## 最佳實踐

### 1. 轉換前備份

```bash
cp -r /data/annotations /data/annotations_backup_$(date +%Y%m%d)
```

### 2. 驗證 → 轉換 → 再驗證

```bash
# 驗證原始資料
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/original/annotations.json \
  --format coco

# 執行轉換
python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/original/annotations.json \
  --output /data/yolo \
  --input-format coco \
  --output-format yolo \
  --images-dir /data/images

# 驗證轉換結果
python scripts/automation/scenarios/annotation_tool.py validate \
  --input /data/yolo/labels \
  --format yolo \
  --classes-file /data/yolo/classes.txt
```

### 3. 使用日誌記錄

```bash
mkdir -p logs

python scripts/automation/scenarios/annotation_tool.py convert \
  --input /data/annotations.json \
  --output /data/converted \
  --input-format coco \
  --output-format yolo \
  --images-dir /data/images \
  --log-level DEBUG \
  2>&1 | tee logs/conversion_$(date +%Y%m%d_%H%M%S).log
```

### 4. 批次處理腳本

```bash
#!/bin/bash
# batch_convert.sh

DATASETS=("train" "val" "test")

for dataset in "${DATASETS[@]}"; do
  echo "Processing $dataset..."

  python scripts/automation/scenarios/annotation_tool.py convert \
    --input /data/coco/annotations/instances_$dataset.json \
    --output /data/yolo/$dataset \
    --input-format coco \
    --output-format yolo \
    --images-dir /data/coco/images/$dataset \
    --output-classes-file /data/yolo/classes.txt

  python scripts/automation/scenarios/annotation_tool.py validate \
    --input /data/yolo/$dataset/labels \
    --format yolo \
    --classes-file /data/yolo/classes.txt \
    --report-file /data/yolo/$dataset/validation_report.json
done
```

---

## 故障排除

### 常見問題

#### Q1: 座標超出影像範圍

**錯誤訊息**：
```
WARNING: Bounding box out of bounds: bbox=[650, 490, 50, 50], image_size=(640, 480)
```

**解決方法**：
1. 檢查原始標註是否正確
2. 使用 validate 命令找出所有問題
3. 手動修正或過濾無效標註

#### Q2: 類別名稱不一致

**錯誤訊息**：
```
ERROR: Category 'Cat' not found in classes file
```

**解決方法**：
1. 統一類別名稱的大小寫
2. 更新 classes.txt 檔案
3. 檢查所有標註檔案

#### Q3: 缺少影像檔案

**錯誤訊息**：
```
WARNING: Image file not found: /data/images/image_001.jpg
```

**解決方法**：
1. 確認影像目錄路徑正確
2. 檢查檔案名稱拼寫
3. 驗證檔案權限

#### Q4: YOLO 座標格式錯誤

**錯誤訊息**：
```
ERROR: Invalid YOLO coordinate: 1.5
```

**解決方法**：
1. 確認座標已標準化到 0-1
2. 檢查標註工具設定
3. 重新生成標註

---

## API 參考

### Python API 使用

```python
from scripts.automation.scenarios.annotation_tool import AnnotationTool

# 創建工具實例
tool = AnnotationTool()

# 格式轉換
result = tool.convert(
    input_path="/data/annotations.json",
    output_path="/data/yolo",
    input_format="coco",
    output_format="yolo",
    images_dir="/data/images",
    output_classes_file="/data/yolo/classes.txt"
)

# 驗證標註
report = tool.validate(
    input_path="/data/annotations.json",
    format="coco",
    images_dir="/data/images"
)

# 分析統計
stats = tool.analyze(
    input_path="/data/annotations.json",
    format="coco"
)
```

### 主要類別

#### BoundingBox

```python
from scripts.automation.scenarios.annotation_tool import BoundingBox

# 創建邊界框
bbox = BoundingBox(x=100, y=100, width=200, height=150)

# 轉換為不同格式
xyxy = bbox.to_xyxy()  # (100, 100, 300, 250)
yolo = bbox.to_yolo(img_width=640, img_height=480)  # 標準化中心點格式

# 從 YOLO 格式創建
bbox = BoundingBox.from_yolo(0.5, 0.5, 0.3, 0.3, 640, 480)
```

#### AnnotationDataset

```python
from scripts.automation.scenarios.annotation_tool import AnnotationDataset

# 創建資料集
dataset = AnnotationDataset(format='coco')

# 添加影像和標註
# ...

# 取得統計資訊
stats = dataset.get_statistics()
```

---

## 效能考量

### 記憶體使用

- 小型資料集 (< 1000 影像): 1-2 GB
- 中型資料集 (1000-10000 影像): 2-4 GB
- 大型資料集 (> 10000 影像): 4-8 GB

### 處理速度

- COCO ↔ YOLO: ~1000 影像/秒
- COCO ↔ VOC: ~500 影像/秒
- 驗證: ~2000 影像/秒

### 優化建議

1. **批次處理**: 分割大型資料集進行處理
2. **增加記憶體**: 提高系統可用記憶體
3. **SSD 儲存**: 使用 SSD 加速檔案讀寫

---

## 相關資源

- [Phase 3 規劃文檔](PHASE3_PLANNING.md)
- [配置檔案說明](../../configs/automation/annotation/README.md)
- [COCO 格式規範](https://cocodataset.org/#format-data)
- [YOLO 格式說明](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Pascal VOC 格式說明](http://host.robots.ox.ac.uk/pascal/VOC/)

---

## 更新日誌

### v1.0.0 (2025-12-02)
- ✅ 初始版本發布
- ✅ 支援 COCO、YOLO、Pascal VOC 格式
- ✅ 實現格式轉換功能
- ✅ 實現驗證功能
- ✅ 實現統計分析功能
- ✅ 完整測試套件
- ✅ 配置檔案範例
- ✅ 中文文檔

---

**文檔維護者**: AI Automation Team
**最後審核**: 2025-12-02
