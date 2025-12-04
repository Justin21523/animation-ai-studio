# Annotation Tool 配置檔案

本目錄包含 Annotation Tool 的各種配置檔案範例，適用於不同的標註格式轉換和驗證場景。

## 配置檔案列表

### 1. coco_to_yolo_config.yaml
**用途**：COCO 格式轉換為 YOLO 格式

**適用場景**：
- 準備 YOLO 系列模型的訓練資料
- 從 COCO 資料集轉換到 YOLO 格式
- 需要標準化座標格式的場景

**關鍵設定**：
- 輸入格式：COCO JSON
- 輸出格式：YOLO txt
- 自動標準化座標（中心點 + 寬高）
- 生成類別檔案（classes.txt）

**使用方法**：
```bash
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /path/to/annotations.json \\
  --output /path/to/output/yolo \\
  --input-format coco \\
  --output-format yolo \\
  --images-dir /path/to/images \\
  --output-classes-file /path/to/output/yolo/classes.txt
```

### 2. yolo_to_coco_config.yaml
**用途**：YOLO 格式轉換為 COCO 格式

**適用場景**：
- YOLO 標註轉換到 COCO 生態系統
- 準備用於 Detectron2、MMDetection 等框架的資料
- 需要統一標註格式的場景
- 資料集分析和視覺化

**關鍵設定**：
- 輸入格式：YOLO txt
- 輸出格式：COCO JSON
- 自動生成影像和標註 ID
- 支援自訂資料集資訊

**使用方法**：
```bash
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /path/to/yolo/labels \\
  --output /path/to/output/annotations.json \\
  --input-format yolo \\
  --output-format coco \\
  --images-dir /path/to/images \\
  --labels-dir /path/to/yolo/labels \\
  --classes-file /path/to/yolo/classes.txt
```

### 3. coco_to_voc_config.yaml
**用途**：COCO 格式轉換為 Pascal VOC 格式

**適用場景**：
- 準備 Pascal VOC 格式的訓練資料
- 使用需要 XML 標註的工具
- 傳統物件偵測模型訓練
- 與 LabelImg 等標註工具整合

**關鍵設定**：
- 輸入格式：COCO JSON
- 輸出格式：Pascal VOC XML
- 自動創建 Annotations/ 和 ImageSets/ 目錄
- 支援資料分割（train/val/test）
- 支援影像複製到 JPEGImages/

**使用方法**：
```bash
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /path/to/annotations.json \\
  --output /path/to/output/voc \\
  --input-format coco \\
  --output-format voc
```

### 4. validation_config.yaml
**用途**：標註驗證和品質檢查

**適用場景**：
- 標註品質保證（QA）
- 資料清理前的預檢
- 發現和修正標註錯誤
- 資料集統計分析

**關鍵設定**：
- 邊界框有效性檢查
- 影像檔案存在性檢查
- 類別一致性驗證
- 重複標註檢測
- 自訂驗證規則

**使用方法**：
```bash
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /path/to/annotations.json \\
  --format coco \\
  --report-file /path/to/validation_report.json
```

## 配置檔案結構

所有配置檔案都遵循以下基本結構：

```yaml
# 專案資訊（可選）
project:
  name: "Configuration Name"
  description: "Configuration description"
  version: "1.0.0"

# 轉換或驗證設定
conversion:  # 或 validation
  input_format: coco
  output_format: yolo
  input_file: /path/to/input
  output_dir: /path/to/output

# 驗證規則
validation:
  validate_bbox: true
  check_image_files: true
  validate_categories: true

# 過濾設定
filtering:
  filter_invalid: true
  filter_small_bbox: true
  min_bbox_area: 100

# 日誌設定
logging:
  level: INFO
  file: logs/annotation_tool.log
  console: true
```

## 自訂配置

### 基於現有配置修改

1. 複製最接近需求的配置檔案：
```bash
cp configs/automation/annotation/coco_to_yolo_config.yaml \\
   configs/automation/annotation/my_custom_config.yaml
```

2. 編輯配置檔案，修改相關參數：
```bash
nano configs/automation/annotation/my_custom_config.yaml
```

3. 使用自訂配置（注意：目前工具尚未完全支援配置檔案載入，需要使用命令列參數）：
```bash
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /path/to/input \\
  --output /path/to/output \\
  --input-format coco \\
  --output-format yolo \\
  --images-dir /path/to/images
```

## 支援的格式

### COCO (Common Objects in Context)
- **檔案格式**：JSON
- **座標格式**：[x, y, width, height]（左上角 + 寬高）
- **特點**：
  - 支援多種標註類型（邊界框、分割、關鍵點）
  - 豐富的元資料
  - 廣泛的生態系統支援

**COCO JSON 結構**：
```json
{
  "images": [
    {"id": 1, "file_name": "image.jpg", "width": 640, "height": 480}
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
    {"id": 1, "name": "cat", "supercategory": "animal"}
  ]
}
```

### YOLO (You Only Look Once)
- **檔案格式**：文字檔案（每個影像一個 .txt 檔案）
- **座標格式**：[x_center, y_center, width, height]（標準化到 0-1）
- **特點**：
  - 簡單高效
  - YOLO 系列模型的原生格式
  - 易於手動編輯

**YOLO 標註格式**：
```
# image.txt
0 0.5 0.5 0.3 0.3
1 0.2 0.3 0.15 0.2
```

**classes.txt**：
```
cat
dog
bird
```

### Pascal VOC (Visual Object Classes)
- **檔案格式**：XML
- **座標格式**：[xmin, ymin, xmax, ymax]（左上角 + 右下角）
- **特點**：
  - 傳統物件偵測標準
  - 豐富的物件屬性（困難、截斷、姿態）
  - 良好的可讀性

**VOC XML 結構**：
```xml
<annotation>
  <folder>VOC2012</folder>
  <filename>image.jpg</filename>
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

## 常見使用場景

### 場景 1：COCO → YOLO 轉換（YOLOv5/v8 訓練）

```bash
# 1. 轉換 COCO 到 YOLO
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /data/coco/annotations/instances_train.json \\
  --output /data/yolo/train \\
  --input-format coco \\
  --output-format yolo \\
  --images-dir /data/coco/images/train \\
  --output-classes-file /data/yolo/classes.txt

# 2. 驗證轉換結果
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/yolo/train/labels \\
  --format yolo \\
  --classes-file /data/yolo/classes.txt \\
  --images-dir /data/coco/images/train

# 3. 分析標註統計
python scripts/automation/scenarios/annotation_tool.py analyze \\
  --input /data/yolo/train/labels \\
  --format yolo \\
  --classes-file /data/yolo/classes.txt \\
  --output /data/yolo/statistics.json
```

### 場景 2：YOLO → COCO 轉換（Detectron2 訓練）

```bash
# 1. 轉換 YOLO 到 COCO
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /data/yolo/labels \\
  --output /data/coco/annotations/instances.json \\
  --input-format yolo \\
  --output-format coco \\
  --images-dir /data/images \\
  --labels-dir /data/yolo/labels \\
  --classes-file /data/yolo/classes.txt

# 2. 驗證 COCO 標註
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/coco/annotations/instances.json \\
  --format coco \\
  --report-file /data/coco/validation_report.json
```

### 場景 3：標註品質檢查

```bash
# 1. 執行全面驗證
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/annotations/instances.json \\
  --format coco \\
  --report-file /data/validation_report.json

# 2. 查看驗證報告
cat /data/validation_report.json

# 3. 根據報告修正標註問題
# （手動或使用其他工具）

# 4. 重新驗證
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/annotations/instances_fixed.json \\
  --format coco \\
  --report-file /data/validation_report_fixed.json
```

### 場景 4：格式統一（多來源資料整合）

```bash
# 1. 轉換來源 A（YOLO）到 COCO
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /data/source_a/labels \\
  --output /data/unified/annotations_a.json \\
  --input-format yolo \\
  --output-format coco \\
  --images-dir /data/source_a/images \\
  --labels-dir /data/source_a/labels \\
  --classes-file /data/source_a/classes.txt

# 2. 轉換來源 B（VOC）到 COCO
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /data/source_b/Annotations \\
  --output /data/unified/annotations_b.json \\
  --input-format voc \\
  --output-format coco \\
  --images-dir /data/source_b/JPEGImages

# 3. 合併 COCO 標註（使用其他工具或腳本）
# 4. 驗證合併後的資料集
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/unified/annotations_merged.json \\
  --format coco
```

## 最佳實踐

### 1. 轉換前先備份

```bash
# 建立備份
cp -r /data/annotations /data/annotations_backup_$(date +%Y%m%d)
```

### 2. 驗證 → 轉換 → 再驗證

```bash
# 步驟 1：驗證原始標註
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/original/annotations.json \\
  --format coco

# 步驟 2：轉換格式
python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /data/original/annotations.json \\
  --output /data/converted/labels \\
  --input-format coco \\
  --output-format yolo \\
  --images-dir /data/images

# 步驟 3：驗證轉換結果
python scripts/automation/scenarios/annotation_tool.py validate \\
  --input /data/converted/labels \\
  --format yolo \\
  --classes-file /data/converted/classes.txt
```

### 3. 使用日誌記錄轉換過程

```bash
mkdir -p logs

python scripts/automation/scenarios/annotation_tool.py convert \\
  --input /data/annotations.json \\
  --output /data/converted \\
  --input-format coco \\
  --output-format yolo \\
  --images-dir /data/images \\
  --log-level DEBUG \\
  2>&1 | tee logs/conversion_$(date +%Y%m%d_%H%M%S).log
```

### 4. 批次轉換多個資料集

```bash
#!/bin/bash
# batch_convert.sh

DATASETS=("train" "val" "test")

for dataset in "${DATASETS[@]}"; do
  echo "Converting $dataset..."
  python scripts/automation/scenarios/annotation_tool.py convert \\
    --input /data/coco/annotations/instances_$dataset.json \\
    --output /data/yolo/$dataset \\
    --input-format coco \\
    --output-format yolo \\
    --images-dir /data/coco/images/$dataset \\
    --output-classes-file /data/yolo/classes.txt

  echo "Validating $dataset..."
  python scripts/automation/scenarios/annotation_tool.py validate \\
    --input /data/yolo/$dataset/labels \\
    --format yolo \\
    --classes-file /data/yolo/classes.txt \\
    --report-file /data/yolo/$dataset/validation_report.json
done

echo "Batch conversion completed!"
```

## 故障排除

### 問題 1：座標超出影像範圍

**錯誤訊息**：
```
WARNING: Bounding box out of bounds: bbox=[650, 490, 50, 50], image_size=(640, 480)
```

**解決方法**：
- 檢查原始標註是否正確
- 使用驗證工具找出所有問題標註
- 修正或移除無效標註

### 問題 2：類別名稱不一致

**錯誤訊息**：
```
ERROR: Category 'Cat' not found in classes file (available: ['cat', 'dog', 'bird'])
```

**解決方法**：
- 統一類別名稱的大小寫
- 更新 classes.txt 檔案
- 使用類別對應功能

### 問題 3：缺少影像檔案

**錯誤訊息**：
```
WARNING: Image file not found: /data/images/image_001.jpg
```

**解決方法**：
- 確認影像目錄路徑正確
- 檢查檔案名稱拼寫
- 驗證影像檔案權限

### 問題 4：YOLO 座標格式錯誤

**錯誤訊息**：
```
ERROR: Invalid YOLO coordinate: 1.5 (must be between 0 and 1)
```

**解決方法**：
- 確認 YOLO 格式標註已標準化
- 使用驗證工具檢查所有標註
- 重新生成標註檔案

## 參考資源

- [Annotation Tool 完整文檔](../../docs/automation/PHASE3_ANNOTATION_TOOL.md)
- [Phase 3 規劃文檔](../../docs/automation/PHASE3_PLANNING.md)
- [COCO 格式規範](https://cocodataset.org/#format-data)
- [YOLO 格式說明](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Pascal VOC 格式說明](http://host.robots.ox.ac.uk/pascal/VOC/)

## 支援

如有問題或建議，請：
1. 查閱完整文檔
2. 查看故障排除部分
3. 提交 GitHub Issue
