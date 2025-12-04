# Dataset Builder 配置檔案

本目錄包含 Dataset Builder 的各種配置檔案範例，適用於不同的使用場景。

## 配置檔案列表

### 1. basic_config.yaml
**用途**：標準的影像分類資料集創建

**適用場景**：
- 中小型資料集（< 100K 影像）
- 標準的 train/val/test 分割
- 基本的品質控制

**關鍵設定**：
- 分割比例：70/20/10
- 分層分割：啟用
- 最小影像尺寸：256x256
- 最大影像尺寸：2048x2048

**使用方法**：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/automation/dataset/basic_config.yaml \
  --skip-preflight
```

### 2. production_config.yaml
**用途**：生產環境的大型資料集管理

**適用場景**：
- 大型資料集（> 100K 影像）
- 多來源資料整合
- 嚴格的品質控制和驗證
- 需要備份和通知機制

**關鍵設定**：
- 多資料來源支援
- 去重功能（pHash）
- 自動驗證
- 備份機制
- 通知系統（Email/Slack）

**使用方法**：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/automation/dataset/production_config.yaml \
  --skip-preflight
```

### 3. kfold_config.yaml
**用途**：K-fold 交叉驗證資料集

**適用場景**：
- 模型選擇和評估
- 超參數調優
- 需要更可靠的性能評估
- 小型資料集（< 10K 影像）

**關鍵設定**：
- 5-fold 交叉驗證
- 分層分割（每個 fold 保持類別比例）
- 為每個 fold 生成獨立報告

**使用方法**：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/automation/dataset/kfold_config.yaml \
  --skip-preflight
```

**輸出結構**：
```
output_dir/
├── fold_1/
│   ├── train/
│   └── val/
├── fold_2/
│   ├── train/
│   └── val/
├── ...
├── fold_5/
│   ├── train/
│   └── val/
├── kfold_metadata.json
└── kfold_statistics.json
```

### 4. quick_prototype_config.yaml
**用途**：快速原型和實驗

**適用場景**：
- 演算法原型測試
- 快速實驗迭代
- 超參數初步探索
- 概念驗證（Proof of Concept）

**關鍵設定**：
- 小型子集（每類最多 50 張）
- 簡化的驗證（節省時間）
- 不計算均值和標準差
- 優化處理速度

**使用方法**：
```bash
python scripts/automation/scenarios/dataset_builder.py extract-subset \
  --config configs/automation/dataset/quick_prototype_config.yaml \
  --skip-preflight
```

**典型執行時間**：< 1 分鐘

## 配置檔案結構

所有配置檔案都遵循以下基本結構：

```yaml
# 專案資訊（可選）
project:
  name: "Dataset Name"
  description: "Dataset description"
  version: "1.0.0"

# 資料集創建設定
creation:
  input_dir: /path/to/input
  output_dir: /path/to/output
  format: imagefolder
  recursive: true
  min_images_per_class: 5

# 分割設定
split:
  enabled: true
  ratio: [0.7, 0.2, 0.1]
  stratify: true
  seed: 42

# 驗證設定
validation:
  check_images: true
  min_width: 256
  min_height: 256
  allowed_formats:
    - jpg
    - png

# 性能設定
performance:
  batch_size: 32
  num_workers: 4
  memory_limit_gb: 8

# 日誌設定
logging:
  level: INFO
  file: logs/dataset_builder.log
  console: true
```

## 自訂配置

### 基於現有配置修改

1. 複製最接近需求的配置檔案：
```bash
cp configs/automation/dataset/basic_config.yaml \
   configs/automation/dataset/my_custom_config.yaml
```

2. 編輯配置檔案，修改相關參數：
```bash
nano configs/automation/dataset/my_custom_config.yaml
```

3. 使用自訂配置：
```bash
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/automation/dataset/my_custom_config.yaml \
  --skip-preflight
```

### 配置繼承（高級）

可以使用配置繼承來避免重複：

**基礎配置（base.yaml）**：
```yaml
# 所有資料集共用的基礎設定
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

**專案配置（my_project.yaml）**：
```yaml
# 繼承基礎配置
extends: configs/automation/dataset/base.yaml

# 專案特定設定
creation:
  input_dir: /data/my_project/images
  output_dir: /data/datasets/my_project

split:
  ratio: [0.8, 0.1, 0.1]  # 覆寫基礎配置中的預設值
```

## 環境變數支援

配置檔案支援環境變數：

```yaml
creation:
  input_dir: ${DATA_ROOT}/raw_images
  output_dir: ${DATA_ROOT}/datasets/${DATASET_NAME}
```

使用方法：
```bash
export DATA_ROOT=/mnt/data/ai_data
export DATASET_NAME=my_dataset
python scripts/automation/scenarios/dataset_builder.py create-from-dir \
  --config configs/automation/dataset/my_config.yaml \
  --skip-preflight
```

## 配置驗證

在執行前驗證配置檔案：

```bash
# 驗證配置檔案語法
python scripts/automation/scenarios/dataset_builder.py validate-config \
  --config configs/automation/dataset/my_config.yaml
```

## 常見問題

### Q1: 如何選擇合適的配置檔案？

**參考以下決策樹**：

```
資料集大小？
├─ < 10K 影像
│  └─ 需要 K-fold？
│     ├─ 是 → kfold_config.yaml
│     └─ 否
│        └─ 快速實驗？
│           ├─ 是 → quick_prototype_config.yaml
│           └─ 否 → basic_config.yaml
└─ > 10K 影像
   └─ 生產環境？
      ├─ 是 → production_config.yaml
      └─ 否 → basic_config.yaml
```

### Q2: 如何調整記憶體使用？

修改 `performance` 部分：

```yaml
performance:
  batch_size: 16        # 減少批次大小
  num_workers: 2        # 減少工作執行緒
  memory_limit_gb: 4    # 降低記憶體限制
```

### Q3: 如何加快處理速度？

1. 增加工作執行緒：
```yaml
performance:
  num_workers: 16       # 使用更多 CPU 核心
```

2. 使用快取：
```yaml
performance:
  use_cache: true
  cache_dir: /tmp/dataset_builder_cache
```

3. 減少驗證強度：
```yaml
validation:
  check_images: false   # 跳過深度檢查
```

### Q4: 如何處理類別不平衡？

1. 使用分層分割：
```yaml
split:
  stratify: true
```

2. 限制每類最大樣本數：
```yaml
class_filter:
  max_samples_per_class: 1000
```

3. 計算類別權重：
```yaml
metadata:
  compute_class_weights: true
```

## 最佳實踐

### 1. 版本控制

將配置檔案納入版本控制：
```bash
git add configs/automation/dataset/my_config.yaml
git commit -m "Add dataset configuration for project X"
```

### 2. 配置文件命名

使用描述性名稱：
- ✅ `animals_classification_v2.yaml`
- ✅ `production_imagenet_subset.yaml`
- ❌ `config1.yaml`
- ❌ `test.yaml`

### 3. 添加註釋

在配置檔案中添加說明：
```yaml
# 專案：動物分類 v2.0
# 日期：2025-12-02
# 作者：AI Team
# 用途：訓練生產環境模型

creation:
  input_dir: /data/animals_v2
  # 輸出到 SSD 以提高速度
  output_dir: /ssd/datasets/animals_v2
```

### 4. 保持簡潔

只包含必要的設定，使用預設值處理其他情況：
```yaml
# 最小化配置
creation:
  input_dir: /data/images
  output_dir: /data/output

split:
  ratio: [0.8, 0.1, 0.1]
  stratify: true
```

## 參考資源

- [Dataset Builder 完整文檔](../../docs/automation/PHASE3_DATASET_BUILDER.md)
- [Phase 3 規劃文檔](../../docs/automation/PHASE3_PLANNING.md)
- [YAML 語法指南](https://yaml.org/spec/1.2/spec.html)

## 支援

如有問題或建議，請：
1. 查閱完整文檔
2. 查看故障排除部分
3. 提交 GitHub Issue
