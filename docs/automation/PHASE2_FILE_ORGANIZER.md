# File Organizer (檔案組織器)

## 概述

File Organizer 是 Phase 2 自動化基礎設施的檔案管理組件，提供全面的檔案組織和管理能力。所有操作都經過 CPU 最佳化，支援大規模檔案操作。

### 核心功能

- ✅ **智能檔案分類**：按類型、日期、大小自動分類
- ✅ **批次重命名**：支援 glob 和 regex 模式匹配
- ✅ **重複檔案偵測**：使用 MD5 雜湊進行內容比對
- ✅ **磁碟空間分析**：按目錄和檔案類型統計空間使用
- ✅ **進階檔案搜尋**：多條件篩選（名稱、大小、日期）
- ✅ **Dry-run 模式**：預覽操作而不實際變更
- ✅ **記憶體安全**：自動檢查可用記憶體
- ✅ **雙語日誌**：中英文雙語輸出

### 系統需求

**必需依賴**：
```bash
# Python 標準函式庫（無額外依賴）
python>=3.10
```

**可選依賴**：
```bash
psutil>=5.9.0      # 記憶體監控（推薦）
pyyaml>=6.0        # YAML 配置支援（批次處理需要）
```

**系統需求**：
- Python 3.10+
- 任何作業系統（Linux, Windows, macOS）
- 磁碟空間：視檔案操作而定

### 安裝

```bash
# 啟動 ai_env 環境
conda activate ai_env

# 基本使用無需額外安裝（使用 Python 標準函式庫）

# 可選：安裝完整功能
pip install psutil>=5.9.0 pyyaml>=6.0

# 驗證安裝
python scripts/automation/scenarios/file_organizer.py --help
```

---

## 快速入門

### 範例 1：按類型組織檔案

```bash
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-type \
  --input /path/to/messy_folder \
  --output /path/to/organized_folder
```

**結果**：
```
organized_folder/
├── images/        # .jpg, .png, .gif, etc.
├── videos/        # .mp4, .avi, .mkv, etc.
├── audio/         # .mp3, .wav, .flac, etc.
├── documents/     # .pdf, .doc, .txt, etc.
├── code/          # .py, .js, .html, etc.
└── other/         # 未分類檔案
```

### 範例 2：尋找重複檔案

```bash
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input /path/to/check \
  --method hash \
  --min-size 1048576  # 只檢查 > 1MB 的檔案
```

**結果**：
```
🔍 Found 5 duplicate groups
   Total wasted space: 2.3 GB

📊 Top duplicate groups:
   1. Group (wasted: 1.5 GB)
      - /path/to/check/video1.mp4
      - /path/to/check/backup/video1_copy.mp4
      - /path/to/check/old/video1_backup.mp4
```

### 範例 3：按日期組織

```bash
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input /path/to/photos \
  --output /path/to/organized \
  --date-format "%Y/%m/%d"
```

**結果**：
```
organized/
├── 2024/
│   ├── 01/
│   │   ├── 01/  # 2024-01-01
│   │   └── 15/  # 2024-01-15
│   └── 12/
│       └── 02/  # 2024-12-02
```

### 範例 4：批次重命名

```bash
# 使用 regex 重命名
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  batch-rename \
  --input /path/to/files \
  --pattern "IMG_(\d+).jpg" \
  --replacement "photo_\1.jpg" \
  --use-regex
```

**結果**：
- `IMG_0001.jpg` → `photo_0001.jpg`
- `IMG_0002.jpg` → `photo_0002.jpg`
- `IMG_0003.jpg` → `photo_0003.jpg`

### 範例 5：磁碟空間分析

```bash
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input /path/to/analyze \
  --depth 3 \
  --top-n 20
```

**結果**：
```
📊 Analysis Results
   Total size: 45.6 GB
   Total files: 12,543
   Total directories: 856

📂 Top file types by size:
   1. .mp4: 1,234 files, 32.5 GB
   2. .jpg: 8,456 files, 8.9 GB
   3. .pdf: 567 files, 2.1 GB
```

---

## 操作詳解

### 1. Organize by Type（按類型組織）

**功能**：將檔案按類型自動分類到對應資料夾

**參數**：
- `--input`：輸入目錄
- `--output`：輸出目錄
- `--no-subdirs`：不建立子目錄（所有檔案放在同一層）
- `--move`：移動檔案而非複製

**支援的檔案分類**：

| 分類 | 副檔名 |
|------|--------|
| **images** | .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .svg |
| **videos** | .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v |
| **audio** | .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma |
| **documents** | .pdf, .doc, .docx, .txt, .rtf, .odt, .pages |
| **spreadsheets** | .xls, .xlsx, .csv, .ods, .numbers |
| **presentations** | .ppt, .pptx, .key, .odp |
| **archives** | .zip, .rar, .7z, .tar, .gz, .bz2, .xz |
| **code** | .py, .js, .html, .css, .java, .cpp, .c, .h, .sh, .yaml, .json |
| **executables** | .exe, .app, .dmg, .deb, .rpm, .apk |
| **other** | 所有未分類檔案 |

**範例**：

```bash
# 基本用法（複製檔案）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-type \
  --input /path/to/downloads \
  --output /path/to/organized

# 移動檔案（不保留原始檔案）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-type \
  --input /path/to/downloads \
  --output /path/to/organized \
  --move

# Dry-run（預覽而不實際操作）
python scripts/automation/scenarios/file_organizer.py \
  --dry-run \
  --skip-preflight \
  organize-by-type \
  --input /path/to/downloads \
  --output /path/to/organized
```

**輸出範例**：
```
📂 Copying files from /path/to/downloads to /path/to/organized
   Organizing by type with subdirectories
📊 Found 100 files to organize

✅ Organization complete!
   Files processed: 100
   Files copied: 98
   Files skipped: 2

📊 Category breakdown:
   images: 45 files
   documents: 23 files
   videos: 15 files
   code: 10 files
   audio: 5 files
```

---

### 2. Organize by Date（按日期組織）

**功能**：按檔案修改日期或建立日期組織到資料夾

**參數**：
- `--input`：輸入目錄
- `--output`：輸出目錄
- `--date-format`：日期格式（預設：`%Y/%m`）
- `--use-created-date`：使用建立日期（預設使用修改日期）
- `--move`：移動檔案而非複製

**日期格式代碼**：
- `%Y`：4 位數年份（2024）
- `%y`：2 位數年份（24）
- `%m`：月份（01-12）
- `%d`：日期（01-31）
- `%b`：月份縮寫（Jan, Feb, etc.）
- `%B`：月份全名（January, February, etc.）

**範例**：

```bash
# YYYY/MM 格式（預設）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input /path/to/photos \
  --output /path/to/by_date

# YYYY/MM/DD 格式
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input /path/to/photos \
  --output /path/to/by_date \
  --date-format "%Y/%m/%d"

# YYYY-Month 格式
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input /path/to/photos \
  --output /path/to/by_date \
  --date-format "%Y-%B"

# 使用建立日期
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input /path/to/photos \
  --output /path/to/by_date \
  --use-created-date
```

**使用情境**：
- 照片整理（按拍攝日期）
- 文件歸檔（按修改日期）
- 專案管理（按建立日期）

---

### 3. Batch Rename（批次重命名）

**功能**：使用模式匹配批次重命名檔案

**參數**：
- `--input`：輸入目錄
- `--pattern`：匹配模式（glob 或 regex）
- `--replacement`：替換模式
- `--use-regex`：使用正規表達式（預設使用 glob）
- `--recursive`：遞迴處理子目錄

**Glob vs Regex**：

| 模式類型 | 適用場景 | 範例 |
|---------|---------|------|
| **Glob** | 簡單模式匹配 | `*.txt`, `IMG_*.jpg` |
| **Regex** | 複雜模式和捕獲群組 | `IMG_(\d+).jpg`, `(\w+)_backup\..*` |

**範例**：

```bash
# Glob：重命名所有 .txt 檔案
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  batch-rename \
  --input /path/to/files \
  --pattern "*.txt" \
  --replacement "document_{}.txt"

# Regex：提取數字並重新格式化
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  batch-rename \
  --input /path/to/photos \
  --pattern "IMG_(\d{4}).jpg" \
  --replacement "photo_\1.jpg" \
  --use-regex

# Regex：添加前綴
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  batch-rename \
  --input /path/to/files \
  --pattern "(.*)\.txt" \
  --replacement "backup_\1.txt" \
  --use-regex

# 遞迴處理所有子目錄
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  batch-rename \
  --input /path/to/root \
  --pattern "old_*" \
  --replacement "new_*" \
  --recursive
```

**重要提示**：
- Glob 模式中的 `*` 會被替換為實際檔名
- Regex 可以使用捕獲群組 `\1`, `\2` 等
- 檔名衝突時會自動添加數字後綴（`_1`, `_2`, etc.）
- 使用 `--dry-run` 預覽結果

---

### 4. Find Duplicates（尋找重複檔案）

**功能**：偵測重複檔案並報告浪費的空間

**參數**：
- `--input`：輸入目錄
- `--method`：偵測方法
  - `hash`：內容雜湊（MD5，最準確）
  - `name`：檔案名稱
  - `size`：檔案大小
- `--no-recursive`：不遞迴處理子目錄
- `--min-size`：最小檔案大小（位元組）
- `--output-json`：輸出 JSON 報告

**偵測方法比較**：

| 方法 | 準確度 | 速度 | 適用場景 |
|------|--------|------|---------|
| **hash** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 精確偵測內容相同的檔案 |
| **name** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 快速找出名稱相同的檔案 |
| **size** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 可能相同的檔案候選 |

**範例**：

```bash
# 基本用法（hash 方法）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input /path/to/check \
  --method hash

# 只檢查大檔案（> 10MB）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input /path/to/check \
  --method hash \
  --min-size 10485760

# 按名稱快速檢查
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input /path/to/check \
  --method name

# 輸出 JSON 報告
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input /path/to/check \
  --method hash \
  --output-json /path/to/duplicates_report.json
```

**輸出範例**：
```
🔍 Finding duplicates in /path/to/check
   Method: hash
   Minimum size: 1.0 MB
📊 Analyzing 5,432 files...

🔍 Found 15 duplicate groups
   Total wasted space: 8.7 GB

📊 Top duplicate groups:

   1. Group (wasted: 4.2 GB)
      - /path/to/check/movies/movie1.mp4
      - /path/to/check/backup/movie1_copy.mp4
      - /path/to/check/archive/movie1_old.mp4

   2. Group (wasted: 2.1 GB)
      - /path/to/check/photos/vacation.jpg
      - /path/to/check/photos/backup/vacation.jpg
```

**JSON 報告格式**：
```json
[
  {
    "hash": "5d41402abc4b2a76b9719d911017c592",
    "size_bytes": 4200000000,
    "files": [
      "/path/to/file1.mp4",
      "/path/to/file2.mp4",
      "/path/to/file3.mp4"
    ],
    "total_wasted_space": 8400000000
  }
]
```

---

### 5. Analyze Disk Space（磁碟空間分析）

**功能**：分析目錄的磁碟空間使用情況

**參數**：
- `--input`：輸入目錄
- `--depth`：最大目錄深度（預設：2）
- `--top-n`：顯示前 N 個項目（預設：20）
- `--output-json`：輸出 JSON 報告

**分析內容**：
- 總大小、檔案數、目錄數
- 按檔案類型統計
- 最大的目錄
- 最大的檔案

**範例**：

```bash
# 基本分析
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input /path/to/analyze

# 深度掃描（3 層）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input /path/to/analyze \
  --depth 3

# 顯示前 50 個項目
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input /path/to/analyze \
  --top-n 50

# 輸出 JSON 報告
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input /path/to/analyze \
  --output-json /path/to/space_report.json
```

**輸出範例**：
```
📊 Analyzing disk space in /path/to/analyze
   Depth: 2, Top items: 20

📊 Analysis Results
   Total size: 156.8 GB
   Total files: 45,678
   Total directories: 1,234

📂 Top 20 file types by size:
   1. .mp4: 2,345 files, 89.4 GB
   2. .jpg: 12,456 files, 34.2 GB
   3. .pdf: 1,567 files, 15.6 GB
   4. .zip: 234 files, 8.9 GB
   5. .docx: 890 files, 3.4 GB

📁 Top 20 largest directories:
   1. videos/raw: 67.8 GB
   2. photos/2024: 23.4 GB
   3. documents/archive: 12.1 GB

📄 Top 20 largest files:
   1. videos/raw/footage_001.mp4: 15.2 GB
   2. videos/raw/footage_002.mp4: 12.8 GB
   3. backups/full_backup.zip: 8.5 GB
```

**使用情境**：
- 磁碟空間清理前的調查
- 尋找佔用空間最多的目錄/檔案
- 定期空間使用報告
- 專案大小評估

---

### 6. Search（進階檔案搜尋）

**功能**：使用多條件篩選搜尋檔案

**參數**：
- `--input`：搜尋目錄
- `--name-pattern`：檔案名稱模式（glob）
- `--extension`：副檔名篩選
- `--min-size`：最小檔案大小（位元組）
- `--max-size`：最大檔案大小（位元組）
- `--modified-after`：修改日期之後（YYYY-MM-DD）
- `--modified-before`：修改日期之前（YYYY-MM-DD）
- `--no-recursive`：不遞迴搜尋子目錄
- `--output-list`：輸出檔案列表到文字檔

**範例**：

```bash
# 按名稱搜尋
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/search \
  --name-pattern "*.jpg"

# 按副檔名和大小搜尋
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/search \
  --extension .mp4 \
  --min-size 104857600  # > 100MB

# 按日期範圍搜尋
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/search \
  --modified-after 2024-01-01 \
  --modified-before 2024-12-31

# 組合條件搜尋
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/search \
  --name-pattern "backup_*" \
  --extension .zip \
  --min-size 10485760 \
  --modified-before 2024-01-01

# 輸出結果到檔案
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/search \
  --name-pattern "*.log" \
  --output-list /path/to/log_files.txt
```

**輸出範例**：
```
🔍 Searching files in /path/to/search
   Filters: name: *.jpg, min: 1.0 MB, after: 2024-01-01

✅ Found 234 matching files

📄 Sample matches:
   - photos/2024/vacation/IMG_001.jpg (2.3 MB)
   - photos/2024/vacation/IMG_002.jpg (1.8 MB)
   - photos/2024/family/DSC_123.jpg (3.1 MB)
   ... and 231 more
```

**檔案列表格式**（`--output-list`）：
```
/path/to/search/photos/2024/vacation/IMG_001.jpg
/path/to/search/photos/2024/vacation/IMG_002.jpg
/path/to/search/photos/2024/family/DSC_123.jpg
...
```

---

## 工作流程範例

### 工作流程 1：清理下載資料夾

**目標**：組織雜亂的下載資料夾

```bash
#!/bin/bash
# cleanup_downloads.sh

DOWNLOADS="/path/to/Downloads"
ORGANIZED="/path/to/Organized"

# Step 1: 尋找並報告重複檔案
echo "🔍 Step 1: Finding duplicates..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input "$DOWNLOADS" \
  --method hash \
  --min-size 1048576 \
  --output-json /tmp/duplicates.json

# Step 2: 按類型組織檔案
echo "📂 Step 2: Organizing by type..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-type \
  --input "$DOWNLOADS" \
  --output "$ORGANIZED" \
  --move

# Step 3: 分析結果
echo "📊 Step 3: Analyzing organized folder..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input "$ORGANIZED" \
  --depth 2

echo "✅ Cleanup complete!"
```

---

### 工作流程 2：照片整理

**目標**：按日期組織照片並尋找重複

```bash
#!/bin/bash
# organize_photos.sh

PHOTOS="/path/to/Photos"
BY_DATE="/path/to/Photos_by_Date"

# Step 1: 按日期組織
echo "📅 Step 1: Organizing by date..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input "$PHOTOS" \
  --output "$BY_DATE" \
  --date-format "%Y/%m"

# Step 2: 尋找重複照片
echo "🔍 Step 2: Finding duplicate photos..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input "$BY_DATE" \
  --method hash

echo "✅ Photo organization complete!"
```

---

### 工作流程 3：專案歸檔

**目標**：整理舊專案檔案

```bash
#!/bin/bash
# archive_old_projects.sh

PROJECTS="/path/to/Projects"
ARCHIVE="/path/to/Archive"

# Step 1: 搜尋 6 個月前的檔案
echo "🔍 Step 1: Finding old files..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input "$PROJECTS" \
  --modified-before 2024-06-01 \
  --output-list /tmp/old_files.txt

# Step 2: 分析空間使用
echo "📊 Step 2: Analyzing space..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input "$PROJECTS" \
  --depth 3

# Step 3: 按日期歸檔
echo "📦 Step 3: Archiving..."
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input "$PROJECTS" \
  --output "$ARCHIVE" \
  --date-format "%Y-Q%m" \
  --move

echo "✅ Archiving complete!"
```

---

## 參數快速參考

### 通用參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--dry-run` | flag | false | 模擬操作（不實際變更） |
| `--skip-preflight` | flag | false | 跳過前置檢查 |

### Organize by Type 參數

| 參數 | 類型 | 必需 | 說明 |
|------|------|------|------|
| `--input` | path | ✅ | 輸入目錄 |
| `--output` | path | ✅ | 輸出目錄 |
| `--no-subdirs` | flag | ❌ | 不建立子目錄 |
| `--move` | flag | ❌ | 移動而非複製 |

### Organize by Date 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--input` | path | 必需 | 輸入目錄 |
| `--output` | path | 必需 | 輸出目錄 |
| `--date-format` | string | %Y/%m | 日期格式 |
| `--use-created-date` | flag | false | 使用建立日期 |
| `--move` | flag | false | 移動而非複製 |

### Batch Rename 參數

| 參數 | 類型 | 必需 | 說明 |
|------|------|------|------|
| `--input` | path | ✅ | 輸入目錄 |
| `--pattern` | string | ✅ | 匹配模式 |
| `--replacement` | string | ✅ | 替換模式 |
| `--use-regex` | flag | ❌ | 使用 regex |
| `--recursive` | flag | ❌ | 遞迴處理 |

### Find Duplicates 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--input` | path | 必需 | 輸入目錄 |
| `--method` | string | hash | 偵測方法 |
| `--no-recursive` | flag | false | 不遞迴 |
| `--min-size` | int | 0 | 最小大小 |
| `--output-json` | path | - | JSON 輸出 |

### Analyze Disk Space 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--input` | path | 必需 | 輸入目錄 |
| `--depth` | int | 2 | 目錄深度 |
| `--top-n` | int | 20 | 顯示項目數 |
| `--output-json` | path | - | JSON 輸出 |

### Search 參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `--input` | path | 搜尋目錄 |
| `--name-pattern` | string | 名稱模式 |
| `--extension` | string | 副檔名 |
| `--min-size` | int | 最小大小 |
| `--max-size` | int | 最大大小 |
| `--modified-after` | date | 之後日期 |
| `--modified-before` | date | 之前日期 |
| `--no-recursive` | flag | 不遞迴 |
| `--output-list` | path | 輸出列表 |

---

## 效能考量

### 處理速度

| 操作 | 1000 檔案 | 10000 檔案 | 100000 檔案 |
|------|----------|-----------|------------|
| **Organize by Type** | ~2s | ~15s | ~2.5min |
| **Organize by Date** | ~2s | ~18s | ~3min |
| **Batch Rename** | ~1s | ~8s | ~1.5min |
| **Find Duplicates (hash)** | ~5s | ~45s | ~8min |
| **Find Duplicates (name)** | ~0.5s | ~4s | ~40s |
| **Analyze Disk Space** | ~3s | ~25s | ~4min |
| **Search** | ~1s | ~10s | ~2min |

### 記憶體使用

File Organizer 的記憶體使用非常低效：

| 操作 | 記憶體使用 |
|------|-----------|
| **基本操作** | < 50 MB |
| **Hash 計算** | < 100 MB |
| **大量檔案** | < 200 MB |

### 最佳化建議

1. **大量檔案**：使用 `--no-recursive` 分批處理
2. **重複檔案偵測**：先用 `--method size` 快速篩選，再用 `--method hash` 確認
3. **磁碟空間分析**：限制 `--depth` 和 `--top-n` 減少處理時間
4. **批次操作**：使用 `--dry-run` 預覽結果
5. **網路磁碟**：避免跨網路操作（速度慢）

---

## 疑難排解

### 問題 1：權限錯誤

**錯誤訊息**：
```
PermissionError: [Errno 13] Permission denied
```

**解決方案**：
```bash
# 檢查檔案權限
ls -la /path/to/file

# 如果需要，添加權限
chmod +r /path/to/file  # 讀取權限
chmod +w /path/to/file  # 寫入權限

# 或使用 sudo（謹慎使用）
sudo python scripts/automation/scenarios/file_organizer.py ...
```

---

### 問題 2：檔名衝突

**問題**：重複檔名導致覆蓋

**解決方案**：
File Organizer 自動處理檔名衝突：
- `file.txt` → `file_1.txt`
- `file.txt` → `file_2.txt`

如果需要手動控制：
```bash
# 使用 dry-run 預覽
python scripts/automation/scenarios/file_organizer.py \
  --dry-run \
  ...
```

---

### 問題 3：處理速度慢

**問題**：大量檔案處理緩慢

**診斷**：
```bash
# 檢查檔案數量
find /path/to/directory -type f | wc -l

# 檢查磁碟速度
dd if=/dev/zero of=/tmp/test bs=1M count=1000 oflag=direct
```

**最佳化**：
```bash
# 1. 減少深度
--depth 2  # 而非 --depth 5

# 2. 使用更快的方法
--method name  # 而非 --method hash

# 3. 限制檔案大小
--min-size 1048576  # 只處理 > 1MB

# 4. 分批處理
# 將大目錄拆分為多個小目錄分別處理
```

---

### 問題 4：記憶體不足

**錯誤訊息**：
```
⚠️ Warning: Low memory (92.3% used)
```

**解決方案**：
```bash
# 1. 關閉其他程式

# 2. 分批處理
# 將操作拆分為多個小批次

# 3. 使用較少記憶體的方法
--method name  # 而非 --method hash
```

---

### 問題 5：特殊字元檔名

**問題**：檔名包含特殊字元（空格、中文等）

**解決方案**：
File Organizer 自動處理特殊字元。如果仍遇到問題：

```bash
# 使用引號包裹路徑
--input "/path/with spaces/folder"
--pattern "中文檔案_*.txt"

# 或使用轉義
--input /path/with\ spaces/folder
```

---

## API 參考

### FileOrganizer 類別

```python
from scripts.automation.scenarios.file_organizer import FileOrganizer

# 初始化
organizer = FileOrganizer(dry_run=False)

# Organize by type
result = organizer.organize_by_type(
    input_dir="/path/to/input",
    output_dir="/path/to/output",
    create_subdirs=True,
    move_files=False
)

# Organize by date
result = organizer.organize_by_date(
    input_dir="/path/to/input",
    output_dir="/path/to/output",
    date_format="%Y/%m",
    use_modified_date=True,
    move_files=False
)

# Batch rename
renamed_files = organizer.batch_rename(
    input_dir="/path/to/files",
    pattern="IMG_(\d+).jpg",
    replacement="photo_\1.jpg",
    use_regex=True,
    recursive=False
)

# Find duplicates
duplicates = organizer.find_duplicates(
    input_dir="/path/to/check",
    method='hash',
    recursive=True,
    min_size=0
)

# Analyze disk space
analysis = organizer.analyze_disk_space(
    input_dir="/path/to/analyze",
    depth=2,
    top_n=20
)

# Search files
matches = organizer.search_files(
    input_dir="/path/to/search",
    name_pattern="*.jpg",
    extension=None,
    min_size=None,
    max_size=None,
    modified_after="2024-01-01",
    modified_before=None,
    recursive=True
)
```

---

## 與其他 Phase 2 組件整合

### 與 Video Processor 整合

```bash
# 1. 從影片提取 frames (Video Processor)
python scripts/automation/scenarios/video_processor.py \
  extract \
  --input /path/to/video.mp4 \
  --output /tmp/frames

# 2. 按日期組織 frames (File Organizer)
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-date \
  --input /tmp/frames \
  --output /path/to/organized_frames \
  --date-format "%Y/%m/%d"
```

### 與 Image Processor 整合

```bash
# 1. 搜尋大型圖像 (File Organizer)
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/photos \
  --extension .jpg \
  --min-size 5242880 \
  --output-list /tmp/large_images.txt

# 2. 批次最佳化 (Image Processor)
while read img; do
  python scripts/automation/scenarios/image_processor.py \
    --operation optimize \
    --input "$img" \
    --output "${img%.jpg}_optimized.jpg" \
    --quality 85
done < /tmp/large_images.txt
```

### 與 Audio Processor 整合

```bash
# 1. 按類型組織多媒體檔案 (File Organizer)
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  organize-by-type \
  --input /path/to/media \
  --output /path/to/organized

# 2. 批次轉換音訊 (Audio Processor)
find /path/to/organized/audio -name "*.wav" | while read audio; do
  python scripts/automation/scenarios/audio_processor.py \
    convert \
    --input "$audio" \
    --output "${audio%.wav}.mp3" \
    --format mp3
done
```

---

## 最佳實踐

### 1. 永遠備份

```bash
# 在進行大規模操作前先備份
cp -r /path/to/important /path/to/backup

# 或使用 rsync
rsync -av /path/to/important/ /path/to/backup/

# 使用 dry-run 預覽
python scripts/automation/scenarios/file_organizer.py \
  --dry-run \
  ...
```

### 2. 逐步操作

```bash
# 不好：一次處理所有檔案
python scripts/automation/scenarios/file_organizer.py \
  organize-by-type \
  --input /huge/directory \
  --output /organized

# 好：先測試小批次
python scripts/automation/scenarios/file_organizer.py \
  --dry-run \
  organize-by-type \
  --input /huge/directory/subfolder \
  --output /organized
```

### 3. 使用有意義的組織結構

```bash
# 組織範例
/Organized/
├── Work/
│   ├── Documents/
│   ├── Presentations/
│   └── Spreadsheets/
├── Personal/
│   ├── Photos/
│   │   ├── 2024/
│   │   │   ├── 01/
│   │   │   └── 02/
│   └── Videos/
└── Archive/
    └── 2023/
```

### 4. 定期清理

```bash
#!/bin/bash
# weekly_cleanup.sh

# 1. 尋找重複檔案
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  find-duplicates \
  --input /path/to/data \
  --method hash \
  --output-json /tmp/duplicates_$(date +%Y%m%d).json

# 2. 分析空間使用
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  analyze-disk-space \
  --input /path/to/data \
  --output-json /tmp/space_$(date +%Y%m%d).json

# 3. 搜尋舊檔案（> 1 年）
python scripts/automation/scenarios/file_organizer.py \
  --skip-preflight \
  search \
  --input /path/to/data \
  --modified-before $(date -d '1 year ago' +%Y-%m-%d) \
  --output-list /tmp/old_files.txt
```

### 5. 記錄操作

```bash
# 將輸出記錄到檔案
python scripts/automation/scenarios/file_organizer.py \
  ... \
  2>&1 | tee -a /var/log/file_organizer.log

# 添加時間戳記
echo "[$(date)] Starting file organization" >> /var/log/file_organizer.log
```

---

## 相關文件

- **Video Processor**: `docs/automation/PHASE2_VIDEO_PROCESSOR.md`
- **Audio Processor**: `docs/automation/PHASE2_AUDIO_PROCESSOR.md`
- **Image Processor**: `docs/automation/PHASE2_IMAGE_PROCESSOR.md`
- **配置範例**: `configs/automation/file_organizer_example.yaml`
- **總體進度**: `AUTOMATION_PROGRESS.md`

---

## 技術支援

遇到問題或需要協助？

1. **檢查日誌**：CLI 輸出包含詳細錯誤訊息
2. **查看疑難排解章節**：本文件「疑難排解」部分
3. **測試基本功能**：
```bash
python scripts/automation/scenarios/file_organizer.py --help
```
4. **使用 dry-run**：預覽操作結果

---

## 更新紀錄

**v1.0.0** (2025-12-02)
- ✅ 初始版本
- ✅ 6 種檔案操作
- ✅ 9 種檔案分類
- ✅ Dry-run 模式
- ✅ 記憶體安全檢查
- ✅ 完整雙語文件

---

*文件版本：1.0.0*
*最後更新：2025-12-02*
*維護者：Animation AI Studio Team*
