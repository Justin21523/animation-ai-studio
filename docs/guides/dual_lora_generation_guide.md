# Dual LoRA Image Generation Guide

這份指南說明如何使用通用的 Dual LoRA 工具來生成不同角色的圖片。

---

## 目錄

1. [快速開始](#快速開始)
2. [工具概覽](#工具概覽)
3. [配置文件格式](#配置文件格式)
4. [Prompt 模板庫](#prompt-模板庫)
5. [常見用例](#常見用例)
6. [最佳實踐](#最佳實踐)

---

## 快速開始

### 1. 測試 LoRA 權重組合

```bash
python scripts/generation/image/dual_lora_weight_tester.py \
  --config configs/generation/test_config.json
```

### 2. 生成角色變化圖片

```bash
python scripts/generation/image/character_variations_generator.py \
  configs/generation/luca_character_config.json
```

---

## 工具概覽

### 工具 1: `dual_lora_weight_tester.py`

**用途**: 測試不同 LoRA 權重組合，找出最佳比例

**輸入**:
- 配置文件（JSON）或命令行參數
- 兩個 LoRA 模型路徑
- 權重範圍列表

**輸出**:
- NxM 張測試圖片（N 個權重 × M 個權重）
- 測試結果 JSON 元數據

**使用場景**:
- 新角色 LoRA 訓練完成後，找出最佳權重
- 嘗試不同 LoRA 組合（角色+風格、角色+背景等）
- 比較不同 base model 的效果

---

### 工具 2: `character_variations_generator.py`

**用途**: 使用最佳權重生成大量角色變化圖片

**輸入**:
- 角色配置文件（JSON）
- 包含表情、動作、環境等變化定義

**輸出**:
- 按類別組織的圖片（expressions、actions、environments等）
- 每個類別的元數據 JSON

**使用場景**:
- 為動畫製作生成參考圖
- 建立角色素材庫
- 測試角色在不同情境下的表現

---

## 配置文件格式

### Dual LoRA 測試配置

文件位置: `configs/generation/`

```json
{
  "base_model": "/path/to/sdxl_base_model.safetensors",

  "lora1": {
    "name": "character",
    "path": "/path/to/character_lora.safetensors",
    "weights": [0.5, 0.7, 0.8, 1.0, 1.2]
  },

  "lora2": {
    "name": "style",
    "path": "/path/to/style_lora.safetensors",
    "weights": [0.5, 0.7, 0.8, 1.0, 1.2]
  },

  "prompt": "character description, environment, style keywords",
  "negative_prompt": "blurry, low quality, ...",

  "output_dir": "outputs/image_generation/test_name",

  "width": 1024,
  "height": 1024,
  "steps": 30,
  "cfg_scale": 7.5,
  "seed": 42
}
```

---

### 角色變化生成配置

文件位置: `configs/generation/`

```json
{
  "character": {
    "name": "Character Name",
    "description": "Brief description"
  },

  "base_model": "/path/to/sdxl_base_model.safetensors",

  "lora_paths": {
    "character": "/path/to/character_lora.safetensors",
    "style": "/path/to/style_lora.safetensors"
  },

  "lora_weights": {
    "character": 0.6,
    "style": 1.2
  },

  "negative_prompt": "...",

  "prompt_templates": {
    "character_base": "{name}, {age} years old, {hair}, {eyes}",
    "default_outfit": "purple striped shirt",
    "default_environment": "italian seaside, sunny day",
    "style_keywords": "pixar 3d animation style, high quality"
  },

  "width": 1024,
  "height": 1024,
  "steps": 30,
  "cfg_scale": 7.5,

  "output_base": "outputs/image_generation/character_variations",

  "variations": [
    {
      "category": "expressions",
      "seed_offset": 0,
      "items": [
        {
          "category": "expression",
          "name": "happy",
          "prompt": "full prompt with character, happy expression, environment, style"
        },
        {
          "category": "expression",
          "name": "sad",
          "prompt": "full prompt with character, sad expression, environment, style"
        }
      ]
    },
    {
      "category": "actions",
      "seed_offset": 100,
      "items": [...]
    }
  ]
}
```

---

## Prompt 模板庫

文件位置: `configs/generation/prompt_templates.json`

### 使用方式

這個文件包含可重複使用的 prompt 片段：

1. **Negative Prompts**: 各種場景的 negative prompt 組合
2. **Style Keywords**: 不同風格的關鍵字（Pixar、Disney、Dreamworks等）
3. **Expression Templates**: 表情描述模板
4. **Action Templates**: 動作姿勢模板
5. **Environment Templates**: 環境場景模板
6. **Outfit Templates**: 服裝模板
7. **Lighting Templates**: 光線條件模板

### 範例

```json
{
  "expression_templates": {
    "happy": "happy expression, big smile, joyful, cheerful",
    "curious": "curious expression, head tilted, inquisitive look"
  },

  "action_templates": {
    "running": "running fast, dynamic pose, arms pumping, energetic",
    "jumping": "jumping in the air, arms raised, joyful, mid-air"
  },

  "lighting_templates": {
    "golden_hour": "golden hour lighting, warm sunset glow, soft shadows",
    "midday_bright": "bright midday sun, strong lighting, high contrast"
  }
}
```

### 組合 Prompt 的建議順序

1. 角色名稱和基本描述
2. 年齡和物理特徵（髮型、眼睛）
3. 表情或情緒
4. 動作或姿勢
5. 服裝
6. 環境/背景
7. 光線條件
8. 風格關鍵字（Pixar、3D等）
9. 質量修飾詞

---

## 常見用例

### 用例 1: 新角色訓練完成

**目標**: 找出角色 LoRA 與風格 LoRA 的最佳權重組合

**步驟**:

1. 創建測試配置文件 `configs/generation/new_character_test.json`
2. 設定權重測試範圍（建議 0.5-1.2）
3. 運行測試：
   ```bash
   python scripts/generation/image/dual_lora_weight_tester.py \
     --config configs/generation/new_character_test.json
   ```
4. 檢查輸出圖片，選擇最佳權重組合
5. 更新角色配置文件中的 `lora_weights`

---

### 用例 2: 建立角色素材庫

**目標**: 為一個角色生成多種表情、動作、環境的圖片

**步驟**:

1. 確認已找到最佳 LoRA 權重（用例 1）
2. 創建角色配置文件 `configs/generation/character_name_config.json`
3. 定義變化類別：
   - expressions（表情）
   - actions（動作）
   - environments（環境）
   - outfits（服裝）
   - lighting（光線）
4. 為每個類別編寫 prompt
5. 運行生成：
   ```bash
   python scripts/generation/image/character_variations_generator.py \
     configs/generation/character_name_config.json
   ```
6. 檢查輸出目錄中的結果

---

### 用例 3: 比較不同 LoRA 組合

**目標**: 測試同一角色配不同風格 LoRA 的效果

**步驟**:

1. 準備多個測試配置文件：
   - `character_pixar.json` (角色 + Pixar 風格)
   - `character_disney.json` (角色 + Disney 風格)
   - `character_realistic.json` (角色 + 寫實風格)
2. 使用相同的 prompt 和 seed
3. 依次運行測試
4. 比較不同風格的表現

---

## 最佳實踐

### 1. Prompt 編寫技巧

**優先順序**:
- CLIP tokenizer 限制為 77 tokens
- 將最重要的描述放在前面
- 次要描述（如 "masterpiece", "best quality"）可能被截斷

**好的 Prompt 結構**:
```
{角色名稱}, {年齡}, {髮型}, {眼睛},
{表情}, {動作}, {服裝},
{環境}, {光線},
{風格關鍵字}, {質量修飾詞}
```

**範例**:
```
luca paguro, young boy, 13 years old, brown curly hair, green eyes,
happy smile, waving hand, purple striped shirt,
italian seaside pier, sunny day, golden hour lighting,
pixar 3d animation style, high quality, vibrant colors
```

---

### 2. 權重調整建議

**Character LoRA 權重**:
- `0.4-0.6`: 輕微的角色特徵，適合混合風格
- `0.6-0.8`: 平衡的角色辨識度（推薦範圍）
- `0.8-1.0`: 強角色特徵
- `1.0-1.2`: 非常強的角色特徵（可能 over-fitting）

**Style LoRA 權重**:
- `0.5-0.7`: 輕微風格影響
- `0.7-1.0`: 明顯風格效果
- `1.0-1.2`: 強烈風格（推薦範圍）
- `1.2-1.5`: 非常強烈（可能失真）

**Luca 的最佳組合**:
- Character LoRA: `0.6`
- Pixar Style LoRA: `1.2`

---

### 3. Seed 管理

**固定 Seed**:
- 用於比較不同參數效果
- 用於重現特定結果

**變化 Seed**:
- 用於生成多樣性
- 建議使用 `seed_offset` 系統性地變化 seed

**範例**:
```json
{
  "category": "expressions",
  "seed_offset": 0,
  "items": [...]  // seed: 42, 43, 44, ...
},
{
  "category": "actions",
  "seed_offset": 100,
  "items": [...]  // seed: 142, 143, 144, ...
}
```

---

### 4. 輸出組織

**建議的目錄結構**:
```
outputs/image_generation/
├── {character}_dual_lora_test/      # 權重測試結果
│   ├── char0.50_style0.50.png
│   ├── ...
│   └── test_results.json
│
├── {character}_single_lora_comparison/  # 單 LoRA 對照
│   ├── character_only/
│   ├── style_only/
│   └── base_model/
│
└── {character}_variations/          # 最終變化圖片
    ├── expressions/
    ├── actions/
    ├── environments/
    ├── outfits/
    └── lighting/
```

---

### 5. 質量控制

**檢查清單**:
- [ ] 角色特徵是否準確（髮型、眼睛、臉型）
- [ ] 風格是否符合預期（Pixar 3D 感）
- [ ] 是否有 artifacts（多餘的肢體、變形）
- [ ] 表情是否清晰表達
- [ ] 動作是否自然
- [ ] 環境是否符合描述
- [ ] 整體構圖是否協調

**問題排查**:
| 問題 | 可能原因 | 解決方案 |
|------|----------|----------|
| 角色不準確 | Character LoRA 權重太低 | 提高至 0.7-0.8 |
| 過度 over-fitting | LoRA 權重太高 | 降低至 0.8-1.0 |
| 風格不明顯 | Style LoRA 權重太低 | 提高至 1.0-1.2 |
| Artifacts 出現 | 權重組合不當 | 調整權重平衡 |
| Prompt 被截斷 | 超過 77 tokens | 精簡 prompt |

---

### 6. 效能優化

**VRAM 使用**:
- 雙 LoRA: ~8GB VRAM
- 單次生成峰值: ~10GB
- 建議 GPU: RTX 3090/4090/5080 (16GB+)

**生成速度**:
- 平均: 10-25 秒/張（1024×1024）
- Pipeline 載入: 35-40 秒
- 批量生成: 可考慮保持 pipeline 載入

**節省時間**:
```python
# 不要每次都重新載入 pipeline
tester.load_pipeline()  # 只載入一次

for variation in variations:
    tester.generate_with_weights(...)  # 重複使用

tester.unload_pipeline()  # 最後才卸載
```

---

## 範例配置文件

### Luca 角色完整配置

參考: `configs/generation/luca_character_config.json`

這個文件包含：
- 完整的角色描述
- 最佳 LoRA 權重設定
- 5 種表情變化
- 5 種動作姿勢
- 5 種環境場景
- 3 種服裝變化
- 3 種光線條件

總共 21 張變化圖片。

---

### 其他角色模板

創建新角色配置時，複製 `character_config_template.json` 並填寫：

1. 角色基本資訊
2. LoRA 模型路徑
3. 最佳權重（需先測試）
4. 自訂變化類別和 prompt

---

## 進階技巧

### 1. 混合多個 LoRA

雖然目前工具支援雙 LoRA，但可以通過程式碼擴展支援 3+ LoRA：

```python
lora_paths = {
    "character": "luca_lora.safetensors",
    "style": "pixar_lora.safetensors",
    "background": "italian_town_lora.safetensors"
}

lora_weights = {
    "character": 0.6,
    "style": 1.2,
    "background": 0.8
}
```

### 2. 動態權重調整

為不同類別使用不同權重：

```json
{
  "category": "close_up_portraits",
  "lora_weights": {"character": 0.8, "style": 1.0},
  "items": [...]
},
{
  "category": "full_body_action",
  "lora_weights": {"character": 0.6, "style": 1.2},
  "items": [...]
}
```

### 3. 批量處理多角色

創建腳本批量處理：

```bash
for config in configs/generation/*_character_config.json; do
    python scripts/generation/image/character_variations_generator.py "$config"
done
```

---

## 相關資源

- **Prompt 模板庫**: `configs/generation/prompt_templates.json`
- **測試總結**: `LUCA_LORA_TEST_SUMMARY.md`
- **項目文檔**: `docs/`

---

**作者**: Animation AI Studio Team
**最後更新**: 2025-11-20
**版本**: 1.0
