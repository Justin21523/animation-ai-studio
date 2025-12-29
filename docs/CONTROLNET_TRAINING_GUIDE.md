# ControlNet 訓練指南（SDXL）

**創建日期：** 2025-12-29  
**狀態：** P4（ControlNet 訓練環境）✅ 可用  
**目的：** 在本專案內建立「可重現、可離線」的 SDXL ControlNet 訓練流程，支援你為遊戲建立資產（角色構圖、景深/深度圖、分割、法線等）。

---

## ✅ 你現在可以做到什麼

1. **把影像+控制圖做成可訓練資料集**（可選：用現成控制圖 / 自動算 Canny 類控制圖）  
   - 工具：`scripts/processing/training/controlnet_dataset_builder.py`
2. **用 diffusers + accelerate 直接訓練 SDXL ControlNet**（離線本機模型）  
   - 工具：`scripts/processing/training/sdxl_controlnet_trainer.py`
3. **訓練完成後自動寫入 registry**（讓生成/agent 直接可用）  
   - 工具：`scripts/processing/training/controlnet_registry_updater.py`
   - Registry：`configs/generation/controlnet_config.yaml`

---

## 0) 前置準備（強烈建議離線）

本專案預設 SDXL 路徑在 `configs/generation/sdxl_config.yaml`：

- `model.base_model`: 單檔 checkpoint（例如 `/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors`）
- `model.base_model_repo`: 本機 SDXL base repo（包含 tokenizers / text encoders / scheduler）

**如果你的 base_model 是單檔 `.safetensors`，訓練一定需要 `base_model_repo`（本機）**，否則會碰到 tokenizer/text encoder/scheduler 的離線相依問題。

---

## 1) 準備訓練資料集

資料集格式（輸出目錄）：

```
<dataset_dir>/
  images/            # 目標圖（target）
  conditioning/      # Control 圖（control map）
  captions/          # 文字提示
  dataset_metadata.json
```

### A. 你已經有控制圖（Depth/Normal/Seg…最推薦）

1) 編輯 `configs/training/controlnet/dataset_builder.yaml`  
2) 執行：

```bash
python scripts/processing/training/controlnet_dataset_builder.py \
  --config configs/training/controlnet/dataset_builder.yaml \
  --images_dir /path/to/target_images \
  --control_images_dir /path/to/control_maps \
  --control_type depth \
  --output_dir outputs/controlnet_datasets/depth_game_v1 \
  --resolution 1024 \
  --overwrite
```

> `control_images_dir` 需要和 `images_dir` **同檔名 stem 對應**（例如 `a.png` ↔ `a.png`）。

### B. 直接從目標圖計算控制圖（canny/scribble/softedge/lineart）

```bash
python scripts/processing/training/controlnet_dataset_builder.py \
  --images_dir /path/to/target_images \
  --control_type canny \
  --output_dir outputs/controlnet_datasets/canny_v1 \
  --resolution 1024 \
  --overwrite
```

---

## 2) 訓練 SDXL ControlNet

1) 編輯 `configs/training/controlnet/trainer.yaml`（至少填 `dataset_dir` / `output_name`）  
2) 執行（單機）：

```bash
python scripts/processing/training/sdxl_controlnet_trainer.py \
  --config configs/training/controlnet/trainer.yaml
```

或（推薦）：

```bash
accelerate launch scripts/processing/training/sdxl_controlnet_trainer.py \
  --config configs/training/controlnet/trainer.yaml
```

輸出會在：

```
outputs/controlnet_training/<output_name>/final/
```

這個 `final/` 目錄就是可被 diffusers `ControlNetModel.from_pretrained()` 載入的格式。

---

## 3) 訓練完成 → 自動更新 registry（讓生成/agent 直接可用）

```bash
python scripts/processing/training/controlnet_registry_updater.py \
  --controlnet_key depth_game_v1 \
  --model_dir outputs/controlnet_training/my_controlnet/final \
  --description "Game depth ControlNet v1" \
  --use_case "Depth maps from engine" \
  --preprocess_as depth
```

如果你要「落盤」到你的模型倉庫（例如 `/mnt/c/ai_models/...`）：

```bash
python scripts/processing/training/controlnet_registry_updater.py \
  --controlnet_key depth_game_v1 \
  --model_dir outputs/controlnet_training/my_controlnet/final \
  --stage_to_dir /mnt/c/ai_models/stable-diffusion/controlnet/trained \
  --stage_name controlnet-depth-game-v1 \
  --stage_overwrite
```

---

## 4) 在生成模組使用（CharacterGenerator / Agent）

- `configs/generation/controlnet_config.yaml` 的 `controlnet_models` key 會被 `ControlNetPipelineManager` 解析
- 若你的 key 是自訂名稱（例如 `depth_game_v1`），建議在 registry entry 裡加：
  - `preprocess_as: depth`（或 canny/pose/seg/normal）

在程式端使用（示意）：

- `scripts/generation/image/character_generator.py`：`use_controlnet=True` + `control_type="<你的key>"`。

---

## ⚠️ 備註（重要）

- 本專案的 `ControlNetPipelineManager` 目前只內建 `canny` 的控制圖前處理；`pose/depth` 仍偏向「使用你已經算好的控制圖」。
- 如果你要完整自動化 `pose` / `depth` 控制圖產生，可以再接：`controlnet_aux`（pose）與深度估計模型（depth），並把結果納入 dataset builder。

