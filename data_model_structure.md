下面這段你可以直接丟給 LLMProvider Tooling / DeepSeek / 任何 code agent 當 system / instructions，用來讓它們**不要亂塞檔案**，而是乖乖照你現在的 AI_WAREHOUSE 3.0 結構來操作。

---

## 🧩 Filesystem & Storage Spec for Code Agents

> **Context:** This machine is an Ubuntu-based AI workstation with three NVMe drives.
> All code, datasets and models **must** follow the layout below.
> Do **not** write large files under `$HOME` unless explicitly allowed.

### 1. Disks & Mount Points

* **System disk (1 TB)**

  * Device: `/dev/nvme0n1`
  * Mount: `/`
  * Usage: OS, small system tools only.
  * ❌ Do **not** store datasets or large models under `/` or `$HOME` by default.

* **Fast SSD for models & code (2 TB)**

  * Device: `/dev/nvme1n1p1`
  * Mount: `/mnt/c`
  * Usage: models, code projects, tools, conda/envs, caches.
  * This is the primary place for anything “model-ish” or “code-ish”.

* **Large SSD for datasets & training outputs (4 TB)**

  * Device: `/dev/nvme2n1p1`
  * Mount: `/mnt/data`
  * Usage: datasets, extracted frames, segmentation masks, training outputs, large media.

---

### 2. Directory Layout – `/mnt/c` (2 TB, models & code)

All **models, tools, projects, caches** must live here:

```text
/mnt/c
 ├── ai_models/          # All model weights / LoRAs / checkpoints
 │   ├── clip/
 │   ├── controlnet/
 │   ├── detection/
 │   ├── embeddings/
 │   ├── flow/
 │   ├── inpainting/
 │   ├── llm/
 │   ├── reranker/
 │   ├── safety/
 │   ├── segmentation/
 │   ├── stable-diffusion/
 │   ├── video/
 │   ├── lora/
 │   └── lora_sdxl/
 │
 ├── ai_projects/        # Git / coding projects (repos, apps, scripts)
 │   └── <project_name>/
 │
 ├── ai_tools/           # Standalone tools
 │   ├── kohya_ss/
 │   ├── comfyui/
 │   ├── sd_scripts/
 │   └── rvc/
 │
 ├── ai_envs/            # (optional) conda / venv dirs if we decide to move them here
 │
 ├── ai_cache/           # All AI-related caches MUST go here (not under $HOME)
 │   ├── huggingface/
 │   ├── pip/
 │   └── torch/
 │
 └── tmp/                # scratch space for temporary downloads / unpacking
```

#### Environment variables (for any code you write):

When configuring Python, HF, transformers, etc, always assume:

```bash
HF_HOME=/mnt/c/ai_cache/huggingface
TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface
TORCH_HOME=/mnt/c/ai_cache/torch
XDG_CACHE_HOME=/mnt/c/ai_cache
```

If you generate scripts / notebooks, please **set these** so no large cache goes to `$HOME/.cache` or `/tmp`.

---

### 3. Directory Layout – `/mnt/data` (4 TB, datasets & training)

All **datasets, training runs, extracted media** go here:

```text
/mnt/data
 ├── datasets/
 │   ├── pixar/
 │   ├── elio/
 │   ├── luca/
 │   ├── audio/
 │   ├── video/
 │   ├── web/
 │   ├── general/              # misc datasets (3d-anime, hunter, inazuma-eleven, yokai-watch, etc.)
 │   └── medical/
 │        ├── aicup_2025_heart_seg/
 │        └── nnUNet_raw/
 │             └── Dataset001_HeartSeg/
 │
 ├── training/
 │   ├── lora/
 │   │   ├── expression_lora/
 │   │   └── evaluation/       # migrated from ai_data/lora_evaluation/*
 │   ├── sd_finetune/
 │   ├── controlnet/
 │   ├── runs/
 │   ├── logs/                 # migrated from ai_data/logs/*
 │   └── nnunet/
 │        └── aicup_2025_heart_seg/
 │
 ├── extracted/
 │   ├── frames/
 │   ├── captions/
 │   └── sam_masks/
 │
 ├── videos/
 │   ├── raw/
 │   ├── processed/
 │   └── ytp/
 │
 ├── audio/
 │   ├── rvc_input/
 │   └── rvc_output/
 │
 ├── backups/
 └── tmp/
```

> **Important:**
> Legacy path `/mnt/data/ai_data/...` exists only as historical data;
> **new code must NOT write there**. Always write into the new structure above.

---

### 4. Rules for Code / Scripts / Agents

1. **Never default to `$HOME` for big stuff.**

   * No large datasets, models or checkpoints under `~` or `/`.
   * Use `/mnt/c` for models/tools, `/mnt/data` for datasets/outputs.

2. **Models go to `/mnt/c/ai_models` only.**

   * If you download HF models, LoRAs, checkpoints, etc, place them in the appropriate subfolder.
   * If you generate new LoRAs, save them under:

     * `/mnt/c/ai_models/lora/…` (SD1.5)
     * `/mnt/c/ai_models/lora_sdxl/…` (SDXL)
     * or other subfolder under `ai_models` as appropriate.

3. **Datasets and training outputs go to `/mnt/data`.**

   * New datasets → `/mnt/data/datasets/...`
   * New training runs → `/mnt/data/training/runs/...`
   * Logs / metrics → `/mnt/data/training/logs/...`
   * Frame extraction / SAM → `/mnt/data/extracted/...`

4. **Cache / temp writes should respect env vars**

   * Use `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME` as above.
   * If you generate a script / Dockerfile, **inject those env vars**.

5. **Conda / venvs**

   * Default is standard `~/miniconda3/envs/...`, **but** if you deliberately create a long-lived env for a big project (e.g. `kohya_ss`), prefer placing it under `/mnt/c/ai_envs/<env_name>` and referencing it explicitly.

6. **Never hardcode absolute home-relative paths for models or datasets.**

   * Use `/mnt/c/...` and `/mnt/data/...` explicitly, or environment variables that point there.

---

### 5. Google Drive / rclone rules

Remote name: **`gdrive`**

Structure in Drive:

```text
gdrive:ai_warehouse/
 ├── logs/
 ├── train/
 ├── cache/
 ├── rag/
 ├── tool-caches/
 ├── checkpoints/
 ├── hf/
 ├── models/
 └── outputs/
```

**When syncing models from Google Drive:**

* Source: `gdrive:ai_warehouse/models`
* Target: `/mnt/c/ai_models`

Example command (for reference):

```bash
rclone sync gdrive:ai_warehouse/models /mnt/c/ai_models \
  --progress --transfers=8 --checkers=16 --drive-chunk-size=256M
```

> Code agents should **not** change the rclone config or remote name.
> If needed, they may assume `gdrive` exists and follows this structure.

---

### 6. Migration note (for agents reading old code)

* Old scripts may reference paths like `/mnt/data/ai_data/datasets/...` or `/mnt/data/ai_data/models/...`.
* New code should translate those into the new layout as described above.
* Do **not** resurrect deprecated directories like `/mnt/data/ai_data` for new work.

---
