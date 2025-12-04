# GPT-SoVITS 自动化训练流程

完整的 GPT-SoVITS 语音克隆训练流程，包含所有预处理步骤。

## 流程概览

GPT-SoVITS 需要 **6个步骤** 完成训练：

### 预处理步骤（Steps 1-3）

1. **提取音素/文本特征**
   - 使用 BERT 模型提取语言特征
   - 输出: `2-name2text.txt`, `3-bert/*.pt`

2. **重采样音频并提取 HuBERT 特征**
   - 音频重采样到 32kHz
   - 提取声学特征
   - 输出: `5-wav32k/*.wav`, `4-cnhubert/*.pt`

3. **提取语义 Token**
   - 生成语义表示
   - 输出: `6-name2semantic.tsv`

### 训练步骤（Steps 4-5）

4. **Stage 1: 训练 GPT 模型**
   - 语义预测模型
   - 输入: 语义 tokens + 音素特征
   - 输出: GPT checkpoint

5. **Stage 2: 训练 SoVITS 声码器**
   - 声学生成模型
   - 输入: GPT 输出 + HuBERT 特征
   - 输出: SoVITS checkpoint

## 必要条件

### 1. 预训练模型

需要下载以下模型到 `GPT-SoVITS/GPT_SoVITS/pretrained_models/`:

```bash
# BERT 模型（用于音素编码）
chinese-roberta-wwm-ext-large/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.txt

# HuBERT 模型（用于声学特征）
chinese-hubert-base/
├── config.json
├── pytorch_model.bin
└── chinese-hubert-base-fairseq-ckpt.pt

# 训练预训练权重
/mnt/c/AI_LLM_projects/ai_warehouse/models/audio/gpt_sovits/pretrained/
├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt  # Stage 1
└── s2G488k.pth  # Stage 2
```

### 2. 训练数据

格式要求：
```
GPT-SoVITS/logs/{character_name}/
├── 0-audio/          # WAV 音频文件
│   ├── {char}_0000.wav
│   ├── {char}_0001.wav
│   └── ...
├── train.list        # 训练列表
└── val.list          # 验证列表
```

`train.list` 和 `val.list` 格式：
```
logs/{char}/0-audio/{char}_0000.wav|{character}|Transcript text here.|en
logs/{char}/0-audio/{char}_0001.wav|{character}|Another sentence.|en
...
```

## 使用方法

### 快速开始

```bash
# 进入项目目录
cd /mnt/c/AI_LLM_projects/animation-ai-studio

# 激活 voice_env 环境
export PATH="/home/b0979/.conda/envs/voice_env/bin:$PATH"

# 运行完整流程（预处理 + 训练）
python scripts/synthesis/tts/gpt_sovits_full_pipeline.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --s1-epochs 15 \
  --s2-epochs 10
```

### 仅运行预处理

如果只想执行预处理步骤（用于调试或准备数据）：

```bash
python scripts/synthesis/tts/gpt_sovits_full_pipeline.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --preprocessing-only
```

### 跳过预处理（直接训练）

如果预处理已完成，只想运行训练：

```bash
python scripts/synthesis/tts/gpt_sovits_full_pipeline.py \
  --character Luca \
  --samples data/films/luca/voice_samples_auto/by_character/Luca \
  --output models/voices/luca/gpt_sovits \
  --skip-preprocessing \
  --s1-epochs 15 \
  --s2-epochs 10
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--character` | str | 必需 | 角色名称（如 "Luca"） |
| `--samples` | str | 必需 | 语音样本目录路径 |
| `--output` | str | 必需 | 模型输出目录 |
| `--gpt-sovits-root` | str | `/mnt/c/AI_LLM_projects/GPT-SoVITS` | GPT-SoVITS 根目录 |
| `--language` | str | `en` | 语言代码（en/zh/ja/ko） |
| `--device` | str | `cuda` | 设备（cuda/cpu） |
| `--s1-epochs` | int | `15` | Stage 1 训练轮数 |
| `--s2-epochs` | int | `10` | Stage 2 训练轮数 |
| `--batch-size` | int | `8` | 批大小 |
| `--skip-preprocessing` | flag | False | 跳过预处理步骤 |
| `--preprocessing-only` | flag | False | 仅运行预处理 |

## 输出文件

### 预处理输出

```
GPT-SoVITS/logs/{character}/
├── 0-audio/              # 原始音频
├── 2-name2text.txt       # 音素特征文件
├── 3-bert/               # BERT 特征
│   ├── {char}_0000.pt
│   └── ...
├── 4-cnhubert/           # HuBERT 特征
│   ├── {char}_0000.pt
│   └── ...
├── 5-wav32k/             # 32kHz 重采样音频
│   ├── {char}_0000.wav
│   └── ...
└── 6-name2semantic.tsv   # 语义 tokens
```

### 训练输出

```
models/voices/{character}/gpt_sovits/
├── training_metadata.json     # 训练元数据
└── checkpoints/               # 训练 checkpoints
    ├── s1_ckpt/              # Stage 1 checkpoints
    │   ├── {char}-e2.ckpt
    │   ├── {char}-e4.ckpt
    │   └── ...
    └── s2_ckpt/              # Stage 2 checkpoints
        ├── {char}-e2.pth
        └── ...
```

## 训练参数

### Stage 1 (GPT 模型)

```yaml
train:
  epochs: 15                 # 训练轮数
  batch_size: 8             # 批大小
  save_every_n_epoch: 2     # 每 N 轮保存一次
  precision: "16-mixed"     # 混合精度训练
  gradient_clip: 1.0        # 梯度裁剪

optimizer:
  lr: 0.01                  # 学习率
  lr_init: 0.00001          # 初始学习率
  lr_end: 0.0001            # 最终学习率
  warmup_steps: 2000        # 预热步数
  decay_steps: 40000        # 衰减步数

model:
  vocab_size: 1025
  phoneme_vocab_size: 512
  embedding_dim: 512
  hidden_dim: 512
  head: 16
  n_layer: 24
```

### Stage 2 (SoVITS 声码器)

```yaml
train:
  epochs: 10
  batch_size: 8
  # ... 类似 Stage 1 配置
```

## 训练时间估计

基于 RTX 5080 16GB GPU：

| 阶段 | 样本数 | 估计时间 |
|------|--------|---------|
| 预处理 Step 1 | 142 | ~10 分钟 |
| 预处理 Step 2 | 142 | ~15 分钟 |
| 预处理 Step 3 | 142 | ~10 分钟 |
| Stage 1 训练 | 127 训练 + 15 验证 | ~1-2 小时 (15 epochs) |
| Stage 2 训练 | 127 训练 + 15 验证 | ~30-60 分钟 (10 epochs) |
| **总计** | | **~2.5-4 小时** |

## 故障排除

### 1. ModuleNotFoundError: 'matplotlib'
```bash
pip install matplotlib seaborn
```

### 2. 预训练模型未找到
确保模型已下载：
```bash
ls /mnt/c/AI_LLM_projects/GPT-SoVITS/GPT_SoVITS/pretrained_models/
# 应该看到 chinese-roberta-wwm-ext-large/ 和 chinese-hubert-base/
```

### 3. CUDA Out of Memory
减小批大小：
```bash
python ... --batch-size 4
```

### 4. 音频文件未找到
检查 train.list 中的路径：
```bash
head /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/Luca/train.list
# 路径应该是相对于 GPT-SoVITS 根目录的
```

### 5. 预处理步骤失败
检查 PYTHONPATH 和工作目录：
```bash
cd /mnt/c/AI_LLM_projects/GPT-SoVITS
export PYTHONPATH=/mnt/c/AI_LLM_projects/GPT-SoVITS:$PYTHONPATH
```

## 监控训练

### GPU 使用情况
```bash
nvidia-smi -l 1  # 每秒更新
```

### 训练日志
查看实时日志：
```bash
tail -f /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/{character}/training.log
```

### TensorBoard
```bash
tensorboard --logdir /mnt/c/AI_LLM_projects/GPT-SoVITS/logs/{character}/s1_output
```

## 下一步

训练完成后：

1. **模型验证**: 使用验证集测试模型质量
2. **推理测试**: 生成测试音频验证音质
3. **模型导出**: 导出最终模型用于部署
4. **参数调优**: 根据结果调整训练参数

## 相关文档

- [GPT-SoVITS 官方文档](https://github.com/RVC-Boss/GPT-SoVITS)
- [语音样本提取指南](./voice_training.md)
- [故障排除详细指南](../reference/voice_implementation_summary.md)

## 版本信息

- GPT-SoVITS: v2 (2025-01)
- PyTorch: 2.7.1+cu128
- PyTorch Lightning: 2.5.6
- 创建日期: 2025-11-20
