# Hardware Specifications and Configuration

**Last Updated:** 2025-11-16
**Project:** Animation AI Studio - LLM Backend

---

## 🖥️ Actual Hardware Configuration

### CPU
```yaml
Model: AMD Ryzen 9 9950X
Cores: 16
Threads: 32
Architecture: Zen 5
```

### RAM
```yaml
Capacity: 64GB DDR5
Use: Model loading, CPU offloading (if needed)
```

### GPU
```yaml
Model: NVIDIA GeForce RTX 5080
VRAM: 16GB GDDR7
Architecture: Ada Lovelace (Compute Capability 8.9)
CUDA Cores: ~10,240
Tensor Cores: 4th Gen
RT Cores: 4th Gen
```

### Software Environment
```yaml
OS: WSL2 (Linux kernel 6.6.87.2-microsoft-standard-WSL2)
Python: 3.10+
PyTorch: 2.7.0
CUDA: 12.8
Conda Environment: ai_env
```

---

## ⚠️ Critical Requirements

### 1. PyTorch Compatibility
```bash
# ABSOLUTELY FORBIDDEN
❌ DO NOT modify PyTorch 2.7.0
❌ DO NOT change CUDA 12.8
❌ DO NOT use xformers

# MUST USE
✅ PyTorch 2.7.0 (installed)
✅ CUDA 12.8 (installed)
✅ PyTorch native SDPA (Scaled Dot Product Attention)
```

### 2. Attention Backend
```yaml
Required: PyTorch Native SDPA
Forbidden: xformers
Reason: PyTorch 2.x has built-in optimized attention

Environment Variables:
  VLLM_ATTENTION_BACKEND: "TORCH_SDPA"
  XFORMERS_DISABLED: "1"
  VLLM_USE_TRITON_FLASH_ATTN: "0"
```

### 3. Memory Management
```yaml
VRAM: 16GB total
Strategy: Use 90% (14.4GB), keep 1.6GB headroom
OOM Prevention: Dynamic batch sizing, prefix caching

PyTorch Settings:
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512,expandable_segments:True"
```

---

## 📊 VRAM Budget (16GB Total)

### Small Models (Recommended)

#### Qwen2.5-VL-7B-Instruct (Multimodal)
```yaml
Model Size: ~14GB FP16
VRAM Usage: ~14GB (88% of total)
Context Length: 32K tokens
Capabilities: Vision, Chat, Embeddings
Quantization: None needed
```

#### Qwen2.5-14B-Instruct (Reasoning)
```yaml
Model Size: ~28GB → ~12GB with optimizations
VRAM Usage: ~12GB (75% of total)
Context Length: 32K tokens
Capabilities: Chat, Reasoning
Quantization: None needed
```

#### Qwen2.5-Coder-7B-Instruct (Code)
```yaml
Model Size: ~14GB FP16
VRAM Usage: ~14GB (88% of total)
Context Length: 32K tokens
Capabilities: Code generation, Chat
Quantization: None needed
```

### Quantized Large Models (Alternative)

#### Qwen2.5-72B-Instruct-AWQ
```yaml
Original Size: ~140GB FP16
Quantized Size: ~16GB (INT4 AWQ)
VRAM Usage: ~15.5GB (97% of total)
Context Length: 32K tokens
Quantization: AWQ INT4
Trade-off: 3-4x slower inference
```

**Note:** Can only load ONE model at a time. Models must be switched dynamically based on task.

---

## 🚀 Deployment Strategy

### Single-Model Loading (Current)
```yaml
Strategy: Load one model at a time
Switching Time: 10-30 seconds
Memory Efficiency: Optimal
Use Case: Sequential tasks

Workflow:
  1. Receive user request
  2. Determine required model
  3. Unload current model (if different)
  4. Load required model
  5. Execute inference
  6. Keep model in memory for next request
```

### Multi-Model (Future, with 24GB+ VRAM)
```yaml
Strategy: Keep multiple small models loaded
Requires: 24GB+ VRAM
Not possible with RTX 5080 16GB
```

---

## ⚡ Performance Expectations

### Small Models (7B/14B)
```yaml
First Token Latency: ~0.5-1.0 seconds
Generation Speed: 30-50 tokens/second
Batch Processing: Up to 32 sequences
Context Window: 32K tokens
Memory Efficiency: Excellent
```

### Quantized Large Models (72B-AWQ)
```yaml
First Token Latency: ~2-4 seconds
Generation Speed: 10-20 tokens/second
Batch Processing: Limited (1-4 sequences)
Context Window: 32K tokens
Memory Efficiency: Good (75% compression)
Quality: ~95% of full precision
```

---

## 🔧 vLLM Configuration Parameters

### For 7B/14B Models
```bash
--gpu-memory-utilization 0.85    # 13.6GB / 16GB
--tensor-parallel-size 1         # Single GPU
--max-model-len 32768            # 32K context
--max-num-batched-tokens 8192    # Batch size
--max-num-seqs 256               # Concurrent requests
--enable-prefix-caching          # KV cache reuse
--dtype auto                     # FP16 or BF16
```

### For Quantized 72B Models
```bash
--gpu-memory-utilization 0.95    # 15.2GB / 16GB
--tensor-parallel-size 1         # Single GPU
--max-model-len 32768            # 32K context
--max-num-batched-tokens 4096    # Smaller batch
--max-num-seqs 32                # Fewer concurrent
--quantization awq               # AWQ INT4
--enable-prefix-caching          # KV cache reuse
```

---

## 📦 Model Storage

### Cache Strategy
```yaml
Primary Cache: /mnt/c/AI_LLM_projects/ai_warehouse/cache/huggingface
VLLM Cache: /mnt/c/AI_LLM_projects/ai_warehouse/cache/vllm
Torch Cache: /mnt/c/AI_LLM_projects/ai_warehouse/cache/torch

Purpose: Avoid duplicate downloads across projects
Shared With: 3d-animation-lora-pipeline, other AI projects
```

### Model Paths
```yaml
Base Path: /mnt/c/AI_LLM_projects/ai_warehouse/models/llm

Models:
  - qwen/Qwen2.5-VL-7B-Instruct
  - qwen/Qwen2.5-14B-Instruct
  - qwen/Qwen2.5-Coder-7B-Instruct
  - qwen/Qwen2.5-72B-Instruct-AWQ (optional)
```

---

## 🛡️ Safety Limits

### OOM Prevention
```yaml
Max Sequence Length: 32768 tokens
VRAM Threshold: 95%
Action on Threshold: Reject new requests
Request Timeout: 300 seconds

Monitoring:
  - GPU memory usage per request
  - Peak memory during generation
  - Automatic cleanup on completion
```

### Concurrent Request Limits
```yaml
Small Models: 10 concurrent requests
Large Models: 2 concurrent requests
Queue Strategy: FIFO with priority
```

---

## 📈 Scaling Options

### Vertical Scaling (Better GPU)
```yaml
RTX 5090 24GB:
  - Load 2-3 small models simultaneously
  - Faster quantized 72B inference

RTX 6000 Ada 48GB:
  - Load full Qwen2.5-72B (no quantization)
  - Multiple models simultaneously
```

### Horizontal Scaling (Multi-GPU, not applicable)
```yaml
Current Setup: Single RTX 5080
Not Supported: Tensor parallelism across multiple GPUs
Reason: Only one GPU available
```

### CPU Offloading (Fallback)
```yaml
Available: 64GB RAM, 32 threads
Strategy: Offload model layers to CPU
Use Case: Running 72B full precision (slow)
Expected Speed: 5-10x slower than GPU-only
```

---

## ✅ Verification Checklist

Before starting vLLM services:
- [ ] CUDA 12.8 installed: `nvidia-smi`
- [ ] PyTorch 2.7.0 verified: `python -c "import torch; print(torch.__version__)"`
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] VRAM available: `nvidia-smi` shows ~16GB free
- [ ] xformers NOT installed: `pip list | grep xformers` returns nothing
- [ ] Environment variables set: Check `.env` file
- [ ] Cache directories exist: Check paths in `paths.yaml`

---

## 📞 Troubleshooting

### OOM Errors
```bash
# Reduce memory utilization
export GPU_MEMORY_UTILIZATION=0.80  # Instead of 0.90

# Reduce batch size
export MAX_NUM_BATCHED_TOKENS=4096  # Instead of 8192

# Reduce concurrent sequences
export MAX_NUM_SEQS=128  # Instead of 256
```

### Slow Inference
```bash
# Check if using correct attention backend
echo $VLLM_ATTENTION_BACKEND  # Should be TORCH_SDPA

# Verify CUDA graphs enabled
# (enforce_eager should be false in config)

# Check GPU utilization
nvidia-smi dmon -s u
```

### Model Loading Fails
```bash
# Check VRAM availability
nvidia-smi

# Verify model path
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/models/llm/qwen/

# Check cache
ls -lh /mnt/c/AI_LLM_projects/ai_warehouse/cache/huggingface/
```

---

**For deployment instructions, see:** `IMPLEMENTATION_ROADMAP.md`
**For vLLM configuration details, see:** `config/vllm_config.yaml`
