# Hardware Optimization Guide

**Purpose:** RTX 5080 16GB VRAM optimization strategies and constraints
**Last Updated:** 2025-11-17

---

## 🖥️ Hardware Specifications

### System Configuration

```yaml
CPU: AMD Ryzen 9 9950X
  Cores: 16 cores / 32 threads
  Base Clock: 4.3 GHz
  Boost Clock: Up to 5.7 GHz

RAM: 64GB DDR5
  Speed: 6000 MHz (typical)
  Configuration: Dual channel

GPU: NVIDIA GeForce RTX 5080
  VRAM: 16GB GDDR7
  CUDA Cores: 10,752
  Tensor Cores: 336 (4th gen)
  RT Cores: 84 (3rd gen)
  Memory Bandwidth: 960 GB/s

Storage:
  System: NVMe SSD (high speed)
  Data: /mnt/data/ai_data/ (shared datasets)
  Models: /mnt/c/AI_LLM_projects/ai_warehouse/

Environment:
  OS: Windows 11 with WSL2
  PyTorch: 2.7.0 (IMMUTABLE)
  CUDA: 12.8 (IMMUTABLE)
  Conda: ai_env
```

---

## ⚠️ Critical Constraints

### 1. Single GPU Operation

**Constraint:** Only ONE GPU available

**Implications:**
- Cannot run multiple heavy models simultaneously
- Must use dynamic model switching
- No model parallelism (tensor_parallel_size=1)
- No pipeline parallelism

**Configuration:**
```python
# vLLM configuration
tensor_parallel_size = 1  # Single GPU
pipeline_parallel_size = 1
gpu_memory_utilization = 0.85  # Conservative
```

### 2. VRAM Limitations (16GB)

**Constraint:** 16GB VRAM must accommodate all operations

**Model VRAM Usage:**
```yaml
LLM Models (one at a time):
  Qwen2.5-VL-7B: ~13.8GB
  Qwen2.5-14B: ~11.5GB
  Qwen2.5-Coder-7B: ~13.5GB

Image Generation (requires LLM shutdown):
  SDXL base: ~10-11GB
  SDXL + LoRA: ~11-12GB
  SDXL + LoRA + ControlNet: ~13-15GB

Voice Synthesis (flexible):
  GPT-SoVITS small: ~3-4GB
  GPT-SoVITS training: ~8-10GB

Maximum Safe Usage: 15.5GB (leave 0.5GB buffer)
```

**Rule:** Only ONE "heavy" model loaded at a time
- Heavy models: LLM (7B/14B), SDXL
- Light models: GPT-SoVITS (can coexist with stopped heavy models)

### 3. PyTorch Immutability

**Constraint:** PyTorch 2.7.0 + CUDA 12.8 CANNOT be modified

**Critical Rules:**
1. **NEVER** upgrade/downgrade PyTorch
2. **NEVER** modify CUDA version
3. Any package conflicts → PyTorch wins
4. Must use PyTorch-compatible versions of all libraries

**Attention Backend:**
```python
# REQUIRED: PyTorch native SDPA
VLLM_ATTENTION_BACKEND=TORCH_SDPA

# FORBIDDEN: xformers
XFORMERS_DISABLED=1
```

**Why xformers is forbidden:**
- Conflicts with PyTorch 2.7.0
- vLLM uses PyTorch SDPA
- Diffusers can use either, but must match vLLM
- Unified attention backend = PyTorch SDPA everywhere

---

## 🔄 Dynamic Model Switching Strategy

### Overview

Since 16GB VRAM cannot fit LLM + SDXL simultaneously, we implement dynamic model loading/unloading:

```
Workflow:
1. User makes request
2. LLM analyzes intent (LLM loaded)
3. LLM plans execution (LLM loaded)
4. Stop LLM service (free ~12-14GB)
5. Load SDXL pipeline (use ~13-15GB)
6. Generate images
7. Unload SDXL (free ~13-15GB)
8. Restart LLM service (load ~12-14GB)
9. LLM evaluates quality (LLM loaded)
10. Return results
```

### Switching Performance

```yaml
Model Switching Times:
  Stop LLM service: ~5-8 seconds
  Unload SDXL pipeline: ~2-3 seconds
  Load SDXL pipeline: ~5-8 seconds
  Start LLM service: ~15-25 seconds

Total switching overhead: 20-35 seconds

Optimization:
  - Use service stop/start (faster than Docker container restart)
  - Keep Redis cache warm
  - Preload common LoRAs
```

### Model Manager Implementation

```python
class ModelManager:
    """
    Manages dynamic model loading/unloading
    Ensures only one heavy model loaded at a time
    """

    def __init__(self):
        self.current_heavy_model: str = None  # "llm" or "sdxl" or None
        self.llm_service_running: bool = False
        self.sdxl_pipeline_loaded: bool = False

    async def switch_to_llm(self):
        """Switch to LLM mode"""
        if self.current_heavy_model == "llm":
            return  # Already in LLM mode

        # Unload SDXL if loaded
        if self.sdxl_pipeline_loaded:
            await self._unload_sdxl()

        # Start LLM service
        await self._start_llm_service()
        self.current_heavy_model = "llm"

    async def switch_to_sdxl(self):
        """Switch to SDXL mode"""
        if self.current_heavy_model == "sdxl":
            return  # Already in SDXL mode

        # Stop LLM service if running
        if self.llm_service_running:
            await self._stop_llm_service()

        # Load SDXL pipeline
        await self._load_sdxl()
        self.current_heavy_model = "sdxl"

    async def switch_to_tts(self, allow_with_llm: bool = True):
        """
        Switch to TTS mode
        TTS is light (~3-4GB), can run with either LLM or standalone
        """
        if allow_with_llm and self.llm_service_running:
            # Keep LLM running, load TTS alongside
            await self._load_tts()
        else:
            # Unload all heavy models first
            if self.current_heavy_model:
                await self._unload_all()
            await self._load_tts()

    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage"""
        import torch
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
            "current_model": self.current_heavy_model
        }
```

---

## 📊 Module VRAM Profiles

### LLM Backend Module

```yaml
Models:
  Qwen2.5-VL-7B:
    VRAM: 13.8GB
    Speed: ~40 tok/s
    Use case: Vision + language tasks

  Qwen2.5-14B:
    VRAM: 11.5GB
    Speed: ~45 tok/s
    Use case: Reasoning, planning

  Qwen2.5-Coder-7B:
    VRAM: 13.5GB
    Speed: ~42 tok/s
    Use case: Prompt engineering

Configuration:
  gpu_memory_utilization: 0.85
  max_model_len: 4096 (VL), 8192 (others)
  tensor_parallel_size: 1
  dtype: "half" (FP16)

Switching Strategy:
  - Only one LLM runs at a time
  - Selection via docker-compose profiles
  - 15-25 seconds switching time
```

### Image Generation Module

```yaml
SDXL Base:
  VRAM: ~10-11GB
  Precision: FP16
  Generation time: ~15s (30 steps)

SDXL + LoRA:
  VRAM: ~11-12GB
  LoRA overhead: ~1GB per adapter
  Max LoRAs: 3-4 simultaneously

SDXL + ControlNet:
  VRAM: ~13-15GB
  ControlNet overhead: ~1-2GB per network
  Max ControlNets: 1-2 safely

Optimization:
  - Use PyTorch SDPA (not xformers)
  - enable_vae_slicing: true
  - enable_vae_tiling: true (for >1024x1024)
  - Sequential LoRA loading

Critical Rule:
  - LLM must be stopped before loading SDXL
  - Cannot coexist with LLM
```

### Voice Synthesis Module

```yaml
GPT-SoVITS Inference:
  VRAM: ~3-4GB
  Can run with: LLM stopped OR SDXL stopped
  Ideal: Run standalone or with LLM stopped

GPT-SoVITS Training:
  VRAM: ~8-10GB
  Requires: All other models stopped
  Training time: ~2-4 hours (100 epochs)

Flexibility:
  - Light enough to coexist with stopped heavy models
  - Can run alongside LLM if needed (tight but possible)
  - Prefer standalone for best performance
```

---

## 🎯 Optimization Techniques

### 1. Model Quantization

```yaml
LLM Models:
  Current: FP16 (half precision)
  Possible: AWQ 4-bit quantization
  Benefit: ~2x VRAM reduction
  Trade-off: Slight quality loss

SDXL Models:
  Current: FP16
  Possible: NF4 quantization for LoRAs
  Benefit: ~2x LoRA VRAM reduction
  Trade-off: Minimal quality impact

Recommendation:
  - Keep FP16 for production quality
  - Use quantization only if VRAM critical
```

### 2. Attention Optimization

```yaml
PyTorch SDPA (Used):
  Advantages:
    - Native PyTorch integration
    - Compatible with PyTorch 2.7.0
    - No external dependencies
    - Consistent across vLLM and Diffusers

  Configuration:
    VLLM_ATTENTION_BACKEND: "TORCH_SDPA"
    XFORMERS_DISABLED: "1"

  Performance:
    - Flash Attention equivalent on RTX 5080
    - Efficient memory usage
    - Optimized for Blackwell architecture

xformers (Forbidden):
  Why not used:
    - Conflicts with PyTorch 2.7.0
    - Inconsistent with vLLM config
    - Installation issues with CUDA 12.8
```

### 3. Memory Management

```yaml
GPU Memory Utilization:
  vLLM: 0.85 (conservative)
  Reasoning: Leave buffer for:
    - CUDA kernel overhead
    - Temporary tensors
    - Gradient accumulation (if training)

VAE Optimizations (SDXL):
  enable_vae_slicing: true
    - Reduces VRAM for VAE decoding
    - Minimal quality impact

  enable_vae_tiling: true
    - For resolutions > 1024x1024
    - Process image in tiles

CPU Offloading:
  enable_cpu_offload: false
    - RTX 5080 has sufficient VRAM
    - CPU offloading adds latency
    - Only use if desperate for VRAM
```

### 4. Batch Processing

```yaml
Image Generation:
  Batch size: 1 (single image at a time)
  Reasoning: 16GB VRAM limits batch size
  Sequential processing for multiple images

Voice Synthesis:
  Batch size: 1-2
  Can batch small texts together
  Longer texts must be processed individually

LLM Inference:
  Batch size: Dynamic (handled by vLLM)
  Request batching for throughput
  Limited by max_model_len
```

---

## 🔧 Configuration Best Practices

### Environment Variables

```bash
# CRITICAL: Always set these
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export XFORMERS_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0

# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Cache paths
export HF_HOME=/mnt/c/AI_LLM_projects/ai_warehouse/cache/huggingface
export VLLM_CACHE_DIR=/mnt/c/AI_LLM_projects/ai_warehouse/cache/vllm
export DIFFUSERS_CACHE=/mnt/c/AI_LLM_projects/ai_warehouse/cache/diffusers

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Docker Configuration

```yaml
# docker-compose.yml snippet
services:
  vllm_service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Single GPU
              capabilities: [gpu]

    environment:
      - VLLM_ATTENTION_BACKEND=TORCH_SDPA
      - XFORMERS_DISABLED=1
      - CUDA_VISIBLE_DEVICES=0
```

### Python Code Configuration

```python
# SDXL Pipeline
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# CRITICAL: Use PyTorch SDPA (not xformers)
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()

# DO NOT call: pipeline.enable_xformers_memory_efficient_attention()

pipeline = pipeline.to("cuda")
```

---

## 📈 Performance Benchmarks

### LLM Backend

```yaml
Qwen2.5-VL-7B:
  Throughput: ~40 tokens/second
  Latency (first token): ~0.8s
  VRAM: 13.8GB
  Batch size: Dynamic (vLLM managed)

Qwen2.5-14B:
  Throughput: ~45 tokens/second
  Latency (first token): ~0.6s
  VRAM: 11.5GB
  Batch size: Dynamic

Qwen2.5-Coder-7B:
  Throughput: ~42 tokens/second
  Latency (first token): ~0.7s
  VRAM: 13.5GB
  Batch size: Dynamic
```

### Image Generation

```yaml
SDXL Base (1024x1024, 30 steps):
  Generation time: ~12-15 seconds
  VRAM: 10-11GB

SDXL + LoRA (1 adapter):
  Generation time: ~15-18 seconds
  VRAM: 11-12GB

SDXL + ControlNet (OpenPose):
  Generation time: ~20-25 seconds
  VRAM: 13-15GB

Model loading time: ~5-8 seconds
```

### Voice Synthesis

```yaml
GPT-SoVITS (short sentence, 1-2s audio):
  Generation time: ~2-3 seconds
  VRAM: 3-4GB

GPT-SoVITS (medium sentence, 3-5s audio):
  Generation time: ~4-5 seconds
  VRAM: 3-4GB

GPT-SoVITS (long sentence, 10s audio):
  Generation time: ~8-10 seconds
  VRAM: 3-4GB
```

---

## 🚨 Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
```python
# 1. Check current VRAM usage
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Ensure only one heavy model loaded
# Use ModelManager to switch properly

# 4. Reduce batch size
# For image generation: batch_size=1
# For LLM: reduce max_model_len

# 5. Lower GPU memory utilization
# vLLM: gpu_memory_utilization=0.80 (instead of 0.85)
```

### Issue 2: xformers Conflict

**Symptoms:**
```
ImportError: xformers not available
AttributeError: 'Attention' object has no attribute 'xformers'
```

**Solutions:**
```bash
# 1. Verify xformers is disabled
pip list | grep xformers  # Should return nothing

# 2. If installed, uninstall
pip uninstall -y xformers

# 3. Ensure environment variables set
export XFORMERS_DISABLED=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# 4. For diffusers, explicitly disable
python -c "
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(...)
pipe.enable_attention_slicing()  # Use this
# NOT: pipe.enable_xformers_memory_efficient_attention()
"
```

### Issue 3: Slow Model Switching

**Symptoms:**
- Model switching takes > 60 seconds
- Services fail to start/stop

**Solutions:**
```bash
# 1. Use service control (faster than container restart)
bash llm_backend/scripts/stop_service.sh qwen-vl
bash llm_backend/scripts/start_service.sh qwen-14b

# 2. Keep Redis running (cache persistence)
# Don't stop Redis between switches

# 3. Preload models in AI Warehouse
# Avoid downloading during switch

# 4. Use SSDs for model storage
# Faster loading from disk
```

### Issue 4: PyTorch Version Conflict

**Symptoms:**
```
ERROR: Package conflicts with PyTorch 2.7.0
```

**Solutions:**
```bash
# 1. NEVER modify PyTorch version
# PyTorch 2.7.0 + CUDA 12.8 is IMMUTABLE

# 2. Find compatible package versions
pip install package_name --no-deps
# Then manually install dependencies compatible with PyTorch 2.7.0

# 3. If absolutely necessary, use older package
pip install package_name==older_version

# 4. Document all version pins in requirements.txt
```

---

## 📋 Pre-Implementation Checklist

Before implementing any module, verify:

### Hardware
- [ ] RTX 5080 16GB VRAM confirmed
- [ ] Single GPU operation configured
- [ ] CUDA 12.8 installed and working
- [ ] Sufficient system RAM (64GB available)

### PyTorch Environment
- [ ] PyTorch 2.7.0 installed
- [ ] CUDA 12.8 compatible
- [ ] xformers uninstalled
- [ ] VLLM_ATTENTION_BACKEND=TORCH_SDPA set
- [ ] XFORMERS_DISABLED=1 set

### Storage
- [ ] AI Warehouse paths configured
- [ ] Cache directories created
- [ ] Sufficient disk space (500GB+ free)
- [ ] SSDs used for model storage

### Model Management
- [ ] Model switching strategy planned
- [ ] VRAM budgets calculated
- [ ] ModelManager implementation ready

### Testing
- [ ] VRAM monitoring tools ready
- [ ] Performance benchmarking plan
- [ ] Failure recovery procedures documented

---

## 📚 References

- **NVIDIA RTX 5080**: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/
- **PyTorch SDPA**: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- **vLLM Configuration**: https://docs.vllm.ai/en/latest/
- **Diffusers Memory Optimization**: https://huggingface.co/docs/diffusers/optimization/memory

---

**Last Updated:** 2025-11-17
**Hardware:** RTX 5080 16GB | AMD R9 9950X | 64GB DDR5
**Environment:** PyTorch 2.7.0 + CUDA 12.8 (IMMUTABLE)
