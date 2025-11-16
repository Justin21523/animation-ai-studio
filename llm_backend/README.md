# LLM Backend - Self-Hosted Inference Services

**Hardware:** RTX 5080 16GB VRAM
**PyTorch:** 2.7.0 + CUDA 12.8
**Attention:** Native PyTorch SDPA (NO xformers)

---

## 📋 Overview

Self-hosted LLM inference backend for Animation AI Studio, optimized for RTX 5080 16GB VRAM with PyTorch 2.7.0 and CUDA 12.8.

### Architecture

```
User Application
       ↓
FastAPI Gateway (Port 7000)
       ↓
   Load Balancer
       ↓
┌──────┴───────┬──────────┐
│              │          │
Qwen-VL-7B  Qwen-14B  Qwen-Coder-7B
(Port 8000)  (8001)   (8002)
```

**Important:** RTX 5080 16GB can only run **ONE model at a time**.

---

## 🚀 Quick Start

### Prerequisites

```bash
# System
- Ubuntu 22.04 / WSL2
- Docker + nvidia-docker2
- NVIDIA RTX 5080 16GB
- PyTorch 2.7.0 + CUDA 12.8 (pre-installed in conda ai_env)

# Storage
- ~50GB disk space for models
- ~2GB disk space for Docker images
```

### Installation

```bash
# 1. Navigate to project root
cd /mnt/c/AI_LLM_projects/animation-ai-studio

# 2. Start services (interactive model selection)
bash llm_backend/scripts/start_all.sh

# 3. Check health
bash llm_backend/scripts/health_check.sh
```

### Choose Your Model

When starting, you'll be asked to select ONE model:

1. **Qwen2.5-VL-7B** - Multimodal (vision + chat)
2. **Qwen2.5-14B** - Reasoning (complex tasks)
3. **Qwen2.5-Coder-7B** - Code generation

---

## 📊 Model Specifications

| Model | VRAM | Context | Speed | Capabilities |
|-------|------|---------|-------|--------------|
| Qwen2.5-VL-7B | ~14GB | 32K | 30-50 tok/s | Vision, Chat, Embeddings |
| Qwen2.5-14B | ~12GB | 32K | 30-50 tok/s | Reasoning, Chat |
| Qwen2.5-Coder-7B | ~14GB | 32K | 30-50 tok/s | Code, Chat |

---

## 🔧 Management Scripts

### Start Services

```bash
# Interactive start (choose model)
bash llm_backend/scripts/start_all.sh
```

### Stop Services

```bash
# Stop all services
bash llm_backend/scripts/stop_all.sh
```

### Switch Models

```bash
# Switch between models
bash llm_backend/scripts/switch_model.sh
```

### Health Check

```bash
# Check all services
bash llm_backend/scripts/health_check.sh
```

### View Logs

```bash
# View gateway logs
bash llm_backend/scripts/logs.sh gateway

# View model logs
bash llm_backend/scripts/logs.sh qwen-vl

# Available: gateway, qwen-vl, qwen-14b, qwen-coder, redis, all
```

---

## 🌐 Service URLs

### Gateway & APIs

```
FastAPI Gateway: http://localhost:7000
Health Check:    http://localhost:7000/health
List Models:     http://localhost:7000/models
```

### Model Services (when loaded)

```
Qwen-VL:    http://localhost:8000
Qwen-14B:   http://localhost:8001
Qwen-Coder: http://localhost:8002
```

### Monitoring

```
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000 (admin/admin)
```

---

## 💻 Usage Examples

### Python Client

```python
import asyncio
from scripts.core.llm_client import LLMClient

async def main():
    async with LLMClient() as client:
        # Check health
        health = await client.health_check()
        print(health)

        # Chat
        response = await client.chat(
            model="qwen-14b",
            messages=[
                {"role": "user", "content": "Explain quantum computing"}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        print(response['choices'][0]['message']['content'])

asyncio.run(main())
```

### cURL Examples

```bash
# Health check
curl http://localhost:7000/health

# List models
curl http://localhost:7000/models

# Chat completion
curl -X POST http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-14b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
# Critical settings (DO NOT MODIFY)
VLLM_ATTENTION_BACKEND=TORCH_SDPA
XFORMERS_DISABLED=1

# GPU configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.85

# Model paths
HF_HOME=/mnt/c/AI_LLM_projects/ai_warehouse/cache/huggingface
```

### Model Configuration

Edit `llm_backend/services/{model}/config.yaml` to adjust:
- GPU memory utilization
- Max context length
- Batch size
- Caching settings

---

## 🔍 Troubleshooting

### OOM Errors

```bash
# Reduce GPU memory utilization
# Edit docker-compose.yml:
GPU_MEMORY_UTILIZATION=0.80  # Instead of 0.85
```

### Model Won't Start

```bash
# Check logs
docker-compose logs qwen-vl

# Check GPU availability
nvidia-smi

# Verify VRAM free
# Should have ~14GB free for 7B/14B models
```

### Slow Inference

```bash
# Verify attention backend
docker exec llm-qwen-vl env | grep VLLM_ATTENTION_BACKEND
# Should show: TORCH_SDPA

# Check GPU utilization
nvidia-smi dmon -s u
```

### Gateway Errors

```bash
# Check Redis
docker exec llm-redis redis-cli ping

# Restart gateway
docker-compose restart gateway
```

---

## 📈 Performance Optimization

### Batch Processing

For multiple requests, use batching:

```python
# Queue multiple requests
tasks = [
    client.chat(model="qwen-14b", messages=[...])
    for _ in range(10)
]
results = await asyncio.gather(*tasks)
```

### Caching

Gateway automatically caches responses. Clear cache if needed:

```bash
docker exec llm-redis redis-cli FLUSHDB
```

### Prefix Caching

vLLM automatically caches KV-cache for common prefixes, improving performance for similar requests.

---

## 🛡️ Safety Limits

### Automatic Protections

- **Max sequence length:** 32,768 tokens
- **GPU memory threshold:** 95%
- **Request timeout:** 300 seconds
- **Concurrent requests:** 10 (configurable)

### Manual Limits

Edit `llm_backend/config/vllm_config.yaml` to adjust safety limits.

---

## 🔄 Upgrading

### Update Models

```bash
# Pull latest model versions
docker exec llm-qwen-vl bash -c \
  "huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct"
```

### Update vLLM

```bash
# Rebuild Docker images
cd llm_backend/docker
docker-compose build --no-cache
```

---

## 📚 Documentation

- **Architecture:** `LLM_BACKEND_ARCHITECTURE.md`
- **Hardware Specs:** `HARDWARE_SPECS.md`
- **Model Options:** `config/model_options.yaml`
- **Implementation Guide:** `IMPLEMENTATION_ROADMAP.md`

---

## ⚠️ Important Notes

### DO NOT

- ❌ Modify PyTorch 2.7.0
- ❌ Change CUDA 12.8
- ❌ Install xformers
- ❌ Run multiple models simultaneously (OOM)

### MUST

- ✅ Use PyTorch native SDPA
- ✅ Set `VLLM_ATTENTION_BACKEND=TORCH_SDPA`
- ✅ Keep 1.5-2GB VRAM headroom
- ✅ Switch models before loading different one

---

## 🤝 Support

**Issues:** Check logs first
```bash
bash llm_backend/scripts/logs.sh {service}
```

**GPU Problems:**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Cache Issues:**
```bash
# Clear HuggingFace cache
rm -rf /mnt/c/AI_LLM_projects/ai_warehouse/cache/huggingface/*

# Clear vLLM cache
rm -rf /mnt/c/AI_LLM_projects/ai_warehouse/cache/vllm/*
```

---

## 📞 Quick Reference

```bash
# Start
bash llm_backend/scripts/start_all.sh

# Stop
bash llm_backend/scripts/stop_all.sh

# Switch
bash llm_backend/scripts/switch_model.sh

# Health
bash llm_backend/scripts/health_check.sh

# Logs
bash llm_backend/scripts/logs.sh {service}

# Python
python -c "from scripts.core.llm_client import LLMClient; import asyncio; asyncio.run(LLMClient().health_check())"
```

---

**Version:** 1.0.0
**Last Updated:** 2025-11-16
**Status:** Production Ready
