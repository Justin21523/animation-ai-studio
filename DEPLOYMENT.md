# Animation AI Studio - Deployment Guide

**Complete deployment guide for production and development environments**

Date: 2025-11-17

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Configuration](#configuration)
5. [Running Services](#running-services)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## 🖥️ System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 22.04+), Windows 11 with WSL2, macOS 13+
- **CPU**: 8 cores (Intel i7 / AMD Ryzen 7 or better)
- **RAM**: 32GB
- **GPU**: NVIDIA RTX 3080 (10GB VRAM) or better
- **Storage**: 100GB SSD free space
- **Python**: 3.10 or 3.11 (NOT 3.12)

### Recommended Requirements (Production)

- **OS**: Ubuntu 22.04 LTS
- **CPU**: 16 cores (Intel Xeon / AMD EPYC)
- **RAM**: 64GB
- **GPU**: NVIDIA RTX 5080 (16GB VRAM) or RTX 4090 (24GB)
- **Storage**: 500GB NVMe SSD
- **Python**: 3.10
- **CUDA**: 12.8

### Software Dependencies

- CUDA Toolkit 12.8
- cuDNN 8.9+
- Docker (optional, for containerized deployment)
- Git
- ffmpeg (for video processing)

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
cd /mnt/c/AI_LLM_projects
git clone <repository-url> animation-ai-studio
cd animation-ai-studio
```

### 2. Run Setup Script

```bash
bash deploy/setup.sh
```

### 3. Activate Environment

```bash
source venv/bin/activate
```

### 4. Start Services

```bash
bash start.sh
```

### 5. Test

```bash
python tests/run_all_tests.py
```

### 6. Try Creative Studio

```bash
python scripts/applications/creative_studio/cli.py list
```

---

## 🔧 Detailed Setup

### Step 1: System Preparation

#### Ubuntu/Linux

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y git ffmpeg libsm6 libxext6 libxrender-dev
sudo apt install -y build-essential cmake

# Install CUDA (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads
```

#### Windows (WSL2)

```powershell
# Install WSL2 with Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Ubuntu instructions above
```

### Step 2: Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install all requirements
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Configure AI Warehouse

```bash
# Set up AI Warehouse path (if not already)
export AI_WAREHOUSE="/mnt/c/AI_LLM_projects/ai_warehouse"

# Create directory structure
mkdir -p $AI_WAREHOUSE/models/{llm,cv,audio,segmentation}
mkdir -p $AI_WAREHOUSE/cache
```

### Step 5: Download Models

Models will be auto-downloaded on first use, or manually:

```bash
# LLM models (Qwen)
# Download from Hugging Face:
# - Qwen/Qwen2.5-VL-7B-Instruct
# - Qwen/Qwen2.5-14B-Instruct
# - Qwen/Qwen2.5-Coder-7B-Instruct

# SAM2 models
# Handled by 3d-animation-lora-pipeline project

# SDXL models
# stabilityai/stable-diffusion-xl-base-1.0
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Project paths
PROJECT_ROOT=/mnt/c/AI_LLM_projects/animation-ai-studio
AI_WAREHOUSE=/mnt/c/AI_LLM_projects/ai_warehouse

# LLM Backend
LLM_BACKEND_HOST=0.0.0.0
LLM_BACKEND_PORT=8000
REDIS_HOST=localhost
REDIS_PORT=6379

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_UTILIZATION=0.85

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Model paths
QWEN_VL_MODEL_PATH=$AI_WAREHOUSE/models/llm/Qwen2.5-VL-7B-Instruct
QWEN_14B_MODEL_PATH=$AI_WAREHOUSE/models/llm/Qwen2.5-14B-Instruct
QWEN_CODER_MODEL_PATH=$AI_WAREHOUSE/models/llm/Qwen2.5-Coder-7B-Instruct

# SAM2 (from LoRA pipeline)
SAM2_CHECKPOINT=$AI_WAREHOUSE/models/segmentation/sam2/sam2_hiera_base_plus.pt

# SDXL
SDXL_MODEL_PATH=$AI_WAREHOUSE/models/diffusion/stable-diffusion-xl-base-1.0
```

### Module Configurations

All module configs are in `configs/` directory:

```
configs/
├── llm_backend/      # LLM Backend configurations
├── generation/       # Image/Voice generation
├── rag/             # RAG System
├── agent/           # Agent Framework
└── model_manager/   # VRAM management
```

Edit as needed for your environment.

---

## 🏃 Running Services

### Development Mode

```bash
# Start all services
bash start.sh

# Start with monitoring
bash start.sh --monitoring

# Start with specific model
bash start.sh --model qwen-vl
```

### Individual Services

```bash
# LLM Backend only
cd llm_backend
bash scripts/start.sh qwen-14b

# Check health
bash scripts/health.sh

# View logs
bash scripts/logs.sh

# Stop
bash scripts/stop.sh
```

### Production Mode

```bash
# Use Docker Compose (recommended)
docker-compose up -d

# Or systemd services (Linux)
sudo systemctl start animation-ai-studio
```

---

## 🧪 Testing

### Run All Tests

```bash
python tests/run_all_tests.py
```

### Run Specific Module Tests

```bash
# Agent Framework
python tests/run_all_tests.py --module agent

# Video Editing
python tests/run_all_tests.py --module editing

# Creative Studio
python tests/run_all_tests.py --module creative
```

### Run with Coverage

```bash
python tests/run_all_tests.py --coverage
```

### Manual Testing

```bash
# Test Creative Studio CLI
python scripts/applications/creative_studio/cli.py list

# Test parody generation (with sample video)
python scripts/applications/creative_studio/cli.py parody \
    test_data/sample.mp4 output.mp4 --style dramatic

# Test analysis
python scripts/applications/creative_studio/cli.py analyze \
    test_data/sample.mp4 --visual --output analysis.json
```

---

## 📊 Monitoring

### Prometheus + Grafana (LLM Backend)

```bash
# Start monitoring stack
bash start.sh --monitoring

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Metrics Endpoints

- LLM Backend metrics: `http://localhost:8000/metrics`
- Gateway metrics: `http://localhost:8000/gateway/metrics`

### Log Monitoring

```bash
# View real-time logs
bash llm_backend/scripts/logs.sh

# View specific service logs
tail -f logs/llm_backend.log
tail -f logs/creative_studio.log
```

### GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or use built-in monitor
python scripts/core/model_management/vram_monitor.py
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Option 1: Use smaller model
bash start.sh --model qwen-vl  # 7B instead of 14B

# Option 2: Reduce GPU memory utilization
# Edit configs/llm_backend/vllm_config.yaml
gpu_memory_utilization: 0.75  # Reduce from 0.85

# Option 3: Use SAM2 small instead of base
# Edit configs (if applicable)
```

#### 2. Port Already in Use

**Symptoms**: `Address already in use`

**Solutions**:
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>

# Or use different port
LLM_BACKEND_PORT=8001 bash start.sh
```

#### 3. Import Errors

**Symptoms**: `ModuleNotFoundError`

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.10 or 3.11

# Activate virtual environment
source venv/bin/activate
```

#### 4. Model Download Failures

**Symptoms**: Models not found or download errors

**Solutions**:
```bash
# Manual download with huggingface-cli
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir $AI_WAREHOUSE/models/llm/Qwen2.5-14B-Instruct

# Or download via browser and place in AI_WAREHOUSE
```

#### 5. Video Processing Errors

**Symptoms**: MoviePy or OpenCV errors

**Solutions**:
```bash
# Install ffmpeg
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS

# Reinstall opencv
pip uninstall opencv-python
pip install opencv-python-headless
```

### Getting Help

1. Check logs: `tail -f logs/*.log`
2. Run health check: `bash llm_backend/scripts/health.sh`
3. Check GPU: `nvidia-smi`
4. Verify environment: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🌐 Production Deployment

### Docker Deployment (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm-backend:
    build: ./llm_backend
    ports:
      - "8000:8000"
    volumes:
      - /mnt/c/AI_LLM_projects/ai_warehouse:/ai_warehouse
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./llm_backend/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./llm_backend/monitoring/grafana:/etc/grafana/provisioning
```

```bash
# Deploy
docker-compose up -d

# Scale (if multiple GPUs)
docker-compose up -d --scale llm-backend=2
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/animation-ai-studio.service
[Unit]
Description=Animation AI Studio - LLM Backend
After=network.target

[Service]
Type=simple
User=ai-studio
WorkingDirectory=/home/ai-studio/animation-ai-studio
Environment="PATH=/home/ai-studio/animation-ai-studio/venv/bin"
ExecStart=/home/ai-studio/animation-ai-studio/start.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable animation-ai-studio
sudo systemctl start animation-ai-studio

# Check status
sudo systemctl status animation-ai-studio
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/animation-ai-studio
server {
    listen 80;
    server_name animation-ai-studio.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/animation-ai-studio /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Security Best Practices

1. **API Authentication**: Add API keys to FastAPI Gateway
2. **HTTPS**: Use SSL certificates (Let's Encrypt)
3. **Firewall**: Restrict ports to necessary services
4. **User Permissions**: Run services as non-root user
5. **Secrets Management**: Use environment variables or secrets manager
6. **Rate Limiting**: Implement rate limiting on API endpoints
7. **Monitoring**: Set up alerts for failures and anomalies

### Backup Strategy

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups/animation-ai-studio"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configurations
tar -czf $BACKUP_DIR/configs_$DATE.tar.gz configs/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Backup generated outputs (optional)
# tar -czf $BACKUP_DIR/outputs_$DATE.tar.gz outputs/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

---

## 📈 Performance Tuning

### GPU Optimization

```yaml
# configs/llm_backend/vllm_config.yaml
gpu_memory_utilization: 0.85  # Adjust based on VRAM
max_model_len: 8192           # Reduce for more VRAM
tensor_parallel_size: 1       # Increase if multiple GPUs
```

### Model Selection

- **Development**: qwen-vl-7b (13.8GB VRAM, ~40 tok/s)
- **Balanced**: qwen-14b (11.5GB VRAM, ~45 tok/s)
- **Code Tasks**: qwen-coder-7b (13.5GB VRAM, ~42 tok/s)

### Caching

```yaml
# Enable Redis caching for repeated queries
redis:
  enabled: true
  ttl: 3600  # Cache TTL in seconds
```

---

## 📝 Summary

**Deployment Checklist**:

- ✅ System requirements met
- ✅ Python 3.10 environment
- ✅ CUDA 12.8 installed
- ✅ Dependencies installed
- ✅ Models downloaded
- ✅ Configuration completed
- ✅ Services started
- ✅ Tests passed
- ✅ Monitoring enabled
- ✅ Backup strategy in place

**Production Ready**: Follow Docker + Nginx + Systemd setup for robust deployment.

---

**Last Updated**: 2025-11-17
**Version**: 1.0.0
**Maintainer**: Animation AI Studio Team
