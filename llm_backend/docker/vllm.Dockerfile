# vLLM Service Dockerfile
# For RTX 5080 16GB with PyTorch 2.7.0 + CUDA 12.8
# CRITICAL: Uses PyTorch native SDPA, NO xformers

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# CRITICAL: Install vLLM compatible with PyTorch 2.7.0 + CUDA 12.8
# Do NOT install xformers
RUN pip3 install --no-cache-dir \
    vllm>=0.3.0 \
    && pip3 uninstall -y xformers || true

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    transformers>=4.40.0 \
    tokenizers>=0.19.0 \
    accelerate>=0.27.0 \
    sentencepiece>=0.2.0 \
    protobuf>=4.25.0

# Create application directory
WORKDIR /app

# Copy service scripts
COPY llm_backend/services /app/llm_backend/services

# Make scripts executable
RUN chmod +x /app/llm_backend/services/*/start.sh

# Create cache directories
RUN mkdir -p /models/cache/{huggingface,vllm,torch}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# CRITICAL: Disable xformers, use PyTorch SDPA
ENV VLLM_ATTENTION_BACKEND=TORCH_SDPA
ENV XFORMERS_DISABLED=1
ENV VLLM_USE_TRITON_FLASH_ATTN=0

# Expose ports (8000, 8001, 8002)
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (override in docker-compose)
CMD ["bash"]
