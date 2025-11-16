#!/bin/bash
#
# Qwen2.5-VL-7B vLLM Service Startup Script
# Optimized for RTX 5080 16GB VRAM
# PyTorch 2.7.0 + CUDA 12.8 + Native SDPA (NO xformers)
#

set -e

# ============================================================================
# CRITICAL: Force PyTorch SDPA, Disable xformers
# ============================================================================
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export XFORMERS_DISABLED=1
export VLLM_USE_TRITON_FLASH_ATTN=0

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Single GPU (RTX 5080)
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# Configuration (can be overridden by environment variables)
# ============================================================================

# Model path - can use HuggingFace ID or local path from AI warehouse
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
# Alternative: Use AI warehouse path
# MODEL_PATH="${MODEL_PATH:-/mnt/c/AI_LLM_projects/ai_warehouse/models/llm/qwen/Qwen2.5-VL-7B-Instruct}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-vl-7b}"

# GPU configuration (RTX 5080 16GB)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"  # Conservative for 16GB
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"  # Single GPU
DTYPE="${DTYPE:-auto}"
QUANTIZATION="${QUANTIZATION:-}"  # No quantization for 7B

# Performance configuration
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Caching
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"

# Other options
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-false}"

# ============================================================================
# Pre-flight checks
# ============================================================================

echo "========================================="
echo "🚀 Starting Qwen2.5-VL-7B vLLM Service"
echo "   Optimized for RTX 5080 16GB"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "  GPU Memory: ${GPU_MEMORY_UTILIZATION}"
echo "  Quantization: $QUANTIZATION"
echo "  Max Context: $MAX_MODEL_LEN"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. CUDA GPUs required."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  WARNING: Port $PORT is already in use"
    echo "   Attempting to continue anyway..."
fi

# ============================================================================
# Build vLLM command
# ============================================================================

VLLM_CMD="python -m vllm.entrypoints.openai.api_server"

# Add all parameters
VLLM_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
)

# Add quantization if specified
if [ "$QUANTIZATION" != "null" ] && [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

# Add prefix caching if enabled
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi

# Add trust remote code if needed
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_ARGS+=(--trust-remote-code)
fi

# Add log stats control
if [ "$DISABLE_LOG_STATS" = "false" ]; then
    # Enable detailed logging
    VLLM_ARGS+=(--disable-log-stats)
fi

# ============================================================================
# Launch vLLM
# ============================================================================

echo "Starting vLLM with command:"
echo "$VLLM_CMD ${VLLM_ARGS[@]}"
echo ""
echo "========================================="
echo ""

# Execute vLLM
exec $VLLM_CMD "${VLLM_ARGS[@]}"

# Note: This script will not reach here as exec replaces the process
