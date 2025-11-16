#!/bin/bash
#
# Qwen2.5-14B vLLM Service Startup Script
# For reasoning tasks on RTX 5080 16GB VRAM
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
# Configuration (environment variable overrides)
# ============================================================================

# Model path
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}"
# Alternative: Use AI warehouse
# MODEL_PATH="${MODEL_PATH:-/mnt/c/AI_LLM_projects/ai_warehouse/models/llm/qwen/Qwen2.5-14B-Instruct}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-14b}"

# GPU configuration (RTX 5080 16GB)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
QUANTIZATION="${QUANTIZATION:-}"  # No quantization for 14B

# Performance configuration
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"  # 32K context
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Advanced options
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# ============================================================================
# Pre-flight checks
# ============================================================================

echo "========================================="
echo "🚀 Starting Qwen2.5-14B vLLM Service"
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

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Verify sufficient VRAM for 14B model
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$TOTAL_MEM" -lt 12000 ]; then
    echo "⚠️  WARNING: GPU memory ($TOTAL_MEM MB) may be insufficient for Qwen2.5-14B"
    echo "   Recommended: 16GB+ VRAM"
    echo ""
fi

# ============================================================================
# Build vLLM command
# ============================================================================

VLLM_CMD="python -m vllm.entrypoints.openai.api_server"

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

# Add quantization
if [ "$QUANTIZATION" != "null" ] && [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

# Add caching options
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi

if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    VLLM_ARGS+=(--enable-chunked-prefill)
fi

# Add trust remote code
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_ARGS+=(--trust-remote-code)
fi

# ============================================================================
# Launch vLLM
# ============================================================================

echo "Starting vLLM with command:"
echo "$VLLM_CMD ${VLLM_ARGS[@]}"
echo ""
echo "========================================="
echo ""

exec $VLLM_CMD "${VLLM_ARGS[@]}"
