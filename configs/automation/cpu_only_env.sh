#!/bin/bash
# CPU-Only Environment Configuration
#
# Source this file before running any automation workflows to ensure CPU-only execution.
# This enforces GPU isolation and sets up proper cache locations per data_model_structure.md.
#
# Usage:
#   source configs/automation/cpu_only_env.sh
#   python scripts/automation/cli.py run workflow.yaml
#
# Author: Animation AI Studio Team
# Last Modified: 2025-12-02

# ============================================================================
# GPU Isolation (CRITICAL)
# ============================================================================

# Hide all CUDA devices from process (most important setting)
export CUDA_VISIBLE_DEVICES=""

# Force PyTorch to use CPU device
export TORCH_DEVICE="cpu"

# Disable CUDA memory configuration
export PYTORCH_CUDA_ALLOC_CONF=""

# JAX CPU-only mode
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="True"

# TensorFlow GPU prevention (if ever needed)
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TF_FORCE_GPU_ALLOW_GROWTH="false"

# ============================================================================
# CPU Threading (utilize all 32 cores)
# ============================================================================

# Utilize all 32 threads for CPU-intensive tasks
# Training uses GPU, so CPU threads are available for automation
export OMP_NUM_THREADS=32

# Limit MKL threads (Intel Math Kernel Library)
export MKL_NUM_THREADS=32

# Limit OpenBLAS threads
export OPENBLAS_NUM_THREADS=32

# NumPy threading
export NUMEXPR_NUM_THREADS=32

# ============================================================================
# Cache Locations (per data_model_structure.md)
# ============================================================================

# HuggingFace cache (models, datasets, transformers)
export HF_HOME=/mnt/c/ai_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface
export HF_DATASETS_CACHE=/mnt/c/ai_cache/huggingface/datasets

# PyTorch cache (models, hub)
export TORCH_HOME=/mnt/c/ai_cache/torch

# General XDG cache (pip, etc.)
export XDG_CACHE_HOME=/mnt/c/ai_cache

# Pip cache
export PIP_CACHE_DIR=/mnt/c/ai_cache/pip

# ============================================================================
# Process Priority (OOM protection)
# ============================================================================

# Set OOM score adjustment (killable before training processes)
# Training processes use -300, automation uses +500
# This ensures automation is killed first if system runs out of memory
if [ -w /proc/self/oom_score_adj ]; then
    echo 500 > /proc/self/oom_score_adj
fi

# Set nice priority (lower priority than training)
# Positive values = lower priority (0 is normal, 19 is lowest)
renice -n 10 $$ > /dev/null 2>&1

# ============================================================================
# Python-specific settings
# ============================================================================

# Disable Python bytecode writing (reduces disk I/O)
export PYTHONDONTWRITEBYTECODE=1

# Unbuffered Python output (for real-time logging)
export PYTHONUNBUFFERED=1

# ============================================================================
# Verification
# ============================================================================

# Print confirmation (if running interactively)
if [ -t 1 ]; then
    echo "✓ CPU-only environment configured"
    echo "  CUDA_VISIBLE_DEVICES: '${CUDA_VISIBLE_DEVICES}'"
    echo "  TORCH_DEVICE: '${TORCH_DEVICE}'"
    echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "  HF_HOME: ${HF_HOME}"
    echo "  TORCH_HOME: ${TORCH_HOME}"
    echo "  OOM score: $(cat /proc/self/oom_score_adj 2>/dev/null || echo 'N/A')"
    echo ""
    echo "Ready to run CPU-only automation workflows."
fi

# ============================================================================
# Helper Functions
# ============================================================================

# Function to verify CPU-only configuration
verify_cpu_only() {
    echo "Verifying CPU-only configuration..."

    # Check CUDA_VISIBLE_DEVICES
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        echo "  ✓ CUDA_VISIBLE_DEVICES is empty (correct)"
    else
        echo "  ✗ CUDA_VISIBLE_DEVICES is not empty: '${CUDA_VISIBLE_DEVICES}'"
        return 1
    fi

    # Check TORCH_DEVICE
    if [ "${TORCH_DEVICE}" = "cpu" ]; then
        echo "  ✓ TORCH_DEVICE is 'cpu' (correct)"
    else
        echo "  ✗ TORCH_DEVICE is not 'cpu': '${TORCH_DEVICE}'"
        return 1
    fi

    # Check cache directories exist
    if [ -d "${HF_HOME}" ]; then
        echo "  ✓ HF_HOME directory exists: ${HF_HOME}"
    else
        echo "  ⚠ HF_HOME directory does not exist: ${HF_HOME}"
        echo "    Creating directory..."
        mkdir -p "${HF_HOME}"
    fi

    if [ -d "${TORCH_HOME}" ]; then
        echo "  ✓ TORCH_HOME directory exists: ${TORCH_HOME}"
    else
        echo "  ⚠ TORCH_HOME directory does not exist: ${TORCH_HOME}"
        echo "    Creating directory..."
        mkdir -p "${TORCH_HOME}"
    fi

    echo "✓ CPU-only configuration verified"
    return 0
}

# Export function for use in shell
export -f verify_cpu_only

# ============================================================================
# Auto-create cache directories
# ============================================================================

# Create cache directories if they don't exist
mkdir -p "${HF_HOME}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${PIP_CACHE_DIR}"

# Set proper permissions (owner read/write/execute)
chmod -R u+rwX /mnt/c/ai_cache 2>/dev/null || true

# ============================================================================
# End of configuration
# ============================================================================
