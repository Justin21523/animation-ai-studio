#!/bin/bash
################################################################################
# GPU Task 4: GPT-SoVITS Voice Training
#
# Train character voice models using GPT-SoVITS for high-quality voice cloning.
#
# Features:
#   - GPT-SoVITS training pipeline
#   - Automatic voice sample processing
#   - ModelManager integration (ensures other models unloaded)
#   - Progress tracking and checkpointing
#   - OPTIONAL task (can run overnight)
#
# Hardware Requirements:
#   - GPU: NVIDIA RTX 5080 16GB
#   - VRAM: 8-10GB for training
#   - Training time: 2-4 hours (depending on sample size)
#   - CUDA: 11.8+
#
# Usage:
#   bash scripts/batch/gpu_task4_voice_training.sh \
#     CHARACTER_NAME \
#     VOICE_SAMPLES_DIR \
#     OUTPUT_DIR \
#     [--epochs N] [--device DEVICE]
#
# Example:
#   bash scripts/batch/gpu_task4_voice_training.sh \
#     luca \
#     /mnt/data/ai_data/datasets/3d-anime/luca/voice_samples \
#     /mnt/data/ai_data/models/voices/luca \
#     --epochs 100 \
#     --device cuda
#
# Author: Animation AI Studio
# Date: 2025-12-04
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# GPT-SoVITS trainer script
TRAINER_SCRIPT="${PROJECT_ROOT}/scripts/synthesis/tts/gpt_sovits_trainer.py"

# Default settings
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=4
DEFAULT_DEVICE="cuda"
VRAM_THRESHOLD_GB=14

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_gpu() { echo -e "${MAGENTA}[GPU]${NC} $1"; }
log_train() { echo -e "${CYAN}[TRAIN]${NC} $1"; }

# ============================================================================
# GPU Monitoring
# ============================================================================

check_gpu_available() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - GPU not available"
        return 1
    fi
    log_success "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    return 0
}

monitor_gpu() {
    local vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    local vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    local vram_pct=$(awk "BEGIN {printf \"%.1f\", ($vram_used/$vram_total)*100}")
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1)
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1)

    log_gpu "VRAM: ${vram_used}MB / ${vram_total}MB (${vram_pct}%) | Util: ${gpu_util}% | Temp: ${gpu_temp}°C"
}

clear_gpu_memory() {
    log_info "Clearing GPU memory..."
    python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
" 2>/dev/null || true
    sleep 2
    monitor_gpu
}

# ============================================================================
# Voice Sample Validation
# ============================================================================

validate_voice_samples() {
    local samples_dir="$1"

    log_info "Validating voice samples in: $samples_dir"

    # Count audio files
    local wav_count=$(find "$samples_dir" -type f -name "*.wav" 2>/dev/null | wc -l)
    local mp3_count=$(find "$samples_dir" -type f -name "*.mp3" 2>/dev/null | wc -l)
    local total_count=$((wav_count + mp3_count))

    log_info "Found $total_count audio files ($wav_count WAV, $mp3_count MP3)"

    if [ $total_count -lt 5 ]; then
        log_error "Insufficient voice samples (need at least 5, found $total_count)"
        return 1
    fi

    if [ $total_count -lt 20 ]; then
        log_warning "Low sample count ($total_count). Recommended: 20-50 samples for good quality"
    fi

    # Check total duration (approximate)
    log_info "Checking sample durations..."
    local total_duration=0

    # Process WAV files
    for audio_file in "$samples_dir"/*.wav; do
        if [ -f "$audio_file" ]; then
            local duration=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$audio_file" 2>/dev/null || echo "0")
            total_duration=$(awk "BEGIN {printf \"%.0f\", $total_duration + $duration}")
        fi
    done

    # Process MP3 files
    for audio_file in "$samples_dir"/*.mp3; do
        if [ -f "$audio_file" ]; then
            local duration=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$audio_file" 2>/dev/null || echo "0")
            total_duration=$(awk "BEGIN {printf \"%.0f\", $total_duration + $duration}")
        fi
    done

    local total_minutes=$((total_duration / 60))
    log_info "Total audio duration: ${total_minutes} minutes"

    if [ $total_duration -lt 60 ]; then
        log_warning "Short total duration (${total_duration}s). Recommended: 5-15 minutes"
    fi

    log_success "Voice sample validation passed"
    return 0
}

# ============================================================================
# Voice Training
# ============================================================================

train_voice_model() {
    local character="$1"
    local samples_dir="$2"
    local output_dir="$3"
    local epochs="$4"
    local batch_size="$5"
    local device="$6"

    log_train "Starting GPT-SoVITS training..."
    log_train "Character: $character"
    log_train "Epochs: $epochs, Batch size: $batch_size"

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/checkpoints"
    mkdir -p "$output_dir/logs"

    # Monitor GPU before
    monitor_gpu

    # Check if trainer script exists
    if [ ! -f "$TRAINER_SCRIPT" ]; then
        log_error "Trainer script not found: $TRAINER_SCRIPT"
        log_warning "Creating placeholder (training not available)"

        cat > "$output_dir/${character}_voice.pth.placeholder" <<EOF
# GPT-SoVITS Voice Model Placeholder
# Character: $character
# Note: Actual training script not available
EOF
        return 1
    fi

    # Run GPT-SoVITS training with ModelManager integration
    if python3 -c "
import sys
import os
import gc
import torch

sys.path.insert(0, '${PROJECT_ROOT}')

from scripts.core.model_management.model_manager import ModelManager

# Initialize ModelManager
manager = ModelManager()

# CRITICAL: Ensure all other heavy models are unloaded before voice training
print('Ensuring GPU is clear for voice training...')
manager._ensure_heavy_model_unloaded()
manager.vram_monitor.clear_cache()

try:
    # Import trainer after clearing GPU
    from scripts.synthesis.tts.gpt_sovits_trainer import GPTSoVITSTrainer

    # Initialize trainer
    trainer = GPTSoVITSTrainer(
        character_name='$character',
        voice_samples_dir='$samples_dir',
        output_dir='$output_dir',
        device='$device'
    )

    # Training configuration
    config = {
        'epochs': $epochs,
        'batch_size': $batch_size,
        'learning_rate': 1e-4,
        'save_frequency': 10,  # Save checkpoint every 10 epochs
        'validation_split': 0.1
    }

    print(f'Training configuration: {config}')

    # Start training (this will take 2-4 hours)
    print('Starting training (this may take 2-4 hours)...')
    trainer.train(config)

    print('Training completed successfully')

    # Cleanup
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.exit(0)

except Exception as e:
    print(f'Error during training: {str(e)}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "$output_dir/logs/training.log"
    then
        log_success "Voice training complete"
        monitor_gpu
        clear_gpu_memory
        return 0
    else
        log_error "Voice training failed"
        return 1
    fi
}

# ============================================================================
# Main Processing Function
# ============================================================================

process_voice_training() {
    local character="$1"
    local samples_dir="$2"
    local output_dir="$3"
    local epochs="$4"
    local batch_size="$5"
    local device="$6"

    log_info "========================================"
    log_info "GPU Task 4: GPT-SoVITS Voice Training"
    log_info "========================================"
    log_info "Character: $character"
    log_info "Samples: $samples_dir"
    log_info "Output: $output_dir"
    log_info "Epochs: $epochs"
    log_info "Batch size: $batch_size"
    log_info "Device: $device"
    echo ""
    log_warning "This task is OPTIONAL and may take 2-4 hours"
    log_warning "Consider running overnight or in tmux/screen"
    echo ""

    # Check GPU
    if ! check_gpu_available; then
        log_error "GPU check failed"
        exit 1
    fi

    # Validate voice samples
    if ! validate_voice_samples "$samples_dir"; then
        log_error "Voice sample validation failed"
        exit 1
    fi

    # Clear GPU before starting
    clear_gpu_memory

    local start_time=$(date +%s)

    # Run training
    if ! train_voice_model "$character" "$samples_dir" "$output_dir" "$epochs" "$batch_size" "$device"; then
        log_error "Training failed"
        return 1
    fi

    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))

    # Check if model file exists
    local model_file="${output_dir}/${character}_voice.pth"
    local model_exists="false"
    if [ -f "$model_file" ]; then
        model_exists="true"
        local model_size=$(du -h "$model_file" | cut -f1)
        log_success "Model file: $model_file ($model_size)"
    fi

    # Create summary
    local summary_file="${output_dir}/training_summary.json"
    cat > "$summary_file" <<EOF
{
  "task": "gpu_task4_voice_training",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "character": "$character",
  "samples_dir": "$samples_dir",
  "output_dir": "$output_dir",
  "epochs": $epochs,
  "batch_size": $batch_size,
  "device": "$device",
  "total_time_seconds": $total_time,
  "model_exists": $model_exists,
  "model_file": "$model_file",
  "completed": true
}
EOF

    log_info ""
    log_info "========================================"
    log_info "Voice Training Complete"
    log_info "========================================"
    log_info "Character: $character"
    log_info "Time: ${hours}h ${minutes}m ${seconds}s"
    log_info "Model: $model_file"
    log_info "Summary: $summary_file"
    echo ""

    return 0
}

# ============================================================================
# Cleanup Handler
# ============================================================================

cleanup_on_error() {
    log_error "Voice training interrupted or failed"
    clear_gpu_memory
    exit 1
}

trap cleanup_on_error ERR SIGTERM SIGINT

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 CHARACTER SAMPLES_DIR OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  CHARACTER     Character name (e.g., luca, alberto)"
    echo "  SAMPLES_DIR   Directory containing voice sample audio files"
    echo "  OUTPUT_DIR    Output directory for trained voice model"
    echo ""
    echo "Options:"
    echo "  --epochs N          Training epochs (default: $DEFAULT_EPOCHS)"
    echo "  --batch-size N      Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --device DEVICE     Device to use (default: $DEFAULT_DEVICE)"
    echo "  --help              Show this help"
    echo ""
    echo "Voice Sample Requirements:"
    echo "  - Minimum: 5 audio files"
    echo "  - Recommended: 20-50 audio files"
    echo "  - Total duration: 5-15 minutes"
    echo "  - Format: WAV or MP3"
    echo "  - Quality: Clean audio, minimal background noise"
    echo ""
    echo "Example:"
    echo "  $0 luca /data/luca/voice_samples /models/voices/luca --epochs 100"
    echo ""
    echo "Note: This task may take 2-4 hours. Consider using tmux/screen."
    exit 1
}

CHARACTER=""
SAMPLES_DIR=""
OUTPUT_DIR=""
EPOCHS=$DEFAULT_EPOCHS
BATCH_SIZE=$DEFAULT_BATCH_SIZE
DEVICE=$DEFAULT_DEVICE

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            if [ -z "$CHARACTER" ]; then
                CHARACTER="$1"
            elif [ -z "$SAMPLES_DIR" ]; then
                SAMPLES_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                log_error "Unknown argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$CHARACTER" ] || [ -z "$SAMPLES_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

if [ ! -d "$SAMPLES_DIR" ]; then
    log_error "Samples directory does not exist: $SAMPLES_DIR"
    exit 1
fi

# Export variables
export RED GREEN YELLOW BLUE CYAN MAGENTA NC

# Run processing
process_voice_training "$CHARACTER" "$SAMPLES_DIR" "$OUTPUT_DIR" "$EPOCHS" "$BATCH_SIZE" "$DEVICE"

log_success "GPU Task 4 (Voice Training) completed successfully"
exit 0
