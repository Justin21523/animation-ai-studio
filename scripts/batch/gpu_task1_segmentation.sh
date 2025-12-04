#!/bin/bash
################################################################################
# GPU Task 1: SAM2 Character Segmentation
#
# Uses SAM2 (Segment Anything Model 2) for automatic character instance
# segmentation with tracking across video frames.
#
# Features:
#   - Automatic character instance detection
#   - Multi-frame tracking for temporal consistency
#   - ModelManager integration for VRAM management
#   - Automatic GPU memory cleanup
#   - Checkpoint support for resume
#
# Hardware Requirements:
#   - GPU: NVIDIA RTX 5080 16GB (or similar)
#   - VRAM: 6-7GB for SAM2-base, 8-10GB for SAM2-large
#   - CUDA: 11.8+
#
# Usage:
#   bash scripts/batch/gpu_task1_segmentation.sh \
#     INPUT_FRAMES_DIR \
#     OUTPUT_DIR \
#     [--model MODEL] [--device DEVICE] [--resume]
#
# Example:
#   bash scripts/batch/gpu_task1_segmentation.sh \
#     /mnt/data/ai_data/datasets/3d-anime/luca/frames \
#     /mnt/data/ai_data/datasets/3d-anime/luca/segmented \
#     --model sam2_hiera_base \
#     --device cuda \
#     --resume
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

# SAM2 script
SEGMENTATION_SCRIPT="${PROJECT_ROOT}/scripts/processing/segmentation/instance_segmentation.py"

# Default settings
DEFAULT_MODEL="sam2_hiera_base"
DEFAULT_DEVICE="cuda"
VRAM_THRESHOLD_GB=14  # Leave ~2GB buffer on 16GB GPU

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

# ============================================================================
# GPU Monitoring
# ============================================================================

check_gpu_available() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - GPU not available"
        return 1
    fi

    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    if [ "$gpu_count" -lt 1 ]; then
        log_error "No GPU detected"
        return 1
    fi

    log_success "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    return 0
}

get_vram_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1
}

get_vram_total() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1
}

get_gpu_utilization() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1
}

get_gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1
}

monitor_gpu() {
    local vram_used=$(get_vram_usage)
    local vram_total=$(get_vram_total)
    local vram_pct=$(awk "BEGIN {printf \"%.1f\", ($vram_used/$vram_total)*100}")
    local gpu_util=$(get_gpu_utilization)
    local gpu_temp=$(get_gpu_temp)

    log_gpu "VRAM: ${vram_used}MB / ${vram_total}MB (${vram_pct}%) | Util: ${gpu_util}% | Temp: ${gpu_temp}°C"

    # Check VRAM threshold
    local vram_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
    if (( $(echo "$vram_gb > $VRAM_THRESHOLD_GB" | bc -l) )); then
        log_warning "VRAM usage high: ${vram_gb}GB > ${VRAM_THRESHOLD_GB}GB"
    fi
}

clear_gpu_memory() {
    log_info "Clearing GPU memory..."

    # Use Python to clear CUDA cache
    python3 -c "
import torch
import gc

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('GPU cache cleared')
else:
    print('CUDA not available')
" 2>/dev/null || true

    sleep 2
    monitor_gpu
}

# ============================================================================
# Checkpoint Management
# ============================================================================

is_video_processed() {
    local checkpoint_file="$1"
    local video_id="$2"

    if [ ! -f "$checkpoint_file" ]; then
        return 1
    fi

    if grep -Fxq "$video_id" "$checkpoint_file"; then
        return 0
    else
        return 1
    fi
}

save_checkpoint() {
    local checkpoint_file="$1"
    local video_id="$2"
    echo "$video_id" >> "$checkpoint_file"
}

# ============================================================================
# SAM2 Segmentation Worker
# ============================================================================

segment_video_frames() {
    local frames_dir="$1"
    local output_dir="$2"
    local model="$3"
    local device="$4"
    local checkpoint_file="$5"

    local video_id=$(basename "$frames_dir")

    # Check if already processed
    if is_video_processed "$checkpoint_file" "$video_id"; then
        log_info "Skipping (already processed): $video_id"
        return 0
    fi

    log_info "Processing: $video_id"
    log_info "Model: $model, Device: $device"

    # Create output directory
    local video_output="${output_dir}/${video_id}"
    mkdir -p "$video_output"

    # Monitor GPU before
    monitor_gpu

    # Run SAM2 segmentation with ModelManager integration
    log_info "Starting SAM2 segmentation..."

    if python3 -c "
import sys
import os
import gc
import torch

# Add project root to path
sys.path.insert(0, '${PROJECT_ROOT}')

from scripts.core.model_management.model_manager import ModelManager
from scripts.processing.segmentation.instance_segmentation import InstanceSegmenter

# Initialize ModelManager
manager = ModelManager()

# Ensure VRAM is clear before loading SAM2
manager._ensure_heavy_model_unloaded()
manager.vram_monitor.clear_cache()

try:
    # Initialize SAM2
    segmenter = InstanceSegmenter(
        model_size='${model##sam2_hiera_}',  # Extract size from model name
        device='$device'
    )

    # Run segmentation
    results = segmenter.segment_directory(
        input_dir='$frames_dir',
        output_dir='$video_output',
        track_instances=True,
        min_mask_area=100
    )

    print(f'Segmented {len(results)} frames')
    print(f'Found {segmenter.num_instances} character instances')

    # Cleanup
    del segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.exit(0)

except Exception as e:
    print(f'Error during segmentation: {str(e)}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | tee "${output_dir}/logs/${video_id}_segmentation.log"
    then
        log_success "Segmentation complete: $video_id"

        # Monitor GPU after
        monitor_gpu

        # Save checkpoint
        save_checkpoint "$checkpoint_file" "$video_id"

        # Clear GPU memory
        clear_gpu_memory

        return 0
    else
        log_error "Segmentation failed: $video_id"
        return 1
    fi
}

# ============================================================================
# Main Processing Function
# ============================================================================

process_segmentation() {
    local input_dir="$1"
    local output_dir="$2"
    local model="$3"
    local device="$4"
    local resume_mode="$5"

    log_info "========================================"
    log_info "GPU Task 1: SAM2 Character Segmentation"
    log_info "========================================"
    log_info "Input: $input_dir"
    log_info "Output: $output_dir"
    log_info "Model: $model"
    log_info "Device: $device"
    log_info "Resume: $resume_mode"
    echo ""

    # Check GPU availability
    if ! check_gpu_available; then
        log_error "GPU check failed"
        exit 1
    fi

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/logs"
    mkdir -p "$output_dir/checkpoints"

    # Checkpoint file
    local checkpoint_file="${output_dir}/checkpoints/segmentation_processed.txt"
    if [ "$resume_mode" = "false" ]; then
        rm -f "$checkpoint_file"
    fi

    # Find all video frame directories
    local frame_dirs=()
    while IFS= read -r -d '' dir; do
        frame_dirs+=("$dir")
    done < <(find "$input_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    local total_videos=${#frame_dirs[@]}
    log_info "Found $total_videos video frame directories"

    if [ $total_videos -eq 0 ]; then
        log_error "No frame directories found in: $input_dir"
        exit 1
    fi

    # Clear GPU memory before starting
    clear_gpu_memory

    # Process each video
    local processed=0
    local failed=0
    local start_time=$(date +%s)

    for frames_dir in "${frame_dirs[@]}"; do
        local video_id=$(basename "$frames_dir")

        log_info ""
        log_info "Processing [$((processed + failed + 1))/$total_videos]: $video_id"

        if segment_video_frames "$frames_dir" "$output_dir" "$model" "$device" "$checkpoint_file"; then
            ((processed++))
        else
            ((failed++))
            echo "$video_id" >> "${output_dir}/failed_videos.txt"
        fi
    done

    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))

    # Generate summary
    log_info ""
    log_info "========================================"
    log_info "Segmentation Complete"
    log_info "========================================"
    log_info "Total videos: $total_videos"
    log_info "Processed: $processed"
    log_info "Failed: $failed"
    log_info "Time: ${hours}h ${minutes}m ${seconds}s"
    echo ""

    # Count instances
    local instance_count=0
    if [ -d "$output_dir" ]; then
        instance_count=$(find "$output_dir" -type f -name "*.json" 2>/dev/null | wc -l)
    fi

    # Create summary JSON
    local summary_file="${output_dir}/segmentation_summary.json"
    cat > "$summary_file" <<EOF
{
  "task": "gpu_task1_segmentation",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "model": "$model",
  "device": "$device",
  "total_videos": $total_videos,
  "processed": $processed,
  "failed": $failed,
  "total_time_seconds": $total_time,
  "instance_files": $instance_count,
  "completed": true
}
EOF

    log_info "Summary: $summary_file"

    if [ $failed -gt 0 ]; then
        log_warning "Some videos failed - see: ${output_dir}/failed_videos.txt"
        return 1
    fi

    return 0
}

# ============================================================================
# Cleanup Handler
# ============================================================================

cleanup_on_error() {
    log_error "Segmentation interrupted or failed"
    clear_gpu_memory
    exit 1
}

trap cleanup_on_error ERR SIGTERM SIGINT

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 INPUT_DIR OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  INPUT_DIR    Directory containing video frame subdirectories"
    echo "  OUTPUT_DIR   Output directory for segmentation results"
    echo ""
    echo "Options:"
    echo "  --model MODEL      SAM2 model variant (default: $DEFAULT_MODEL)"
    echo "                     Options: sam2_hiera_tiny, sam2_hiera_small,"
    echo "                              sam2_hiera_base, sam2_hiera_large"
    echo "  --device DEVICE    Device to use (default: $DEFAULT_DEVICE)"
    echo "                     Options: cuda, cpu (cpu not recommended)"
    echo "  --resume           Resume from checkpoint (skip processed)"
    echo "  --help             Show this help"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/frames /path/to/output --model sam2_hiera_base --resume"
    exit 1
}

INPUT_DIR=""
OUTPUT_DIR=""
MODEL=$DEFAULT_MODEL
DEVICE=$DEFAULT_DEVICE
RESUME_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME_MODE="true"
            shift
            ;;
        --help)
            usage
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
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
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

if [ ! -d "$INPUT_DIR" ]; then
    log_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Validate model
valid_models=("sam2_hiera_tiny" "sam2_hiera_small" "sam2_hiera_base" "sam2_hiera_large")
if [[ ! " ${valid_models[@]} " =~ " ${MODEL} " ]]; then
    log_error "Invalid model: $MODEL"
    log_error "Valid models: ${valid_models[*]}"
    exit 1
fi

# Export variables
export SEGMENTATION_SCRIPT
export RED GREEN YELLOW BLUE CYAN MAGENTA NC

# Run processing
process_segmentation "$INPUT_DIR" "$OUTPUT_DIR" "$MODEL" "$DEVICE" "$RESUME_MODE"

log_success "GPU Task 1 (Segmentation) completed successfully"
exit 0
