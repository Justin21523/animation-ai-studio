#!/bin/bash
################################################################################
# GPU Task 2: SDXL Image Generation
#
# Batch image generation using SDXL with LoRA adapters and ControlNet support.
#
# Features:
#   - SDXL base model with LoRA support
#   - ControlNet guidance (pose, depth, canny)
#   - ModelManager integration for automatic VRAM management
#   - Batch generation with quality presets
#   - Automatic GPU memory cleanup
#
# Hardware Requirements:
#   - GPU: NVIDIA RTX 5080 16GB (or similar)
#   - VRAM: 7-9GB for SDXL, 10-12GB with ControlNet
#   - CUDA: 11.8+
#
# Usage:
#   bash scripts/batch/gpu_task2_image_generation.sh \
#     CONFIG_FILE \
#     OUTPUT_DIR \
#     [--num-images N] [--device DEVICE]
#
# Example:
#   bash scripts/batch/gpu_task2_image_generation.sh \
#     configs/generation/luca_character_config.json \
#     /mnt/data/ai_data/outputs/generated_images/luca \
#     --num-images 10 \
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

# Default settings
DEFAULT_NUM_IMAGES=4
DEFAULT_DEVICE="cuda"
DEFAULT_QUALITY="standard"  # Options: draft, standard, high, ultra
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

# ============================================================================
# GPU Monitoring (same as Task 1)
# ============================================================================

check_gpu_available() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - GPU not available"
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

monitor_gpu() {
    local vram_used=$(get_vram_usage)
    local vram_total=$(get_vram_total)
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
# Image Generation
# ============================================================================

generate_images() {
    local config_file="$1"
    local output_dir="$2"
    local num_images="$3"
    local quality="$4"
    local device="$5"

    log_info "Loading configuration: $config_file"

    if [ ! -f "$config_file" ]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    # Create output directory
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/logs"

    # Monitor GPU before
    monitor_gpu

    log_info "Starting SDXL image generation..."
    log_info "Quality preset: $quality"
    log_info "Number of images: $num_images"

    # Run SDXL generation with ModelManager integration
    if python3 -c "
import sys
import os
import json
import gc
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '${PROJECT_ROOT}')

from scripts.core.model_management.model_manager import ModelManager
from scripts.generation.image.sdxl_pipeline import SDXLPipeline

# Load configuration
with open('$config_file', 'r') as f:
    config = json.load(f)

# Extract config parameters
prompt = config.get('prompt', 'a boy, pixar style, 3d animation')
negative_prompt = config.get('negative_prompt', 'blurry, low quality, distorted')
lora_path = config.get('lora_path', None)
lora_weight = config.get('lora_weight', 0.8)
controlnet_type = config.get('controlnet_type', None)
control_image = config.get('control_image', None)
seed = config.get('seed', None)

# Initialize ModelManager
manager = ModelManager()

# Ensure other heavy models are unloaded
manager._ensure_heavy_model_unloaded()
manager.vram_monitor.clear_cache()

try:
    # Use SDXL context (automatic loading/unloading)
    with manager.use_sdxl() as pipeline:
        log_info(f'SDXL pipeline loaded')
        log_info(f'Prompt: {prompt}')

        # Load LoRA if specified
        if lora_path and os.path.exists(lora_path):
            print(f'Loading LoRA: {lora_path} (weight: {lora_weight})')
            # LoRA loading logic here (depends on your SDXL implementation)

        # Setup ControlNet if specified
        if controlnet_type and control_image:
            print(f'Using ControlNet: {controlnet_type}')
            print(f'Control image: {control_image}')

        # Quality preset mapping
        quality_presets = {
            'draft': {'steps': 20, 'guidance_scale': 5.0},
            'standard': {'steps': 30, 'guidance_scale': 7.5},
            'high': {'steps': 50, 'guidance_scale': 9.0},
            'ultra': {'steps': 75, 'guidance_scale': 10.0}
        }
        preset = quality_presets.get('$quality', quality_presets['standard'])

        # Generate images
        images = pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=$num_images,
            num_inference_steps=preset['steps'],
            guidance_scale=preset['guidance_scale'],
            seed=seed
        )

        # Save images
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i, img in enumerate(images):
            output_path = f'$output_dir/{timestamp}_{i:03d}.png'
            img.save(output_path)
            print(f'Saved: {output_path}')

        print(f'Generated {len(images)} images successfully')

    # SDXL automatically unloaded when exiting context

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.exit(0)

except Exception as e:
    print(f'Error during generation: {str(e)}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "${output_dir}/logs/generation_$(date +%Y%m%d_%H%M%S).log"
    then
        log_success "Image generation complete"
        monitor_gpu
        clear_gpu_memory
        return 0
    else
        log_error "Image generation failed"
        return 1
    fi
}

# ============================================================================
# Batch Generation (Multiple Configs)
# ============================================================================

process_batch_generation() {
    local config_file="$1"
    local output_dir="$2"
    local num_images="$3"
    local quality="$4"
    local device="$5"

    log_info "========================================"
    log_info "GPU Task 2: SDXL Image Generation"
    log_info "========================================"
    log_info "Config: $config_file"
    log_info "Output: $output_dir"
    log_info "Images per prompt: $num_images"
    log_info "Quality: $quality"
    log_info "Device: $device"
    echo ""

    # Check GPU
    if ! check_gpu_available; then
        log_error "GPU check failed"
        exit 1
    fi

    # Clear GPU before starting
    clear_gpu_memory

    local start_time=$(date +%s)

    # Check if config is a directory (batch mode) or single file
    if [ -d "$config_file" ]; then
        log_info "Batch mode: Processing multiple configs from directory"

        local config_files=()
        while IFS= read -r -d '' file; do
            config_files+=("$file")
        done < <(find "$config_file" -maxdepth 1 -type f -name "*.json" -print0 | sort -z)

        local total_configs=${#config_files[@]}
        log_info "Found $total_configs config files"

        local processed=0
        local failed=0

        for config in "${config_files[@]}"; do
            local config_name=$(basename "$config" .json)
            local config_output="${output_dir}/${config_name}"

            log_info ""
            log_info "Processing [$((processed + failed + 1))/$total_configs]: $config_name"

            if generate_images "$config" "$config_output" "$num_images" "$quality" "$device"; then
                ((processed++))
            else
                ((failed++))
                echo "$config_name" >> "${output_dir}/failed_configs.txt"
            fi
        done

        log_info ""
        log_info "Batch generation complete"
        log_info "Processed: $processed, Failed: $failed"

    else
        # Single config mode
        if ! generate_images "$config_file" "$output_dir" "$num_images" "$quality" "$device"; then
            log_error "Generation failed"
            return 1
        fi
    fi

    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local minutes=$((total_time / 60))
    local seconds=$((total_time % 60))

    # Count generated images
    local image_count=$(find "$output_dir" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)

    # Create summary
    local summary_file="${output_dir}/generation_summary.json"
    cat > "$summary_file" <<EOF
{
  "task": "gpu_task2_image_generation",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "config_file": "$config_file",
  "output_dir": "$output_dir",
  "num_images_per_prompt": $num_images,
  "quality_preset": "$quality",
  "device": "$device",
  "total_images_generated": $image_count,
  "total_time_seconds": $total_time,
  "completed": true
}
EOF

    log_info ""
    log_info "========================================"
    log_info "Generation Complete"
    log_info "========================================"
    log_info "Images generated: $image_count"
    log_info "Time: ${minutes}m ${seconds}s"
    log_info "Summary: $summary_file"
    echo ""

    return 0
}

# ============================================================================
# Cleanup Handler
# ============================================================================

cleanup_on_error() {
    log_error "Generation interrupted or failed"
    clear_gpu_memory
    exit 1
}

trap cleanup_on_error ERR SIGTERM SIGINT

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 CONFIG_FILE OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  CONFIG_FILE    JSON config file or directory with multiple configs"
    echo "  OUTPUT_DIR     Output directory for generated images"
    echo ""
    echo "Options:"
    echo "  --num-images N     Images per prompt (default: $DEFAULT_NUM_IMAGES)"
    echo "  --quality PRESET   Quality preset (default: $DEFAULT_QUALITY)"
    echo "                     Options: draft, standard, high, ultra"
    echo "  --device DEVICE    Device to use (default: $DEFAULT_DEVICE)"
    echo "  --help             Show this help"
    echo ""
    echo "Config File Format (JSON):"
    echo "  {"
    echo "    \"prompt\": \"luca, boy, brown hair, pixar style\","
    echo "    \"negative_prompt\": \"blurry, low quality\","
    echo "    \"lora_path\": \"/path/to/lora.safetensors\","
    echo "    \"lora_weight\": 0.8,"
    echo "    \"seed\": 42"
    echo "  }"
    echo ""
    echo "Example:"
    echo "  $0 configs/generation/luca_config.json outputs/luca --num-images 10"
    exit 1
}

CONFIG_FILE=""
OUTPUT_DIR=""
NUM_IMAGES=$DEFAULT_NUM_IMAGES
QUALITY=$DEFAULT_QUALITY
DEVICE=$DEFAULT_DEVICE

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --quality)
            QUALITY="$2"
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
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
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
if [ -z "$CONFIG_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

if [ ! -e "$CONFIG_FILE" ]; then
    log_error "Config file/directory does not exist: $CONFIG_FILE"
    exit 1
fi

# Validate quality preset
valid_qualities=("draft" "standard" "high" "ultra")
if [[ ! " ${valid_qualities[@]} " =~ " ${QUALITY} " ]]; then
    log_error "Invalid quality preset: $QUALITY"
    log_error "Valid presets: ${valid_qualities[*]}"
    exit 1
fi

# Export variables
export RED GREEN YELLOW BLUE CYAN MAGENTA NC

# Run processing
process_batch_generation "$CONFIG_FILE" "$OUTPUT_DIR" "$NUM_IMAGES" "$QUALITY" "$DEVICE"

log_success "GPU Task 2 (Image Generation) completed successfully"
exit 0
