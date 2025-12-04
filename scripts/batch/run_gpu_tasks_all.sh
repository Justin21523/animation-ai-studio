#!/bin/bash
################################################################################
# Master GPU Tasks Orchestration Script
#
# Executes all GPU tasks in optimal sequence with automatic model switching:
#   Task 1: SAM2 Character Segmentation (6-7GB VRAM)
#   Task 2: SDXL Image Generation (7-9GB VRAM)
#   Task 3: LLM Video Analysis (6-14GB VRAM)
#   Task 4: Voice Training (OPTIONAL, 8-10GB VRAM)
#
# Features:
#   - Sequential execution with automatic VRAM management
#   - ModelManager integration for model switching
#   - GPU memory cleanup between tasks
#   - Comprehensive error handling
#   - Progress tracking and reporting
#   - Skip completed tasks with --resume
#
# Hardware Requirements:
#   - GPU: NVIDIA RTX 5080 16GB (or similar)
#   - Single GPU - tasks execute sequentially (NOT parallel)
#   - CUDA: 11.8+
#
# Usage:
#   bash scripts/batch/run_gpu_tasks_all.sh \
#     INPUT_DATA_DIR \
#     OUTPUT_BASE_DIR \
#     [OPTIONS]
#
# Example:
#   bash scripts/batch/run_gpu_tasks_all.sh \
#     /mnt/data/ai_data/datasets/3d-anime/luca \
#     /mnt/data/ai_data/outputs/luca \
#     --enable-voice-training \
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

# GPU task scripts
TASK1_SCRIPT="${SCRIPT_DIR}/gpu_task1_segmentation.sh"
TASK2_SCRIPT="${SCRIPT_DIR}/gpu_task2_image_generation.sh"
TASK3_SCRIPT="${SCRIPT_DIR}/gpu_task3_llm_analysis.sh"
TASK4_SCRIPT="${SCRIPT_DIR}/gpu_task4_voice_training.sh"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor_resources.sh"

# Default settings
DEFAULT_SAM2_MODEL="sam2_hiera_base"
DEFAULT_SDXL_QUALITY="standard"
DEFAULT_LLM_MODEL="qwen-vl-7b"
DEFAULT_VOICE_EPOCHS=100
MONITOR_INTERVAL=30

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

log_header() { echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
log_section() { echo -e "${BOLD}${MAGENTA}▶ $1${NC}"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_task() { echo -e "${BOLD}${BLUE}[GPU TASK $1]${NC} $2"; }

# ============================================================================
# Time Tracking
# ============================================================================

TASK_START_TIME=0
TOTAL_START_TIME=0

start_timer() {
    local timer_name="$1"
    if [ "$timer_name" = "total" ]; then
        TOTAL_START_TIME=$(date +%s)
    else
        TASK_START_TIME=$(date +%s)
    fi
}

get_elapsed_time() {
    local start_time="$1"
    local current_time=$(date +%s)
    local elapsed=$((current_time - start_time))

    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))

    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# ============================================================================
# GPU Monitoring
# ============================================================================

MONITOR_PID=""

start_monitoring() {
    local log_dir="$1"

    if [ ! -f "$MONITOR_SCRIPT" ]; then
        log_warning "Monitor script not found, skipping GPU monitoring"
        return
    fi

    log_info "Starting GPU resource monitor (daemon mode)"

    bash "$MONITOR_SCRIPT" \
        --daemon \
        --interval "$MONITOR_INTERVAL" \
        --log-dir "$log_dir" &

    MONITOR_PID=$!
    log_info "Monitor started (PID: $MONITOR_PID)"
    sleep 2
}

stop_monitoring() {
    if [ -n "$MONITOR_PID" ] && ps -p "$MONITOR_PID" > /dev/null 2>&1; then
        log_info "Stopping GPU resource monitor (PID: $MONITOR_PID)"
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

# ============================================================================
# Task Execution
# ============================================================================

execute_task() {
    local task_num="$1"
    local task_name="$2"
    local task_script="$3"
    shift 3
    local task_args=("$@")

    log_header
    log_task "$task_num" "$task_name"
    log_header
    echo ""

    # Check if script exists
    if [ ! -f "$task_script" ]; then
        log_error "Task script not found: $task_script"
        return 1
    fi

    # Start timer
    start_timer "task"

    # Execute task
    log_info "Executing: bash $task_script ${task_args[*]}"
    echo ""

    if bash "$task_script" "${task_args[@]}"; then
        local elapsed=$(get_elapsed_time "$TASK_START_TIME")
        echo ""
        log_success "GPU Task $task_num completed in $elapsed"
        return 0
    else
        local elapsed=$(get_elapsed_time "$TASK_START_TIME")
        echo ""
        log_error "GPU Task $task_num failed after $elapsed"
        return 1
    fi
}

# ============================================================================
# Main GPU Pipeline
# ============================================================================

run_gpu_pipeline() {
    local input_dir="$1"
    local output_dir="$2"
    local enable_voice="$3"
    local resume_mode="$4"
    local sam2_model="$5"
    local sdxl_quality="$6"
    local llm_model="$7"
    local voice_epochs="$8"

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/logs"
    mkdir -p "$output_dir/monitoring"
    mkdir -p "$output_dir/checkpoints"

    # Save execution metadata
    local metadata_file="${output_dir}/gpu_execution_metadata.json"
    cat > "$metadata_file" <<EOF
{
  "pipeline": "gpu_tasks_all",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "enable_voice_training": $enable_voice,
  "resume_mode": $resume_mode,
  "models": {
    "sam2": "$sam2_model",
    "sdxl_quality": "$sdxl_quality",
    "llm": "$llm_model"
  },
  "voice_training_epochs": $voice_epochs,
  "start_time": "$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")",
  "hostname": "$(hostname)",
  "user": "$(whoami)"
}
EOF

    # Display configuration
    log_header
    log_section "GPU PIPELINE CONFIGURATION"
    log_header
    echo ""
    log_info "Input Directory:     $input_dir"
    log_info "Output Directory:    $output_dir"
    log_info "SAM2 Model:          $sam2_model"
    log_info "SDXL Quality:        $sdxl_quality"
    log_info "LLM Model:           $llm_model"
    log_info "Voice Training:      $enable_voice"
    log_info "Resume Mode:         $resume_mode"
    log_info "Start Time:          $(date)"
    echo ""
    log_warning "IMPORTANT: GPU tasks run SEQUENTIALLY (one at a time)"
    log_warning "Single RTX 5080 16GB can only load one heavy model at a time"
    echo ""

    # Start total timer
    start_timer "total"

    # Start resource monitoring
    start_monitoring "${output_dir}/monitoring"

    # ========================================================================
    # Task 1: SAM2 Character Segmentation
    # ========================================================================

    local task1_args=(
        "${input_dir}/frames"
        "${output_dir}/segmented"
        "--model" "$sam2_model"
        "--device" "cuda"
    )

    if [ "$resume_mode" = "true" ]; then
        task1_args+=("--resume")
    fi

    if ! execute_task "1" "SAM2 Character Segmentation (VRAM: 6-7GB)" "$TASK1_SCRIPT" "${task1_args[@]}"; then
        log_error "Pipeline failed at Task 1"
        stop_monitoring
        return 1
    fi

    # ========================================================================
    # Task 2: SDXL Image Generation
    # ========================================================================

    # Check if generation config exists
    local gen_config="${input_dir}/generation_config.json"
    if [ ! -f "$gen_config" ]; then
        log_warning "No generation config found, skipping SDXL task"
        log_info "To enable: create $gen_config"
    else
        local task2_args=(
            "$gen_config"
            "${output_dir}/generated_images"
            "--num-images" "4"
            "--quality" "$sdxl_quality"
            "--device" "cuda"
        )

        if ! execute_task "2" "SDXL Image Generation (VRAM: 7-9GB)" "$TASK2_SCRIPT" "${task2_args[@]}"; then
            log_warning "Task 2 failed, continuing pipeline..."
        fi
    fi

    # ========================================================================
    # Task 3: LLM Video Analysis
    # ========================================================================

    local task3_args=(
        "$input_dir"
        "${output_dir}/llm_analysis"
        "--model" "$llm_model"
        "--task" "scene_analysis"
    )

    if ! execute_task "3" "LLM Video Analysis (VRAM: 6-14GB)" "$TASK3_SCRIPT" "${task3_args[@]}"; then
        log_warning "Task 3 failed, continuing pipeline..."
    fi

    # ========================================================================
    # Task 4: Voice Training (OPTIONAL)
    # ========================================================================

    if [ "$enable_voice" = "true" ]; then
        # Check if voice samples exist
        local voice_samples="${input_dir}/voice_samples"
        if [ ! -d "$voice_samples" ]; then
            log_warning "No voice samples found, skipping voice training"
            log_info "To enable: create directory $voice_samples with audio files"
        else
            local character_name=$(basename "$input_dir")

            local task4_args=(
                "$character_name"
                "$voice_samples"
                "${output_dir}/voice_model"
                "--epochs" "$voice_epochs"
                "--device" "cuda"
            )

            log_warning "Voice training may take 2-4 hours"
            read -p "Continue with voice training? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if ! execute_task "4" "Voice Training (VRAM: 8-10GB)" "$TASK4_SCRIPT" "${task4_args[@]}"; then
                    log_warning "Task 4 failed"
                fi
            else
                log_info "Skipping voice training (user choice)"
            fi
        fi
    else
        log_info "Voice training disabled (use --enable-voice-training to enable)"
    fi

    # ========================================================================
    # Pipeline Complete
    # ========================================================================

    # Stop monitoring
    stop_monitoring

    # Calculate total time
    local total_elapsed=$(get_elapsed_time "$TOTAL_START_TIME")

    # Update metadata
    local start_time_iso
    if command -v date &> /dev/null; then
        start_time_iso=$(date -u -d "@${TOTAL_START_TIME}" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")
    else
        start_time_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    fi

    cat > "$metadata_file" <<EOF
{
  "pipeline": "gpu_tasks_all",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "enable_voice_training": $enable_voice,
  "resume_mode": $resume_mode,
  "models": {
    "sam2": "$sam2_model",
    "sdxl_quality": "$sdxl_quality",
    "llm": "$llm_model"
  },
  "voice_training_epochs": $voice_epochs,
  "start_time": "$start_time_iso",
  "end_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "total_duration": "$total_elapsed",
  "status": "completed",
  "hostname": "$(hostname)",
  "user": "$(whoami)"
}
EOF

    # Final summary
    echo ""
    log_header
    log_section "GPU PIPELINE COMPLETED SUCCESSFULLY"
    log_header
    echo ""
    log_success "All GPU tasks completed"
    log_info "Total execution time: $total_elapsed"
    log_info "Output directory: $output_dir"
    echo ""

    # Show output summary
    log_section "OUTPUT SUMMARY"
    echo ""

    local seg_count=$(find "${output_dir}/segmented" -type f -name "*.json" 2>/dev/null | wc -l)
    local img_count=$(find "${output_dir}/generated_images" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    local llm_count=$(find "${output_dir}/llm_analysis" -type f -name "*.json" 2>/dev/null | wc -l)

    log_info "Segmentation results:  $seg_count files"
    log_info "Generated images:      $img_count files"
    log_info "LLM analysis results:  $llm_count files"

    if [ "$enable_voice" = "true" ] && [ -f "${output_dir}/voice_model/training_summary.json" ]; then
        log_info "Voice model trained:   ${output_dir}/voice_model/"
    fi

    echo ""
    log_info "Monitoring logs:       ${output_dir}/monitoring/"
    log_info "Metadata:              $metadata_file"
    echo ""
    log_header

    return 0
}

# ============================================================================
# Error Handler
# ============================================================================

cleanup_on_error() {
    log_error "GPU pipeline interrupted or failed"
    stop_monitoring
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
    echo "  INPUT_DIR     Input data directory (must contain frames/)"
    echo "  OUTPUT_DIR    Base output directory for all GPU results"
    echo ""
    echo "Options:"
    echo "  --sam2-model MODEL          SAM2 model variant (default: $DEFAULT_SAM2_MODEL)"
    echo "  --sdxl-quality PRESET       SDXL quality preset (default: $DEFAULT_SDXL_QUALITY)"
    echo "  --llm-model MODEL           LLM model (default: $DEFAULT_LLM_MODEL)"
    echo "  --enable-voice-training     Enable voice training (Task 4)"
    echo "  --voice-epochs N            Voice training epochs (default: $DEFAULT_VOICE_EPOCHS)"
    echo "  --resume                    Resume from checkpoints"
    echo "  --help                      Show this help"
    echo ""
    echo "Example:"
    echo "  $0 /data/luca /outputs/luca --enable-voice-training --resume"
    echo ""
    echo "Note: GPU tasks run SEQUENTIALLY (not parallel) on single RTX 5080"
    exit 1
}

INPUT_DIR=""
OUTPUT_DIR=""
SAM2_MODEL=$DEFAULT_SAM2_MODEL
SDXL_QUALITY=$DEFAULT_SDXL_QUALITY
LLM_MODEL=$DEFAULT_LLM_MODEL
VOICE_EPOCHS=$DEFAULT_VOICE_EPOCHS
ENABLE_VOICE="false"
RESUME_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sam2-model)
            SAM2_MODEL="$2"
            shift 2
            ;;
        --sdxl-quality)
            SDXL_QUALITY="$2"
            shift 2
            ;;
        --llm-model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --voice-epochs)
            VOICE_EPOCHS="$2"
            shift 2
            ;;
        --enable-voice-training)
            ENABLE_VOICE="true"
            shift
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

# Validate required arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

if [ ! -d "$INPUT_DIR" ]; then
    log_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check for frames directory
if [ ! -d "${INPUT_DIR}/frames" ]; then
    log_error "Frames directory not found: ${INPUT_DIR}/frames"
    log_error "Run CPU Stage 1 first to extract frames"
    exit 1
fi

# ============================================================================
# Main Execution
# ============================================================================

run_gpu_pipeline "$INPUT_DIR" "$OUTPUT_DIR" "$ENABLE_VOICE" "$RESUME_MODE" \
    "$SAM2_MODEL" "$SDXL_QUALITY" "$LLM_MODEL" "$VOICE_EPOCHS"

log_success "GPU pipeline execution completed successfully"
exit 0
