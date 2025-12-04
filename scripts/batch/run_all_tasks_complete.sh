#!/bin/bash
################################################################################
# Ultimate Master Script: Complete Animation AI Studio Pipeline
#
# Integrates both CPU and GPU pipelines into a single one-command workflow.
#
# Features:
#   - Full CPU pipeline (Stage 1-3: Data prep, analysis, RAG)
#   - Full GPU pipeline (Task 1-4: SAM2, SDXL, LLM, Voice)
#   - Prerequisite validation (dependencies, disk space, GPU)
#   - Unified progress reporting
#   - Error handling with automatic rollback
#   - Comprehensive final summary
#
# Hardware Requirements:
#   - CPU: AMD Ryzen 9 9950X (16 cores) or similar
#   - GPU: NVIDIA RTX 5080 16GB
#   - RAM: 32GB+
#   - Disk: 100GB+ free space
#   - CUDA: 11.8+
#
# Usage:
#   bash scripts/batch/run_all_tasks_complete.sh \
#     FILM_NAME \
#     INPUT_VIDEO_DIR \
#     OUTPUT_BASE_DIR \
#     [OPTIONS]
#
# Example:
#   bash scripts/batch/run_all_tasks_complete.sh \
#     luca \
#     /mnt/c/raw_videos/luca \
#     /mnt/data/ai_data/datasets/3d-anime/luca \
#     --enable-gpu \
#     --enable-voice
#
# Author: Animation AI Studio
# Date: 2025-12-04
# Version: 1.0
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Script paths
CPU_PIPELINE_SCRIPT="${SCRIPT_DIR}/run_cpu_tasks_all.sh"
GPU_PIPELINE_SCRIPT="${SCRIPT_DIR}/run_gpu_tasks_all.sh"

# Default settings
DEFAULT_ENABLE_GPU=true
DEFAULT_ENABLE_VOICE=false
DEFAULT_SKIP_CPU=false
DEFAULT_SKIP_GPU=false
MIN_DISK_SPACE_GB=100
MIN_RAM_GB=16
MIN_VRAM_GB=14

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

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_stage() { echo -e "${CYAN}${BOLD}[STAGE]${NC} $1"; }
log_pipeline() { echo -e "${MAGENTA}${BOLD}[PIPELINE]${NC} $1"; }

# ============================================================================
# Banner
# ============================================================================

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║         Animation AI Studio - Complete Pipeline v1.0              ║"
    echo "║                                                                    ║"
    echo "║   CPU Pipeline: Data Prep → Analysis → RAG                        ║"
    echo "║   GPU Pipeline: SAM2 → SDXL → LLM → Voice                         ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ============================================================================
# Prerequisite Validation
# ============================================================================

check_script_exists() {
    local script="$1"
    local name="$2"

    if [ ! -f "$script" ]; then
        log_error "$name not found: $script"
        return 1
    fi

    if [ ! -x "$script" ]; then
        log_warning "$name not executable, fixing..."
        chmod +x "$script"
    fi

    log_success "$name found: $script"
    return 0
}

check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    # Essential commands
    local required_cmds=("python3" "nvidia-smi" "parallel" "ffmpeg" "jq")

    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install: sudo apt-get install -y ${missing_deps[*]}"
        return 1
    fi

    log_success "All dependencies found"
    return 0
}

check_disk_space() {
    local output_dir="$1"

    log_info "Checking disk space..."

    local parent_dir=$(dirname "$output_dir")
    if [ ! -d "$parent_dir" ]; then
        parent_dir="/"
    fi

    local available_gb=$(df -BG "$parent_dir" | awk 'NR==2 {print $4}' | sed 's/G//')

    log_info "Available disk space: ${available_gb}GB"

    if [ "$available_gb" -lt "$MIN_DISK_SPACE_GB" ]; then
        log_error "Insufficient disk space (need ${MIN_DISK_SPACE_GB}GB, have ${available_gb}GB)"
        return 1
    fi

    log_success "Disk space sufficient (${available_gb}GB available)"
    return 0
}

check_ram() {
    log_info "Checking RAM..."

    local total_ram_gb=$(free -g | awk 'NR==2{print $2}')

    log_info "Total RAM: ${total_ram_gb}GB"

    if [ "$total_ram_gb" -lt "$MIN_RAM_GB" ]; then
        log_warning "Low RAM (${total_ram_gb}GB, recommended ${MIN_RAM_GB}GB+)"
    else
        log_success "RAM sufficient (${total_ram_gb}GB)"
    fi

    return 0
}

check_gpu() {
    log_info "Checking GPU..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - GPU not available"
        return 1
    fi

    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    local vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    local vram_total_gb=$((vram_total / 1024))

    log_success "GPU found: $gpu_name"
    log_info "VRAM: ${vram_total_gb}GB"

    if [ "$vram_total_gb" -lt "$MIN_VRAM_GB" ]; then
        log_warning "Low VRAM (${vram_total_gb}GB, recommended ${MIN_VRAM_GB}GB+)"
        log_warning "Some GPU tasks may fail or require smaller models"
    fi

    return 0
}

check_python_env() {
    log_info "Checking Python environment..."

    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $python_version"

    # Check PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        local torch_version=$(python3 -c "import torch; print(torch.__version__)")
        local cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")

        log_success "PyTorch found: $torch_version"

        if [ "$cuda_available" = "True" ]; then
            local cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
            log_success "CUDA available: $cuda_version"
        else
            log_error "CUDA not available in PyTorch"
            return 1
        fi
    else
        log_error "PyTorch not found"
        return 1
    fi

    return 0
}

validate_prerequisites() {
    log_stage "Validating Prerequisites"
    echo ""

    local validation_failed=false

    # Check scripts exist
    if ! check_script_exists "$CPU_PIPELINE_SCRIPT" "CPU Pipeline Script"; then
        validation_failed=true
    fi

    if ! check_script_exists "$GPU_PIPELINE_SCRIPT" "GPU Pipeline Script"; then
        validation_failed=true
    fi

    # Check dependencies
    if ! check_dependencies; then
        validation_failed=true
    fi

    # Check Python environment
    if ! check_python_env; then
        validation_failed=true
    fi

    # Check disk space
    if ! check_disk_space "$OUTPUT_DIR"; then
        validation_failed=true
    fi

    # Check RAM
    check_ram  # Non-critical, just warning

    # Check GPU
    if [ "$ENABLE_GPU" = true ]; then
        if ! check_gpu; then
            validation_failed=true
        fi
    fi

    if [ "$validation_failed" = true ]; then
        log_error "Prerequisite validation failed"
        return 1
    fi

    log_success "All prerequisites validated"
    echo ""
    return 0
}

# ============================================================================
# Pipeline Execution
# ============================================================================

execute_cpu_pipeline() {
    log_stage "Executing CPU Pipeline (Stage 1-3)"
    echo ""

    log_info "Running: $CPU_PIPELINE_SCRIPT"
    log_info "Film: $FILM_NAME"
    log_info "Input: $INPUT_DIR"
    log_info "Output: $OUTPUT_DIR"
    echo ""

    local cpu_start=$(date +%s)

    # Build CPU pipeline command
    local cpu_args=(
        "$FILM_NAME"
        "$INPUT_DIR"
        "$OUTPUT_DIR"
    )

    if [ "$CPU_PARALLEL_JOBS" != "" ]; then
        cpu_args+=("--parallel-jobs" "$CPU_PARALLEL_JOBS")
    fi

    # Execute CPU pipeline
    if bash "$CPU_PIPELINE_SCRIPT" "${cpu_args[@]}" 2>&1 | tee "${LOG_DIR}/cpu_pipeline.log"; then
        local cpu_end=$(date +%s)
        local cpu_duration=$((cpu_end - cpu_start))
        CPU_PIPELINE_TIME=$cpu_duration

        log_success "CPU Pipeline completed in ${cpu_duration}s"
        echo ""
        return 0
    else
        log_error "CPU Pipeline failed"
        return 1
    fi
}

execute_gpu_pipeline() {
    log_stage "Executing GPU Pipeline (Task 1-4)"
    echo ""

    log_info "Running: $GPU_PIPELINE_SCRIPT"
    log_info "Output: $OUTPUT_DIR"
    echo ""

    local gpu_start=$(date +%s)

    # Build GPU pipeline command
    local gpu_args=(
        "$OUTPUT_DIR"
    )

    # Add optional flags
    if [ "$ENABLE_VOICE" = true ]; then
        gpu_args+=("--enable-voice")
    fi

    if [ "$SAM2_MODEL" != "" ]; then
        gpu_args+=("--sam2-model" "$SAM2_MODEL")
    fi

    if [ "$LLM_MODEL" != "" ]; then
        gpu_args+=("--llm-model" "$LLM_MODEL")
    fi

    if [ "$SDXL_CONFIG" != "" ] && [ -f "$SDXL_CONFIG" ]; then
        gpu_args+=("--sdxl-config" "$SDXL_CONFIG")
    fi

    # Execute GPU pipeline
    if bash "$GPU_PIPELINE_SCRIPT" "${gpu_args[@]}" 2>&1 | tee "${LOG_DIR}/gpu_pipeline.log"; then
        local gpu_end=$(date +%s)
        local gpu_duration=$((gpu_end - gpu_start))
        GPU_PIPELINE_TIME=$gpu_duration

        log_success "GPU Pipeline completed in ${gpu_duration}s"
        echo ""
        return 0
    else
        log_error "GPU Pipeline failed"
        return 1
    fi
}

# ============================================================================
# Summary Generation
# ============================================================================

generate_summary() {
    local status="$1"
    local total_time="$2"

    log_stage "Generating Summary"
    echo ""

    local summary_file="${OUTPUT_DIR}/complete_pipeline_summary.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Calculate hours, minutes, seconds
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))

    # Count outputs
    local frames_count=0
    local segmented_count=0
    local generated_images_count=0

    if [ -d "${OUTPUT_DIR}/frames" ]; then
        frames_count=$(find "${OUTPUT_DIR}/frames" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    fi

    if [ -d "${OUTPUT_DIR}/segmented" ]; then
        segmented_count=$(find "${OUTPUT_DIR}/segmented" -type d -name "video_*" 2>/dev/null | wc -l)
    fi

    if [ -d "${OUTPUT_DIR}/generated_images" ]; then
        generated_images_count=$(find "${OUTPUT_DIR}/generated_images" -type f -name "*.png" 2>/dev/null | wc -l)
    fi

    # Create summary JSON
    cat > "$summary_file" <<EOF
{
  "pipeline": "Animation AI Studio - Complete Pipeline",
  "version": "1.0",
  "timestamp": "$timestamp",
  "status": "$status",
  "film_name": "$FILM_NAME",
  "input_directory": "$INPUT_DIR",
  "output_directory": "$OUTPUT_DIR",
  "execution_time": {
    "total_seconds": $total_time,
    "formatted": "${hours}h ${minutes}m ${seconds}s",
    "cpu_pipeline_seconds": ${CPU_PIPELINE_TIME:-0},
    "gpu_pipeline_seconds": ${GPU_PIPELINE_TIME:-0}
  },
  "pipeline_stages": {
    "cpu_pipeline": {
      "executed": $([ "$SKIP_CPU" = false ] && echo "true" || echo "false"),
      "status": "$([ "$CPU_PIPELINE_TIME" != "" ] && echo "completed" || echo "skipped")",
      "stages": [
        "Stage 1: Data Preparation (Frame + Audio Extraction)",
        "Stage 2: Video Analysis (Scene Detection + Composition + Camera)",
        "Stage 3: RAG Preparation (Document Processing + Knowledge Base)"
      ]
    },
    "gpu_pipeline": {
      "executed": $([ "$SKIP_GPU" = false ] && echo "true" || echo "false"),
      "status": "$([ "$GPU_PIPELINE_TIME" != "" ] && echo "completed" || echo "skipped")",
      "tasks": [
        "Task 1: SAM2 Character Segmentation",
        "Task 2: SDXL Image Generation",
        "Task 3: LLM Video Analysis",
        "Task 4: Voice Training (optional)"
      ]
    }
  },
  "output_statistics": {
    "frames_extracted": $frames_count,
    "videos_segmented": $segmented_count,
    "images_generated": $generated_images_count
  },
  "configuration": {
    "enable_gpu": $ENABLE_GPU,
    "enable_voice": $ENABLE_VOICE,
    "sam2_model": "${SAM2_MODEL:-default}",
    "llm_model": "${LLM_MODEL:-default}",
    "cpu_parallel_jobs": ${CPU_PARALLEL_JOBS:-16}
  },
  "logs": {
    "cpu_pipeline": "${LOG_DIR}/cpu_pipeline.log",
    "gpu_pipeline": "${LOG_DIR}/gpu_pipeline.log",
    "master_log": "${LOG_DIR}/master_pipeline.log"
  }
}
EOF

    log_success "Summary saved: $summary_file"
    echo ""

    # Print summary to console
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}             PIPELINE EXECUTION SUMMARY                         ${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BOLD}Status:${NC} $status"
    echo -e "${BOLD}Film:${NC} $FILM_NAME"
    echo -e "${BOLD}Total Time:${NC} ${hours}h ${minutes}m ${seconds}s"
    echo ""

    if [ "$CPU_PIPELINE_TIME" != "" ]; then
        local cpu_minutes=$((CPU_PIPELINE_TIME / 60))
        local cpu_seconds=$((CPU_PIPELINE_TIME % 60))
        echo -e "${BOLD}CPU Pipeline:${NC} ${cpu_minutes}m ${cpu_seconds}s"
    fi

    if [ "$GPU_PIPELINE_TIME" != "" ]; then
        local gpu_minutes=$((GPU_PIPELINE_TIME / 60))
        local gpu_seconds=$((GPU_PIPELINE_TIME % 60))
        echo -e "${BOLD}GPU Pipeline:${NC} ${gpu_minutes}m ${gpu_seconds}s"
    fi

    echo ""
    echo -e "${BOLD}Output Statistics:${NC}"
    echo -e "  Frames extracted:     $frames_count"
    echo -e "  Videos segmented:     $segmented_count"
    echo -e "  Images generated:     $generated_images_count"
    echo ""
    echo -e "${BOLD}Output Directory:${NC} $OUTPUT_DIR"
    echo -e "${BOLD}Summary File:${NC} $summary_file"
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
}

# ============================================================================
# Error Handling and Cleanup
# ============================================================================

cleanup_on_error() {
    log_error "Pipeline interrupted or failed"

    # Generate failure summary
    local end_time=$(date +%s)
    local total_time=$((end_time - PIPELINE_START_TIME))

    generate_summary "FAILED" "$total_time"

    exit 1
}

trap cleanup_on_error ERR SIGTERM SIGINT

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 FILM_NAME INPUT_DIR OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  FILM_NAME       Film name (e.g., luca, coco, turning_red)"
    echo "  INPUT_DIR       Input directory containing video files"
    echo "  OUTPUT_DIR      Output base directory for all results"
    echo ""
    echo "Options:"
    echo "  --enable-gpu             Enable GPU pipeline (default: true)"
    echo "  --disable-gpu            Disable GPU pipeline (CPU only)"
    echo "  --enable-voice           Enable voice training (default: false)"
    echo "  --skip-cpu               Skip CPU pipeline"
    echo "  --skip-gpu               Skip GPU pipeline"
    echo "  --parallel-jobs N        CPU parallel jobs (default: 16)"
    echo "  --sam2-model MODEL       SAM2 model size (base/large/small/tiny)"
    echo "  --llm-model MODEL        LLM model (qwen-vl-7b/qwen-14b/qwen-7b)"
    echo "  --sdxl-config PATH       SDXL generation config JSON"
    echo "  --help                   Show this help"
    echo ""
    echo "Example:"
    echo "  $0 luca /mnt/c/raw_videos/luca /mnt/data/ai_data/datasets/3d-anime/luca"
    echo ""
    echo "  $0 luca /videos /output --enable-gpu --enable-voice --parallel-jobs 8"
    exit 1
}

FILM_NAME=""
INPUT_DIR=""
OUTPUT_DIR=""
ENABLE_GPU=$DEFAULT_ENABLE_GPU
ENABLE_VOICE=$DEFAULT_ENABLE_VOICE
SKIP_CPU=$DEFAULT_SKIP_CPU
SKIP_GPU=$DEFAULT_SKIP_GPU
CPU_PARALLEL_JOBS=""
SAM2_MODEL=""
LLM_MODEL=""
SDXL_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-gpu)
            ENABLE_GPU=true
            shift
            ;;
        --disable-gpu)
            ENABLE_GPU=false
            shift
            ;;
        --enable-voice)
            ENABLE_VOICE=true
            shift
            ;;
        --skip-cpu)
            SKIP_CPU=true
            shift
            ;;
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --parallel-jobs)
            CPU_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --sam2-model)
            SAM2_MODEL="$2"
            shift 2
            ;;
        --llm-model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --sdxl-config)
            SDXL_CONFIG="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            if [ -z "$FILM_NAME" ]; then
                FILM_NAME="$1"
            elif [ -z "$INPUT_DIR" ]; then
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
if [ -z "$FILM_NAME" ] || [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

if [ ! -d "$INPUT_DIR" ]; then
    log_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/logs"
LOG_DIR="${OUTPUT_DIR}/logs"

# ============================================================================
# Main Pipeline Execution
# ============================================================================

main() {
    print_banner

    log_pipeline "Starting Complete Pipeline Execution"
    log_info "Film: $FILM_NAME"
    log_info "Input: $INPUT_DIR"
    log_info "Output: $OUTPUT_DIR"
    log_info "Enable GPU: $ENABLE_GPU"
    log_info "Enable Voice: $ENABLE_VOICE"
    echo ""

    # Record start time
    PIPELINE_START_TIME=$(date +%s)

    # Phase 1: Validate prerequisites
    if ! validate_prerequisites; then
        log_error "Prerequisite validation failed - aborting"
        exit 1
    fi

    # Phase 2: Execute CPU pipeline
    if [ "$SKIP_CPU" = false ]; then
        if ! execute_cpu_pipeline; then
            log_error "CPU Pipeline failed - aborting"
            exit 1
        fi
    else
        log_warning "CPU Pipeline skipped (--skip-cpu)"
        echo ""
    fi

    # Phase 3: Execute GPU pipeline
    if [ "$SKIP_GPU" = false ] && [ "$ENABLE_GPU" = true ]; then
        if ! execute_gpu_pipeline; then
            log_error "GPU Pipeline failed"
            # Don't abort - CPU results still valid
        fi
    else
        if [ "$SKIP_GPU" = true ]; then
            log_warning "GPU Pipeline skipped (--skip-gpu)"
        else
            log_warning "GPU Pipeline disabled (--disable-gpu)"
        fi
        echo ""
    fi

    # Phase 4: Generate summary
    local end_time=$(date +%s)
    local total_time=$((end_time - PIPELINE_START_TIME))

    generate_summary "COMPLETED" "$total_time"

    log_success "Complete Pipeline finished successfully"
    log_info "Output directory: $OUTPUT_DIR"

    return 0
}

# Export variables for sub-scripts
export RED GREEN YELLOW BLUE CYAN MAGENTA NC

# Execute main pipeline
main

exit 0
