#!/bin/bash
################################################################################
# Master CPU Tasks Orchestration Script
#
# Executes all CPU-only processing stages in sequence:
#   Stage 1: Data Preparation (frame/audio extraction)
#   Stage 2: Video Analysis (scene/composition/camera)
#   Stage 3: RAG Preparation (documents/knowledge base)
#
# Features:
#   - Automatic stage execution with dependency handling
#   - Integrated resource monitoring
#   - Comprehensive error handling and rollback
#   - Progress tracking and reporting
#   - Email notifications (optional)
#   - Execution time estimation and tracking
#
# Hardware Requirements:
#   - CPU: 8+ cores recommended
#   - RAM: 16GB+ recommended
#   - Disk: 50-100GB free space per film
#   - NO GPU required (100% CPU-only)
#
# Usage:
#   bash scripts/batch/run_cpu_tasks_all.sh \
#     FILM_NAME \
#     INPUT_VIDEO_DIR \
#     OUTPUT_BASE_DIR \
#     [OPTIONS]
#
# Example:
#   bash scripts/batch/run_cpu_tasks_all.sh \
#     luca \
#     /mnt/c/raw_videos/luca \
#     /mnt/data/ai_data/datasets/3d-anime/luca \
#     --workers 8 \
#     --monitor
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

# Stage scripts
STAGE1_SCRIPT="${SCRIPT_DIR}/cpu_tasks_stage1_data_prep.sh"
STAGE2_SCRIPT="${SCRIPT_DIR}/cpu_tasks_stage2_analysis.sh"
STAGE3_SCRIPT="${SCRIPT_DIR}/cpu_tasks_stage3_rag_prep.sh"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor_resources.sh"

# Default settings
DEFAULT_WORKERS=8
MONITOR_INTERVAL=30  # Monitor every 30 seconds

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
log_stage() { echo -e "${BOLD}${BLUE}[STAGE $1]${NC} $2"; }

# ============================================================================
# Time Tracking
# ============================================================================

STAGE_START_TIME=0
TOTAL_START_TIME=0

start_timer() {
    local timer_name="$1"
    if [ "$timer_name" = "total" ]; then
        TOTAL_START_TIME=$(date +%s)
    else
        STAGE_START_TIME=$(date +%s)
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
# Resource Monitoring
# ============================================================================

MONITOR_PID=""

start_monitoring() {
    local log_dir="$1"

    if [ ! -f "$MONITOR_SCRIPT" ]; then
        log_warning "Monitor script not found, skipping resource monitoring"
        return
    fi

    log_info "Starting resource monitor (daemon mode)"

    # Start monitor in background
    bash "$MONITOR_SCRIPT" \
        --daemon \
        --interval "$MONITOR_INTERVAL" \
        --log-dir "$log_dir" &

    MONITOR_PID=$!
    log_info "Monitor started (PID: $MONITOR_PID)"

    # Wait a moment for monitor to initialize
    sleep 2
}

stop_monitoring() {
    if [ -n "$MONITOR_PID" ] && ps -p "$MONITOR_PID" > /dev/null 2>&1; then
        log_info "Stopping resource monitor (PID: $MONITOR_PID)"
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

# ============================================================================
# Stage Execution
# ============================================================================

execute_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local stage_script="$3"
    shift 3
    local stage_args=("$@")

    log_header
    log_stage "$stage_num" "$stage_name"
    log_header
    echo ""

    # Check if script exists
    if [ ! -f "$stage_script" ]; then
        log_error "Stage script not found: $stage_script"
        return 1
    fi

    # Start timer
    start_timer "stage"

    # Execute stage
    log_info "Executing: bash $stage_script ${stage_args[*]}"
    echo ""

    if bash "$stage_script" "${stage_args[@]}"; then
        local elapsed=$(get_elapsed_time "$STAGE_START_TIME")
        echo ""
        log_success "Stage $stage_num completed in $elapsed"
        return 0
    else
        local elapsed=$(get_elapsed_time "$STAGE_START_TIME")
        echo ""
        log_error "Stage $stage_num failed after $elapsed"
        return 1
    fi
}

# ============================================================================
# Main Processing Pipeline
# ============================================================================

run_cpu_pipeline() {
    local film_name="$1"
    local input_dir="$2"
    local output_dir="$3"
    local num_workers="$4"
    local enable_monitoring="$5"
    local resume_mode="$6"

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/logs"
    mkdir -p "$output_dir/monitoring"

    # Save execution metadata
    local metadata_file="${output_dir}/execution_metadata.json"
    cat > "$metadata_file" <<EOF
{
  "pipeline": "cpu_tasks_all",
  "film_name": "$film_name",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "workers": $num_workers,
  "monitoring_enabled": $enable_monitoring,
  "resume_mode": $resume_mode,
  "start_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "hostname": "$(hostname)",
  "user": "$(whoami)"
}
EOF

    # Display configuration
    log_header
    log_section "CPU PIPELINE CONFIGURATION"
    log_header
    echo ""
    log_info "Film Name:        $film_name"
    log_info "Input Directory:  $input_dir"
    log_info "Output Directory: $output_dir"
    log_info "Workers:          $num_workers"
    log_info "Monitoring:       $enable_monitoring"
    log_info "Resume Mode:      $resume_mode"
    log_info "Start Time:       $(date)"
    echo ""

    # Start total timer
    start_timer "total"

    # Start resource monitoring (if enabled)
    if [ "$enable_monitoring" = "true" ]; then
        start_monitoring "${output_dir}/monitoring"
    fi

    # ========================================================================
    # Stage 1: Data Preparation
    # ========================================================================

    local stage1_args=(
        "$input_dir"
        "$output_dir"
        "--workers" "$num_workers"
    )

    if [ "$resume_mode" = "true" ]; then
        stage1_args+=("--resume")
    fi

    if ! execute_stage "1" "Data Preparation (Frame/Audio Extraction)" "$STAGE1_SCRIPT" "${stage1_args[@]}"; then
        log_error "Pipeline failed at Stage 1"
        stop_monitoring
        return 1
    fi

    # ========================================================================
    # Stage 2: Video Analysis
    # ========================================================================

    local stage2_args=(
        "$output_dir"
        "--workers" "$num_workers"
    )

    if [ "$resume_mode" = "true" ]; then
        stage2_args+=("--resume")
    fi

    if ! execute_stage "2" "Video Analysis (Scene/Composition/Camera)" "$STAGE2_SCRIPT" "${stage2_args[@]}"; then
        log_error "Pipeline failed at Stage 2"
        stop_monitoring
        return 1
    fi

    # ========================================================================
    # Stage 3: RAG Preparation
    # ========================================================================

    local stage3_args=(
        "$film_name"
        "$output_dir"
    )

    if ! execute_stage "3" "RAG Preparation (Documents/Knowledge Base)" "$STAGE3_SCRIPT" "${stage3_args[@]}"; then
        log_error "Pipeline failed at Stage 3"
        stop_monitoring
        return 1
    fi

    # ========================================================================
    # Pipeline Complete
    # ========================================================================

    # Stop monitoring
    if [ "$enable_monitoring" = "true" ]; then
        stop_monitoring
    fi

    # Calculate total time
    local total_elapsed=$(get_elapsed_time "$TOTAL_START_TIME")

    # Update metadata
    cat > "$metadata_file" <<EOF
{
  "pipeline": "cpu_tasks_all",
  "film_name": "$film_name",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "workers": $num_workers,
  "monitoring_enabled": $enable_monitoring,
  "resume_mode": $resume_mode,
  "start_time": "$(date -u -d @${TOTAL_START_TIME} +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")",
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
    log_section "PIPELINE COMPLETED SUCCESSFULLY"
    log_header
    echo ""
    log_success "All CPU stages completed"
    log_info "Total execution time: $total_elapsed"
    log_info "Output directory: $output_dir"
    echo ""

    # Show output summary
    log_section "OUTPUT SUMMARY"
    echo ""

    local frame_count=$(find "$output_dir/frames" -type f -name "*.jpg" 2>/dev/null | wc -l)
    local audio_count=$(find "$output_dir/audio" -type f -name "*.wav" 2>/dev/null | wc -l)
    local scene_count=$(find "$output_dir/analysis/scenes" -type f -name "*.json" 2>/dev/null | wc -l)
    local rag_docs=$(find "$output_dir/rag/documents" -type f 2>/dev/null | wc -l)

    log_info "Frames extracted:     $frame_count"
    log_info "Audio files:          $audio_count"
    log_info "Scene analyses:       $scene_count"
    log_info "RAG documents:        $rag_docs"
    echo ""

    if [ "$enable_monitoring" = "true" ]; then
        log_info "Resource logs:        ${output_dir}/monitoring/"
    fi

    log_info "Metadata:             $metadata_file"
    echo ""
    log_header

    return 0
}

# ============================================================================
# Error Handler
# ============================================================================

cleanup_on_error() {
    log_error "Pipeline interrupted or failed"
    stop_monitoring
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
    echo "  FILM_NAME     Name of the film (e.g., luca, coco)"
    echo "  INPUT_DIR     Directory containing raw video files"
    echo "  OUTPUT_DIR    Base output directory for all results"
    echo ""
    echo "Options:"
    echo "  --workers N        Number of parallel workers (default: $DEFAULT_WORKERS)"
    echo "  --monitor          Enable resource monitoring (daemon mode)"
    echo "  --resume           Resume from checkpoints (skip completed stages)"
    echo "  --help             Show this help"
    echo ""
    echo "Example:"
    echo "  $0 luca /mnt/c/raw_videos/luca /mnt/data/ai_data/datasets/3d-anime/luca \\"
    echo "     --workers 8 --monitor --resume"
    echo ""
    echo "Output Structure:"
    echo "  OUTPUT_DIR/"
    echo "  ├── frames/              # Stage 1: Extracted frames"
    echo "  ├── audio/               # Stage 1: Extracted audio"
    echo "  ├── analysis/            # Stage 2: Video analysis"
    echo "  │   ├── scenes/"
    echo "  │   ├── composition/"
    echo "  │   └── camera/"
    echo "  ├── rag/                 # Stage 3: RAG preparation"
    echo "  │   ├── documents/"
    echo "  │   └── knowledge_base/"
    echo "  ├── monitoring/          # Resource monitoring logs"
    echo "  ├── logs/                # Stage execution logs"
    echo "  └── execution_metadata.json"
    exit 1
}

FILM_NAME=""
INPUT_DIR=""
OUTPUT_DIR=""
NUM_WORKERS=$DEFAULT_WORKERS
ENABLE_MONITORING="false"
RESUME_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --monitor)
            ENABLE_MONITORING="true"
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

# Validate required arguments
if [ -z "$FILM_NAME" ] || [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    log_error "Missing required arguments"
    usage
fi

if [ ! -d "$INPUT_DIR" ]; then
    log_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Validate workers
if [ "$NUM_WORKERS" -lt 1 ]; then
    log_error "Workers must be >= 1"
    exit 1
fi

# ============================================================================
# Main Execution
# ============================================================================

run_cpu_pipeline "$FILM_NAME" "$INPUT_DIR" "$OUTPUT_DIR" "$NUM_WORKERS" "$ENABLE_MONITORING" "$RESUME_MODE"

log_success "CPU pipeline execution completed successfully"
exit 0
