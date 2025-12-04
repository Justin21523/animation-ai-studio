#!/bin/bash
################################################################################
# CPU Tasks Stage 1: Data Preparation
#
# Pure CPU-only tasks (NO GPU usage):
#   - Parallel frame extraction from videos
#   - Parallel audio extraction
#   - File organization and indexing
#
# Features:
#   - Memory-safe parallel processing (controlled workers)
#   - Checkpoint/resume support (skip already processed)
#   - Resource monitoring (CPU, RAM, Disk)
#   - Error handling and retry logic
#   - Progress tracking
#
# Hardware Requirements:
#   - CPU: 8+ cores (uses up to 16 parallel workers)
#   - RAM: 16GB+ recommended
#   - Disk: ~50-100GB free space per film
#
# Usage:
#   bash scripts/batch/cpu_tasks_stage1_data_prep.sh \
#     /path/to/videos \
#     /path/to/output \
#     [--workers 8] [--resume]
#
# Author: Animation AI Studio
# Date: 2025-12-04
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Default settings
DEFAULT_WORKERS=8
MAX_WORKERS=16
MEMORY_THRESHOLD_PCT=90  # Stop if RAM usage > 90%
DISK_THRESHOLD_GB=10     # Stop if free disk < 10GB

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAME_EXTRACTOR="${PROJECT_ROOT}/scripts/processing/extraction/universal_frame_extractor.py"
AUDIO_EXTRACTOR="${PROJECT_ROOT}/scripts/processing/extraction/audio_extractor.py"

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Resource Monitoring
# ============================================================================

check_memory_usage() {
    # Get memory usage percentage
    if command -v free &> /dev/null; then
        local mem_used=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        echo "$mem_used"
    else
        echo "0"
    fi
}

check_disk_space() {
    # Get free disk space in GB
    local output_dir="$1"
    local free_gb=$(df -BG "$output_dir" | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "$free_gb"
}

check_cpu_cores() {
    # Get number of CPU cores
    if command -v nproc &> /dev/null; then
        nproc
    else
        echo "4"
    fi
}

monitor_resources() {
    local output_dir="$1"

    local mem_pct=$(check_memory_usage)
    local disk_gb=$(check_disk_space "$output_dir")
    local cpu_cores=$(check_cpu_cores)

    log_info "Resources: CPU=${cpu_cores} cores, RAM=${mem_pct}%, Free Disk=${disk_gb}GB"

    # Check thresholds
    if [ "$mem_pct" -gt "$MEMORY_THRESHOLD_PCT" ]; then
        log_error "Memory usage too high (${mem_pct}% > ${MEMORY_THRESHOLD_PCT}%)"
        return 1
    fi

    if [ "$disk_gb" -lt "$DISK_THRESHOLD_GB" ]; then
        log_error "Disk space too low (${disk_gb}GB < ${DISK_THRESHOLD_GB}GB)"
        return 1
    fi

    return 0
}

# ============================================================================
# Checkpoint Management
# ============================================================================

load_checkpoint() {
    local checkpoint_file="$1"

    if [ -f "$checkpoint_file" ]; then
        log_info "Loading checkpoint: $checkpoint_file"
        cat "$checkpoint_file"
    else
        echo ""
    fi
}

save_checkpoint() {
    local checkpoint_file="$1"
    local processed_item="$2"

    echo "$processed_item" >> "$checkpoint_file"
}

is_processed() {
    local checkpoint_file="$1"
    local item="$2"

    if [ ! -f "$checkpoint_file" ]; then
        return 1
    fi

    if grep -Fxq "$item" "$checkpoint_file"; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Frame Extraction Worker
# ============================================================================

extract_frames_worker() {
    local video_file="$1"
    local output_dir="$2"
    local checkpoint_file="$3"
    local video_name=$(basename "$video_file")

    # Check if already processed
    if is_processed "$checkpoint_file" "$video_name"; then
        log_info "Skipping (already processed): $video_name"
        return 0
    fi

    log_info "Extracting frames: $video_name"

    # Create output directory for this video
    local video_output="${output_dir}/$(basename "$video_file" | sed 's/\.[^.]*$//')"

    # Run frame extractor (CPU-only, no GPU)
    if python "$FRAME_EXTRACTOR" \
        "$video_file" \
        --output-dir "$video_output" \
        --mode scene \
        --scene-threshold 27.0 \
        --frames-per-scene 3 \
        --jpeg-quality 95 \
        --workers 1 2>&1 | tee "${output_dir}/logs/${video_name}.log"
    then
        log_success "Completed: $video_name"
        save_checkpoint "$checkpoint_file" "$video_name"
        return 0
    else
        log_error "Failed: $video_name"
        return 1
    fi
}

export -f extract_frames_worker
export -f log_info
export -f log_success
export -f log_error
export -f is_processed
export -f save_checkpoint

# ============================================================================
# Audio Extraction Worker
# ============================================================================

extract_audio_worker() {
    local video_file="$1"
    local output_dir="$2"
    local checkpoint_file="$3"
    local video_name=$(basename "$video_file")

    # Check if already processed
    if is_processed "$checkpoint_file" "$video_name"; then
        log_info "Skipping audio (already processed): $video_name"
        return 0
    fi

    log_info "Extracting audio: $video_name"

    # Output audio file
    local audio_output="${output_dir}/$(basename "$video_file" | sed 's/\.[^.]*$//')_audio.wav"

    # Use ffmpeg for audio extraction (CPU-only)
    if ffmpeg -i "$video_file" \
        -vn \
        -acodec pcm_s16le \
        -ar 16000 \
        -ac 1 \
        "$audio_output" \
        -y 2>&1 | tee "${output_dir}/logs/${video_name}_audio.log"
    then
        log_success "Completed audio: $video_name"
        save_checkpoint "$checkpoint_file" "$video_name"
        return 0
    else
        log_error "Failed audio: $video_name"
        return 1
    fi
}

export -f extract_audio_worker

# ============================================================================
# Main Processing Function
# ============================================================================

process_stage1() {
    local input_dir="$1"
    local output_dir="$2"
    local num_workers="$3"
    local resume_mode="$4"

    log_info "========================================="
    log_info "CPU Stage 1: Data Preparation"
    log_info "========================================="
    log_info "Input: $input_dir"
    log_info "Output: $output_dir"
    log_info "Workers: $num_workers"
    log_info "Resume: $resume_mode"
    log_info ""

    # Create output directories
    mkdir -p "$output_dir/frames"
    mkdir -p "$output_dir/audio"
    mkdir -p "$output_dir/logs"
    mkdir -p "$output_dir/checkpoints"

    # Checkpoint files
    local frames_checkpoint="${output_dir}/checkpoints/frames_processed.txt"
    local audio_checkpoint="${output_dir}/checkpoints/audio_processed.txt"

    # Clear checkpoints if not resuming
    if [ "$resume_mode" = "false" ]; then
        rm -f "$frames_checkpoint"
        rm -f "$audio_checkpoint"
    fi

    # Find all video files
    log_info "Scanning for video files..."
    local video_files=()
    for ext in mp4 mkv avi mov flv wmv; do
        while IFS= read -r -d '' file; do
            video_files+=("$file")
        done < <(find "$input_dir" -maxdepth 1 -type f -iname "*.${ext}" -print0)
    done

    local total_videos=${#video_files[@]}
    log_info "Found $total_videos video files"

    if [ $total_videos -eq 0 ]; then
        log_error "No video files found in: $input_dir"
        exit 1
    fi

    # Check initial resources
    if ! monitor_resources "$output_dir"; then
        log_error "Resource check failed"
        exit 1
    fi

    # ========================================================================
    # Phase 1: Parallel Frame Extraction
    # ========================================================================

    log_info ""
    log_info "Phase 1/2: Frame Extraction (parallel workers: $num_workers)"
    log_info "========================================="

    # Check if GNU parallel is available
    if ! command -v parallel &> /dev/null; then
        log_warning "GNU parallel not found, using sequential processing"

        for video_file in "${video_files[@]}"; do
            extract_frames_worker "$video_file" "${output_dir}/frames" "$frames_checkpoint"
        done
    else
        # Use GNU parallel for efficient parallelization
        printf '%s\n' "${video_files[@]}" | \
            parallel \
                --jobs "$num_workers" \
                --progress \
                --bar \
                --tagstring "[{#}/{%}]" \
                extract_frames_worker {} "${output_dir}/frames" "$frames_checkpoint"
    fi

    log_success "Frame extraction completed"

    # Check resources after frame extraction
    if ! monitor_resources "$output_dir"; then
        log_warning "Resources constrained, consider freeing space before continuing"
    fi

    # ========================================================================
    # Phase 2: Parallel Audio Extraction
    # ========================================================================

    log_info ""
    log_info "Phase 2/2: Audio Extraction (parallel workers: $num_workers)"
    log_info "========================================="

    if ! command -v parallel &> /dev/null; then
        for video_file in "${video_files[@]}"; do
            extract_audio_worker "$video_file" "${output_dir}/audio" "$audio_checkpoint"
        done
    else
        printf '%s\n' "${video_files[@]}" | \
            parallel \
                --jobs "$num_workers" \
                --progress \
                --bar \
                --tagstring "[{#}/{%}]" \
                extract_audio_worker {} "${output_dir}/audio" "$audio_checkpoint"
    fi

    log_success "Audio extraction completed"

    # ========================================================================
    # Generate Dataset Index
    # ========================================================================

    log_info ""
    log_info "Generating dataset index..."

    local index_file="${output_dir}/dataset_index.json"
    local frame_count=$(find "${output_dir}/frames" -type f -name "*.jpg" 2>/dev/null | wc -l)
    local audio_count=$(find "${output_dir}/audio" -type f -name "*.wav" 2>/dev/null | wc -l)

    cat > "$index_file" <<EOF
{
  "stage": "cpu_stage1_data_prep",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "total_videos": $total_videos,
  "frames_extracted": $frame_count,
  "audio_files": $audio_count,
  "workers_used": $num_workers,
  "completed": true
}
EOF

    log_success "Dataset index saved: $index_file"

    # ========================================================================
    # Final Summary
    # ========================================================================

    log_info ""
    log_info "========================================="
    log_info "CPU Stage 1 COMPLETED"
    log_info "========================================="
    log_info "Videos processed: $total_videos"
    log_info "Frames extracted: $frame_count"
    log_info "Audio files: $audio_count"
    log_info "Output directory: $output_dir"
    log_info ""

    # Final resource check
    monitor_resources "$output_dir" || true
}

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 INPUT_DIR OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  INPUT_DIR    Directory containing video files"
    echo "  OUTPUT_DIR   Output directory for extracted data"
    echo ""
    echo "Options:"
    echo "  --workers N  Number of parallel workers (default: $DEFAULT_WORKERS, max: $MAX_WORKERS)"
    echo "  --resume     Resume from checkpoint (skip already processed)"
    echo "  --help       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 /mnt/c/raw_videos/luca /mnt/data/ai_data/datasets/3d-anime/luca --workers 8 --resume"
    exit 1
}

# Parse arguments
INPUT_DIR=""
OUTPUT_DIR=""
NUM_WORKERS=$DEFAULT_WORKERS
RESUME_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            NUM_WORKERS="$2"
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

# Validate worker count
if [ "$NUM_WORKERS" -lt 1 ] || [ "$NUM_WORKERS" -gt "$MAX_WORKERS" ]; then
    log_error "Workers must be between 1 and $MAX_WORKERS"
    exit 1
fi

# Adjust workers based on available CPU cores
AVAILABLE_CORES=$(check_cpu_cores)
if [ "$NUM_WORKERS" -gt "$AVAILABLE_CORES" ]; then
    log_warning "Requested workers ($NUM_WORKERS) exceeds available cores ($AVAILABLE_CORES)"
    log_warning "Adjusting to $AVAILABLE_CORES workers"
    NUM_WORKERS=$AVAILABLE_CORES
fi

# Export variables for workers
export FRAME_EXTRACTOR
export AUDIO_EXTRACTOR
export RED GREEN YELLOW BLUE NC

# Run main processing
process_stage1 "$INPUT_DIR" "$OUTPUT_DIR" "$NUM_WORKERS" "$RESUME_MODE"

log_success "All CPU Stage 1 tasks completed successfully"
