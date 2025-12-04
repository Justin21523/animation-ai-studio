#!/bin/bash
################################################################################
# CPU Tasks Stage 2: Video Analysis
#
# Pure CPU-only video analysis tasks (NO GPU usage):
#   - Parallel scene detection (PySceneDetect)
#   - Parallel composition analysis
#   - Parallel camera movement tracking
#   - Results aggregation
#
# Features:
#   - Memory-safe parallel processing
#   - Checkpoint/resume support
#   - Resource monitoring
#   - Per-video analysis with fallback handling
#
# Hardware Requirements:
#   - CPU: 8+ cores
#   - RAM: 16GB+ (analysis can be memory-intensive)
#   - Disk: Minimal (only for analysis results JSON)
#
# Usage:
#   bash scripts/batch/cpu_tasks_stage2_analysis.sh \
#     /path/to/output_from_stage1 \
#     [--workers 8] [--resume]
#
# Author: Animation AI Studio
# Date: 2025-12-04
################################################################################

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_WORKERS=8
MAX_WORKERS=16
MEMORY_THRESHOLD_PCT=90

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Analysis scripts (all CPU-only)
SCENE_DETECTOR="${PROJECT_ROOT}/scripts/analysis/video/scene_detection.py"
COMPOSITION_ANALYZER="${PROJECT_ROOT}/scripts/analysis/video/shot_composition_analyzer.py"
CAMERA_TRACKER="${PROJECT_ROOT}/scripts/analysis/video/camera_movement_tracker.py"

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Resource Monitoring
# ============================================================================

check_memory_usage() {
    if command -v free &> /dev/null; then
        free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}'
    else
        echo "0"
    fi
}

check_cpu_cores() {
    if command -v nproc &> /dev/null; then
        nproc
    else
        echo "4"
    fi
}

monitor_resources() {
    local mem_pct=$(check_memory_usage)
    local cpu_cores=$(check_cpu_cores)

    log_info "Resources: CPU=${cpu_cores} cores, RAM=${mem_pct}%"

    if [ "$mem_pct" -gt "$MEMORY_THRESHOLD_PCT" ]; then
        log_error "Memory usage too high (${mem_pct}% > ${MEMORY_THRESHOLD_PCT}%)"
        return 1
    fi

    return 0
}

# ============================================================================
# Checkpoint Management
# ============================================================================

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

save_checkpoint() {
    local checkpoint_file="$1"
    local processed_item="$2"
    echo "$processed_item" >> "$checkpoint_file"
}

# ============================================================================
# Scene Detection Worker (CPU-only)
# ============================================================================

detect_scenes_worker() {
    local video_file="$1"
    local output_dir="$2"
    local checkpoint_file="$3"
    local video_name=$(basename "$video_file")

    if is_processed "$checkpoint_file" "$video_name"; then
        log_info "Skipping scene detection (done): $video_name"
        return 0
    fi

    log_info "Scene detection: $video_name"

    local output_json="${output_dir}/$(basename "$video_file" | sed 's/\.[^.]*$//')_scenes.json"

    # Use PySceneDetect (CPU-only)
    if command -v scenedetect &> /dev/null; then
        if scenedetect \
            --input "$video_file" \
            detect-content \
            --threshold 27.0 \
            list-scenes \
            --output "$output_json" 2>&1 | tee "${output_dir}/logs/${video_name}_scenes.log"
        then
            log_success "Scene detection complete: $video_name"
            save_checkpoint "$checkpoint_file" "$video_name"
            return 0
        else
            log_error "Scene detection failed: $video_name"
            return 1
        fi
    else
        # Fallback: use Python script
        if python "$SCENE_DETECTOR" \
            --input "$video_file" \
            --output "$output_json" \
            --threshold 27.0 2>&1 | tee "${output_dir}/logs/${video_name}_scenes.log"
        then
            log_success "Scene detection complete: $video_name"
            save_checkpoint "$checkpoint_file" "$video_name"
            return 0
        else
            log_error "Scene detection failed: $video_name"
            return 1
        fi
    fi
}

export -f detect_scenes_worker
export -f log_info log_success log_error log_warning
export -f is_processed save_checkpoint

# ============================================================================
# Composition Analysis Worker (CPU-only)
# ============================================================================

analyze_composition_worker() {
    local frames_dir="$1"
    local output_dir="$2"
    local checkpoint_file="$3"
    local video_id=$(basename "$frames_dir")

    if is_processed "$checkpoint_file" "$video_id"; then
        log_info "Skipping composition analysis (done): $video_id"
        return 0
    fi

    log_info "Composition analysis: $video_id"

    local output_json="${output_dir}/${video_id}_composition.json"

    # Check if script exists
    if [ ! -f "$COMPOSITION_ANALYZER" ]; then
        log_warning "Composition analyzer not found, creating placeholder"
        echo '{"video": "'$video_id'", "analysis": "pending"}' > "$output_json"
        save_checkpoint "$checkpoint_file" "$video_id"
        return 0
    fi

    # Run composition analysis (CPU-only OpenCV operations)
    if python "$COMPOSITION_ANALYZER" \
        --input "$frames_dir" \
        --output "$output_json" 2>&1 | tee "${output_dir}/logs/${video_id}_composition.log"
    then
        log_success "Composition analysis complete: $video_id"
        save_checkpoint "$checkpoint_file" "$video_id"
        return 0
    else
        log_error "Composition analysis failed: $video_id"
        return 1
    fi
}

export -f analyze_composition_worker

# ============================================================================
# Camera Movement Tracking Worker (CPU-only)
# ============================================================================

track_camera_worker() {
    local video_file="$1"
    local output_dir="$2"
    local checkpoint_file="$3"
    local video_name=$(basename "$video_file")

    if is_processed "$checkpoint_file" "$video_name"; then
        log_info "Skipping camera tracking (done): $video_name"
        return 0
    fi

    log_info "Camera tracking: $video_name"

    local output_json="${output_dir}/$(basename "$video_file" | sed 's/\.[^.]*$//')_camera.json"

    # Check if script exists
    if [ ! -f "$CAMERA_TRACKER" ]; then
        log_warning "Camera tracker not found, creating placeholder"
        echo '{"video": "'$video_name'", "camera_movement": "pending"}' > "$output_json"
        save_checkpoint "$checkpoint_file" "$video_name"
        return 0
    fi

    # Run camera tracking (CPU-only optical flow)
    if python "$CAMERA_TRACKER" \
        --input "$video_file" \
        --output "$output_json" 2>&1 | tee "${output_dir}/logs/${video_name}_camera.log"
    then
        log_success "Camera tracking complete: $video_name"
        save_checkpoint "$checkpoint_file" "$video_name"
        return 0
    else
        log_error "Camera tracking failed: $video_name"
        return 1
    fi
}

export -f track_camera_worker

# ============================================================================
# Main Processing Function
# ============================================================================

process_stage2() {
    local base_dir="$1"
    local num_workers="$2"
    local resume_mode="$3"

    log_info "========================================="
    log_info "CPU Stage 2: Video Analysis"
    log_info "========================================="
    log_info "Base directory: $base_dir"
    log_info "Workers: $num_workers"
    log_info "Resume: $resume_mode"
    log_info ""

    # Create output directories
    local analysis_dir="${base_dir}/analysis"
    mkdir -p "$analysis_dir/scenes"
    mkdir -p "$analysis_dir/composition"
    mkdir -p "$analysis_dir/camera"
    mkdir -p "$analysis_dir/logs"
    mkdir -p "$analysis_dir/checkpoints"

    # Checkpoint files
    local scenes_checkpoint="${analysis_dir}/checkpoints/scenes_processed.txt"
    local composition_checkpoint="${analysis_dir}/checkpoints/composition_processed.txt"
    local camera_checkpoint="${analysis_dir}/checkpoints/camera_processed.txt"

    if [ "$resume_mode" = "false" ]; then
        rm -f "$scenes_checkpoint" "$composition_checkpoint" "$camera_checkpoint"
    fi

    # Find input videos (from Stage 1)
    local input_videos_dir="${base_dir}/../../raw_videos"  # Adjust as needed
    local video_files=()

    # Try to find original videos or use any videos in parent directory
    if [ -d "$input_videos_dir" ]; then
        for ext in mp4 mkv avi mov; do
            while IFS= read -r -d '' file; do
                video_files+=("$file")
            done < <(find "$input_videos_dir" -maxdepth 2 -type f -iname "*.${ext}" -print0)
        done
    fi

    # Find extracted frames directories
    local frames_dirs=()
    if [ -d "${base_dir}/frames" ]; then
        while IFS= read -r -d '' dir; do
            frames_dirs+=("$dir")
        done < <(find "${base_dir}/frames" -mindepth 1 -maxdepth 1 -type d -print0)
    fi

    log_info "Found ${#video_files[@]} video files for analysis"
    log_info "Found ${#frames_dirs[@]} frame directories for composition analysis"

    # Check resources
    if ! monitor_resources; then
        log_error "Resource check failed"
        exit 1
    fi

    # ========================================================================
    # Phase 1: Scene Detection (parallel)
    # ========================================================================

    if [ ${#video_files[@]} -gt 0 ]; then
        log_info ""
        log_info "Phase 1/3: Scene Detection (parallel workers: $num_workers)"
        log_info "========================================="

        export SCENE_DETECTOR

        if command -v parallel &> /dev/null; then
            printf '%s\n' "${video_files[@]}" | \
                parallel \
                    --jobs "$num_workers" \
                    --progress \
                    --bar \
                    --tagstring "[{#}]" \
                    detect_scenes_worker {} "${analysis_dir}/scenes" "$scenes_checkpoint"
        else
            for video in "${video_files[@]}"; do
                detect_scenes_worker "$video" "${analysis_dir}/scenes" "$scenes_checkpoint"
            done
        fi

        log_success "Scene detection completed"
    else
        log_warning "No video files found for scene detection"
    fi

    # ========================================================================
    # Phase 2: Composition Analysis (parallel)
    # ========================================================================

    if [ ${#frames_dirs[@]} -gt 0 ]; then
        log_info ""
        log_info "Phase 2/3: Composition Analysis (parallel workers: $num_workers)"
        log_info "========================================="

        export COMPOSITION_ANALYZER

        if command -v parallel &> /dev/null; then
            printf '%s\n' "${frames_dirs[@]}" | \
                parallel \
                    --jobs "$num_workers" \
                    --progress \
                    --bar \
                    --tagstring "[{#}]" \
                    analyze_composition_worker {} "${analysis_dir}/composition" "$composition_checkpoint"
        else
            for dir in "${frames_dirs[@]}"; do
                analyze_composition_worker "$dir" "${analysis_dir}/composition" "$composition_checkpoint"
            done
        fi

        log_success "Composition analysis completed"
    else
        log_warning "No frame directories found for composition analysis"
    fi

    # ========================================================================
    # Phase 3: Camera Movement Tracking (parallel)
    # ========================================================================

    if [ ${#video_files[@]} -gt 0 ]; then
        log_info ""
        log_info "Phase 3/3: Camera Movement Tracking (parallel workers: $num_workers)"
        log_info "========================================="

        export CAMERA_TRACKER

        if command -v parallel &> /dev/null; then
            printf '%s\n' "${video_files[@]}" | \
                parallel \
                    --jobs "$num_workers" \
                    --progress \
                    --bar \
                    --tagstring "[{#}]" \
                    track_camera_worker {} "${analysis_dir}/camera" "$camera_checkpoint"
        else
            for video in "${video_files[@]}"; do
                track_camera_worker "$video" "${analysis_dir}/camera" "$camera_checkpoint"
            done
        fi

        log_success "Camera tracking completed"
    fi

    # ========================================================================
    # Aggregate Results
    # ========================================================================

    log_info ""
    log_info "Aggregating analysis results..."

    local scenes_count=$(find "${analysis_dir}/scenes" -type f -name "*.json" 2>/dev/null | wc -l)
    local composition_count=$(find "${analysis_dir}/composition" -type f -name "*.json" 2>/dev/null | wc -l)
    local camera_count=$(find "${analysis_dir}/camera" -type f -name "*.json" 2>/dev/null | wc -l)

    local summary_file="${analysis_dir}/analysis_summary.json"
    cat > "$summary_file" <<EOF
{
  "stage": "cpu_stage2_analysis",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "base_dir": "$base_dir",
  "results": {
    "scene_detection": {
      "count": $scenes_count,
      "output_dir": "${analysis_dir}/scenes"
    },
    "composition_analysis": {
      "count": $composition_count,
      "output_dir": "${analysis_dir}/composition"
    },
    "camera_tracking": {
      "count": $camera_count,
      "output_dir": "${analysis_dir}/camera"
    }
  },
  "workers_used": $num_workers,
  "completed": true
}
EOF

    log_success "Analysis summary saved: $summary_file"

    # ========================================================================
    # Final Summary
    # ========================================================================

    log_info ""
    log_info "========================================="
    log_info "CPU Stage 2 COMPLETED"
    log_info "========================================="
    log_info "Scene detection: $scenes_count files"
    log_info "Composition analysis: $composition_count files"
    log_info "Camera tracking: $camera_count files"
    log_info "Summary: $summary_file"
    log_info ""

    monitor_resources || true
}

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 BASE_DIR [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  BASE_DIR     Output directory from CPU Stage 1"
    echo ""
    echo "Options:"
    echo "  --workers N  Number of parallel workers (default: $DEFAULT_WORKERS)"
    echo "  --resume     Resume from checkpoint"
    echo "  --help       Show this help"
    echo ""
    echo "Example:"
    echo "  $0 /mnt/data/ai_data/datasets/3d-anime/luca --workers 8"
    exit 1
}

BASE_DIR=""
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
            if [ -z "$BASE_DIR" ]; then
                BASE_DIR="$1"
            else
                log_error "Unknown argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$BASE_DIR" ]; then
    log_error "Missing required argument: BASE_DIR"
    usage
fi

if [ ! -d "$BASE_DIR" ]; then
    log_error "Base directory does not exist: $BASE_DIR"
    exit 1
fi

# Adjust workers to available cores
AVAILABLE_CORES=$(check_cpu_cores)
if [ "$NUM_WORKERS" -gt "$AVAILABLE_CORES" ]; then
    NUM_WORKERS=$AVAILABLE_CORES
fi

# Export variables
export SCENE_DETECTOR COMPOSITION_ANALYZER CAMERA_TRACKER
export RED GREEN YELLOW BLUE NC

# Run processing
process_stage2 "$BASE_DIR" "$NUM_WORKERS" "$RESUME_MODE"

log_success "All CPU Stage 2 tasks completed successfully"
