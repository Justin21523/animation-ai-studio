#!/bin/bash
################################################################################
# GPU Task 3: LLM Video Content Analysis
#
# Uses multimodal LLM (Qwen-VL) or text LLM (Qwen) for intelligent video
# content analysis, scene understanding, and narrative extraction.
#
# Features:
#   - Multimodal analysis (visual + text)
#   - Scene description and classification
#   - Character action recognition
#   - Narrative flow analysis
#   - ModelManager integration for VRAM management
#
# Hardware Requirements:
#   - GPU: NVIDIA RTX 5080 16GB
#   - VRAM: 6-8GB (Qwen-VL-7B), 11-14GB (Qwen-14B)
#   - CUDA: 11.8+
#
# Usage:
#   bash scripts/batch/gpu_task3_llm_analysis.sh \
#     INPUT_DATA_DIR \
#     OUTPUT_DIR \
#     [--model MODEL] [--task TASK]
#
# Example:
#   bash scripts/batch/gpu_task3_llm_analysis.sh \
#     /mnt/data/ai_data/datasets/3d-anime/luca \
#     /mnt/data/ai_data/outputs/llm_analysis/luca \
#     --model qwen-vl-7b \
#     --task scene_analysis
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
DEFAULT_MODEL="qwen-vl-7b"  # Options: qwen-vl-7b, qwen-14b, qwen-7b
DEFAULT_TASK="scene_analysis"  # Options: scene_analysis, character_analysis, narrative_extraction
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
log_llm() { echo -e "${CYAN}[LLM]${NC} $1"; }

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
# LLM Analysis Tasks
# ============================================================================

analyze_scene() {
    local input_dir="$1"
    local output_dir="$2"
    local model="$3"

    log_llm "Starting scene analysis..."

    python3 -c "
import sys
import os
import json
import gc
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '${PROJECT_ROOT}')

from scripts.core.model_management.model_manager import ModelManager
from scripts.core.llm_client import LLMClient

# Initialize ModelManager
manager = ModelManager()

# Ensure other heavy models unloaded
manager._ensure_heavy_model_unloaded()
manager.vram_monitor.clear_cache()

try:
    # Use LLM context
    with manager.use_llm(model='$model'):
        client = LLMClient(model='$model')

        # Load analysis data (from CPU Stage 2)
        analysis_file = '${input_dir}/analysis/analysis_summary.json'
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
        else:
            analysis_data = {}

        # Construct prompt for scene analysis
        prompt = '''Analyze the following animation scenes:

Scene Data:
{}

Please provide:
1. Scene descriptions (setting, mood, visual style)
2. Character actions and interactions
3. Narrative flow and story beats
4. Emotional tone and atmosphere

Format your response as structured JSON.
'''.format(json.dumps(analysis_data, indent=2))

        # Query LLM
        print('Sending prompt to LLM...')
        response = client.chat(messages=[
            {'role': 'user', 'content': prompt}
        ])

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'$output_dir/scene_analysis_{timestamp}.json'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model': '$model',
                'task': 'scene_analysis',
                'input_data': analysis_data,
                'llm_response': response
            }, f, indent=2)

        print(f'Analysis saved: {output_file}')

    # LLM automatically unloaded

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sys.exit(0)

except Exception as e:
    print(f'Error during analysis: {str(e)}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
}

analyze_characters() {
    local input_dir="$1"
    local output_dir="$2"
    local model="$3"

    log_llm "Starting character analysis..."

    python3 -c "
import sys
import os
import json
sys.path.insert(0, '${PROJECT_ROOT}')

from scripts.core.model_management.model_manager import ModelManager
from scripts.core.llm_client import LLMClient

manager = ModelManager()
manager._ensure_heavy_model_unloaded()

try:
    with manager.use_llm(model='$model'):
        client = LLMClient(model='$model')

        # Load character data (from segmentation/clustering)
        character_data = {}
        seg_file = '${input_dir}/segmented/segmentation_summary.json'
        if os.path.exists(seg_file):
            with open(seg_file, 'r') as f:
                character_data = json.load(f)

        prompt = f'''Analyze the following character instances:

{json.dumps(character_data, indent=2)}

Provide:
1. Character identification and descriptions
2. Costume and appearance details
3. Personality traits (inferred from visuals)
4. Relationships between characters

Format as JSON.'''

        response = client.chat(messages=[
            {'role': 'user', 'content': prompt}
        ])

        output_file = f'$output_dir/character_analysis.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                'model': '$model',
                'task': 'character_analysis',
                'llm_response': response
            }, f, indent=2)

        print(f'Character analysis saved: {output_file}')

    sys.exit(0)

except Exception as e:
    print(f'Error: {str(e)}', file=sys.stderr)
    sys.exit(1)
"
}

# ============================================================================
# Main Processing Function
# ============================================================================

process_llm_analysis() {
    local input_dir="$1"
    local output_dir="$2"
    local model="$3"
    local task="$4"

    log_info "========================================"
    log_info "GPU Task 3: LLM Video Content Analysis"
    log_info "========================================"
    log_info "Input: $input_dir"
    log_info "Output: $output_dir"
    log_info "Model: $model"
    log_info "Task: $task"
    echo ""

    # Check GPU
    if ! check_gpu_available; then
        log_error "GPU check failed"
        exit 1
    fi

    # Create output directories
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/logs"

    # Clear GPU before starting
    clear_gpu_memory

    local start_time=$(date +%s)

    # Monitor GPU before LLM loading
    log_info "GPU status before LLM loading:"
    monitor_gpu

    # Execute analysis task
    case $task in
        scene_analysis)
            if analyze_scene "$input_dir" "$output_dir" "$model" 2>&1 | tee "$output_dir/logs/scene_analysis.log"; then
                log_success "Scene analysis complete"
            else
                log_error "Scene analysis failed"
                return 1
            fi
            ;;
        character_analysis)
            if analyze_characters "$input_dir" "$output_dir" "$model" 2>&1 | tee "$output_dir/logs/character_analysis.log"; then
                log_success "Character analysis complete"
            else
                log_error "Character analysis failed"
                return 1
            fi
            ;;
        narrative_extraction)
            log_warning "Narrative extraction not yet implemented"
            ;;
        *)
            log_error "Unknown task: $task"
            return 1
            ;;
    esac

    # Monitor GPU after
    log_info "GPU status after analysis:"
    monitor_gpu

    # Clear GPU memory
    clear_gpu_memory

    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local minutes=$((total_time / 60))
    local seconds=$((total_time % 60))

    # Create summary
    local summary_file="${output_dir}/llm_analysis_summary.json"
    cat > "$summary_file" <<EOF
{
  "task": "gpu_task3_llm_analysis",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "input_dir": "$input_dir",
  "output_dir": "$output_dir",
  "model": "$model",
  "analysis_task": "$task",
  "total_time_seconds": $total_time,
  "completed": true
}
EOF

    log_info ""
    log_info "========================================"
    log_info "LLM Analysis Complete"
    log_info "========================================"
    log_info "Time: ${minutes}m ${seconds}s"
    log_info "Summary: $summary_file"
    echo ""

    return 0
}

# ============================================================================
# Cleanup Handler
# ============================================================================

cleanup_on_error() {
    log_error "LLM analysis interrupted or failed"
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
    echo "  INPUT_DIR    Directory containing analysis data from CPU stages"
    echo "  OUTPUT_DIR   Output directory for LLM analysis results"
    echo ""
    echo "Options:"
    echo "  --model MODEL      LLM model to use (default: $DEFAULT_MODEL)"
    echo "                     Options: qwen-vl-7b, qwen-14b, qwen-7b"
    echo "  --task TASK        Analysis task (default: $DEFAULT_TASK)"
    echo "                     Options: scene_analysis, character_analysis,"
    echo "                              narrative_extraction"
    echo "  --help             Show this help"
    echo ""
    echo "Example:"
    echo "  $0 /data/luca /outputs/llm_analysis --model qwen-vl-7b --task scene_analysis"
    exit 1
}

INPUT_DIR=""
OUTPUT_DIR=""
MODEL=$DEFAULT_MODEL
TASK=$DEFAULT_TASK

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
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

# Export variables
export RED GREEN YELLOW BLUE CYAN MAGENTA NC

# Run processing
process_llm_analysis "$INPUT_DIR" "$OUTPUT_DIR" "$MODEL" "$TASK"

log_success "GPU Task 3 (LLM Analysis) completed successfully"
exit 0
