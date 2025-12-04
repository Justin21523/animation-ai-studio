#!/bin/bash
################################################################################
# Smoke Test Script - Batch Automation Pipeline
#
# Verifies that all batch scripts are working correctly without running
# full processing pipelines.
#
# Tests:
#   1. Script executability
#   2. Bash syntax validation
#   3. Help/usage display
#   4. Python imports
#   5. Basic functionality checks
#
# Usage:
#   bash scripts/batch/smoke_test.sh
#
# Author: Animation AI Studio
# Date: 2025-12-04
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_test() { echo -e "${BOLD}[TEST]${NC} $1"; }

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# ============================================================================
# Test Functions
# ============================================================================

test_script_exists() {
    local script="$1"
    local name="$2"

    log_test "Checking if $name exists..."

    if [ -f "$script" ]; then
        log_success "$name found"
        return 0
    else
        log_error "$name not found: $script"
        return 1
    fi
}

test_script_executable() {
    local script="$1"
    local name="$2"

    log_test "Checking if $name is executable..."

    if [ -x "$script" ]; then
        log_success "$name is executable"
        return 0
    else
        log_error "$name is not executable: $script"
        return 1
    fi
}

test_bash_syntax() {
    local script="$1"
    local name="$2"

    log_test "Validating bash syntax for $name..."

    if bash -n "$script" 2>/dev/null; then
        log_success "$name has valid bash syntax"
        return 0
    else
        log_error "$name has syntax errors"
        bash -n "$script"
        return 1
    fi
}

test_help_output() {
    local script="$1"
    local name="$2"

    log_test "Testing help output for $name..."

    if bash "$script" --help &>/dev/null || bash "$script" 2>&1 | grep -q "Usage:"; then
        log_success "$name displays help/usage"
        return 0
    else
        log_warning "$name does not display help (may be normal)"
        return 0  # Non-critical
    fi
}

# ============================================================================
# Batch Script Tests
# ============================================================================

test_batch_scripts() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "  Testing Batch Scripts"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""

    local scripts=(
        "cpu_task1_data_prep.sh"
        "cpu_task2_analysis.sh"
        "cpu_task3_rag_prep.sh"
        "run_cpu_tasks_all.sh"
        "gpu_task1_segmentation.sh"
        "gpu_task2_image_generation.sh"
        "gpu_task3_llm_analysis.sh"
        "gpu_task4_voice_training.sh"
        "run_gpu_tasks_all.sh"
        "run_all_tasks_complete.sh"
    )

    for script_name in "${scripts[@]}"; do
        local script_path="${SCRIPT_DIR}/${script_name}"

        echo ""
        log_info "Testing: $script_name"
        echo "──────────────────────────────────────────────────────────"

        local script_passed=true

        # Test 1: Exists
        if ! test_script_exists "$script_path" "$script_name"; then
            ((TESTS_FAILED++))
            script_passed=false
        else
            ((TESTS_PASSED++))

            # Test 2: Executable
            if ! test_script_executable "$script_path" "$script_name"; then
                ((TESTS_FAILED++))
                script_passed=false
            else
                ((TESTS_PASSED++))
            fi

            # Test 3: Syntax
            if ! test_bash_syntax "$script_path" "$script_name"; then
                ((TESTS_FAILED++))
                script_passed=false
            else
                ((TESTS_PASSED++))
            fi

            # Test 4: Help (non-critical)
            if test_help_output "$script_path" "$script_name"; then
                ((TESTS_PASSED++))
            fi
        fi

        if [ "$script_passed" = true ]; then
            log_success "All tests passed for $script_name"
        else
            log_error "Some tests failed for $script_name"
        fi
    done
}

# ============================================================================
# Python Import Tests
# ============================================================================

test_python_imports() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "  Testing Python Imports"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""

    local python_modules=(
        "torch"
        "torchvision"
        "numpy"
        "PIL"
        "cv2"
        "tqdm"
    )

    for module in "${python_modules[@]}"; do
        log_test "Importing $module..."

        if python3 -c "import $module" 2>/dev/null; then
            log_success "$module imported successfully"
            ((TESTS_PASSED++))
        else
            log_error "Failed to import $module"
            ((TESTS_FAILED++))
        fi
    done
}

# ============================================================================
# CUDA Tests
# ============================================================================

test_cuda() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "  Testing CUDA"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""

    # Test nvidia-smi
    log_test "Checking nvidia-smi..."
    if command -v nvidia-smi &> /dev/null; then
        log_success "nvidia-smi found"
        ((TESTS_PASSED++))

        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
        log_info "GPU: $gpu_name"
    else
        log_warning "nvidia-smi not found (GPU tests skipped)"
        ((TESTS_SKIPPED++))
        return 0
    fi

    # Test PyTorch CUDA
    log_test "Checking PyTorch CUDA availability..."
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
        log_success "CUDA available in PyTorch (version: $cuda_version)"
        ((TESTS_PASSED++))
    else
        log_error "CUDA not available in PyTorch"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# Dependency Tests
# ============================================================================

test_dependencies() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "  Testing Dependencies"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""

    local required_commands=(
        "python3"
        "ffmpeg"
        "parallel"
        "jq"
    )

    for cmd in "${required_commands[@]}"; do
        log_test "Checking $cmd..."

        if command -v "$cmd" &> /dev/null; then
            local version=""
            case $cmd in
                python3)
                    version=$(python3 --version 2>&1 | awk '{print $2}')
                    ;;
                ffmpeg)
                    version=$(ffmpeg -version 2>&1 | head -n1 | awk '{print $3}')
                    ;;
                parallel)
                    version=$(parallel --version 2>&1 | head -n1 | awk '{print $3}')
                    ;;
                jq)
                    version=$(jq --version 2>&1 | awk -F'-' '{print $2}')
                    ;;
            esac

            log_success "$cmd found (version: $version)"
            ((TESTS_PASSED++))
        else
            log_error "$cmd not found"
            ((TESTS_FAILED++))
        fi
    done
}

# ============================================================================
# Directory Structure Tests
# ============================================================================

test_directory_structure() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "  Testing Directory Structure"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""

    local required_dirs=(
        "${PROJECT_ROOT}/scripts/batch"
        "${PROJECT_ROOT}/scripts/processing"
        "${PROJECT_ROOT}/scripts/core"
        "${PROJECT_ROOT}/configs"
        "${PROJECT_ROOT}/docs"
    )

    for dir in "${required_dirs[@]}"; do
        log_test "Checking directory: $(basename $dir)..."

        if [ -d "$dir" ]; then
            log_success "$(basename $dir) exists"
            ((TESTS_PASSED++))
        else
            log_error "$(basename $dir) not found: $dir"
            ((TESTS_FAILED++))
        fi
    done
}

# ============================================================================
# Main Test Execution
# ============================================================================

main() {
    echo ""
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║                                                                ║${NC}"
    echo -e "${BOLD}${BLUE}║         Animation AI Studio - Smoke Test Suite                ║${NC}"
    echo -e "${BOLD}${BLUE}║                                                                ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    log_info "Starting smoke tests..."
    log_info "Project root: $PROJECT_ROOT"
    echo ""

    # Run all test suites
    test_directory_structure
    test_dependencies
    test_python_imports
    test_cuda
    test_batch_scripts

    # Summary
    echo ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "  Test Summary"
    log_info "════════════════════════════════════════════════════════════"
    echo ""

    local total_tests=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

    echo -e "${BOLD}Total Tests:${NC}   $total_tests"
    echo -e "${GREEN}${BOLD}Passed:${NC}        $TESTS_PASSED"
    echo -e "${RED}${BOLD}Failed:${NC}        $TESTS_FAILED"
    echo -e "${YELLOW}${BOLD}Skipped:${NC}       $TESTS_SKIPPED"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ All tests passed!${NC}"
        echo ""
        log_success "Batch automation pipeline is ready to use"
        return 0
    else
        echo -e "${RED}${BOLD}✗ Some tests failed${NC}"
        echo ""
        log_error "Please fix the failed tests before running the pipeline"
        return 1
    fi
}

# Run main
main

exit $?
