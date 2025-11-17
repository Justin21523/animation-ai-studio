#!/bin/bash
#
# Animation AI Studio - Setup Script
#
# Complete setup for Animation AI Studio including:
# - Environment setup
# - Dependencies installation
# - Model downloads
# - Configuration
# - Health checks
#
# Usage:
#   bash deploy/setup.sh
#   bash deploy/setup.sh --quick  # Skip model downloads
#
# Author: Animation AI Studio
# Date: 2025-11-17
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🎬 ANIMATION AI STUDIO - SETUP${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Project Root: $PROJECT_ROOT"
echo -e "${BLUE}======================================================================${NC}"

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}⚡ Quick mode enabled (skipping model downloads)${NC}"
fi

# ============================================================================
# Step 1: Check Python version
# ============================================================================

echo -e "\n${BLUE}Step 1/8: Checking Python version...${NC}"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) and sys.version_info < (3, 12) else 1)"; then
    echo -e "${GREEN}✅ Python $PYTHON_VERSION (Compatible)${NC}"
else
    echo -e "${RED}❌ Python $PYTHON_VERSION detected${NC}"
    echo -e "${RED}   Required: Python >= 3.10, < 3.12${NC}"
    exit 1
fi

# ============================================================================
# Step 2: Check CUDA availability
# ============================================================================

echo -e "\n${BLUE}Step 2/8: Checking CUDA availability...${NC}"

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    echo -e "${GREEN}✅ CUDA $CUDA_VERSION detected${NC}"
    echo -e "   GPU: $GPU_NAME"
    echo -e "   VRAM: ${GPU_MEMORY}MB"

    if (( GPU_MEMORY < 16000 )); then
        echo -e "${YELLOW}⚠️  Warning: Less than 16GB VRAM${NC}"
        echo -e "${YELLOW}   Some models may require adjustment${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  CUDA not detected - CPU mode only${NC}"
    echo -e "${YELLOW}   Performance will be significantly slower${NC}"
fi

# ============================================================================
# Step 3: Create virtual environment (if needed)
# ============================================================================

echo -e "\n${BLUE}Step 3/8: Setting up virtual environment...${NC}"

if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ ! -d "venv" ]]; then
        echo -e "Creating virtual environment..."
        python3 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✅ Virtual environment exists${NC}"
    fi

    echo -e "${YELLOW}⚠️  Activate with: source venv/bin/activate${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already active${NC}"
fi

# ============================================================================
# Step 4: Upgrade pip
# ============================================================================

echo -e "\n${BLUE}Step 4/8: Upgrading pip...${NC}"

python3 -m pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✅ pip upgraded${NC}"

# ============================================================================
# Step 5: Install dependencies
# ============================================================================

echo -e "\n${BLUE}Step 5/8: Installing dependencies...${NC}"

echo -e "Installing core requirements..."
python3 -m pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# ============================================================================
# Step 6: Create directories
# ============================================================================

echo -e "\n${BLUE}Step 6/8: Creating directory structure...${NC}"

mkdir -p outputs/{llm_backend,image_generation,voice_synthesis,video_analysis,video_editing,creative_studio}
mkdir -p outputs/tests/{agent,editing,creative_studio}
mkdir -p logs
mkdir -p data/cache
mkdir -p data/temp

echo -e "${GREEN}✅ Directories created${NC}"

# ============================================================================
# Step 7: Download models (unless quick mode)
# ============================================================================

if [[ "$QUICK_MODE" == false ]]; then
    echo -e "\n${BLUE}Step 7/8: Checking models...${NC}"

    AI_WAREHOUSE="/mnt/c/AI_LLM_projects/ai_warehouse"

    if [[ -d "$AI_WAREHOUSE" ]]; then
        echo -e "${GREEN}✅ AI Warehouse found at: $AI_WAREHOUSE${NC}"

        # Check critical models
        MODELS_OK=true

        if [[ ! -d "$AI_WAREHOUSE/models/llm/Qwen" ]]; then
            echo -e "${YELLOW}⚠️  LLM models not found${NC}"
            MODELS_OK=false
        fi

        if [[ ! -d "$AI_WAREHOUSE/models/segmentation/sam2" ]]; then
            echo -e "${YELLOW}⚠️  SAM2 models not found${NC}"
            MODELS_OK=false
        fi

        if [[ "$MODELS_OK" == true ]]; then
            echo -e "${GREEN}✅ Critical models present${NC}"
        else
            echo -e "${YELLOW}⚠️  Some models missing${NC}"
            echo -e "${YELLOW}   Download with: python scripts/core/utils/download_models.py${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  AI Warehouse not found${NC}"
        echo -e "${YELLOW}   Expected at: $AI_WAREHOUSE${NC}"
        echo -e "${YELLOW}   Models will be downloaded on first use${NC}"
    fi
else
    echo -e "\n${BLUE}Step 7/8: Skipping model check (quick mode)${NC}"
fi

# ============================================================================
# Step 8: Health checks
# ============================================================================

echo -e "\n${BLUE}Step 8/8: Running health checks...${NC}"

# Check if key modules can be imported
python3 -c "
import sys
errors = []

try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
except Exception as e:
    errors.append(f'❌ PyTorch: {e}')

try:
    import numpy
    print(f'✅ NumPy {numpy.__version__}')
except Exception as e:
    errors.append(f'❌ NumPy: {e}')

try:
    import cv2
    print(f'✅ OpenCV {cv2.__version__}')
except Exception as e:
    errors.append(f'❌ OpenCV: {e}')

try:
    import fastapi
    print(f'✅ FastAPI {fastapi.__version__}')
except Exception as e:
    errors.append(f'❌ FastAPI: {e}')

if errors:
    print('\n'.join(errors))
    sys.exit(1)
" || {
    echo -e "${RED}❌ Import checks failed${NC}"
    exit 1
}

echo -e "${GREEN}✅ Health checks passed${NC}"

# ============================================================================
# Summary
# ============================================================================

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}🎉 SETUP COMPLETE!${NC}"
echo -e "${BLUE}======================================================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Activate virtual environment (if not already):"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo -e ""
echo -e "2. Start LLM Backend:"
echo -e "   ${BLUE}bash llm_backend/scripts/start.sh${NC}"
echo -e ""
echo -e "3. Run tests:"
echo -e "   ${BLUE}python tests/run_all_tests.py${NC}"
echo -e ""
echo -e "4. Try Creative Studio CLI:"
echo -e "   ${BLUE}python scripts/applications/creative_studio/cli.py list${NC}"
echo -e ""
echo -e "5. Generate parody video:"
echo -e "   ${BLUE}python scripts/applications/creative_studio/cli.py parody input.mp4 output.mp4 --style dramatic${NC}"

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}✨ Animation AI Studio is ready!${NC}"
echo -e "${BLUE}======================================================================${NC}\n"
