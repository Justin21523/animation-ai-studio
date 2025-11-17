#!/bin/bash
#
# Animation AI Studio - Master Startup Script
#
# Starts all required services for Animation AI Studio:
# - LLM Backend (vLLM + FastAPI Gateway)
# - Optional: Monitoring (Prometheus + Grafana)
#
# Usage:
#   bash start.sh                    # Start LLM backend only
#   bash start.sh --monitoring       # Start with monitoring
#   bash start.sh --model qwen-vl    # Start with specific model
#
# Author: Animation AI Studio
# Date: 2025-11-17
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🚀 ANIMATION AI STUDIO - STARTUP${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Parse arguments
START_MONITORING=false
MODEL_NAME="qwen-14b"

while [[ $# -gt 0 ]]; do
    case $1 in
        --monitoring|-m)
            START_MONITORING=true
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash start.sh [options]"
            echo ""
            echo "Options:"
            echo "  --monitoring, -m    Start with monitoring (Prometheus + Grafana)"
            echo "  --model NAME        Start with specific model (qwen-14b, qwen-vl, qwen-coder)"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ============================================================================
# Pre-flight checks
# ============================================================================

echo -e "\n${BLUE}Pre-flight checks...${NC}"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo -e "${YELLOW}   Activate with: source venv/bin/activate${NC}"
    echo -e "${YELLOW}   Or run: bash deploy/setup.sh${NC}"
    exit 1
fi

# Check Python version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${RED}❌ Python 3.10+ required${NC}"
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✅ ${GPU_COUNT} GPU(s) detected${NC}"
else
    echo -e "${YELLOW}⚠️  No CUDA detected - running in CPU mode${NC}"
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"

# ============================================================================
# Start LLM Backend
# ============================================================================

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${BLUE}Starting LLM Backend...${NC}"
echo -e "${BLUE}======================================================================${NC}"

if [[ -f "llm_backend/scripts/start.sh" ]]; then
    echo -e "Model: ${YELLOW}$MODEL_NAME${NC}"

    cd llm_backend
    bash scripts/start.sh "$MODEL_NAME"
    cd "$PROJECT_ROOT"

    echo -e "${GREEN}✅ LLM Backend started${NC}"
else
    echo -e "${RED}❌ LLM Backend scripts not found${NC}"
    exit 1
fi

# ============================================================================
# Start Monitoring (optional)
# ============================================================================

if [[ "$START_MONITORING" == true ]]; then
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}Starting Monitoring...${NC}"
    echo -e "${BLUE}======================================================================${NC}"

    if [[ -f "llm_backend/monitoring/start_monitoring.sh" ]]; then
        cd llm_backend/monitoring
        bash start_monitoring.sh
        cd "$PROJECT_ROOT"

        echo -e "${GREEN}✅ Monitoring started${NC}"
        echo -e "   Prometheus: http://localhost:9090"
        echo -e "   Grafana: http://localhost:3000 (admin/admin)"
    else
        echo -e "${YELLOW}⚠️  Monitoring scripts not found${NC}"
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}🎉 ANIMATION AI STUDIO - READY!${NC}"
echo -e "${BLUE}======================================================================${NC}"

echo -e "\n${YELLOW}Services:${NC}"
echo -e "  ✅ LLM Backend: http://localhost:8000"
echo -e "  ✅ OpenAI API: http://localhost:8000/v1"

if [[ "$START_MONITORING" == true ]]; then
    echo -e "  ✅ Prometheus: http://localhost:9090"
    echo -e "  ✅ Grafana: http://localhost:3000"
fi

echo -e "\n${YELLOW}Quick Commands:${NC}"
echo -e "  Check status:  ${BLUE}bash llm_backend/scripts/health.sh${NC}"
echo -e "  View logs:     ${BLUE}bash llm_backend/scripts/logs.sh${NC}"
echo -e "  Stop services: ${BLUE}bash stop.sh${NC}"
echo -e "  Run tests:     ${BLUE}python tests/run_all_tests.py${NC}"

echo -e "\n${YELLOW}Try Creative Studio:${NC}"
echo -e "  List features: ${BLUE}python scripts/applications/creative_studio/cli.py list${NC}"
echo -e "  Parody video:  ${BLUE}python scripts/applications/creative_studio/cli.py parody input.mp4 output.mp4${NC}"

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}✨ Ready to create!${NC}"
echo -e "${BLUE}======================================================================${NC}\n"
