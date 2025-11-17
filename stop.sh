#!/bin/bash
#
# Animation AI Studio - Master Shutdown Script
#
# Stops all running services gracefully
#
# Usage:
#   bash stop.sh
#   bash stop.sh --force  # Force kill all processes
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
echo -e "${BLUE}🛑 ANIMATION AI STUDIO - SHUTDOWN${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Parse arguments
FORCE_KILL=false
if [[ "$1" == "--force" ]]; then
    FORCE_KILL=true
    echo -e "${YELLOW}⚠️  Force mode enabled${NC}"
fi

# ============================================================================
# Stop LLM Backend
# ============================================================================

echo -e "\n${BLUE}Stopping LLM Backend...${NC}"

if [[ -f "llm_backend/scripts/stop.sh" ]]; then
    cd llm_backend
    bash scripts/stop.sh
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✅ LLM Backend stopped${NC}"
else
    echo -e "${YELLOW}⚠️  LLM Backend stop script not found${NC}"
fi

# ============================================================================
# Stop Monitoring
# ============================================================================

echo -e "\n${BLUE}Stopping Monitoring...${NC}"

if [[ -f "llm_backend/monitoring/stop_monitoring.sh" ]]; then
    cd llm_backend/monitoring
    bash stop_monitoring.sh
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✅ Monitoring stopped${NC}"
else
    echo -e "${YELLOW}⚠️  Monitoring not running${NC}"
fi

# ============================================================================
# Force kill if requested
# ============================================================================

if [[ "$FORCE_KILL" == true ]]; then
    echo -e "\n${YELLOW}Force killing remaining processes...${NC}"

    # Kill vllm processes
    pkill -f "vllm.entrypoints" 2>/dev/null || true

    # Kill FastAPI processes
    pkill -f "uvicorn.*llm_backend" 2>/dev/null || true

    # Kill monitoring
    pkill -f "prometheus" 2>/dev/null || true
    pkill -f "grafana" 2>/dev/null || true

    echo -e "${GREEN}✅ Force kill complete${NC}"
fi

# ============================================================================
# Verify shutdown
# ============================================================================

echo -e "\n${BLUE}Verifying shutdown...${NC}"

STILL_RUNNING=false

if pgrep -f "vllm.entrypoints" > /dev/null; then
    echo -e "${YELLOW}⚠️  vLLM still running${NC}"
    STILL_RUNNING=true
fi

if pgrep -f "uvicorn.*llm_backend" > /dev/null; then
    echo -e "${YELLOW}⚠️  FastAPI Gateway still running${NC}"
    STILL_RUNNING=true
fi

if [[ "$STILL_RUNNING" == true ]]; then
    echo -e "${YELLOW}⚠️  Some processes still running${NC}"
    echo -e "${YELLOW}   Use: bash stop.sh --force${NC}"
else
    echo -e "${GREEN}✅ All processes stopped${NC}"
fi

# ============================================================================
# Summary
# ============================================================================

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}✅ SHUTDOWN COMPLETE${NC}"
echo -e "${BLUE}======================================================================${NC}\n"
