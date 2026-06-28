#!/bin/bash
# Start Animation AI Studio Web UI Backend

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Conda environment
CONDA_ENV="ai_env"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting Animation AI Studio Web UI ===${NC}"
echo ""

# Check conda environment
if [ ! -d "$HOME/miniconda3/envs/$CONDA_ENV" ]; then
    echo -e "${RED}Error: Conda environment '$CONDA_ENV' not found${NC}"
    exit 1
fi

# Activate conda environment
echo -e "${YELLOW}Activating conda environment: $CONDA_ENV${NC}"
export PATH="$HOME/miniconda3/envs/$CONDA_ENV/bin:$PATH"

# Check Python
PYTHON_BIN="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${RED}Error: Python not found at $PYTHON_BIN${NC}"
    exit 1
fi

echo "Python: $($PYTHON_BIN --version)"
echo ""

# Check database directory
DB_PATH="/mnt/data/training/runs/animation-ai-studio/workflows.db"
DB_DIR="$(dirname "$DB_PATH")"

if [ ! -d "$DB_DIR" ]; then
    echo -e "${YELLOW}Creating database directory: $DB_DIR${NC}"
    mkdir -p "$DB_DIR"
fi

# Check config file
CONFIG_FILE="$PROJECT_ROOT/configs/web_ui/backend.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Project root: $PROJECT_ROOT"
echo "  Database: $DB_PATH"
echo "  Config: $CONFIG_FILE"
echo ""

# Start server
echo -e "${GREEN}Starting FastAPI server...${NC}"
cd "$SCRIPT_DIR/backend"

exec "$PYTHON_BIN" main.py
