#!/bin/bash
#
# Switch between models (RTX 5080 16GB can only run one at a time)
#

set -e

echo "========================================="
echo "🔄 Switch LLM Model"
echo "   RTX 5080 16GB: One model at a time"
echo "========================================="
echo ""

# Check if running in project root
if [ ! -f "llm_backend/docker/docker-compose.yml" ]; then
    echo "❌ Error: Must run from project root"
    exit 1
fi

cd llm_backend/docker

# Show current models
echo "Current running models:"
docker-compose ps --filter "status=running" | grep "llm-qwen" || echo "  None"
echo ""

# Model selection
echo "Available models:"
echo "  1) Qwen2.5-VL-7B (Multimodal - vision + chat)"
echo "  2) Qwen2.5-14B (Reasoning - complex tasks)"
echo "  3) Qwen2.5-Coder-7B (Code generation)"
echo "  4) Stop all models"
echo ""
read -p "Select model [1-4]: " choice

# Stop all current models
echo ""
echo "🛑 Stopping current models..."
docker-compose stop qwen-vl qwen-14b qwen-coder 2>/dev/null || true

case $choice in
    1)
        echo "🤖 Starting Qwen2.5-VL-7B..."
        docker-compose up -d qwen-vl
        MODEL_NAME="qwen-vl"
        MODEL_PORT=8000
        ;;
    2)
        echo "🤖 Starting Qwen2.5-14B..."
        docker-compose --profile manual up -d qwen-14b
        MODEL_NAME="qwen-14b"
        MODEL_PORT=8001
        ;;
    3)
        echo "🤖 Starting Qwen2.5-Coder-7B..."
        docker-compose --profile manual up -d qwen-coder
        MODEL_NAME="qwen-coder"
        MODEL_PORT=8002
        ;;
    4)
        echo "✅ All models stopped"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Wait for model
echo ""
echo "⏳ Loading model (1-3 minutes)..."

WAIT_TIME=0
MAX_WAIT=300

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Model ready!"
        break
    fi
    echo -n "."
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo ""
    echo "❌ Model failed to start"
    echo "   Check logs: bash llm_backend/scripts/logs.sh $MODEL_NAME"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ Model switched successfully"
echo "========================================="
echo ""
echo "Active model: $MODEL_NAME"
echo "Port: $MODEL_PORT"
echo "URL: http://localhost:$MODEL_PORT"
echo ""
echo "Test: curl http://localhost:7000/health"
