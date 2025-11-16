#!/bin/bash
#
# Start all LLM Backend services
# For RTX 5080 16GB - Can only run ONE model at a time
#

set -e

echo "========================================="
echo "🚀 Starting LLM Backend Services"
echo "   Hardware: RTX 5080 16GB VRAM"
echo "========================================="
echo ""

# Check if running in project root
if [ ! -f "llm_backend/docker/docker-compose.yml" ]; then
    echo "❌ Error: Must run from project root"
    echo "   cd /mnt/c/AI_LLM_projects/animation-ai-studio"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found"
    exit 1
fi

# Check nvidia-docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-docker not working"
    echo "   Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✅ Docker and nvidia-docker ready"
echo ""

# Navigate to docker directory
cd llm_backend/docker

# Start infrastructure services
echo "📦 Starting infrastructure services..."
docker-compose up -d redis prometheus grafana
sleep 5

# Check Redis
if ! docker exec llm-redis redis-cli ping &> /dev/null; then
    echo "❌ Redis failed to start"
    exit 1
fi
echo "✅ Redis ready"

# Start Gateway
echo "🌐 Starting API Gateway..."
docker-compose up -d gateway
sleep 5

# Check Gateway
if ! curl -s http://localhost:7000/health > /dev/null; then
    echo "❌ Gateway failed to start"
    echo "   Check logs: docker-compose logs gateway"
    exit 1
fi
echo "✅ Gateway ready"

# Ask user which model to start
echo ""
echo "========================================="
echo "⚠️  IMPORTANT: RTX 5080 16GB can only run ONE model at a time"
echo "========================================="
echo ""
echo "Choose which model to start:"
echo "  1) Qwen2.5-VL-7B (Multimodal - vision + chat)"
echo "  2) Qwen2.5-14B (Reasoning - complex tasks)"
echo "  3) Qwen2.5-Coder-7B (Code generation)"
echo "  4) None (infrastructure only)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "🤖 Starting Qwen2.5-VL-7B (Multimodal)..."
        docker-compose up -d qwen-vl
        MODEL_NAME="qwen-vl"
        MODEL_PORT=8000
        ;;
    2)
        echo "🤖 Starting Qwen2.5-14B (Reasoning)..."
        docker-compose --profile manual up -d qwen-14b
        MODEL_NAME="qwen-14b"
        MODEL_PORT=8001
        ;;
    3)
        echo "🤖 Starting Qwen2.5-Coder-7B (Code)..."
        docker-compose --profile manual up -d qwen-coder
        MODEL_NAME="qwen-coder"
        MODEL_PORT=8002
        ;;
    4)
        echo "✅ Infrastructure services started (no model loaded)"
        echo ""
        echo "Service URLs:"
        echo "  Gateway:    http://localhost:7000"
        echo "  Redis:      localhost:6379"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana:    http://localhost:3000 (admin/admin)"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Wait for model to load
echo ""
echo "⏳ Waiting for model to load (this may take 1-3 minutes)..."
echo "   Model loading progress..."

WAIT_TIME=0
MAX_WAIT=300  # 5 minutes max

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Model loaded successfully!"
        break
    fi

    # Show dots for progress
    echo -n "."
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo ""
    echo "❌ Model failed to start within timeout"
    echo "   Check logs: docker-compose logs $MODEL_NAME"
    exit 1
fi

# Final health check
echo ""
echo "========================================="
echo "✅ All services started successfully!"
echo "========================================="
echo ""

# Show status
echo "Service Status:"
curl -s http://localhost:7000/health | python3 -m json.tool 2>/dev/null || echo "  (Gateway health check pending)"

echo ""
echo "Service URLs:"
echo "  Gateway:    http://localhost:7000"
echo "  Model:      http://localhost:$MODEL_PORT"
echo "  Redis:      localhost:6379"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "Quick Tests:"
echo "  Health: curl http://localhost:7000/health"
echo "  Models: curl http://localhost:7000/models"
echo ""
echo "Logs:"
echo "  Gateway: docker-compose logs -f gateway"
echo "  Model:   docker-compose logs -f $MODEL_NAME"
echo ""
echo "🎉 LLM Backend ready for use!"
