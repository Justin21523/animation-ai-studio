#!/bin/bash
#
# Health check for all LLM Backend services
#

echo "========================================="
echo "🔍 LLM Backend Health Check"
echo "========================================="
echo ""

# Function to check HTTP service
check_http() {
    local name=$1
    local url=$2

    if curl -sf "$url" > /dev/null 2>&1; then
        echo "✅ $name: healthy"
        return 0
    else
        echo "❌ $name: down"
        return 1
    fi
}

# Function to check Docker container
check_container() {
    local name=$1

    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "✅ $name: running"
        return 0
    else
        echo "❌ $name: not running"
        return 1
    fi
}

# Check Docker containers
echo "Docker Containers:"
check_container "llm-redis"
check_container "llm-gateway"
check_container "llm-qwen-vl" || echo "   (Optional - may not be started)"
check_container "llm-qwen-14b" || echo "   (Optional - may not be started)"
check_container "llm-qwen-coder" || echo "   (Optional - may not be started)"
check_container "llm-prometheus"
check_container "llm-grafana"

echo ""
echo "HTTP Services:"
check_http "Gateway" "http://localhost:7000/health"
check_http "Qwen-VL (8000)" "http://localhost:8000/health" || echo "   (May not be started)"
check_http "Qwen-14B (8001)" "http://localhost:8001/health" || echo "   (May not be started)"
check_http "Qwen-Coder (8002)" "http://localhost:8002/health" || echo "   (May not be started)"
check_http "Prometheus" "http://localhost:9090/-/healthy"
check_http "Grafana" "http://localhost:3000/api/health"

# Redis check
echo ""
echo "Redis:"
if docker exec llm-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: responsive"
else
    echo "❌ Redis: not responding"
fi

# Detailed Gateway health
echo ""
echo "========================================="
echo "Gateway Detailed Health:"
echo "========================================="
curl -s http://localhost:7000/health 2>/dev/null | python3 -m json.tool || echo "Gateway not available"

# GPU status
echo ""
echo "========================================="
echo "GPU Status:"
echo "========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
else
    echo "nvidia-smi not available"
fi

echo ""
echo "========================================="
echo "Health check complete"
echo "========================================="
