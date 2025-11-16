#!/bin/bash
#
# View logs for LLM Backend services
#

# Default to gateway if no argument
SERVICE=${1:-gateway}

cd llm_backend/docker

echo "========================================="
echo "📋 Viewing logs for: $SERVICE"
echo "   Press Ctrl+C to exit"
echo "========================================="
echo ""

case $SERVICE in
    gateway)
        docker-compose logs -f gateway
        ;;
    qwen-vl|vl)
        docker-compose logs -f qwen-vl
        ;;
    qwen-14b|14b|reasoning)
        docker-compose --profile manual logs -f qwen-14b
        ;;
    qwen-coder|coder|code)
        docker-compose --profile manual logs -f qwen-coder
        ;;
    redis)
        docker-compose logs -f redis
        ;;
    prometheus)
        docker-compose logs -f prometheus
        ;;
    grafana)
        docker-compose logs -f grafana
        ;;
    all)
        docker-compose --profile manual logs -f
        ;;
    *)
        echo "❌ Unknown service: $SERVICE"
        echo ""
        echo "Available services:"
        echo "  gateway, qwen-vl, qwen-14b, qwen-coder"
        echo "  redis, prometheus, grafana, all"
        exit 1
        ;;
esac
