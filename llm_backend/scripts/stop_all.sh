#!/bin/bash
#
# Stop all LLM Backend services
#

set -e

echo "========================================="
echo "🛑 Stopping LLM Backend Services"
echo "========================================="
echo ""

# Check if running in project root
if [ ! -f "llm_backend/docker/docker-compose.yml" ]; then
    echo "❌ Error: Must run from project root"
    exit 1
fi

# Navigate to docker directory
cd llm_backend/docker

# Stop all services
echo "Stopping all services..."
docker-compose --profile manual down

echo ""
echo "✅ All services stopped"
echo ""
echo "Note: Volumes are preserved. To remove volumes:"
echo "  docker-compose --profile manual down -v"
