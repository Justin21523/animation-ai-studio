# Gateway Dockerfile
# FastAPI Gateway for LLM services

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements/llm_backend.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy gateway code
COPY llm_backend/gateway /app/gateway

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose gateway port
EXPOSE 7000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7000/health || exit 1

# Run gateway
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "7000", "--log-level", "info"]
