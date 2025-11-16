"""
FastAPI Gateway for LLM Services
Unified entry point for all LLM inference requests
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import hashlib
from typing import List
from contextlib import asynccontextmanager
from loguru import logger

from .cache import RedisCache
from .load_balancer import LoadBalancer
from .models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelInfo,
    ErrorResponse
)

# Service Registry
# URLs point to vLLM services (OpenAI-compatible API)
# Optimized for RTX 5080 16GB VRAM
SERVICES = {
    "qwen-vl-7b": {
        "url": "http://localhost:8000/v1",
        "type": "multimodal",
        "priority": 1,
        "capabilities": ["chat", "vision", "embeddings"],
        "vram_gb": 14
    },
    "qwen-14b": {
        "url": "http://localhost:8001/v1",
        "type": "reasoning",
        "priority": 1,
        "capabilities": ["chat", "reasoning"],
        "vram_gb": 12
    },
    "qwen-coder-7b": {
        "url": "http://localhost:8002/v1",
        "type": "code",
        "priority": 2,
        "capabilities": ["chat", "code_generation"],
        "vram_gb": 14
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting LLM Gateway...")

    # Initialize Redis cache
    app.state.cache = RedisCache(host="localhost", port=6379)
    try:
        await app.state.cache.connect()
    except Exception as e:
        logger.warning(f"⚠️ Redis not available: {e}. Continuing without cache.")
        app.state.cache = None

    # Initialize load balancer
    app.state.load_balancer = LoadBalancer(SERVICES)

    # Initialize HTTP client
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    logger.info("✅ LLM Gateway started successfully")

    yield

    # Shutdown
    logger.info("🛑 Shutting down LLM Gateway...")
    if app.state.cache:
        await app.state.cache.disconnect()
    await app.state.http_client.aclose()
    logger.info("✅ LLM Gateway stopped")


# Create FastAPI application
app = FastAPI(
    title="LLM Service Gateway",
    description="Unified gateway for self-hosted LLM services",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_cache_key(request: ChatRequest) -> str:
    """Generate cache key from request"""
    content = f"{request.model}:{request.messages}:{request.temperature}:{request.max_tokens}"
    return f"chat:{hashlib.md5(content.encode()).hexdigest()}"


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint (OpenAI-compatible)

    Handles LLM chat requests with caching and load balancing
    """
    start_time = time.time()

    # Check cache (only for non-streaming requests)
    if not request.stream and app.state.cache:
        cache_key = generate_cache_key(request)
        cached = await app.state.cache.get(cache_key)
        if cached:
            logger.info(f"✅ Cache hit for {request.model}")
            return JSONResponse(content=cached)

    # Get service URL
    service_url = app.state.load_balancer.get_service(request.model)
    if not service_url:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found or unavailable"
        )

    # Forward request to vLLM service
    try:
        response = await app.state.http_client.post(
            f"{service_url}/chat/completions",
            json=request.dict(),
            timeout=300.0
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Service error: {response.text}"
            )

        result = response.json()

        # Cache result
        if not request.stream and app.state.cache:
            await app.state.cache.set(cache_key, result, ttl=3600)

        # Log metrics
        duration = time.time() - start_time
        logger.info(
            f"✅ Chat completion: model={request.model}, "
            f"duration={duration:.2f}s, "
            f"tokens={result.get('usage', {}).get('total_tokens', 'N/A')}"
        )

        return result

    except httpx.RequestError as e:
        logger.error(f"❌ Request failed for {request.model}: {e}")
        app.state.load_balancer.mark_unhealthy(request.model)
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings endpoint

    Uses multimodal model (Qwen-VL) for embeddings
    """
    # Route to multimodal model
    model_name = "qwen-vl-72b"
    service_url = app.state.load_balancer.get_service(model_name)

    if not service_url:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable"
        )

    try:
        response = await app.state.http_client.post(
            f"{service_url}/embeddings",
            json=request.dict()
        )
        return response.json()

    except httpx.RequestError as e:
        logger.error(f"❌ Embedding request failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Checks gateway and all LLM services
    """
    health = {
        "status": "healthy",
        "gateway": "healthy",
        "services": {}
    }

    # Check each service
    for name, service in SERVICES.items():
        try:
            response = await app.state.http_client.get(
                f"{service['url']}/health",
                timeout=5.0
            )
            health["services"][name] = (
                "healthy" if response.status_code == 200 else "unhealthy"
            )
        except Exception as e:
            health["services"][name] = f"down: {str(e)}"
            logger.warning(f"⚠️ Service {name} health check failed: {e}")

    # Overall status
    all_healthy = all(
        status == "healthy"
        for status in health["services"].values()
    )
    health["status"] = "healthy" if all_healthy else "degraded"

    return health


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List available models

    Returns all registered LLM models
    """
    models = [
        ModelInfo(
            id=name,
            object="model",
            type=info["type"],
            capabilities=info["capabilities"]
        )
        for name, info in SERVICES.items()
    ]
    return models


@app.get("/stats")
async def get_stats():
    """
    Get gateway statistics

    Returns cache and load balancer stats
    """
    stats = {
        "load_balancer": app.state.load_balancer.get_stats()
    }

    if app.state.cache:
        stats["cache"] = await app.state.cache.get_stats()

    return stats


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code
            }
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logger.add(
        "logs/gateway_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7000,
        reload=True,
        log_level="info"
    )
