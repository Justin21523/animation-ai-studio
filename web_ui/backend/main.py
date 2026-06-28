"""
FastAPI Backend for Animation AI Studio Web UI.

This backend provides REST API and SSE endpoints for job management,
integrating with existing batch scripts via BashRunnerAdapter.
"""
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web_ui.backend.db.schema import init_database
from web_ui.backend.db.operations import JobDatabase
from web_ui.backend.services.job_service import JobService
from web_ui.backend.api.jobs import router as jobs_router, set_job_service
from web_ui.backend.api.action import router as action_router
from web_ui.backend.api.creative import router as creative_router
from web_ui.backend.api.image import router as image_router
from web_ui.backend.api.results import router as results_router
from web_ui.backend.api.results import set_allowed_base_dirs
from web_ui.backend.api.system import router as system_router
from web_ui.backend.api.stats import router as stats_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """
    Load backend configuration from YAML file.

    Args:
        config_path: Path to config file (default: configs/web_ui/backend.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        env_path = os.environ.get("WEB_UI_CONFIG_PATH")
        config_path = Path(env_path) if env_path else (project_root / "configs" / "web_ui" / "backend.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks:
    - Initialize database
    - Create JobService
    - Set up dependency injection
    """
    # Startup
    logger.info("Starting Animation AI Studio Web UI Backend...")

    # Load configuration
    config = load_config()

    # Initialize database
    db_path = config['database']['path']
    if not Path(db_path).is_absolute():
        db_path = str(project_root / db_path)
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    init_database(db_path)
    logger.info(f"Database initialized: {db_path}")

    # Create JobDatabase
    job_db = JobDatabase(db_path)

    # Configure result browser roots from config.
    paths_config = config.get("paths", {}) or {}
    result_roots = paths_config.get("results_roots") or [
        paths_config.get("outputs_root"),
        paths_config.get("extracted_root"),
    ]
    resolved_roots = []
    for root in result_roots:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        root_path.mkdir(parents=True, exist_ok=True)
        resolved_roots.append(str(root_path))
    set_allowed_base_dirs(resolved_roots)

    # Create JobService
    job_service = JobService(db=job_db, config=config)

    # Set job service for dependency injection
    set_job_service(job_service)

    logger.info("Backend initialization complete")

    # Store in app state for access
    app.state.config = config
    app.state.job_service = job_service

    yield

    # Shutdown
    logger.info("Shutting down backend...")
    await job_db.close()
    logger.info("Backend shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Animation AI Studio Web UI",
    description="Web interface for Animation AI Studio batch processing pipelines",
    version="0.1.0",
    lifespan=lifespan
)


# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8100",
        "http://127.0.0.1:8100",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routers
app.include_router(jobs_router)
app.include_router(action_router)
app.include_router(creative_router)
app.include_router(image_router)
app.include_router(results_router)
app.include_router(system_router)
app.include_router(stats_router)


# Serve frontend static files
frontend_dir = project_root / "web_ui" / "frontend"
if frontend_dir.exists():
    # Mount CSS directory
    css_dir = frontend_dir / "css"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
        logger.info(f"Serving CSS from {css_dir}")

    # Mount JS directory
    js_dir = frontend_dir / "js"
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
        logger.info(f"Serving JS from {js_dir}")

    # Mount assets directory
    assets_dir = frontend_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        logger.info(f"Serving assets from {assets_dir}")

    logger.info(f"Frontend resources mounted")


@app.get("/")
async def root():
    """Root endpoint - serve frontend index.html."""
    from fastapi.responses import FileResponse

    frontend_dir = project_root / "web_ui" / "frontend"
    index_path = frontend_dir / "index.html"

    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {
            "message": "Animation AI Studio Web UI",
            "version": "0.1.0",
            "docs": "/docs",
            "api": "/api",
            "error": "Frontend index.html not found"
        }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "animation-ai-studio-web-ui",
        "version": "0.1.0"
    }


@app.get("/api/config")
async def get_config():
    """
    Get public configuration information.

    Returns subset of configuration safe to expose to frontend.
    """
    config = app.state.config

    return {
        "pipeline_types": ["cpu_only", "cpu_gpu_full", "image_generate"],
        "cpu_pipeline": {
            "default_workers": config['cpu_pipeline']['default_workers'],
            "max_workers": config['cpu_pipeline']['max_workers']
        },
        "gpu_pipeline": {
            "sam2_models": ["base", "large", "small", "tiny"],
            "llm_models": ["qwen-vl-7b", "qwen-14b"],
            "enable_voice": config['gpu_pipeline']['enable_voice']
        },
        "resources": {
            "max_concurrent_jobs": config['resources']['max_concurrent_jobs']
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Load config to get server settings
    config = load_config()
    server_config = config['server']

    # Run server
    uvicorn.run(
        "main:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config.get('reload', False),
        log_level="info"
    )
