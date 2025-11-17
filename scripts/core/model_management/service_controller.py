"""
Service Controller for Dynamic Model Switching

Controls LLM backend and SDXL pipeline services for VRAM management.

Author: Animation AI Studio
Date: 2025-11-17
"""

import subprocess
import time
import logging
import requests
from typing import Optional, Literal
from pathlib import Path
from dataclasses import dataclass

from .vram_monitor import VRAMMonitor


logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    """Status of a model service"""
    service_name: str
    is_running: bool
    vram_usage_gb: Optional[float]
    uptime_seconds: Optional[float]
    endpoint: Optional[str]


class ServiceController:
    """
    Controls LLM and SDXL services for dynamic model switching

    Features:
    - Start/stop LLM backend (vLLM + FastAPI Gateway)
    - Load/unload SDXL pipeline
    - Service health checking
    - VRAM-aware switching
    - Service status monitoring
    """

    # Service endpoints
    LLM_GATEWAY_URL = "http://localhost:8000"
    LLM_HEALTH_ENDPOINT = f"{LLM_GATEWAY_URL}/health"

    # Script paths (relative to project root)
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    LLM_START_SCRIPT = PROJECT_ROOT / "llm_backend" / "scripts" / "start_service.sh"
    LLM_STOP_SCRIPT = PROJECT_ROOT / "llm_backend" / "scripts" / "stop_service.sh"

    # Timing constants (seconds)
    LLM_START_TIMEOUT = 60
    LLM_STOP_TIMEOUT = 15
    HEALTH_CHECK_INTERVAL = 2
    HEALTH_CHECK_MAX_RETRIES = 10

    def __init__(self, vram_monitor: Optional[VRAMMonitor] = None):
        """
        Initialize service controller

        Args:
            vram_monitor: VRAMMonitor instance (creates new if None)
        """
        self.vram_monitor = vram_monitor or VRAMMonitor()

        # Service state
        self.llm_running = False
        self.llm_start_time: Optional[float] = None

        # Check if scripts exist
        if not self.LLM_START_SCRIPT.exists():
            logger.warning(f"LLM start script not found: {self.LLM_START_SCRIPT}")
        if not self.LLM_STOP_SCRIPT.exists():
            logger.warning(f"LLM stop script not found: {self.LLM_STOP_SCRIPT}")

        logger.info("ServiceController initialized")

    def is_llm_running(self) -> bool:
        """
        Check if LLM backend is running

        Returns:
            True if LLM is healthy and responding
        """
        try:
            response = requests.get(self.LLM_HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                self.llm_running = True
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        self.llm_running = False
        return False

    def start_llm(
        self,
        model: Optional[Literal["qwen-vl-7b", "qwen-14b", "qwen-coder-7b"]] = None,
        wait: bool = True
    ) -> bool:
        """
        Start LLM backend service

        Args:
            model: Specific model to start (None = default from config)
            wait: Wait for service to be healthy before returning

        Returns:
            True if service started successfully

        Raises:
            RuntimeError: If service fails to start
        """
        if self.is_llm_running():
            logger.info("LLM backend already running")
            return True

        logger.info("Starting LLM backend service...")

        # Check VRAM availability
        model_key = model or "qwen-14b"  # Default
        if not self.vram_monitor.can_fit_model(model_key):
            raise RuntimeError(
                f"Insufficient VRAM to start {model_key}. "
                "Consider stopping other models first."
            )

        # Execute start script
        try:
            cmd = [str(self.LLM_START_SCRIPT)]
            if model:
                cmd.extend(["--model", model])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.LLM_START_TIMEOUT,
                cwd=str(self.PROJECT_ROOT)
            )

            if result.returncode != 0:
                logger.error(f"Failed to start LLM: {result.stderr}")
                return False

            logger.info("LLM start script executed")

        except subprocess.TimeoutExpired:
            logger.error(f"LLM start timeout ({self.LLM_START_TIMEOUT}s)")
            return False
        except Exception as e:
            logger.error(f"Error starting LLM: {e}")
            return False

        # Wait for service to be healthy
        if wait:
            logger.info("Waiting for LLM to be ready...")
            for i in range(self.HEALTH_CHECK_MAX_RETRIES):
                time.sleep(self.HEALTH_CHECK_INTERVAL)

                if self.is_llm_running():
                    self.llm_start_time = time.time()
                    logger.info(f"LLM backend ready (took {(i+1) * self.HEALTH_CHECK_INTERVAL}s)")
                    return True

            logger.error("LLM failed to become healthy")
            return False

        return True

    def stop_llm(self, wait: bool = True) -> bool:
        """
        Stop LLM backend service

        Args:
            wait: Wait for service to fully stop before returning

        Returns:
            True if service stopped successfully
        """
        if not self.is_llm_running():
            logger.info("LLM backend already stopped")
            return True

        logger.info("Stopping LLM backend service...")

        try:
            result = subprocess.run(
                [str(self.LLM_STOP_SCRIPT)],
                capture_output=True,
                text=True,
                timeout=self.LLM_STOP_TIMEOUT,
                cwd=str(self.PROJECT_ROOT)
            )

            if result.returncode != 0:
                logger.warning(f"LLM stop script returned non-zero: {result.stderr}")

            logger.info("LLM stop script executed")

        except subprocess.TimeoutExpired:
            logger.warning(f"LLM stop timeout ({self.LLM_STOP_TIMEOUT}s)")
        except Exception as e:
            logger.error(f"Error stopping LLM: {e}")
            return False

        # Wait for service to stop
        if wait:
            logger.info("Waiting for LLM to stop...")
            for i in range(5):
                time.sleep(2)

                if not self.is_llm_running():
                    self.llm_running = False
                    self.llm_start_time = None
                    logger.info("LLM backend stopped")

                    # Clear CUDA cache
                    self.vram_monitor.clear_cache()
                    return True

            logger.warning("LLM may still be running")

        return True

    def restart_llm(
        self,
        model: Optional[Literal["qwen-vl-7b", "qwen-14b", "qwen-coder-7b"]] = None
    ) -> bool:
        """
        Restart LLM backend service

        Args:
            model: Specific model to start (None = keep current)

        Returns:
            True if restart successful
        """
        logger.info("Restarting LLM backend...")

        if not self.stop_llm(wait=True):
            logger.error("Failed to stop LLM during restart")
            return False

        # Give system time to release resources
        time.sleep(3)

        if not self.start_llm(model=model, wait=True):
            logger.error("Failed to start LLM during restart")
            return False

        logger.info("LLM backend restarted successfully")
        return True

    def get_llm_status(self) -> ServiceStatus:
        """
        Get current LLM service status

        Returns:
            ServiceStatus with LLM state
        """
        is_running = self.is_llm_running()
        vram_usage = None
        uptime = None

        if is_running:
            # Estimate VRAM (approximate - we don't know which model is running)
            snapshot = self.vram_monitor.get_snapshot()
            vram_usage = snapshot.allocated_gb

            if self.llm_start_time:
                uptime = time.time() - self.llm_start_time

        return ServiceStatus(
            service_name="LLM Backend",
            is_running=is_running,
            vram_usage_gb=vram_usage,
            uptime_seconds=uptime,
            endpoint=self.LLM_GATEWAY_URL if is_running else None
        )

    def print_status(self):
        """Print human-readable service status"""
        llm_status = self.get_llm_status()

        print("=" * 60)
        print("Service Controller Status")
        print("=" * 60)

        print(f"\nLLM Backend:")
        print(f"  Running: {'✓ Yes' if llm_status.is_running else '✗ No'}")
        if llm_status.is_running:
            print(f"  Endpoint: {llm_status.endpoint}")
            if llm_status.vram_usage_gb:
                print(f"  VRAM Usage: ~{llm_status.vram_usage_gb:.1f} GB")
            if llm_status.uptime_seconds:
                print(f"  Uptime: {llm_status.uptime_seconds:.0f}s")

        print("\nVRAM Status:")
        stats = self.vram_monitor.get_detailed_stats()
        print(f"  Allocated: {stats['allocated_gb']:.2f} GB / {stats['total_gb']:.2f} GB")
        print(f"  Available: {stats['available_for_new_model_gb']:.2f} GB")

        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Initializing Service Controller...")
    controller = ServiceController()

    print("\nCurrent Status:")
    controller.print_status()

    # Example: Check if LLM is running
    if controller.is_llm_running():
        print("\n✓ LLM backend is healthy")
    else:
        print("\n✗ LLM backend is not running")
        print("To start: bash llm_backend/scripts/start_all.sh")
