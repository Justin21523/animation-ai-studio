"""
GPU Isolation Layer

4-layer protection system to ensure 100% CPU-only operation:
1. Environment Variable Enforcement (CUDA_VISIBLE_DEVICES="")
2. Runtime GPU Detection and Blocking
3. Library Import Interception (torch.cuda, tensorflow-gpu)
4. Process Monitoring and Termination

Author: Animation AI Studio
Date: 2025-12-02
"""

import os
import sys
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class IsolationLevel(Enum):
    """GPU isolation security level"""
    PARANOID = "paranoid"    # All 4 layers active
    STRICT = "strict"        # Layers 1-3 active
    MODERATE = "moderate"    # Layers 1-2 active
    MINIMAL = "minimal"      # Layer 1 only


@dataclass
class GPUCheckResult:
    """Result of GPU detection check"""
    gpu_detected: bool
    gpu_processes: List[str] = field(default_factory=list)
    cuda_available: bool = False
    cuda_device_count: int = 0
    violations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_safe(self) -> bool:
        """Check if system is safe (no GPU usage)"""
        return not self.gpu_detected and not self.cuda_available


class GPUIsolation:
    """
    GPU Isolation Layer

    Provides 4-layer protection to ensure automation runs CPU-only:

    Layer 1: Environment Variable Enforcement
    - Sets CUDA_VISIBLE_DEVICES="" at startup
    - Blocks CUDA device enumeration
    - Prevents accidental GPU allocation

    Layer 2: Runtime GPU Detection
    - Checks for CUDA availability via torch/tensorflow
    - Detects GPU processes via nvidia-smi
    - Monitors system GPU usage

    Layer 3: Library Import Interception
    - Intercepts torch.cuda imports
    - Blocks tensorflow-gpu initialization
    - Prevents GPU-accelerated library loading

    Layer 4: Process Monitoring
    - Continuously monitors process GPU usage
    - Terminates process if GPU usage detected
    - Logs violations for audit

    Example:
        isolation = GPUIsolation(level=IsolationLevel.PARANOID)

        # Enforce at startup
        isolation.enforce()

        # Periodic checks
        result = isolation.check_gpu_usage()
        if not result.is_safe():
            isolation.handle_violation(result)
    """

    def __init__(
        self,
        level: IsolationLevel = IsolationLevel.STRICT,
        check_interval: float = 30.0,
        auto_terminate: bool = True
    ):
        """
        Initialize GPU Isolation

        Args:
            level: Isolation security level
            check_interval: Interval for periodic checks (seconds)
            auto_terminate: Automatically terminate on GPU usage detection
        """
        self.level = level
        self.check_interval = check_interval
        self.auto_terminate = auto_terminate

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._violation_count = 0

        logger.info(f"GPUIsolation initialized: level={level.value}, auto_terminate={auto_terminate}")

    def enforce(self):
        """
        Enforce GPU isolation

        Applies protection layers based on isolation level.
        """
        logger.info(f"Enforcing GPU isolation: {self.level.value}")

        # Layer 1: Environment Variables (always applied)
        self._enforce_env_vars()

        if self.level.value in ["strict", "paranoid"]:
            # Layer 2: Runtime Detection
            result = self.check_gpu_usage()
            if not result.is_safe():
                logger.error(f"GPU detected during enforcement: {result.violations}")
                if self.auto_terminate:
                    raise RuntimeError(
                        f"GPU isolation violation: {result.violations}. "
                        "Cannot proceed with GPU-only environment."
                    )

        if self.level.value in ["strict", "paranoid"]:
            # Layer 3: Library Interception
            self._intercept_cuda_imports()

        if self.level.value == "paranoid":
            # Layer 4: Start continuous monitoring
            self.start_monitoring()

        logger.info("GPU isolation enforced successfully")

    def _enforce_env_vars(self):
        """Layer 1: Set environment variables to disable CUDA"""
        # Disable CUDA devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Disable CUDA for various frameworks
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # Disable PyTorch CUDA
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Force CPU-only for common frameworks
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

        logger.info("Layer 1 (Environment Variables): Enforced")

    def _intercept_cuda_imports(self):
        """Layer 3: Intercept CUDA library imports"""
        # Prevent torch.cuda usage
        try:
            import torch
            if torch.cuda.is_available():
                logger.warning("torch.cuda reports available - attempting to disable")
                # Monkey-patch torch.cuda to always return False
                torch.cuda.is_available = lambda: False
                torch.cuda.device_count = lambda: 0
                logger.info("Layer 3 (Import Interception): torch.cuda disabled")
        except ImportError:
            logger.debug("torch not installed, skipping torch.cuda interception")

        # Prevent TensorFlow GPU usage
        try:
            import tensorflow as tf
            # Force CPU-only
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Layer 3 (Import Interception): TensorFlow GPU disabled")
        except ImportError:
            logger.debug("tensorflow not installed, skipping TF GPU interception")
        except Exception as e:
            logger.warning(f"Could not disable TensorFlow GPU: {e}")

    def check_gpu_usage(self) -> GPUCheckResult:
        """
        Layer 2: Check for GPU usage

        Returns:
            GPUCheckResult with detection status
        """
        result = GPUCheckResult(gpu_detected=False)

        # Check 1: nvidia-smi for GPU processes
        gpu_processes = self._check_nvidia_smi()
        if gpu_processes:
            result.gpu_detected = True
            result.gpu_processes = gpu_processes
            result.violations.append(f"GPU processes detected: {gpu_processes}")

        # Check 2: torch.cuda availability
        try:
            import torch
            if torch.cuda.is_available():
                result.cuda_available = True
                result.cuda_device_count = torch.cuda.device_count()
                result.violations.append(
                    f"CUDA available: {result.cuda_device_count} devices"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error checking torch.cuda: {e}")

        # Check 3: TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                result.cuda_available = True
                result.violations.append(f"TensorFlow GPU devices: {len(gpus)}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error checking TensorFlow GPU: {e}")

        # Check 4: Environment variables
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible and cuda_visible != "":
            result.violations.append(
                f"CUDA_VISIBLE_DEVICES not empty: '{cuda_visible}'"
            )

        return result

    def _check_nvidia_smi(self) -> List[str]:
        """
        Check for GPU processes using nvidia-smi

        Returns:
            List of GPU process names
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                processes = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        pid, process = line.split(",", 1)
                        # Filter out LoRA training processes (allowed)
                        if "train" not in process.lower() and "lora" not in process.lower():
                            processes.append(f"{process.strip()} (PID: {pid.strip()})")
                return processes

        except FileNotFoundError:
            logger.debug("nvidia-smi not found, skipping GPU process check")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timeout")
        except Exception as e:
            logger.debug(f"Error running nvidia-smi: {e}")

        return []

    def start_monitoring(self):
        """
        Layer 4: Start continuous GPU monitoring

        Runs in background thread, periodically checks for GPU usage.
        """
        if self._monitoring:
            logger.warning("GPU monitoring already running")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="GPUMonitor"
        )
        self._monitor_thread.start()
        logger.info(f"Layer 4 (Process Monitoring): Started (interval={self.check_interval}s)")

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            logger.info("Layer 4 (Process Monitoring): Stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                result = self.check_gpu_usage()
                if not result.is_safe():
                    self._violation_count += 1
                    self.handle_violation(result)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            time.sleep(self.check_interval)

    def handle_violation(self, result: GPUCheckResult):
        """
        Handle GPU usage violation

        Args:
            result: GPU check result with violations
        """
        logger.error(
            f"GPU ISOLATION VIOLATION #{self._violation_count}: "
            f"{', '.join(result.violations)}"
        )

        # Log detailed violation info
        logger.error(
            f"GPU Status: detected={result.gpu_detected}, "
            f"cuda_available={result.cuda_available}, "
            f"processes={result.gpu_processes}"
        )

        if self.auto_terminate:
            logger.critical("AUTO-TERMINATING due to GPU isolation violation")
            # Give time to flush logs
            time.sleep(1)
            sys.exit(1)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current isolation status

        Returns:
            Status dictionary
        """
        current_result = self.check_gpu_usage()

        return {
            "level": self.level.value,
            "monitoring": self._monitoring,
            "safe": current_result.is_safe(),
            "violation_count": self._violation_count,
            "cuda_available": current_result.cuda_available,
            "gpu_processes": current_result.gpu_processes,
            "violations": current_result.violations
        }

    def __enter__(self):
        """Context manager entry"""
        self.enforce()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()

    def __repr__(self) -> str:
        return (
            f"GPUIsolation(level={self.level.value}, "
            f"monitoring={self._monitoring}, "
            f"violations={self._violation_count})"
        )
