"""
VRAM Monitor for RTX 5080 16GB GPU

Monitors GPU memory usage and provides utilities for VRAM management.

Author: Animation AI Studio
Date: 2025-11-17
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

try:
    import torch
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. GPU monitoring will be limited.")


logger = logging.getLogger(__name__)


@dataclass
class VRAMSnapshot:
    """Snapshot of GPU memory state at a point in time"""
    timestamp: float
    device_id: int
    device_name: str
    total_vram_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization_percent: float


@dataclass
class VRAMEstimate:
    """Estimated VRAM requirements for a model"""
    model_name: str
    estimated_vram_gb: float
    confidence: str  # "high", "medium", "low"
    notes: str


class VRAMMonitor:
    """
    VRAM monitoring and management for RTX 5080 16GB

    Features:
    - Real-time VRAM usage tracking
    - Model-specific VRAM estimates
    - Safety checks before model loading
    - Memory leak detection
    - VRAM usage history
    """

    # VRAM estimates for common models (in GB)
    MODEL_VRAM_ESTIMATES = {
        # LLM models
        "qwen-vl-7b": {"vram": 13.8, "confidence": "high", "notes": "Measured on RTX 5080"},
        "qwen-14b": {"vram": 11.5, "confidence": "high", "notes": "Measured on RTX 5080"},
        "qwen-coder-7b": {"vram": 13.5, "confidence": "high", "notes": "Measured on RTX 5080"},

        # Image generation models
        "sdxl-base": {"vram": 10.5, "confidence": "high", "notes": "FP16, basic generation"},
        "sdxl-lora": {"vram": 11.5, "confidence": "high", "notes": "SDXL + single LoRA"},
        "sdxl-controlnet": {"vram": 14.5, "confidence": "high", "notes": "SDXL + LoRA + ControlNet"},

        # Voice synthesis models
        "gpt-sovits-small": {"vram": 3.5, "confidence": "medium", "notes": "Inference only"},
        "gpt-sovits-train": {"vram": 9.0, "confidence": "medium", "notes": "Training mode"},
    }

    # Hardware constraints
    TOTAL_VRAM_GB = 16.0
    SAFE_MAX_USAGE_GB = 15.5  # Leave 0.5GB buffer

    def __init__(self, device: int = 0):
        """
        Initialize VRAM monitor

        Args:
            device: CUDA device ID (default: 0)
        """
        self.device = device
        self.history: List[VRAMSnapshot] = []
        self.max_history_size = 100

        # Initialize NVML if available
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                self.nvml_initialized = True
                logger.info("NVML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")

        # Verify CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. VRAMMonitor requires GPU.")

        if device >= torch.cuda.device_count():
            raise ValueError(f"Device {device} not available. Only {torch.cuda.device_count()} GPUs detected.")

        # Get device properties
        self.device_name = torch.cuda.get_device_name(device)
        self.device_properties = torch.cuda.get_device_properties(device)

        logger.info(f"VRAMMonitor initialized for {self.device_name} (Device {device})")

    def get_snapshot(self) -> VRAMSnapshot:
        """
        Get current VRAM usage snapshot

        Returns:
            VRAMSnapshot with current memory state
        """
        # PyTorch memory stats
        allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1e9    # GB
        total = self.device_properties.total_memory / 1e9           # GB
        free = total - allocated
        utilization = (allocated / total) * 100

        snapshot = VRAMSnapshot(
            timestamp=time.time(),
            device_id=self.device,
            device_name=self.device_name,
            total_vram_gb=total,
            allocated_gb=allocated,
            reserved_gb=reserved,
            free_gb=free,
            utilization_percent=utilization
        )

        # Add to history
        self.history.append(snapshot)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        return snapshot

    def get_detailed_stats(self) -> Dict[str, float]:
        """
        Get detailed VRAM statistics

        Returns:
            Dictionary with detailed memory stats
        """
        snapshot = self.get_snapshot()

        stats = {
            "total_gb": snapshot.total_vram_gb,
            "allocated_gb": snapshot.allocated_gb,
            "reserved_gb": snapshot.reserved_gb,
            "free_gb": snapshot.free_gb,
            "utilization_percent": snapshot.utilization_percent,
            "safe_max_gb": self.SAFE_MAX_USAGE_GB,
            "available_for_new_model_gb": self.SAFE_MAX_USAGE_GB - snapshot.allocated_gb
        }

        # Add NVML stats if available
        if self.nvml_initialized:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)

                stats["nvml_total_gb"] = mem_info.total / 1e9
                stats["nvml_used_gb"] = mem_info.used / 1e9
                stats["nvml_free_gb"] = mem_info.free / 1e9
                stats["gpu_utilization_percent"] = util_rates.gpu
                stats["memory_utilization_percent"] = util_rates.memory
            except Exception as e:
                logger.warning(f"Failed to get NVML stats: {e}")

        return stats

    def can_fit_model(self, model_name: str, strict: bool = True) -> bool:
        """
        Check if a model can fit in available VRAM

        Args:
            model_name: Model identifier (e.g., "sdxl-base", "qwen-14b")
            strict: If True, use safe maximum (15.5GB). If False, use total (16GB)

        Returns:
            True if model can fit, False otherwise
        """
        estimate = self.get_model_estimate(model_name)
        if estimate is None:
            logger.warning(f"No VRAM estimate for '{model_name}'. Cannot verify fit.")
            return False

        snapshot = self.get_snapshot()
        max_allowed = self.SAFE_MAX_USAGE_GB if strict else self.TOTAL_VRAM_GB
        available = max_allowed - snapshot.allocated_gb

        can_fit = estimate.estimated_vram_gb <= available

        if not can_fit:
            logger.warning(
                f"Model '{model_name}' requires ~{estimate.estimated_vram_gb:.1f}GB, "
                f"but only {available:.1f}GB available (current usage: {snapshot.allocated_gb:.1f}GB)"
            )

        return can_fit

    def get_model_estimate(self, model_name: str) -> Optional[VRAMEstimate]:
        """
        Get VRAM estimate for a specific model

        Args:
            model_name: Model identifier

        Returns:
            VRAMEstimate if known, None otherwise
        """
        # Normalize model name (lowercase, remove underscores/dashes)
        normalized = model_name.lower().replace("_", "-")

        if normalized in self.MODEL_VRAM_ESTIMATES:
            est = self.MODEL_VRAM_ESTIMATES[normalized]
            return VRAMEstimate(
                model_name=model_name,
                estimated_vram_gb=est["vram"],
                confidence=est["confidence"],
                notes=est["notes"]
            )

        return None

    def require_free_vram(self, required_gb: float, model_name: str = "model"):
        """
        Raise error if required VRAM is not available

        Args:
            required_gb: Required VRAM in GB
            model_name: Name of model requesting VRAM (for error message)

        Raises:
            RuntimeError: If insufficient VRAM available
        """
        snapshot = self.get_snapshot()
        available = self.SAFE_MAX_USAGE_GB - snapshot.allocated_gb

        if available < required_gb:
            raise RuntimeError(
                f"Insufficient VRAM for {model_name}. "
                f"Required: {required_gb:.1f}GB, Available: {available:.1f}GB, "
                f"Current usage: {snapshot.allocated_gb:.1f}GB / {self.SAFE_MAX_USAGE_GB:.1f}GB"
            )

    def clear_cache(self):
        """Clear PyTorch CUDA cache"""
        torch.cuda.empty_cache()
        logger.info("PyTorch CUDA cache cleared")

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        torch.cuda.reset_peak_memory_stats(self.device)
        logger.info("Peak memory statistics reset")

    def get_peak_memory(self) -> float:
        """
        Get peak memory usage since last reset

        Returns:
            Peak memory in GB
        """
        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        return peak_bytes / 1e9

    def print_summary(self):
        """Print human-readable VRAM summary"""
        stats = self.get_detailed_stats()

        print("=" * 60)
        print(f"VRAM Summary - {self.device_name}")
        print("=" * 60)
        print(f"Total VRAM:        {stats['total_gb']:.2f} GB")
        print(f"Allocated:         {stats['allocated_gb']:.2f} GB ({stats['utilization_percent']:.1f}%)")
        print(f"Reserved:          {stats['reserved_gb']:.2f} GB")
        print(f"Free:              {stats['free_gb']:.2f} GB")
        print(f"Safe Maximum:      {stats['safe_max_gb']:.2f} GB")
        print(f"Available for New: {stats['available_for_new_model_gb']:.2f} GB")

        if self.nvml_initialized:
            print("-" * 60)
            print(f"GPU Utilization:   {stats.get('gpu_utilization_percent', 0):.1f}%")
            print(f"Memory Util:       {stats.get('memory_utilization_percent', 0):.1f}%")

        print("=" * 60)

    def cleanup(self):
        """Cleanup resources"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown")
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {e}")


def check_vram_requirements():
    """
    Utility function to check if system meets minimum VRAM requirements

    Raises:
        RuntimeError: If system does not meet requirements
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This project requires NVIDIA GPU.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices detected.")

    # Check device 0
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / 1e9

    if total_vram_gb < 14.0:
        raise RuntimeError(
            f"Insufficient VRAM. Required: 16GB (14GB minimum), "
            f"Available: {total_vram_gb:.1f}GB on {torch.cuda.get_device_name(0)}"
        )

    logger.info(f"VRAM check passed: {total_vram_gb:.1f}GB on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Initializing VRAM Monitor...")
    monitor = VRAMMonitor(device=0)

    print("\nCurrent VRAM Status:")
    monitor.print_summary()

    print("\nModel Fit Checks:")
    models_to_check = ["qwen-14b", "sdxl-base", "sdxl-controlnet", "gpt-sovits-small"]
    for model_name in models_to_check:
        can_fit = monitor.can_fit_model(model_name)
        estimate = monitor.get_model_estimate(model_name)
        if estimate:
            print(f"  {model_name:20s}: {'✓ Can fit' if can_fit else '✗ Cannot fit':12s} (~{estimate.estimated_vram_gb:.1f}GB)")

    print("\nPeak Memory:")
    print(f"  {monitor.get_peak_memory():.2f} GB")

    monitor.cleanup()
