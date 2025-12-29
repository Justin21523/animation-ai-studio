"""
Model Manager for Dynamic VRAM-Constrained Model Switching

Orchestrates loading/unloading of LLM, SDXL, and TTS models within 16GB VRAM constraint.

Author: Animation AI Studio
Date: 2025-11-17
"""

import time
import logging
from typing import Optional, Literal, Any, Dict
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

from .vram_monitor import VRAMMonitor
from .service_controller import ServiceController


logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Current state of loaded models"""
    active_model: Optional[str]
    model_type: Optional[Literal["llm", "sdxl", "tts"]]
    vram_usage_gb: float
    loaded_at: Optional[float]


class ModelManager:
    """
    Dynamic model loading/unloading for RTX 5080 16GB VRAM

    Rules:
    1. Only ONE heavy model at a time (LLM XOR SDXL)
    2. TTS can coexist with stopped heavy models
    3. Automatic VRAM safety checks
    4. Service orchestration for model switching

    Usage:
        manager = ModelManager()

        # Use LLM
        with manager.use_llm(model="qwen-14b"):
            # LLM is active, SDXL is stopped
            response = llm_client.chat(...)

        # Switch to SDXL
        with manager.use_sdxl():
            # SDXL is loaded, LLM is stopped
            image = sdxl_pipeline.generate(...)

        # Use TTS (lightweight)
        with manager.use_tts():
            # TTS loaded, can run with LLM stopped
            audio = tts_model.synthesize(...)
    """

    # Model categories
    HEAVY_MODELS = ["llm", "sdxl"]
    LIGHT_MODELS = ["tts"]

    def __init__(
        self,
        vram_monitor: Optional[VRAMMonitor] = None,
        service_controller: Optional[ServiceController] = None,
        strict_vram_checks: bool = True
    ):
        """
        Initialize Model Manager

        Args:
            vram_monitor: VRAMMonitor instance (creates new if None)
            service_controller: ServiceController instance (creates new if None)
            strict_vram_checks: If True, enforce strict VRAM limits (15.5GB)
        """
        self.vram_monitor = vram_monitor or VRAMMonitor()
        self.service_controller = service_controller or ServiceController(self.vram_monitor)
        self.strict_vram_checks = strict_vram_checks

        # State tracking
        self.state = ModelState(
            active_model=None,
            model_type=None,
            vram_usage_gb=0.0,
            loaded_at=None
        )

        # SDXL pipeline state (loaded in-process)
        self.sdxl_pipeline: Optional[Any] = None
        self.tts_model: Optional[Any] = None

        logger.info("ModelManager initialized")

    def get_current_state(self) -> ModelState:
        """
        Get current model loading state

        Returns:
            ModelState with current active model
        """
        snapshot = self.vram_monitor.get_snapshot()
        self.state.vram_usage_gb = snapshot.allocated_gb
        return self.state

    def can_load_model(self, model_type: Literal["llm", "sdxl", "tts"], model_name: str) -> bool:
        """
        Check if a model can be loaded in current VRAM state

        Args:
            model_type: Type of model to load
            model_name: Specific model identifier

        Returns:
            True if model can be loaded safely
        """
        # Check if heavy model would conflict
        if model_type in self.HEAVY_MODELS:
            if self.state.active_model and self.state.model_type in self.HEAVY_MODELS:
                logger.warning(
                    f"Cannot load {model_type} '{model_name}': "
                    f"{self.state.model_type} '{self.state.active_model}' is already loaded"
                )
                return False

        # Check VRAM availability
        return self.vram_monitor.can_fit_model(model_name, strict=self.strict_vram_checks)

    def _ensure_heavy_model_unloaded(self):
        """
        Ensure no heavy model is loaded before loading another

        Raises:
            RuntimeError: If a heavy model is loaded and cannot be unloaded
        """
        # Stop LLM service if it's actually running (state may be stale).
        if self.service_controller.is_llm_running() or self.state.model_type == "llm":
            logger.info("Stopping LLM to free VRAM...")
            if not self.service_controller.stop_llm(wait=True):
                raise RuntimeError("Failed to stop LLM service")
            if self.state.model_type == "llm":
                self.state.active_model = None
                self.state.model_type = None
                self.state.loaded_at = None

        # Unload SDXL if present (in-process heavy model).
        if self.state.model_type == "sdxl" or self.sdxl_pipeline is not None:
            logger.info("Unloading SDXL to free VRAM...")
            self.unload_sdxl()

        # Clear CUDA cache
        self.vram_monitor.clear_cache()
        time.sleep(1)  # Give system time to release memory

    @contextmanager
    def use_llm(
        self,
        model: Optional[Literal["qwen-vl-7b", "qwen-14b", "qwen-coder-7b"]] = None,
        auto_unload: bool = True
    ):
        """
        Context manager for using LLM

        Args:
            model: Specific LLM model to use
            auto_unload: Automatically unload heavy models before loading LLM

        Yields:
            None (LLM is available via service endpoint)

        Example:
            with manager.use_llm(model="qwen-14b"):
                response = llm_client.chat(messages=[...])
        """
        model_key = model or "qwen-14b"

        # Check if LLM is already running
        if self.service_controller.is_llm_running():
            logger.info("LLM already running")
            # Best-effort state sync (we cannot reliably detect exact model variant here).
            self.state.active_model = model_key
            self.state.model_type = "llm"
            if self.state.loaded_at is None:
                self.state.loaded_at = time.time()
            yield
            return

        # Ensure no heavy model is loaded
        if auto_unload:
            self._ensure_heavy_model_unloaded()

        # Check VRAM
        if not self.can_load_model("llm", model_key):
            raise RuntimeError(f"Insufficient VRAM to load LLM '{model_key}'")

        # Start LLM
        logger.info(f"Starting LLM: {model_key}")
        if not self.service_controller.start_llm(model=model, wait=True):
            raise RuntimeError(f"Failed to start LLM '{model_key}'")

        # Update state
        self.state.active_model = model_key
        self.state.model_type = "llm"
        self.state.loaded_at = time.time()

        try:
            yield
        finally:
            # Note: We don't auto-stop LLM on context exit
            # User can keep it running or manually stop
            logger.info("LLM context exited (service still running)")

    @contextmanager
    def use_sdxl(self, auto_unload: bool = True):
        """
        Context manager for using SDXL

        Args:
            auto_unload: Automatically unload heavy models before loading SDXL

        Yields:
            SDXLPipeline instance

        Example:
            with manager.use_sdxl() as pipeline:
                image = pipeline.generate(prompt="...")
        """
        # Check if SDXL is already loaded
        if self.sdxl_pipeline is not None:
            logger.info("SDXL already loaded")
            yield self.sdxl_pipeline
            return

        # Ensure no heavy model is loaded
        if auto_unload:
            self._ensure_heavy_model_unloaded()

        # Check VRAM
        if not self.can_load_model("sdxl", "sdxl-base"):
            raise RuntimeError("Insufficient VRAM to load SDXL")

        # Load SDXL
        logger.info("Loading SDXL pipeline...")
        self.load_sdxl()

        # Update state
        self.state.active_model = "sdxl-base"
        self.state.model_type = "sdxl"
        self.state.loaded_at = time.time()

        try:
            yield self.sdxl_pipeline
        finally:
            # Auto-unload on context exit to free VRAM
            logger.info("SDXL context exited, unloading pipeline")
            self.unload_sdxl()

    @contextmanager
    def use_tts(self, auto_unload_heavy: bool = True):
        """
        Context manager for using TTS

        Args:
            auto_unload_heavy: Automatically unload heavy models before loading TTS

        Yields:
            TTS model instance

        Example:
            with manager.use_tts() as tts:
                audio = tts.synthesize(text="Hello")
        """
        # Check if TTS is already loaded
        if self.tts_model is not None:
            logger.info("TTS already loaded")
            yield self.tts_model
            return

        # TTS is lightweight, but might need heavy models unloaded
        if auto_unload_heavy:
            snapshot = self.vram_monitor.get_snapshot()
            if snapshot.allocated_gb > 12.0:  # If high usage, unload heavy models
                self._ensure_heavy_model_unloaded()

        # Check VRAM
        if not self.can_load_model("tts", "gpt-sovits-small"):
            raise RuntimeError("Insufficient VRAM to load TTS")

        # Load TTS
        logger.info("Loading TTS model...")
        self.load_tts()

        # Update state (TTS doesn't change active heavy model)
        previous_model = self.state.active_model
        previous_type = self.state.model_type

        try:
            yield self.tts_model
        finally:
            logger.info("TTS context exited, unloading model")
            self.unload_tts()

            # Restore previous state
            self.state.active_model = previous_model
            self.state.model_type = previous_type

    def load_sdxl(self):
        """
        Load SDXL pipeline

        Loads SDXLPipelineManager from `configs/generation/sdxl_config.yaml`.
        """
        if self.sdxl_pipeline is not None:
            logger.warning("SDXL already loaded")
            return

        from scripts.generation.image.sdxl_pipeline import SDXLPipelineManager
        import torch
        import yaml

        config_path = Path("configs/generation/sdxl_config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"SDXL config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        model_cfg = config.get("model") or {}
        device_cfg = config.get("device") or {}
        vram_cfg = config.get("vram") or {}

        model_path = model_cfg.get("base_model")
        if not model_path:
            raise ValueError("Missing 'model.base_model' in configs/generation/sdxl_config.yaml")

        model_path_p = Path(model_path)
        if not model_path_p.exists():
            raise FileNotFoundError(
                f"SDXL base model not found: {model_path_p}\n"
                "Update configs/generation/sdxl_config.yaml:model.base_model to a valid local path."
            )

        dtype_str = str(device_cfg.get("dtype", "float16")).lower()
        dtype = torch.float16 if dtype_str in ("float16", "fp16") else torch.float32
        device = str(device_cfg.get("device", "cuda"))

        logger.info("Loading SDXL pipeline...")
        self.sdxl_pipeline = SDXLPipelineManager(
            model_path=str(model_path_p),
            device=device,
            dtype=dtype,
            use_sdpa=True,
            enable_model_cpu_offload=bool(vram_cfg.get("enable_model_cpu_offload", False)),
            enable_vae_slicing=bool(vram_cfg.get("enable_vae_slicing", True)),
            enable_vae_tiling=bool(vram_cfg.get("enable_vae_tiling", True)),
            variant=str(model_cfg.get("variant", "fp16")),
        )
        self.sdxl_pipeline.load_pipeline()

        logger.info("SDXL pipeline loaded")

    def unload_sdxl(self):
        """Unload SDXL pipeline to free VRAM"""
        if self.sdxl_pipeline is None:
            logger.debug("SDXL not loaded")
            return

        logger.info("Unloading SDXL pipeline...")

        try:
            if hasattr(self.sdxl_pipeline, "unload_pipeline"):
                self.sdxl_pipeline.unload_pipeline()
        finally:
            self.sdxl_pipeline = None

        self.vram_monitor.clear_cache()

        # Update state
        if self.state.model_type == "sdxl":
            self.state.active_model = None
            self.state.model_type = None
            self.state.loaded_at = None

        logger.info("SDXL pipeline unloaded")

    def load_tts(self):
        """
        Load TTS model

        Loads a unified TTS adapter (prefers XTTS-v2 if available).
        """
        if self.tts_model is not None:
            logger.warning("TTS already loaded")
            return

        logger.info("Loading TTS model...")
        import yaml
        from scripts.synthesis.tts.unified_tts import UnifiedTTS

        config_path = Path("configs/generation/tts_config.yaml")
        config: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        tts_cfg = config.get("tts") or {}
        language = str(tts_cfg.get("language", "en"))
        backend = str(tts_cfg.get("backend", "auto")).lower()
        cache_dir = tts_cfg.get("cache_dir")
        prefer_device = tts_cfg.get("prefer_device")

        self.tts_model = UnifiedTTS(
            language=language,
            prefer_device=str(prefer_device) if prefer_device else None,
            backend=backend if backend in ("auto", "xtts", "silence") else "auto",
            cache_dir=str(cache_dir) if cache_dir else None,
        )

        logger.info("TTS model loaded")

    def unload_tts(self):
        """Unload TTS model"""
        if self.tts_model is None:
            logger.debug("TTS not loaded")
            return

        logger.info("Unloading TTS model...")

        try:
            if hasattr(self.tts_model, "cleanup"):
                self.tts_model.cleanup()
        except Exception as e:
            logger.warning(f"TTS cleanup failed: {e}")
        self.tts_model = None
        self.vram_monitor.clear_cache()

        logger.info("TTS model unloaded")

    def cleanup(self):
        """
        Cleanup all loaded models and resources

        Call this before exiting to ensure clean shutdown
        """
        logger.info("Cleaning up ModelManager...")

        # Unload SDXL
        if self.sdxl_pipeline is not None:
            self.unload_sdxl()

        # Unload TTS
        if self.tts_model is not None:
            self.unload_tts()

        # Stop LLM if running
        if self.service_controller.is_llm_running():
            logger.info("Stopping LLM service...")
            self.service_controller.stop_llm(wait=True)

        # Cleanup VRAM monitor
        self.vram_monitor.cleanup()

        logger.info("ModelManager cleanup complete")

    def print_summary(self):
        """Print human-readable summary of model manager state"""
        state = self.get_current_state()

        print("=" * 60)
        print("Model Manager Summary")
        print("=" * 60)

        print(f"\nActive Model: {state.active_model or 'None'}")
        print(f"Model Type:   {state.model_type or 'None'}")
        print(f"VRAM Usage:   {state.vram_usage_gb:.2f} GB")

        if state.loaded_at:
            uptime = time.time() - state.loaded_at
            print(f"Loaded:       {uptime:.0f}s ago")

        print("\nService Status:")
        self.service_controller.print_status()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Initializing Model Manager...")
    manager = ModelManager()

    print("\nCurrent State:")
    manager.print_summary()

    print("\n" + "=" * 60)
    print("Example Usage:")
    print("=" * 60)

    print("""
# Use LLM
with manager.use_llm(model="qwen-14b"):
    # LLM is active, call API
    response = llm_client.chat(messages=[...])

# Switch to SDXL
with manager.use_sdxl() as pipeline:
    # SDXL is loaded, LLM is stopped
    image = pipeline.generate(prompt="...")

# Use TTS (lightweight)
with manager.use_tts() as tts:
    # TTS loaded
    audio = tts.synthesize(text="Hello")
    """)

    print("\nCleaning up...")
    manager.cleanup()
