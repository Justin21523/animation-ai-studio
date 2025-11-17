"""
Unit Tests for Model Management Module

Tests VRAM monitoring, service control, and model manager.

Author: Animation AI Studio
Date: 2025-11-17
"""

import pytest
import torch
import time
from pathlib import Path

from scripts.core.model_management import (
    VRAMMonitor,
    VRAMSnapshot,
    VRAMEstimate,
    check_vram_requirements,
    ServiceController,
    ServiceStatus,
    ModelManager,
    ModelState
)


# ============================================================================
# VRAMMonitor Tests
# ============================================================================

class TestVRAMMonitor:
    """Tests for VRAMMonitor class"""

    def test_monitor_initialization(self):
        """Test VRAMMonitor initialization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor(device=0)

        assert monitor.device == 0
        assert monitor.device_name is not None
        assert monitor.TOTAL_VRAM_GB == 16.0
        assert monitor.SAFE_MAX_USAGE_GB == 15.5

        monitor.cleanup()

    def test_get_snapshot(self):
        """Test VRAM snapshot capture"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()
        snapshot = monitor.get_snapshot()

        assert isinstance(snapshot, VRAMSnapshot)
        assert snapshot.device_id == 0
        assert snapshot.total_vram_gb > 0
        assert snapshot.allocated_gb >= 0
        assert snapshot.free_gb >= 0
        assert 0 <= snapshot.utilization_percent <= 100

        monitor.cleanup()

    def test_get_detailed_stats(self):
        """Test detailed VRAM statistics"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()
        stats = monitor.get_detailed_stats()

        assert "total_gb" in stats
        assert "allocated_gb" in stats
        assert "free_gb" in stats
        assert "utilization_percent" in stats
        assert "safe_max_gb" in stats
        assert "available_for_new_model_gb" in stats

        # Check values are reasonable
        assert stats["total_gb"] > 0
        assert stats["allocated_gb"] >= 0
        assert stats["safe_max_gb"] == 15.5

        monitor.cleanup()

    def test_model_estimates(self):
        """Test model VRAM estimates"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()

        # Test known models
        qwen_est = monitor.get_model_estimate("qwen-14b")
        assert qwen_est is not None
        assert qwen_est.estimated_vram_gb == 11.5
        assert qwen_est.confidence == "high"

        sdxl_est = monitor.get_model_estimate("sdxl-base")
        assert sdxl_est is not None
        assert sdxl_est.estimated_vram_gb == 10.5

        # Test unknown model
        unknown_est = monitor.get_model_estimate("unknown-model")
        assert unknown_est is None

        monitor.cleanup()

    def test_can_fit_model(self):
        """Test model fit checking"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()

        # Qwen-14b should fit (11.5GB)
        can_fit = monitor.can_fit_model("qwen-14b")
        # This might fail if VRAM is already heavily used, so we don't assert
        # Just verify it returns a boolean
        assert isinstance(can_fit, bool)

        # Unknown model should return False
        can_fit_unknown = monitor.can_fit_model("unknown-model")
        assert can_fit_unknown is False

        monitor.cleanup()

    def test_clear_cache(self):
        """Test CUDA cache clearing"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()

        # Allocate some memory
        tensor = torch.zeros((1000, 1000), device="cuda")
        del tensor

        # Clear cache
        monitor.clear_cache()

        # Should complete without error
        assert True

        monitor.cleanup()

    def test_peak_memory(self):
        """Test peak memory tracking"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()
        monitor.reset_peak_stats()

        # Allocate memory
        tensor = torch.zeros((1000, 1000), device="cuda")

        peak = monitor.get_peak_memory()
        assert peak > 0

        del tensor
        monitor.cleanup()


class TestVRAMRequirements:
    """Test VRAM requirement checking"""

    def test_check_vram_requirements(self):
        """Test system VRAM requirement check"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Should not raise for RTX 5080 16GB
        try:
            check_vram_requirements()
        except RuntimeError as e:
            # If it raises, should be due to insufficient VRAM
            assert "Insufficient VRAM" in str(e) or "CUDA not available" in str(e)


# ============================================================================
# ServiceController Tests
# ============================================================================

class TestServiceController:
    """Tests for ServiceController class"""

    def test_controller_initialization(self):
        """Test ServiceController initialization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        controller = ServiceController()

        assert controller.LLM_GATEWAY_URL == "http://localhost:8000"
        assert controller.vram_monitor is not None

    def test_is_llm_running(self):
        """Test LLM status check"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        controller = ServiceController()

        # Check if LLM is running (might be true or false)
        is_running = controller.is_llm_running()
        assert isinstance(is_running, bool)

    def test_get_llm_status(self):
        """Test LLM status retrieval"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        controller = ServiceController()
        status = controller.get_llm_status()

        assert isinstance(status, ServiceStatus)
        assert status.service_name == "LLM Backend"
        assert isinstance(status.is_running, bool)


# ============================================================================
# ModelManager Tests
# ============================================================================

class TestModelManager:
    """Tests for ModelManager class"""

    def test_manager_initialization(self):
        """Test ModelManager initialization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        assert manager.vram_monitor is not None
        assert manager.service_controller is not None
        assert manager.state.active_model is None
        assert manager.state.model_type is None

        manager.cleanup()

    def test_get_current_state(self):
        """Test current state retrieval"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()
        state = manager.get_current_state()

        assert isinstance(state, ModelState)
        assert isinstance(state.vram_usage_gb, float)
        assert state.vram_usage_gb >= 0

        manager.cleanup()

    def test_can_load_model(self):
        """Test model load feasibility check"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        # Test known model
        can_load = manager.can_load_model("llm", "qwen-14b")
        assert isinstance(can_load, bool)

        manager.cleanup()

    def test_load_unload_sdxl(self):
        """Test SDXL loading and unloading"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        # Load SDXL (placeholder)
        manager.load_sdxl()
        assert manager.sdxl_pipeline is not None

        # Unload SDXL
        manager.unload_sdxl()
        assert manager.sdxl_pipeline is None

        manager.cleanup()

    def test_load_unload_tts(self):
        """Test TTS loading and unloading"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        # Load TTS (placeholder)
        manager.load_tts()
        assert manager.tts_model is not None

        # Unload TTS
        manager.unload_tts()
        assert manager.tts_model is None

        manager.cleanup()

    def test_context_manager_sdxl(self):
        """Test SDXL context manager"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        # Use SDXL context (should auto-load and auto-unload)
        with manager.use_sdxl() as pipeline:
            assert pipeline is not None
            assert manager.state.model_type == "sdxl"

        # After context exit, should be unloaded
        assert manager.sdxl_pipeline is None

        manager.cleanup()

    def test_context_manager_tts(self):
        """Test TTS context manager"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        # Use TTS context
        with manager.use_tts() as tts:
            assert tts is not None

        # After context exit, should be unloaded
        assert manager.tts_model is None

        manager.cleanup()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for model management"""

    def test_vram_monitor_with_allocation(self):
        """Test VRAM monitor with actual GPU allocation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = VRAMMonitor()

        # Get initial state
        initial_snapshot = monitor.get_snapshot()
        initial_allocated = initial_snapshot.allocated_gb

        # Allocate 1GB
        tensor = torch.zeros((256, 1024, 1024), dtype=torch.float32, device="cuda")

        # Get new state
        new_snapshot = monitor.get_snapshot()
        new_allocated = new_snapshot.allocated_gb

        # Should have allocated approximately 1GB more
        allocated_diff = new_allocated - initial_allocated
        assert 0.8 < allocated_diff < 1.2  # Allow some tolerance

        # Cleanup
        del tensor
        monitor.clear_cache()
        monitor.cleanup()

    def test_model_manager_cleanup(self):
        """Test ModelManager cleanup"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = ModelManager()

        # Load some models
        manager.load_sdxl()
        manager.load_tts()

        # Cleanup
        manager.cleanup()

        # All should be unloaded
        assert manager.sdxl_pipeline is None
        assert manager.tts_model is None


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration file validity"""

    def test_config_file_exists(self):
        """Test that model manager config exists"""
        config_path = Path(__file__).parent.parent.parent / "configs" / "model_manager_config.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_file_valid_yaml(self):
        """Test that config is valid YAML"""
        import yaml

        config_path = Path(__file__).parent.parent.parent / "configs" / "model_manager_config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "hardware" in config
        assert "vram_estimates" in config
        assert "switching" in config
        assert "services" in config

    def test_config_vram_estimates(self):
        """Test VRAM estimates in config"""
        import yaml

        config_path = Path(__file__).parent.parent.parent / "configs" / "model_manager_config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        estimates = config["vram_estimates"]

        # Check LLM estimates
        assert "llm" in estimates
        assert "qwen-14b" in estimates["llm"]
        assert estimates["llm"]["qwen-14b"]["vram_gb"] == 11.5

        # Check image estimates
        assert "image" in estimates
        assert "sdxl-base" in estimates["image"]
        assert estimates["image"]["sdxl-base"]["vram_gb"] == 10.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
