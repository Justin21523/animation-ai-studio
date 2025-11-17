"""
Integration Tests for Image Generation Module

Tests complete image generation pipeline with real models.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image import (
    SDXLPipelineManager,
    LoRAManager,
    ControlNetPipelineManager,
    CharacterGenerator,
    CharacterConsistencyChecker,
    BatchImageGenerator
)


logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """
    Integration test suite for image generation

    Tests:
    1. SDXL base generation
    2. SDXL + LoRA generation
    3. SDXL + ControlNet generation
    4. Character consistency checking
    5. Batch generation with filtering
    """

    def __init__(
        self,
        sdxl_model_path: str,
        device: str = "cuda",
        output_dir: str = "outputs/integration_tests"
    ):
        """
        Initialize integration test suite

        Args:
            sdxl_model_path: Path to SDXL model
            device: Device to use
            output_dir: Output directory for test results
        """
        self.sdxl_model_path = Path(sdxl_model_path)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Integration Test Suite initialized")
        logger.info(f"SDXL model: {self.sdxl_model_path}")
        logger.info(f"Device: {self.device}")

    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met"""
        logger.info("Checking prerequisites...")

        # Check CUDA
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA not available but device='cuda'")
            return False

        # Check SDXL model
        if not self.sdxl_model_path.exists():
            logger.error(f"SDXL model not found: {self.sdxl_model_path}")
            logger.info("Download from: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0")
            return False

        # Check VRAM
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Total VRAM: {total_vram:.1f} GB")

            if total_vram < 14.0:
                logger.warning(f"Low VRAM ({total_vram:.1f}GB). Recommend 16GB for SDXL.")

        logger.info("✓ Prerequisites check passed")
        return True

    def test_sdxl_base(self) -> bool:
        """Test SDXL base generation"""
        logger.info("\n" + "=" * 60)
        logger.info("Test 1: SDXL Base Generation")
        logger.info("=" * 60)

        try:
            # Initialize pipeline
            pipeline_manager = SDXLPipelineManager(
                model_path=str(self.sdxl_model_path),
                device=self.device,
                dtype=torch.float16
            )

            logger.info("Loading SDXL pipeline...")
            pipeline = pipeline_manager.load_pipeline()

            # Generate image
            logger.info("Generating image...")
            prompt = "a young boy with brown hair and green eyes, pixar style, 3d animation, high quality"
            negative_prompt = "blurry, low quality, distorted"

            output_path = self.output_dir / "test_sdxl_base.png"

            image = pipeline_manager.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                quality_preset="standard",
                seed=42,
                output_path=str(output_path)
            )

            logger.info(f"✓ Generated: {output_path}")
            logger.info(f"VRAM usage: {pipeline_manager.get_vram_usage():.2f} GB")

            # Cleanup
            pipeline_manager.cleanup()

            return True

        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return False

    def test_lora_loading(self) -> bool:
        """Test LoRA loading (without actual generation)"""
        logger.info("\n" + "=" * 60)
        logger.info("Test 2: LoRA Manager")
        logger.info("=" * 60)

        try:
            # Initialize LoRA manager
            lora_manager = LoRAManager(
                registry_path="configs/generation/lora_registry.yaml"
            )

            logger.info(f"Loaded registry: {len(lora_manager.registry.loras)} LoRAs")

            # Check if any character LoRAs are registered
            luca_lora = lora_manager.registry.get_character_lora("luca")
            if luca_lora:
                logger.info(f"✓ Found Luca LoRA: {luca_lora.name}")
            else:
                logger.warning("No Luca LoRA found (expected - LoRAs not yet trained)")

            # List all registered LoRAs
            for lora_name in lora_manager.registry.loras.keys():
                logger.info(f"  - {lora_name}")

            logger.info("✓ LoRA manager test passed")
            return True

        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return False

    def test_character_generator(self) -> bool:
        """Test character generator (high-level API)"""
        logger.info("\n" + "=" * 60)
        logger.info("Test 3: Character Generator")
        logger.info("=" * 60)

        try:
            # Initialize generator
            generator = CharacterGenerator(
                sdxl_model_path=str(self.sdxl_model_path),
                lora_registry_path="configs/generation/lora_registry.yaml",
                device=self.device
            )

            logger.info("Generating character image...")

            output_path = self.output_dir / "test_character.png"

            # Note: This will work without LoRA, using base SDXL
            image = generator.generate_character(
                character="luca",  # Will use base model if LoRA not found
                scene_description="standing on italian beach, smiling",
                style="pixar_3d",
                quality_preset="standard",
                seed=123,
                output_path=str(output_path)
            )

            logger.info(f"✓ Generated: {output_path}")

            # Cleanup
            generator.cleanup()

            return True

        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return False

    def test_consistency_checker(self) -> bool:
        """Test consistency checker"""
        logger.info("\n" + "=" * 60)
        logger.info("Test 4: Consistency Checker")
        logger.info("=" * 60)

        try:
            # Initialize checker
            checker = CharacterConsistencyChecker(device=self.device)

            logger.info("Consistency checker initialized")
            logger.info(f"Model: {checker.model_name}")

            # Note: Actual consistency testing requires reference images
            logger.info("✓ Consistency checker test passed (initialization)")

            return True

        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return False

    def run_all_tests(self) -> dict:
        """Run all integration tests"""
        logger.info("\n" + "=" * 60)
        logger.info("Running Integration Tests")
        logger.info("=" * 60)

        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting tests.")
            return {"passed": 0, "failed": 1, "total": 1}

        results = {
            "sdxl_base": self.test_sdxl_base(),
            "lora_loading": self.test_lora_loading(),
            "character_generator": self.test_character_generator(),
            "consistency_checker": self.test_consistency_checker()
        }

        # Summary
        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed

        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"Passed: {passed}/{len(results)}")
        logger.info(f"Failed: {failed}/{len(results)}")

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"  {test_name}: {status}")

        return {
            "passed": passed,
            "failed": failed,
            "total": len(results),
            "results": results
        }


def main():
    parser = argparse.ArgumentParser(description="Image Generation Integration Tests")
    parser.add_argument(
        "--sdxl-model",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/stable-diffusion-xl-base-1.0",
        help="Path to SDXL model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/integration_tests",
        help="Output directory"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    suite = IntegrationTestSuite(
        sdxl_model_path=args.sdxl_model,
        device=args.device,
        output_dir=args.output_dir
    )

    results = suite.run_all_tests()

    # Exit code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
