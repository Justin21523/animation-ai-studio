"""
Performance Benchmarking for Image Generation

Measure generation speed, VRAM usage, and quality metrics.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image import (
    SDXLPipelineManager,
    CharacterGenerator
)


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    test_name: str
    total_time: float
    generation_time: float
    loading_time: float
    peak_vram_gb: float
    avg_vram_gb: float
    steps: int
    resolution: tuple
    success: bool
    error_message: Optional[str] = None


class ImageGenerationBenchmark:
    """
    Benchmark image generation performance

    Metrics:
    - Generation speed (seconds)
    - VRAM usage (peak, average)
    - Quality-vs-speed tradeoffs
    - LoRA overhead
    - ControlNet overhead
    """

    def __init__(
        self,
        sdxl_model_path: str,
        device: str = "cuda",
        output_dir: str = "outputs/benchmarks"
    ):
        """
        Initialize benchmark

        Args:
            sdxl_model_path: Path to SDXL model
            device: Device to use
            output_dir: Output directory for results
        """
        self.sdxl_model_path = Path(sdxl_model_path)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BenchmarkResult] = []

        logger.info("Benchmark initialized")

    def measure_vram(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1e9
        return 0.0

    def benchmark_sdxl_base(
        self,
        quality_preset: str = "standard",
        num_runs: int = 3
    ) -> BenchmarkResult:
        """
        Benchmark SDXL base generation

        Args:
            quality_preset: Quality preset to use
            num_runs: Number of runs for averaging

        Returns:
            BenchmarkResult
        """
        logger.info(f"\nBenchmarking SDXL base ({quality_preset})...")

        try:
            # Initialize pipeline
            start_load = time.time()

            pipeline_manager = SDXLPipelineManager(
                model_path=str(self.sdxl_model_path),
                device=self.device,
                dtype=torch.float16
            )

            pipeline = pipeline_manager.load_pipeline()
            loading_time = time.time() - start_load

            logger.info(f"Loading time: {loading_time:.2f}s")

            # Warmup run
            logger.info("Warmup run...")
            _ = pipeline_manager.generate(
                prompt="test warmup",
                quality_preset=quality_preset,
                seed=42
            )

            # Benchmark runs
            generation_times = []
            vram_measurements = []

            for i in range(num_runs):
                torch.cuda.reset_peak_memory_stats()

                start_gen = time.time()

                _ = pipeline_manager.generate(
                    prompt="a young boy with brown hair, pixar style, 3d animation",
                    negative_prompt="blurry, low quality",
                    quality_preset=quality_preset,
                    seed=42 + i
                )

                gen_time = time.time() - start_gen
                generation_times.append(gen_time)

                peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
                vram_measurements.append(peak_vram)

                logger.info(f"Run {i+1}/{num_runs}: {gen_time:.2f}s, VRAM: {peak_vram:.2f}GB")

            # Get config
            config = pipeline_manager.quality_presets[quality_preset]

            result = BenchmarkResult(
                test_name=f"sdxl_base_{quality_preset}",
                total_time=loading_time + np.mean(generation_times),
                generation_time=np.mean(generation_times),
                loading_time=loading_time,
                peak_vram_gb=np.max(vram_measurements),
                avg_vram_gb=np.mean(vram_measurements),
                steps=config["steps"],
                resolution=(1024, 1024),
                success=True
            )

            # Cleanup
            pipeline_manager.cleanup()

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result = BenchmarkResult(
                test_name=f"sdxl_base_{quality_preset}",
                total_time=0.0,
                generation_time=0.0,
                loading_time=0.0,
                peak_vram_gb=0.0,
                avg_vram_gb=0.0,
                steps=0,
                resolution=(0, 0),
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            return result

    def benchmark_quality_presets(self, num_runs: int = 3):
        """Benchmark all quality presets"""
        logger.info("\n" + "=" * 60)
        logger.info("Benchmarking Quality Presets")
        logger.info("=" * 60)

        presets = ["draft", "standard", "high", "ultra"]

        for preset in presets:
            self.benchmark_sdxl_base(quality_preset=preset, num_runs=num_runs)

    def print_summary(self):
        """Print benchmark summary"""
        logger.info("\n" + "=" * 60)
        logger.info("Benchmark Summary")
        logger.info("=" * 60)

        if not self.results:
            logger.info("No results to display")
            return

        # Table header
        print(f"\n{'Test':<25} {'Gen Time':<12} {'VRAM Peak':<12} {'Steps':<8} {'Status'}")
        print("-" * 70)

        for result in self.results:
            if result.success:
                print(
                    f"{result.test_name:<25} "
                    f"{result.generation_time:>8.2f}s    "
                    f"{result.peak_vram_gb:>8.2f}GB    "
                    f"{result.steps:>5}    "
                    f"✓"
                )
            else:
                print(f"{result.test_name:<25} {'FAILED':<12} {'-':<12} {'-':<8} ✗")

        # Performance targets
        print("\n" + "=" * 60)
        print("Performance Targets (RTX 5080 16GB)")
        print("=" * 60)
        print("SDXL base:         < 15s (30 steps @ 1024x1024)")
        print("SDXL + LoRA:       < 20s")
        print("SDXL + ControlNet: < 25s")
        print("Peak VRAM:         < 15.5GB (safe for 16GB)")

    def save_results(self, output_path: Optional[str] = None):
        """Save results to JSON"""
        if output_path is None:
            timestamp = int(time.time())
            output_path = self.output_dir / f"benchmark_results_{timestamp}.json"

        output_path = Path(output_path)

        results_data = {
            "timestamp": time.time(),
            "device": self.device,
            "sdxl_model": str(self.sdxl_model_path),
            "results": [
                {
                    "test_name": r.test_name,
                    "total_time": r.total_time,
                    "generation_time": r.generation_time,
                    "loading_time": r.loading_time,
                    "peak_vram_gb": r.peak_vram_gb,
                    "avg_vram_gb": r.avg_vram_gb,
                    "steps": r.steps,
                    "resolution": r.resolution,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"\n✓ Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Image Generation Benchmark")
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
        default="outputs/benchmarks",
        help="Output directory"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per test"
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check prerequisites
    if not Path(args.sdxl_model).exists():
        logger.error(f"SDXL model not found: {args.sdxl_model}")
        logger.info("Download from: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0")
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)

    # Run benchmark
    benchmark = ImageGenerationBenchmark(
        sdxl_model_path=args.sdxl_model,
        device=args.device,
        output_dir=args.output_dir
    )

    benchmark.benchmark_quality_presets(num_runs=args.num_runs)
    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()
