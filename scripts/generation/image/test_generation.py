"""
Test Script for Image Generation Module

Tests SDXL pipeline, LoRA loading, and character generation.

NOTE: This script requires:
1. SDXL base model downloaded to AI Warehouse
2. Conda environment: ai_env
3. RTX 5080 16GB VRAM available

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.generation.image import (
    SDXLPipelineManager,
    LoRAManager,
    CharacterGenerator
)


def test_sdxl_basic():
    """Test basic SDXL generation"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic SDXL Generation")
    print("=" * 60)

    model_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/stable-diffusion-xl-base-1.0"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ SDXL model not found at: {model_path}")
        print("Please download SDXL base model first.")
        return False

    try:
        # Initialize pipeline
        pipeline_manager = SDXLPipelineManager(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_sdpa=True
        )

        # Load pipeline
        pipeline_manager.load_pipeline()

        # Check VRAM
        vram = pipeline_manager.get_vram_usage()
        print(f"VRAM allocated: {vram.get('allocated', 0):.2f}GB / {vram.get('total', 0):.2f}GB")

        # Generate test image
        print("\nGenerating test image...")
        image = pipeline_manager.generate(
            prompt="a cute cartoon character, 3d animation style, high quality",
            negative_prompt="blurry, low quality",
            quality_preset="draft",  # Use draft for fast testing
            seed=42,
            output_path="outputs/test_sdxl_basic.png"
        )

        # Unload pipeline
        pipeline_manager.unload_pipeline()

        print("✅ Test 1 PASSED: Basic SDXL generation successful")
        return True

    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_registry():
    """Test LoRA registry loading"""
    print("\n" + "=" * 60)
    print("TEST 2: LoRA Registry")
    print("=" * 60)

    try:
        from scripts.generation.image.lora_manager import LoRARegistry

        registry_path = "configs/generation/lora_registry.yaml"

        # Check if registry exists
        if not Path(registry_path).exists():
            print(f"❌ LoRA registry not found at: {registry_path}")
            return False

        # Load registry
        registry = LoRARegistry(registry_path)

        print(f"Loaded {len(registry.loras)} LoRAs:")
        for name, lora in registry.loras.items():
            print(f"  - {name} ({lora.type}): {lora.description}")

        # Test character LoRA lookup
        luca_lora = registry.get_character_lora("luca")
        if luca_lora:
            print(f"\nFound Luca LoRA:")
            print(f"  Trigger words: {luca_lora.trigger_words}")
            print(f"  Recommended weight: {luca_lora.recommended_weight}")

        print("✅ Test 2 PASSED: LoRA registry loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_character_generator():
    """Test character generator (dry run - no actual generation)"""
    print("\n" + "=" * 60)
    print("TEST 3: Character Generator (Configuration Test)")
    print("=" * 60)

    try:
        # Initialize generator (without loading models)
        generator = CharacterGenerator()

        print("✓ Character generator initialized")
        print(f"✓ SDXL config loaded: {len(generator.sdxl_config)} keys")
        print(f"✓ LoRA registry loaded: {len(generator.lora_registry.loras)} LoRAs")

        # Test prompt building
        prompt = generator._build_prompt(
            character="luca",
            scene_description="running on the beach, excited expression",
            style="pixar_3d",
            include_trigger_words=True
        )

        print(f"\nGenerated prompt:")
        print(f"  {prompt}")

        # Test negative prompt
        negative_prompt = generator._get_negative_prompt("character")
        print(f"\nNegative prompt:")
        print(f"  {negative_prompt}")

        print("\n✅ Test 3 PASSED: Character generator configuration valid")
        return True

    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vram_check():
    """Check VRAM availability"""
    print("\n" + "=" * 60)
    print("TEST 4: VRAM Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    try:
        # Get GPU info
        device_count = torch.cuda.device_count()
        print(f"CUDA devices: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / 1e9
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total VRAM: {total_vram:.2f}GB")
            print(f"  Compute capability: {props.major}.{props.minor}")

            # Check if sufficient for SDXL
            if total_vram < 14.0:
                print(f"  ⚠️  WARNING: VRAM may be insufficient for SDXL (recommended: 14GB+)")
            else:
                print(f"  ✓ VRAM sufficient for SDXL")

        # Check PyTorch version
        print(f"\nPyTorch version: {torch.__version__}")
        if torch.__version__.startswith("2.7") or torch.__version__.startswith("2.8"):
            print("  ✓ PyTorch 2.7.0+ detected (SDPA support)")
        else:
            print(f"  ⚠️  WARNING: PyTorch 2.7.0+ recommended (current: {torch.__version__})")

        print("\n✅ Test 4 PASSED: VRAM check complete")
        return True

    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Image Generation Module - Test Suite")
    print("=" * 60)

    tests = [
        ("VRAM Check", test_vram_check),
        ("LoRA Registry", test_lora_registry),
        ("Character Generator Config", test_character_generator),
        # ("Basic SDXL Generation", test_sdxl_basic),  # Commented out - requires model download
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Image Generation Module")
    parser.add_argument("--test", choices=["all", "vram", "registry", "generator", "sdxl"], default="all",
                        help="Which test to run")

    args = parser.parse_args()

    if args.test == "all":
        success = run_all_tests()
    elif args.test == "vram":
        success = test_vram_check()
    elif args.test == "registry":
        success = test_lora_registry()
    elif args.test == "generator":
        success = test_character_generator()
    elif args.test == "sdxl":
        success = test_sdxl_basic()

    sys.exit(0 if success else 1)
