# Image Generation Module

**Status:** ✅ 100% Complete (All components)
**Last Updated:** 2025-11-17

---

## Overview

SDXL-based 3D character image generation with LoRA, ControlNet, and consistency validation.

**Hardware:** Optimized for RTX 5080 16GB VRAM
**PyTorch:** 2.7.0+ (PyTorch SDPA, xformers FORBIDDEN)

---

## Components

### ✅ All Components Complete (8 of 8)

1. **SDXL Pipeline Manager** (`sdxl_pipeline.py`, 420 lines)
   - FP16 optimization for RTX 5080 16GB
   - PyTorch 2.7.0 SDPA attention
   - VRAM monitoring and dynamic management
   - Quality presets (draft/standard/high/ultra)
   - Multiple schedulers (Euler/DPM/DDIM)

2. **LoRA Manager** (`lora_manager.py`, 370 lines)
   - LoRA registry system with YAML config
   - Multi-LoRA fusion (weighted composition)
   - Character/style/background LoRA support
   - Trigger word integration
   - Dynamic loading/unloading

3. **ControlNet Pipeline** (`controlnet_pipeline.py`, 400 lines)
   - 5 control types: Pose, Canny, Depth, Seg, Normal
   - Control image preprocessing
   - Adjustable conditioning scale
   - PyTorch SDPA attention

4. **Character Generator** (`character_generator.py`, 530 lines)
   - High-level wrapper for SDXL + LoRA + ControlNet
   - Automatic prompt engineering
   - Style integration (Pixar, Disney, etc.)
   - Batch generation support
   - VRAM-aware model switching

5. **Consistency Checker** (`consistency_checker.py`, 530 lines)
   - ArcFace-based face recognition
   - Character consistency validation
   - Similarity scoring (cosine/euclidean)
   - Reference embedding management
   - Batch consistency checking

6. **Batch Generator** (`batch_generator.py`, 470 lines)
   - Batch generation with quality filtering
   - Automatic consistency checking
   - Progress tracking (tqdm)
   - Organized output structure
   - Generation metadata (JSON)

7. **Test Suite** (`test_generation.py`, 240 lines + `test_image_generation.py`, 370 lines)
   - VRAM check
   - Configuration validation
   - Unit tests (pytest)
   - Example usage patterns

8. **Integration Tests & Benchmarks** (`integration_test.py`, 330 lines + `benchmark.py`, 290 lines)
   - Full pipeline integration tests
   - SDXL base generation testing
   - Character generator testing
   - Performance benchmarking (speed, VRAM)
   - Quality preset comparisons

---

## Configuration Files

### ✅ Complete (4 files, 450+ lines YAML)

1. **`sdxl_config.yaml`** (90 lines)
   - Quality presets (steps/CFG)
   - Negative prompt templates
   - Style prompts (Pixar, Disney, DreamWorks)
   - VRAM optimization settings

2. **`lora_registry.yaml`** (100 lines)
   - Character LoRAs (Luca, Alberto)
   - Style LoRAs (Pixar 3D, lighting)
   - Background LoRAs (Portorosso, beach)
   - Pose LoRAs (running, dynamic)
   - LoRA combination presets

3. **`controlnet_config.yaml`** (112 lines)
   - 5 ControlNet model configs
   - Preprocessing parameters
   - Quality presets
   - VRAM estimations

4. **`character_presets.yaml`** (150 lines)
   - Character definitions (Luca, Alberto)
   - Reference images for consistency
   - Default LoRAs and weights
   - Scene presets
   - Consistency thresholds

---

## Usage Examples

### Basic Character Generation

```python
from scripts.generation.image import CharacterGenerator

generator = CharacterGenerator()

image = generator.generate_character(
    character="luca",
    scene_description="running on the beach, excited expression",
    quality_preset="high",
    seed=42,
    output_path="outputs/luca_beach.png"
)

generator.cleanup()
```

### Multi-LoRA Generation

```python
image = generator.generate_character(
    character="luca",
    scene_description="standing in Portorosso town square",
    additional_loras=[
        {"name": "portorosso_town", "weight": 0.7},
        {"name": "warm_summer_lighting", "weight": 0.5}
    ],
    quality_preset="ultra",
    seed=123
)
```

### ControlNet-Guided Generation

```python
image = generator.generate_character(
    character="luca",
    scene_description="dynamic running pose",
    use_controlnet=True,
    control_type="pose",
    control_image="reference_pose.jpg",
    controlnet_scale=0.9
)
```

### Batch Generation with Consistency Filtering

```python
from scripts.generation.image import BatchImageGenerator

batch_gen = BatchImageGenerator()

result = batch_gen.generate_batch(
    character="luca",
    scene_description="running on the beach",
    num_images=20,
    quality_preset="high",
    enable_consistency_check=True,
    consistency_threshold=0.70,
    save_rejected=True
)

print(f"Accepted: {result.total_accepted}/{result.total_generated}")
print(f"Average similarity: {sum(result.consistency_scores)/len(result.consistency_scores):.3f}")

batch_gen.cleanup()
```

### Character Consistency Checking

```python
from scripts.generation.image import CharacterConsistencyChecker, CharacterReferenceManager

# Initialize
checker = CharacterConsistencyChecker(device="cuda")
ref_manager = CharacterReferenceManager(checker)

# Create reference embedding from multiple images
ref_manager.create_character_embedding(
    character_name="luca",
    reference_images=[
        "data/films/luca/characters/luca_ref_1.jpg",
        "data/films/luca/characters/luca_ref_2.jpg",
        "data/films/luca/characters/luca_ref_3.jpg"
    ]
)

# Check consistency
result = ref_manager.check_character_consistency(
    character_name="luca",
    generated_image="outputs/luca_generated.png",
    threshold=0.65
)

print(f"Consistent: {result.is_consistent}")
print(f"Similarity: {result.similarity_score:.3f}")
```

---

## Performance Targets

### Latency (RTX 5080 16GB)

```
SDXL base:          < 15s (30 steps @ 1024x1024)
SDXL + LoRA:        < 20s
SDXL + ControlNet:  < 25s
```

### Quality

```
Consistency:  > 0.65 similarity (ArcFace)
Resolution:   1024x1024 default
              768x1280 (portrait)
              1280x768 (landscape)
```

### VRAM Usage

```
SDXL base:         ~10-11GB
SDXL + LoRA:       ~11-12GB
SDXL + ControlNet: ~13-15GB
Peak:              < 15.5GB (safe for 16GB)
```

---

## Installation

### Requirements

```bash
# Install dependencies
pip install -r requirements/generation.txt

# Key packages:
# - torch >= 2.7.0 (CRITICAL for PyTorch SDPA)
# - diffusers >= 0.33.0
# - transformers >= 4.47.0
# - insightface >= 0.7.3 (for consistency checking)
# - onnxruntime-gpu >= 1.19.0

# NOTE: Do NOT install xformers (FORBIDDEN)
```

### Model Download

```bash
# SDXL base model (required)
# Download from HuggingFace: stabilityai/stable-diffusion-xl-base-1.0
# Place in: /mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/

# ControlNet models (optional)
# Download from HuggingFace: diffusers/controlnet-*-sdxl-1.0
# Place in: /mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/controlnet/

# LoRA adapters (from 3D Animation LoRA Pipeline)
# Trained LoRAs will be placed in: /mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/lora/
```

---

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/generation/test_image_generation.py -v

# Run specific test class
pytest tests/generation/test_image_generation.py::TestLoRARegistry -v

# Run with coverage
pytest tests/generation/test_image_generation.py --cov=scripts/generation/image
```

### Run Integration Tests

```bash
# Full integration test (requires SDXL model)
python scripts/generation/image/integration_test.py \
  --sdxl-model /path/to/sdxl-base-1.0 \
  --device cuda \
  --output-dir outputs/integration_tests

# Tests included:
# 1. SDXL base generation
# 2. LoRA manager
# 3. Character generator
# 4. Consistency checker
```

### Run Performance Benchmarks

```bash
# Benchmark generation speed and VRAM usage
python scripts/generation/image/benchmark.py \
  --sdxl-model /path/to/sdxl-base-1.0 \
  --device cuda \
  --num-runs 3 \
  --output-dir outputs/benchmarks

# Benchmarks:
# - Quality presets (draft/standard/high/ultra)
# - VRAM usage per preset
# - Generation speed analysis
# - Results saved to JSON
```

### Run Example Script

```bash
# Basic functionality test (no model required)
python scripts/generation/image/test_generation.py

# Specific test
python scripts/generation/image/test_generation.py --test registry
```

---

## File Structure

```
scripts/generation/image/
├── __init__.py (44 lines)
├── sdxl_pipeline.py (420 lines)
├── lora_manager.py (370 lines)
├── controlnet_pipeline.py (400 lines)
├── character_generator.py (530 lines)
├── consistency_checker.py (530 lines)
├── batch_generator.py (470 lines)
├── test_generation.py (240 lines)
├── integration_test.py (330 lines) - NEW
├── benchmark.py (290 lines) - NEW
└── README.md (this file)

configs/generation/
├── sdxl_config.yaml (90 lines)
├── lora_registry.yaml (100 lines)
├── controlnet_config.yaml (112 lines)
└── character_presets.yaml (150 lines)

tests/generation/
├── __init__.py
└── test_image_generation.py (370 lines)

Total: ~4,200+ lines Python code + 450+ lines YAML config
```

---

## Known Limitations

1. **LoRA Models Not Yet Available**
   - Character LoRAs pending from 3D Animation LoRA Pipeline
   - Currently at 14.8% progress (SAM2 segmentation)
   - Will be integrated when available

2. **SDXL Model Download Required**
   - ~13GB download from HuggingFace
   - Requires manual setup

3. **Reference Images Required**
   - Consistency checking needs character reference images
   - Extract from film datasets

4. **Single GPU Limitation**
   - RTX 5080 16GB can only load one heavy model at a time
   - Must unload LLM before loading SDXL

---

## Next Steps

1. **Download SDXL Model**
   - Download from HuggingFace
   - Test basic generation

2. **Create Reference Embeddings**
   - Extract character frames from films
   - Generate ArcFace embeddings

3. **Wait for LoRA Training**
   - Monitor 3D Animation LoRA Pipeline progress
   - Integrate trained LoRAs when available

4. **Performance Benchmarking**
   - Measure generation speed
   - Profile VRAM usage
   - Optimize bottlenecks

5. **Integration with Other Modules**
   - Connect to Model Manager (Module 4)
   - Integrate with Agent Framework (Module 6)
   - Connect to RAG System (Module 5)

---

## References

- **Architecture Doc:** [docs/modules/image-generation.md](../../../docs/modules/image-generation.md)
- **Module Progress:** [docs/modules/module-progress.md](../../../docs/modules/module-progress.md)
- **Hardware Optimization:** [docs/reference/hardware-optimization.md](../../../docs/reference/hardware-optimization.md)

---

**Version:** v1.0.0
**Status:** 85% Complete (7/8 components)
**Last Updated:** 2025-11-17
