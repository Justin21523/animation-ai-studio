# Image Generation Module

**Purpose:** SDXL-based 3D character image generation with LoRA and ControlNet support
**Status:** 🔄 In Progress (15% Complete)
**Hardware:** RTX 5080 16GB VRAM
**VRAM Usage:** 13-15GB (requires LLM shutdown)

---

## 📊 Module Overview

### Core Capabilities

```
User Request → LLM Analysis → Prompt Engineering
                                    ↓
                       ┌────────────────────────┐
                       │  SDXL Pipeline Manager │
                       │  - Base model: SDXL 1.0│
                       │  - LoRA injection      │
                       │  - ControlNet guidance │
                       │  - Character consistency│
                       └────────────────────────┘
                                    ↓
                    Image Generation → Quality Evaluation
```

### Technologies

- **Base Model**: Stable Diffusion XL 1.0
- **Adapters**: LoRA (character, background, style)
- **Guidance**: ControlNet (OpenPose, Depth, Canny)
- **Consistency**: InstantID, ArcFace embeddings
- **Attention**: PyTorch SDPA (xformers FORBIDDEN)
- **Precision**: FP16

---

## 🎯 Functional Requirements

### 1. Basic Generation
- Text-to-image with SDXL base model
- Prompt engineering and negative prompts
- Quality presets (draft, standard, high quality)
- Resolution: 1024x1024 (SDXL native)
- Generation speed: < 20 seconds (30 steps)

### 2. LoRA Character Generation
- Load character-specific LoRA adapters
- Multi-LoRA composition (character + background + style)
- Dynamic weight adjustment (0.0-1.0)
- Trigger word injection
- Character consistency validation

### 3. ControlNet Guided Generation
- **OpenPose**: Pose control from reference images
- **Depth**: Depth map guided composition
- **Canny**: Edge-based structure control
- Multi-ControlNet composition
- Conditioning strength adjustment

### 4. Character Consistency
- Face embedding extraction (ArcFace)
- Similarity scoring between generated and reference
- Automatic filtering (threshold 0.60-0.65)
- Multiple reference embeddings per character

### 5. Batch Processing
- Sequential image generation
- Consistent parameters across batch
- Progress tracking
- Quality filtering

---

## 🔧 Technical Architecture

### Component Structure

```
scripts/generation/image/
├── __init__.py
├── sdxl_pipeline.py           # SDXL base pipeline manager
├── lora_loader.py             # LoRA adapter loading
├── controlnet_generator.py    # ControlNet integration
├── character_generator.py     # Character-specific generation
├── consistency_checker.py     # Character consistency validation
└── batch_generator.py         # Batch processing pipeline

configs/generation/
├── sdxl_config.yaml           # SDXL configuration
├── controlnet_config.yaml     # ControlNet settings
├── lora_registry.yaml         # Available LoRAs
└── character_presets.yaml     # Per-character generation settings
```

### Core Classes

#### SDXLPipelineManager
```python
class SDXLPipelineManager:
    """Manages SDXL base model loading and generation"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_sdpa: bool = True  # PyTorch SDPA required
    )

    def load_pipeline(self) -> StableDiffusionXLPipeline:
        """Load SDXL with PyTorch SDPA attention"""

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 30,
        cfg_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: int = None
    ) -> PIL.Image:
        """Generate image from prompt"""

    def unload_pipeline(self):
        """Free VRAM"""

    def get_vram_usage(self) -> float:
        """Current VRAM consumption"""
```

#### LoRALoader
```python
class LoRALoader:
    """Manages LoRA adapter loading and composition"""

    def __init__(self, pipeline: StableDiffusionXLPipeline)

    def load_lora(
        self,
        lora_path: str,
        adapter_name: str,
        weight: float = 0.8
    ):
        """Load single LoRA adapter"""

    def load_multiple_loras(
        self,
        loras: List[Dict[str, Any]]  # [{"path": "...", "name": "...", "weight": 0.8}]
    ):
        """Load and compose multiple LoRAs"""

    def set_lora_weight(self, adapter_name: str, weight: float):
        """Adjust LoRA influence"""

    def unload_lora(self, adapter_name: str):
        """Remove specific LoRA"""

    def unload_all_loras(self):
        """Remove all LoRAs"""
```

#### ControlNetGenerator
```python
class ControlNetGenerator:
    """ControlNet-guided image generation"""

    def __init__(self, base_pipeline: StableDiffusionXLPipeline)

    def generate_with_pose(
        self,
        reference_image: Union[str, PIL.Image],
        prompt: str,
        character_lora: str = None,
        controlnet_strength: float = 1.0,
        **generation_kwargs
    ) -> PIL.Image:
        """Generate with OpenPose control"""

    def generate_with_depth(
        self,
        reference_image: Union[str, PIL.Image],
        prompt: str,
        **kwargs
    ) -> PIL.Image:
        """Generate with depth map control"""

    def generate_with_canny(
        self,
        reference_image: Union[str, PIL.Image],
        prompt: str,
        **kwargs
    ) -> PIL.Image:
        """Generate with edge control"""

    def generate_multi_controlnet(
        self,
        controls: List[Dict],  # [{"type": "pose", "image": "...", "strength": 0.8}]
        prompt: str,
        **kwargs
    ) -> PIL.Image:
        """Multi-ControlNet composition"""
```

#### CharacterConsistencyChecker
```python
class CharacterConsistencyChecker:
    """Validates character identity consistency"""

    def __init__(self, model_name: str = 'buffalo_l')

    def extract_embedding(self, image: Union[str, PIL.Image]) -> np.ndarray:
        """Extract ArcFace embedding"""

    def compute_similarity(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray
    ) -> float:
        """Cosine similarity score"""

    def verify_character_match(
        self,
        reference_image: Union[str, PIL.Image],
        generated_image: Union[str, PIL.Image],
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "is_match": bool,
                "similarity": float,
                "threshold": float
            }
        """

    def filter_consistent_images(
        self,
        reference_image: Union[str, PIL.Image],
        generated_images: List[Union[str, PIL.Image]],
        threshold: float = 0.65
    ) -> List[Union[str, PIL.Image]]:
        """Filter images matching character identity"""
```

#### CharacterGenerator
```python
class CharacterGenerator:
    """High-level character image generation"""

    def __init__(
        self,
        sdxl_manager: SDXLPipelineManager,
        lora_loader: LoRALoader,
        consistency_checker: CharacterConsistencyChecker
    )

    async def generate_character(
        self,
        character: str,  # From character_presets.yaml
        scene_description: str,
        background: str = None,
        pose_reference: Union[str, PIL.Image] = None,
        quality: str = "high",  # draft, standard, high
        num_candidates: int = 1,
        filter_consistency: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Complete character generation pipeline

        Returns:
            {
                "image_path": str,
                "prompt_used": str,
                "loras_used": List[str],
                "consistency_score": float,
                "generation_time": float
            }
        """
```

---

## ⚙️ Configuration System

### sdxl_config.yaml

```yaml
model:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  cache_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/sdxl"
  variant: "fp16"

hardware:
  device: "cuda"
  dtype: "float16"
  vram_optimization: true
  enable_cpu_offload: false
  enable_sequential_cpu_offload: false

attention:
  backend: "sdpa"  # PyTorch SDPA (REQUIRED)
  enable_xformers: false  # FORBIDDEN

generation:
  quality_presets:
    draft:
      steps: 20
      cfg_scale: 7.0
    standard:
      steps: 30
      cfg_scale: 7.5
    high:
      steps: 40
      cfg_scale: 8.0

  default_size: [1024, 1024]
  scheduler: "DPMSolverMultistepScheduler"

safety:
  enable_safety_checker: false
  nsfw_threshold: 0.0
```

### lora_registry.yaml

```yaml
characters:
  luca:
    name: "Luca Paguro"
    lora_path: "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/outputs/loras/luca_character.safetensors"
    default_weight: 0.8
    trigger_words: ["luca", "boy", "brown hair", "green eyes"]

  alberto:
    name: "Alberto Scorfano"
    lora_path: "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/outputs/loras/alberto_character.safetensors"
    default_weight: 0.8
    trigger_words: ["alberto", "italian boy", "curly hair"]

backgrounds:
  portorosso:
    name: "Portorosso Town"
    lora_path: "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/outputs/loras/portorosso_background.safetensors"
    default_weight: 0.6
    trigger_words: ["portorosso", "italian seaside town", "colorful buildings"]

styles:
  pixar_3d:
    name: "Pixar 3D Animation Style"
    lora_path: "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/loras/pixar_style.safetensors"
    default_weight: 0.7
    trigger_words: ["pixar style", "3d animation", "smooth shading"]
```

### character_presets.yaml

```yaml
luca:
  character_name: "Luca Paguro"
  lora_name: "luca"
  lora_weight: 0.85

  base_prompt: "luca, boy, brown hair, green eyes, pixar style, 3d animation, smooth shading, cinematic lighting"
  negative_prompt: "2d, flat, anime, low quality, blurry, distorted"

  reference_embeddings:
    - /mnt/c/AI_LLM_projects/animation-ai-studio/data/films/luca/references/luca_face_1.npy
    - /mnt/c/AI_LLM_projects/animation-ai-studio/data/films/luca/references/luca_face_2.npy

  consistency_threshold: 0.65

  generation_params:
    steps: 35
    cfg_scale: 7.5
    size: [1024, 1024]
    scheduler: "DPMSolverMultistepScheduler"

alberto:
  character_name: "Alberto Scorfano"
  lora_name: "alberto"
  lora_weight: 0.85

  base_prompt: "alberto, italian boy, curly hair, green eyes, pixar style, 3d animation"
  negative_prompt: "2d, flat, anime, low quality"

  consistency_threshold: 0.65
```

### controlnet_config.yaml

```yaml
controlnets:
  openpose:
    model: "thibaud/controlnet-openpose-sdxl-1.0"
    cache_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/controlnet"
    default_strength: 1.0
    preprocessor: "openpose_full"

  depth:
    model: "diffusers/controlnet-depth-sdxl-1.0"
    cache_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/controlnet"
    default_strength: 0.8
    preprocessor: "depth_midas"

  canny:
    model: "diffusers/controlnet-canny-sdxl-1.0"
    cache_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/controlnet"
    default_strength: 0.7
    preprocessor: "canny"

preprocessing:
  openpose:
    detect_hands: true
    detect_face: true

  depth:
    model_type: "DPT_Large"

  canny:
    low_threshold: 100
    high_threshold: 200
```

---

## 📦 Dependencies

### requirements/image_generation.txt

```
# Base diffusion
diffusers>=0.25.0
transformers>=4.37.0
accelerate>=0.26.0
safetensors>=0.4.1

# LoRA support
peft>=0.8.0

# ControlNet
controlnet_aux>=0.0.7
opencv-python>=4.8.0
mediapipe>=0.10.0

# Character consistency
insightface>=0.7.3
onnxruntime-gpu>=1.16.0

# Prompt engineering
compel>=2.0.2

# Image processing
Pillow>=10.0.0
numpy>=1.24.0
```

---

## 🚀 Usage Examples

### Basic SDXL Generation

```python
from scripts.generation.image import SDXLPipelineManager

# Initialize pipeline
manager = SDXLPipelineManager(
    model_path="stabilityai/stable-diffusion-xl-base-1.0",
    device="cuda",
    dtype=torch.float16
)

# Load pipeline
pipeline = manager.load_pipeline()

# Generate image
image = manager.generate(
    prompt="pixar style 3d animation, boy, smooth shading, cinematic lighting",
    negative_prompt="2d, flat, anime, low quality",
    num_steps=30,
    cfg_scale=7.5
)

# Save
image.save("outputs/test/basic_generation.png")

# Cleanup
manager.unload_pipeline()
```

### Character Generation with LoRA

```python
from scripts.generation.image import CharacterGenerator

# Initialize
generator = CharacterGenerator()

# Generate character
result = await generator.generate_character(
    character="luca",
    scene_description="running on the beach, excited expression",
    background="portorosso",
    quality="high",
    output_path="outputs/characters/luca_beach.png"
)

print(f"Generated: {result['image_path']}")
print(f"Consistency score: {result['consistency_score']}")
print(f"Generation time: {result['generation_time']:.2f}s")
```

### ControlNet Pose Generation

```python
from scripts.generation.image import ControlNetGenerator

# Initialize
controlnet_gen = ControlNetGenerator(base_pipeline)

# Generate with pose control
image = controlnet_gen.generate_with_pose(
    reference_image="/path/to/pose_reference.jpg",
    prompt="luca, boy, excited expression, pixar style",
    character_lora="luca",
    controlnet_strength=1.0,
    num_steps=30
)

image.save("outputs/controlnet/luca_pose.png")
```

### Batch Generation with Consistency Filtering

```python
from scripts.generation.image import CharacterGenerator, CharacterConsistencyChecker

generator = CharacterGenerator()
checker = CharacterConsistencyChecker()

# Generate multiple candidates
candidates = []
for i in range(5):
    result = await generator.generate_character(
        character="luca",
        scene_description="smiling, looking at camera",
        quality="standard",
        filter_consistency=False  # We'll filter manually
    )
    candidates.append(result['image_path'])

# Filter consistent images
reference_image = "data/films/luca/references/luca_ref.jpg"
consistent_images = checker.filter_consistent_images(
    reference_image=reference_image,
    generated_images=candidates,
    threshold=0.65
)

print(f"Kept {len(consistent_images)} of {len(candidates)} images")
```

---

## 📈 Performance Targets

### Latency

```yaml
SDXL base generation: < 15 seconds (30 steps)
LoRA generation: < 20 seconds (30 steps)
ControlNet generation: < 25 seconds (30 steps)
Model loading: ~5-8 seconds
Model unloading: ~2-3 seconds
```

### Quality

```yaml
Character consistency: > 0.65 similarity score
Resolution: 1024x1024 (SDXL native)
Steps: 30-35 (quality/speed balance)
```

### VRAM

```yaml
SDXL base: ~10-11GB
+LoRA: ~11-12GB
+ControlNet: ~13-15GB
Peak usage: < 15.5GB (safety margin)
```

---

## 🚨 Known Challenges

### 1. VRAM Constraints
**Issue**: RTX 5080 16GB cannot run LLM + SDXL simultaneously

**Solution**:
- Stop LLM service before loading SDXL (see Model Manager module)
- Dynamic model switching (20-35 seconds)
- Sequential workflow: plan → generate → evaluate

### 2. LoRA Availability
**Issue**: Character LoRAs not yet trained (LoRA pipeline at 14.8%)

**Solution**:
- Build infrastructure with placeholder LoRAs
- Test with public LoRAs from HuggingFace
- Integrate trained LoRAs when pipeline completes

### 3. PyTorch Compatibility
**Issue**: SDXL uses xformers by default, conflicts with vLLM

**Solution**:
- Force PyTorch SDPA: `attn_processor = AttnProcessor2_0()`
- Keep xformers disabled globally
- Thoroughly test compatibility

### 4. Character Consistency
**Issue**: Maintaining identity across poses/scenes

**Solution**:
- Multiple reference embeddings per character
- Lower threshold (0.60-0.65) for flexibility
- Manual curation of best outputs
- Consider InstantID for stronger preservation

---

## ✅ Implementation Checklist

### Phase 1: SDXL Base (Estimated: 3 days)
- [ ] Install dependencies
- [ ] Create SDXLPipelineManager class
- [ ] Implement PyTorch SDPA attention
- [ ] Create sdxl_config.yaml
- [ ] Test basic generation
- [ ] Validate VRAM usage

### Phase 2: LoRA Integration (Estimated: 3 days)
- [ ] Create LoRALoader class
- [ ] Implement multi-LoRA composition
- [ ] Create lora_registry.yaml
- [ ] Create character_presets.yaml
- [ ] Create CharacterGenerator class
- [ ] Test character generation

### Phase 3: ControlNet (Estimated: 3 days)
- [ ] Install ControlNet dependencies
- [ ] Create ControlNetGenerator class
- [ ] Implement OpenPose, Depth, Canny
- [ ] Create controlnet_config.yaml
- [ ] Test pose-controlled generation
- [ ] Test multi-ControlNet composition

### Phase 4: Consistency Validation (Estimated: 3 days)
- [ ] Install InsightFace
- [ ] Create CharacterConsistencyChecker class
- [ ] Extract reference embeddings
- [ ] Test similarity scoring
- [ ] Integrate with CharacterGenerator
- [ ] Validate filtering accuracy

### Phase 5: Batch Processing (Estimated: 2 days)
- [ ] Create BatchGenerator class
- [ ] Implement progress tracking
- [ ] Add quality filtering
- [ ] Test batch workflows

### Phase 6: Testing & Documentation (Estimated: 2 days)
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Performance benchmarking
- [ ] Create usage examples
- [ ] API documentation

---

## 🔄 Integration Points

### With LLM Backend Module
- LLM analyzes user intent
- LLM engineers prompts
- LLM evaluates generated images (Qwen-VL-7B)
- Requires model switching (stop LLM → load SDXL)

### With Model Manager Module
- Request SDXL pipeline load
- Monitor VRAM usage
- Automatic unloading when done

### With RAG System Module (Future)
- Retrieve character descriptions
- Retrieve style guides
- Retrieve past successful prompts

### With Agent Framework Module (Future)
- Agent decides when to generate images
- Agent iterates based on quality evaluation
- Agent selects appropriate LoRAs and ControlNets

---

## 📚 References

- **SDXL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **ControlNet**: https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0
- **Diffusers**: https://huggingface.co/docs/diffusers
- **PEFT (LoRA)**: https://huggingface.co/docs/peft
- **InstantID**: https://github.com/instantX/InstantID
- **InsightFace**: https://github.com/deepinsight/insightface

---

**Last Updated:** 2025-11-17
**Status:** 🔄 In Progress (Architecture Complete, Implementation Pending)
