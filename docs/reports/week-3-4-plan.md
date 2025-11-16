# Week 3-4: 3D Character Generation Tools - Implementation Plan

**Date Created:** 2025-11-16
**Phase:** 3D Character Tools Integration
**Hardware:** RTX 5080 16GB VRAM (single GPU)
**PyTorch:** 2.7.0 + CUDA 12.8
**Status:** Planning Complete, Ready for Implementation

---

## 🎯 Objectives

Build comprehensive 3D character generation capabilities:
1. **Image Generation**: SDXL + LoRA for character images
2. **Pose Control**: ControlNet for guided generation
3. **Character Consistency**: Maintain identity across generations
4. **Voice Synthesis**: GPT-SoVITS for character voices

---

## 📦 Deliverables

### 1. Image Generation System
- **SDXL Base Integration** (Stable Diffusion XL 1.0)
- **LoRA Adapter Loading** (character, background, pose)
- **ControlNet Support** (OpenPose, Depth, Canny)
- **Character Consistency** (InstantID, IP-Adapter)

### 2. Voice Synthesis System
- **GPT-SoVITS Wrapper** (text-to-speech)
- **Voice Cloning Pipeline** (train character voices)
- **Emotion Control** (happy, sad, excited, etc.)
- **Lip-sync Generation** (Wav2Lip integration)

### 3. Configuration System
- **Model Configuration** (paths, parameters, settings)
- **Generation Presets** (character-specific settings)
- **Quality Control** (validation, filtering)

### 4. Integration
- **LLM Client Integration** (creative intent → generation)
- **Batch Processing** (multiple character generation)
- **Caching** (generated images, voice samples)

---

## 🔧 Technical Architecture

### Image Generation Pipeline

```
User Request
     ↓
LLM Intent Analysis (Qwen-14B)
     ↓
Prompt Engineering (Qwen-Coder-7B)
     ↓
┌──────────────────────────┐
│  SDXL Pipeline Manager   │
│  - Base model: SDXL 1.0  │
│  - LoRA injection        │
│  - ControlNet guidance   │
│  - Character consistency │
└──────────────────────────┘
     ↓
Image Generation
     ↓
Quality Evaluation (Qwen-VL-7B)
     ↓
Output / Iterate
```

### Voice Synthesis Pipeline

```
User Request (text + character + emotion)
     ↓
LLM Intent Analysis
     ↓
┌──────────────────────────┐
│  GPT-SoVITS Engine       │
│  - Voice model selection │
│  - Emotion control       │
│  - Language handling     │
└──────────────────────────┘
     ↓
Audio Generation
     ↓
Quality Check
     ↓
Output (WAV file)
```

---

## 🖥️ Hardware Optimization (RTX 5080 16GB)

### VRAM Management Strategy

**Critical Constraint:** 16GB VRAM must accommodate:
1. Either LLM (7B/14B) OR SDXL, not both simultaneously
2. SDXL base (~10-11GB) + LoRA (~1GB) + ControlNet (~1-2GB)
3. GPT-SoVITS (~3-4GB for small model)

**Solution: Dynamic Model Switching**

```yaml
Workflow 1: Image Generation
  1. Stop LLM service (free 12-14GB)
  2. Load SDXL pipeline (use 13-15GB)
  3. Generate images
  4. Unload SDXL
  5. Restart LLM service

Workflow 2: Voice Synthesis
  1. Can run alongside stopped SDXL
  2. GPT-SoVITS small model (~4GB)
  3. Or run with LLM stopped for large model

Workflow 3: LLM-Only Tasks
  1. Stop SDXL/GPT-SoVITS
  2. Run LLM for analysis/planning
```

### Memory Optimization Techniques

1. **Model Quantization**
   - SDXL: FP16 (default)
   - Option: NF4 quantization (4-bit) for larger LoRAs

2. **Sequential Loading**
   - Load ControlNet only when needed
   - Release LoRA weights after generation

3. **Attention Optimization**
   - Use PyTorch SDPA (required for compatibility)
   - Enable xformers ONLY for Diffusers (NOT vLLM)
   - Memory-efficient attention for long prompts

4. **Batch Size Control**
   - Single image generation (batch=1)
   - Sequential processing for multiple images

---

## 📂 Directory Structure

```
scripts/
├── generation/
│   └── image/
│       ├── __init__.py
│       ├── sdxl_pipeline.py           # SDXL base pipeline
│       ├── lora_loader.py             # LoRA adapter loading
│       ├── controlnet_generator.py    # ControlNet integration
│       ├── character_generator.py     # Character-specific generation
│       ├── consistency_checker.py     # Character consistency validation
│       └── batch_generator.py         # Batch processing
│
├── synthesis/
│   └── tts/
│       ├── __init__.py
│       ├── gpt_sovits_wrapper.py      # GPT-SoVITS Python wrapper
│       ├── voice_model_trainer.py     # Train character voices
│       ├── emotion_controller.py      # Emotion-aware synthesis
│       └── voice_dataset_builder.py   # Extract voice samples from film
│
└── core/
    └── generation/
        ├── __init__.py
        ├── model_manager.py           # Dynamic model loading/unloading
        └── generation_cache.py        # Cache generated content

configs/
└── generation/
    ├── sdxl_config.yaml               # SDXL configuration
    ├── controlnet_config.yaml         # ControlNet settings
    ├── lora_registry.yaml             # Available LoRAs
    ├── character_presets.yaml         # Per-character generation settings
    └── tts_config.yaml                # Voice synthesis settings

requirements/
└── generation.txt                     # Image/voice generation dependencies
```

---

## 🚀 Implementation Plan

### Phase 1: SDXL Base Integration (Days 1-3)

#### Step 1.1: Install Dependencies
```bash
# Create requirements/generation.txt
diffusers>=0.25.0
transformers>=4.37.0
accelerate>=0.26.0
safetensors>=0.4.1
peft>=0.8.0               # For LoRA
controlnet_aux>=0.0.7     # ControlNet preprocessors
insightface>=0.7.3        # For InstantID
onnxruntime-gpu>=1.16.0   # For InsightFace
compel>=2.0.2             # Prompt weighting
```

#### Step 1.2: Create SDXL Pipeline Manager
**File:** `scripts/generation/image/sdxl_pipeline.py`

**Core Features:**
- Load SDXL base model (stabilityai/stable-diffusion-xl-base-1.0)
- PyTorch SDPA attention (compatibility with our stack)
- FP16 precision
- VRAM-efficient loading
- Safety checker (optional)
- Negative prompt support

**Key Methods:**
```python
class SDXLPipelineManager:
    def __init__(self, model_path, device="cuda", dtype=torch.float16)
    def load_pipeline(self) -> StableDiffusionXLPipeline
    def generate(self, prompt, negative_prompt, num_steps, cfg_scale, ...)
    def unload_pipeline(self)  # Free VRAM
    def get_vram_usage(self) -> float
```

#### Step 1.3: Create Configuration
**File:** `configs/generation/sdxl_config.yaml`

```yaml
model:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  cache_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/diffusion/sdxl"
  variant: "fp16"

hardware:
  device: "cuda"
  dtype: "float16"
  vram_optimization: true
  enable_cpu_offload: false  # RTX 5080 has enough VRAM
  enable_sequential_cpu_offload: false

attention:
  backend: "sdpa"  # PyTorch SDPA
  enable_xformers: false  # Conflicts with vLLM settings

generation:
  default_steps: 30
  default_cfg_scale: 7.5
  default_size: [1024, 1024]
  scheduler: "DPMSolverMultistepScheduler"

safety:
  enable_safety_checker: false
  nsfw_threshold: 0.0
```

#### Step 1.4: Test SDXL Base Generation
```bash
# Test script
python scripts/generation/image/sdxl_pipeline.py \
  --prompt "luca, a boy with brown hair, green eyes, pixar style, 3d animation" \
  --negative-prompt "2d, anime, flat, low quality" \
  --steps 30 \
  --cfg-scale 7.5 \
  --output outputs/test/sdxl_base_test.png
```

---

### Phase 2: LoRA Integration (Days 4-6)

#### Step 2.1: Create LoRA Loader
**File:** `scripts/generation/image/lora_loader.py`

**Core Features:**
- Load LoRA adapters from safetensors
- Multi-LoRA composition (character + background + style)
- Weight adjustment (0.0-1.0)
- LoRA unloading
- LoRA registry management

**Key Methods:**
```python
class LoRALoader:
    def __init__(self, pipeline: StableDiffusionXLPipeline)
    def load_lora(self, lora_path: str, adapter_name: str, weight: float = 0.8)
    def load_multiple_loras(self, loras: List[Dict])
    def set_lora_weight(self, adapter_name: str, weight: float)
    def unload_lora(self, adapter_name: str)
    def unload_all_loras(self)
    def list_loaded_loras(self) -> List[str]
```

#### Step 2.2: Create LoRA Registry
**File:** `configs/generation/lora_registry.yaml`

```yaml
# LoRA models available for use
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

  giulia:
    name: "Giulia Marcovaldo"
    lora_path: "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/outputs/loras/giulia_character.safetensors"
    default_weight: 0.8
    trigger_words: ["giulia", "red hair", "freckles"]

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

#### Step 2.3: Create Character Generator
**File:** `scripts/generation/image/character_generator.py`

**Features:**
- Character-specific prompt engineering
- Multi-LoRA composition (character + background + style)
- Automatic trigger word injection
- Quality presets (draft, standard, high quality)

**Usage Example:**
```python
from scripts.generation.image import CharacterGenerator

generator = CharacterGenerator()
generator.generate_character(
    character="luca",
    scene_description="running on the beach, excited expression",
    background="portorosso",
    quality="high",
    output_path="outputs/characters/luca_beach.png"
)
```

---

### Phase 3: ControlNet Integration (Days 7-9)

#### Step 3.1: Install ControlNet Dependencies
```bash
# Add to requirements/generation.txt
controlnet-aux>=0.0.7
opencv-python>=4.8.0
mediapipe>=0.10.0  # For pose detection
```

#### Step 3.2: Create ControlNet Generator
**File:** `scripts/generation/image/controlnet_generator.py`

**Supported ControlNets:**
1. **OpenPose** - Pose control
2. **Depth** - Depth map guidance
3. **Canny** - Edge detection

**Core Features:**
- Preprocessor integration (pose detection, depth estimation)
- Multi-ControlNet composition
- Conditioning strength control
- Reference image processing

**Key Methods:**
```python
class ControlNetGenerator:
    def __init__(self, base_pipeline: StableDiffusionXLPipeline)

    def generate_with_pose(
        self,
        reference_image: str,  # Extract pose from this
        prompt: str,
        character_lora: str = None,
        controlnet_strength: float = 1.0
    )

    def generate_with_depth(
        self,
        reference_image: str,
        prompt: str,
        ...
    )

    def generate_with_canny(
        self,
        reference_image: str,
        prompt: str,
        ...
    )

    def generate_multi_controlnet(
        self,
        controls: List[Dict],  # [{"type": "pose", "image": "...", "strength": 0.8}, ...]
        prompt: str,
        ...
    )
```

#### Step 3.3: Configuration
**File:** `configs/generation/controlnet_config.yaml`

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

#### Step 3.4: Test ControlNet Generation
```bash
# Test with pose control
python scripts/generation/image/controlnet_generator.py \
  --mode pose \
  --reference-image /path/to/pose_reference.jpg \
  --character luca \
  --prompt "luca, running pose, excited expression, pixar style" \
  --output outputs/test/controlnet_pose_test.png
```

---

### Phase 4: Character Consistency (Days 10-12)

#### Step 4.1: Install InstantID
```bash
# Add to requirements/generation.txt
insightface>=0.7.3
onnxruntime-gpu>=1.16.0
```

#### Step 4.2: Create Consistency Checker
**File:** `scripts/generation/image/consistency_checker.py`

**Features:**
- Face embedding extraction (ArcFace)
- Character identity verification
- Similarity scoring
- Quality filtering

**Key Methods:**
```python
class CharacterConsistencyChecker:
    def __init__(self, model_name='buffalo_l')

    def extract_embedding(self, image_path: str) -> np.ndarray

    def compute_similarity(
        self,
        reference_embedding: np.ndarray,
        generated_embedding: np.ndarray
    ) -> float

    def verify_character_match(
        self,
        reference_image: str,
        generated_image: str,
        threshold: float = 0.6
    ) -> Dict

    def filter_consistent_images(
        self,
        reference_image: str,
        generated_images: List[str],
        threshold: float = 0.6
    ) -> List[str]
```

#### Step 4.3: Create Character Presets
**File:** `configs/generation/character_presets.yaml`

```yaml
# Per-character generation settings
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

giulia:
  character_name: "Giulia Marcovaldo"
  lora_name: "giulia"
  lora_weight: 0.80

  base_prompt: "giulia, girl, red hair, freckles, pixar style, 3d animation"
  negative_prompt: "2d, flat, anime, low quality"

  consistency_threshold: 0.60
```

---

### Phase 5: GPT-SoVITS Integration (Days 13-16)

#### Step 5.1: Install GPT-SoVITS

**Option 1: Git Clone (Recommended)**
```bash
cd /mnt/c/AI_LLM_projects/ai_warehouse/tools
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Python Package**
```bash
# Add to requirements/tts.txt
GPT_SoVITS>=0.1.0
TTS>=0.22.0
librosa>=0.10.0
soundfile>=0.12.0
```

#### Step 5.2: Create GPT-SoVITS Wrapper
**File:** `scripts/synthesis/tts/gpt_sovits_wrapper.py`

**Core Features:**
- Interface to GPT-SoVITS inference
- Voice model loading
- Emotion control
- Language support (EN, IT)
- Batch synthesis

**Key Methods:**
```python
class GPTSoVITSWrapper:
    def __init__(self, model_dir: str, device: str = "cuda")

    def load_voice_model(
        self,
        character_name: str,
        sovits_path: str,
        gpt_path: str
    )

    def synthesize(
        self,
        text: str,
        character: str,
        emotion: str = "neutral",
        language: str = "en",
        speed: float = 1.0,
        output_path: str = None
    ) -> str

    def synthesize_batch(
        self,
        texts: List[str],
        character: str,
        output_dir: str
    ) -> List[str]

    def get_available_voices(self) -> List[str]
```

#### Step 5.3: Create Voice Model Trainer
**File:** `scripts/synthesis/tts/voice_model_trainer.py`

**Features:**
- Extract voice samples from film audio
- Clean and preprocess audio
- Train GPT-SoVITS voice model
- Validate quality

**Workflow:**
```python
# 1. Extract voice samples from film
extractor = VoiceSampleExtractor()
samples = extractor.extract_from_film(
    video_path="/mnt/data/ai_data/datasets/3d-anime/luca/film.mp4",
    transcript_path="data/films/luca/transcript.json",
    character_name="Luca",
    output_dir="data/films/luca/voice_samples"
)

# 2. Train voice model
trainer = VoiceModelTrainer()
trainer.train(
    voice_samples_dir="data/films/luca/voice_samples",
    character_name="Luca",
    output_dir="/mnt/c/AI_LLM_projects/ai_warehouse/models/tts/luca",
    epochs=100
)
```

#### Step 5.4: Configuration
**File:** `configs/generation/tts_config.yaml`

```yaml
gpt_sovits:
  repo_path: "/mnt/c/AI_LLM_projects/ai_warehouse/tools/GPT-SoVITS"
  models_dir: "/mnt/c/AI_LLM_projects/ai_warehouse/models/tts"

  base_models:
    gpt: "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    sovits: "pretrained_models/s2G488k.pth"

  inference:
    device: "cuda"
    dtype: "float16"
    max_length: 1024
    top_k: 15
    top_p: 1.0
    temperature: 1.0

  languages:
    - "en"  # English
    - "it"  # Italian

character_voices:
  luca:
    name: "Luca Paguro"
    sovits_model: "luca_sovits.pth"
    gpt_model: "luca_gpt.ckpt"
    reference_audio: "luca_reference.wav"
    language: "en"

  alberto:
    name: "Alberto Scorfano"
    sovits_model: "alberto_sovits.pth"
    gpt_model: "alberto_gpt.ckpt"
    reference_audio: "alberto_reference.wav"
    language: "en"

emotions:
  neutral:
    description: "Normal speaking voice"
    temperature: 1.0

  happy:
    description: "Excited, joyful"
    temperature: 1.2

  sad:
    description: "Sad, melancholic"
    temperature: 0.8

  excited:
    description: "Very energetic"
    temperature: 1.3
```

---

### Phase 6: LLM Client Integration (Days 17-18)

#### Step 6.1: Extend LLM Client
**File:** `scripts/core/llm_client/llm_client.py`

**Add New Methods:**
```python
class LLMClient:
    # Existing methods...

    async def generate_character_image(
        self,
        character: str,
        scene_description: str,
        pose_reference: str = None,
        quality: str = "high"
    ) -> Dict:
        """
        Use LLM to understand intent, then generate character image

        Returns:
            {
                "image_path": "outputs/...",
                "prompt_used": "...",
                "loras_used": ["luca", "portorosso"],
                "consistency_score": 0.87,
                "generation_time": 12.5
            }
        """
        # 1. Analyze intent with Qwen-14B
        # 2. Generate prompt with Qwen-Coder-7B
        # 3. Stop LLM service
        # 4. Load SDXL + LoRAs
        # 5. Generate image
        # 6. Evaluate with Qwen-VL-7B (restart LLM)
        # 7. Return result

    async def generate_character_voice(
        self,
        character: str,
        text: str,
        emotion: str = "neutral"
    ) -> Dict:
        """
        Generate character voice with emotion

        Returns:
            {
                "audio_path": "outputs/...",
                "character": "luca",
                "emotion": "happy",
                "duration": 3.2,
                "quality_score": 0.92
            }
        """
        # 1. Load GPT-SoVITS
        # 2. Synthesize voice
        # 3. Return result

    async def create_character_scene(
        self,
        characters: List[str],
        scene_description: str,
        dialogue: List[Dict]  # [{"character": "luca", "text": "...", "emotion": "happy"}]
    ) -> Dict:
        """
        Full pipeline: images + voices for a scene

        Returns:
            {
                "images": ["luca.png", "alberto.png"],
                "audio_files": ["luca_voice.wav", "alberto_voice.wav"],
                "metadata": {...}
            }
        """
```

#### Step 6.2: Create Model Manager
**File:** `scripts/core/generation/model_manager.py`

**Purpose:** Manage dynamic model loading/unloading

```python
class GenerationModelManager:
    """
    Manages VRAM by loading/unloading models dynamically

    Only one "heavy" model can be loaded at a time:
    - LLM (12-14GB)
    - SDXL (13-15GB)
    - GPT-SoVITS can run alongside stopped LLM (~4GB)
    """

    def __init__(self):
        self.current_model: str = None
        self.llm_service_running: bool = False

    async def switch_to_image_generation(self):
        # Stop LLM service
        # Load SDXL pipeline

    async def switch_to_llm(self):
        # Unload SDXL
        # Start LLM service

    async def load_voice_synthesis(self):
        # Can run with LLM stopped
        # Load GPT-SoVITS

    def get_vram_usage(self) -> Dict:
        # Return current VRAM usage
```

---

### Phase 7: Testing & Validation (Days 19-20)

#### Test Cases

**Test 1: Basic SDXL Generation**
```bash
python scripts/generation/image/sdxl_pipeline.py \
  --prompt "pixar style 3d animation, boy, smooth shading" \
  --steps 30 \
  --output outputs/test/basic_sdxl.png
```

**Test 2: LoRA Character Generation**
```bash
python scripts/generation/image/character_generator.py \
  --character luca \
  --scene "standing on the beach, smiling" \
  --quality high \
  --output outputs/test/luca_beach.png
```

**Test 3: ControlNet Pose**
```bash
python scripts/generation/image/controlnet_generator.py \
  --mode pose \
  --reference /path/to/pose.jpg \
  --character luca \
  --output outputs/test/luca_pose.png
```

**Test 4: Character Consistency**
```bash
python scripts/generation/image/consistency_checker.py \
  --reference data/films/luca/references/luca_ref.jpg \
  --generated outputs/test/luca_*.png \
  --threshold 0.65
```

**Test 5: Voice Synthesis**
```bash
python scripts/synthesis/tts/gpt_sovits_wrapper.py \
  --character luca \
  --text "Silenzio, Bruno!" \
  --emotion excited \
  --output outputs/test/luca_voice.wav
```

**Test 6: End-to-End Scene Creation**
```python
import asyncio
from scripts.core.llm_client import LLMClient

async def test_scene_creation():
    async with LLMClient() as client:
        result = await client.create_character_scene(
            characters=["luca", "alberto"],
            scene_description="two boys talking on the beach at sunset",
            dialogue=[
                {"character": "luca", "text": "We can do this!", "emotion": "excited"},
                {"character": "alberto", "text": "Piacere, Girolamo Trombetta!", "emotion": "happy"}
            ]
        )
        print(result)

asyncio.run(test_scene_creation())
```

---

## 📊 Success Criteria

### Week 3-4 Completion Checklist

**Image Generation:**
- [ ] SDXL base model integrated and working
- [ ] LoRA loading functional (character, background, style)
- [ ] ControlNet generation working (OpenPose, Depth, Canny)
- [ ] Character consistency validation operational
- [ ] Batch generation pipeline functional

**Voice Synthesis:**
- [ ] GPT-SoVITS wrapper implemented
- [ ] Voice model training pipeline working
- [ ] Character voice models trained (at least Luca)
- [ ] Emotion control functional
- [ ] Multi-language support (EN, IT)

**Integration:**
- [ ] LLM client extended with generation methods
- [ ] Model manager for dynamic VRAM allocation
- [ ] Generation caching implemented
- [ ] End-to-end scene creation pipeline working

**Configuration:**
- [ ] All config files created
- [ ] LoRA registry populated
- [ ] Character presets defined
- [ ] TTS configuration complete

**Testing:**
- [ ] All test cases passing
- [ ] Performance benchmarks documented
- [ ] VRAM usage validated
- [ ] Quality metrics established

**Documentation:**
- [ ] API documentation for all modules
- [ ] Usage examples
- [ ] Troubleshooting guide
- [ ] Week 3-4 completion summary

---

## 📈 Performance Targets

### Image Generation

```yaml
Latency:
  - SDXL base generation: < 15 seconds (30 steps)
  - LoRA generation: < 20 seconds (30 steps)
  - ControlNet generation: < 25 seconds (30 steps)

Quality:
  - Character consistency: > 0.65 similarity score
  - Resolution: 1024x1024 (SDXL native)
  - Steps: 30-35 (good quality/speed balance)

VRAM:
  - SDXL base: ~10-11GB
  - +LoRA: ~11-12GB
  - +ControlNet: ~13-15GB
  - Peak usage: < 15.5GB (safety margin)
```

### Voice Synthesis

```yaml
Latency:
  - Short sentence (1-2s audio): < 3 seconds generation
  - Medium sentence (3-5s audio): < 5 seconds generation
  - Long sentence (10s audio): < 10 seconds generation

Quality:
  - Voice similarity: > 85% (subjective evaluation)
  - Naturalness: > 4.0/5.0 MOS score
  - Emotion accuracy: Clear distinction between emotions

VRAM:
  - GPT-SoVITS small: ~3-4GB
  - Can run with SDXL unloaded
```

---

## 🚨 Known Challenges & Solutions

### Challenge 1: VRAM Constraints
**Issue:** RTX 5080 16GB cannot run LLM + SDXL simultaneously

**Solution:**
- Dynamic model switching via ModelManager
- Stop LLM service before loading SDXL
- Cache LLM responses for reuse
- Sequential workflow (plan → generate → evaluate)

### Challenge 2: LoRA Availability
**Issue:** LoRA models not yet trained (LoRA pipeline at 14.8%)

**Solution:**
- Build infrastructure first (can test with public LoRAs)
- Use placeholder LoRAs for testing
- Integrate trained LoRAs when pipeline completes
- Test with generic character LoRAs from HuggingFace

### Challenge 3: PyTorch Compatibility
**Issue:** SDXL uses xformers by default, conflicts with vLLM settings

**Solution:**
- Force PyTorch SDPA for SDXL: `attn_processor = AttnProcessor2_0()`
- Keep xformers disabled globally
- Test compatibility thoroughly
- Document attention backend choice

### Challenge 4: Voice Sample Extraction
**Issue:** Need clean voice samples from film for GPT-SoVITS training

**Solution:**
- Use Whisper + Pyannote for speaker diarization
- Extract segments with low background noise
- Manually curate best samples (1-5 minutes per character)
- Use audio enhancement (noise reduction) if needed

### Challenge 5: Character Consistency
**Issue:** Maintaining identity across different poses/scenes

**Solution:**
- Multiple reference embeddings per character
- Lower consistency threshold (0.60-0.65)
- Use InstantID for strong identity preservation
- Manual curation of best outputs

---

## 🔄 Integration with Week 1-2 (LLM Backend)

### Workflow Example: AI-Driven Character Image Generation

```
1. User Request:
   "Generate an image of Luca running excitedly on the beach"

2. LLM Analysis (Qwen-14B):
   - Character: Luca
   - Action: Running
   - Emotion: Excited
   - Location: Beach
   - Decide: Use ControlNet (OpenPose) for dynamic pose

3. Prompt Engineering (Qwen-Coder-7B):
   - Positive: "luca, boy, brown hair, running pose, excited expression,
               beach background, sunny day, pixar style, 3d animation"
   - Negative: "2d, anime, flat, low quality, static pose"
   - ControlNet: Find running pose reference

4. Model Switching:
   - Stop LLM service (free VRAM)
   - Load SDXL + Luca LoRA + ControlNet

5. Image Generation:
   - Generate with pose control
   - Steps: 35, CFG: 7.5, Size: 1024x1024

6. Quality Evaluation (Qwen-VL-7B):
   - Restart LLM service
   - Evaluate: "Does this image match the description?"
   - Check character consistency
   - Decide: Accept or regenerate with adjustments

7. Output:
   - Image saved to outputs/characters/luca_beach_running.png
   - Metadata stored for caching
```

---

## 📝 Next Steps After Week 3-4

**Week 5-6: LangGraph Agent Decision Engine**
- Use generated images/voices as tools
- LLM decides WHEN to generate (not just respond to requests)
- Autonomous creative iteration
- Multi-tool composition

**Week 7-8: End-to-End Creative Workflows**
- Parody video generation
- Automated character scene creation
- Style remix and effects
- Complete user-facing application

---

## 📚 References

### Models
- SDXL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- ControlNet: https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- InstantID: https://github.com/instantX/InstantID

### Documentation
- Diffusers: https://huggingface.co/docs/diffusers
- PEFT (LoRA): https://huggingface.co/docs/peft
- ControlNet Aux: https://github.com/patrickvonplaten/controlnet_aux

---

**Created by:** Claude Code
**Date:** 2025-11-16
**Version:** 1.0.0
**Status:** ✅ Ready for Implementation
