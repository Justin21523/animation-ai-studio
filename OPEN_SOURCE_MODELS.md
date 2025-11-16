# Open-Source Models and Tools Reference

**Complete list of all open-source models and tools for Animation AI Studio**

---

## 🧠 LLM Decision Engines (Core Brain)

### Qwen2.5-VL (Multimodal Understanding)

```yaml
Model: Qwen/Qwen2.5-VL-72B-Instruct
Size: 72B parameters
Modality: Text + Image + Video
Deployment: Ollama or vLLM

Ollama Command:
  ollama pull qwen2.5-vl:72b

VRAM Requirements:
  - 72B full: ~144GB
  - 72B quantized (INT4): ~48GB
  - 32B full: ~64GB
  - 32B quantized: ~20GB

Capabilities:
  - Long video analysis
  - Image understanding
  - Multimodal reasoning
  - Tool calling

GitHub: https://github.com/QwenLM/Qwen2.5-VL
HuggingFace: Qwen/Qwen2.5-VL-72B-Instruct
```

### DeepSeek-V3 (Complex Reasoning)

```yaml
Model: deepseek-ai/DeepSeek-V3
Size: 671B MoE (37B activated per token)
Modality: Text
Deployment: Ollama or vLLM with FP8

Ollama Command:
  ollama pull deepseek-v3:671b

VRAM Requirements:
  - FP8 quantization: ~80GB (single A100)
  - Full precision: ~1.3TB (multi-GPU)
  - Practical: Use FP8 on A100 80GB

Capabilities:
  - Superior reasoning
  - Strategic planning
  - Complex decision making
  - Mathematics and logic

GitHub: https://github.com/deepseek-ai/DeepSeek-V3
HuggingFace: deepseek-ai/DeepSeek-V3
Paper: https://arxiv.org/abs/2412.19437
```

### Qwen2.5-Coder (Tool Orchestration)

```yaml
Model: Qwen/Qwen2.5-Coder-32B-Instruct
Size: 32B parameters
Modality: Text (code-specialized)
Deployment: Ollama

Ollama Command:
  ollama pull qwen2.5-coder:32b

VRAM Requirements:
  - 32B full: ~64GB
  - 32B quantized: ~20GB

Capabilities:
  - Code generation
  - API calling
  - Workflow automation
  - Tool chain scripting

GitHub: https://github.com/QwenLM/Qwen2.5-Coder
HuggingFace: Qwen/Qwen2.5-Coder-32B-Instruct
```

### Alternative LLMs

```yaml
Llama 3.3 (70B):
  Provider: Meta
  Ollama: ollama pull llama3.3:70b
  VRAM: ~40GB (quantized)
  Use: General reasoning, dialogue

Mixtral 8x22B:
  Provider: Mistral AI
  Ollama: ollama pull mixtral:8x22b
  VRAM: ~90GB (FP8)
  Use: Multi-expert reasoning

Phi-4 (14B):
  Provider: Microsoft
  Ollama: ollama pull phi4:14b
  VRAM: ~8GB (quantized)
  Use: Fast inference, lightweight tasks
```

---

## 🤖 Agent Frameworks

### LangGraph (Primary)

```yaml
Name: LangGraph
Provider: LangChain
License: MIT
Installation: pip install langgraph

Features:
  - ReAct reasoning pattern
  - State management
  - Tool calling
  - Multi-agent collaboration
  - Graph visualization
  - Streaming support

GitHub: https://github.com/langchain-ai/langgraph
Docs: https://langchain-ai.github.io/langgraph/

Example:
  from langgraph.prebuilt import create_react_agent
  agent = create_react_agent(llm, tools)
```

### AutoGen (Secondary)

```yaml
Name: AutoGen
Provider: Microsoft
License: MIT
Installation: pip install pyautogen

Features:
  - Multi-agent conversations
  - Code execution
  - Human-in-the-loop
  - Group chat

GitHub: https://github.com/microsoft/autogen
Docs: https://microsoft.github.io/autogen/

Example:
  from autogen import AssistantAgent, UserProxyAgent
```

### CrewAI (Alternative)

```yaml
Name: CrewAI
License: MIT
Installation: pip install crewai

Features:
  - Role-based agents
  - Task delegation
  - Sequential/parallel execution

GitHub: https://github.com/joaomdmoura/crewAI
```

---

## 🎨 Image Generation (3D Animation Optimized)

### Stable Diffusion XL (Base)

```yaml
Model: stabilityai/stable-diffusion-xl-base-1.0
Size: 6.9GB
License: OpenRAIL++-M
VRAM: 12GB+ (16GB recommended)

Installation:
  from diffusers import StableDiffusionXLPipeline
  pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
  )

HuggingFace: stabilityai/stable-diffusion-xl-base-1.0
```

### ControlNet (Guided Generation)

```yaml
OpenPose:
  Model: lllyasviel/control_v11p_sd15_openpose
  Purpose: Pose control

Depth:
  Model: lllyasviel/control_v11f1p_sd15_depth
  Purpose: Depth/composition

Canny:
  Model: lllyasviel/control_v11p_sd15_canny
  Purpose: Edge detection

Normal Map:
  Model: lllyasviel/control_v11p_sd15_normalbae
  Purpose: 3D surface normals

Installation:
  from controlnet_aux import OpenposeDetector, DepthEstimator
  from diffusers import ControlNetModel

GitHub: https://github.com/lllyasviel/ControlNet
HuggingFace: lllyasviel/ControlNet-v1-1
```

### InstantID (Character Consistency)

```yaml
Model: InstantX/InstantID
Purpose: Maintain character identity across generations
License: Apache 2.0
VRAM: 12GB+

Features:
  - Single image reference
  - Pose control
  - Style transfer
  - High fidelity

Installation:
  from diffusers import StableDiffusionXLInstantIDPipeline

GitHub: https://github.com/InstantID/InstantID
HuggingFace: InstantX/InstantID
Paper: https://arxiv.org/abs/2401.07519
```

### LoRA Training (Kohya_ss)

```yaml
Tool: Kohya_ss GUI
License: Apache 2.0
Purpose: Train custom LoRA adapters

Features:
  - Character LoRA
  - Background LoRA
  - Style LoRA
  - Emotion LoRA
  - Pose LoRA

Installation:
  git clone https://github.com/bmaltais/kohya_ss
  cd kohya_ss
  ./setup.sh

GitHub: https://github.com/bmaltais/kohya_ss

Training Parameters (3D Animation):
  Learning Rate: 1e-4 to 2e-4
  Network Rank: 32-64
  Dataset: 200-500 images
  Epochs: 10-20
  Alpha Threshold: 0.15 (soft edges)
  No color jitter, no horizontal flip
```

### AnimateDiff (Video Generation)

```yaml
Model: guoyww/AnimateDiff-Lightning
Purpose: Animate still images
License: Apache 2.0

Features:
  - Temporal consistency
  - Motion control
  - LoRA compatible

Installation:
  from diffusers import AnimateDiffPipeline

GitHub: https://github.com/guoyww/AnimateDiff
HuggingFace: guoyww/animatediff-motion-adapter-v1-5-2
```

---

## 🎙️ Voice Synthesis

### GPT-SoVITS (Primary)

```yaml
Name: GPT-SoVITS
License: MIT
VRAM: 8GB+ (16GB recommended)

Features:
  - 1-minute voice cloning
  - 17 languages
  - Emotion control
  - Real-time synthesis (RTX 4090: RTF 0.014)
  - Cross-lingual inference

Installation:
  git clone https://github.com/RVC-Boss/GPT-SoVITS
  cd GPT-SoVITS
  pip install -r requirements.txt

Usage:
  # WebUI
  python webui.py

  # API
  python api.py

GitHub: https://github.com/RVC-Boss/GPT-SoVITS

Voice Cloning Steps:
  1. Prepare 1-5 min character audio
  2. Automatic segmentation
  3. Train voice model
  4. Synthesize with emotion control
```

### Coqui TTS (XTTS-v2)

```yaml
Name: Coqui TTS
Model: XTTS-v2
License: MPL 2.0
VRAM: 6GB+

Features:
  - Multi-speaker
  - Voice cloning
  - 17 languages
  - Zero-shot

Installation:
  pip install TTS

Usage:
  from TTS.api import TTS
  tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
  tts.tts_to_file(
    text="Hello",
    speaker_wav="reference.wav",
    file_path="output.wav"
  )

GitHub: https://github.com/coqui-ai/TTS
```

### EmoKnob (Emotion Control)

```yaml
Name: EmoKnob
Purpose: Fine-grained emotion control for TTS
License: Research (check paper)

Features:
  - Few-shot emotion learning (1-5 examples)
  - Emotion strength parameter
  - Works with foundation models
  - Training-free

Paper: https://arxiv.org/abs/2410.00316
Code: (Check paper for official implementation)

Integration:
  - Extract emotion direction vectors
  - Apply to GPT-SoVITS or XTTS
  - Control emotion intensity
```

---

## 🎬 Video Processing

### SAM2 (Instance Segmentation)

```yaml
Name: Segment Anything Model 2
Provider: Meta
License: Apache 2.0
VRAM: 16GB+ (for large model)

Models:
  - sam2_hiera_large (best quality)
  - sam2_hiera_base (balanced)
  - sam2_hiera_small (fast)

Installation:
  git clone https://github.com/facebookresearch/sam2
  cd sam2
  pip install -e .

Usage:
  from sam2.build_sam import build_sam2_video_predictor
  predictor = build_sam2_video_predictor("sam2_hiera_large")

GitHub: https://github.com/facebookresearch/sam2
Paper: https://arxiv.org/abs/2408.00714

Use Cases:
  - Character tracking
  - Multi-character separation
  - Background removal
  - Temporal consistency
```

### PySceneDetect (Scene Detection)

```yaml
Name: PySceneDetect
License: BSD 3-Clause
Installation: pip install scenedetect[opencv]

Features:
  - Content-aware detection
  - Threshold-based detection
  - Automatic video splitting

Usage:
  from scenedetect import detect, ContentDetector
  scenes = detect(video_path, ContentDetector())

GitHub: https://github.com/Breakthrough/PySceneDetect
Docs: https://scenedetect.com/
```

### MoviePy (Video Editing)

```yaml
Name: MoviePy
License: MIT
Installation: pip install moviepy

Features:
  - Video concatenation
  - Speed ramping
  - Effects and transitions
  - Audio mixing

Usage:
  from moviepy.editor import VideoFileClip
  clip = VideoFileClip("input.mp4")
  slow = clip.fx(vfx.speedx, 0.5)

GitHub: https://github.com/Zulko/moviepy
Docs: https://zulko.github.io/moviepy/
```

### FFmpeg (Professional Processing)

```yaml
Name: FFmpeg
License: LGPL 2.1+
Installation: apt install ffmpeg (Linux)

Capabilities:
  - Format conversion
  - Codec encoding
  - Audio/video sync
  - Streaming

Usage:
  import subprocess
  subprocess.run([
    "ffmpeg", "-i", "input.mp4",
    "-c:v", "libx264", "-crf", "18",
    "output.mp4"
  ])

Website: https://ffmpeg.org/
```

### Real-ESRGAN (Upscaling)

```yaml
Name: Real-ESRGAN
License: BSD 3-Clause
VRAM: 8GB+

Models:
  - RealESRGAN_x4plus (general)
  - RealESRGAN_x4plus_anime_6B (anime)
  - RealESRGAN_x2plus (moderate)

Installation:
  pip install realesrgan

Usage:
  from realesrgan import RealESRGANer
  upsampler = RealESRGANer(scale=4, model_path="...")
  output = upsampler.enhance(image)

GitHub: https://github.com/xinntao/Real-ESRGAN
```

---

## 🎭 Lip-Sync and Facial Animation

### Wav2Lip (Basic Lip-Sync)

```yaml
Name: Wav2Lip
License: Custom (check repo)
VRAM: 8GB+

Installation:
  git clone https://github.com/Rudrabha/Wav2Lip
  cd Wav2Lip
  pip install -r requirements.txt

Usage:
  python inference.py \
    --checkpoint_path checkpoints/wav2lip_gan.pth \
    --face input_video.mp4 \
    --audio input_audio.wav \
    --outfile output.mp4

GitHub: https://github.com/Rudrabha/Wav2Lip
Paper: https://arxiv.org/abs/2008.10010
```

### VividWav2Lip (Improved)

```yaml
Name: VividWav2Lip
Improvement: 5% better accuracy, reduced jitter
Paper: Electronics 2024
Status: Check for open-source implementation

Enhancements:
  - Cross-attention mechanism
  - SE residual blocks
  - Better dataset optimization
```

### UniTalker (Audio → 3D Face)

```yaml
Name: UniTalker
Conference: ECCV 2024
Purpose: Audio-driven 3D facial animation

Features:
  - Unified model
  - Multi-speaker
  - Emotion expression
  - 100 fps viseme output

Paper: https://arxiv.org/abs/2407.11036
Code: (Check paper for official repo)
```

---

## 🔍 Multimodal Analysis

### MediaPipe (Face Mesh)

```yaml
Name: MediaPipe
Provider: Google
License: Apache 2.0
Installation: pip install mediapipe

Features:
  - 478 3D facial landmarks
  - Real-time performance
  - Multi-face tracking

Usage:
  import mediapipe as mp
  face_mesh = mp.solutions.face_mesh.FaceMesh()
  results = face_mesh.process(image)

GitHub: https://github.com/google/mediapipe
Docs: https://google.github.io/mediapipe/
```

### InsightFace (Face Analysis)

```yaml
Name: InsightFace
License: Custom (check repo)
VRAM: 4GB+

Models:
  - buffalo_l (large, best quality)
  - buffalo_s (small, fast)

Features:
  - Face detection
  - Recognition
  - Age/gender estimation
  - 3D reconstruction
  - Expression analysis

Installation:
  pip install insightface

Usage:
  from insightface.app import FaceAnalysis
  app = FaceAnalysis(providers=['CUDAExecutionProvider'])
  faces = app.get(image)

GitHub: https://github.com/deepinsight/insightface
```

### Wav2Vec2 (Audio Features)

```yaml
Name: Wav2Vec2
Provider: Meta
License: MIT
Purpose: Audio feature extraction

Model: facebook/wav2vec2-base
Installation: pip install transformers

Usage:
  from transformers import Wav2Vec2Processor, Wav2Vec2Model
  processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
  model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

HuggingFace: facebook/wav2vec2-base
```

---

## 🛠️ Utility Tools

### LaMa (Inpainting)

```yaml
Name: LaMa (Large Mask Inpainting)
License: Apache 2.0
VRAM: 4GB+

Purpose:
  - Background inpainting
  - Object removal
  - High-quality filling

Installation:
  git clone https://github.com/advimman/lama
  cd lama
  pip install -r requirements.txt

GitHub: https://github.com/advimman/lama
```

### GFPGAN (Face Restoration)

```yaml
Name: GFPGAN
License: Custom
Purpose: Face enhancement and restoration

Installation:
  pip install gfpgan

Usage:
  from gfpgan import GFPGANer
  restorer = GFPGANer(model_path="...")
  restored = restorer.enhance(face_image)

GitHub: https://github.com/TencentARC/GFPGAN
```

### CodeFormer (Face Enhancement)

```yaml
Name: CodeFormer
License: Custom
Purpose: Robust face restoration

Installation:
  git clone https://github.com/sczhou/CodeFormer
  cd CodeFormer
  pip install -r requirements.txt

GitHub: https://github.com/sczhou/CodeFormer
```

---

## 📦 Deployment Tools

### Ollama (LLM Deployment)

```yaml
Name: Ollama
License: MIT
Platform: Linux, macOS, Windows

Installation:
  curl -fsSL https://ollama.com/install.sh | sh

Features:
  - One-command model download
  - OpenAI-compatible API
  - GPU acceleration
  - Model quantization

Usage:
  ollama pull qwen2.5-vl:72b
  ollama serve

Website: https://ollama.com
GitHub: https://github.com/ollama/ollama
```

### vLLM (High-Performance Inference)

```yaml
Name: vLLM
License: Apache 2.0
Purpose: Fast LLM inference

Features:
  - PagedAttention
  - Continuous batching
  - Multi-GPU support
  - OpenAI API compatible

Installation:
  pip install vllm

Usage:
  from vllm import LLM
  llm = LLM(model="Qwen/Qwen2.5-VL-72B")

GitHub: https://github.com/vllm-project/vllm
```

---

## 🎓 Training and Fine-tuning

### Kohya_ss (LoRA Training)

```yaml
Name: Kohya_ss
License: Apache 2.0
Purpose: LoRA/LoCon training GUI

Installation:
  git clone https://github.com/bmaltais/kohya_ss
  cd kohya_ss && ./setup.sh

GitHub: https://github.com/bmaltais/kohya_ss
```

### Axolotl (LLM Fine-tuning)

```yaml
Name: Axolotl
License: Apache 2.0
Purpose: LLM fine-tuning framework

Installation:
  git clone https://github.com/OpenAccess-AI-Collective/axolotl
  pip install -e .

GitHub: https://github.com/OpenAccess-AI-Collective/axolotl
```

---

## 📊 Summary Table

| Category | Tool | VRAM | License | Priority |
|----------|------|------|---------|----------|
| **LLM Brain** | Qwen2.5-VL 72B | 48GB+ | Apache 2.0 | ⭐⭐⭐ |
| **LLM Reasoning** | DeepSeek-V3 671B | 80GB+ | MIT | ⭐⭐⭐ |
| **Agent Framework** | LangGraph | - | MIT | ⭐⭐⭐ |
| **Image Gen** | SDXL | 12GB | OpenRAIL++ | ⭐⭐⭐ |
| **Character ID** | InstantID | 12GB | Apache 2.0 | ⭐⭐⭐ |
| **Voice Clone** | GPT-SoVITS | 8GB | MIT | ⭐⭐⭐ |
| **Segmentation** | SAM2 | 16GB | Apache 2.0 | ⭐⭐⭐ |
| **Lip-Sync** | Wav2Lip | 8GB | Custom | ⭐⭐ |
| **Video Edit** | MoviePy | - | MIT | ⭐⭐ |
| **Face Mesh** | MediaPipe | - | Apache 2.0 | ⭐⭐ |

---

**Last Updated:** 2025-11-16
**Total Tools:** 50+
**All Open-Source:** ✅
