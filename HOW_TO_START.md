# How to Start - Instructions for Claude Code

**This document explains how to quickly get Claude Code up to speed on the Animation AI Studio project.**

---

## 📝 Quick Context for New Claude Code Session

### Step 1: Read Core Documents (Order Matters!)

```bash
# In new WSL terminal
cd /mnt/c/AI_LLM_projects/animation-ai-studio

# Read in this order:
1. PROJECT_STATUS.md      # Current status and architecture
2. CLAUDE.md              # Complete project instructions
3. OPEN_SOURCE_MODELS.md  # All models and tools
4. README.md              # Quick start guide
```

### Step 2: Key Information to Understand

**Core Requirements:**
- ✅ All open-source LLMs (Qwen2.5-VL, DeepSeek-V3)
- ✅ LLM as creative decision-making brain
- ✅ Agent-based architecture (LangGraph)
- ✅ 3D animation optimization (Pixar-style)
- ❌ NO closed-source APIs (GPT-4, Claude, Gemini)

**Project Purpose:**
- Create, analyze, and transform animated content
- LLM agents make creative decisions autonomously
- Coordinate 50+ specialized AI tools
- Optimize for 3D character consistency

**Current Status:**
- ✅ Project structure created
- ✅ Comprehensive research completed
- ✅ Documentation written
- 🔄 Ready for implementation

---

## 💬 What to Tell Claude Code

### Opening Message Template

```
I'm working on Animation AI Studio, an LLM-driven AI platform for
3D animation content creation.

Key points:
1. All models must be OPEN-SOURCE (Qwen2.5-VL, DeepSeek-V3, etc.)
2. LLM agents make creative decisions autonomously
3. Optimized for Pixar-style 3D animation
4. Uses LangGraph for agent framework

Please read:
- PROJECT_STATUS.md (current status)
- CLAUDE.md (complete instructions)
- OPEN_SOURCE_MODELS.md (all tools)

Current task: [describe what you want to work on]
```

### Example Tasks You Might Request

**Task 1: Set Up LLM Decision Engine**
```
I need to implement the LLM decision engine using:
- Qwen2.5-VL (multimodal understanding)
- DeepSeek-V3 (reasoning)
- LangGraph (agent framework)

Please:
1. Set up Ollama with these models
2. Create LangGraph agent structure
3. Implement ReAct reasoning loop
4. Add tool registration system
```

**Task 2: Implement Tool for Character Generation**
```
I need a tool that generates 3D animation style character images using:
- SDXL base model
- Character LoRA (from LoRA pipeline)
- OpenPose ControlNet

The tool should:
1. Take prompt + character name + pose reference
2. Load appropriate LoRA weights
3. Generate with Pixar-style optimization
4. Return high-quality image
```

**Task 3: Create Parody Video Generator**
```
I need an automated parody video generator that:
1. Analyzes source video (SAM2 + scene detection)
2. LLM identifies funny moments
3. Applies effects (slow-mo, exaggeration, speed ramping)
4. Generates final parody video

Use open-source tools only.
```

---

## 🎯 Implementation Priorities

Based on research and project goals, suggest this order:

### Week 1-2: Foundation
```
Priority: High
Tasks:
- [ ] Install Ollama + download models (Qwen2.5-VL, DeepSeek-V3)
- [ ] Set up LangGraph agent framework
- [ ] Create tool registration system
- [ ] Test basic LLM decision flow

Files to create:
- scripts/ai_editing/decision_engine/llm_engine.py
- scripts/ai_editing/decision_engine/tool_registry.py
- scripts/ai_editing/decision_engine/agent_graph.py
```

### Week 3-4: 3D Character Tools
```
Priority: High
Tasks:
- [ ] Integrate SDXL + LoRA loading
- [ ] Set up GPT-SoVITS for voice cloning
- [ ] Connect to LoRA pipeline models
- [ ] Test character generation consistency

Files to create:
- scripts/generation/image/sdxl_lora_generator.py
- scripts/synthesis/tts/gpt_sovits_wrapper.py
- scripts/core/utils/lora_loader.py
```

### Week 5-6: Agent Decision System
```
Priority: High
Tasks:
- [ ] Implement ReAct reasoning loop
- [ ] Create quality evaluation system
- [ ] Add automatic iteration logic
- [ ] Test end-to-end creative workflow

Files to create:
- scripts/ai_editing/decision_engine/react_agent.py
- scripts/ai_editing/decision_engine/quality_evaluator.py
```

### Week 7-8: Integration & Applications
```
Priority: Medium
Tasks:
- [ ] Build parody generator
- [ ] Create voice synthesis pipeline
- [ ] Implement multimodal analysis
- [ ] User interface (optional)

Files to create:
- scripts/ai_editing/style_remix/parody_generator.py
- scripts/applications/creative_studio_app.py
```

---

## 🔗 Related Projects Context

### 3D Animation LoRA Pipeline

**Location:** `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline`

**Relationship:**
- Shares film datasets
- Provides trained LoRA models
- Common character metadata
- Shared AI warehouse

**Current Status:**
- Luca SAM2 segmentation: 14.8% complete
- Smart batch launcher: Waiting for GPU
- Next: LaMa inpainting + batch processing

**DO NOT interfere with running processes!**

---

## 📚 Research Findings Summary

Comprehensive SOTA research was completed covering:

1. **Multi-Type LoRA Training**
   - Character, Background, Pose, Emotion, Style LoRAs
   - Best practices for 3D animation
   - Multi-LoRA composition strategies

2. **Open-Source LLMs**
   - Qwen2.5-VL: Multimodal understanding
   - DeepSeek-V3: Superior reasoning
   - Deployment via Ollama

3. **Voice Synthesis**
   - GPT-SoVITS: 1-min voice cloning
   - EmoKnob: Emotion control
   - Real-time performance

4. **Lip-Sync**
   - Wav2Lip variants
   - VividWav2Lip improvements
   - UniTalker for 3D faces

5. **LLM-Powered Editing**
   - Automated decision making
   - Tool selection strategies
   - Quality evaluation

6. **Parody Generation**
   - Expression exaggeration
   - Comedy timing analysis
   - Meme-style effects

**Full research report:** See previous conversation or ask for summary

---

## 🎨 3D Animation Specific Settings

### Critical Parameters (DO NOT CHANGE without reason)

```yaml
Segmentation:
  alpha_threshold: 0.15    # Soft anti-aliased edges
  blur_threshold: 80       # Tolerate DoF blur
  min_size: 128           # Minimum character size

Clustering:
  min_cluster_size: 10-15  # Smaller than 2D anime
  min_samples: 2          # Tighter identity clusters

Training:
  dataset_size: 200-500    # Fewer images needed for 3D
  color_jitter: false     # Breaks PBR materials
  horizontal_flip: false  # Breaks asymmetric details
  caption_length: 40-77   # Stable with SD trainers

LoRA:
  learning_rate: 1e-4 to 2e-4
  network_rank: 32-64
  epochs: 10-20
```

### Prompt Engineering for 3D

```yaml
Positive Keywords:
  - "pixar style"
  - "3d animation"
  - "smooth shading"
  - "pbr materials"
  - "cinematic lighting"
  - "rendered"

Negative Keywords:
  - "2d"
  - "flat"
  - "anime"
  - "sketchy"
  - "low quality"
  - "watermark"
```

---

## 🚀 Quick Start Commands

### Terminal Setup

```bash
# Open new WSL terminal
cd /mnt/c/AI_LLM_projects/animation-ai-studio

# Activate conda environment (shared with LoRA pipeline)
conda activate ai_env

# Or create new environment
conda create -n ai_studio python=3.10
conda activate ai_studio
```

### Install Dependencies

```bash
# Install core requirements
pip install -r requirements/core.txt

# Install LangGraph
pip install langgraph

# Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh
```

### Download Models

```bash
# Start Ollama
ollama serve &

# Download LLMs (this will take time!)
ollama pull qwen2.5-vl:72b        # ~48GB
ollama pull deepseek-v3:671b      # ~80GB with FP8
ollama pull qwen2.5-coder:32b     # ~20GB
```

### Test Setup

```bash
# Test Ollama
ollama list

# Test LangGraph
python -c "from langgraph.prebuilt import create_react_agent; print('✅ LangGraph ready')"

# Test GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 💡 Common Questions & Answers

**Q: Can I use GPT-4 or Claude for development?**
A: No. All core functionality must use open-source models (Qwen, DeepSeek). This is a hard requirement.

**Q: What if open-source models aren't good enough?**
A: Use Qwen2.5-VL 72B and DeepSeek-V3. They are comparable to GPT-4 in many tasks. If specific limitations exist, document them and we'll find solutions.

**Q: Can I create new tools or must I use existing ones?**
A: You can create new tools, but prioritize integration of existing open-source tools first. Document all new tools in the tool registry.

**Q: How do I access the LoRA pipeline's data?**
A: Shared resources are in:
- Film data: `/mnt/data/ai_data/datasets/3d-anime/`
- Models: `/mnt/c/AI_LLM_projects/ai_warehouse/`
- Character info: `data/films/`

**Q: What if I need to modify the LoRA pipeline?**
A: Coordinate with that project. The AI Studio focuses on content creation/analysis, LoRA pipeline on training.

---

## 🎯 Success Criteria

You'll know the project is progressing well when:

1. **Week 2:** LLM agent can understand creative intent and suggest tools
2. **Week 4:** Can generate 3D character images with LoRA consistency
3. **Week 6:** LLM makes autonomous editing decisions and iterates
4. **Week 8:** End-to-end parody generation works automatically

---

## 📞 Getting Help

**Documentation:**
- CLAUDE.md - Complete project instructions
- PROJECT_STATUS.md - Current status
- OPEN_SOURCE_MODELS.md - All tools reference

**External Resources:**
- LangGraph: https://langchain-ai.github.io/langgraph/
- Ollama: https://ollama.com/
- Qwen2.5: https://github.com/QwenLM/Qwen2.5

**Previous Research:**
- Full SOTA research report in previous conversation
- Can be summarized on request

---

## ✅ Checklist for New Claude Code Session

Before starting work, ensure Claude Code understands:

- [ ] Project uses ONLY open-source models
- [ ] LLM agents make creative decisions (not just tools)
- [ ] Optimized for 3D animation (specific parameters)
- [ ] Agent framework is LangGraph
- [ ] Current status is "ready for implementation"
- [ ] Shared resources with LoRA pipeline
- [ ] Implementation priority order (Week 1-8)

---

**Ready to start? Ask Claude Code to begin with Week 1-2 foundation tasks!**
