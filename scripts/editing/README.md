# Module 8: Video Editing

**AI-Driven Autonomous Video Editing System**

Module 8 provides intelligent video editing capabilities powered by LLM decision-making, combining computer vision (SAM2), professional editing operations (MoviePy), and quality-driven iterative improvement.

## 🎯 Core Innovation

**LLM as Editor Brain**: Unlike traditional video editors, Module 8 uses LLMs to make ALL editing decisions autonomously based on user goals and video analysis.

```
User Goal: "Create funny 30s highlight reel"
         ↓
    LLM Analyzes Video
         ↓
    LLM Plans Edits
         ↓
    Execute Operations
         ↓
    LLM Evaluates Quality
         ↓
    Iterate Until Perfect
```

## 📁 Architecture

```
scripts/editing/
├── segmentation/          # Character segmentation & tracking
│   └── character_segmenter.py
├── engine/               # Video editing operations
│   └── video_editor.py
├── decision/             # LLM decision-making (CORE INNOVATION)
│   └── llm_decision_engine.py
├── quality/              # Quality evaluation
│   └── quality_evaluator.py
├── effects/              # Special effects
│   └── parody_generator.py
└── tests/                # Test suite
    └── test_module8.py
```

## 🔧 Components

### 1. Character Segmenter (`segmentation/character_segmenter.py`)

**Purpose**: Segment and track characters in video using SAM2

**Key Features**:
- Integrates LoRA pipeline's SAM2InstanceSegmenter
- Video-specific character tracking
- IoU-based matching across frames
- Temporal consistency validation

**Usage**:
```python
from scripts.editing.segmentation.character_segmenter import CharacterSegmenter

segmenter = CharacterSegmenter(model_size="base")
result = segmenter.segment_video(
    video_path="video.mp4",
    sample_interval=1,
    track_characters=True
)

print(f"Found {len(result.character_tracks)} characters")
for track in result.character_tracks:
    print(f"  Character {track.character_id}: {track.total_segments} segments")
```

**Model Sizes**:
- `tiny`: 2GB VRAM, fastest
- `small`: 4GB VRAM, good balance
- `base`: 6GB VRAM, **recommended for RTX 5080**
- `large`: 16GB VRAM, best quality (uses entire GPU)

### 2. Video Editor (`engine/video_editor.py`)

**Purpose**: Professional video editing operations using MoviePy

**Operations**:
- `cut_clip`: Extract segments
- `change_speed`: Slow motion / fast forward
- `composite_layers`: Multi-layer compositing
- `concatenate_clips`: Join clips
- `add_text_overlay`: Add text
- `apply_effect`: Apply visual effects

**Usage**:
```python
from scripts.editing.engine.video_editor import VideoEditor

editor = VideoEditor()

# Cut segment
result = editor.cut_clip(
    video_path="video.mp4",
    start_time=10.0,
    end_time=30.0,
    output_path="segment.mp4"
)

# Slow motion
result = editor.change_speed(
    video_path="video.mp4",
    speed_factor=0.5,  # 0.5x = half speed
    output_path="slow_mo.mp4"
)

# Composite layers
result = editor.composite_layers(
    layers=[
        {"video_path": "background.mp4", "position": (0, 0), "z_index": 0},
        {"video_path": "character.mp4", "position": (100, 50), "z_index": 1}
    ],
    output_path="composite.mp4"
)
```

### 3. LLM Decision Engine (`decision/llm_decision_engine.py`) ⭐

**Purpose**: AI brain that makes ALL editing decisions

**Key Methods**:
- `create_edit_plan`: Generate complete edit plan from goal
- `evaluate_edit_quality`: Assess quality of edited video
- `suggest_improvements`: Propose refinements

**Workflow**:
```python
from scripts.editing.decision.llm_decision_engine import LLMDecisionEngine

async with LLMDecisionEngine() as engine:
    # Create edit plan
    plan = await engine.create_edit_plan(
        video_path="video.mp4",
        goal="Create a funny 30-second highlight reel with dramatic moments",
        analysis_results=analysis_from_module7
    )

    print(f"Plan: {len(plan.decisions)} decisions")
    for decision in plan.decisions:
        print(f"  {decision.decision_type}: {decision.reasoning}")
        print(f"    Confidence: {decision.confidence:.2f}")
        print(f"    Parameters: {decision.parameters}")
```

**Decision Types**:
- `cut`: Extract segment
- `speed`: Change playback speed
- `composite`: Layer compositing
- `concatenate`: Join clips
- `text_overlay`: Add text
- `effect`: Apply visual effect

**Example Edit Plan**:
```json
{
  "goal": "Create funny highlight reel",
  "decisions": [
    {
      "decision_type": "cut",
      "reasoning": "Extract the funniest moment where Luca discovers pasta",
      "confidence": 0.92,
      "parameters": {
        "start_time": 45.2,
        "end_time": 52.8
      }
    },
    {
      "decision_type": "speed",
      "reasoning": "Slow down for dramatic effect",
      "confidence": 0.85,
      "parameters": {
        "speed_factor": 0.7,
        "segment": [48.0, 50.0]
      }
    }
  ]
}
```

### 4. Quality Evaluator (`quality/quality_evaluator.py`)

**Purpose**: Evaluate video quality with technical and creative metrics

**Metrics**:
- **Technical**: Composition, temporal coherence, pacing
- **Creative**: Goal achievement, artistic quality
- **Overall**: Weighted combination

**Usage**:
```python
from scripts.editing.quality.quality_evaluator import QualityEvaluator

evaluator = QualityEvaluator()
metrics = evaluator.evaluate(
    video_path="edited.mp4",
    goal="Create funny highlight reel",
    quality_threshold=0.7
)

print(f"Overall Score: {metrics.overall_score:.3f}")
print(f"Needs Improvement: {metrics.needs_improvement}")
print(f"Feedback: {metrics.feedback}")

if metrics.suggestions:
    print("Suggestions:")
    for suggestion in metrics.suggestions:
        print(f"  - {suggestion}")
```

**Quality Thresholds**:
- `0.9+`: Excellent
- `0.7-0.9`: Good
- `0.5-0.7`: Acceptable but needs improvement
- `<0.5`: Poor, significant improvements needed

### 5. Parody Generator (`effects/parody_generator.py`)

**Purpose**: Create funny/parody videos with comedic effects

**Effects**:
- `zoom_punch`: Dramatic zoom in/out
- `speed_ramp`: Slow-mo and fast-forward segments
- `meme_style`: Automated meme video generation

**Usage**:
```python
from scripts.editing.effects.parody_generator import ParodyGenerator

generator = ParodyGenerator()

# Zoom punch at dramatic moment
result = generator.apply_zoom_punch(
    video_path="video.mp4",
    zoom_time=5.0,  # When to zoom
    output_path="funny.mp4",
    zoom_factor=1.5
)

# Speed ramping for comedy
result = generator.apply_speed_ramp(
    video_path="video.mp4",
    output_path="funny.mp4",
    slow_mo_segments=[(2.0, 5.0)],
    fast_segments=[(10.0, 15.0)]
)

# Auto meme video
result = generator.create_meme_video(
    video_path="video.mp4",
    output_path="meme.mp4",
    meme_style="dramatic"  # or "chaotic", "wholesome"
)
```

## 🤖 Agent Framework Integration

Module 8 provides 7 tools for autonomous agent workflows:

### Tool 1: `segment_characters`
```python
await segment_characters(
    video_path="video.mp4",
    model_size="base"
)
```

### Tool 2: `cut_video_clip`
```python
await cut_video_clip(
    video_path="video.mp4",
    start_time=10.0,
    end_time=30.0,
    output_path="segment.mp4"
)
```

### Tool 3: `change_video_speed`
```python
await change_video_speed(
    video_path="video.mp4",
    speed_factor=0.5,
    output_path="slow_mo.mp4"
)
```

### Tool 4: `create_edit_plan` ⭐
```python
await create_edit_plan(
    video_path="video.mp4",
    goal="Create funny 30s highlight reel",
    analysis_results=analysis
)
```

### Tool 5: `evaluate_video_quality`
```python
await evaluate_video_quality(
    video_path="edited.mp4",
    goal="Funny highlight reel",
    quality_threshold=0.7
)
```

### Tool 6: `create_parody_video`
```python
await create_parody_video(
    video_path="video.mp4",
    output_path="parody.mp4",
    parody_style="dramatic"
)
```

### Tool 7: `auto_edit_video` ⭐⭐⭐
**Complete autonomous workflow**:
```python
await auto_edit_video(
    video_path="video.mp4",
    goal="Create a funny 30-second highlight reel",
    output_path="final.mp4",
    quality_threshold=0.7,
    max_iterations=3,
    analyze_first=True
)
```

**What it does**:
1. Analyzes video (Module 7)
2. Creates LLM edit plan
3. Executes edits
4. Evaluates quality
5. Iterates if quality below threshold
6. Returns final high-quality result

## 🚀 Complete Workflows

### Workflow 1: Manual Editing Pipeline
```python
from scripts.editing.segmentation.character_segmenter import CharacterSegmenter
from scripts.editing.engine.video_editor import VideoEditor

# 1. Segment characters
segmenter = CharacterSegmenter()
seg_result = segmenter.segment_video("video.mp4")

# 2. Cut highlight segment
editor = VideoEditor()
cut_result = editor.cut_clip(
    video_path="video.mp4",
    start_time=10.0,
    end_time=40.0,
    output_path="highlight.mp4"
)

# 3. Add slow motion
final_result = editor.change_speed(
    video_path="highlight.mp4",
    speed_factor=0.7,
    output_path="final.mp4"
)
```

### Workflow 2: LLM-Driven Editing
```python
from scripts.editing.decision.llm_decision_engine import LLMDecisionEngine
from scripts.editing.engine.video_editor import VideoEditor
from scripts.editing.quality.quality_evaluator import QualityEvaluator

async def ai_edit():
    # 1. Create edit plan with LLM
    async with LLMDecisionEngine() as engine:
        plan = await engine.create_edit_plan(
            video_path="video.mp4",
            goal="Create dramatic 30s trailer"
        )

    # 2. Execute decisions
    editor = VideoEditor()
    for decision in plan.decisions:
        if decision.decision_type == "cut":
            editor.cut_clip(
                video_path="video.mp4",
                start_time=decision.parameters["start_time"],
                end_time=decision.parameters["end_time"],
                output_path="segment.mp4"
            )
        # ... execute other decisions

    # 3. Evaluate quality
    evaluator = QualityEvaluator()
    metrics = evaluator.evaluate(
        video_path="final.mp4",
        goal="Create dramatic 30s trailer",
        quality_threshold=0.7
    )

    if metrics.needs_improvement:
        print("Needs improvement:")
        for suggestion in metrics.suggestions:
            print(f"  - {suggestion}")
```

### Workflow 3: Fully Autonomous (Recommended)
```python
from scripts.agent.tools.video_editing_tools import auto_edit_video

# One function call does EVERYTHING
result = await auto_edit_video(
    video_path="video.mp4",
    goal="Create a funny 30-second highlight reel with best moments",
    output_path="final.mp4",
    quality_threshold=0.8,
    max_iterations=3,
    analyze_first=True
)

print(f"Success: {result['success']}")
print(f"Quality: {result['quality_evaluation']['overall_score']:.3f}")
print(f"Iterations: {result['total_iterations']}")
```

## 📊 Performance

### GPU Memory Usage (RTX 5080 16GB)

| Component | VRAM | Time (30s video) |
|-----------|------|------------------|
| SAM2 Base | 6GB | 60s |
| SAM2 Large | 16GB | 90s |
| MoviePy Operations | 0GB | 5-30s |
| LLM Decision | 0GB* | 10-15s |
| Quality Eval | 0GB | 5s |
| Parody Effects | 0GB | 20-40s |

*Uses LLM Backend (separate service)

### Recommended Configurations

**RTX 5080 16GB** (your setup):
- SAM2: `base` (6GB)
- Can run segmentation + editing simultaneously
- Total pipeline: ~2-3 minutes for 30s video

**Lower VRAM**:
- SAM2: `small` (4GB) or `tiny` (2GB)
- Reduce sample_interval for faster processing

**Higher VRAM** (24GB+):
- SAM2: `large` (16GB) for best quality
- Can run multiple tools in parallel

## 🧪 Testing

Run test suite:
```bash
# All tests
python scripts/editing/tests/test_module8.py

# Specific test class
pytest scripts/editing/tests/test_module8.py::TestLLMDecisionEngine -v

# With coverage
pytest scripts/editing/tests/test_module8.py --cov=scripts.editing
```

## 🔗 Integration with Other Modules

### Module 7: Video Analysis
```python
# Module 7 provides analysis
from scripts.agent.tools.video_analysis_tools import analyze_video_complete

analysis = await analyze_video_complete(video_path="video.mp4")

# Module 8 uses analysis for decisions
plan = await create_edit_plan(
    video_path="video.mp4",
    goal="Create highlight reel",
    analysis_results=analysis["analyses"]
)
```

### LoRA Pipeline: SAM2 Integration
```python
# Module 8 reuses LoRA pipeline's SAM2
from scripts.editing.segmentation.character_segmenter import CharacterSegmenter

# Under the hood, imports:
# from 3d-animation-lora-pipeline.scripts.generic.segmentation.instance_segmentation import SAM2InstanceSegmenter
```

## 📝 Configuration

Edit `configs/editing_config.yaml`:
```yaml
editing:
  segmentation:
    model_size: base  # tiny, small, base, large
    device: cuda
    sample_interval: 1
    min_mask_size: 100

  decision_engine:
    model: qwen2.5-14b-instruct  # LLM for decisions
    temperature: 0.7
    quality_threshold: 0.7
    max_iterations: 3

  quality:
    enable_automated_checks: true
    enable_llm_evaluation: true

  effects:
    default_zoom_factor: 1.5
    default_speed_slow: 0.5
    default_speed_fast: 2.0
```

## 🎬 Example: Create Funny Parody

```python
import asyncio
from scripts.agent.tools.video_editing_tools import auto_edit_video

async def create_funny_video():
    result = await auto_edit_video(
        video_path="data/films/luca/scenes/pasta_discovery.mp4",
        goal="Create hilarious parody with zoom punches and slow motion at funniest moments",
        output_path="outputs/parody/luca_pasta_funny.mp4",
        quality_threshold=0.8,
        max_iterations=3,
        analyze_first=True
    )

    if result["success"]:
        print("✅ Parody created successfully!")
        print(f"📊 Quality: {result['quality_evaluation']['overall_score']:.3f}")
        print(f"🔄 Iterations: {result['total_iterations']}")
    else:
        print(f"❌ Failed: {result['error']}")

asyncio.run(create_funny_video())
```

## 🚨 Troubleshooting

### Out of VRAM
```python
# Solution 1: Use smaller SAM2 model
segmenter = CharacterSegmenter(model_size="small")  # or "tiny"

# Solution 2: Increase sample interval
result = segmenter.segment_video(sample_interval=5)  # Process every 5th frame

# Solution 3: Don't run segmentation + generation simultaneously
```

### MoviePy Errors
```bash
# Install MoviePy
pip install moviepy

# For audio support
pip install moviepy[optional]
```

### LLM Decision Quality Poor
```python
# Solution 1: Use better LLM
engine = LLMDecisionEngine(model="qwen2.5-14b-instruct")  # vs 7b

# Solution 2: Provide analysis results
plan = await engine.create_edit_plan(
    video_path="video.mp4",
    goal="...",
    analysis_results=module7_analysis  # More context
)

# Solution 3: Be more specific in goal
goal = "Create a 30-second funny highlight reel focusing on the pasta eating scene with dramatic slow motion"
```

## 📚 References

- **SAM2**: [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2)
- **MoviePy**: [Video editing library](https://zulko.github.io/moviepy/)
- **LoRA Pipeline SAM2**: `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/generic/segmentation/`

## 📄 License

Part of Animation AI Studio - See main project LICENSE

---

**Module Status**: ✅ Complete (2025-11-17)

**Dependencies**:
- Module 7: Video Analysis (provides analysis input)
- LoRA Pipeline: SAM2 implementation (code reuse)
- LLM Backend: Decision-making (inference service)

**Next**: Module 9 or Week 3-4 implementation
