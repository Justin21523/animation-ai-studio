# Video Analysis Module (Module 7)

Comprehensive video analysis tools for intelligent video understanding and quality assessment.

## Features

### 1. Scene Detection
- **Content-aware scene detection** using PySceneDetect
- **Adaptive thresholding** for optimal results
- **Keyframe extraction** (representative frame per scene)
- **JSON output** for Agent Framework integration

**Use Cases:**
- Automatic scene segmentation for editing
- Keyframe extraction for thumbnails
- Content-based video organization

### 2. Composition Analysis
- **Rule of thirds** compliance scoring
- **Visual balance** analysis (left/right, top/bottom)
- **Depth layer detection** (foreground, midground, background)
- **Subject position identification**
- **Power point usage** (intersection of thirds)

**Use Cases:**
- Evaluate video composition quality
- Identify well-composed frames
- Provide composition feedback for AI editing

### 3. Camera Movement Tracking
- **Pan detection** (horizontal movement)
- **Tilt detection** (vertical movement)
- **Zoom detection**
- **Camera style classification** (static, smooth, dynamic, handheld)
- **Movement velocity** and smoothness metrics

**Use Cases:**
- Characterize camera work style
- Detect handheld vs tripod footage
- Analyze camera movement for editing decisions

### 4. Temporal Coherence Checking
- **Color stability** across frames
- **Motion smoothness** analysis
- **Flicker detection**
- **Abrupt transition detection**
- **Frame-to-frame similarity** (SSIM)
- **Temporal artifacts** (ghosting, judder)

**Use Cases:**
- **Critical for AI-generated videos** where temporal coherence is often poor
- Quality assessment of video generation
- Detect and quantify temporal artifacts

## Installation

Required dependencies:
```bash
pip install opencv-python numpy scipy scenedetect[opencv]
```

All dependencies are already included in the project environment.

## Usage

### 1. Scene Detection

```python
from scripts.analysis.video.scene_detector import SceneDetector

detector = SceneDetector(
    threshold=27.0,
    min_scene_length=15,
    adaptive_threshold=True
)

result = detector.detect(
    video_path="path/to/video.mp4",
    extract_keyframes=True
)

# Access results
print(f"Detected {result.total_scenes} scenes")
for scene in result.scenes:
    print(f"Scene {scene.scene_id}: {scene.start_time:.2f}s - {scene.end_time:.2f}s")
    print(f"  Keyframe: {scene.keyframe_path}")

# Save to JSON
result.save_json("outputs/scene_detection.json")
```

### 2. Composition Analysis

```python
from scripts.analysis.video.composition_analyzer import CompositionAnalyzer

analyzer = CompositionAnalyzer(enable_visualization=True)

result = analyzer.analyze_video(
    video_path="path/to/video.mp4",
    sample_rate=30,  # Analyze every 30th frame
    visualization_output_dir="outputs/composition_vis"
)

# Access results
print(f"Average composition score: {result.avg_composition_score:.3f}")
print(f"Best frame: {result.best_composition_frame}")

# Get detailed metrics for specific frame
first_frame = result.frame_metrics[0]
print(f"Rule of thirds score: {first_frame.rule_of_thirds.overall_score:.3f}")
print(f"Visual balance: {first_frame.visual_balance.overall_balance_score:.3f}")
print(f"Depth complexity: {first_frame.depth_layers.depth_complexity:.3f}")

result.save_json("outputs/composition_analysis.json")
```

### 3. Camera Movement Tracking

```python
from scripts.analysis.video.camera_movement_tracker import CameraMovementTracker

tracker = CameraMovementTracker()

result = tracker.track_video(
    video_path="path/to/video.mp4",
    sample_interval=1  # Analyze every frame
)

# Access results
print(f"Camera style: {result.camera_style}")
print(f"Total shots: {len(result.shots)}")
print(f"Static duration: {result.total_static_duration:.2f}s")
print(f"Moving duration: {result.total_moving_duration:.2f}s")

# Examine individual shots
for shot in result.shots:
    print(f"Shot {shot.shot_id}: {shot.dominant_movement}")
    print(f"  Duration: {shot.duration:.2f}s")
    print(f"  Handheld: {shot.is_handheld}")
    print(f"  Smoothness: {shot.smoothness_score:.3f}")

result.save_json("outputs/camera_tracking.json")
```

### 4. Temporal Coherence Checking

```python
from scripts.analysis.video.temporal_coherence_checker import TemporalCoherenceChecker

checker = TemporalCoherenceChecker()

result = checker.check_video(
    video_path="path/to/video.mp4",
    sample_interval=1
)

# Access results
print(f"Temporal stability: {result.temporal_stability_rating}")
print(f"Average coherence: {result.avg_coherence_score:.3f}")
print(f"Flicker frames: {result.total_flicker_frames}")
print(f"Abrupt transitions: {result.total_abrupt_transitions}")
print(f"Problem segments: {len(result.problem_segments)}")

# Examine segment quality
for segment in result.segments:
    print(f"Segment {segment.segment_id}: {segment.quality_rating}")
    print(f"  Average coherence: {segment.avg_coherence_score:.3f}")
    print(f"  Issues: {segment.has_issues}")

result.save_json("outputs/temporal_coherence.json")
```

### 5. Agent Framework Integration

```python
from scripts.agent.tools.video_analysis_tools import (
    detect_scenes,
    analyze_composition,
    track_camera_movement,
    check_temporal_coherence,
    analyze_video_complete
)

# Run individual analysis
scene_result = await detect_scenes("path/to/video.mp4")
print(scene_result)

# Run complete analysis suite
complete_result = await analyze_video_complete(
    video_path="path/to/video.mp4",
    scene_detection=True,
    composition_analysis=True,
    camera_tracking=True,
    temporal_coherence=True,
    sample_rate=30
)

# Access summary
print(complete_result["summary"])
```

## CLI Usage

All modules can be run standalone for testing:

```bash
# Scene Detection
python scripts/analysis/video/scene_detector.py

# Composition Analysis
python scripts/analysis/video/composition_analyzer.py

# Camera Tracking
python scripts/analysis/video/camera_movement_tracker.py

# Temporal Coherence
python scripts/analysis/video/temporal_coherence_checker.py
```

## Running Tests

```bash
python scripts/analysis/video/test_video_analysis.py
```

**Note:** Edit the test file to provide a valid video path before running.

## Output Format

All analysis tools output **structured JSON** that is compatible with the Agent Framework:

```json
{
  "video_path": "path/to/video.mp4",
  "success": true,
  "total_scenes": 15,
  "avg_composition_score": 0.762,
  "camera_style": "smooth",
  "temporal_stability_rating": "excellent",
  "summary": {
    "scenes": {
      "total_scenes": 15,
      "avg_duration": "3.45s",
      "description": "Video contains 15 distinct scenes..."
    },
    "composition": {
      "quality": "good",
      "score": "0.762",
      "description": "Composition quality is good..."
    },
    "camera": {
      "style": "smooth",
      "shots": 8,
      "description": "Camera style is smooth..."
    },
    "temporal_quality": {
      "rating": "excellent",
      "score": "0.923",
      "description": "Temporal stability is excellent..."
    }
  }
}
```

## Performance Considerations

### CPU-Based Processing
All video analysis modules are **CPU-based** and do not require GPU:
- **Scene Detection:** ~30-60s per minute of video
- **Composition Analysis:** ~45-90s per minute (sample_rate=30)
- **Camera Tracking:** ~60-120s per minute
- **Temporal Coherence:** ~90-180s per minute

**Tip:** Use `sample_rate` parameter to speed up frame-based analyses.

### Memory Usage
- **Scene Detection:** ~500MB RAM
- **Composition Analysis:** ~800MB RAM
- **Camera Tracking:** ~600MB RAM
- **Temporal Coherence:** ~700MB RAM

All analyses can run simultaneously without conflicts.

## Integration with Agent Framework

Video analysis tools are registered in the Agent Framework tool registry:

- **Tool Category:** `VIDEO_ANALYSIS`
- **Registered Tools:**
  - `detect_scenes`
  - `analyze_composition`
  - `track_camera_movement`
  - `check_temporal_coherence`
  - `analyze_video_complete`

The LLM can automatically select and use these tools when analyzing videos.

## Architecture

```
scripts/analysis/video/
├── scene_detector.py              # PySceneDetect wrapper
├── composition_analyzer.py         # Visual composition analysis
├── camera_movement_tracker.py      # Optical flow-based tracking
├── temporal_coherence_checker.py   # Frame-to-frame consistency
├── test_video_analysis.py          # Test suite
└── README.md                       # This file

scripts/agent/tools/
└── video_analysis_tools.py         # Agent Framework wrappers
```

## Future Enhancements

Potential additions to Module 7:
- **Object detection** (YOLO, SAM2) for character tracking
- **Action recognition** for content understanding
- **Audio-visual sync** analysis
- **Shot type classification** (close-up, medium, wide)
- **Editing pattern detection** (cuts, fades, transitions)
- **Multi-modal analysis** (combining audio + visual features)

## Technical Details

### Scene Detection
- **Method:** Content-based detection (color histogram changes)
- **Algorithm:** PySceneDetect ContentDetector
- **Threshold:** Configurable (default 27.0, lower = more sensitive)

### Composition Analysis
- **Rule of Thirds:** Edge detection along third lines
- **Visual Balance:** Pixel intensity distribution
- **Depth Layers:** Laplacian sharpness analysis
- **Subject Detection:** Saliency maps (spectral residual)

### Camera Tracking
- **Feature Detection:** Shi-Tomasi corners
- **Tracking:** Lucas-Kanade optical flow
- **Transformation:** Affine transformation estimation
- **Classification:** Heuristic-based movement type classification

### Temporal Coherence
- **SSIM:** Structural similarity index
- **Color Stability:** HSV color space analysis
- **Motion Smoothness:** Optical flow variance
- **Flicker Detection:** Brightness oscillation detection

## Dependencies

All dependencies are included in `requirements/video.txt`:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scenedetect >= 0.6.0

## License

Part of Animation AI Studio project.

## Author

Animation AI Studio Team
Date: 2025-11-17
