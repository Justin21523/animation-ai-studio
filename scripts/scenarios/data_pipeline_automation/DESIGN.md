# Data Pipeline Automation - Design Document

**Version:** 1.0
**Date:** 2025-12-03
**Status:** Design Phase

---

## Executive Summary

The **Data Pipeline Automation** system provides a comprehensive framework for defining, executing, and monitoring multi-stage data processing pipelines. It enables declarative pipeline definitions using DAG (Directed Acyclic Graph) structures, automatic stage orchestration, checkpoint management, and failure recovery.

**Target Use Cases:**
- Video → Frame Extraction → Segmentation → Clustering → Training Data
- Audio → Transcription → Subtitle Generation → Translation
- Multi-character dataset preparation workflows
- Batch processing with dependency management

**Key Features:**
- DAG-based pipeline definition with stage dependencies
- Automatic stage orchestration and execution
- Checkpoint/resume support for long-running pipelines
- Parallel stage execution where dependencies allow
- Real-time progress monitoring and event emission
- Integration with EventBus for system-wide coordination
- CLI and programmatic API

---

## Architecture Overview

### High-Level Flow

```
Pipeline Definition (YAML/Python)
    ↓
Pipeline Builder (validates DAG, resolves dependencies)
    ↓
Pipeline Orchestrator (schedules and executes stages)
    ↓
Stage Executors (frame extraction, segmentation, clustering, etc.)
    ↓
Checkpoint Manager (saves intermediate state)
    ↓
Progress Monitor (tracks metrics, emits events)
    ↓
Results (outputs, logs, metrics)
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline Automation                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐         ┌──────────────────┐            │
│  │ Pipeline       │────────▶│ Stage            │            │
│  │ Builder        │         │ Registry         │            │
│  └────────────────┘         └──────────────────┘            │
│         │                            │                       │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌────────────────┐         ┌──────────────────┐            │
│  │ DAG            │◀────────│ Stage            │            │
│  │ Validator      │         │ Executors        │            │
│  └────────────────┘         └──────────────────┘            │
│         │                            │                       │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌────────────────────────────────────────────┐             │
│  │      Pipeline Orchestrator                 │             │
│  │  - Topological sort                        │             │
│  │  - Parallel execution                      │             │
│  │  - Dependency resolution                   │             │
│  │  - Error handling & retry                  │             │
│  └────────────────────────────────────────────┘             │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌────────────────┐         ┌──────────────────┐            │
│  │ Checkpoint     │         │ Progress         │            │
│  │ Manager        │         │ Monitor          │            │
│  └────────────────┘         └──────────────────┘            │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌────────────────┐         ┌──────────────────┐            │
│  │ EventBus       │         │ CLI/API          │            │
│  │ Integration    │         │ Interface        │            │
│  └────────────────┘         └──────────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Pipeline Definition

**Purpose:** Declarative specification of processing workflows

**Features:**
- YAML-based pipeline definitions
- Stage dependencies (DAG structure)
- Input/output path configuration
- Stage-specific parameters
- Environment variables and secrets

**Example Pipeline Definition:**

```yaml
pipeline:
  name: "character_dataset_preparation"
  version: "1.0"

stages:
  - id: "extract_frames"
    type: "frame_extraction"
    depends_on: []
    config:
      input_video: "/path/to/video.mp4"
      output_dir: "/path/to/frames"
      mode: "scene"
      scene_threshold: 0.3
      quality: "high"

  - id: "segment_characters"
    type: "segmentation"
    depends_on: ["extract_frames"]
    config:
      input_dir: "{extract_frames.output_dir}"
      output_dir: "/path/to/segmented"
      model: "isnet"
      alpha_threshold: 0.15
      extract_characters: true

  - id: "cluster_identities"
    type: "clustering"
    depends_on: ["segment_characters"]
    config:
      input_dir: "{segment_characters.output_dir}/characters"
      output_dir: "/path/to/clustered"
      min_cluster_size: 12
      use_face_detection: true

  - id: "prepare_training_data"
    type: "training_data_prep"
    depends_on: ["cluster_identities"]
    config:
      input_dir: "{cluster_identities.output_dir}"
      output_dir: "/path/to/training_data"
      generate_captions: true
      target_size: 400
```

**Pipeline Object Structure:**

```python
@dataclass
class Pipeline:
    id: str
    name: str
    version: str
    stages: List[PipelineStage]
    config: Dict[str, Any]
    created_at: float
    state: PipelineState
    checkpoint_dir: Optional[Path] = None
```

### 2. Pipeline Builder

**Purpose:** Construct and validate pipeline DAGs

**Responsibilities:**
- Load pipeline definitions from YAML/Python
- Validate DAG structure (no cycles, valid dependencies)
- Resolve stage configurations
- Register custom stage executors
- Build execution plan

**Key Methods:**

```python
class PipelineBuilder:
    def load_from_yaml(self, path: Path) -> Pipeline
    def load_from_dict(self, config: dict) -> Pipeline
    def validate_dag(self, pipeline: Pipeline) -> bool
    def register_stage_executor(self, stage_type: str, executor_class: Type[StageExecutor])
    def build_execution_plan(self, pipeline: Pipeline) -> List[List[str]]  # Parallel groups
```

**Validation Checks:**
- No circular dependencies (topological sort succeeds)
- All dependencies exist
- Required configuration parameters present
- Input paths exist (optional validation)
- Stage types are registered

### 3. Stage Executors

**Purpose:** Execute individual pipeline stages

**Built-in Stage Types:**
1. **Frame Extraction** - Extract frames from video
2. **Segmentation** - Segment characters/backgrounds
3. **Clustering** - Cluster by identity/pose
4. **Training Data Prep** - Organize and caption datasets
5. **Custom** - User-defined stages

**Base Interface:**

```python
class StageExecutor(ABC):
    def __init__(self, stage_id: str, config: Dict[str, Any]):
        self.stage_id = stage_id
        self.config = config

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate stage configuration"""
        pass

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        """Execute stage and return results"""
        pass

    @abstractmethod
    def estimate_duration(self) -> float:
        """Estimate stage duration in seconds"""
        pass
```

**StageResult Structure:**

```python
@dataclass
class StageResult:
    stage_id: str
    status: ExecutionStatus  # SUCCESS, FAILED, SKIPPED
    duration: float
    outputs: Dict[str, Any]  # Stage outputs (paths, metrics, etc.)
    metrics: Dict[str, float]  # Performance metrics
    error_message: Optional[str] = None
    checkpoints: List[Path] = field(default_factory=list)
```

### 4. Pipeline Orchestrator

**Purpose:** Coordinate multi-stage execution

**Features:**
- Topological sort for execution order
- Parallel execution of independent stages
- Dependency resolution
- Checkpoint management
- Failure handling and retry
- Progress tracking

**Key Methods:**

```python
class PipelineOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.executor_registry: Dict[str, Type[StageExecutor]] = {}
        self.checkpoint_manager = CheckpointManager()
        self.progress_monitor = PipelineProgressMonitor()

    def execute_pipeline(self, pipeline: Pipeline, resume: bool = False) -> PipelineResult:
        """Execute complete pipeline"""
        pass

    def execute_stage(self, stage: PipelineStage, inputs: Dict[str, Any]) -> StageResult:
        """Execute single stage"""
        pass

    def resume_from_checkpoint(self, checkpoint_path: Path) -> Pipeline:
        """Resume pipeline from checkpoint"""
        pass
```

**Execution Algorithm:**

1. Load pipeline definition
2. Validate DAG structure
3. Build execution plan (topological sort + grouping)
4. For each execution group (parallel stages):
   - Check if stage completed (checkpoint exists)
   - If not completed:
     - Allocate resources
     - Execute stage
     - Save checkpoint
     - Update progress
5. Aggregate results and emit events

### 5. Checkpoint Manager

**Purpose:** Save and restore pipeline state

**Features:**
- Stage-level checkpoints
- Automatic checkpoint creation
- Resume from failure
- Checkpoint pruning (keep last N)
- Metadata tracking

**Checkpoint Structure:**

```json
{
  "pipeline_id": "character_dataset_prep_20251203_143022",
  "checkpoint_time": 1701614422.5,
  "completed_stages": ["extract_frames", "segment_characters"],
  "stage_outputs": {
    "extract_frames": {
      "output_dir": "/path/to/frames",
      "frame_count": 1523
    },
    "segment_characters": {
      "output_dir": "/path/to/segmented",
      "character_count": 428
    }
  },
  "pipeline_state": "running",
  "next_stage": "cluster_identities"
}
```

**Key Methods:**

```python
class CheckpointManager:
    def save_checkpoint(self, pipeline: Pipeline, completed_stages: List[str]) -> Path
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]
    def list_checkpoints(self, pipeline_id: str) -> List[Path]
    def prune_checkpoints(self, pipeline_id: str, keep_last: int = 3)
```

### 6. Progress Monitor

**Purpose:** Track and report pipeline progress

**Features:**
- Real-time stage progress tracking
- Metrics collection (throughput, ETA)
- EventBus integration
- Dashboard data aggregation
- Log streaming

**Tracked Metrics:**
- Pipeline progress percentage
- Current stage
- Stage duration
- Estimated time remaining
- Resource utilization
- Throughput (items/sec)

**Key Methods:**

```python
class PipelineProgressMonitor:
    def start_pipeline(self, pipeline: Pipeline)
    def update_stage_progress(self, stage_id: str, progress: float, metrics: dict)
    def complete_stage(self, stage_id: str, result: StageResult)
    def get_pipeline_status(self, pipeline_id: str) -> dict
    def emit_event(self, event_type: str, data: dict)
```

---

## Data Structures

### Enums

```python
class PipelineState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RUNNING = "running"

class StageType(Enum):
    FRAME_EXTRACTION = "frame_extraction"
    SEGMENTATION = "segmentation"
    CLUSTERING = "clustering"
    TRAINING_DATA_PREP = "training_data_prep"
    CUSTOM = "custom"
```

### Core Dataclasses

```python
@dataclass
class PipelineStage:
    id: str
    type: StageType
    depends_on: List[str]
    config: Dict[str, Any]
    status: ExecutionStatus = ExecutionStatus.PENDING
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class Pipeline:
    id: str
    name: str
    version: str
    stages: List[PipelineStage]
    config: Dict[str, Any]
    created_at: float
    state: PipelineState
    checkpoint_dir: Optional[Path] = None

    @property
    def progress_percent(self) -> float:
        completed = sum(1 for s in self.stages if s.status == ExecutionStatus.SUCCESS)
        return (completed / len(self.stages)) * 100 if self.stages else 0

@dataclass
class StageResult:
    stage_id: str
    status: ExecutionStatus
    duration: float
    outputs: Dict[str, Any]
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    checkpoints: List[Path] = field(default_factory=list)

@dataclass
class PipelineResult:
    pipeline_id: str
    status: PipelineState
    total_duration: float
    stage_results: List[StageResult]
    final_outputs: Dict[str, Any]
    error_message: Optional[str] = None
```

### Configuration

```python
@dataclass
class OrchestratorConfig:
    checkpoint_dir: Path = Path("/tmp/pipeline_checkpoints")
    enable_parallel_execution: bool = True
    max_parallel_stages: int = 4
    checkpoint_interval: int = 300  # Save checkpoint every 5 minutes
    enable_auto_retry: bool = True
    max_retries: int = 2
    event_bus_enabled: bool = True
```

---

## Stage Executors (Built-in)

### 1. Frame Extraction Stage

**Type:** `frame_extraction`

**Config Parameters:**
- `input_video`: Path to video file
- `output_dir`: Output directory for frames
- `mode`: Extraction mode (scene, interval, hybrid)
- `scene_threshold`: Scene detection threshold
- `quality`: Output quality (high, medium, low)

**Outputs:**
- `output_dir`: Path to extracted frames
- `frame_count`: Number of frames extracted

**Implementation:**
```python
class FrameExtractionExecutor(StageExecutor):
    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        # Call universal_frame_extractor.py
        # Return StageResult with outputs
        pass
```

### 2. Segmentation Stage

**Type:** `segmentation`

**Config Parameters:**
- `input_dir`: Directory of frames
- `output_dir`: Output directory
- `model`: Segmentation model (isnet, sam2, u2net)
- `alpha_threshold`: Alpha channel threshold
- `extract_characters`: Extract character instances

**Outputs:**
- `output_dir`: Path to segmented outputs
- `character_count`: Number of character instances

**Implementation:**
```python
class SegmentationExecutor(StageExecutor):
    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        # Call layered_segmentation.py
        # Return StageResult with outputs
        pass
```

### 3. Clustering Stage

**Type:** `clustering`

**Config Parameters:**
- `input_dir`: Directory of character instances
- `output_dir`: Output directory
- `min_cluster_size`: Minimum cluster size
- `use_face_detection`: Enable face detection

**Outputs:**
- `output_dir`: Path to clustered outputs
- `cluster_count`: Number of clusters
- `cluster_sizes`: List of cluster sizes

**Implementation:**
```python
class ClusteringExecutor(StageExecutor):
    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        # Call character_clustering.py
        # Return StageResult with outputs
        pass
```

### 4. Training Data Prep Stage

**Type:** `training_data_prep`

**Config Parameters:**
- `input_dir`: Directory of clustered data
- `output_dir`: Output directory
- `generate_captions`: Generate captions
- `target_size`: Target dataset size

**Outputs:**
- `output_dir`: Path to training data
- `dataset_size`: Number of training examples

---

## CLI Interface

### Commands

```bash
# Create pipeline from YAML
python -m scripts.scenarios.data_pipeline_automation create \
  --config pipeline_definition.yaml \
  --output-dir /path/to/outputs

# Execute pipeline
python -m scripts.scenarios.data_pipeline_automation run \
  --pipeline-id PIPELINE_ID

# Resume from checkpoint
python -m scripts.scenarios.data_pipeline_automation resume \
  --checkpoint /path/to/checkpoint.json

# Show pipeline status
python -m scripts.scenarios.data_pipeline_automation status \
  --pipeline-id PIPELINE_ID

# List pipelines
python -m scripts.scenarios.data_pipeline_automation list \
  --state running

# Cancel pipeline
python -m scripts.scenarios.data_pipeline_automation cancel \
  --pipeline-id PIPELINE_ID
```

---

## Integration Points

### EventBus Integration

```python
# Pipeline events
events.emit("pipeline.started", {"pipeline_id": "...", "name": "..."})
events.emit("pipeline.stage_completed", {"stage_id": "...", "duration": 123.5})
events.emit("pipeline.completed", {"pipeline_id": "...", "duration": 3600.0})
events.emit("pipeline.failed", {"pipeline_id": "...", "error": "..."})
```

### Safety System Integration

- Memory monitoring during stage execution
- Resource guards for CPU/GPU/disk
- Automatic pause on resource exhaustion

---

## Error Handling

### Failure Recovery

1. **Stage Failure:**
   - Log error and metrics
   - Save checkpoint at last successful stage
   - Optionally retry (configurable)
   - Emit failure event

2. **Pipeline Failure:**
   - Save checkpoint
   - Preserve partial outputs
   - Generate error report
   - Allow manual intervention

3. **Resume Strategy:**
   - Load checkpoint
   - Skip completed stages
   - Re-execute from failure point

### Retry Policy

```python
@dataclass
class RetryPolicy:
    max_retries: int = 2
    retry_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    retry_on_errors: List[str] = field(default_factory=lambda: ["timeout", "resource_exhausted"])
```

---

## Performance Considerations

### Parallel Execution

- Identify independent stages (no dependencies between them)
- Execute in parallel using ThreadPoolExecutor
- Resource-aware scheduling (don't exceed max_parallel_stages)

### Checkpoint Optimization

- Save checkpoints asynchronously
- Compress checkpoint data
- Prune old checkpoints automatically

### Progress Monitoring

- Update progress every N items processed
- Use sampling for large datasets
- Cache metrics to reduce computation

---

## Testing Strategy

### Unit Tests
- Stage executor validation
- DAG validation logic
- Checkpoint save/load
- Progress calculation

### Integration Tests
- End-to-end pipeline execution
- Checkpoint resume
- Failure recovery
- Parallel stage execution

### Performance Tests
- Large-scale pipeline (1000+ stages)
- Checkpoint overhead
- Parallel execution speedup

---

## Future Enhancements

1. **Dynamic Resource Allocation:**
   - Adjust parallel execution based on available resources
   - GPU-aware stage scheduling

2. **Pipeline Templates:**
   - Pre-built templates for common workflows
   - Template parameterization

3. **Web Dashboard:**
   - Real-time pipeline visualization
   - Interactive stage inspection
   - Manual intervention controls

4. **Distributed Execution:**
   - Execute stages across multiple machines
   - Shared checkpoint storage

5. **Advanced Scheduling:**
   - Cost-based optimization
   - Priority-based stage scheduling
   - Resource affinity

---

## Dependencies

- **Core:**
  - Python 3.10+
  - pyyaml
  - dataclasses
  - typing
  - pathlib

- **Integration:**
  - EventBus (from core.orchestration)
  - Safety System (from core.safety)
  - Logger (from core.utils)

---

## File Structure

```
scripts/scenarios/data_pipeline_automation/
├── common.py                    # Enums, dataclasses, helpers
├── pipeline_builder.py          # Pipeline definition and validation
├── stage_executors/
│   ├── base_executor.py        # Base StageExecutor interface
│   ├── frame_extraction.py     # Frame extraction stage
│   ├── segmentation.py         # Segmentation stage
│   ├── clustering.py           # Clustering stage
│   ├── training_data_prep.py   # Training data prep stage
│   └── __init__.py
├── orchestrator/
│   ├── pipeline_orchestrator.py  # Main orchestrator
│   ├── checkpoint_manager.py     # Checkpoint management
│   ├── progress_monitor.py       # Progress tracking
│   └── __init__.py
├── __main__.py                  # CLI interface
├── __init__.py
├── DESIGN.md                    # This document
└── IMPLEMENTATION_SUMMARY.md    # Implementation tracking
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-03
**Status:** Ready for implementation
