# Data Pipeline Automation

**Status:** Core framework complete (2,541 LOC / 115.5% of target)
**Author:** Animation AI Studio
**Date:** 2025-12-03

---

## Overview

A DAG-based pipeline orchestration system for automating multi-stage data processing workflows. The system provides:

- **YAML-based configuration** for declarative pipeline definitions
- **Dependency management** with topological sorting and parallel execution
- **Checkpoint/resume capability** for long-running pipelines
- **Progress tracking** with ETA calculation
- **Extensible executor framework** for custom stage types
- **CLI interface** for pipeline operations

---

## Features

### Core Capabilities

1. **DAG Validation**
   - Automatic cycle detection using DFS algorithm
   - Dependency existence validation
   - Topological sorting using Kahn's algorithm

2. **Parallel Execution**
   - Automatic identification of independent stages
   - ThreadPoolExecutor-based parallel execution
   - Configurable parallelism limits

3. **Checkpoint Management**
   - Automatic checkpoint saving after each stage group
   - Resume from failure point
   - Checkpoint pruning (keep last N)

4. **Progress Monitoring**
   - Real-time progress tracking
   - ETA calculation based on historical performance
   - Per-stage metrics collection

5. **Template Variables**
   - Reference outputs from previous stages: `{stage_id.output_key}`
   - Automatic variable resolution during execution

---

## Architecture

```
data_pipeline_automation/
├── common.py                    # Core types and utilities (592 LOC)
├── __init__.py                  # Package exports (74 LOC)
├── pipeline_builder.py          # Pipeline construction (422 LOC)
├── validators.py                # Validation utilities (144 LOC)
├── stage_executors/
│   ├── base_executor.py        # Abstract base class (266 LOC)
│   └── __init__.py             # Executor exports (14 LOC)
├── orchestrator/
│   ├── pipeline_orchestrator.py # Main orchestrator (681 LOC)
│   └── __init__.py             # Orchestrator exports (20 LOC)
├── __main__.py                  # CLI interface (328 LOC)
├── examples/
│   └── simple_pipeline.yaml    # Example configuration
├── DESIGN.md                    # Architecture specification
├── IMPLEMENTATION_SUMMARY.md    # Implementation tracking
└── README.md                    # This file
```

**Total:** 2,541 LOC (115.5% of 2,200 target)

---

## Quick Start

### 1. Validate Pipeline Configuration

```bash
python -m scripts.scenarios.data_pipeline_automation validate \
  --config examples/simple_pipeline.yaml
```

### 2. Run Pipeline

```bash
python -m scripts.scenarios.data_pipeline_automation run \
  --config examples/simple_pipeline.yaml \
  --max-parallel 4
```

### 3. Check Pipeline Status

```bash
python -m scripts.scenarios.data_pipeline_automation status \
  --pipeline-id PIPELINE_ID
```

### 4. List All Pipelines

```bash
python -m scripts.scenarios.data_pipeline_automation list
```

---

## Pipeline Configuration

### Basic Structure

```yaml
name: my_pipeline
version: 1.0.0

stages:
  - id: stage_1
    type: frame_extraction
    depends_on: []
    config:
      input_video: /path/to/video.mp4
      output_dir: /path/to/frames

  - id: stage_2
    type: segmentation
    depends_on:
      - stage_1
    config:
      input_dir: "{stage_1.output_dir}"
      output_dir: /path/to/segmented
```

### Stage Types

Currently defined (executors to be implemented):

- `frame_extraction` - Extract frames from video
- `segmentation` - Segment characters from frames
- `clustering` - Cluster by character identity
- `training_data_prep` - Prepare training datasets
- `custom` - Custom stage type

### Template Variables

Reference outputs from previous stages:

```yaml
config:
  input_dir: "{previous_stage.output_dir}"
  character_dirs: "{clustering_stage.cluster_dirs}"
```

---

## CLI Commands

### validate

Validate pipeline configuration and show execution plan:

```bash
python -m scripts.scenarios.data_pipeline_automation validate \
  --config pipeline.yaml
```

**Output:**
- Validation results (DAG structure, dependencies)
- Execution plan (parallel stage groups)
- Pipeline summary (stages, dependencies)

### run

Execute pipeline:

```bash
python -m scripts.scenarios.data_pipeline_automation run \
  --config pipeline.yaml \
  --checkpoint-dir /path/to/checkpoints \
  --max-parallel 4 \
  --resume
```

**Options:**
- `--config` - Pipeline YAML configuration (required)
- `--checkpoint-dir` - Checkpoint directory (default: `checkpoints`)
- `--max-parallel` - Maximum parallel stages (default: 4)
- `--no-checkpoint` - Disable checkpoint saving
- `--resume` - Resume from latest checkpoint

### status

Show pipeline status:

```bash
python -m scripts.scenarios.data_pipeline_automation status \
  --pipeline-id PIPELINE_ID \
  --checkpoint-dir /path/to/checkpoints
```

**Output:**
- Pipeline name and ID
- Current state (RUNNING, COMPLETED, FAILED, etc.)
- Progress percentage
- Completed/total stages
- Elapsed time and ETA

### list

List all pipelines:

```bash
python -m scripts.scenarios.data_pipeline_automation list \
  --checkpoint-dir /path/to/checkpoints
```

**Output:**
- All pipeline IDs
- Number of checkpoints per pipeline
- Latest checkpoint timestamp

---

## Extending with Custom Executors

### 1. Implement Executor Class

```python
from scripts.scenarios.data_pipeline_automation.stage_executors import StageExecutor
from scripts.scenarios.data_pipeline_automation.common import StageResult, ExecutionStatus

class MyCustomExecutor(StageExecutor):
    def validate_config(self) -> bool:
        # Validate configuration
        required = ["input_dir", "output_dir"]
        for key in required:
            self._get_config_value(key, required=True)
        return True

    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        self._mark_started()

        # Get config values
        input_dir = self._get_config_value("input_dir")
        output_dir = self._get_config_value("output_dir")

        # Execute stage logic
        try:
            # ... your processing code ...

            self._mark_completed()
            return self._create_success_result(
                outputs={"output_dir": output_dir},
                metrics={"processed_files": 100}
            )

        except Exception as e:
            self._mark_completed()
            return self._create_failure_result(str(e))

    def estimate_duration(self) -> float:
        # Estimate execution time in seconds
        return 300.0  # 5 minutes
```

### 2. Register Executor

```python
from scripts.scenarios.data_pipeline_automation.orchestrator import PipelineOrchestrator
from my_custom_executor import MyCustomExecutor

orchestrator = PipelineOrchestrator(config)
orchestrator.register_executor("my_custom_type", MyCustomExecutor)
```

### 3. Use in Pipeline

```yaml
stages:
  - id: custom_stage
    type: my_custom_type
    depends_on: []
    config:
      input_dir: /path/to/input
      output_dir: /path/to/output
```

---

## Core Types

### Pipeline States

- `PENDING` - Pipeline created, not started
- `RUNNING` - Pipeline executing
- `COMPLETED` - Pipeline finished successfully
- `FAILED` - Pipeline failed
- `CANCELLED` - Pipeline cancelled by user
- `PAUSED` - Pipeline paused (future feature)

### Execution Status

- `SUCCESS` - Stage completed successfully
- `FAILED` - Stage failed
- `SKIPPED` - Stage skipped
- `RUNNING` - Stage currently executing

### Stage Types

- `FRAME_EXTRACTION` - Frame extraction from video
- `SEGMENTATION` - Character segmentation
- `CLUSTERING` - Identity clustering
- `TRAINING_DATA_PREP` - Training data preparation
- `CUSTOM` - Custom stage type

---

## Implementation Status

### Phase 1: Foundation & Common ✅ (666 LOC)
- Core enums and dataclasses
- Helper functions (DAG validation, topological sort, parallel grouping)
- Package structure

### Phase 2: Pipeline Builder ✅ (566 LOC)
- YAML/dict pipeline loading
- DAG validation with cycle detection
- Template variable resolution
- Execution plan generation

### Phase 3: Stage Executors ✅ (280 LOC - base framework)
- Abstract StageExecutor base class
- Common utilities (config validation, subprocess execution, result creation)
- **Note:** Concrete executors (frame extraction, segmentation, clustering, training prep) to be implemented

### Phase 4: Pipeline Orchestrator ✅ (701 LOC)
- Main orchestration engine
- Checkpoint manager
- Progress monitor
- Parallel execution with ThreadPoolExecutor

### Phase 5: CLI + Integration ✅ (328 LOC)
- CLI commands (validate, run, status, list)
- Argparse integration
- User-friendly output formatting

---

## Next Steps

### Immediate (Required for Production)

1. **Implement Concrete Executors** (~520 LOC)
   - `FrameExtractionExecutor` - Wrap `universal_frame_extractor.py`
   - `SegmentationExecutor` - Wrap `layered_segmentation.py`
   - `ClusteringExecutor` - Wrap `character_clustering.py`
   - `TrainingDataPrepExecutor` - Wrap `prepare_training_data.py`

2. **Integration Testing**
   - End-to-end pipeline test
   - Checkpoint resume test
   - Parallel execution test
   - Error recovery test

3. **EventBus Integration**
   - Emit pipeline events (started, stage_completed, failed, completed)
   - Progress updates for dashboard

### Future Enhancements

1. **Resource Management**
   - GPU allocation
   - Memory limits
   - Disk space checks

2. **Advanced Retry**
   - Configurable retry policies
   - Exponential backoff
   - Per-stage retry limits

3. **Pipeline Pause/Resume**
   - Manual pause capability
   - Graceful shutdown on SIGTERM

4. **Web Dashboard**
   - Real-time progress visualization
   - Pipeline history
   - Log viewer

5. **Notification System**
   - Email/Slack notifications
   - Failure alerts
   - Completion reports

---

## Technical Details

### Topological Sort Algorithm

Uses **Kahn's algorithm** for dependency resolution:

1. Calculate in-degree for each stage
2. Start with stages having in-degree 0
3. Process stages in order, decrementing dependencies
4. Detect cycles if not all stages processed

### Parallel Execution Strategy

Stages are grouped into execution levels:

1. Build dependency graph
2. Group stages by maximum dependency depth
3. Execute each group in parallel using ThreadPoolExecutor
4. Wait for group completion before next group

### Checkpoint Format

Checkpoints are JSON files containing:

```json
{
  "pipeline_id": "PIPELINE_ID",
  "pipeline_name": "my_pipeline",
  "completed_stages": ["stage_1", "stage_2"],
  "stage_outputs": {
    "stage_1": {"output_dir": "/path/to/frames"},
    "stage_2": {"output_dir": "/path/to/segmented"}
  },
  "timestamp": 1701619200.0
}
```

---

## Example Workflows

### Simple Character Dataset Pipeline

```yaml
name: character_dataset_pipeline
version: 1.0.0

stages:
  - id: extract
    type: frame_extraction
    depends_on: []
    config:
      input_video: /data/luca.mp4
      output_dir: /data/frames
      mode: scene

  - id: segment
    type: segmentation
    depends_on: [extract]
    config:
      input_dir: "{extract.output_dir}"
      output_dir: /data/segmented
      model: isnet

  - id: cluster
    type: clustering
    depends_on: [segment]
    config:
      input_dir: "{segment.character_dir}"
      output_dir: /data/clustered

  - id: prepare
    type: training_data_prep
    depends_on: [cluster]
    config:
      character_dirs: "{cluster.cluster_dirs}"
      output_dir: /data/training
```

---

## Troubleshooting

### Pipeline Validation Fails

**Error:** "Circular dependency detected"

**Solution:** Check stage dependencies for cycles. Use `validate` command to see dependency graph.

### Stage Execution Fails

**Error:** "No executor registered for stage type"

**Solution:** Ensure executor is registered before running pipeline. Concrete executors need to be implemented.

### Checkpoint Not Loading

**Error:** "Checkpoint not found"

**Solution:** Check `--checkpoint-dir` path. Ensure checkpoint files exist.

---

## Performance Considerations

### Parallelism

- Default max parallel stages: 4
- Adjust based on available CPU/GPU resources
- Monitor system resources during execution

### Checkpointing

- Checkpoints saved after each stage group
- Overhead: ~100ms per checkpoint
- Disable with `--no-checkpoint` for short pipelines

### Memory

- Each parallel stage runs in separate thread
- Ensure sufficient memory for parallel execution
- Consider reducing `--max-parallel` if OOM errors occur

---

## Contributing

When adding new stage types:

1. Implement executor class inheriting from `StageExecutor`
2. Add to executor registry in `__main__.py`
3. Document in this README
4. Add example pipeline configuration
5. Write integration test

---

## License

Part of Animation AI Studio project.

---

**Document Version:** 1.0
**Last Updated:** 2025-12-03
**Status:** Core framework complete, ready for executor implementation
