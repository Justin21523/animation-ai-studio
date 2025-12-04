# Batch Training Orchestrator - Architecture Design

**Author:** Animation AI Studio
**Date:** 2025-12-03
**Version:** 1.0.0

---

## Overview

Distributed batch training orchestrator for LoRA training jobs with GPU resource management, job scheduling, progress monitoring, and automatic checkpoint evaluation.

**Key Features:**
- Multi-GPU job scheduling and execution
- Resource allocation and conflict resolution
- Progress monitoring and logging
- Automatic checkpoint evaluation
- Job queue management (priority, dependencies)
- Failure recovery and retry logic
- Integration with Orchestration Layer and Safety System

---

## Architecture

```
Job Submission → Job Queue → Scheduler → Resource Allocator → Job Executor
                                              ↓
                                        GPU Monitor
                                              ↓
                                    Progress Tracker → Event Bus
                                              ↓
                                    Checkpoint Evaluator
```

### Components

1. **Job Manager** (`jobs/job_manager.py`)
   - Job definition and validation
   - Job queue management (FIFO, priority, dependency-based)
   - Job state tracking (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
   - Job persistence and recovery

2. **Resource Manager** (`resources/resource_manager.py`)
   - GPU discovery and monitoring (nvidia-smi, torch.cuda)
   - Memory tracking (VRAM, system RAM)
   - Resource allocation and locking
   - Conflict resolution

3. **Scheduler** (`schedulers/job_scheduler.py`)
   - Job scheduling strategies (FIFO, priority, fair-share)
   - Resource-aware scheduling
   - Dependency resolution
   - Load balancing across GPUs

4. **Job Executor** (`jobs/job_executor.py`)
   - subprocess/tmux-based execution
   - Environment setup and cleanup
   - Log capture and streaming
   - Error handling and retry logic

5. **Progress Monitor** (`monitors/progress_monitor.py`)
   - Training metrics parsing (loss, accuracy, epoch)
   - Real-time progress updates
   - Event emission (EventBus integration)
   - Dashboard data aggregation

6. **Checkpoint Evaluator** (`jobs/checkpoint_evaluator.py`)
   - Automatic checkpoint testing
   - Quality metrics computation (CLIP score, FID, consistency)
   - Best checkpoint selection
   - Evaluation result storage

7. **Main Orchestrator** (`batch_training_orchestrator.py`)
   - Coordinates all components
   - API for job submission and management
   - Status queries and reporting
   - Configuration management

---

## Data Flow

```
1. User submits training jobs (batch config)
2. JobManager validates and queues jobs
3. Scheduler selects next job based on strategy + resources
4. ResourceManager allocates GPU + memory
5. JobExecutor launches training process
6. ProgressMonitor tracks metrics and emits events
7. CheckpointEvaluator tests checkpoints on completion
8. JobManager updates job status and frees resources
```

---

## Job Definition

```python
@dataclass
class TrainingJob:
    id: str
    name: str
    config_path: Path
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    gpu_requirements: int = 1  # Number of GPUs
    vram_requirements: int = 16  # GB
    retry_count: int = 3
    timeout: Optional[int] = None  # seconds
    state: JobState = JobState.PENDING
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output_dir: Path
    log_file: Path
```

---

## Resource Management

- **GPU Discovery:** Query available GPUs via `nvidia-smi` or `torch.cuda`
- **Memory Tracking:** Monitor VRAM usage per GPU
- **Allocation:** Reserve GPU(s) for job, mark as busy
- **Conflict Resolution:** Queue jobs if resources unavailable
- **Release:** Free GPU(s) when job completes/fails

---

## Scheduling Strategies

1. **FIFO (First-In-First-Out):** Simple queue, no prioritization
2. **Priority:** Jobs with higher priority run first
3. **Fair-Share:** Distribute resources evenly across users/projects
4. **Shortest Job First:** Estimate job duration, prioritize short jobs
5. **Dependency-Based:** Run jobs in topological order of dependencies

---

## Progress Monitoring

- **Log Parsing:** Extract metrics from training logs (regex patterns)
- **Metrics Tracked:**
  - Current epoch / total epochs
  - Training loss, learning rate
  - Time per epoch, ETA
  - Memory usage
- **Event Emission:** Publish progress updates to EventBus
- **Dashboard:** Aggregate data for web UI

---

## Checkpoint Evaluation

- **Trigger:** Automatically run after training completes
- **Test Suite:**
  - Generate test images with fixed prompts
  - Compute CLIP similarity score
  - Check visual quality (blur, artifacts)
- **Metrics:**
  - CLIP score (text-image alignment)
  - FID (Fréchet Inception Distance)
  - Consistency score (variance across seeds)
- **Selection:** Identify best checkpoint based on combined score

---

## Integration Points

### Orchestration Layer
- Events: `job_submitted`, `job_started`, `job_completed`, `job_failed`
- Workflows: Trigger downstream tasks on job completion

### Safety System
- Memory limits: Respect max VRAM/RAM constraints
- OOM protection: Monitor memory, pause/kill jobs if needed
- Watchdog: Restart stuck jobs

### Agent Framework
- Job optimization: AI-powered hyperparameter suggestions
- Anomaly detection: Identify failed runs early
- Auto-tuning: Adjust learning rate based on loss curves

---

## Configuration

```yaml
# batch_training_orchestrator_config.yaml
scheduler:
  strategy: priority  # fifo, priority, fair_share
  max_concurrent_jobs: 4
  check_interval: 10  # seconds

resources:
  gpus: [0, 1]  # GPU IDs to use
  max_vram_per_gpu: 16  # GB
  max_system_ram: 64  # GB

monitoring:
  log_parse_interval: 5  # seconds
  metrics_retention: 7  # days

evaluation:
  auto_evaluate: true
  test_prompts: configs/test_prompts.json
  num_samples_per_checkpoint: 5
```

---

## CLI Commands

```bash
# Submit single job
python -m scripts.scenarios.batch_training_orchestrator submit \
  --config training_config.toml \
  --name "character_lora_v1" \
  --priority 10

# Submit batch of jobs
python -m scripts.scenarios.batch_training_orchestrator batch \
  --batch-config batch_jobs.yaml

# List jobs
python -m scripts.scenarios.batch_training_orchestrator list \
  --status running

# Cancel job
python -m scripts.scenarios.batch_training_orchestrator cancel \
  --job-id abc123

# Show job status
python -m scripts.scenarios.batch_training_orchestrator status \
  --job-id abc123

# Monitor progress
python -m scripts.scenarios.batch_training_orchestrator monitor \
  --job-id abc123 --follow
```

---

## Success Criteria

- ✅ Submit and queue training jobs
- ✅ Allocate GPU resources dynamically
- ✅ Schedule jobs based on strategy
- ✅ Execute jobs with progress tracking
- ✅ Parse training logs and extract metrics
- ✅ Automatically evaluate checkpoints
- ✅ Handle failures and retries
- ✅ Provide CLI for job management
- ✅ Support multi-GPU jobs
- ✅ Integration with Orchestration Layer

---

**Next:** See `IMPLEMENTATION_SUMMARY.md` for implementation roadmap.
