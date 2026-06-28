# Animation AI Studio - Web UI

Web-based interface for Animation AI Studio batch processing pipelines.

## Architecture

```
Frontend (Vanilla JS)
    ↓ HTTP REST API + SSE
Backend (FastAPI)
    ↓ BashRunnerAdapter
Existing Bash Scripts (scripts/batch/)
```

## Features

- **Job Submission**: Submit CPU-only or CPU+GPU processing jobs
- **Real-time Monitoring**: Live progress updates via Server-Sent Events (SSE)
- **Job History**: View all submitted jobs and their status
- **Result Management**: Browse and manage processing outputs

## Quick Start

### 1. Install Dependencies

```bash
# Activate conda environment
conda activate ai_env

# Install Web UI dependencies
pip install -r requirements/web_ui.txt
```

### 2. Start Server

```bash
# From project root
cd web_ui
./start_server.sh

# Or manually
cd web_ui/backend
python main.py
```

The server will start at: http://localhost:8100

### 3. Access Web UI

Open your browser and navigate to:
- **Web UI**: http://localhost:8100/static/index.html
- **API Docs**: http://localhost:8100/docs
- **Health Check**: http://localhost:8100/api/health

## Testing with CPU-Only Pipeline

### Prepare Test Data

1. **Small test video** (recommended for initial testing):
   ```bash
   # Use a short video clip (10-30 seconds)
   INPUT_VIDEO="/mnt/data/videos/test_clip.mp4"
   OUTPUT_DIR="/mnt/data/extracted/test_output"
   ```

2. **Full film video** (after CPU-only testing succeeds):
   ```bash
   INPUT_VIDEO="/mnt/data/videos/luca_full.mp4"
   OUTPUT_DIR="/mnt/data/extracted/luca"
   ```

### Submit CPU-Only Job via Web UI

1. Open http://localhost:8100/static/index.html
2. Fill in the form:
   - **Film Name**: `test_film`
   - **Pipeline Type**: `CPU Only`
   - **Input Video Path**: `/mnt/data/videos/test_clip.mp4`
   - **Output Directory**: `/mnt/data/extracted/test_output`
   - **Workers**: `8`
3. Click "Submit Job"
4. Watch real-time progress in the UI

### Expected CPU-Only Pipeline Stages

```
1. Frame Extraction (CPU)
   - Extract frames at 2 FPS
   - Detect scenes
   - Output: {output_dir}/frames/

2. Frame Analysis (CPU)
   - Analyze shot composition
   - Detect camera movement
   - Output: {output_dir}/analysis/

3. RAG Preparation (CPU)
   - Create vector database
   - Index frame embeddings
   - Output: {output_dir}/rag/
```

### Verify Results

After job completion, check:

```bash
# Frame extraction
ls -lh /mnt/data/extracted/test_output/frames/

# Analysis results
cat /mnt/data/extracted/test_output/analysis/summary.json

# RAG database
ls -lh /mnt/data/extracted/test_output/rag/
```

## API Endpoints

### Jobs

- `POST /api/jobs` - Submit new job
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/jobs` - List all jobs
- `DELETE /api/jobs/{job_id}` - Cancel job
- `GET /api/jobs/{job_id}/progress` - SSE progress stream
- `GET /api/jobs/{job_id}/outputs` - Get job outputs

### Config

- `GET /api/config` - Get public configuration

### Health

- `GET /api/health` - Health check

## Configuration

Configuration file: `configs/web_ui/backend.yaml`

Key settings:

```yaml
server:
  host: "127.0.0.1"
  port: 8000

database:
  path: "/mnt/data/training/runs/animation-ai-studio/workflows.db"

conda:
  env_name: "ai_env"

paths:
  project_root: "/mnt/c/ai_projects/animation-ai-studio"
  scripts_root: "/mnt/c/ai_projects/animation-ai-studio/scripts"
  datasets_root: "/mnt/data/datasets"
  training_root: "/mnt/data/training"

cpu_pipeline:
  default_workers: 8
  max_workers: 16

resources:
  max_concurrent_jobs: 1  # Single GPU, one job at a time
```

## Database Schema

Web UI extends existing `workflows.db` with 3 tables:

### jobs
- Job metadata (film_name, pipeline_type, status, progress)
- Timestamps (created_at, started_at, completed_at)
- Error tracking (error_message)

### job_outputs
- Output tracking per stage
- File paths, counts, sizes

### system_metrics
- Historical resource monitoring
- CPU, RAM, GPU, VRAM metrics

## Troubleshooting

### Server Won't Start

```bash
# Check conda environment
conda env list

# Check Python
/home/justin/miniconda3/envs/ai_env/bin/python --version

# Check dependencies
pip list | grep fastapi
```

### Database Issues

```bash
# Check database directory
ls -la /mnt/data/training/runs/animation-ai-studio/

# Reset database (WARNING: deletes all job history)
rm /mnt/data/training/runs/animation-ai-studio/workflows.db
```

### Job Fails Immediately

Check:
1. Input video path exists
2. Output directory parent exists
3. Batch scripts are executable: `chmod +x scripts/batch/*.sh`
4. Conda environment activated in bash scripts

### SSE Connection Errors

- Ensure CORS is configured correctly
- Check browser console for errors
- Verify job_id is correct

## Development

### Backend Structure

```
web_ui/backend/
├── main.py              # FastAPI app
├── api/
│   └── jobs.py          # Jobs API endpoints
├── services/
│   └── job_service.py   # Job execution logic
├── adapters/
│   └── bash_runner.py   # Bash script adapter
└── db/
    ├── schema.py        # Database schema
    ├── models.py        # Pydantic models
    └── operations.py    # CRUD operations
```

### Frontend Structure

```
web_ui/frontend/
└── index.html           # Single-page test UI
```

### Adding New Pipeline Types

1. Add to `configs/web_ui/backend.yaml`:
   ```yaml
   pipeline_types:
     - custom_pipeline
   ```

2. Add handler in `job_service.py`:
   ```python
   def _build_script_config(self, submission):
       if submission.pipeline_type == 'custom_pipeline':
           script_path = scripts_root / "batch" / "custom_script.sh"
           # ...
   ```

## Next Steps

After CPU-only testing succeeds:

1. **Test GPU Pipeline**: Submit `cpu_gpu_full` jobs
2. **Implement Full Frontend**: Build complete UI with all features
3. **Add Result Browsing**: File browser for outputs
4. **Add Config Editor**: YAML/JSON editor with validation
5. **Add System Monitoring**: Real-time GPU/CPU metrics dashboard
6. **Add Statistics**: Job history analytics and visualizations

## License

Part of Animation AI Studio project.

---

**Version**: 0.1.0 (Week 1 Backend Complete)
**Status**: Ready for CPU-only testing
**Last Updated**: 2025-12-04
