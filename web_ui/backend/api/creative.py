"""
Creative Studio API: submit Creative Studio workflows as queued Web UI jobs.

These jobs run via `scripts/batch/run_webui_creative_task.sh` and are executed by
JobService/BashRunnerAdapter so they share the same queue, logs, and outputs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..db.models import JobSubmission
from ..services.job_service import JobService
from .jobs import get_job_service


router = APIRouter(prefix="/api/creative", tags=["creative"])


def _safe_name(name: str, fallback: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return fallback
    keep = []
    for ch in raw:
        if ch.isalnum() or ch in ("_", "-", ".", " "):
            keep.append(ch)
    cleaned = "".join(keep).strip().replace(" ", "_")
    return cleaned or fallback


def _make_output_dir(service: JobService, subdir: str) -> Path:
    outputs_root = (service.config.get("paths") or {}).get("outputs_root") or "/mnt/data/ai_data/animation-ai-studio/outputs"
    tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return Path(outputs_root) / str(subdir).strip("/") / tag


class ParodyJobRequest(BaseModel):
    input_video_path: str
    style: str = Field("dramatic", description="dramatic|chaotic|wholesome")
    duration: Optional[float] = None
    effects: Optional[List[str]] = None


class AnalyzeJobRequest(BaseModel):
    input_video_path: str
    visual: bool = True
    audio: bool = False
    context: bool = False
    sample_rate: int = 30


class VoiceJobRequest(BaseModel):
    character: str
    text: str
    emotion: str = "neutral"
    intensity: float = 0.8


class WorkflowJobRequest(BaseModel):
    workflow_type: str = Field(..., description="parody|analyze")
    input_video_path: str
    style: str = "dramatic"
    duration: Optional[float] = None
    audio: bool = False
    context: bool = False


@router.post("/parody", status_code=201)
async def submit_parody_job(req: ParodyJobRequest, service: JobService = Depends(get_job_service)):
    out_dir = _make_output_dir(service, "webui/creative_parody")
    video_name = Path(req.input_video_path).stem
    output_video = out_dir / f"{_safe_name(video_name, 'video')}_parody.mp4"

    submission = JobSubmission(
        film_name=f"creative_parody_{_safe_name(video_name, 'video')}",
        pipeline_type="creative_parody",
        input_video_path=str(req.input_video_path),
        output_base_dir=str(out_dir),
        options={
            "input": str(req.input_video_path),
            "output": str(output_video),
            "style": req.style,
            "duration": req.duration,
            "effects": ",".join(req.effects) if req.effects else None,
        },
    )

    try:
        job_id = await service.submit_job(submission)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {e}") from e

    return {"job_id": job_id, "output_dir": str(out_dir), "output_video": str(output_video)}


@router.post("/analyze", status_code=201)
async def submit_analyze_job(req: AnalyzeJobRequest, service: JobService = Depends(get_job_service)):
    out_dir = _make_output_dir(service, "webui/creative_analyze")
    video_name = Path(req.input_video_path).stem
    output_json = out_dir / f"{_safe_name(video_name, 'video')}_analysis.json"

    submission = JobSubmission(
        film_name=f"creative_analyze_{_safe_name(video_name, 'video')}",
        pipeline_type="creative_analyze",
        input_video_path=str(req.input_video_path),
        output_base_dir=str(out_dir),
        options={
            "input": str(req.input_video_path),
            "output": str(output_json),
            "visual": bool(req.visual),
            "audio": bool(req.audio),
            "context": bool(req.context),
            "sample_rate": int(req.sample_rate),
        },
    )

    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir), "output_json": str(output_json)}


@router.post("/voice", status_code=201)
async def submit_voice_job(req: VoiceJobRequest, service: JobService = Depends(get_job_service)):
    out_dir = _make_output_dir(service, "webui/creative_voice")
    output_wav = out_dir / f"{_safe_name(req.character, 'character')}_voice.wav"

    # input_video_path is required by the DB schema; use output directory as a placeholder.
    submission = JobSubmission(
        film_name=f"creative_voice_{_safe_name(req.character, 'character')}",
        pipeline_type="creative_voice",
        input_video_path=str(out_dir),
        output_base_dir=str(out_dir),
        options={
            "character": req.character,
            "text": req.text,
            "emotion": req.emotion,
            "intensity": float(req.intensity),
            "output": str(output_wav),
        },
    )

    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir), "output_wav": str(output_wav)}


@router.post("/workflow", status_code=201)
async def submit_workflow_job(req: WorkflowJobRequest, service: JobService = Depends(get_job_service)):
    out_dir = _make_output_dir(service, "webui/creative_workflow")
    video_name = Path(req.input_video_path).stem
    output_path = out_dir / f"{_safe_name(video_name, 'video')}_workflow_output"

    submission = JobSubmission(
        film_name=f"creative_workflow_{_safe_name(video_name, 'video')}",
        pipeline_type="creative_workflow",
        input_video_path=str(req.input_video_path),
        output_base_dir=str(out_dir),
        options={
            "workflow_type": req.workflow_type,
            "input": str(req.input_video_path),
            "output": str(output_path),
            "style": req.style,
            "duration": req.duration,
            "audio": bool(req.audio),
            "context": bool(req.context),
        },
    )

    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir), "output_path": str(output_path)}
