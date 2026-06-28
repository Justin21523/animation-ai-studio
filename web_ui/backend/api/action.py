"""
Action API: Submit Action ControlNet jobs via the shared job queue.

This wraps the CLI entrypoints under `scripts/generation/action/` as web jobs:
- generate.py (single image)
- animate.py (frame sequence)
- inpaint.py (mask editing)
- multichar.py (layered composition)
- regress.py (batch regression)
- extract_controls.py (video/frames -> control frames)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..db.models import JobSubmission
from ..services.job_service import JobService
from .jobs import get_job_service


router = APIRouter(prefix="/api/action", tags=["action"])


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
    cfg_paths = service.config.get("paths") or {}
    outputs_root = Path(cfg_paths.get("outputs_root") or "/mnt/data/ai_data/animation-ai-studio/outputs")
    if not outputs_root.is_absolute():
        project_root = Path(cfg_paths.get("project_root") or Path(__file__).resolve().parents[3])
        if not project_root.is_absolute():
            project_root = Path(__file__).resolve().parents[3] / project_root
        outputs_root = project_root / outputs_root
    tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return outputs_root / str(subdir).strip("/") / tag


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = await upload.read()
    finally:
        await upload.close()
    dest.write_bytes(data)


class AnimateRequest(BaseModel):
    character: str
    control_type: str = Field("pose", description="pose|canny|softedge|lineart|tile|auto")
    control_dir: str
    pattern: str = "*.png"
    every: int = 1
    limit: int = 0

    action: Optional[str] = None
    scene: str = ""
    prompt: Optional[str] = None
    extra: str = ""
    style: str = "pixar_3d"
    negative_prompt: Optional[str] = None
    negative_prompt_key: str = "character"

    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    controlnet_scale: Optional[float] = None
    seed: Optional[int] = None
    seed_mode: str = "fixed"
    no_preprocess: bool = False

    # Multi-pass refinement
    refine_sequence: Optional[List[str]] = None
    refine_strength: Optional[float] = None
    refine_steps: Optional[int] = None
    refine_guidance_scale: Optional[float] = None
    refine_controlnet_scale: Optional[float] = None
    controls_root: Optional[str] = None
    save_intermediate: bool = False

    write_video: bool = False
    fps: int = 12
    ffmpeg: Optional[str] = None

    reference_image: Optional[str] = None
    consistency_threshold: Optional[float] = None
    consistency_device: Optional[str] = None
    max_retries: Optional[int] = None

    timeout: Optional[int] = None


class ExtractControlsRequest(BaseModel):
    mode: str = Field("video", description="video|frames")
    video: Optional[str] = None
    frames_dir: Optional[str] = None
    pattern: str = "*.png"
    fps: Optional[int] = None
    types: Optional[List[str]] = None
    detect_resolution: Optional[int] = None
    image_resolution: Optional[int] = None
    overwrite: bool = False
    timeout: Optional[int] = None


@router.get("/metadata")
async def get_action_metadata(service: JobService = Depends(get_job_service)):
    cfg_paths = service.config.get("paths") or {}
    project_root = Path(cfg_paths.get("project_root") or Path(__file__).resolve().parents[3])
    if not project_root.is_absolute():
        project_root = Path(__file__).resolve().parents[3] / project_root
    project_root = project_root.resolve()
    registry_path = project_root / "configs" / "generation" / "controlnet_config.yaml"
    action_cfg_path = project_root / "configs" / "generation" / "action_config.yaml"
    sdxl_cfg_path = project_root / "configs" / "generation" / "sdxl_config.yaml"

    from scripts.generation.action.action_registry import (  # noqa: WPS433
        ACTION_CONTROL_TYPES,
        ActionControlNetRegistry,
    )

    reg = ActionControlNetRegistry(str(registry_path))
    action_cfg = yaml.safe_load(action_cfg_path.read_text(encoding="utf-8")) if action_cfg_path.exists() else {}
    sdxl_cfg = yaml.safe_load(sdxl_cfg_path.read_text(encoding="utf-8")) if sdxl_cfg_path.exists() else {}

    actions = (action_cfg.get("actions") or {}) if isinstance(action_cfg, dict) else {}
    styles = list((sdxl_cfg.get("style_prompts") or {}).keys()) if isinstance(sdxl_cfg, dict) else []
    neg_keys = list((sdxl_cfg.get("negative_prompts") or {}).keys()) if isinstance(sdxl_cfg, dict) else []

    return {
        "characters": reg.list_action_characters(),
        "actions": actions,
        "control_types": ["auto"] + list(ACTION_CONTROL_TYPES),
        "styles": styles,
        "negative_prompt_keys": neg_keys,
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "guidance_scale": 7.5,
            "style": "pixar_3d",
            "negative_prompt_key": "character",
        },
    }


@router.post("/generate", status_code=201)
async def submit_generate_job(
    character: str = Form(...),
    control_type: str = Form("auto"),
    action: str = Form(""),
    scene: str = Form(""),
    prompt: str = Form(""),
    extra: str = Form(""),
    style: str = Form("pixar_3d"),
    negative_prompt: str = Form(""),
    negative_prompt_key: str = Form("character"),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    controlnet_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1),
    no_preprocess: bool = Form(False),
    consistency_threshold: float = Form(0.65),
    max_retries: int = Form(0),
    consistency_device: str = Form("cpu"),
    timeout: Optional[int] = Form(None),
    control_image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    service: JobService = Depends(get_job_service),
):
    out_dir = _make_output_dir(service, "webui/action_generate")
    inputs_dir = out_dir / "inputs"
    control_name = _safe_name(control_image.filename or "", "control.png")
    control_path = inputs_dir / f"control_{control_name}"
    await _save_upload(control_image, control_path)

    ref_path = None
    if reference_image is not None:
        ref_name = _safe_name(reference_image.filename or "", "reference.png")
        ref_path = inputs_dir / f"reference_{ref_name}"
        await _save_upload(reference_image, ref_path)

    film_name = f"action_generate_{_safe_name(character, 'character')}"
    submission = JobSubmission(
        film_name=film_name,
        pipeline_type="action_generate",
        input_video_path=str(control_path),
        output_base_dir=str(out_dir),
        options={
            "character": character,
            "control_type": control_type,
            "action": action.strip() or None,
            "scene": scene,
            "prompt": prompt.strip() or None,
            "extra": extra,
            "style": style,
            "negative_prompt": negative_prompt.strip() or None,
            "negative_prompt_key": negative_prompt_key,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "controlnet_scale": controlnet_scale,
            "seed": seed,
            "num_images": num_images,
            "no_preprocess": bool(no_preprocess),
            "reference_image": str(ref_path) if ref_path else None,
            "consistency_threshold": float(consistency_threshold),
            "max_retries": int(max_retries),
            "consistency_device": consistency_device,
            "timeout": int(timeout) if timeout else None,
        },
    )

    try:
        job_id = await service.submit_job(submission)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {e}") from e

    return {"job_id": job_id, "output_dir": str(out_dir)}


@router.post("/animate", status_code=201)
async def submit_animate_job(req: AnimateRequest, service: JobService = Depends(get_job_service)):
    out_dir = _make_output_dir(service, "webui/action_animate")

    options = req.model_dump(exclude_none=True)
    options["control_dir"] = str(req.control_dir)

    submission = JobSubmission(
        film_name=f"action_animate_{_safe_name(req.character, 'character')}",
        pipeline_type="action_animate",
        input_video_path=str(req.control_dir),
        output_base_dir=str(out_dir),
        options=options,
    )
    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir)}


@router.post("/extract_controls", status_code=201)
async def submit_extract_controls_job(req: ExtractControlsRequest, service: JobService = Depends(get_job_service)):
    out_dir = _make_output_dir(service, "webui/action_controls")
    options = req.model_dump(exclude_none=True)

    input_path = None
    if str(req.mode).lower() == "frames":
        input_path = req.frames_dir
    else:
        input_path = req.video

    if not input_path:
        raise HTTPException(status_code=400, detail="Missing input path for extract_controls")

    submission = JobSubmission(
        film_name="action_extract_controls",
        pipeline_type="action_extract_controls",
        input_video_path=str(input_path),
        output_base_dir=str(out_dir),
        options=options,
    )
    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir)}


@router.post("/inpaint", status_code=201)
async def submit_inpaint_job(
    character: str = Form(...),
    control_type: str = Form("auto"),
    action: str = Form(""),
    scene: str = Form(""),
    prompt: str = Form(""),
    extra: str = Form(""),
    style: str = Form("pixar_3d"),
    negative_prompt: str = Form(""),
    negative_prompt_key: str = Form("character"),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    strength: float = Form(0.55),
    controlnet_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    no_preprocess: bool = Form(False),
    timeout: Optional[int] = Form(None),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    control_image: Optional[UploadFile] = File(None),
    service: JobService = Depends(get_job_service),
):
    out_dir = _make_output_dir(service, "webui/action_inpaint")
    inputs_dir = out_dir / "inputs"

    image_path = inputs_dir / f"image_{_safe_name(image.filename or '', 'image.png')}"
    mask_path = inputs_dir / f"mask_{_safe_name(mask.filename or '', 'mask.png')}"
    await _save_upload(image, image_path)
    await _save_upload(mask, mask_path)

    control_path = None
    if control_image is not None:
        control_path = inputs_dir / f"control_{_safe_name(control_image.filename or '', 'control.png')}"
        await _save_upload(control_image, control_path)

    submission = JobSubmission(
        film_name=f"action_inpaint_{_safe_name(character, 'character')}",
        pipeline_type="action_inpaint",
        input_video_path=str(image_path),
        output_base_dir=str(out_dir),
        options={
            "character": character,
            "control_type": control_type,
            "image": str(image_path),
            "mask": str(mask_path),
            "control_image": str(control_path) if control_path else None,
            "action": action.strip() or None,
            "scene": scene,
            "prompt": prompt.strip() or None,
            "extra": extra,
            "style": style,
            "negative_prompt": negative_prompt.strip() or None,
            "negative_prompt_key": negative_prompt_key,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "controlnet_scale": controlnet_scale,
            "seed": seed,
            "no_preprocess": bool(no_preprocess),
            "timeout": int(timeout) if timeout else None,
        },
    )

    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir)}


@router.post("/multichar", status_code=201)
async def submit_multichar_job(
    config: UploadFile = File(...),
    timeout: Optional[int] = Form(None),
    service: JobService = Depends(get_job_service),
):
    out_dir = _make_output_dir(service, "webui/action_multichar")
    inputs_dir = out_dir / "inputs"
    cfg_path = inputs_dir / f"multichar_{_safe_name(config.filename or '', 'config.yaml')}"
    await _save_upload(config, cfg_path)

    submission = JobSubmission(
        film_name="action_multichar",
        pipeline_type="action_multichar",
        input_video_path=str(cfg_path),
        output_base_dir=str(out_dir),
        options={"config": str(cfg_path), "timeout": int(timeout) if timeout else None},
    )
    job_id = await service.submit_job(submission)
    return {"job_id": job_id, "output_dir": str(out_dir)}
