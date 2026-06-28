"""
General image generation API for multi-provider backends.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scripts.generation.image.registry import ImageProviderRegistry
from scripts.generation.image.strategy import ImageWorkflowStrategy
from ..db.models import JobSubmission
from ..services.job_service import JobService
from .jobs import get_job_service


router = APIRouter(prefix="/api/image", tags=["image"])


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


class ImageGenerateRequest(BaseModel):
    provider: Optional[str] = None
    workflow: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    style: Optional[str] = None
    width: int = Field(1024, ge=256)
    height: int = Field(1024, ge=256)
    steps: Optional[int] = Field(None, ge=1)
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    num_images: int = Field(1, ge=1, le=4)
    timeout: Optional[int] = None


@router.get("/metadata")
async def get_image_metadata():
    registry = ImageProviderRegistry()
    strategy = ImageWorkflowStrategy(registry=registry)
    providers = []
    for item in registry.get_capabilities():
        providers.append(
            {
                "provider": item.provider,
                "label": item.label,
                "mode": item.mode,
                "description": item.description,
                "enabled": item.enabled,
                "available": item.available,
                "status": item.status,
                "availability_reason": item.availability_reason,
                "supports_negative_prompt": item.supports_negative_prompt,
                "supports_styles": item.supports_styles,
                "supports_seed": item.supports_seed,
                "supports_num_images": item.supports_num_images,
                "supports_guidance_scale": item.supports_guidance_scale,
                "supports_steps": item.supports_steps,
                "styles": item.styles,
                "default_width": item.default_width,
                "default_height": item.default_height,
                "max_images_per_request": item.max_images_per_request,
                "recommended_for": item.recommended_for,
                "action_compatible": item.action_compatible,
                "ui_flags": item.ui_flags,
                "usage_notes": item.usage_notes,
            }
        )
    return {
        "providers": providers,
        "workflow_strategy": strategy.summary(),
        "workflow_list": strategy.list_workflows(),
        "workflow_defaults": strategy.default_provider_map(),
        "workflow_task_defaults": strategy.task_defaults_map(),
        "gui_workflows": strategy.gui_workflow_statuses(),
    }


@router.get("/resolve-workflow")
async def resolve_image_workflow(workflow: str):
    workflow_name = (workflow or "").strip()
    if not workflow_name:
        raise HTTPException(status_code=400, detail="workflow is required")
    strategy = ImageWorkflowStrategy(registry=ImageProviderRegistry())
    return strategy.resolve_task_config(workflow_name)


@router.post("/generate", status_code=201)
async def submit_image_generate_job(req: ImageGenerateRequest, service: JobService = Depends(get_job_service)):
    strategy = ImageWorkflowStrategy(registry=ImageProviderRegistry())
    resolved = None
    selected_provider = (req.provider or "").strip() or None

    if req.workflow:
        workflow_name = req.workflow.strip()
        if workflow_name:
            resolved = strategy.resolve_task_config(workflow_name)
            resolved_provider = str(resolved.get("provider") or "").strip() or None
            resolved_mode = str(resolved.get("mode") or "").strip()
            if resolved_mode == "fixed" and selected_provider and resolved_provider and selected_provider != resolved_provider:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Workflow '{workflow_name}' is fixed to provider '{resolved_provider}', "
                        f"but got '{selected_provider}'."
                    ),
                )
            if not selected_provider and resolved_provider:
                selected_provider = resolved_provider

    if not selected_provider:
        raise HTTPException(status_code=400, detail="provider is required (or provide workflow with resolvable provider)")

    try:
        registry = ImageProviderRegistry()
        if not registry.is_enabled(selected_provider):
            raise ValueError(f"Provider '{selected_provider}' is disabled in configs/generation/image_providers.yaml")
        provider = registry.get_provider(selected_provider)
        caps = provider.get_capabilities()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.num_images > caps.max_images_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{selected_provider}' allows at most {caps.max_images_per_request} images per request",
        )

    out_dir = _make_output_dir(service, "webui/image_generate")
    submission = JobSubmission(
        film_name=f"image_generate_{_safe_name(selected_provider, 'provider')}",
        pipeline_type="image_generate",
        input_video_path=str(out_dir),
        output_base_dir=str(out_dir),
        options={
            "provider": selected_provider,
            "workflow": req.workflow,
            "resolved_strategy": resolved,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "style": req.style,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
            "num_images": req.num_images,
            "timeout": req.timeout,
        },
    )

    try:
        job_id = await service.submit_job(submission)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {exc}") from exc

    return {"job_id": job_id, "output_dir": str(out_dir)}
