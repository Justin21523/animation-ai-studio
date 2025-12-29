#!/usr/bin/env python3
"""
SDXL ControlNet Trainer (Diffusers + Accelerate)

Goals:
- Train ControlNet weights for SDXL using an on-disk dataset produced by
  `scripts/processing/training/controlnet_dataset_builder.py`.
- Prefer local model paths (offline/reproducible); no dataset downloads.
- Save output as a diffusers ControlNet directory (controlnet.save_pretrained()).

Notes:
- This is a minimal trainer intended to be extended as needed (validation, schedulers, checkpoints).
- For multi-GPU, run via `accelerate launch ...`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


logger = logging.getLogger(__name__)


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict)")
    return data


def _is_local_path(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def _load_sdxl_pipeline(
    *,
    base_model_path: str,
    base_repo_path: Optional[str],
    dtype,
):
    """
    Load SDXL pipeline in a way that works with single-file checkpoints.
    """
    from diffusers import StableDiffusionXLPipeline

    base_model = Path(base_model_path)
    if base_model.is_file():
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

        if not base_repo_path or not Path(base_repo_path).exists():
            raise FileNotFoundError(
                "SDXL single-file checkpoints require a local `base_repo_path` (tokenizers/text encoders/scheduler) "
                f"but got base_repo_path={base_repo_path!r}"
            )
        base_repo = str(base_repo_path)
        logger.info("Loading SDXL tokenizers/text encoders from base repo: %s", base_repo)
        tokenizer = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(base_repo, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_repo, subfolder="text_encoder_2", torch_dtype=dtype
        )

        return StableDiffusionXLPipeline.from_single_file(
            str(base_model),
            torch_dtype=dtype,
            use_safetensors=True,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
        )

    return StableDiffusionXLPipeline.from_pretrained(
        str(base_model),
        torch_dtype=dtype,
        use_safetensors=True,
    )


def _load_noise_scheduler(base_model_path: str, base_repo_path: Optional[str]):
    from diffusers import DDPMScheduler

    candidate_roots: List[str] = []
    if base_repo_path:
        candidate_roots.append(str(base_repo_path))
    if base_model_path and Path(base_model_path).exists() and Path(base_model_path).is_dir():
        candidate_roots.append(str(base_model_path))

    for root in candidate_roots:
        try:
            return DDPMScheduler.from_pretrained(root, subfolder="scheduler")
        except Exception:
            continue

    raise RuntimeError(
        "Failed to load DDPMScheduler. Provide a local `base_repo_path` that contains `scheduler/`."
    )


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path
    base_model_path: str
    base_repo_path: Optional[str]
    output_dir: Path
    output_name: str

    resolution: int = 1024
    init_controlnet_path: Optional[str] = None

    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    max_train_steps: int = 1000
    save_steps: int = 500
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "no" | "fp16" | "bf16"
    seed: int = 42
    num_workers: int = 0
    enable_gradient_checkpointing: bool = True


class ControlNetDiskDataset:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        meta_path = self.dataset_dir / "dataset_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        items = meta.get("items") or []
        if not isinstance(items, list) or not items:
            raise ValueError(f"Dataset metadata has no items: {meta_path}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        item = self.items[idx]
        img_path = self.dataset_dir / item["image"]
        cond_path = self.dataset_dir / item["conditioning_image"]
        caption = (item.get("caption") or "").strip()

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_np = np.array(img).astype("float32") / 255.0
        pixel_values = torch.from_numpy(img_np).permute(2, 0, 1)
        pixel_values = pixel_values * 2.0 - 1.0  # [-1, 1] for VAE

        with Image.open(cond_path) as cimg:
            cimg = cimg.convert("RGB")
            c_np = np.array(cimg).astype("float32") / 255.0
        conditioning_pixel_values = torch.from_numpy(c_np).permute(2, 0, 1)  # [0, 1] for ControlNet

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "caption": caption,
        }


def _collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch

    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    conditioning_pixel_values = torch.stack([ex["conditioning_pixel_values"] for ex in examples])
    captions = [ex.get("caption", "") for ex in examples]
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "captions": captions,
    }


def train(config: TrainConfig) -> Path:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import ControlNetModel

    mp = None if config.mixed_precision == "no" else config.mixed_precision
    accelerator = Accelerator(mixed_precision=mp, gradient_accumulation_steps=int(config.gradient_accumulation_steps))
    set_seed(int(config.seed))

    dtype = torch.float16 if config.mixed_precision == "fp16" else (torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32)

    pipeline = _load_sdxl_pipeline(
        base_model_path=config.base_model_path,
        base_repo_path=config.base_repo_path,
        dtype=dtype,
    )
    noise_scheduler = _load_noise_scheduler(config.base_model_path, config.base_repo_path)

    vae = pipeline.vae
    unet = pipeline.unet

    if config.init_controlnet_path:
        logger.info("Initializing ControlNet from: %s", config.init_controlnet_path)
        controlnet = ControlNetModel.from_pretrained(config.init_controlnet_path, torch_dtype=dtype)
    else:
        logger.info("Initializing ControlNet from SDXL UNet weights (from_unet)")
        controlnet = ControlNetModel.from_unet(unet)

    if config.enable_gradient_checkpointing and hasattr(controlnet, "enable_gradient_checkpointing"):
        controlnet.enable_gradient_checkpointing()

    # Freeze everything except ControlNet.
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    if getattr(pipeline, "text_encoder", None) is not None:
        pipeline.text_encoder.requires_grad_(False)
    if getattr(pipeline, "text_encoder_2", None) is not None:
        pipeline.text_encoder_2.requires_grad_(False)

    controlnet.train()
    unet.eval()
    vae.eval()
    if getattr(pipeline, "text_encoder", None) is not None:
        pipeline.text_encoder.eval()
    if getattr(pipeline, "text_encoder_2", None) is not None:
        pipeline.text_encoder_2.eval()

    dataset = ControlNetDiskDataset(config.dataset_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.train_batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate_fn,
    )

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=float(config.learning_rate))

    controlnet, optimizer, dataloader = accelerator.prepare(controlnet, optimizer, dataloader)

    # Move SDXL components to device/dtype (not wrapped by accelerator).
    pipeline.to(accelerator.device, dtype=dtype)
    # DDPMScheduler stores tensors (betas/alphas/alphas_cumprod) on CPU by default.
    for attr in ("betas", "alphas", "alphas_cumprod"):
        if hasattr(noise_scheduler, attr):
            val = getattr(noise_scheduler, attr)
            if isinstance(val, torch.Tensor):
                setattr(noise_scheduler, attr, val.to(accelerator.device))

    projection_dim = getattr(getattr(pipeline, "text_encoder_2", None), "config", None)
    projection_dim = getattr(projection_dim, "projection_dim", None)

    global_step = 0  # optimizer steps (after grad accumulation)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    save_root = output_root / config.output_name
    save_root.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training: max_train_steps=%d", config.max_train_steps)

    while global_step < int(config.max_train_steps):
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=dtype)
                control_values = batch["conditioning_pixel_values"].to(device=accelerator.device, dtype=dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.int64,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                    batch["captions"],
                    device=accelerator.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                add_time_ids = pipeline._get_add_time_ids(
                    (int(config.resolution), int(config.resolution)),
                    (0, 0),
                    (int(config.resolution), int(config.resolution)),
                    prompt_embeds.dtype,
                    text_encoder_projection_dim=projection_dim,
                ).to(device=accelerator.device)
                add_time_ids = add_time_ids.repeat(prompt_embeds.shape[0], 1)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

                down_samples, mid_sample = controlnet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_values,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction_type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), float(config.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % 10 == 0:
                        logger.info("step=%d loss=%.6f", global_step, loss.detach().item())

                    if config.save_steps and global_step > 0 and global_step % int(config.save_steps) == 0:
                        ckpt_dir = save_root / f"checkpoint-{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(controlnet)
                        unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
                        (ckpt_dir / "training_args.json").write_text(
                            json.dumps(
                                {
                                    "base_model_path": config.base_model_path,
                                    "base_repo_path": config.base_repo_path,
                                    "dataset_dir": str(config.dataset_dir),
                                    "resolution": int(config.resolution),
                                    "learning_rate": float(config.learning_rate),
                                    "train_batch_size": int(config.train_batch_size),
                                    "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
                                    "mixed_precision": config.mixed_precision,
                                    "seed": int(config.seed),
                                    "global_step": int(global_step),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        logger.info("Saved checkpoint to %s", ckpt_dir)

                if global_step >= int(config.max_train_steps):
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = save_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(controlnet)
        unwrapped.save_pretrained(final_dir, safe_serialization=True)
        (final_dir / "training_args.json").write_text(
            json.dumps(
                {
                    "base_model_path": config.base_model_path,
                    "base_repo_path": config.base_repo_path,
                    "dataset_dir": str(config.dataset_dir),
                    "resolution": int(config.resolution),
                    "learning_rate": float(config.learning_rate),
                    "train_batch_size": int(config.train_batch_size),
                    "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
                    "mixed_precision": config.mixed_precision,
                    "seed": int(config.seed),
                    "global_step": int(global_step),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved final model to %s", final_dir)
        return final_dir

    return save_root / "final"


def _build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ControlNet for SDXL (diffusers + accelerate).")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file")

    parser.add_argument("--dataset_dir", type=Path, required=defaults.get("dataset_dir") is None)
    parser.add_argument("--base_model_path", type=str, required=defaults.get("base_model_path") is None)
    parser.add_argument("--base_repo_path", type=str, default=defaults.get("base_repo_path"))
    parser.add_argument("--init_controlnet_path", type=str, default=defaults.get("init_controlnet_path"))

    parser.add_argument("--output_dir", type=Path, required=defaults.get("output_dir") is None)
    parser.add_argument("--output_name", type=str, required=defaults.get("output_name") is None)

    parser.add_argument("--resolution", type=int, default=int(defaults.get("resolution", 1024)))
    parser.add_argument("--train_batch_size", type=int, default=int(defaults.get("train_batch_size", 1)))
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=int(defaults.get("gradient_accumulation_steps", 1)),
    )
    parser.add_argument("--learning_rate", type=float, default=float(defaults.get("learning_rate", 1e-5)))
    parser.add_argument("--max_train_steps", type=int, default=int(defaults.get("max_train_steps", 1000)))
    parser.add_argument("--save_steps", type=int, default=int(defaults.get("save_steps", 500)))
    parser.add_argument("--max_grad_norm", type=float, default=float(defaults.get("max_grad_norm", 1.0)))
    parser.add_argument("--mixed_precision", type=str, default=str(defaults.get("mixed_precision", "fp16")))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 0)))
    parser.add_argument(
        "--enable_gradient_checkpointing",
        action="store_true",
        default=bool(defaults.get("enable_gradient_checkpointing", True)),
    )
    parser.add_argument("--log_level", type=str, default=str(defaults.get("log_level", "INFO")))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = _load_yaml(known.config)

    parser = _build_parser(defaults)
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    cfg = TrainConfig(
        dataset_dir=Path(args.dataset_dir or defaults.get("dataset_dir")),
        base_model_path=str(args.base_model_path or defaults.get("base_model_path")),
        base_repo_path=str(args.base_repo_path or defaults.get("base_repo_path")) if (args.base_repo_path or defaults.get("base_repo_path")) else None,
        output_dir=Path(args.output_dir or defaults.get("output_dir")),
        output_name=str(args.output_name or defaults.get("output_name")),
        resolution=int(args.resolution),
        init_controlnet_path=str(args.init_controlnet_path) if args.init_controlnet_path else None,
        train_batch_size=int(args.train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        max_train_steps=int(args.max_train_steps),
        save_steps=int(args.save_steps),
        max_grad_norm=float(args.max_grad_norm),
        mixed_precision=str(args.mixed_precision),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        enable_gradient_checkpointing=bool(args.enable_gradient_checkpointing),
    )

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
