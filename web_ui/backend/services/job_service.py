"""
JobService: Business logic for job submission, execution, and monitoring.
Integrates BashRunnerAdapter with database operations.
"""
import asyncio
import uuid
import time
import logging
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..db.models import JobSubmission, JobInfo
from ..db.operations import JobDatabase
from ..adapters.bash_runner import BashRunnerAdapter, BashScriptConfig, ScriptResult

logger = logging.getLogger(__name__)


class JobService:
    """Service for job management."""

    def __init__(
        self,
        db: JobDatabase,
        config: Dict[str, Any]
    ):
        """
        Initialize job service.

        Args:
            db: Job database instance
            config: Backend configuration dict
        """
        self.db = db
        self.config = config
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.running_adapters: Dict[str, BashRunnerAdapter] = {}
        self.progress_callbacks: Dict[str, List] = {}

    def _project_root(self) -> Path:
        configured = Path(self.config.get("paths", {}).get("project_root") or ".")
        if configured.is_absolute():
            return configured.resolve()
        return (Path(__file__).resolve().parents[3] / configured).resolve()

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self._project_root() / path

    def _demo_mode(self) -> bool:
        return bool(self.config.get("demo_mode") or self.config.get("demo", {}).get("enabled"))

    async def submit_job(self, submission: JobSubmission) -> str:
        """
        Submit a new processing job.

        Args:
            submission: Job submission details

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())

        logger.info(f"Submitting job {job_id}: {submission.film_name} ({submission.pipeline_type})")

        # Validate input paths (some job types don't require a pre-existing input file)
        requires_input = not (
            (submission.pipeline_type or "").startswith("creative_voice")
            or (submission.pipeline_type or "") == "image_generate"
        )
        input_path = self._resolve_path(submission.input_video_path)
        if requires_input and not self._demo_mode() and not input_path.exists():
            raise ValueError(f"Input path not found: {submission.input_video_path}")

        # Create output directory
        output_dir = self._resolve_path(submission.output_base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission.output_base_dir = str(output_dir)
        if not Path(submission.input_video_path).is_absolute():
            submission.input_video_path = str(input_path)

        # Save job to database
        await self.db.create_job(
            job_id=job_id,
            film_name=submission.film_name,
            pipeline_type=submission.pipeline_type,
            input_video_path=submission.input_video_path,
            output_base_dir=submission.output_base_dir,
            options=submission.options or {}
        )

        # Start job execution asynchronously
        task = asyncio.create_task(self._execute_job(job_id, submission))
        self.running_jobs[job_id] = task

        return job_id

    async def _execute_job(self, job_id: str, submission: JobSubmission):
        """
        Execute job asynchronously.

        Args:
            job_id: Job identifier
            submission: Job submission details
        """
        try:
            # Update status to running
            await self.db.update_job(job_id, {
                'status': 'running',
                'started_at': time.time()
            })

            # Build script config based on pipeline type
            if self._demo_mode():
                await self._execute_demo_job(job_id, submission)
                return

            # Build script config based on pipeline type
            script_config = self._build_script_config(submission)
            script_config.log_dir = str(Path(submission.output_base_dir) / "webui_logs" / job_id)

            # Create progress callback
            async def progress_callback(event: Dict[str, Any]):
                await self._handle_progress_event(job_id, event)

            # Execute bash script
            adapter = BashRunnerAdapter(
                job_id=job_id,
                progress_callback=progress_callback,
                conda_env=self.config.get('conda', {}).get('env_name', 'ai_env')
            )
            self.running_adapters[job_id] = adapter

            result: ScriptResult = await adapter.execute(script_config)

            # Process result
            if result.success:
                await self._handle_success(job_id, result)
            else:
                await self._handle_failure(job_id, result)

        except asyncio.CancelledError:
            logger.info(f"Job {job_id} was cancelled")
            await self.db.update_job(job_id, {
                'status': 'cancelled',
                'completed_at': time.time()
            })
            raise

        except Exception as e:
            logger.exception(f"Job {job_id} failed with exception")
            await self.db.update_job(job_id, {
                'status': 'failed',
                'completed_at': time.time(),
                'error_message': str(e)
            })

        finally:
            # Remove from running jobs
            self.running_jobs.pop(job_id, None)
            self.running_adapters.pop(job_id, None)

    def _build_script_config(self, submission: JobSubmission) -> BashScriptConfig:
        """
        Build bash script configuration from job submission.

        Args:
            submission: Job submission details

        Returns:
            BashScriptConfig ready for execution
        """
        scripts_root = Path(self.config['paths']['scripts_root'])
        options = submission.options or {}

        if submission.pipeline_type == 'cpu_only':
            # CPU-only pipeline: run_cpu_tasks_all.sh
            script_path = scripts_root / "batch" / "run_cpu_tasks_all.sh"

            # run_cpu_tasks_all.sh expects INPUT_DIR (directory containing videos)
            # If input_video_path is a file, use its parent directory
            input_path = Path(submission.input_video_path)
            if input_path.is_file():
                input_dir = str(input_path.parent)
            else:
                input_dir = str(input_path)

            args = [
                submission.film_name,
                input_dir,
                submission.output_base_dir,
                "--workers",
                str(options.get('workers', 8))
            ]

            if options.get('monitor'):
                args.append("--monitor")
            if options.get('resume'):
                args.append("--resume")

            # Add environment variables from config
            env_vars = self.config.get('environment', {})

            return BashScriptConfig(
                script_path=str(script_path),
                args=args,
                env=env_vars,
                timeout=7200,  # 2 hours timeout for CPU pipeline
                parse_json_output=True,
                emit_progress_events=True,
            )

        elif submission.pipeline_type == 'cpu_gpu_full':
            # Full pipeline: run_all_tasks_complete.sh
            script_path = scripts_root / "batch" / "run_all_tasks_complete.sh"

            # run_all_tasks_complete.sh expects INPUT_DIR (directory containing videos)
            input_path = Path(submission.input_video_path)
            if input_path.is_file():
                input_dir = str(input_path.parent)
            else:
                input_dir = str(input_path)

            args = [
                submission.film_name,
                input_dir,
                submission.output_base_dir
            ]

            if options.get('enable_gpu', True):
                args.append("--enable-gpu")
            else:
                args.append("--disable-gpu")

            if options.get('enable_voice', False):
                args.append("--enable-voice")

            if 'parallel_jobs' in options:
                args.extend(["--parallel-jobs", str(options['parallel_jobs'])])
            if 'sam2_model' in options:
                args.extend(["--sam2-model", options['sam2_model']])
            if 'llm_model' in options:
                args.extend(["--llm-model", options['llm_model']])

            # Add environment variables from config
            env_vars = self.config.get('environment', {})

            return BashScriptConfig(
                script_path=str(script_path),
                args=args,
                env=env_vars,
                timeout=14400,  # 4 hours timeout for full pipeline
                parse_json_output=True,
                emit_progress_events=True,
            )

        elif submission.pipeline_type in (
            "action_generate",
            "action_animate",
            "action_inpaint",
            "action_multichar",
            "action_regress",
            "action_extract_controls",
        ):
            script_path = scripts_root / "batch" / "run_webui_action_task.sh"
            env_vars = dict(self.config.get("environment", {}) or {})
            env_vars["WEBUI_OUTPUT_DIR"] = str(submission.output_base_dir)

            task = submission.pipeline_type.replace("action_", "", 1)
            options = submission.options or {}

            args: List[str] = [task]

            def add_flag(flag: str, value: Optional[Any]):
                if value is None:
                    return
                if isinstance(value, str) and not value.strip():
                    return
                args.extend([flag, str(value)])

            def add_bool_flag(flag: str, enabled: bool):
                if enabled:
                    args.append(flag)

            # Common config overrides (optional)
            add_flag("--registry", options.get("registry"))
            add_flag("--sdxl-config", options.get("sdxl_config"))
            add_flag("--action-config", options.get("action_config"))
            add_flag("--lora-registry", options.get("lora_registry"))
            add_flag("--identity-lora", options.get("identity_lora"))
            add_flag("--identity-lora-weight", options.get("identity_lora_weight"))

            if submission.pipeline_type == "action_generate":
                add_flag("--character", options.get("character"))
                add_flag("--control-type", options.get("control_type", "auto"))
                add_flag("--control-image", submission.input_video_path)
                add_flag("--output-dir", submission.output_base_dir)

                add_flag("--action", options.get("action"))
                add_flag("--scene", options.get("scene"))
                add_flag("--prompt", options.get("prompt"))
                add_flag("--extra", options.get("extra"))
                add_flag("--style", options.get("style"))
                add_flag("--negative-prompt", options.get("negative_prompt"))
                add_flag("--negative-prompt-key", options.get("negative_prompt_key"))

                add_flag("--width", options.get("width"))
                add_flag("--height", options.get("height"))
                add_flag("--steps", options.get("steps"))
                add_flag("--guidance-scale", options.get("guidance_scale"))
                add_flag("--controlnet-scale", options.get("controlnet_scale"))
                add_flag("--seed", options.get("seed"))
                add_flag("--num-images", options.get("num_images"))
                add_bool_flag("--no-preprocess", bool(options.get("no_preprocess", False)))

                add_flag("--reference-image", options.get("reference_image"))
                add_flag("--consistency-threshold", options.get("consistency_threshold"))
                add_flag("--max-retries", options.get("max_retries"))
                add_flag("--consistency-device", options.get("consistency_device"))

            elif submission.pipeline_type == "action_animate":
                add_flag("--character", options.get("character"))
                add_flag("--control-type", options.get("control_type", "pose"))
                add_flag("--control-dir", options.get("control_dir") or submission.input_video_path)
                add_flag("--pattern", options.get("pattern"))
                add_flag("--every", options.get("every"))
                add_flag("--limit", options.get("limit"))
                add_flag("--output-dir", submission.output_base_dir)

                add_flag("--action", options.get("action"))
                add_flag("--scene", options.get("scene"))
                add_flag("--prompt", options.get("prompt"))
                add_flag("--extra", options.get("extra"))
                add_flag("--style", options.get("style"))
                add_flag("--negative-prompt", options.get("negative_prompt"))
                add_flag("--negative-prompt-key", options.get("negative_prompt_key"))

                add_flag("--width", options.get("width"))
                add_flag("--height", options.get("height"))
                add_flag("--steps", options.get("steps"))
                add_flag("--guidance-scale", options.get("guidance_scale"))
                add_flag("--controlnet-scale", options.get("controlnet_scale"))
                add_flag("--seed", options.get("seed"))
                add_flag("--seed-mode", options.get("seed_mode"))
                add_bool_flag("--no-preprocess", bool(options.get("no_preprocess", False)))

                # Multi-pass refinement (optional; supported by updated animate.py)
                refine_sequence = options.get("refine_sequence")
                if isinstance(refine_sequence, list) and refine_sequence:
                    args.append("--refine-sequence")
                    args.extend([str(x) for x in refine_sequence])
                add_flag("--refine-strength", options.get("refine_strength"))
                add_flag("--refine-steps", options.get("refine_steps"))
                add_flag("--refine-guidance-scale", options.get("refine_guidance_scale"))
                add_flag("--refine-controlnet-scale", options.get("refine_controlnet_scale"))
                add_flag("--controls-root", options.get("controls_root"))
                add_bool_flag("--save-intermediate", bool(options.get("save_intermediate", False)))

                add_bool_flag("--write-video", bool(options.get("write_video", False)))
                add_flag("--fps", options.get("fps"))
                add_flag("--ffmpeg", options.get("ffmpeg"))

                add_flag("--reference-image", options.get("reference_image"))
                add_flag("--consistency-threshold", options.get("consistency_threshold"))
                add_flag("--consistency-device", options.get("consistency_device"))
                add_flag("--max-retries", options.get("max_retries"))

            elif submission.pipeline_type == "action_inpaint":
                add_flag("--character", options.get("character"))
                add_flag("--control-type", options.get("control_type", "auto"))
                add_flag("--image", options.get("image") or submission.input_video_path)
                add_flag("--mask", options.get("mask"))
                add_flag("--control-image", options.get("control_image"))
                add_flag("--output-dir", submission.output_base_dir)

                add_flag("--action", options.get("action"))
                add_flag("--scene", options.get("scene"))
                add_flag("--prompt", options.get("prompt"))
                add_flag("--extra", options.get("extra"))
                add_flag("--style", options.get("style"))
                add_flag("--negative-prompt", options.get("negative_prompt"))
                add_flag("--negative-prompt-key", options.get("negative_prompt_key"))

                add_flag("--width", options.get("width"))
                add_flag("--height", options.get("height"))
                add_flag("--steps", options.get("steps"))
                add_flag("--guidance-scale", options.get("guidance_scale"))
                add_flag("--strength", options.get("strength"))
                add_flag("--controlnet-scale", options.get("controlnet_scale"))
                add_flag("--seed", options.get("seed"))
                add_bool_flag("--no-preprocess", bool(options.get("no_preprocess", False)))

            elif submission.pipeline_type == "action_multichar":
                add_flag("--config", options.get("config") or submission.input_video_path)
                add_flag("--output-dir", submission.output_base_dir)

            elif submission.pipeline_type == "action_regress":
                # Backwards compatible defaults.
                add_flag("--output-dir", submission.output_base_dir)
                add_flag("--control-image", options.get("control_image") or submission.input_video_path)
                add_flag("--control-image-pose", options.get("control_image_pose"))
                add_flag("--control-image-canny", options.get("control_image_canny"))
                add_flag("--control-image-softedge", options.get("control_image_softedge"))
                add_flag("--control-image-lineart", options.get("control_image_lineart"))
                add_flag("--control-image-tile", options.get("control_image_tile"))
                add_flag("--action", options.get("action"))
                add_flag("--scene", options.get("scene"))
                add_flag("--prompt", options.get("prompt"))
                add_flag("--extra", options.get("extra"))
                add_flag("--style", options.get("style"))
                add_flag("--negative-prompt", options.get("negative_prompt"))
                add_flag("--negative-prompt-key", options.get("negative_prompt_key"))
                add_flag("--width", options.get("width"))
                add_flag("--height", options.get("height"))
                add_flag("--steps", options.get("steps"))
                add_flag("--guidance-scale", options.get("guidance_scale"))
                add_flag("--seed", options.get("seed"))
                add_bool_flag("--no-preprocess", bool(options.get("no_preprocess", False)))
                add_bool_flag("--skip-existing", bool(options.get("skip_existing", False)))
                add_bool_flag("--dry-run", bool(options.get("dry_run", False)))
                # Suite config (new regress.py supports this; ignored otherwise).
                add_flag("--suite", options.get("suite"))
                add_flag("--reference-image", options.get("reference_image"))
                add_flag("--score-threshold", options.get("score_threshold"))

            elif submission.pipeline_type == "action_extract_controls":
                add_flag("--output-dir", submission.output_base_dir)
                mode = str(options.get("mode") or "video").strip().lower()
                if mode == "frames":
                    add_flag("--frames-dir", options.get("frames_dir") or submission.input_video_path)
                else:
                    add_flag("--video", options.get("video") or submission.input_video_path)
                add_flag("--pattern", options.get("pattern"))
                add_flag("--fps", options.get("fps"))
                types = options.get("types")
                if isinstance(types, list) and types:
                    args.append("--types")
                    args.extend([str(x) for x in types])
                add_flag("--detect-resolution", options.get("detect_resolution"))
                add_flag("--image-resolution", options.get("image_resolution"))
                add_bool_flag("--overwrite", bool(options.get("overwrite", False)))

            timeout_opt = options.get("timeout", None)
            timeout = int(timeout_opt) if timeout_opt is not None else 21600  # default 6 hours
            return BashScriptConfig(
                script_path=str(script_path),
                args=args,
                env=env_vars,
                timeout=timeout,
                parse_json_output=True,
                emit_progress_events=True,
            )

        elif submission.pipeline_type in (
            "creative_parody",
            "creative_analyze",
            "creative_voice",
            "creative_workflow",
        ):
            script_path = scripts_root / "batch" / "run_webui_creative_task.sh"
            env_vars = dict(self.config.get("environment", {}) or {})
            env_vars["WEBUI_OUTPUT_DIR"] = str(submission.output_base_dir)

            task = submission.pipeline_type.replace("creative_", "", 1)
            options = submission.options or {}

            args: List[str] = [task]

            def add_flag(flag: str, value: Optional[Any]):
                if value is None:
                    return
                if isinstance(value, str) and not value.strip():
                    return
                args.extend([flag, str(value)])

            def add_bool_flag(flag: str, enabled: bool):
                if enabled:
                    args.append(flag)

            # Common
            add_flag("--input", options.get("input"))
            add_flag("--output-dir", submission.output_base_dir)

            if submission.pipeline_type == "creative_parody":
                add_flag("--style", options.get("style"))
                add_flag("--duration", options.get("duration"))
                add_flag("--effects", options.get("effects"))
                add_flag("--output", options.get("output"))

            elif submission.pipeline_type == "creative_analyze":
                add_bool_flag("--visual", bool(options.get("visual", True)))
                add_bool_flag("--audio", bool(options.get("audio", False)))
                add_bool_flag("--context", bool(options.get("context", False)))
                add_flag("--sample-rate", options.get("sample_rate"))
                add_flag("--output", options.get("output"))

            elif submission.pipeline_type == "creative_voice":
                add_flag("--character", options.get("character"))
                add_flag("--text", options.get("text"))
                add_flag("--emotion", options.get("emotion"))
                add_flag("--intensity", options.get("intensity"))
                add_flag("--output", options.get("output"))

            elif submission.pipeline_type == "creative_workflow":
                add_flag("--workflow-type", options.get("workflow_type"))
                add_flag("--style", options.get("style"))
                add_flag("--duration", options.get("duration"))
                add_bool_flag("--audio", bool(options.get("audio", False)))
                add_bool_flag("--context", bool(options.get("context", False)))
                add_flag("--output", options.get("output"))

            timeout_opt = options.get("timeout", None)
            timeout = int(timeout_opt) if timeout_opt is not None else 21600
            return BashScriptConfig(
                script_path=str(script_path),
                args=args,
                env=env_vars,
                timeout=timeout,
                parse_json_output=True,
                emit_progress_events=True,
            )

        elif submission.pipeline_type == "image_generate":
            script_path = scripts_root / "batch" / "run_webui_image_task.sh"
            env_vars = dict(self.config.get("environment", {}) or {})
            env_vars["WEBUI_OUTPUT_DIR"] = str(submission.output_base_dir)
            options = submission.options or {}

            args: List[str] = ["generate"]

            def add_flag(flag: str, value: Optional[Any]):
                if value is None:
                    return
                if isinstance(value, str) and not value.strip():
                    return
                args.extend([flag, str(value)])

            add_flag("--provider", options.get("provider"))
            add_flag("--workflow", options.get("workflow"))
            add_flag("--prompt", options.get("prompt"))
            add_flag("--negative-prompt", options.get("negative_prompt"))
            add_flag("--style", options.get("style"))
            add_flag("--width", options.get("width"))
            add_flag("--height", options.get("height"))
            add_flag("--steps", options.get("steps"))
            add_flag("--guidance-scale", options.get("guidance_scale"))
            add_flag("--seed", options.get("seed"))
            add_flag("--num-images", options.get("num_images"))
            add_flag("--timeout", options.get("timeout"))

            timeout_opt = options.get("timeout", None)
            timeout = int(timeout_opt) if timeout_opt is not None else 3600
            return BashScriptConfig(
                script_path=str(script_path),
                args=args,
                env=env_vars,
                timeout=timeout,
                parse_json_output=True,
                emit_progress_events=True,
            )

        else:
            raise ValueError(f"Unsupported pipeline type: {submission.pipeline_type}")

    async def _execute_demo_job(self, job_id: str, submission: JobSubmission) -> None:
        """Run a deterministic mock job for portfolio demos."""
        output_dir = Path(submission.output_base_dir)
        logs_dir = output_dir / "webui_logs" / job_id
        gallery_dir = output_dir / "gallery"
        manifests_dir = output_dir / "manifests"
        logs_dir.mkdir(parents=True, exist_ok=True)
        gallery_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = logs_dir / "stdout.log"
        stderr_path = logs_dir / "stderr.log"

        stages = [
            ("intake", 0.18, "Validated request and selected demo scenario"),
            ("planning", 0.38, "Resolved provider strategy and shot plan"),
            ("generation", 0.68, "Rendered sample frames and metadata"),
            ("quality", 0.86, "Collected QC signals and result summaries"),
            ("complete", 1.0, "Demo job completed"),
        ]

        stdout_lines = [
            f"Demo mode job: {submission.pipeline_type}",
            f"Film/project: {submission.film_name}",
            f"Output: {output_dir}",
        ]
        for index, (stage, progress, message) in enumerate(stages, start=1):
            line = f"Processing stage {index}/{len(stages)} - {message}"
            stdout_lines.append(line)
            await self._handle_progress_event(
                job_id,
                {
                    "type": "progress",
                    "job_id": job_id,
                    "progress": progress,
                    "stage": stage,
                    "message": message,
                },
            )
            await self.db.update_job(
                job_id,
                {
                    "progress_percent": progress * 100,
                    "current_stage": stage,
                },
            )
            await asyncio.sleep(0.08)

        summary = {
            "job_id": job_id,
            "film_name": submission.film_name,
            "pipeline_type": submission.pipeline_type,
            "mode": "demo",
            "scenario": (submission.options or {}).get("scenario", "portfolio_interview"),
            "highlights": [
                "Provider-aware routing",
                "Queued job orchestration",
                "Realtime progress stream",
                "Result artifact browser",
                "System resource dashboard",
            ],
            "outputs": {
                "gallery": str(gallery_dir),
                "storyboard": str(manifests_dir / "storyboard.json"),
                "quality_report": str(manifests_dir / "quality_report.json"),
            },
        }
        storyboard = {
            "title": "Demo: Shot-Based Animation Pipeline",
            "shots": [
                {
                    "shot_id": "shot-001",
                    "title": "Character entrance",
                    "provider": "flux2_klein",
                    "status": "preview_ready",
                    "notes": "Mocked still used for demo-safe portfolio flow.",
                },
                {
                    "shot_id": "shot-002",
                    "title": "Dialogue beat",
                    "provider": "ltx23_api",
                    "status": "manifest_ready",
                    "notes": "Shows audio-video task preparation without external API calls.",
                },
                {
                    "shot_id": "shot-003",
                    "title": "Export package",
                    "provider": "ffmpeg",
                    "status": "assembled",
                    "notes": "Composition/export artifacts are represented as metadata.",
                },
            ],
        }
        quality = {
            "continuity_score": 0.91,
            "prompt_alignment": 0.88,
            "asset_completeness": 1.0,
            "review_status": "demo_ready",
        }

        def write_json(path: Path, payload: Dict[str, Any]) -> None:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        write_json(output_dir / "summary.json", summary)
        write_json(manifests_dir / "storyboard.json", storyboard)
        write_json(manifests_dir / "quality_report.json", quality)
        (gallery_dir / "shot-001-preview.svg").write_text(
            """<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<rect width="1280" height="720" fill="#101820"/>
<rect x="80" y="80" width="1120" height="560" rx="24" fill="#f4f7fb"/>
<text x="140" y="170" font-family="Arial" font-size="52" font-weight="700" fill="#101820">Animation AI Studio</text>
<text x="140" y="240" font-family="Arial" font-size="30" fill="#415063">Shot-based demo artifact</text>
<rect x="140" y="320" width="280" height="180" rx="18" fill="#2f80ed"/>
<rect x="500" y="320" width="280" height="180" rx="18" fill="#27ae60"/>
<rect x="860" y="320" width="280" height="180" rx="18" fill="#f2994a"/>
<text x="170" y="425" font-family="Arial" font-size="28" fill="#ffffff">Plan</text>
<text x="530" y="425" font-family="Arial" font-size="28" fill="#ffffff">Generate</text>
<text x="890" y="425" font-family="Arial" font-size="28" fill="#ffffff">Review</text>
</svg>
""",
            encoding="utf-8",
        )

        stdout_lines.append(f"Summary: {output_dir / 'summary.json'}")
        stdout_path.write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        await self.db.add_job_output(job_id, "execution", "stdout_log", str(stdout_path))
        await self.db.add_job_output(job_id, "execution", "stderr_log", str(stderr_path))
        await self.db.add_job_output(job_id, "summary", "json", str(output_dir / "summary.json"))
        await self.db.add_job_output(job_id, "storyboard", "json", str(manifests_dir / "storyboard.json"))
        await self.db.add_job_output(job_id, "gallery", "preview", str(gallery_dir), file_count=1)

        await self.db.update_job(
            job_id,
            {
                "status": "completed",
                "completed_at": time.time(),
                "progress_percent": 100.0,
                "current_stage": "complete",
            },
        )
        await self._handle_progress_event(
            job_id,
            {
                "type": "completed",
                "job_id": job_id,
                "progress": 1.0,
                "stage": "complete",
                "message": "Demo job completed",
            },
        )

    async def _handle_progress_event(self, job_id: str, event: Dict[str, Any]):
        """
        Handle progress event from script execution.

        Args:
            job_id: Job identifier
            event: Progress event data
        """
        event_type = event.get('type')

        if event_type == 'progress':
            # Update progress in database
            progress = event.get('progress', 0.0)
            stage = event.get('stage', '')

            await self.db.update_job(job_id, {
                'progress_percent': progress * 100,
                'current_stage': stage
            })

        # Notify subscribers (for SSE streaming)
        if job_id in self.progress_callbacks:
            for callback in self.progress_callbacks[job_id]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

    async def _handle_success(self, job_id: str, result: ScriptResult):
        """
        Handle successful job completion.

        Args:
            job_id: Job identifier
            result: Script execution result
        """
        logger.info(f"Job {job_id} completed successfully in {result.execution_time:.1f}s")

        # Update job status
        await self.db.update_job(job_id, {
            'status': 'completed',
            'completed_at': time.time(),
            'progress_percent': 100.0
        })

        # Record logs (if available)
        if result.stdout_log_path:
            await self.db.add_job_output(
                job_id=job_id,
                stage='execution',
                output_type='stdout_log',
                path=str(result.stdout_log_path),
            )
        if result.stderr_log_path:
            await self.db.add_job_output(
                job_id=job_id,
                stage='execution',
                output_type='stderr_log',
                path=str(result.stderr_log_path),
            )

        # Parse and store outputs
        outputs = result.outputs
        if 'output_dir' in outputs:
            output_dir = Path(outputs['output_dir'])

            # Record frame extraction output
            frames_dir = output_dir / "frames"
            if frames_dir.exists():
                frame_count = len(list(frames_dir.glob("**/*.jpg"))) + len(list(frames_dir.glob("**/*.png")))
                await self.db.add_job_output(
                    job_id=job_id,
                    stage='frame_extraction',
                    output_type='frames',
                    path=str(frames_dir),
                    file_count=frame_count
                )

            # Record analysis output
            analysis_dir = output_dir / "analysis"
            if analysis_dir.exists():
                await self.db.add_job_output(
                    job_id=job_id,
                    stage='analysis',
                    output_type='json',
                    path=str(analysis_dir)
                )

            # Record RAG output
            rag_dir = output_dir / "rag"
            if rag_dir.exists():
                await self.db.add_job_output(
                    job_id=job_id,
                    stage='rag_preparation',
                    output_type='knowledge_base',
                    path=str(rag_dir)
                )

    async def _handle_failure(self, job_id: str, result: ScriptResult):
        """
        Handle job failure.

        Args:
            job_id: Job identifier
            result: Script execution result
        """
        logger.error(f"Job {job_id} failed: {result.error}")

        await self.db.update_job(job_id, {
            'status': 'failed',
            'completed_at': time.time(),
            'error_message': result.error
        })

        # Record logs (if available)
        if result.stdout_log_path:
            await self.db.add_job_output(
                job_id=job_id,
                stage='execution',
                output_type='stdout_log',
                path=str(result.stdout_log_path),
            )
        if result.stderr_log_path:
            await self.db.add_job_output(
                job_id=job_id,
                stage='execution',
                output_type='stderr_log',
                path=str(result.stderr_log_path),
            )

    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """
        Get job information by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobInfo if found, None otherwise
        """
        return await self.db.get_job(job_id)

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[JobInfo]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status (optional)
            limit: Maximum number of jobs to return
            offset: Pagination offset

        Returns:
            List of JobInfo objects
        """
        return await self.db.list_jobs(status=status, limit=limit, offset=offset)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not running
        """
        if job_id not in self.running_jobs:
            logger.warning(f"Cannot cancel job {job_id}: not running")
            return False

        adapter = self.running_adapters.get(job_id)
        if adapter is not None:
            try:
                await adapter.cancel()
            except Exception as e:
                logger.warning(f"Failed to cancel subprocess for job {job_id}: {e}")

        task = self.running_jobs[job_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        return True

    def subscribe_progress(self, job_id: str, callback):
        """
        Subscribe to progress events for a job.

        Args:
            job_id: Job identifier
            callback: Async callback function
        """
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []

        self.progress_callbacks[job_id].append(callback)

    def unsubscribe_progress(self, job_id: str, callback):
        """
        Unsubscribe from progress events.

        Args:
            job_id: Job identifier
            callback: Callback to remove
        """
        if job_id in self.progress_callbacks:
            try:
                self.progress_callbacks[job_id].remove(callback)
            except ValueError:
                pass

            # Clean up empty lists
            if not self.progress_callbacks[job_id]:
                del self.progress_callbacks[job_id]
