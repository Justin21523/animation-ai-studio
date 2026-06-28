"""Seed portfolio demo jobs and outputs for the Web UI."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_ui.backend.db.operations import JobDatabase
from web_ui.backend.db.schema import init_database


DEFAULT_CONFIG = REPO_ROOT / "configs" / "web_ui" / "demo.yaml"


SCENARIOS = [
    {
        "job_id": "demo-shot-planning",
        "film_name": "Portfolio Scenario - Shot Planning",
        "pipeline_type": "creative_analyze",
        "status": "completed",
        "progress_percent": 100.0,
        "stage": "complete",
        "summary": {
            "value": "Turns story beats into shot manifests, prompts, and reviewable production tasks.",
            "highlights": ["typed project schema", "shot manifests", "provider routing"],
        },
    },
    {
        "job_id": "demo-image-provider-routing",
        "film_name": "Portfolio Scenario - Provider Routing",
        "pipeline_type": "image_generate",
        "status": "completed",
        "progress_percent": 100.0,
        "stage": "complete",
        "summary": {
            "value": "Chooses local image providers by workflow intent and model availability.",
            "highlights": ["FLUX.2", "Qwen-Image", "Z-Image", "ComfyUI workflows"],
        },
    },
    {
        "job_id": "demo-live-monitoring",
        "film_name": "Portfolio Scenario - Live Monitoring",
        "pipeline_type": "cpu_only",
        "status": "running",
        "progress_percent": 64.0,
        "stage": "quality",
        "summary": {
            "value": "Shows queued execution, SSE progress, logs, and system monitoring.",
            "highlights": ["FastAPI", "SQLite", "SSE", "resource dashboard"],
        },
    },
    {
        "job_id": "demo-failure-handling",
        "film_name": "Portfolio Scenario - Failure Handling",
        "pipeline_type": "action_generate",
        "status": "failed",
        "progress_percent": 42.0,
        "stage": "generation",
        "error_message": "Demo failure: missing external ComfyUI runtime in full mode.",
        "summary": {
            "value": "Documents clear failure states for external model/runtime dependencies.",
            "highlights": ["safe error reporting", "logs", "retry-friendly job model"],
        },
    },
]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_svg(path: Path, title: str, accent: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<rect width="1280" height="720" fill="#101820"/>
<rect x="78" y="76" width="1124" height="568" rx="22" fill="#f7fafc"/>
<text x="130" y="165" font-family="Arial" font-size="50" font-weight="700" fill="#101820">{title}</text>
<rect x="130" y="245" width="1020" height="20" rx="10" fill="{accent}"/>
<text x="130" y="350" font-family="Arial" font-size="32" fill="#344054">Demo-safe generated artifact</text>
<text x="130" y="405" font-family="Arial" font-size="24" fill="#667085">Use this state for screenshots and interview walkthroughs.</text>
<circle cx="990" cy="450" r="88" fill="{accent}" opacity="0.92"/>
<text x="942" y="462" font-family="Arial" font-size="30" font-weight="700" fill="#ffffff">AAS</text>
</svg>
""",
        encoding="utf-8",
    )


async def seed(config_path: Path, reset: bool) -> None:
    config = load_config(config_path)
    db_path = resolve_repo_path(config["database"]["path"])
    outputs_root = resolve_repo_path(config["paths"]["outputs_root"])

    if reset and outputs_root.exists():
        shutil.rmtree(outputs_root)

    outputs_root.mkdir(parents=True, exist_ok=True)
    init_database(str(db_path))
    db = JobDatabase(str(db_path))

    base_time = time.time() - 3600
    for index, scenario in enumerate(SCENARIOS):
        job_dir = outputs_root / "scenarios" / scenario["job_id"]
        logs_dir = job_dir / "webui_logs" / scenario["job_id"]
        gallery_dir = job_dir / "gallery"
        manifests_dir = job_dir / "manifests"
        logs_dir.mkdir(parents=True, exist_ok=True)
        gallery_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "job_id": scenario["job_id"],
            "title": scenario["film_name"],
            "mode": "seeded_demo",
            **scenario["summary"],
        }
        quality = {
            "demo_readiness": "ready" if scenario["status"] != "failed" else "documented_failure",
            "screenshot_score": 0.95,
            "review_notes": scenario["summary"]["highlights"],
        }
        write_json(job_dir / "summary.json", summary)
        write_json(manifests_dir / "quality_report.json", quality)
        write_svg(gallery_dir / "preview.svg", scenario["film_name"], ["#2f80ed", "#27ae60", "#f2994a", "#eb5757"][index])

        stdout = [
            f"Seeded demo job: {scenario['film_name']}",
            f"Processing stage 1/4 - intake",
            f"Processing stage 2/4 - planning",
            f"Processing stage 3/4 - generation",
            f"Processing stage 4/4 - review",
            f"Output: {job_dir}",
            f"Summary: {job_dir / 'summary.json'}",
        ]
        (logs_dir / "stdout.log").write_text("\n".join(stdout) + "\n", encoding="utf-8")
        (logs_dir / "stderr.log").write_text((scenario.get("error_message") or "") + "\n", encoding="utf-8")

        await db.create_job(
            job_id=scenario["job_id"],
            film_name=scenario["film_name"],
            pipeline_type=scenario["pipeline_type"],
            input_video_path=str(outputs_root / "demo_input.mp4"),
            output_base_dir=str(job_dir),
            options={"scenario": scenario["job_id"], "seeded": True},
        )
        started = base_time + (index * 240)
        completed = started + 95 if scenario["status"] != "running" else None
        updates = {
            "status": scenario["status"],
            "progress_percent": scenario["progress_percent"],
            "current_stage": scenario["stage"],
            "started_at": started,
            "completed_at": completed,
            "error_message": scenario.get("error_message"),
        }
        await db.update_job(scenario["job_id"], updates)
        await db.add_job_output(scenario["job_id"], "execution", "stdout_log", str(logs_dir / "stdout.log"))
        await db.add_job_output(scenario["job_id"], "execution", "stderr_log", str(logs_dir / "stderr.log"))
        await db.add_job_output(scenario["job_id"], "summary", "json", str(job_dir / "summary.json"))
        await db.add_job_output(scenario["job_id"], "quality", "json", str(manifests_dir / "quality_report.json"))
        await db.add_job_output(scenario["job_id"], "gallery", "preview", str(gallery_dir), file_count=1)

    await db.close()
    print(f"Seeded {len(SCENARIOS)} demo jobs into {db_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Animation AI Studio portfolio demo data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reset", action="store_true", help="Delete existing demo outputs first")
    args = parser.parse_args()
    asyncio.run(seed(args.config, args.reset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
