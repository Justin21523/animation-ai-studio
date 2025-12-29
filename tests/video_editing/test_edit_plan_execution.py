"""
Tests for edit plan execution (Module 8 → Creative Studio P2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_dummy_video(path: Path, duration: float = 2.0):
    try:
        from moviepy.editor import ColorClip
    except Exception as e:  # pragma: no cover
        pytest.skip(f"moviepy not available: {e}")

    path.parent.mkdir(parents=True, exist_ok=True)
    clip = ColorClip(size=(160, 90), color=(255, 0, 0)).set_duration(duration)
    clip.write_videofile(str(path), fps=12, codec="libx264", audio=False, logger=None)
    clip.close()


@pytest.mark.asyncio
async def test_execute_edit_plan_cut_speed_text(tmp_path: Path):
    from scripts.agent.tools.video_editing_tools import execute_edit_plan

    video_path = tmp_path / "input.mp4"
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "output.mp4"

    _write_dummy_video(video_path, duration=4.0)

    plan = {
        "plan_id": "plan_test",
        "video_path": str(video_path),
        "goal": "test",
        "decisions": [
            {
                "decision_id": "cut_001",
                "decision_type": "cut",
                "confidence": 0.9,
                "reasoning": "first half",
                "parameters": {"start_time": 0.0, "end_time": 2.0},
                "priority": 9,
            },
            {
                "decision_id": "cut_002",
                "decision_type": "cut",
                "confidence": 0.9,
                "reasoning": "second half",
                "parameters": {"start_time": 2.0, "end_time": 4.0},
                "priority": 8,
            },
            {
                "decision_id": "speed_001",
                "decision_type": "speed",
                "confidence": 0.8,
                "reasoning": "speed up a bit",
                "parameters": {"start_time": 0.5, "end_time": 1.5, "speed_factor": 2.0},
                "priority": 6,
            },
            {
                "decision_id": "transition_001",
                "decision_type": "transition",
                "confidence": 0.7,
                "reasoning": "smooth join",
                "parameters": {"transition_type": "crossfade", "duration": 0.2},
                "priority": 5,
            },
            {
                "decision_id": "text_001",
                "decision_type": "text_overlay",
                "confidence": 0.7,
                "reasoning": "caption",
                "parameters": {"text": "HELLO", "start_time": 0.2, "duration": 0.5, "position": [10, 10]},
                "priority": 4,
            },
        ],
        "quality_threshold": 0.7,
        "max_iterations": 1,
        "metadata": {},
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = await execute_edit_plan(
        video_path=str(video_path),
        plan_json_path=str(plan_path),
        output_path=str(output_path),
        target_duration=3.0,
    )

    assert result["success"] is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0

