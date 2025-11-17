"""
Test Suite for Module 8: Video Editing

Tests all components:
- Character Segmentation
- Video Editing Operations
- LLM Decision Engine
- Quality Evaluation
- Parody Generation
- Agent Framework Integration

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.editing.segmentation.character_segmenter import CharacterSegmenter
from scripts.editing.engine.video_editor import VideoEditor
from scripts.editing.decision.llm_decision_engine import LLMDecisionEngine
from scripts.editing.quality.quality_evaluator import QualityEvaluator
from scripts.editing.effects.parody_generator import ParodyGenerator


# Test fixtures
@pytest.fixture
def sample_video_path():
    """Sample video path (mock for testing)"""
    return "test_data/sample_video.mp4"


@pytest.fixture
def output_dir():
    """Output directory for test results"""
    output = Path("outputs/tests/module8")
    output.mkdir(parents=True, exist_ok=True)
    return output


# Character Segmentation Tests
class TestCharacterSegmentation:
    """Test character segmentation"""

    def test_segmenter_initialization(self):
        """Test segmenter initialization"""
        segmenter = CharacterSegmenter(
            model_size="base",
            device="cuda" if os.path.exists("/dev/nvidia0") else "cpu"
        )

        assert segmenter is not None
        assert segmenter.model_type in ["tiny", "small", "base", "large"]

    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    def test_video_segmentation(self, sample_video_path, output_dir):
        """Test video segmentation"""
        segmenter = CharacterSegmenter(model_size="base")

        result = segmenter.segment_video(
            video_path=sample_video_path,
            sample_interval=10,
            output_masks_dir=str(output_dir / "masks"),
            track_characters=True
        )

        assert result.success
        assert result.total_frames > 0
        assert len(result.character_tracks) >= 0


# Video Editing Tests
class TestVideoEditor:
    """Test video editing operations"""

    def test_editor_initialization(self):
        """Test editor initialization"""
        editor = VideoEditor()
        assert editor is not None

    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    def test_cut_clip(self, sample_video_path, output_dir):
        """Test cutting clip"""
        editor = VideoEditor()

        result = editor.cut_clip(
            video_path=sample_video_path,
            start_time=0.0,
            end_time=5.0,
            output_path=str(output_dir / "cut_clip.mp4")
        )

        assert result.success
        assert result.output_duration == pytest.approx(5.0, abs=0.5)

    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    def test_change_speed(self, sample_video_path, output_dir):
        """Test speed change"""
        editor = VideoEditor()

        result = editor.change_speed(
            video_path=sample_video_path,
            speed_factor=2.0,
            output_path=str(output_dir / "speed_2x.mp4")
        )

        assert result.success
        assert result.output_path is not None


# LLM Decision Engine Tests
class TestLLMDecisionEngine:
    """Test LLM decision engine"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization"""
        async with LLMDecisionEngine() as engine:
            assert engine is not None
            assert engine.model is not None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    async def test_create_edit_plan(self, sample_video_path):
        """Test creating edit plan"""
        async with LLMDecisionEngine() as engine:
            plan = await engine.create_edit_plan(
                video_path=sample_video_path,
                goal="Create a 30-second highlight reel"
            )

            assert plan is not None
            assert plan.goal == "Create a 30-second highlight reel"
            assert len(plan.decisions) > 0

    @pytest.mark.asyncio
    async def test_decision_validation(self):
        """Test decision validation"""
        async with LLMDecisionEngine() as engine:
            # Test with various decision types
            decision_types = ["cut", "speed", "composite", "text_overlay", "effect"]

            for dt in decision_types:
                is_valid = engine._validate_decision_parameters(dt, {})
                # Should return bool
                assert isinstance(is_valid, bool)


# Quality Evaluation Tests
class TestQualityEvaluator:
    """Test quality evaluation"""

    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = QualityEvaluator()
        assert evaluator is not None

    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    def test_quality_evaluation(self, sample_video_path):
        """Test quality evaluation"""
        evaluator = QualityEvaluator()

        metrics = evaluator.evaluate(
            video_path=sample_video_path,
            goal="Test evaluation",
            quality_threshold=0.7
        )

        assert metrics is not None
        assert 0.0 <= metrics.overall_score <= 1.0
        assert 0.0 <= metrics.technical_score <= 1.0
        assert 0.0 <= metrics.creative_score <= 1.0
        assert isinstance(metrics.needs_improvement, bool)

    def test_feedback_generation(self):
        """Test feedback generation"""
        evaluator = QualityEvaluator()

        technical_metrics = {
            "composition_score": 0.8,
            "temporal_coherence_score": 0.7,
            "pacing_score": 0.6
        }

        feedback = evaluator._generate_feedback(
            overall_score=0.75,
            technical_metrics=technical_metrics,
            creative_score=0.7,
            goal_achievement_score=0.8,
            needs_improvement=False
        )

        assert isinstance(feedback, str)
        assert len(feedback) > 0


# Parody Generation Tests
class TestParodyGenerator:
    """Test parody generation"""

    def test_generator_initialization(self):
        """Test generator initialization"""
        try:
            generator = ParodyGenerator()
            assert generator is not None
        except ImportError:
            pytest.skip("MoviePy not available")

    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    def test_zoom_punch(self, sample_video_path, output_dir):
        """Test zoom punch effect"""
        try:
            generator = ParodyGenerator()

            result = generator.apply_zoom_punch(
                video_path=sample_video_path,
                zoom_time=2.0,
                output_path=str(output_dir / "zoom_punch.mp4"),
                zoom_factor=1.5
            )

            assert result["success"]
            assert result["effect"] == "zoom_punch"
        except ImportError:
            pytest.skip("MoviePy not available")


# Agent Integration Tests
class TestAgentIntegration:
    """Test Agent Framework integration"""

    @pytest.mark.asyncio
    async def test_segment_characters_tool(self, sample_video_path):
        """Test character segmentation tool"""
        from scripts.agent.tools.video_editing_tools import segment_characters

        if not Path(sample_video_path).exists():
            pytest.skip("Sample video not available")

        result = await segment_characters(
            video_path=sample_video_path,
            model_size="base",
            sample_interval=10
        )

        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_cut_video_tool(self, sample_video_path, output_dir):
        """Test cut video tool"""
        from scripts.agent.tools.video_editing_tools import cut_video_clip

        if not Path(sample_video_path).exists():
            pytest.skip("Sample video not available")

        result = await cut_video_clip(
            video_path=sample_video_path,
            start_time=0.0,
            end_time=5.0,
            output_path=str(output_dir / "cut_tool.mp4")
        )

        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_create_edit_plan_tool(self, sample_video_path):
        """Test create edit plan tool"""
        from scripts.agent.tools.video_editing_tools import create_edit_plan

        if not Path(sample_video_path).exists():
            pytest.skip("Sample video not available")

        result = await create_edit_plan(
            video_path=sample_video_path,
            goal="Create highlight reel"
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_tool_registration(self):
        """Test tool registration"""
        from scripts.agent.tools.tool_registry import create_default_tool_registry

        registry = create_default_tool_registry()

        # Check that video editing tools are registered
        editing_tools = [
            "segment_characters",
            "cut_video_clip",
            "change_video_speed",
            "create_edit_plan",
            "evaluate_video_quality",
            "create_parody_video",
            "auto_edit_video"
        ]

        for tool_name in editing_tools:
            tool = registry.get_tool(tool_name)
            assert tool is not None, f"Tool {tool_name} not registered"
            assert tool.name == tool_name


# Performance Tests
class TestPerformance:
    """Test performance characteristics"""

    def test_vram_estimates(self):
        """Test VRAM estimates are reasonable"""
        from scripts.agent.tools.tool_registry import create_default_tool_registry

        registry = create_default_tool_registry()

        # SAM2 base should use ~6GB
        segment_tool = registry.get_tool("segment_characters")
        assert segment_tool.estimated_vram_gb == pytest.approx(6.0, abs=2.0)

        # Auto edit should account for SAM2
        auto_edit_tool = registry.get_tool("auto_edit_video")
        assert auto_edit_tool.estimated_vram_gb == pytest.approx(6.0, abs=2.0)

    def test_gpu_availability_check(self):
        """Test GPU availability checking"""
        from scripts.agent.tools.tool_registry import create_default_tool_registry

        registry = create_default_tool_registry()

        # Should fit on RTX 5080 16GB
        can_run = registry.check_gpu_availability([
            "segment_characters",
            "cut_video_clip"  # No GPU
        ])

        assert can_run  # 6GB + 0GB = 6GB < 15.5GB

        # Should NOT fit if multiple heavy models
        can_run_heavy = registry.check_gpu_availability([
            "generate_character_image",  # 13GB
            "segment_characters"  # 6GB
        ])

        assert not can_run_heavy  # 19GB > 15.5GB


def main():
    """Run tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
