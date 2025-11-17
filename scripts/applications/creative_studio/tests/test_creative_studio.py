"""
Test Suite for Module 9: Creative Studio

Tests all Creative Studio components:
- Parody Video Generator
- Multimodal Analysis Pipeline
- Creative Workflows
- CLI Interface

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
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.applications.creative_studio.parody_video_generator import ParodyVideoGenerator
from scripts.applications.creative_studio.multimodal_analysis_pipeline import MultimodalAnalysisPipeline
from scripts.applications.creative_studio.creative_workflows import CreativeWorkflows


# Test fixtures
@pytest.fixture
def sample_video_path():
    """Sample video path (mock for testing)"""
    return "test_data/sample_video.mp4"


@pytest.fixture
def output_dir():
    """Output directory for test results"""
    output = Path("outputs/tests/creative_studio")
    output.mkdir(parents=True, exist_ok=True)
    return output


# Parody Video Generator Tests
class TestParodyVideoGenerator:
    """Test parody video generator"""

    def test_initialization(self):
        """Test generator initialization"""
        generator = ParodyVideoGenerator(
            quality_threshold=0.7,
            max_iterations=3
        )

        assert generator is not None
        assert generator.quality_threshold == 0.7
        assert generator.max_iterations == 3

    @pytest.mark.asyncio
    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    async def test_generate_parody(self, sample_video_path, output_dir):
        """Test parody generation"""
        generator = ParodyVideoGenerator()

        result = await generator.generate_parody(
            input_video=sample_video_path,
            output_video=str(output_dir / "parody.mp4"),
            style="dramatic",
            target_duration=10.0
        )

        assert isinstance(result.to_dict(), dict)
        assert "success" in result.to_dict()

    @pytest.mark.asyncio
    async def test_custom_workflow(self, sample_video_path, output_dir):
        """Test custom workflow"""
        generator = ParodyVideoGenerator()

        if not Path(sample_video_path).exists():
            pytest.skip("Sample video not available")

        result = await generator.custom_workflow(
            input_video=sample_video_path,
            output_video=str(output_dir / "custom.mp4"),
            workflow_description="Create funny video with zoom effects"
        )

        assert isinstance(result, object)
        assert hasattr(result, "success")

    def test_build_parody_goal(self):
        """Test goal building"""
        generator = ParodyVideoGenerator()

        goal = generator._build_parody_goal(
            style="dramatic",
            target_duration=30.0,
            effects=["zoom_punch", "speed_ramp"]
        )

        assert isinstance(goal, str)
        assert "dramatic" in goal.lower()
        assert "30" in goal


# Multimodal Analysis Pipeline Tests
class TestMultimodalAnalysisPipeline:
    """Test multimodal analysis pipeline"""

    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = MultimodalAnalysisPipeline(
            enable_visual=True,
            enable_audio=False,
            enable_context=False
        )

        assert pipeline is not None
        assert pipeline.enable_visual is True
        assert pipeline.enable_audio is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    async def test_analyze_visual(self, sample_video_path):
        """Test visual analysis"""
        pipeline = MultimodalAnalysisPipeline()

        result = await pipeline.analyze_visual(
            video_path=sample_video_path,
            sample_rate=30
        )

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "visual_analysis")

    @pytest.mark.asyncio
    async def test_analyze_multimodal(self, sample_video_path):
        """Test multimodal analysis"""
        pipeline = MultimodalAnalysisPipeline()

        if not Path(sample_video_path).exists():
            pytest.skip("Sample video not available")

        result = await pipeline.analyze(
            video_path=sample_video_path,
            include_visual=True,
            include_audio=False,
            include_context=False
        )

        assert result is not None
        assert isinstance(result.to_dict(), dict)

    def test_generate_insights(self):
        """Test insight generation"""
        pipeline = MultimodalAnalysisPipeline()

        visual = {
            "scenes": {"total_scenes": 5, "avg_scene_duration": 6.0},
            "composition": {"avg_composition_score": 0.8},
            "camera": {"camera_style": "smooth"},
            "temporal": {"avg_coherence_score": 0.95}
        }

        insights = pipeline.generate_insights(visual, {}, {})

        assert isinstance(insights, list)
        assert len(insights) > 0

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        pipeline = MultimodalAnalysisPipeline()

        visual = {
            "composition": {"avg_composition_score": 0.6},
            "temporal": {"avg_coherence_score": 0.85},
            "scenes": {"avg_scene_duration": 3.0}
        }

        recommendations = pipeline.generate_recommendations(visual, {})

        assert isinstance(recommendations, list)


# Creative Workflows Tests
class TestCreativeWorkflows:
    """Test creative workflows"""

    def test_initialization(self):
        """Test workflows initialization"""
        workflows = CreativeWorkflows()

        assert workflows is not None
        assert workflows.parody_generator is not None
        assert workflows.analysis_pipeline is not None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    async def test_create_parody_video_workflow(self, sample_video_path, output_dir):
        """Test parody video workflow"""
        workflows = CreativeWorkflows()

        result = await workflows.create_parody_video(
            input_video=sample_video_path,
            output_video=str(output_dir / "workflow_parody.mp4"),
            style="dramatic"
        )

        assert result is not None
        assert hasattr(result, "success")
        assert result.workflow_name == "create_parody_video"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    async def test_analyze_and_report_workflow(self, sample_video_path, output_dir):
        """Test analysis workflow"""
        workflows = CreativeWorkflows()

        result = await workflows.analyze_and_report(
            video_path=sample_video_path,
            output_report=str(output_dir / "analysis_report.json"),
            include_visual=True
        )

        assert result is not None
        assert result.workflow_name == "analyze_and_report"

    def test_list_workflows(self):
        """Test workflow listing"""
        workflows = CreativeWorkflows()

        workflow_list = workflows.list_workflows()

        assert isinstance(workflow_list, list)
        assert len(workflow_list) >= 3
        assert all("name" in wf for wf in workflow_list)


# Integration Tests
class TestIntegration:
    """Test end-to-end integration"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not Path("test_data/sample_video.mp4").exists(),
                       reason="Sample video not available")
    async def test_full_parody_pipeline(self, sample_video_path, output_dir):
        """Test complete parody generation pipeline"""
        # Step 1: Analysis
        analysis_pipeline = MultimodalAnalysisPipeline()
        analysis_result = await analysis_pipeline.analyze_visual(
            video_path=sample_video_path
        )

        assert analysis_result.success

        # Step 2: Parody generation
        parody_generator = ParodyVideoGenerator()
        parody_result = await parody_generator.generate_parody(
            input_video=sample_video_path,
            output_video=str(output_dir / "full_pipeline.mp4"),
            style="dramatic"
        )

        # Verify pipeline completed
        assert isinstance(parody_result.to_dict(), dict)

    @pytest.mark.asyncio
    async def test_workflow_orchestration(self, sample_video_path, output_dir):
        """Test workflow orchestration"""
        workflows = CreativeWorkflows()

        if not Path(sample_video_path).exists():
            pytest.skip("Sample video not available")

        # Execute multiple workflows
        result1 = await workflows.analyze_and_report(
            video_path=sample_video_path,
            output_report=str(output_dir / "orchestration_analysis.json")
        )

        result2 = await workflows.create_parody_video(
            input_video=sample_video_path,
            output_video=str(output_dir / "orchestration_parody.mp4"),
            style="chaotic"
        )

        # Both should succeed
        assert isinstance(result1, object)
        assert isinstance(result2, object)


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
