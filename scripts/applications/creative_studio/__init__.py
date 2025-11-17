"""
Module 9: Creative Studio

Complete AI-Powered Creative Content Generation Platform

Integrates all modules for end-to-end creative workflows.

Author: Animation AI Studio
Date: 2025-11-17
"""

from scripts.applications.creative_studio.parody_video_generator import (
    ParodyVideoGenerator,
    ParodyGenerationResult
)

from scripts.applications.creative_studio.multimodal_analysis_pipeline import (
    MultimodalAnalysisPipeline,
    MultimodalAnalysisResult
)

from scripts.applications.creative_studio.creative_workflows import (
    CreativeWorkflows,
    WorkflowResult
)


__all__ = [
    "ParodyVideoGenerator",
    "ParodyGenerationResult",
    "MultimodalAnalysisPipeline",
    "MultimodalAnalysisResult",
    "CreativeWorkflows",
    "WorkflowResult"
]

__version__ = "1.0.0"
