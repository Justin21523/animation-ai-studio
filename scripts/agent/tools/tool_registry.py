"""
Tool Registry for Agent Framework

Manages available tools and their metadata.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories"""
    IMAGE_GENERATION = "image_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    VIDEO_ANALYSIS = "video_analysis"
    ANALYSIS = "analysis"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


@dataclass
class Tool:
    """
    Tool definition

    Represents a callable tool with metadata.
    """
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    function: Optional[Callable] = None
    examples: List[str] = field(default_factory=list)

    # Resource requirements
    requires_gpu: bool = False
    estimated_vram_gb: float = 0.0
    estimated_time_seconds: float = 1.0

    # Dependencies
    requires_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum
                }
                for p in self.parameters
            ],
            "examples": self.examples,
            "requires_gpu": self.requires_gpu,
            "estimated_vram_gb": self.estimated_vram_gb,
            "estimated_time_seconds": self.estimated_time_seconds
        }


class ToolRegistry:
    """
    Tool Registry

    Central registry of all available tools.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        logger.info("ToolRegistry initialized")

    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in category"""
        return [tool for tool in self.tools.values() if tool.category == category]

    def get_all_tools(self) -> List[Tool]:
        """Get all tools"""
        return list(self.tools.values())

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools formatted for LLM"""
        return [tool.to_dict() for tool in self.tools.values()]

    def check_gpu_availability(self, tools: List[str]) -> bool:
        """Check if GPU tools can run simultaneously"""
        gpu_tools = [self.tools[name] for name in tools if name in self.tools and self.tools[name].requires_gpu]

        total_vram = sum(tool.estimated_vram_gb for tool in gpu_tools)

        # RTX 5080 has 16GB VRAM
        return total_vram <= 15.5  # Leave some margin

    def suggest_tool_for_task(self, task_description: str) -> List[str]:
        """Suggest tools based on task description (simple heuristic)"""
        task_lower = task_description.lower()
        suggestions = []

        if any(keyword in task_lower for keyword in ["image", "generate", "picture", "visual", "draw"]):
            suggestions.extend([tool.name for tool in self.get_tools_by_category(ToolCategory.IMAGE_GENERATION)])

        if any(keyword in task_lower for keyword in ["voice", "speech", "audio", "say", "speak"]):
            suggestions.extend([tool.name for tool in self.get_tools_by_category(ToolCategory.VOICE_SYNTHESIS)])

        if any(keyword in task_lower for keyword in ["search", "find", "information", "know", "about"]):
            suggestions.extend([tool.name for tool in self.get_tools_by_category(ToolCategory.KNOWLEDGE_RETRIEVAL)])

        return suggestions


def create_default_tool_registry() -> ToolRegistry:
    """Create registry with default tools"""
    registry = ToolRegistry()

    # Image Generation Tools
    registry.register_tool(Tool(
        name="generate_character_image",
        description="Generate an image of a character using SDXL + LoRA",
        category=ToolCategory.IMAGE_GENERATION,
        parameters=[
            ToolParameter("character", "string", "Character name (e.g., 'luca', 'alberto')"),
            ToolParameter("scene_description", "string", "Description of the scene"),
            ToolParameter("style", "string", "Style to use", required=False, default="pixar_3d",
                         enum=["pixar_3d", "italian_summer", "disney", "dreamworks"]),
            ToolParameter("quality_preset", "string", "Quality preset", required=False, default="high",
                         enum=["draft", "standard", "high", "ultra"]),
            ToolParameter("seed", "integer", "Random seed for reproducibility", required=False),
        ],
        examples=[
            "Generate an image of Luca running on the beach",
            "Create a picture of Alberto smiling in Portorosso town square"
        ],
        requires_gpu=True,
        estimated_vram_gb=13.0,
        estimated_time_seconds=15.0
    ))

    registry.register_tool(Tool(
        name="generate_scene_with_controlnet",
        description="Generate image with pose/depth control using ControlNet",
        category=ToolCategory.IMAGE_GENERATION,
        parameters=[
            ToolParameter("character", "string", "Character name"),
            ToolParameter("scene_description", "string", "Scene description"),
            ToolParameter("control_type", "string", "Control type",
                         enum=["pose", "depth", "canny", "normal", "seg"]),
            ToolParameter("control_image_path", "string", "Path to control image"),
            ToolParameter("controlnet_scale", "float", "ControlNet strength", required=False, default=0.9),
        ],
        examples=[
            "Generate Luca in a running pose using pose control"
        ],
        requires_gpu=True,
        estimated_vram_gb=14.5,
        estimated_time_seconds=20.0
    ))

    registry.register_tool(Tool(
        name="batch_generate_character_images",
        description="Generate multiple character images with consistency checking",
        category=ToolCategory.IMAGE_GENERATION,
        parameters=[
            ToolParameter("character", "string", "Character name"),
            ToolParameter("scene_description", "string", "Scene description"),
            ToolParameter("num_images", "integer", "Number of images to generate"),
            ToolParameter("consistency_threshold", "float", "Minimum consistency score",
                         required=False, default=0.70),
        ],
        examples=[
            "Generate 10 consistent images of Luca on the beach"
        ],
        requires_gpu=True,
        estimated_vram_gb=13.0,
        estimated_time_seconds=150.0  # 15s per image * 10
    ))

    # Voice Synthesis Tools
    registry.register_tool(Tool(
        name="synthesize_character_voice",
        description="Synthesize speech in character's voice using GPT-SoVITS",
        category=ToolCategory.VOICE_SYNTHESIS,
        parameters=[
            ToolParameter("character", "string", "Character name (e.g., 'luca', 'alberto')"),
            ToolParameter("text", "string", "Text to synthesize"),
            ToolParameter("emotion", "string", "Emotion/tone", required=False, default="neutral",
                         enum=["neutral", "happy", "sad", "excited", "angry", "surprised", "scared", "calm"]),
            ToolParameter("intensity", "float", "Emotion intensity (0.0-1.0)", required=False, default=0.8),
        ],
        examples=[
            "Synthesize Luca saying 'Silenzio, Bruno!' with excited emotion",
            "Generate Alberto's voice saying 'Welcome to Portorosso!' in a happy tone"
        ],
        requires_gpu=True,
        estimated_vram_gb=3.5,
        estimated_time_seconds=5.0
    ))

    registry.register_tool(Tool(
        name="batch_synthesize_script",
        description="Synthesize multiple lines from a script",
        category=ToolCategory.VOICE_SYNTHESIS,
        parameters=[
            ToolParameter("character", "string", "Character name"),
            ToolParameter("script_lines", "array", "List of text lines to synthesize"),
            ToolParameter("default_emotion", "string", "Default emotion", required=False, default="neutral"),
        ],
        examples=[
            "Synthesize a dialogue script for Luca"
        ],
        requires_gpu=True,
        estimated_vram_gb=3.5,
        estimated_time_seconds=30.0
    ))

    # Knowledge Retrieval Tools
    registry.register_tool(Tool(
        name="search_character_knowledge",
        description="Search knowledge base for character information",
        category=ToolCategory.KNOWLEDGE_RETRIEVAL,
        parameters=[
            ToolParameter("character", "string", "Character name"),
            ToolParameter("aspect", "string", "Specific aspect", required=False,
                         enum=["appearance", "personality", "relationships", "voice", "all"]),
        ],
        examples=[
            "Search for Luca's appearance details",
            "Find information about Alberto's personality"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=0.5
    ))

    registry.register_tool(Tool(
        name="search_style_guide",
        description="Search for style guide information",
        category=ToolCategory.KNOWLEDGE_RETRIEVAL,
        parameters=[
            ToolParameter("style_name", "string", "Style name (e.g., 'pixar_3d', 'italian_summer')"),
        ],
        examples=[
            "Get the Pixar 3D style guide",
            "Find Italian summer coastal style characteristics"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=0.5
    ))

    registry.register_tool(Tool(
        name="search_technical_parameters",
        description="Search for technical generation parameters",
        category=ToolCategory.KNOWLEDGE_RETRIEVAL,
        parameters=[
            ToolParameter("task_type", "string", "Type of task",
                         enum=["image_generation", "voice_synthesis"]),
        ],
        examples=[
            "Get recommended parameters for SDXL image generation"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=0.5
    ))

    registry.register_tool(Tool(
        name="answer_question",
        description="Answer a question using the knowledge base",
        category=ToolCategory.KNOWLEDGE_RETRIEVAL,
        parameters=[
            ToolParameter("question", "string", "Question to answer"),
        ],
        examples=[
            "Who is Luca's best friend?",
            "What is the setting of the film?"
        ],
        requires_gpu=False,
        estimated_vram_gb=0.0,
        estimated_time_seconds=1.0
    ))

    # Video Analysis Tools
    from scripts.agent.tools.video_analysis_tools import register_video_analysis_tools
    register_video_analysis_tools(registry)

    # Video Editing Tools
    from scripts.agent.tools.video_editing_tools import register_video_editing_tools
    register_video_editing_tools(registry)

    logger.info(f"Created default tool registry with {len(registry.tools)} tools")
    return registry


# Global registry instance
_default_registry: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Get the global default registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_tool_registry()
    return _default_registry
