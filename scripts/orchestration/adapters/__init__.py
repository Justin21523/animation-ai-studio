"""
Orchestration Module Adapters

Adapters for integrating existing modules with the orchestration layer.
"""

from .agent_adapter import AgentAdapter
from .rag_adapter import RAGAdapter
from .scenario_adapter import ScenarioAdapter

__all__ = [
    "AgentAdapter",
    "RAGAdapter",
    "ScenarioAdapter"
]
