"""
Agent Framework for Animation AI Studio

LLM-powered autonomous agent for creative decision-making.

Phase 1 Implementation:
- Thinking Module (intent understanding, task decomposition)
- RAG Usage Module (knowledge retrieval)
- Agent Orchestrator (core workflow)

Future Phases:
- Reasoning Module (ReAct, CoT, ToT)
- Tool Calling Module
- Function Calling Module
- Multi-Step Reasoning Module

Author: Animation AI Studio
Date: 2025-11-17
"""

from scripts.agent.agent import Agent, AgentConfig
from scripts.agent.core.types import (
    Task,
    TaskType,
    Thought,
    AgentState,
    AgentResponse,
    ReasoningTrace,
    ReasoningStrategy,
    StepStatus,
    ToolCall,
    FunctionCall,
    Workflow,
    WorkflowStep
)
from scripts.agent.thinking.thinking_module import (
    ThinkingModule,
    IntentAnalysis,
    TaskDecomposition
)
from scripts.agent.rag_usage.rag_module import (
    RAGUsageModule,
    RAGRetrievalResult
)

__all__ = [
    # Main agent
    "Agent",
    "AgentConfig",

    # Core types
    "Task",
    "TaskType",
    "Thought",
    "AgentState",
    "AgentResponse",
    "ReasoningTrace",
    "ReasoningStrategy",
    "StepStatus",
    "ToolCall",
    "FunctionCall",
    "Workflow",
    "WorkflowStep",

    # Thinking module
    "ThinkingModule",
    "IntentAnalysis",
    "TaskDecomposition",

    # RAG module
    "RAGUsageModule",
    "RAGRetrievalResult",
]
