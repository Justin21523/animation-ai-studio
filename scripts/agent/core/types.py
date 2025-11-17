"""
Core Types and Data Structures for Agent Framework

Defines fundamental types used across all agent modules.

Author: Animation AI Studio
Date: 2025-11-17
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime


class TaskType(Enum):
    """Types of tasks the agent can handle"""
    IMAGE_GENERATION = "image_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    CHARACTER_CREATION = "character_creation"
    SCENE_GENERATION = "scene_generation"
    STYLE_TRANSFER = "style_transfer"
    QUESTION_ANSWERING = "question_answering"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    MIXED = "mixed"


class ReasoningStrategy(Enum):
    """Reasoning strategies"""
    REACT = "react"  # Reason + Act
    CHAIN_OF_THOUGHT = "chain_of_thought"  # CoT
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # ToT
    REFLEXION = "reflexion"  # Execute + Reflect + Refine
    DIRECT = "direct"  # Direct execution without reasoning


class StepStatus(Enum):
    """Status of workflow step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Thought:
    """
    Single reasoning thought

    Represents one step in the agent's thinking process.
    """
    content: str
    thought_type: str  # observation, reasoning, action, reflection
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.thought_type}] {self.content}"


@dataclass
class Task:
    """
    Represents a task to be executed

    Tasks can be decomposed into subtasks.
    """
    task_id: str
    description: str
    task_type: TaskType
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def mark_completed(self, result: Any):
        """Mark task as completed"""
        self.status = StepStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now().isoformat()

    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.now().isoformat()


@dataclass
class ToolCall:
    """
    Record of a tool being called

    Tracks tool execution for transparency and debugging.
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


@dataclass
class FunctionCall:
    """
    Record of a function being called

    Similar to ToolCall but for type-safe function calling.
    """
    function_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "arguments": self.arguments,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


@dataclass
class ReasoningTrace:
    """
    Complete reasoning trace

    Records the entire reasoning process for transparency.
    """
    task_description: str
    strategy: ReasoningStrategy
    thoughts: List[Thought] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    function_calls: List[FunctionCall] = field(default_factory=list)
    final_result: Optional[Any] = None
    success: bool = False
    total_time: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_thought(self, content: str, thought_type: str, confidence: float = 1.0):
        """Add a thought to the trace"""
        self.thoughts.append(Thought(
            content=content,
            thought_type=thought_type,
            confidence=confidence
        ))

    def add_tool_call(self, tool_call: ToolCall):
        """Add a tool call to the trace"""
        self.tool_calls.append(tool_call)

    def add_function_call(self, function_call: FunctionCall):
        """Add a function call to the trace"""
        self.function_calls.append(function_call)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_description": self.task_description,
            "strategy": self.strategy.value,
            "thoughts": [str(t) for t in self.thoughts],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "function_calls": [fc.to_dict() for fc in self.function_calls],
            "final_result": str(self.final_result)[:500] if self.final_result else None,
            "success": self.success,
            "total_time": self.total_time,
            "created_at": self.created_at
        }


@dataclass
class WorkflowStep:
    """
    Single step in a workflow

    Part of a multi-step execution plan.
    """
    step_id: str
    description: str
    action: str  # What to do
    dependencies: List[str] = field(default_factory=list)  # Step IDs
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    reasoning_trace: Optional[ReasoningTrace] = None


@dataclass
class Workflow:
    """
    Complete multi-step workflow

    Represents a complex task broken down into steps.
    """
    workflow_id: str
    description: str
    steps: List[WorkflowStep] = field(default_factory=list)
    current_step: int = 0
    status: StepStatus = StepStatus.PENDING
    final_result: Optional[Any] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def get_next_step(self) -> Optional[WorkflowStep]:
        """Get next executable step"""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # Check if dependencies are met
                deps_met = all(
                    any(s.step_id == dep_id and s.status == StepStatus.COMPLETED
                        for s in self.steps)
                    for dep_id in step.dependencies
                ) if step.dependencies else True

                if deps_met:
                    return step
        return None

    def is_complete(self) -> bool:
        """Check if workflow is complete"""
        return all(
            s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
            for s in self.steps
        )


@dataclass
class AgentState:
    """
    Current state of the agent

    Maintains context across multiple interactions.
    """
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_task: Optional[Task] = None
    active_workflow: Optional[Workflow] = None
    knowledge_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


@dataclass
class AgentResponse:
    """
    Agent's response to a request

    Contains the result and all metadata about the execution.
    """
    content: str  # Main response
    success: bool
    reasoning_trace: Optional[ReasoningTrace] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    function_calls: List[FunctionCall] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "success": self.success,
            "reasoning_trace": self.reasoning_trace.to_dict() if self.reasoning_trace else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "function_calls": [fc.to_dict() for fc in self.function_calls],
            "confidence": self.confidence,
            "metadata": self.metadata
        }
