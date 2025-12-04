"""
Agent Framework Adapter

Adapter for integrating the Agent Framework with the orchestration layer.
Wraps the existing Agent module to provide standardized Task/TaskResult interface.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.orchestration.module_registry import (
    ModuleAdapter,
    ModuleType,
    ModuleStatus,
    ModuleCapabilities,
    HealthCheckResult,
    Task,
    TaskResult
)
from scripts.agent.agent import Agent, AgentConfig
from scripts.agent.core.types import ReasoningStrategy
from scripts.core.llm_client import LLMClient
from scripts.rag import KnowledgeBase

logger = logging.getLogger(__name__)


class AgentAdapter(ModuleAdapter):
    """
    Adapter for Agent Framework (6,681 LOC)

    Wraps the existing Agent system to provide standardized interface for:
    - Natural language understanding
    - RAG-based knowledge retrieval
    - Multi-step reasoning (CoT, ReAct, ToT)
    - Tool/function calling
    - Web search integration

    Task Types Supported:
    - "analyze": Analyze and understand content
    - "question": Answer questions using RAG
    - "creative": Creative generation tasks
    - "complex": Multi-step complex workflows
    - "search": Web search and information retrieval

    Task Parameters:
    - user_request (str): Natural language request
    - context (dict, optional): Additional context
    - strategy (str, optional): Reasoning strategy (cot, react, tot, auto)
    - enable_tools (bool, optional): Enable tool/function calling
    - enable_web_search (bool, optional): Enable web search
    - enable_rag (bool, optional): Enable RAG retrieval

    Example:
        adapter = AgentAdapter(config=AgentConfig(
            llm_model="qwen-14b",
            enable_rag=True,
            enable_web_search=True
        ))

        await adapter.initialize()

        task = Task(
            task_id="1",
            task_type="analyze",
            parameters={
                "user_request": "Tell me about Luca's appearance",
                "context": {"character": "luca"}
            }
        )

        result = await adapter.execute(task)
        print(result.output["response"])
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        Initialize Agent Adapter

        Args:
            config: Agent configuration (will use defaults if not provided)
            llm_client: Optional shared LLM client
            knowledge_base: Optional shared knowledge base
        """
        super().__init__(
            module_name="agent",
            module_type=ModuleType.AGENT
        )

        self.config = config or AgentConfig()
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.agent: Optional[Agent] = None

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_time = 0.0

        logger.info("AgentAdapter created")

    async def initialize(self) -> bool:
        """
        Initialize Agent Framework

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Agent Framework...")

            # Create agent instance
            self.agent = Agent(
                config=self.config,
                llm_client=self.llm_client,
                knowledge_base=self.knowledge_base
            )

            # Initialize agent (async context manager)
            await self.agent.__aenter__()

            self._initialized = True
            logger.info("Agent Framework initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Agent Framework: {e}", exc_info=True)
            return False

    async def execute(self, task: Task) -> TaskResult:
        """
        Execute task using Agent Framework

        Args:
            task: Task to execute

        Returns:
            TaskResult with agent response
        """
        if not self._initialized:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error="Agent not initialized"
            )

        start_time = time.time()
        self.total_tasks += 1

        try:
            # Extract parameters
            user_request = task.parameters.get("user_request")
            if not user_request:
                raise ValueError("Missing required parameter: user_request")

            context = task.parameters.get("context")
            strategy_name = task.parameters.get("strategy", "auto")
            enable_tools = task.parameters.get("enable_tools", True)
            enable_web_search = task.parameters.get("enable_web_search", self.config.enable_web_search)
            enable_rag = task.parameters.get("enable_rag", self.config.enable_rag)

            # Map strategy name to enum
            strategy_map = {
                "cot": ReasoningStrategy.CHAIN_OF_THOUGHT,
                "react": ReasoningStrategy.REACT,
                "tot": ReasoningStrategy.TREE_OF_THOUGHTS,
                "auto": None  # Auto-select
            }
            strategy = strategy_map.get(strategy_name)

            # Temporarily override config if needed
            original_web_search = self.config.enable_web_search
            original_rag = self.config.enable_rag
            self.config.enable_web_search = enable_web_search
            self.config.enable_rag = enable_rag

            try:
                # Choose processing mode based on task type
                if task.task_type in ["complex", "creative"] or enable_tools:
                    # Advanced processing with tools
                    agent_response = await self.agent.process_advanced(
                        user_request=user_request,
                        context=context,
                        strategy=strategy,
                        enable_tools=enable_tools
                    )
                else:
                    # Basic processing
                    agent_response = await self.agent.process(
                        user_request=user_request,
                        context=context
                    )
            finally:
                # Restore original config
                self.config.enable_web_search = original_web_search
                self.config.enable_rag = original_rag

            # Convert agent response to TaskResult
            execution_time = time.time() - start_time
            self.total_time += execution_time

            if agent_response.success:
                self.successful_tasks += 1
            else:
                self.failed_tasks += 1

            return TaskResult(
                task_id=task.task_id,
                success=agent_response.success,
                output={
                    "response": agent_response.content,
                    "confidence": agent_response.confidence,
                    "reasoning_trace": {
                        "thoughts": [
                            {
                                "content": thought.content,
                                "type": thought.thought_type,
                                "confidence": thought.confidence
                            }
                            for thought in agent_response.reasoning_trace.thoughts
                        ],
                        "strategy": agent_response.reasoning_trace.strategy.value,
                        "total_time": agent_response.reasoning_trace.total_time
                    }
                },
                error=None if agent_response.success else agent_response.content,
                execution_time=execution_time,
                metadata={
                    "task_type": task.task_type,
                    "strategy": strategy_name,
                    "enable_tools": enable_tools,
                    "enable_web_search": enable_web_search,
                    "enable_rag": enable_rag
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.total_time += execution_time
            self.failed_tasks += 1

            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def health_check(self) -> HealthCheckResult:
        """
        Check Agent Framework health

        Returns:
            HealthCheckResult with status and details
        """
        try:
            if not self._initialized:
                return HealthCheckResult(
                    module_name=self.module_name,
                    status=ModuleStatus.UNHEALTHY,
                    message="Agent not initialized"
                )

            # Check agent is available
            if not self.agent:
                return HealthCheckResult(
                    module_name=self.module_name,
                    status=ModuleStatus.UNHEALTHY,
                    message="Agent instance not available"
                )

            # Check sub-modules
            issues = []

            if not self.agent.thinking_module:
                issues.append("Thinking module not initialized")

            if self.config.enable_rag and not self.agent.rag_module:
                issues.append("RAG module not initialized")

            if self.config.enable_tool_calling and not self.agent.tool_calling_module:
                issues.append("Tool calling module not initialized")

            # Determine status
            if issues:
                status = ModuleStatus.DEGRADED
                message = f"Agent partially functional: {', '.join(issues)}"
            else:
                status = ModuleStatus.HEALTHY
                message = "Agent fully functional"

            # Calculate success rate
            success_rate = (
                self.successful_tasks / self.total_tasks * 100
                if self.total_tasks > 0 else 100.0
            )

            avg_time = (
                self.total_time / self.total_tasks
                if self.total_tasks > 0 else 0.0
            )

            return HealthCheckResult(
                module_name=self.module_name,
                status=status,
                message=message,
                details={
                    "total_tasks": self.total_tasks,
                    "successful_tasks": self.successful_tasks,
                    "failed_tasks": self.failed_tasks,
                    "success_rate": f"{success_rate:.1f}%",
                    "avg_execution_time": f"{avg_time:.2f}s",
                    "modules": {
                        "thinking": self.agent.thinking_module is not None,
                        "rag": self.agent.rag_module is not None,
                        "reasoning": self.agent.reasoning_module is not None,
                        "tool_calling": self.agent.tool_calling_module is not None,
                        "function_calling": self.agent.function_calling_module is not None,
                        "multi_step": self.agent.multi_step_module is not None,
                        "web_search": self.agent.web_search_module is not None
                    }
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return HealthCheckResult(
                module_name=self.module_name,
                status=ModuleStatus.UNKNOWN,
                message=f"Health check error: {str(e)}"
            )

    def get_capabilities(self) -> ModuleCapabilities:
        """
        Get Agent Framework capabilities

        Returns:
            ModuleCapabilities describing agent features
        """
        return ModuleCapabilities(
            module_name=self.module_name,
            module_type=self.module_type,
            supported_operations=[
                "analyze",          # Content analysis
                "question",         # Question answering
                "creative",         # Creative generation
                "complex",          # Multi-step workflows
                "search",           # Web search
                "character_query",  # Character information
                "style_query",      # Style guide queries
                "technical_query"   # Technical parameters
            ],
            requires_gpu=False,  # CPU-only LLM inference
            max_concurrent_tasks=3,  # Limit concurrent agent tasks
            estimated_memory_mb=2048,  # ~2GB for LLM + embeddings
            description=(
                "Agent Framework for natural language understanding, reasoning, "
                "and task execution. Supports RAG, multi-step workflows, "
                "tool/function calling, and web search. CPU-only operation."
            )
        )

    async def cleanup(self):
        """Cleanup Agent Framework resources"""
        try:
            if self.agent:
                await self.agent.__aexit__(None, None, None)
                logger.info("Agent Framework cleaned up")
        except Exception as e:
            logger.error(f"Error during Agent cleanup: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get adapter statistics

        Returns:
            Statistics dictionary
        """
        success_rate = (
            self.successful_tasks / self.total_tasks * 100
            if self.total_tasks > 0 else 0.0
        )

        avg_time = (
            self.total_time / self.total_tasks
            if self.total_tasks > 0 else 0.0
        )

        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": success_rate,
            "total_time": self.total_time,
            "avg_execution_time": avg_time
        }

    def __repr__(self) -> str:
        return (
            f"AgentAdapter(initialized={self._initialized}, "
            f"tasks={self.total_tasks}, "
            f"success_rate={self.successful_tasks}/{self.total_tasks})"
        )
