"""
Main Agent Orchestrator

Coordinates all agent modules to execute complex tasks.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient
from scripts.rag import KnowledgeBase, KnowledgeBaseConfig
from scripts.agent.core.types import (
    Task,
    TaskType,
    Thought,
    AgentState,
    AgentResponse,
    ReasoningTrace,
    ReasoningStrategy,
    StepStatus
)
from scripts.agent.thinking.thinking_module import ThinkingModule
from scripts.agent.rag_usage.rag_module import RAGUsageModule
from scripts.agent.reasoning.reasoning_module import ReasoningModule
from scripts.agent.tools.tool_calling_module import ToolCallingModule
from scripts.agent.functions.function_calling_module import FunctionCallingModule
from scripts.agent.multi_step.multi_step_module import MultiStepModule


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agent"""
    # LLM settings
    llm_model: str = "qwen-14b"
    temperature: float = 0.7
    max_tokens: int = 2000

    # RAG settings
    enable_rag: bool = True
    rag_top_k: int = 5

    # Reasoning settings
    default_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    enable_reflection: bool = True

    # Phase 2 settings
    enable_tool_calling: bool = True
    enable_function_calling: bool = True
    enable_multi_step: bool = True

    # Quality settings
    quality_threshold: float = 0.7
    max_iterations: int = 3
    max_retries: int = 3


class Agent:
    """
    Main Agent Orchestrator (Phase 1 + Phase 2 Complete)

    Coordinates all sub-modules to:
    1. Understand user intent (Thinking Module)
    2. Retrieve relevant knowledge (RAG Module)
    3. Advanced reasoning (Reasoning Module - ReAct, CoT, ToT)
    4. Tool/function calling (Tool Calling, Function Calling)
    5. Multi-step execution (Multi-Step Module)
    6. Reflect and iterate

    Phase 1 (Core Infrastructure):
    - Thinking Module: Intent understanding, task decomposition, reflection
    - RAG Usage Module: Knowledge retrieval and context management

    Phase 2 (Advanced Reasoning & Execution):
    - Reasoning Module: ReAct, Chain-of-Thought, Tree-of-Thoughts
    - Tool Calling Module: LLM-powered tool selection and execution
    - Function Calling Module: Type-safe function calling with auto-schema
    - Multi-Step Module: Stateful workflow execution with quality checks
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        Initialize agent

        Args:
            config: Agent configuration
            llm_client: LLM client (will create if not provided)
            knowledge_base: Knowledge base (will create if not provided)
        """
        self.config = config or AgentConfig()
        self.state = AgentState()

        # Core components
        self._llm_client = llm_client
        self._knowledge_base = knowledge_base

        # Phase 1 modules
        self.thinking_module: Optional[ThinkingModule] = None
        self.rag_module: Optional[RAGUsageModule] = None

        # Phase 2 modules
        self.reasoning_module: Optional[ReasoningModule] = None
        self.tool_calling_module: Optional[ToolCallingModule] = None
        self.function_calling_module: Optional[FunctionCallingModule] = None
        self.multi_step_module: Optional[MultiStepModule] = None

        self._own_llm = llm_client is None
        self._own_kb = knowledge_base is None

        logger.info("Agent initialized (Phase 1 + Phase 2)")

    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize LLM client
        if self._own_llm:
            self._llm_client = LLMClient()
            await self._llm_client.__aenter__()

        # Initialize knowledge base
        if self._own_kb and self.config.enable_rag:
            self._knowledge_base = KnowledgeBase()
            await self._knowledge_base.__aenter__()

        # Initialize Phase 1 modules
        self.thinking_module = ThinkingModule(llm_client=self._llm_client)
        await self.thinking_module.__aenter__()

        if self.config.enable_rag:
            self.rag_module = RAGUsageModule(knowledge_base=self._knowledge_base)
            await self.rag_module.__aenter__()

        # Initialize Phase 2 modules
        self.reasoning_module = ReasoningModule(llm_client=self._llm_client)
        await self.reasoning_module.__aenter__()

        if self.config.enable_tool_calling:
            self.tool_calling_module = ToolCallingModule(llm_client=self._llm_client)
            await self.tool_calling_module.__aenter__()

        if self.config.enable_function_calling:
            self.function_calling_module = FunctionCallingModule(llm_client=self._llm_client)
            await self.function_calling_module.__aenter__()

        if self.config.enable_multi_step:
            self.multi_step_module = MultiStepModule(
                llm_client=self._llm_client,
                quality_threshold=self.config.quality_threshold,
                max_retries=self.config.max_retries
            )
            await self.multi_step_module.__aenter__()

        logger.info("Agent fully initialized and ready (Phase 1 + Phase 2)")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Close Phase 1 modules
        if self.thinking_module:
            await self.thinking_module.__aexit__(exc_type, exc_val, exc_tb)

        if self.rag_module:
            await self.rag_module.__aexit__(exc_type, exc_val, exc_tb)

        # Close Phase 2 modules
        if self.reasoning_module:
            await self.reasoning_module.__aexit__(exc_type, exc_val, exc_tb)

        if self.tool_calling_module:
            await self.tool_calling_module.__aexit__(exc_type, exc_val, exc_tb)

        if self.function_calling_module:
            await self.function_calling_module.__aexit__(exc_type, exc_val, exc_tb)

        if self.multi_step_module:
            await self.multi_step_module.__aexit__(exc_type, exc_val, exc_tb)

        # Close shared resources
        if self._own_kb and self._knowledge_base:
            await self._knowledge_base.__aexit__(exc_type, exc_val, exc_tb)

        if self._own_llm and self._llm_client:
            await self._llm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def process_advanced(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[ReasoningStrategy] = None,
        enable_tools: bool = True
    ) -> AgentResponse:
        """
        Process user request with Phase 2 advanced features

        Uses advanced reasoning, tool calling, and multi-step execution.

        Args:
            user_request: User's request
            context: Additional context
            strategy: Reasoning strategy to use (None = auto-select)
            enable_tools: Whether to use tool/function calling

        Returns:
            AgentResponse with result
        """
        start_time = time.time()

        # Select strategy if not provided
        if strategy is None:
            strategy = self.config.default_strategy

        trace = ReasoningTrace(
            task_description=user_request,
            strategy=strategy
        )

        try:
            # Add to conversation history
            self.state.add_message("user", user_request)

            # Step 1: Understand intent
            trace.add_thought("Analyzing user request with advanced reasoning...", "thinking")
            intent = await self.thinking_module.understand_intent(user_request, context)

            # Step 2: Retrieve knowledge
            knowledge_context = ""
            if self.config.enable_rag and self.rag_module:
                rag_result = await self.rag_module.retrieve_context_for_task(user_request)
                knowledge_context = rag_result.context

            # Step 3: Reasoning with selected strategy
            trace.add_thought(f"Applying {strategy.value} reasoning...", "action")
            reasoning_trace = await self.reasoning_module.reason(
                task=user_request,
                strategy=strategy,
                context=knowledge_context
            )

            # Merge reasoning thoughts into main trace
            for thought in reasoning_trace.thoughts:
                trace.thoughts.append(thought)

            # Step 4: Multi-step execution (for complex tasks)
            if self.config.enable_multi_step and intent.task_type.value in ["creative", "complex"]:
                trace.add_thought("Creating multi-step workflow plan...", "action")

                plan = await self.multi_step_module.create_workflow_plan(
                    task=user_request,
                    context=knowledge_context
                )

                trace.add_thought(f"Executing workflow with {len(plan.steps)} steps...", "action")

                workflow_trace = await self.multi_step_module.execute_workflow(plan)

                # Merge workflow thoughts
                for thought in workflow_trace.thoughts:
                    trace.thoughts.append(thought)

                result_content = workflow_trace.final_result or "Workflow completed"
            else:
                # Simple response generation
                result_content = await self._generate_response(
                    user_request, intent, knowledge_context, trace
                )

            # Step 5: Reflection
            if self.config.enable_reflection:
                reflection = await self.thinking_module.reflect_on_result(
                    task=Task(
                        task_id="main",
                        description=user_request,
                        task_type=intent.task_type
                    ),
                    result=result_content
                )
                trace.add_thought(reflection.content, reflection.thought_type)

            trace.final_result = result_content
            trace.success = True
            trace.total_time = time.time() - start_time

            self.state.add_message("assistant", str(result_content))

            logger.info(f"Advanced processing completed in {trace.total_time:.2f}s")

            return AgentResponse(
                content=str(result_content),
                success=True,
                reasoning_trace=trace,
                confidence=intent.confidence
            )

        except Exception as e:
            logger.error(f"Error in advanced processing: {e}")
            trace.success = False
            trace.total_time = time.time() - start_time

            return AgentResponse(
                content=f"Error: {str(e)}",
                success=False,
                reasoning_trace=trace,
                confidence=0.0
            )

    async def process(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process user request (Phase 1 compatible interface)

        Main entry point for agent execution.
        Uses basic processing without advanced Phase 2 features.

        Args:
            user_request: User's request
            context: Additional context

        Returns:
            AgentResponse with result
        """
        start_time = time.time()

        # Create reasoning trace
        trace = ReasoningTrace(
            task_description=user_request,
            strategy=self.config.default_strategy
        )

        try:
            # Add to conversation history
            self.state.add_message("user", user_request)

            # Step 1: Understand intent
            trace.add_thought("Analyzing user request...", "thinking")
            intent = await self.thinking_module.understand_intent(
                user_request,
                context
            )

            trace.add_thought(
                f"Identified intent: {intent.primary_intent} (type: {intent.task_type.value})",
                "observation",
                intent.confidence
            )

            # Step 2: Retrieve relevant knowledge (if RAG enabled)
            knowledge_context = ""
            if self.config.enable_rag and self.rag_module:
                trace.add_thought("Retrieving relevant knowledge...", "action")

                rag_result = await self.rag_module.retrieve_context_for_task(
                    user_request
                )

                knowledge_context = rag_result.context

                retrieval_thought = self.rag_module.create_retrieval_thought(rag_result)
                trace.add_thought(
                    retrieval_thought.content,
                    retrieval_thought.thought_type,
                    retrieval_thought.confidence
                )

            # Step 3: Generate response
            trace.add_thought("Generating response...", "action")

            response_content = await self._generate_response(
                user_request=user_request,
                intent=intent,
                knowledge_context=knowledge_context,
                trace=trace
            )

            # Step 4: Reflect (if enabled)
            if self.config.enable_reflection:
                trace.add_thought("Reflecting on response...", "thinking")

                reflection = await self.thinking_module.reflect_on_result(
                    task=Task(
                        task_id="main",
                        description=user_request,
                        task_type=intent.task_type
                    ),
                    result=response_content
                )

                trace.add_thought(
                    reflection.content,
                    reflection.thought_type,
                    reflection.confidence
                )

            # Finalize trace
            trace.final_result = response_content
            trace.success = True
            trace.total_time = time.time() - start_time

            # Add to conversation history
            self.state.add_message("assistant", response_content)

            logger.info(f"Request processed successfully in {trace.total_time:.2f}s")

            return AgentResponse(
                content=response_content,
                success=True,
                reasoning_trace=trace,
                confidence=intent.confidence
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}")

            trace.success = False
            trace.total_time = time.time() - start_time

            error_message = f"I encountered an error while processing your request: {str(e)}"

            return AgentResponse(
                content=error_message,
                success=False,
                reasoning_trace=trace,
                confidence=0.0
            )

    async def _generate_response(
        self,
        user_request: str,
        intent: Any,
        knowledge_context: str,
        trace: ReasoningTrace
    ) -> str:
        """
        Generate response using LLM

        Args:
            user_request: Original user request
            intent: Intent analysis result
            knowledge_context: Retrieved knowledge
            trace: Reasoning trace

        Returns:
            Response string
        """
        # Build prompt with context
        system_prompt = """You are an advanced AI assistant for animation content creation.

You have access to:
- Character knowledge (appearance, personality, relationships)
- Style guides (Pixar 3D, Italian Summer, etc.)
- Technical parameters (SDXL, GPT-SoVITS settings)

Provide helpful, accurate responses based on the provided context."""

        user_prompt = f"""User request: {user_request}

Intent: {intent.primary_intent}
Task type: {intent.task_type.value}

"""

        if knowledge_context:
            user_prompt += f"""Relevant knowledge:
{knowledge_context}

"""

        user_prompt += """Please provide a helpful response."""

        # Query LLM
        response = await self._llm_client.chat(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response["content"]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.state.conversation_history

    def clear_conversation(self):
        """Clear conversation history"""
        self.state.clear_history()
        logger.info("Conversation history cleared")


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create agent
    async with Agent() as agent:
        # Example 1: Simple question
        print("\n" + "=" * 60)
        print("Example 1: Character Question")
        print("=" * 60)

        response = await agent.process(
            "Tell me about Luca's appearance and personality"
        )

        print(f"\nResponse: {response.content}")
        print(f"Success: {response.success}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"\nReasoning trace:")
        for thought in response.reasoning_trace.thoughts:
            print(f"  [{thought.thought_type}] {thought.content}")

        # Example 2: Creative request
        print("\n" + "=" * 60)
        print("Example 2: Creative Request")
        print("=" * 60)

        response = await agent.process(
            "I want to generate an image of Luca running on the beach"
        )

        print(f"\nResponse: {response.content}")
        print(f"Reasoning steps: {len(response.reasoning_trace.thoughts)}")

        # Example 3: Conversation
        print("\n" + "=" * 60)
        print("Example 3: Conversation")
        print("=" * 60)

        response = await agent.process("Who is Alberto?")
        print(f"\nQ: Who is Alberto?")
        print(f"A: {response.content[:200]}...")

        response = await agent.process("What's his relationship with Luca?")
        print(f"\nQ: What's his relationship with Luca?")
        print(f"A: {response.content[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
