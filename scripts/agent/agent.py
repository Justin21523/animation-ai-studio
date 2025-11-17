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
    default_strategy: ReasoningStrategy = ReasoningStrategy.DIRECT
    enable_reflection: bool = True

    # Quality settings
    quality_threshold: float = 0.7
    max_iterations: int = 3


class Agent:
    """
    Main Agent Orchestrator

    Coordinates all sub-modules to:
    1. Understand user intent (Thinking Module)
    2. Retrieve relevant knowledge (RAG Module)
    3. Plan execution
    4. Execute tasks
    5. Reflect and iterate

    This is Phase 1 implementation with core functionality.
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

        # Components
        self._llm_client = llm_client
        self._knowledge_base = knowledge_base
        self.thinking_module: Optional[ThinkingModule] = None
        self.rag_module: Optional[RAGUsageModule] = None

        self._own_llm = llm_client is None
        self._own_kb = knowledge_base is None

        logger.info("Agent initialized")

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

        # Initialize modules
        self.thinking_module = ThinkingModule(llm_client=self._llm_client)
        await self.thinking_module.__aenter__()

        if self.config.enable_rag:
            self.rag_module = RAGUsageModule(knowledge_base=self._knowledge_base)
            await self.rag_module.__aenter__()

        logger.info("Agent fully initialized and ready")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.thinking_module:
            await self.thinking_module.__aexit__(exc_type, exc_val, exc_tb)

        if self.rag_module:
            await self.rag_module.__aexit__(exc_type, exc_val, exc_tb)

        if self._own_kb and self._knowledge_base:
            await self._knowledge_base.__aexit__(exc_type, exc_val, exc_tb)

        if self._own_llm and self._llm_client:
            await self._llm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def process(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process user request

        Main entry point for agent execution.

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
