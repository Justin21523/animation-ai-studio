"""
Reasoning Module for Agent Framework

Implements multiple reasoning strategies:
- ReAct (Reason + Act)
- Chain-of-Thought (CoT)
- Tree-of-Thoughts (ToT)
- Reflexion (Execute + Reflect + Refine)

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient
from scripts.agent.core.types import (
    Thought,
    ReasoningTrace,
    ReasoningStrategy,
    Task,
    ToolCall
)


logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Single reasoning step"""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 1.0


class ReActReasoner:
    """
    ReAct (Reason + Act) Reasoning

    Interleaves reasoning and action execution:
    1. Thought: Reasoning about current situation
    2. Action: Execute an action
    3. Observation: Observe result
    4. Repeat until task complete

    Paper: https://arxiv.org/abs/2210.03629
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.max_steps = 10

    async def reason(
        self,
        task: str,
        context: Optional[str] = None,
        available_actions: Optional[List[str]] = None
    ) -> ReasoningTrace:
        """
        Execute ReAct reasoning

        Args:
            task: Task description
            context: Additional context
            available_actions: List of available actions

        Returns:
            ReasoningTrace with complete reasoning
        """
        trace = ReasoningTrace(
            task_description=task,
            strategy=ReasoningStrategy.REACT
        )

        start_time = time.time()

        # Build initial prompt
        prompt = self._build_react_prompt(task, context, available_actions)

        steps: List[ReasoningStep] = []

        for step_num in range(1, self.max_steps + 1):
            # Get reasoning step from LLM
            trace.add_thought(f"ReAct Step {step_num}: Thinking...", "thinking")

            response = await self.llm_client.chat(
                model="qwen-14b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            # Parse response
            reasoning_step = self._parse_react_response(response["content"], step_num)
            steps.append(reasoning_step)

            # Add to trace
            trace.add_thought(
                f"Thought: {reasoning_step.thought}",
                "reasoning",
                reasoning_step.confidence
            )

            if reasoning_step.action:
                trace.add_thought(
                    f"Action: {reasoning_step.action}",
                    "action"
                )

            # Check if task is complete
            if self._is_task_complete(reasoning_step, task):
                trace.add_thought("Task complete!", "observation")
                break

            # Update prompt for next iteration
            prompt = self._update_react_prompt(prompt, reasoning_step)

        trace.success = True
        trace.total_time = time.time() - start_time

        logger.info(f"ReAct reasoning completed in {len(steps)} steps")
        return trace

    def _build_react_prompt(
        self,
        task: str,
        context: Optional[str],
        available_actions: Optional[List[str]]
    ) -> str:
        """Build ReAct prompt"""
        actions_str = "\n".join(f"- {action}" for action in (available_actions or ["search", "generate", "analyze"]))

        prompt = f"""You are solving the following task using ReAct (Reason + Act) approach.

Task: {task}

Context: {context or "None"}

Available Actions:
{actions_str}

Use this format:
Thought: [your reasoning about what to do next]
Action: [the action to take]
Observation: [what you observe after the action]

Let's begin. What is your first thought and action?"""

        return prompt

    def _parse_react_response(self, response: str, step_num: int) -> ReasoningStep:
        """Parse LLM response into ReasoningStep"""
        lines = response.strip().split('\n')

        thought = ""
        action = None
        observation = None

        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Observation:"):
                observation = line.replace("Observation:", "").strip()

        return ReasoningStep(
            step_number=step_num,
            thought=thought or response[:100],
            action=action,
            observation=observation,
            confidence=0.8
        )

    def _is_task_complete(self, step: ReasoningStep, task: str) -> bool:
        """Check if task is complete"""
        # Simple heuristic: check for completion keywords
        completion_keywords = ["complete", "done", "finished", "success", "final answer"]
        thought_lower = step.thought.lower()
        return any(keyword in thought_lower for keyword in completion_keywords)

    def _update_react_prompt(self, original_prompt: str, step: ReasoningStep) -> str:
        """Update prompt with new step"""
        update = f"""
Previous step:
Thought: {step.thought}
Action: {step.action or "None"}
Observation: {step.observation or "Waiting for observation"}

What is your next thought and action?"""

        return original_prompt + update


class ChainOfThoughtReasoner:
    """
    Chain-of-Thought (CoT) Reasoning

    Generates explicit step-by-step reasoning before answering.
    Improves LLM's ability to solve complex problems.

    Paper: https://arxiv.org/abs/2201.11903
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def reason(
        self,
        task: str,
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> ReasoningTrace:
        """
        Execute Chain-of-Thought reasoning

        Args:
            task: Task description
            context: Additional context
            examples: Few-shot examples (optional)

        Returns:
            ReasoningTrace with step-by-step reasoning
        """
        trace = ReasoningTrace(
            task_description=task,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )

        start_time = time.time()

        # Build CoT prompt
        prompt = self._build_cot_prompt(task, context, examples)

        trace.add_thought("Generating step-by-step reasoning...", "thinking")

        # Get reasoning from LLM
        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        # Parse reasoning steps
        steps = self._parse_cot_response(response["content"])

        for i, step in enumerate(steps, 1):
            trace.add_thought(
                f"Step {i}: {step}",
                "reasoning",
                confidence=0.9
            )

        trace.success = True
        trace.total_time = time.time() - start_time

        logger.info(f"CoT reasoning completed with {len(steps)} steps")
        return trace

    def _build_cot_prompt(
        self,
        task: str,
        context: Optional[str],
        examples: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build Chain-of-Thought prompt"""
        prompt = f"""Let's solve this step by step.

Task: {task}

Context: {context or "None"}
"""

        if examples:
            prompt += "\nExamples of step-by-step reasoning:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"\nExample {i}:\n"
                prompt += f"Question: {example.get('question', '')}\n"
                prompt += f"Reasoning: {example.get('reasoning', '')}\n"
                prompt += f"Answer: {example.get('answer', '')}\n"

        prompt += """\nNow, let's solve the task step by step:
Step 1:"""

        return prompt

    def _parse_cot_response(self, response: str) -> List[str]:
        """Parse CoT response into steps"""
        steps = []

        # Split by "Step" markers
        lines = response.split('\n')
        current_step = ""

        for line in lines:
            line = line.strip()
            if line.startswith("Step ") and current_step:
                steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line

        if current_step:
            steps.append(current_step.strip())

        # Remove "Step N:" prefix
        steps = [step.split(":", 1)[-1].strip() for step in steps if step]

        return steps if steps else [response]


class TreeOfThoughtsReasoner:
    """
    Tree-of-Thoughts (ToT) Reasoning

    Explores multiple reasoning paths in parallel,
    evaluates each path, and selects the best one.

    Paper: https://arxiv.org/abs/2305.10601
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.num_branches = 3  # Number of alternative reasoning paths
        self.max_depth = 3  # Maximum reasoning depth

    async def reason(
        self,
        task: str,
        context: Optional[str] = None
    ) -> ReasoningTrace:
        """
        Execute Tree-of-Thoughts reasoning

        Args:
            task: Task description
            context: Additional context

        Returns:
            ReasoningTrace with best reasoning path
        """
        trace = ReasoningTrace(
            task_description=task,
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS
        )

        start_time = time.time()

        trace.add_thought("Exploring multiple reasoning paths...", "thinking")

        # Generate multiple reasoning paths
        paths = await self._generate_reasoning_paths(task, context)

        # Evaluate each path
        trace.add_thought(f"Generated {len(paths)} reasoning paths, evaluating...", "observation")

        best_path = await self._select_best_path(paths, task)

        # Add best path to trace
        for i, step in enumerate(best_path, 1):
            trace.add_thought(
                f"Best path step {i}: {step}",
                "reasoning",
                confidence=0.85
            )

        trace.success = True
        trace.total_time = time.time() - start_time

        logger.info(f"ToT reasoning completed, selected best of {len(paths)} paths")
        return trace

    async def _generate_reasoning_paths(
        self,
        task: str,
        context: Optional[str]
    ) -> List[List[str]]:
        """Generate multiple reasoning paths"""
        paths = []

        for i in range(self.num_branches):
            prompt = f"""Generate a reasoning path for solving this task.

Task: {task}
Context: {context or "None"}

Provide {self.max_depth} reasoning steps (one per line):"""

            response = await self.llm_client.chat(
                model="qwen-14b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8 + (i * 0.1),  # Different temperatures for diversity
                max_tokens=500
            )

            # Parse steps
            steps = [s.strip() for s in response["content"].split('\n') if s.strip()]
            paths.append(steps[:self.max_depth])

        return paths

    async def _select_best_path(
        self,
        paths: List[List[str]],
        task: str
    ) -> List[str]:
        """Select best reasoning path using LLM evaluation"""
        # Build evaluation prompt
        paths_str = "\n\n".join(
            f"Path {i+1}:\n" + "\n".join(f"{j+1}. {step}" for j, step in enumerate(path))
            for i, path in enumerate(paths)
        )

        prompt = f"""Evaluate which reasoning path is best for solving this task.

Task: {task}

{paths_str}

Which path is most likely to solve the task correctly? Respond with just the number (1, 2, or 3):"""

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )

        # Parse selection
        try:
            selected = int(response["content"].strip()) - 1
            if 0 <= selected < len(paths):
                return paths[selected]
        except:
            pass

        # Fallback: return first path
        return paths[0] if paths else []


class ReasoningModule:
    """
    Main Reasoning Module

    Coordinates different reasoning strategies.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize reasoning module

        Args:
            llm_client: LLM client (will create if not provided)
        """
        self._llm_client = llm_client
        self._own_client = llm_client is None

        # Initialize reasoners
        self.react_reasoner: Optional[ReActReasoner] = None
        self.cot_reasoner: Optional[ChainOfThoughtReasoner] = None
        self.tot_reasoner: Optional[TreeOfThoughtsReasoner] = None

        logger.info("ReasoningModule initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_client:
            self._llm_client = LLMClient()
            await self._llm_client.__aenter__()

        # Initialize reasoners
        self.react_reasoner = ReActReasoner(self._llm_client)
        self.cot_reasoner = ChainOfThoughtReasoner(self._llm_client)
        self.tot_reasoner = TreeOfThoughtsReasoner(self._llm_client)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._own_client and self._llm_client:
            await self._llm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def reason(
        self,
        task: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        context: Optional[str] = None,
        **kwargs
    ) -> ReasoningTrace:
        """
        Execute reasoning with specified strategy

        Args:
            task: Task description
            strategy: Reasoning strategy to use
            context: Additional context
            **kwargs: Strategy-specific arguments

        Returns:
            ReasoningTrace with reasoning process
        """
        logger.info(f"Starting {strategy.value} reasoning for: {task[:50]}...")

        if strategy == ReasoningStrategy.REACT:
            return await self.react_reasoner.reason(
                task,
                context,
                kwargs.get("available_actions")
            )

        elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            return await self.cot_reasoner.reason(
                task,
                context,
                kwargs.get("examples")
            )

        elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
            return await self.tot_reasoner.reason(task, context)

        else:
            # Fallback to CoT
            logger.warning(f"Unknown strategy {strategy}, falling back to CoT")
            return await self.cot_reasoner.reason(task, context)


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async with ReasoningModule() as reasoning:
        # Example 1: Chain-of-Thought
        print("\n" + "=" * 60)
        print("Example 1: Chain-of-Thought")
        print("=" * 60)

        trace = await reasoning.reason(
            task="Generate an image of Luca running on the beach",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context="Character: Luca, Style: Pixar 3D"
        )

        print(f"\nReasoning steps ({len(trace.thoughts)}):")
        for thought in trace.thoughts:
            print(f"  [{thought.thought_type}] {thought.content}")

        # Example 2: ReAct
        print("\n" + "=" * 60)
        print("Example 2: ReAct")
        print("=" * 60)

        trace = await reasoning.reason(
            task="Find information about Alberto's personality",
            strategy=ReasoningStrategy.REACT,
            available_actions=["search_knowledge_base", "ask_question", "complete"]
        )

        print(f"\nReAct steps ({len(trace.thoughts)}):")
        for thought in trace.thoughts:
            print(f"  {thought.content}")

        # Example 3: Tree-of-Thoughts
        print("\n" + "=" * 60)
        print("Example 3: Tree-of-Thoughts")
        print("=" * 60)

        trace = await reasoning.reason(
            task="Create a creative scene combining Luca and Alberto",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS
        )

        print(f"\nToT best path ({len(trace.thoughts)}):")
        for thought in trace.thoughts:
            print(f"  {thought.content}")


if __name__ == "__main__":
    asyncio.run(main())
