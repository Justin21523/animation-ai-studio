"""
Thinking Module for Agent Framework

Handles intent understanding, task decomposition, and reflection.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient
from scripts.agent.core.types import (
    Task,
    TaskType,
    Thought,
    StepStatus
)


logger = logging.getLogger(__name__)


@dataclass
class IntentAnalysis:
    """Result of intent analysis"""
    primary_intent: str
    task_type: TaskType
    entities: Dict[str, Any]  # Extracted entities (character, style, etc.)
    constraints: List[str]  # User constraints
    confidence: float


@dataclass
class TaskDecomposition:
    """Result of task decomposition"""
    main_task: Task
    subtasks: List[Task]
    execution_order: List[str]  # Task IDs in order
    estimated_time: float  # seconds


class ThinkingModule:
    """
    Thinking Module

    Responsible for:
    - Understanding user intent
    - Decomposing complex tasks
    - Reflecting on progress
    - Generating explanations

    Uses LLM for cognitive operations.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize thinking module

        Args:
            llm_client: LLM client (will create if not provided)
        """
        self._llm_client = llm_client
        self._own_client = llm_client is None
        logger.info("ThinkingModule initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_client:
            self._llm_client = LLMClient()
            await self._llm_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._own_client and self._llm_client:
            await self._llm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def understand_intent(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysis:
        """
        Understand user's intent

        Args:
            user_request: User's request
            context: Additional context (conversation history, etc.)

        Returns:
            IntentAnalysis with extracted information
        """
        # Build prompt for intent analysis
        prompt = self._build_intent_prompt(user_request, context)

        # Query LLM
        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        # Parse response
        intent_analysis = self._parse_intent_response(response["content"])

        logger.info(f"Intent: {intent_analysis.primary_intent}, Type: {intent_analysis.task_type.value}")
        return intent_analysis

    async def decompose_task(
        self,
        task_description: str,
        intent: Optional[IntentAnalysis] = None
    ) -> TaskDecomposition:
        """
        Decompose complex task into subtasks

        Args:
            task_description: Description of the task
            intent: Intent analysis result

        Returns:
            TaskDecomposition with subtasks
        """
        # Build prompt for task decomposition
        prompt = self._build_decomposition_prompt(task_description, intent)

        # Query LLM
        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        # Parse response
        decomposition = self._parse_decomposition_response(
            response["content"],
            task_description,
            intent
        )

        logger.info(f"Decomposed into {len(decomposition.subtasks)} subtasks")
        return decomposition

    async def reflect_on_result(
        self,
        task: Task,
        result: Any,
        expected_quality: Optional[Dict[str, Any]] = None
    ) -> Thought:
        """
        Reflect on task result

        Args:
            task: Completed task
            result: Task result
            expected_quality: Expected quality metrics

        Returns:
            Reflection thought
        """
        # Build reflection prompt
        prompt = self._build_reflection_prompt(task, result, expected_quality)

        # Query LLM
        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )

        reflection = Thought(
            content=response["content"],
            thought_type="reflection",
            confidence=0.8
        )

        logger.debug(f"Reflection: {reflection.content[:100]}...")
        return reflection

    async def generate_explanation(
        self,
        action: str,
        reasoning: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate human-readable explanation

        Args:
            action: Action taken
            reasoning: Internal reasoning
            context: Additional context

        Returns:
            Human-readable explanation
        """
        prompt = f"""Generate a clear, concise explanation for the user.

Action taken: {action}
Internal reasoning: {reasoning}
Context: {json.dumps(context) if context else "None"}

Provide a user-friendly explanation (2-3 sentences):"""

        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        return response["content"]

    def _build_intent_prompt(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for intent analysis"""
        prompt = f"""Analyze the user's request and extract key information.

User request: "{user_request}"

Context: {json.dumps(context) if context else "None"}

Please analyze and provide:
1. Primary intent (what the user wants to do)
2. Task type (image_generation, voice_synthesis, character_creation, etc.)
3. Entities mentioned (character names, styles, locations, etc.)
4. Constraints or requirements

Respond in JSON format:
{{
    "primary_intent": "...",
    "task_type": "...",
    "entities": {{}},
    "constraints": [],
    "confidence": 0.9
}}"""
        return prompt

    def _parse_intent_response(self, response: str) -> IntentAnalysis:
        """Parse LLM response for intent"""
        try:
            # Try to parse JSON
            data = json.loads(response)

            # Map task type string to enum
            task_type_str = data.get("task_type", "mixed").lower()
            task_type_map = {
                "image_generation": TaskType.IMAGE_GENERATION,
                "voice_synthesis": TaskType.VOICE_SYNTHESIS,
                "character_creation": TaskType.CHARACTER_CREATION,
                "scene_generation": TaskType.SCENE_GENERATION,
                "style_transfer": TaskType.STYLE_TRANSFER,
                "question_answering": TaskType.QUESTION_ANSWERING,
                "creative_writing": TaskType.CREATIVE_WRITING,
                "analysis": TaskType.ANALYSIS,
            }
            task_type = task_type_map.get(task_type_str, TaskType.MIXED)

            return IntentAnalysis(
                primary_intent=data.get("primary_intent", "Unknown"),
                task_type=task_type,
                entities=data.get("entities", {}),
                constraints=data.get("constraints", []),
                confidence=data.get("confidence", 0.7)
            )

        except json.JSONDecodeError:
            # Fallback: basic parsing
            logger.warning("Failed to parse intent JSON, using fallback")
            return IntentAnalysis(
                primary_intent=response[:100],
                task_type=TaskType.MIXED,
                entities={},
                constraints=[],
                confidence=0.5
            )

    def _build_decomposition_prompt(
        self,
        task_description: str,
        intent: Optional[IntentAnalysis] = None
    ) -> str:
        """Build prompt for task decomposition"""
        intent_info = ""
        if intent:
            intent_info = f"""
Intent analysis:
- Primary intent: {intent.primary_intent}
- Task type: {intent.task_type.value}
- Entities: {json.dumps(intent.entities)}
- Constraints: {intent.constraints}
"""

        prompt = f"""Break down the following task into concrete, executable subtasks.

Task: "{task_description}"
{intent_info}

Consider:
1. What needs to be done step-by-step
2. Dependencies between steps
3. Estimated time for each step

Respond in JSON format:
{{
    "main_task": {{
        "id": "main",
        "description": "..."
    }},
    "subtasks": [
        {{
            "id": "task_1",
            "description": "...",
            "dependencies": [],
            "estimated_time": 10.0
        }}
    ],
    "execution_order": ["task_1", "task_2", ...]
}}"""
        return prompt

    def _parse_decomposition_response(
        self,
        response: str,
        task_description: str,
        intent: Optional[IntentAnalysis] = None
    ) -> TaskDecomposition:
        """Parse LLM response for task decomposition"""
        try:
            data = json.loads(response)

            # Create main task
            task_type = intent.task_type if intent else TaskType.MIXED
            main_task = Task(
                task_id="main",
                description=task_description,
                task_type=task_type,
                priority=0
            )

            # Create subtasks
            subtasks = []
            for i, subtask_data in enumerate(data.get("subtasks", [])):
                subtask = Task(
                    task_id=subtask_data.get("id", f"task_{i+1}"),
                    description=subtask_data.get("description", ""),
                    task_type=task_type,
                    dependencies=subtask_data.get("dependencies", []),
                    priority=i
                )
                subtasks.append(subtask)

            execution_order = data.get("execution_order", [s.task_id for s in subtasks])

            # Estimate total time
            total_time = sum(
                subtask_data.get("estimated_time", 5.0)
                for subtask_data in data.get("subtasks", [])
            )

            return TaskDecomposition(
                main_task=main_task,
                subtasks=subtasks,
                execution_order=execution_order,
                estimated_time=total_time
            )

        except json.JSONDecodeError:
            # Fallback: create single subtask
            logger.warning("Failed to parse decomposition JSON, using fallback")
            task_type = intent.task_type if intent else TaskType.MIXED
            main_task = Task(
                task_id="main",
                description=task_description,
                task_type=task_type
            )
            subtask = Task(
                task_id="task_1",
                description=task_description,
                task_type=task_type
            )
            return TaskDecomposition(
                main_task=main_task,
                subtasks=[subtask],
                execution_order=["task_1"],
                estimated_time=10.0
            )

    def _build_reflection_prompt(
        self,
        task: Task,
        result: Any,
        expected_quality: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for reflection"""
        quality_info = ""
        if expected_quality:
            quality_info = f"\nExpected quality: {json.dumps(expected_quality)}"

        prompt = f"""Reflect on the completed task and its result.

Task: {task.description}
Result: {str(result)[:300]}
{quality_info}

Provide a brief reflection (2-3 sentences):
1. Was the task completed successfully?
2. Does the result meet expectations?
3. Any improvements needed?"""

        return prompt


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async with ThinkingModule() as thinking:
        # Example 1: Understand intent
        user_request = "Generate an image of Luca running on the beach with a happy expression"
        intent = await thinking.understand_intent(user_request)

        print(f"\nIntent Analysis:")
        print(f"  Primary intent: {intent.primary_intent}")
        print(f"  Task type: {intent.task_type.value}")
        print(f"  Entities: {intent.entities}")
        print(f"  Confidence: {intent.confidence}")

        # Example 2: Decompose task
        decomposition = await thinking.decompose_task(user_request, intent)

        print(f"\nTask Decomposition:")
        print(f"  Main task: {decomposition.main_task.description}")
        print(f"  Subtasks: {len(decomposition.subtasks)}")
        for subtask in decomposition.subtasks:
            print(f"    - {subtask.task_id}: {subtask.description}")
        print(f"  Estimated time: {decomposition.estimated_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
