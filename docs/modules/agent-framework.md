# Module 6: Agent Framework Architecture

**Status:** 📋 PLANNED (0%)
**Dependencies:** Module 1 (LLM Backend), Module 2 (Image Generation), Module 3 (Voice Synthesis), Module 5 (RAG System)
**VRAM:** Uses loaded LLM (12-14GB for Qwen2.5-14B)
**Version:** v0.1.0
**Last Updated:** 2025-11-17

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Philosophy](#architecture-philosophy)
3. [Module Components](#module-components)
4. [Sub-Module 1: Thinking Module](#sub-module-1-thinking-module)
5. [Sub-Module 2: Reasoning Module](#sub-module-2-reasoning-module)
6. [Sub-Module 3: Web Search Module](#sub-module-3-web-search-module)
7. [Sub-Module 4: RAG Usage Module](#sub-module-4-rag-usage-module)
8. [Sub-Module 5: Tool Calling Module](#sub-module-5-tool-calling-module)
9. [Sub-Module 6: Function Calling Module](#sub-module-6-function-calling-module)
10. [Sub-Module 7: Multi-Step Reasoning Module](#sub-module-7-multi-step-reasoning-module)
11. [LLM Decision-Making Engine](#llm-decision-making-engine)
12. [Integration Examples](#integration-examples)
13. [Configuration](#configuration)
14. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Goal

The **Agent Framework** is the intelligent orchestration layer that enables **autonomous creative decision-making** for animation AI tasks. The LLM acts as the **central brain**, coordinating multiple specialized modules to understand intent, plan execution, and iterate towards quality results.

### Core Philosophy: LLM + RAG + Agent (缺一不可)

```
User Request
    ↓
[1] LLM Thinking Module → Understand intent, decompose task
    ↓
[2] RAG Usage Module → Retrieve relevant knowledge (characters, styles, past work)
    ↓
[3] Reasoning Module → Plan execution strategy (ReAct, Chain-of-Thought)
    ↓
[4] Tool Calling Module → Select appropriate tools (SDXL, GPT-SoVITS, ControlNet)
    ↓
[5] Multi-Step Execution → Execute, evaluate, iterate
    ↓
[6] Quality Assessment → LLM judges quality, decides next action
    ↓
Final Result (or Loop back to step 3)
```

### Key Principles

1. **LLM as Central Decision Maker**: Not just a tool, but the intelligent coordinator
2. **Autonomous Iteration**: Agent continues until quality threshold met
3. **Multi-Modal Reasoning**: Combines text, image, audio understanding
4. **Tool Agnostic**: Flexibly selects best tool for each sub-task
5. **Self-Improving**: Learns from past executions via RAG

---

## Architecture Philosophy

### Design Principles

```python
class AgentFramework:
    """
    Agent Framework follows these principles:

    1. LLM Decision-Making: All major decisions go through LLM reasoning
    2. Modular Tools: Each capability is a separate, swappable module
    3. Stateful Planning: Maintains context across multi-step workflows
    4. Quality-Driven: Iterates until quality criteria satisfied
    5. Hardware-Aware: Respects RTX 5080 16GB VRAM constraints
    """
```

### LLM as Central Brain

The LLM (Qwen2.5-14B or Qwen2.5-VL-7B) makes ALL critical decisions:

```
Creative Decisions:
- "Should I use ControlNet for pose consistency?"
- "Does this generated image match the character description?"
- "Which voice emotion best fits this dialogue context?"

Technical Decisions:
- "Should I switch from LLM to SDXL model now?"
- "Do I need more context from RAG before proceeding?"
- "Should I re-generate or refine existing output?"

Quality Decisions:
- "Is this image quality acceptable? (Check: character likeness, pose accuracy, style consistency)"
- "Does the voice match the emotion and character personality?"
- "Should I iterate again or move to next step?"
```

### Agent Workflow Patterns

**Pattern 1: ReAct (Reason + Act)**
```
1. Thought: "User wants to generate Luca running on beach"
2. Action: Query RAG for Luca character details
3. Observation: Retrieved Luca appearance, personality traits
4. Thought: "Need to emphasize brown hair, green eyes, excited expression"
5. Action: Call SDXL with ControlNet (running pose reference)
6. Observation: Generated image has correct pose but expression too neutral
7. Thought: "Need to refine expression prompt"
8. Action: Regenerate with stronger emotion keywords
9. Observation: Image now matches requirements
10. Finish: Return final image
```

**Pattern 2: Plan-Execute-Reflect**
```
1. Plan: Break down "Create animated dialogue scene" into steps
   - Generate character images (2 characters)
   - Generate dialogue audio (with emotions)
   - Generate lip-sync animations
   - Composite into final video

2. Execute: Run each step sequentially
   - Check VRAM before each model switch
   - Use RAG to retrieve character/scene context

3. Reflect: After each step, assess quality
   - "Does character A look consistent with training data?"
   - "Is dialogue emotion appropriate for scene?"
   - If NO → iterate, if YES → proceed

4. Final Reflection: Review complete scene
   - "Is overall scene coherent?"
   - "Any improvements needed?"
```

---

## Module Components

### Sub-Module Architecture

```
Agent Framework (Module 6)
├── 1. Thinking Module          # Intent understanding, task decomposition
├── 2. Reasoning Module         # ReAct, CoT, planning strategies
├── 3. Web Search Module        # Real-time information retrieval
├── 4. RAG Usage Module         # Knowledge base retrieval
├── 5. Tool Calling Module      # Dynamic tool selection
├── 6. Function Calling Module  # Standardized function interfaces
├── 7. Multi-Step Reasoning     # Stateful workflow execution
└── LLM Decision Engine         # Central coordinator (Qwen2.5)
```

### Module Dependencies

```yaml
# Agent Framework Dependencies
dependencies:
  required:
    - Module 1 (LLM Backend): Qwen2.5-14B or Qwen2.5-VL-7B
    - Module 5 (RAG System): ChromaDB + embeddings

  optional_tools:
    - Module 2 (Image Generation): SDXL + LoRA + ControlNet
    - Module 3 (Voice Synthesis): GPT-SoVITS
    - Module 4 (Model Manager): Dynamic VRAM management
    - Module 7 (Video Analysis): Scene understanding
    - Module 8 (Video Editing): AI-powered editing

  frameworks:
    - LangGraph: Stateful agent workflows
    - LangChain: Tool/function calling abstractions
```

---

## Sub-Module 1: Thinking Module

### Purpose

The **Thinking Module** enables the LLM to **understand user intent**, **decompose complex tasks**, and **maintain context** across multi-turn conversations.

### Architecture

```python
# scripts/ai_editing/agent/thinking_module.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scripts.core.llm_client import LLMClient

@dataclass
class Thought:
    """Represents a single reasoning thought"""
    content: str
    thought_type: str  # "analysis", "planning", "reflection", "decision"
    context: Dict[str, Any]
    timestamp: float
    confidence: float  # 0.0-1.0

@dataclass
class TaskDecomposition:
    """Result of decomposing complex task into sub-tasks"""
    original_request: str
    intent: str
    sub_tasks: List[Dict[str, Any]]
    dependencies: List[tuple[int, int]]  # (from_task_idx, to_task_idx)
    estimated_steps: int
    requires_iteration: bool

class ThinkingModule:
    """
    LLM-powered thinking and intent understanding

    Capabilities:
    - Understand complex, ambiguous user requests
    - Decompose into actionable sub-tasks
    - Maintain conversation context
    - Generate reasoning traces
    - Self-explain decision-making process
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "qwen-14b",
        temperature: float = 0.7,
        max_context_messages: int = 10
    ):
        """
        Initialize Thinking Module

        Args:
            llm_client: LLM client for inference
            model: LLM model to use (qwen-14b, qwen-vl-7b, qwen-coder-7b)
            temperature: Sampling temperature for creative thinking
            max_context_messages: Maximum conversation history to maintain
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_context_messages = max_context_messages
        self.conversation_history: List[Dict[str, str]] = []
        self.thought_chain: List[Thought] = []

    async def understand_intent(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Understand user's creative intent

        Args:
            user_request: Raw user input
            context: Additional context (current project, past interactions)

        Returns:
            {
                "intent": "generate_character_image",
                "parameters": {
                    "character": "luca",
                    "scene": "running on beach",
                    "emotion": "excited"
                },
                "clarifications_needed": [],
                "confidence": 0.95
            }
        """
        system_prompt = """You are an AI creative assistant specialized in animation and multimedia content creation.

Your task is to understand the user's creative intent from their request.

Analyze the request and extract:
1. Primary intent (what they want to create/do)
2. Key parameters (character, scene, style, emotion, etc.)
3. Any ambiguities that need clarification
4. Confidence in understanding (0.0-1.0)

Respond in JSON format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: {user_request}\n\nContext: {context or {}}"}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        # Parse and validate response
        intent_data = response["choices"][0]["message"]["content"]

        # Store thought
        self.thought_chain.append(Thought(
            content=f"Understood intent: {intent_data.get('intent', 'unknown')}",
            thought_type="analysis",
            context={"user_request": user_request},
            timestamp=time.time(),
            confidence=intent_data.get("confidence", 0.5)
        ))

        return intent_data

    async def decompose_task(
        self,
        intent: str,
        parameters: Dict[str, Any],
        available_tools: List[str]
    ) -> TaskDecomposition:
        """
        Decompose complex task into actionable sub-tasks

        Args:
            intent: Identified user intent
            parameters: Extracted parameters
            available_tools: List of available tools/modules

        Returns:
            TaskDecomposition with sub-tasks and dependencies
        """
        system_prompt = """You are a task planning expert for animation AI workflows.

Given a high-level intent, break it down into concrete sub-tasks that can be executed by available tools.

Consider:
1. Task dependencies (some tasks must complete before others)
2. Hardware constraints (RTX 5080 16GB, only one heavy model at a time)
3. Quality checkpoints (when to evaluate and potentially iterate)
4. Tool availability

Respond with a structured plan in JSON format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Intent: {intent}
Parameters: {parameters}
Available tools: {available_tools}

Create a detailed execution plan."""}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3,  # Lower temperature for planning
            response_format={"type": "json_object"}
        )

        plan_data = response["choices"][0]["message"]["content"]

        # Parse into TaskDecomposition
        decomposition = TaskDecomposition(
            original_request=f"{intent} with {parameters}",
            intent=intent,
            sub_tasks=plan_data.get("sub_tasks", []),
            dependencies=plan_data.get("dependencies", []),
            estimated_steps=len(plan_data.get("sub_tasks", [])),
            requires_iteration=plan_data.get("requires_iteration", False)
        )

        # Store thought
        self.thought_chain.append(Thought(
            content=f"Decomposed into {len(decomposition.sub_tasks)} sub-tasks",
            thought_type="planning",
            context={"decomposition": decomposition},
            timestamp=time.time(),
            confidence=0.9
        ))

        return decomposition

    async def reflect_on_progress(
        self,
        completed_tasks: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reflect on execution progress and decide next actions

        Args:
            completed_tasks: List of completed sub-tasks with results
            current_state: Current workflow state

        Returns:
            {
                "assessment": "on_track" | "needs_adjustment" | "quality_issue",
                "next_action": "continue" | "iterate" | "abort",
                "reasoning": "...",
                "adjustments": [...]
            }
        """
        system_prompt = """You are a quality assessment expert for animation AI workflows.

Review the progress of an ongoing task execution and assess:
1. Are we on track to meet the goal?
2. Is the quality of intermediate results acceptable?
3. Should we continue, iterate on previous steps, or adjust the plan?

Be critical but constructive. Provide clear reasoning for your assessment."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Completed tasks: {completed_tasks}
Current state: {current_state}

Assess progress and recommend next action."""}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        reflection = response["choices"][0]["message"]["content"]

        # Store thought
        self.thought_chain.append(Thought(
            content=f"Reflection: {reflection.get('assessment', 'unknown')}",
            thought_type="reflection",
            context={"reflection": reflection},
            timestamp=time.time(),
            confidence=0.85
        ))

        return reflection

    def get_thought_chain(self) -> List[Thought]:
        """Get full reasoning trace for transparency"""
        return self.thought_chain

    def clear_thought_chain(self):
        """Clear thought chain (start fresh reasoning)"""
        self.thought_chain = []
```

### Usage Example

```python
# Example: Understanding and decomposing a complex creative request

from scripts.ai_editing.agent.thinking_module import ThinkingModule
from scripts.core.llm_client import LLMClient

async def main():
    async with LLMClient() as llm_client:
        thinking = ThinkingModule(llm_client, model="qwen-14b")

        # User request
        user_request = "Create a scene where Luca and Alberto are talking on the beach at sunset"

        # 1. Understand intent
        intent_data = await thinking.understand_intent(user_request)
        print(f"Intent: {intent_data['intent']}")
        print(f"Parameters: {intent_data['parameters']}")

        # 2. Decompose into sub-tasks
        decomposition = await thinking.decompose_task(
            intent=intent_data['intent'],
            parameters=intent_data['parameters'],
            available_tools=["sdxl", "controlnet", "gpt_sovits", "video_editor"]
        )

        print(f"\nExecution Plan ({decomposition.estimated_steps} steps):")
        for i, task in enumerate(decomposition.sub_tasks):
            print(f"  {i+1}. {task['description']}")

        # 3. Get reasoning trace
        thought_chain = thinking.get_thought_chain()
        print(f"\nThought Chain ({len(thought_chain)} thoughts):")
        for thought in thought_chain:
            print(f"  [{thought.thought_type}] {thought.content} (confidence: {thought.confidence})")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/thinking_config.yaml

thinking_module:
  llm:
    model: "qwen-14b"  # or "qwen-vl-7b" for multimodal
    temperature: 0.7   # Creative thinking
    max_tokens: 2048

  context:
    max_history_messages: 10
    max_thought_chain_length: 50

  intent_understanding:
    confidence_threshold: 0.7  # Minimum confidence to proceed
    ask_clarification_below: 0.6  # Ask user if confidence below this

  task_decomposition:
    max_subtasks: 20
    planning_temperature: 0.3  # Lower for structured planning

  reflection:
    reflection_interval: 3  # Reflect after every N sub-tasks
    quality_threshold: 0.8  # Minimum quality score to proceed
```

---

## Sub-Module 2: Reasoning Module

### Purpose

The **Reasoning Module** implements various **reasoning strategies** (ReAct, Chain-of-Thought, Tree-of-Thoughts) to enable the LLM to **plan, execute, and adapt** complex multi-step workflows.

### Architecture

```python
# scripts/ai_editing/agent/reasoning_module.py

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

class ReasoningStrategy(Enum):
    """Supported reasoning strategies"""
    REACT = "react"  # Reason + Act (interleaved reasoning and action)
    CHAIN_OF_THOUGHT = "cot"  # Step-by-step reasoning
    TREE_OF_THOUGHTS = "tot"  # Explore multiple reasoning paths
    PLAN_EXECUTE = "plan_execute"  # Plan first, then execute
    REFLEXION = "reflexion"  # Execute, reflect, refine

@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_number: int
    thought: str
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[str]
    reflection: Optional[str]
    timestamp: float

@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a task"""
    strategy: ReasoningStrategy
    steps: List[ReasoningStep]
    final_answer: Any
    success: bool
    total_steps: int
    reasoning_time: float

class ReActReasoner:
    """
    ReAct (Reason + Act) Reasoning Strategy

    Interleaves reasoning (thoughts) with actions (tool calls)

    Flow:
    1. Thought: "What should I do next?"
    2. Action: Call a tool
    3. Observation: Receive tool output
    4. Thought: "What does this mean? Should I continue?"
    5. (Repeat until task complete)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "qwen-14b",
        max_iterations: int = 10
    ):
        self.llm_client = llm_client
        self.model = model
        self.max_iterations = max_iterations

    async def reason(
        self,
        task: str,
        available_actions: Dict[str, Callable],
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Execute ReAct reasoning loop

        Args:
            task: Task description
            available_actions: Dict mapping action names to callable functions
            context: Additional context

        Returns:
            ReasoningTrace with complete reasoning process
        """
        steps: List[ReasoningStep] = []
        step_number = 0
        start_time = time.time()

        system_prompt = """You are an AI agent using ReAct (Reason + Act) reasoning.

For each step, you must:
1. Think: Reason about what to do next
2. Act: Choose an action and provide input
3. Observe: Receive action result
4. Reflect: Decide if task is complete or continue

Available actions: {available_actions}

Respond in this format:
{{
    "thought": "I need to...",
    "action": "action_name",
    "action_input": {{"param": "value"}},
    "is_complete": false
}}"""

        conversation_history = [
            {"role": "system", "content": system_prompt.format(
                available_actions=list(available_actions.keys())
            )},
            {"role": "user", "content": f"Task: {task}\nContext: {context or {}}"}
        ]

        for iteration in range(self.max_iterations):
            # LLM generates thought + action
            response = await self.llm_client.chat(
                model=self.model,
                messages=conversation_history,
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            step_data = response["choices"][0]["message"]["content"]

            # Execute action
            action_name = step_data.get("action")
            action_input = step_data.get("action_input", {})

            if action_name and action_name in available_actions:
                observation = await available_actions[action_name](**action_input)
            else:
                observation = "Action not available or not specified"

            # Record step
            step = ReasoningStep(
                step_number=step_number,
                thought=step_data.get("thought", ""),
                action=action_name,
                action_input=action_input,
                observation=str(observation),
                reflection=None,
                timestamp=time.time()
            )
            steps.append(step)

            # Add observation to conversation
            conversation_history.append({
                "role": "assistant",
                "content": f"Thought: {step.thought}\nAction: {action_name}\nInput: {action_input}"
            })
            conversation_history.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

            # Check if complete
            if step_data.get("is_complete", False):
                break

            step_number += 1

        reasoning_time = time.time() - start_time

        return ReasoningTrace(
            strategy=ReasoningStrategy.REACT,
            steps=steps,
            final_answer=steps[-1].observation if steps else None,
            success=True,
            total_steps=len(steps),
            reasoning_time=reasoning_time
        )

class ChainOfThoughtReasoner:
    """
    Chain-of-Thought (CoT) Reasoning Strategy

    Breaks down complex reasoning into explicit step-by-step thoughts
    before taking action

    Flow:
    1. Analyze the problem
    2. Break into logical steps
    3. Reason through each step
    4. Synthesize final answer
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "qwen-14b"
    ):
        self.llm_client = llm_client
        self.model = model

    async def reason(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Execute Chain-of-Thought reasoning

        Args:
            task: Task description
            context: Additional context

        Returns:
            ReasoningTrace with step-by-step reasoning
        """
        start_time = time.time()

        system_prompt = """You are an AI agent using Chain-of-Thought reasoning.

Break down the problem into logical steps and reason through each step explicitly.

Format your response as:
{{
    "problem_analysis": "...",
    "reasoning_steps": [
        {{"step": 1, "thought": "...", "conclusion": "..."}},
        {{"step": 2, "thought": "...", "conclusion": "..."}},
        ...
    ],
    "final_answer": "..."
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\nContext: {context or {}}"}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        reasoning_data = response["choices"][0]["message"]["content"]

        # Convert to ReasoningStep format
        steps = []
        for i, step_data in enumerate(reasoning_data.get("reasoning_steps", [])):
            step = ReasoningStep(
                step_number=i,
                thought=step_data.get("thought", ""),
                action=None,
                action_input=None,
                observation=step_data.get("conclusion", ""),
                reflection=None,
                timestamp=time.time()
            )
            steps.append(step)

        reasoning_time = time.time() - start_time

        return ReasoningTrace(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            steps=steps,
            final_answer=reasoning_data.get("final_answer"),
            success=True,
            total_steps=len(steps),
            reasoning_time=reasoning_time
        )

class ReasoningModule:
    """
    Central reasoning module supporting multiple strategies
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "qwen-14b",
        default_strategy: ReasoningStrategy = ReasoningStrategy.REACT
    ):
        self.llm_client = llm_client
        self.model = model
        self.default_strategy = default_strategy

        # Initialize reasoners
        self.react_reasoner = ReActReasoner(llm_client, model)
        self.cot_reasoner = ChainOfThoughtReasoner(llm_client, model)

    async def reason(
        self,
        task: str,
        strategy: Optional[ReasoningStrategy] = None,
        available_actions: Optional[Dict[str, Callable]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Execute reasoning with specified strategy

        Args:
            task: Task to reason about
            strategy: Reasoning strategy to use (defaults to self.default_strategy)
            available_actions: Available actions for ReAct
            context: Additional context

        Returns:
            ReasoningTrace with complete reasoning process
        """
        strategy = strategy or self.default_strategy

        if strategy == ReasoningStrategy.REACT:
            if not available_actions:
                raise ValueError("ReAct strategy requires available_actions")
            return await self.react_reasoner.reason(task, available_actions, context)

        elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            return await self.cot_reasoner.reason(task, context)

        else:
            raise NotImplementedError(f"Strategy {strategy} not yet implemented")

    async def adaptive_reason(
        self,
        task: str,
        available_actions: Optional[Dict[str, Callable]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Automatically select best reasoning strategy for task

        Uses LLM to analyze task and choose appropriate strategy
        """
        # Ask LLM to choose strategy
        system_prompt = """Analyze the task and choose the best reasoning strategy:

- ReAct: Good for tasks requiring interleaved reasoning and tool use
- Chain-of-Thought: Good for complex analytical tasks
- Plan-Execute: Good for tasks with clear sequential steps

Respond with: {{"strategy": "react|cot|plan_execute", "reasoning": "..."}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        strategy_data = response["choices"][0]["message"]["content"]
        chosen_strategy = ReasoningStrategy(strategy_data.get("strategy", "react"))

        # Execute with chosen strategy
        return await self.reason(task, chosen_strategy, available_actions, context)
```

### Usage Example

```python
# Example: ReAct reasoning for image generation task

from scripts.ai_editing.agent.reasoning_module import ReasoningModule, ReasoningStrategy
from scripts.core.llm_client import LLMClient

async def main():
    async with LLMClient() as llm_client:
        reasoning = ReasoningModule(llm_client, model="qwen-14b")

        # Define available actions
        async def retrieve_character_info(character: str) -> Dict:
            """Mock RAG retrieval"""
            return {
                "name": character,
                "appearance": "brown hair, green eyes, slim build",
                "personality": "curious, adventurous, kind"
            }

        async def generate_image(prompt: str, controlnet: str = None) -> str:
            """Mock image generation"""
            return f"Generated image with prompt: {prompt}"

        async def evaluate_quality(image_path: str, criteria: List[str]) -> float:
            """Mock quality evaluation"""
            return 0.85

        available_actions = {
            "retrieve_character_info": retrieve_character_info,
            "generate_image": generate_image,
            "evaluate_quality": evaluate_quality
        }

        # Execute ReAct reasoning
        task = "Generate a high-quality image of Luca running on the beach with excited expression"

        trace = await reasoning.reason(
            task=task,
            strategy=ReasoningStrategy.REACT,
            available_actions=available_actions
        )

        print(f"Reasoning completed in {trace.reasoning_time:.2f}s with {trace.total_steps} steps")
        print("\nReasoning Trace:")
        for step in trace.steps:
            print(f"\nStep {step.step_number}:")
            print(f"  Thought: {step.thought}")
            print(f"  Action: {step.action}({step.action_input})")
            print(f"  Observation: {step.observation}")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/reasoning_config.yaml

reasoning_module:
  llm:
    model: "qwen-14b"
    temperature: 0.5

  react:
    max_iterations: 10
    reflection_frequency: 3  # Reflect every N steps

  chain_of_thought:
    max_reasoning_steps: 15
    require_explicit_conclusion: true

  adaptive:
    enable_auto_strategy_selection: true
    fallback_strategy: "react"
```

---

## Sub-Module 3: Web Search Module

### Purpose

The **Web Search Module** enables the agent to retrieve **real-time information** from the internet when knowledge is needed beyond the LLM's training data or local RAG knowledge base.

### Architecture

```python
# scripts/ai_editing/agent/web_search_module.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp
from bs4 import BeautifulSoup

@dataclass
class SearchResult:
    """Single search result"""
    title: str
    url: str
    snippet: str
    relevance_score: float
    source: str  # "google", "bing", "duckduckgo"
    timestamp: float

@dataclass
class WebContent:
    """Extracted web content"""
    url: str
    title: str
    text_content: str
    images: List[str]
    metadata: Dict[str, Any]

class WebSearchModule:
    """
    Web search and content retrieval module

    Capabilities:
    - Search web for relevant information
    - Extract and clean web content
    - Rank results by relevance
    - Synthesize information from multiple sources
    """

    def __init__(
        self,
        llm_client: LLMClient,
        search_engine: str = "duckduckgo",  # Privacy-respecting default
        max_results: int = 5
    ):
        """
        Initialize Web Search Module

        Args:
            llm_client: LLM client for relevance ranking and synthesis
            search_engine: Search engine to use
            max_results: Maximum search results to retrieve
        """
        self.llm_client = llm_client
        self.search_engine = search_engine
        self.max_results = max_results

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search web for query

        Args:
            query: Search query
            filters: Optional filters (date range, site, etc.)

        Returns:
            List of ranked SearchResult objects
        """
        # Implementation depends on search engine API
        # Example with DuckDuckGo (privacy-respecting)

        if self.search_engine == "duckduckgo":
            results = await self._search_duckduckgo(query)
        else:
            raise NotImplementedError(f"Search engine {self.search_engine} not supported")

        # Rank results by relevance using LLM
        ranked_results = await self._rank_results(query, results)

        return ranked_results[:self.max_results]

    async def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """DuckDuckGo search implementation"""
        # Use duckduckgo_search library
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=self.max_results * 2)

            for i, result in enumerate(search_results):
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("href", ""),
                    snippet=result.get("body", ""),
                    relevance_score=1.0 - (i / len(search_results)),  # Initial scoring
                    source="duckduckgo",
                    timestamp=time.time()
                ))

        return results

    async def _rank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Use LLM to rank search results by relevance"""

        system_prompt = """You are a search result relevance evaluator.

Given a search query and a list of search results, score each result's relevance from 0.0 to 1.0.

Consider:
- Title and snippet relevance to query
- Source credibility
- Information freshness

Respond with: {{"scores": [0.95, 0.82, ...]}}"""

        results_text = "\n".join([
            f"{i+1}. Title: {r.title}\n   Snippet: {r.snippet}\n   URL: {r.url}"
            for i, r in enumerate(results)
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nResults:\n{results_text}"}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        scores = response["choices"][0]["message"]["content"].get("scores", [])

        # Update relevance scores
        for i, score in enumerate(scores):
            if i < len(results):
                results[i].relevance_score = score

        # Sort by relevance
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    async def fetch_content(self, url: str) -> WebContent:
        """
        Fetch and extract content from URL

        Args:
            url: URL to fetch

        Returns:
            WebContent with extracted text and metadata
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                html = await response.text()

        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        title = soup.find('title')
        title = title.get_text() if title else ""

        # Extract main content (remove scripts, styles, etc.)
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        text_content = soup.get_text(separator="\n", strip=True)

        # Extract images
        images = [img.get('src') for img in soup.find_all('img') if img.get('src')]

        return WebContent(
            url=url,
            title=title,
            text_content=text_content,
            images=images,
            metadata={"fetch_time": time.time()}
        )

    async def synthesize_information(
        self,
        query: str,
        search_results: List[SearchResult],
        web_contents: List[WebContent]
    ) -> str:
        """
        Synthesize information from multiple web sources

        Args:
            query: Original query
            search_results: Search results metadata
            web_contents: Fetched web content

        Returns:
            Synthesized answer with citations
        """
        system_prompt = """You are an information synthesis expert.

Given a query and content from multiple web sources, synthesize a comprehensive answer.

Requirements:
- Cite sources using [1], [2], etc.
- Integrate information from all sources
- Resolve contradictions if any
- Provide balanced perspective"""

        sources_text = "\n\n".join([
            f"[{i+1}] {content.title}\nURL: {content.url}\nContent:\n{content.text_content[:1000]}"
            for i, content in enumerate(web_contents)
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nSources:\n{sources_text}"}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.5,
            max_tokens=1024
        )

        return response["choices"][0]["message"]["content"]
```

### Usage Example

```python
# Example: Search for animation technique information

from scripts.ai_editing.agent.web_search_module import WebSearchModule
from scripts.core.llm_client import LLMClient

async def main():
    async with LLMClient() as llm_client:
        web_search = WebSearchModule(llm_client, search_engine="duckduckgo")

        # Search for information
        query = "best practices for ControlNet pose consistency in animation"

        results = await web_search.search(query)

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.title} (relevance: {result.relevance_score:.2f})")
            print(f"   {result.url}")
            print(f"   {result.snippet[:100]}...")

        # Fetch top 3 results
        contents = []
        for result in results[:3]:
            try:
                content = await web_search.fetch_content(result.url)
                contents.append(content)
            except Exception as e:
                print(f"Failed to fetch {result.url}: {e}")

        # Synthesize information
        answer = await web_search.synthesize_information(query, results, contents)

        print(f"\nSynthesized Answer:\n{answer}")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/web_search_config.yaml

web_search_module:
  search_engine: "duckduckgo"  # Privacy-respecting
  max_results: 5
  timeout: 10  # seconds

  content_extraction:
    max_content_length: 5000  # characters
    remove_boilerplate: true

  synthesis:
    model: "qwen-14b"
    temperature: 0.5
    max_synthesis_length: 1024
```

---

## Sub-Module 4: RAG Usage Module

### Purpose

The **RAG Usage Module** enables the agent to retrieve relevant information from the **local knowledge base** (character metadata, style guides, past generations, film analysis) to inform creative decisions.

### Architecture

```python
# scripts/ai_editing/agent/rag_usage_module.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions

@dataclass
class RetrievalResult:
    """Single retrieval result from RAG"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str  # "character_db", "style_guide", "past_generations"
    timestamp: float

class RAGUsageModule:
    """
    RAG (Retrieval-Augmented Generation) Usage Module

    Capabilities:
    - Retrieve character information
    - Retrieve style guides and references
    - Retrieve past successful generations
    - Retrieve film analysis and insights
    - Context-aware retrieval with re-ranking
    """

    def __init__(
        self,
        llm_client: LLMClient,
        chroma_client: chromadb.Client,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5
    ):
        """
        Initialize RAG Usage Module

        Args:
            llm_client: LLM client for re-ranking and synthesis
            chroma_client: ChromaDB client
            embedding_model: Embedding model for retrieval
            top_k: Number of top results to retrieve
        """
        self.llm_client = llm_client
        self.chroma_client = chroma_client
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.top_k = top_k

        # Initialize collections
        self.character_collection = chroma_client.get_or_create_collection(
            name="character_knowledge",
            embedding_function=self.embedding_fn
        )
        self.style_collection = chroma_client.get_or_create_collection(
            name="style_guides",
            embedding_function=self.embedding_fn
        )
        self.generations_collection = chroma_client.get_or_create_collection(
            name="past_generations",
            embedding_function=self.embedding_fn
        )

    async def retrieve_character_info(
        self,
        character_name: str,
        context: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve character information

        Args:
            character_name: Character name (e.g., "luca", "alberto")
            context: Optional context to refine retrieval

        Returns:
            List of relevant character information
        """
        query = f"{character_name}"
        if context:
            query += f" {context}"

        results = self.character_collection.query(
            query_texts=[query],
            n_results=self.top_k
        )

        retrieval_results = []
        for i in range(len(results['documents'][0])):
            retrieval_results.append(RetrievalResult(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                relevance_score=1.0 - results['distances'][0][i],  # Convert distance to score
                source="character_db",
                timestamp=time.time()
            ))

        # LLM re-ranking for context-aware retrieval
        if context:
            retrieval_results = await self._rerank_results(query, retrieval_results)

        return retrieval_results

    async def retrieve_style_guide(
        self,
        style_query: str
    ) -> List[RetrievalResult]:
        """
        Retrieve style guide information

        Args:
            style_query: Style description (e.g., "pixar 3d animation lighting")

        Returns:
            List of relevant style guide entries
        """
        results = self.style_collection.query(
            query_texts=[style_query],
            n_results=self.top_k
        )

        retrieval_results = []
        for i in range(len(results['documents'][0])):
            retrieval_results.append(RetrievalResult(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                relevance_score=1.0 - results['distances'][0][i],
                source="style_guide",
                timestamp=time.time()
            ))

        return retrieval_results

    async def retrieve_similar_generations(
        self,
        description: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve similar past successful generations

        Args:
            description: Generation description
            filters: Optional filters (character, quality_score, etc.)

        Returns:
            List of similar past generations
        """
        where_clause = filters if filters else None

        results = self.generations_collection.query(
            query_texts=[description],
            n_results=self.top_k,
            where=where_clause
        )

        retrieval_results = []
        for i in range(len(results['documents'][0])):
            retrieval_results.append(RetrievalResult(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                relevance_score=1.0 - results['distances'][0][i],
                source="past_generations",
                timestamp=time.time()
            ))

        return retrieval_results

    async def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Re-rank results using LLM for better context relevance

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Re-ranked results
        """
        system_prompt = """You are a relevance scoring expert.

Given a query and a list of retrieved documents, score each document's relevance from 0.0 to 1.0.

Consider:
- Semantic relevance to query
- Information completeness
- Contextual appropriateness

Respond with: {{"scores": [0.95, 0.82, ...]}}"""

        results_text = "\n".join([
            f"{i+1}. {r.content[:200]}... (source: {r.source})"
            for i, r in enumerate(results)
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{results_text}"}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        scores = response["choices"][0]["message"]["content"].get("scores", [])

        # Update scores
        for i, score in enumerate(scores):
            if i < len(results):
                results[i].relevance_score = score

        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    async def synthesize_context(
        self,
        query: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """
        Synthesize retrieved information into coherent context

        Args:
            query: Original query
            retrieval_results: Retrieved information

        Returns:
            Synthesized context string
        """
        system_prompt = """You are a context synthesis expert.

Given a query and retrieved information from multiple sources, synthesize a coherent context summary.

Focus on:
- Key facts relevant to the query
- Consistent information across sources
- Actionable insights"""

        results_text = "\n\n".join([
            f"[{r.source}] {r.content}"
            for r in retrieval_results
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nRetrieved Information:\n{results_text}"}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.5,
            max_tokens=512
        )

        return response["choices"][0]["message"]["content"]
```

### Usage Example

```python
# Example: Retrieve character information with RAG

from scripts.ai_editing.agent.rag_usage_module import RAGUsageModule
from scripts.core.llm_client import LLMClient
import chromadb

async def main():
    # Initialize ChromaDB
    chroma_client = chromadb.Client()

    async with LLMClient() as llm_client:
        rag = RAGUsageModule(llm_client, chroma_client, top_k=3)

        # Retrieve character information
        character_results = await rag.retrieve_character_info(
            character_name="luca",
            context="running on beach excited expression"
        )

        print("Character Information Retrieved:")
        for result in character_results:
            print(f"  [Score: {result.relevance_score:.2f}] {result.content[:100]}...")

        # Retrieve style guide
        style_results = await rag.retrieve_style_guide(
            style_query="pixar 3d animation warm summer lighting"
        )

        print("\nStyle Guide Retrieved:")
        for result in style_results:
            print(f"  [Score: {result.relevance_score:.2f}] {result.content[:100]}...")

        # Synthesize context
        all_results = character_results + style_results
        context = await rag.synthesize_context(
            query="Generate Luca running on beach",
            retrieval_results=all_results
        )

        print(f"\nSynthesized Context:\n{context}")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/rag_config.yaml

rag_usage_module:
  embedding:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cuda"

  retrieval:
    top_k: 5
    rerank_with_llm: true
    rerank_model: "qwen-14b"

  collections:
    character_knowledge:
      path: "data/rag/character_db"
      auto_update: true

    style_guides:
      path: "data/rag/style_guides"

    past_generations:
      path: "data/rag/generations"
      max_entries: 10000
```

---

Due to length constraints, I'll continue this document in a follow-up. The remaining sub-modules to cover are:

- **Sub-Module 5: Tool Calling Module**
- **Sub-Module 6: Function Calling Module**
- **Sub-Module 7: Multi-Step Reasoning Module**
- **LLM Decision-Making Engine**
- **Integration Examples**
- **Configuration**
- **Implementation Checklist**

Would you like me to continue with the complete document?
# Module 6: Agent Framework Architecture (Part 2)

**Continuation of agent-framework.md**

This document contains the remaining sub-modules and integration details for the Agent Framework.

---

## Sub-Module 5: Tool Calling Module

### Purpose

The **Tool Calling Module** enables the agent to **dynamically select and execute tools** (SDXL, GPT-SoVITS, ControlNet, video analysis, etc.) based on task requirements and current context.

### Architecture

```python
# scripts/ai_editing/agent/tool_calling_module.py

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

class ToolCategory(Enum):
    """Tool categories"""
    IMAGE_GENERATION = "image_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    VIDEO_ANALYSIS = "video_analysis"
    VIDEO_EDITING = "video_editing"
    ENHANCEMENT = "enhancement"
    UTILITY = "utility"

@dataclass
class Tool:
    """Tool definition"""
    name: str
    category: ToolCategory
    description: str
    parameters: Dict[str, Any]
    vram_required: float  # GB
    execution_time_estimate: float  # seconds
    callable: Callable
    dependencies: List[str]  # Other tools required

@dataclass
class ToolCall:
    """Record of a tool execution"""
    tool_name: str
    input_parameters: Dict[str, Any]
    output: Any
    execution_time: float
    success: bool
    error_message: Optional[str]
    timestamp: float

class ToolRegistry:
    """
    Central registry of all available tools

    Manages tool registration, discovery, and metadata
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(
        self,
        name: str,
        category: ToolCategory,
        description: str,
        parameters: Dict[str, Any],
        vram_required: float,
        execution_time_estimate: float,
        callable: Callable,
        dependencies: List[str] = None
    ):
        """Register a new tool"""
        self.tools[name] = Tool(
            name=name,
            category=category,
            description=description,
            parameters=parameters,
            vram_required=vram_required,
            execution_time_estimate=execution_time_estimate,
            callable=callable,
            dependencies=dependencies or []
        )

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category"""
        return [tool for tool in self.tools.values() if tool.category == category]

    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all registered tools"""
        return self.tools

class ToolCallingModule:
    """
    Tool calling and execution module

    Capabilities:
    - LLM-powered tool selection
    - Hardware-aware tool execution (VRAM constraints)
    - Tool dependency resolution
    - Parallel tool execution where possible
    - Error handling and retry logic
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        model_manager: Optional[Any] = None,  # Module 4: Model Manager
        max_concurrent_tools: int = 1  # RTX 5080 16GB limitation
    ):
        """
        Initialize Tool Calling Module

        Args:
            llm_client: LLM client for tool selection
            tool_registry: Registry of available tools
            model_manager: Optional model manager for VRAM-aware switching
            max_concurrent_tools: Maximum concurrent tools (1 for single GPU)
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.model_manager = model_manager
        self.max_concurrent_tools = max_concurrent_tools
        self.tool_call_history: List[ToolCall] = []

    async def select_tool(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use LLM to select appropriate tool for task

        Args:
            task_description: Description of what needs to be done
            context: Current context (previous steps, available resources)
            constraints: Constraints (VRAM limit, time limit, quality requirements)

        Returns:
            Selected tool name
        """
        # Get available tools
        available_tools = self.tool_registry.get_all_tools()

        # Format tools for LLM
        tools_description = "\n".join([
            f"- {name}: {tool.description} (VRAM: {tool.vram_required}GB, ~{tool.execution_time_estimate}s)"
            for name, tool in available_tools.items()
        ])

        system_prompt = """You are a tool selection expert for animation AI workflows.

Given a task description and available tools, select the SINGLE MOST APPROPRIATE tool to use.

Consider:
1. Tool capability match to task requirements
2. VRAM constraints (RTX 5080 16GB, only one heavy model at a time)
3. Execution time estimates
4. Tool dependencies
5. Current context

Respond with: {{"tool_name": "...", "reasoning": "..."}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Task: {task_description}
Context: {context or {}}
Constraints: {constraints or {}}

Available Tools:
{tools_description}

Select the best tool."""}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        selection_data = response["choices"][0]["message"]["content"]
        return selection_data.get("tool_name")

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolCall:
        """
        Execute a tool with given parameters

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            timeout: Optional timeout in seconds

        Returns:
            ToolCall record with execution results
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found in registry")

        # Check VRAM availability (if model manager available)
        if self.model_manager:
            current_vram = self.model_manager.get_vram_usage()
            if current_vram['used'] + tool.vram_required > current_vram['total'] * 0.9:
                # Need to free VRAM first
                await self.model_manager.free_vram_for_tool(tool.vram_required)

        # Execute tool
        start_time = time.time()
        success = True
        error_message = None
        output = None

        try:
            if asyncio.iscoroutinefunction(tool.callable):
                if timeout:
                    output = await asyncio.wait_for(
                        tool.callable(**parameters),
                        timeout=timeout
                    )
                else:
                    output = await tool.callable(**parameters)
            else:
                output = tool.callable(**parameters)

        except Exception as e:
            success = False
            error_message = str(e)

        execution_time = time.time() - start_time

        # Record tool call
        tool_call = ToolCall(
            tool_name=tool_name,
            input_parameters=parameters,
            output=output,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            timestamp=time.time()
        )

        self.tool_call_history.append(tool_call)

        return tool_call

    async def execute_tool_chain(
        self,
        tasks: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[ToolCall]:
        """
        Execute a chain of tool calls

        Args:
            tasks: List of tasks, each with {"description": "...", "parameters": {...}}
            parallel: Whether to execute in parallel (if hardware allows)

        Returns:
            List of ToolCall records
        """
        tool_calls = []

        if parallel and self.max_concurrent_tools > 1:
            # Parallel execution (not applicable for RTX 5080 16GB single GPU)
            # But included for future multi-GPU support
            tasks_coros = []
            for task in tasks:
                tool_name = await self.select_tool(
                    task_description=task["description"],
                    context=task.get("context")
                )
                tasks_coros.append(
                    self.execute_tool(tool_name, task.get("parameters", {}))
                )

            tool_calls = await asyncio.gather(*tasks_coros)

        else:
            # Sequential execution (RTX 5080 16GB constraint)
            for task in tasks:
                # Select tool
                tool_name = await self.select_tool(
                    task_description=task["description"],
                    context=task.get("context")
                )

                # Execute tool
                tool_call = await self.execute_tool(
                    tool_name=tool_name,
                    parameters=task.get("parameters", {})
                )

                tool_calls.append(tool_call)

                # If tool failed, decide whether to retry or abort
                if not tool_call.success:
                    # Ask LLM whether to retry or abort
                    should_retry = await self._should_retry_failed_tool(tool_call, task)
                    if should_retry:
                        # Retry with adjusted parameters
                        adjusted_params = await self._adjust_parameters_for_retry(
                            tool_call, task
                        )
                        retry_call = await self.execute_tool(
                            tool_name=tool_name,
                            parameters=adjusted_params
                        )
                        tool_calls.append(retry_call)
                    else:
                        # Abort chain
                        break

        return tool_calls

    async def _should_retry_failed_tool(
        self,
        failed_call: ToolCall,
        original_task: Dict[str, Any]
    ) -> bool:
        """Ask LLM whether to retry failed tool"""

        system_prompt = """You are an error recovery expert for animation AI workflows.

A tool execution failed. Decide whether to:
1. Retry with adjusted parameters
2. Abort and report error

Consider:
- Error type (recoverable vs fatal)
- Retry potential
- Time constraints

Respond with: {{"should_retry": true/false, "reasoning": "..."}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Tool: {failed_call.tool_name}
Error: {failed_call.error_message}
Task: {original_task['description']}

Should we retry?"""}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        decision = response["choices"][0]["message"]["content"]
        return decision.get("should_retry", False)

    async def _adjust_parameters_for_retry(
        self,
        failed_call: ToolCall,
        original_task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask LLM to adjust parameters for retry"""

        system_prompt = """You are a parameter adjustment expert for animation AI workflows.

A tool execution failed. Suggest adjusted parameters for retry.

Consider the error message and original parameters to propose improvements."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Tool: {failed_call.tool_name}
Error: {failed_call.error_message}
Original Parameters: {failed_call.input_parameters}

Suggest adjusted parameters (return full parameter dict)."""}
        ]

        response = await self.llm_client.chat(
            model="qwen-14b",
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        adjusted = response["choices"][0]["message"]["content"]
        return adjusted.get("parameters", failed_call.input_parameters)

    def get_tool_call_history(self) -> List[ToolCall]:
        """Get all tool call history for this session"""
        return self.tool_call_history
```

### Tool Registration Example

```python
# Example: Register image generation tools

from scripts.ai_editing.agent.tool_calling_module import ToolRegistry, ToolCategory
from scripts.generation.image.sdxl_pipeline import SDXLPipelineManager

# Initialize registry
registry = ToolRegistry()

# Create SDXL pipeline
sdxl_pipeline = SDXLPipelineManager(
    model_path="/path/to/sdxl",
    device="cuda",
    dtype=torch.float16
)

# Register SDXL tool
registry.register_tool(
    name="sdxl_generate",
    category=ToolCategory.IMAGE_GENERATION,
    description="Generate high-quality images with SDXL base model",
    parameters={
        "prompt": {"type": "string", "required": True},
        "negative_prompt": {"type": "string", "required": False},
        "width": {"type": "integer", "default": 1024},
        "height": {"type": "integer", "default": 1024},
        "num_inference_steps": {"type": "integer", "default": 30},
        "guidance_scale": {"type": "float", "default": 7.5}
    },
    vram_required=13.0,  # GB
    execution_time_estimate=15.0,  # seconds
    callable=sdxl_pipeline.generate
)

# Register ControlNet tool
registry.register_tool(
    name="controlnet_pose",
    category=ToolCategory.IMAGE_GENERATION,
    description="Generate images with pose control using ControlNet",
    parameters={
        "prompt": {"type": "string", "required": True},
        "control_image": {"type": "string", "required": True},  # Path to pose reference
        "controlnet_conditioning_scale": {"type": "float", "default": 1.0}
    },
    vram_required=14.5,  # GB (SDXL + ControlNet)
    execution_time_estimate=20.0,
    callable=sdxl_pipeline.generate_with_controlnet,
    dependencies=["sdxl_generate"]
)

# Register GPT-SoVITS tool
from scripts.synthesis.tts.gpt_sovits_wrapper import GPTSoVITSWrapper

tts_wrapper = GPTSoVITSWrapper()

registry.register_tool(
    name="gpt_sovits_synthesize",
    category=ToolCategory.VOICE_SYNTHESIS,
    description="Synthesize character voice with emotion control",
    parameters={
        "text": {"type": "string", "required": True},
        "character": {"type": "string", "required": True},
        "emotion": {"type": "string", "default": "neutral"},
        "language": {"type": "string", "default": "en"},
        "speed": {"type": "float", "default": 1.0}
    },
    vram_required=3.5,  # GB
    execution_time_estimate=5.0,
    callable=tts_wrapper.synthesize
)
```

### Usage Example

```python
# Example: LLM-driven tool selection and execution

from scripts.ai_editing.agent.tool_calling_module import ToolCallingModule
from scripts.core.llm_client import LLMClient

async def main():
    # Initialize
    async with LLMClient() as llm_client:
        tool_calling = ToolCallingModule(
            llm_client=llm_client,
            tool_registry=registry,
            max_concurrent_tools=1  # RTX 5080 16GB
        )

        # Task: Generate character image
        task_description = "Generate an image of Luca running on the beach with excited expression"

        # LLM selects appropriate tool
        selected_tool = await tool_calling.select_tool(
            task_description=task_description,
            constraints={"vram_limit": 16.0, "quality": "high"}
        )

        print(f"Selected tool: {selected_tool}")

        # Execute tool
        tool_call = await tool_calling.execute_tool(
            tool_name=selected_tool,
            parameters={
                "prompt": "luca, boy with brown hair, running on beach, excited expression, pixar style, 3d animation",
                "negative_prompt": "blurry, low quality, distorted",
                "num_inference_steps": 40,
                "guidance_scale": 8.0
            }
        )

        if tool_call.success:
            print(f"Tool executed successfully in {tool_call.execution_time:.2f}s")
            print(f"Output: {tool_call.output}")
        else:
            print(f"Tool failed: {tool_call.error_message}")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/tool_calling_config.yaml

tool_calling_module:
  llm:
    model: "qwen-14b"
    temperature: 0.3  # Lower for deterministic tool selection

  hardware:
    max_concurrent_tools: 1  # RTX 5080 16GB limitation
    vram_safety_margin: 0.1  # Reserve 10% VRAM

  execution:
    default_timeout: 300  # seconds
    enable_retry_on_failure: true
    max_retries_per_tool: 2

  tool_registry:
    auto_discover_tools: true
    tools_directory: "scripts/generation"
```

---

## Sub-Module 6: Function Calling Module

### Purpose

The **Function Calling Module** provides a **standardized interface** for the LLM to call Python functions with **type-safe parameters**, **automatic validation**, and **structured responses**.

### Architecture

```python
# scripts/ai_editing/agent/function_calling_module.py

from typing import List, Dict, Any, Optional, Callable, get_type_hints
from dataclasses import dataclass
from pydantic import BaseModel, create_model, ValidationError
import inspect
import json

@dataclass
class FunctionDefinition:
    """OpenAI-compatible function definition"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format

@dataclass
class FunctionCall:
    """Record of a function call"""
    function_name: str
    arguments: Dict[str, Any]
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str]
    timestamp: float

class FunctionRegistry:
    """
    Registry for callable functions with automatic schema generation

    Capabilities:
    - Auto-generate JSON schemas from Python type hints
    - Validate function arguments
    - Convert between Python and LLM function calling formats
    """

    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, FunctionDefinition] = {}

    def register_function(
        self,
        func: Callable,
        description: Optional[str] = None,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Register a function for LLM calling

        Args:
            func: Python function to register
            description: Function description (defaults to docstring)
            parameter_descriptions: Dict mapping param names to descriptions
        """
        # Get function metadata
        func_name = func.__name__
        func_description = description or func.__doc__ or "No description"

        # Get type hints
        type_hints = get_type_hints(func)
        signature = inspect.signature(func)

        # Generate JSON schema for parameters
        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, Any)
            param_schema = self._python_type_to_json_schema(param_type)

            # Add description if provided
            if parameter_descriptions and param_name in parameter_descriptions:
                param_schema["description"] = parameter_descriptions[param_name]

            properties[param_name] = param_schema

            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # Create function definition
        function_def = FunctionDefinition(
            name=func_name,
            description=func_description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )

        # Store
        self.functions[func_name] = func
        self.schemas[func_name] = function_def

    def _python_type_to_json_schema(self, python_type: type) -> Dict[str, Any]:
        """Convert Python type hint to JSON schema"""
        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"}
        }

        # Handle Optional types
        if hasattr(python_type, "__origin__"):
            if python_type.__origin__ is list:
                return {"type": "array"}
            elif python_type.__origin__ is dict:
                return {"type": "object"}

        return type_mapping.get(python_type, {"type": "string"})

    def get_function_definitions(self) -> List[FunctionDefinition]:
        """Get all function definitions in OpenAI format"""
        return list(self.schemas.values())

    def get_function(self, name: str) -> Optional[Callable]:
        """Get registered function by name"""
        return self.functions.get(name)

class FunctionCallingModule:
    """
    Function calling module with LLM integration

    Capabilities:
    - LLM decides which function to call
    - Automatic argument validation
    - Type-safe execution
    - Error handling and reporting
    """

    def __init__(
        self,
        llm_client: LLMClient,
        function_registry: FunctionRegistry,
        model: str = "qwen-14b"
    ):
        """
        Initialize Function Calling Module

        Args:
            llm_client: LLM client
            function_registry: Registry of callable functions
            model: LLM model to use
        """
        self.llm_client = llm_client
        self.function_registry = function_registry
        self.model = model
        self.call_history: List[FunctionCall] = []

    async def chat_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[str]] = None,
        function_call: str = "auto"
    ) -> Dict[str, Any]:
        """
        Chat with LLM with function calling enabled

        Args:
            messages: Conversation messages
            functions: Optional list of function names to make available (None = all)
            function_call: "auto" or "none" or {"name": "function_name"}

        Returns:
            LLM response with potential function call
        """
        # Get function definitions
        if functions:
            available_functions = [
                self.function_registry.schemas[name]
                for name in functions
                if name in self.function_registry.schemas
            ]
        else:
            available_functions = self.function_registry.get_function_definitions()

        # Convert to OpenAI format
        functions_schema = [
            {
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters
            }
            for func in available_functions
        ]

        # Call LLM with functions (Qwen2.5 supports function calling)
        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            functions=functions_schema if functions_schema else None,
            function_call=function_call,
            temperature=0.3
        )

        return response

    async def execute_function_call(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> FunctionCall:
        """
        Execute a function call

        Args:
            function_name: Name of function to call
            arguments: Function arguments

        Returns:
            FunctionCall record with results
        """
        func = self.function_registry.get_function(function_name)
        if not func:
            return FunctionCall(
                function_name=function_name,
                arguments=arguments,
                result=None,
                execution_time=0.0,
                success=False,
                error_message=f"Function {function_name} not found",
                timestamp=time.time()
            )

        # Execute function
        start_time = time.time()
        success = True
        error_message = None
        result = None

        try:
            # Handle async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

        except Exception as e:
            success = False
            error_message = str(e)

        execution_time = time.time() - start_time

        # Record call
        function_call = FunctionCall(
            function_name=function_name,
            arguments=arguments,
            result=result,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            timestamp=time.time()
        )

        self.call_history.append(function_call)

        return function_call

    async def run_conversation_with_functions(
        self,
        initial_message: str,
        max_turns: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Run a full conversation with function calling

        Args:
            initial_message: User's initial message
            max_turns: Maximum conversation turns

        Returns:
            Full conversation history
        """
        conversation_history = [
            {"role": "user", "content": initial_message}
        ]

        for turn in range(max_turns):
            # Get LLM response
            response = await self.chat_with_functions(
                messages=conversation_history
            )

            message = response["choices"][0]["message"]

            # Check if LLM wants to call a function
            if message.get("function_call"):
                function_call_data = message["function_call"]
                function_name = function_call_data["name"]
                arguments = json.loads(function_call_data["arguments"])

                # Execute function
                function_result = await self.execute_function_call(
                    function_name=function_name,
                    arguments=arguments
                )

                # Add function call to conversation
                conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call_data
                })

                # Add function result
                conversation_history.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps({
                        "success": function_result.success,
                        "result": function_result.result,
                        "error": function_result.error_message
                    })
                })

            else:
                # LLM provided final answer
                conversation_history.append({
                    "role": "assistant",
                    "content": message["content"]
                })
                break

        return conversation_history
```

### Usage Example

```python
# Example: Register and use functions with LLM

from scripts.ai_editing.agent.function_calling_module import (
    FunctionRegistry, FunctionCallingModule
)
from scripts.core.llm_client import LLMClient

# Define functions
async def get_character_info(character_name: str) -> Dict[str, Any]:
    """
    Retrieve character information from database

    Args:
        character_name: Name of the character

    Returns:
        Character information dict
    """
    # Mock implementation
    return {
        "name": character_name,
        "appearance": "brown hair, green eyes",
        "personality": "curious, adventurous"
    }

async def generate_character_image(
    character: str,
    scene_description: str,
    quality: str = "high"
) -> str:
    """
    Generate character image

    Args:
        character: Character name
        scene_description: Scene description
        quality: Quality level (draft, standard, high)

    Returns:
        Path to generated image
    """
    # Mock implementation
    return f"/outputs/images/{character}_{int(time.time())}.png"

# Register functions
async def main():
    registry = FunctionRegistry()

    registry.register_function(
        get_character_info,
        parameter_descriptions={
            "character_name": "Name of the character (e.g., 'luca', 'alberto')"
        }
    )

    registry.register_function(
        generate_character_image,
        parameter_descriptions={
            "character": "Character name",
            "scene_description": "Detailed scene description",
            "quality": "Quality level: draft, standard, or high"
        }
    )

    # Use with LLM
    async with LLMClient() as llm_client:
        function_calling = FunctionCallingModule(llm_client, registry)

        # Run conversation
        conversation = await function_calling.run_conversation_with_functions(
            initial_message="Generate an image of Luca running on the beach"
        )

        print("Conversation History:")
        for i, message in enumerate(conversation):
            print(f"\n{i+1}. {message.get('role', 'unknown')}")
            if message.get('content'):
                print(f"   Content: {message['content']}")
            if message.get('function_call'):
                print(f"   Function Call: {message['function_call']}")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/function_calling_config.yaml

function_calling_module:
  llm:
    model: "qwen-14b"  # Qwen2.5 supports function calling
    temperature: 0.3

  validation:
    strict_type_checking: true
    validate_before_execution: true

  execution:
    max_conversation_turns: 10
    timeout_per_function: 60  # seconds
```

---

## Sub-Module 7: Multi-Step Reasoning Module

### Purpose

The **Multi-Step Reasoning Module** enables the agent to execute **complex multi-step workflows** with **stateful context**, **dynamic planning**, and **quality-driven iteration**.

### Architecture

```python
# scripts/ai_editing/agent/multi_step_reasoning_module.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx

class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    """Single step in a multi-step workflow"""
    step_id: str
    description: str
    action: str  # Function/tool to execute
    parameters: Dict[str, Any]
    dependencies: List[str]  # step_ids that must complete first
    status: StepStatus
    result: Optional[Any]
    error_message: Optional[str]
    execution_time: float
    quality_score: float  # 0.0-1.0

@dataclass
class Workflow:
    """Complete multi-step workflow"""
    workflow_id: str
    goal: str
    steps: List[WorkflowStep]
    dependency_graph: nx.DiGraph
    current_step_idx: int
    status: str  # "running", "completed", "failed"
    total_execution_time: float
    final_result: Optional[Any]

class MultiStepReasoningModule:
    """
    Multi-step reasoning and workflow execution module

    Capabilities:
    - Plan multi-step workflows
    - Execute steps with dependency resolution
    - Maintain state across steps
    - Quality-driven iteration (retry if quality below threshold)
    - Dynamic re-planning based on intermediate results
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_calling_module: Any,  # ToolCallingModule
        function_calling_module: Any,  # FunctionCallingModule
        thinking_module: Any,  # ThinkingModule
        rag_module: Any,  # RAGUsageModule
        model: str = "qwen-14b"
    ):
        """
        Initialize Multi-Step Reasoning Module

        Args:
            llm_client: LLM client
            tool_calling_module: Tool calling module
            function_calling_module: Function calling module
            thinking_module: Thinking module for reflection
            rag_module: RAG module for context retrieval
            model: LLM model to use
        """
        self.llm_client = llm_client
        self.tool_calling = tool_calling_module
        self.function_calling = function_calling_module
        self.thinking = thinking_module
        self.rag = rag_module
        self.model = model

    async def plan_workflow(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        Plan a multi-step workflow to achieve goal

        Args:
            goal: High-level goal description
            context: Additional context

        Returns:
            Workflow with planned steps
        """
        # Use thinking module to decompose task
        decomposition = await self.thinking.decompose_task(
            intent=goal,
            parameters=context or {},
            available_tools=self.tool_calling.tool_registry.get_all_tools().keys()
        )

        # Convert to workflow steps
        steps = []
        for i, sub_task in enumerate(decomposition.sub_tasks):
            step = WorkflowStep(
                step_id=f"step_{i}",
                description=sub_task["description"],
                action=sub_task["action"],
                parameters=sub_task.get("parameters", {}),
                dependencies=sub_task.get("dependencies", []),
                status=StepStatus.PENDING,
                result=None,
                error_message=None,
                execution_time=0.0,
                quality_score=0.0
            )
            steps.append(step)

        # Build dependency graph
        dependency_graph = nx.DiGraph()
        for step in steps:
            dependency_graph.add_node(step.step_id)
            for dep in step.dependencies:
                dependency_graph.add_edge(dep, step.step_id)

        # Create workflow
        workflow = Workflow(
            workflow_id=f"workflow_{int(time.time())}",
            goal=goal,
            steps=steps,
            dependency_graph=dependency_graph,
            current_step_idx=0,
            status="planned",
            total_execution_time=0.0,
            final_result=None
        )

        return workflow

    async def execute_workflow(
        self,
        workflow: Workflow,
        quality_threshold: float = 0.7,
        max_iterations_per_step: int = 3
    ) -> Workflow:
        """
        Execute a planned workflow

        Args:
            workflow: Planned workflow
            quality_threshold: Minimum quality score to proceed (0.0-1.0)
            max_iterations_per_step: Maximum iterations per step if quality low

        Returns:
            Completed workflow with results
        """
        workflow.status = "running"
        start_time = time.time()

        # Get topologically sorted steps (respecting dependencies)
        try:
            execution_order = list(nx.topological_sort(workflow.dependency_graph))
        except nx.NetworkXError:
            # Cycle detected
            workflow.status = "failed"
            return workflow

        # Execute steps in order
        for step_id in execution_order:
            # Find step
            step = next(s for s in workflow.steps if s.step_id == step_id)

            # Check if dependencies completed successfully
            deps_completed = all(
                next(s for s in workflow.steps if s.step_id == dep_id).status == StepStatus.COMPLETED
                for dep_id in step.dependencies
            )

            if not deps_completed:
                step.status = StepStatus.SKIPPED
                continue

            # Execute step with quality-driven iteration
            step.status = StepStatus.IN_PROGRESS

            for iteration in range(max_iterations_per_step):
                step_start_time = time.time()

                # Execute action
                if step.action in self.tool_calling.tool_registry.get_all_tools():
                    # Tool call
                    tool_call = await self.tool_calling.execute_tool(
                        tool_name=step.action,
                        parameters=step.parameters
                    )
                    step.result = tool_call.output
                    step.error_message = tool_call.error_message
                    success = tool_call.success

                elif step.action in self.function_calling.function_registry.functions:
                    # Function call
                    func_call = await self.function_calling.execute_function_call(
                        function_name=step.action,
                        arguments=step.parameters
                    )
                    step.result = func_call.result
                    step.error_message = func_call.error_message
                    success = func_call.success

                else:
                    step.error_message = f"Action {step.action} not found"
                    success = False

                step.execution_time = time.time() - step_start_time

                # Evaluate quality
                if success:
                    step.quality_score = await self._evaluate_step_quality(
                        step=step,
                        goal=workflow.goal
                    )

                    # Check if quality sufficient
                    if step.quality_score >= quality_threshold:
                        step.status = StepStatus.COMPLETED
                        break
                    else:
                        # Quality too low, ask LLM how to improve
                        if iteration < max_iterations_per_step - 1:
                            step.parameters = await self._improve_step_parameters(
                                step=step,
                                quality_score=step.quality_score,
                                quality_threshold=quality_threshold
                            )
                        else:
                            # Max iterations reached, accept current quality
                            step.status = StepStatus.COMPLETED
                            break
                else:
                    # Step failed
                    step.status = StepStatus.FAILED
                    break

            # If step failed critically, abort workflow
            if step.status == StepStatus.FAILED and step.dependencies:
                workflow.status = "failed"
                break

        # Workflow complete
        workflow.total_execution_time = time.time() - start_time

        if all(s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] for s in workflow.steps):
            workflow.status = "completed"
            # Final result is the result of the last step
            workflow.final_result = workflow.steps[-1].result
        else:
            workflow.status = "failed"

        return workflow

    async def _evaluate_step_quality(
        self,
        step: WorkflowStep,
        goal: str
    ) -> float:
        """
        Evaluate quality of step result using LLM

        Args:
            step: Completed step
            goal: Overall workflow goal

        Returns:
            Quality score (0.0-1.0)
        """
        system_prompt = """You are a quality evaluation expert for animation AI workflows.

Evaluate the quality of a workflow step result on a scale of 0.0 to 1.0.

Consider:
- Does the result achieve the step's goal?
- Is the result relevant to the overall workflow goal?
- Is the quality of the result acceptable?

Respond with: {{"quality_score": 0.85, "reasoning": "..."}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Step Description: {step.description}
Step Result: {step.result}
Overall Goal: {goal}

Evaluate quality."""}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        evaluation = response["choices"][0]["message"]["content"]
        return evaluation.get("quality_score", 0.5)

    async def _improve_step_parameters(
        self,
        step: WorkflowStep,
        quality_score: float,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """
        Ask LLM to improve step parameters for better quality

        Args:
            step: Current step
            quality_score: Current quality score
            quality_threshold: Target quality threshold

        Returns:
            Improved parameters
        """
        system_prompt = """You are a parameter optimization expert for animation AI workflows.

A workflow step achieved quality score {quality_score} but needs {quality_threshold}.

Suggest improved parameters to increase quality.

Respond with complete improved parameter dict."""

        messages = [
            {"role": "system", "content": system_prompt.format(
                quality_score=quality_score,
                quality_threshold=quality_threshold
            )},
            {"role": "user", "content": f"""
Step: {step.description}
Action: {step.action}
Current Parameters: {step.parameters}
Current Result: {step.result}

Suggest improved parameters."""}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        improvements = response["choices"][0]["message"]["content"]
        return improvements.get("parameters", step.parameters)

    async def dynamic_replan(
        self,
        workflow: Workflow,
        current_step_idx: int,
        new_context: Dict[str, Any]
    ) -> Workflow:
        """
        Dynamically re-plan remaining steps based on intermediate results

        Args:
            workflow: Current workflow
            current_step_idx: Index of current step
            new_context: New information from completed steps

        Returns:
            Updated workflow with re-planned steps
        """
        # Get completed steps
        completed_steps = workflow.steps[:current_step_idx]

        # Ask LLM to re-plan remaining steps
        system_prompt = """You are a dynamic planning expert for animation AI workflows.

Given a workflow goal and results from completed steps, re-plan the remaining steps.

Consider:
- What has been achieved so far
- What remains to be done
- How to best achieve the remaining goals

Respond with list of remaining steps in JSON format."""

        completed_summary = "\n".join([
            f"{s.step_id}: {s.description} -> {s.result}"
            for s in completed_steps
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Goal: {workflow.goal}

Completed Steps:
{completed_summary}

New Context: {new_context}

Re-plan remaining steps."""}
        ]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        replanned_data = response["choices"][0]["message"]["content"]

        # Create new steps
        new_steps = []
        for i, step_data in enumerate(replanned_data.get("steps", [])):
            new_step = WorkflowStep(
                step_id=f"step_{current_step_idx + i}",
                description=step_data["description"],
                action=step_data["action"],
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                status=StepStatus.PENDING,
                result=None,
                error_message=None,
                execution_time=0.0,
                quality_score=0.0
            )
            new_steps.append(new_step)

        # Update workflow
        workflow.steps = completed_steps + new_steps

        # Rebuild dependency graph
        workflow.dependency_graph = nx.DiGraph()
        for step in workflow.steps:
            workflow.dependency_graph.add_node(step.step_id)
            for dep in step.dependencies:
                workflow.dependency_graph.add_edge(dep, step.step_id)

        return workflow
```

### Usage Example

```python
# Example: Execute multi-step creative workflow

from scripts.ai_editing.agent.multi_step_reasoning_module import MultiStepReasoningModule
from scripts.core.llm_client import LLMClient

async def main():
    # Initialize modules (simplified)
    async with LLMClient() as llm_client:
        multi_step = MultiStepReasoningModule(
            llm_client=llm_client,
            tool_calling_module=tool_calling,
            function_calling_module=function_calling,
            thinking_module=thinking,
            rag_module=rag
        )

        # Plan workflow
        workflow = await multi_step.plan_workflow(
            goal="Create a scene with Luca and Alberto talking on the beach at sunset",
            context={
                "characters": ["luca", "alberto"],
                "setting": "beach",
                "time": "sunset",
                "action": "dialogue"
            }
        )

        print(f"Planned workflow with {len(workflow.steps)} steps:")
        for step in workflow.steps:
            print(f"  {step.step_id}: {step.description}")

        # Execute workflow
        completed_workflow = await multi_step.execute_workflow(
            workflow=workflow,
            quality_threshold=0.8,
            max_iterations_per_step=2
        )

        print(f"\nWorkflow {completed_workflow.status} in {completed_workflow.total_execution_time:.2f}s")

        for step in completed_workflow.steps:
            print(f"\n{step.step_id} ({step.status}):")
            print(f"  Quality: {step.quality_score:.2f}")
            print(f"  Time: {step.execution_time:.2f}s")
            if step.result:
                print(f"  Result: {step.result}")

asyncio.run(main())
```

### Configuration

```yaml
# configs/agent/multi_step_reasoning_config.yaml

multi_step_reasoning_module:
  llm:
    model: "qwen-14b"
    temperature: 0.5

  execution:
    quality_threshold: 0.7  # Minimum quality to proceed
    max_iterations_per_step: 3
    enable_dynamic_replanning: true

  quality_evaluation:
    use_llm_evaluation: true
    evaluation_model: "qwen-14b"
    evaluation_temperature: 0.3
```

---

## LLM Decision-Making Engine

### Architecture

The **LLM Decision-Making Engine** is the central coordinator that integrates all sub-modules and makes high-level creative decisions.

```python
# scripts/ai_editing/agent/decision_engine.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Decision:
    """LLM decision record"""
    decision_type: str  # "tool_selection", "quality_assessment", "strategy", etc.
    decision: Any
    reasoning: str
    confidence: float
    timestamp: float

class LLMDecisionEngine:
    """
    Central LLM-powered decision-making engine

    Integrates all agent sub-modules and makes high-level creative decisions
    """

    def __init__(
        self,
        llm_client: LLMClient,
        thinking_module: Any,
        reasoning_module: Any,
        web_search_module: Any,
        rag_module: Any,
        tool_calling_module: Any,
        function_calling_module: Any,
        multi_step_module: Any,
        model: str = "qwen-14b"
    ):
        """Initialize Decision Engine with all sub-modules"""
        self.llm_client = llm_client
        self.thinking = thinking_module
        self.reasoning = reasoning_module
        self.web_search = web_search_module
        self.rag = rag_module
        self.tool_calling = tool_calling_module
        self.function_calling = function_calling_module
        self.multi_step = multi_step_module
        self.model = model
        self.decision_history: List[Decision] = []

    async def process_user_request(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Process user request through full agent framework

        This is the main entry point for the agent

        Flow:
        1. Understand intent (Thinking Module)
        2. Retrieve context (RAG + Web Search if needed)
        3. Plan execution (Multi-Step Reasoning)
        4. Execute with quality iteration
        5. Return final result

        Args:
            user_request: User's request
            context: Optional additional context

        Returns:
            Final result (image path, audio path, video path, etc.)
        """
        # 1. Understand intent
        intent_data = await self.thinking.understand_intent(
            user_request=user_request,
            context=context
        )

        # 2. Retrieve context
        # 2a. RAG retrieval
        rag_results = []
        if "character" in intent_data.get("parameters", {}):
            rag_results.extend(
                await self.rag.retrieve_character_info(
                    character_name=intent_data["parameters"]["character"]
                )
            )

        # 2b. Web search (if needed for latest info)
        if intent_data.get("requires_web_search", False):
            web_results = await self.web_search.search(
                query=intent_data.get("search_query", user_request)
            )
            # Synthesize web info
            web_context = await self.web_search.synthesize_information(
                query=user_request,
                search_results=web_results,
                web_contents=[]
            )

        # 3. Plan workflow
        workflow = await self.multi_step.plan_workflow(
            goal=intent_data["intent"],
            context=intent_data.get("parameters", {})
        )

        # 4. Execute workflow
        completed_workflow = await self.multi_step.execute_workflow(
            workflow=workflow,
            quality_threshold=0.8,
            max_iterations_per_step=2
        )

        # 5. Return final result
        if completed_workflow.status == "completed":
            return completed_workflow.final_result
        else:
            raise Exception(f"Workflow failed: {completed_workflow.steps[-1].error_message}")
```

---

## Integration Examples

### Example 1: Complete Creative Workflow

```python
# Example: Full agent-powered character image generation

from scripts.ai_editing.agent.decision_engine import LLMDecisionEngine
from scripts.core.llm_client import LLMClient

async def main():
    # Initialize all modules
    async with LLMClient() as llm_client:
        # ... initialize all sub-modules ...

        # Create decision engine
        decision_engine = LLMDecisionEngine(
            llm_client=llm_client,
            thinking_module=thinking,
            reasoning_module=reasoning,
            web_search_module=web_search,
            rag_module=rag,
            tool_calling_module=tool_calling,
            function_calling_module=function_calling,
            multi_step_module=multi_step
        )

        # User request
        result = await decision_engine.process_user_request(
            user_request="Create a high-quality scene of Luca and Alberto racing on Vespas through Portorosso"
        )

        print(f"Final result: {result}")

asyncio.run(main())
```

---

## Configuration

Complete configuration file for the entire Agent Framework:

```yaml
# configs/agent/agent_framework_config.yaml

agent_framework:
  # LLM configuration
  llm:
    primary_model: "qwen-14b"  # Main decision-making model
    vision_model: "qwen-vl-7b"  # For multimodal tasks
    coder_model: "qwen-coder-7b"  # For code generation
    temperature: 0.5
    max_tokens: 2048

  # Sub-modules
  thinking_module:
    model: "qwen-14b"
    temperature: 0.7
    max_context_messages: 10
    confidence_threshold: 0.7

  reasoning_module:
    default_strategy: "react"
    max_react_iterations: 10
    max_cot_steps: 15
    enable_adaptive_strategy: true

  web_search_module:
    search_engine: "duckduckgo"
    max_results: 5
    timeout: 10

  rag_module:
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    top_k: 5
    rerank_with_llm: true

  tool_calling_module:
    max_concurrent_tools: 1  # RTX 5080 16GB
    enable_retry: true
    max_retries: 2

  function_calling_module:
    strict_validation: true
    max_conversation_turns: 10

  multi_step_reasoning_module:
    quality_threshold: 0.7
    max_iterations_per_step: 3
    enable_dynamic_replanning: true

  # Decision engine
  decision_engine:
    enable_web_search: true
    enable_rag: true
    default_quality_threshold: 0.8
    max_workflow_steps: 20
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Set up agent directory structure (`scripts/ai_editing/agent/`)
- [ ] Create base classes and data structures
- [ ] Implement LangGraph integration
- [ ] Set up configuration system

### Phase 2: Individual Sub-Modules
- [ ] Implement Thinking Module
  - [ ] Intent understanding
  - [ ] Task decomposition
  - [ ] Reflection capabilities
- [ ] Implement Reasoning Module
  - [ ] ReAct reasoner
  - [ ] Chain-of-Thought reasoner
  - [ ] Adaptive strategy selection
- [ ] Implement Web Search Module
  - [ ] DuckDuckGo integration
  - [ ] Content extraction
  - [ ] LLM-powered synthesis
- [ ] Implement RAG Usage Module
  - [ ] ChromaDB integration
  - [ ] Character/style retrieval
  - [ ] Context synthesis
- [ ] Implement Tool Calling Module
  - [ ] Tool registry
  - [ ] LLM-powered tool selection
  - [ ] Hardware-aware execution
- [ ] Implement Function Calling Module
  - [ ] Function registry with schema generation
  - [ ] Type-safe execution
  - [ ] Conversation with functions
- [ ] Implement Multi-Step Reasoning Module
  - [ ] Workflow planning
  - [ ] Quality-driven iteration
  - [ ] Dynamic re-planning

### Phase 3: Integration
- [ ] Implement LLM Decision Engine
- [ ] Create end-to-end workflows
- [ ] Integration testing with real tasks

### Phase 4: Testing & Optimization
- [ ] Unit tests for each sub-module
- [ ] Integration tests for workflows
- [ ] Performance optimization
- [ ] VRAM usage optimization
- [ ] Error handling and recovery

### Phase 5: Documentation & Examples
- [ ] Usage examples for each module
- [ ] End-to-end workflow tutorials
- [ ] Best practices guide
- [ ] Troubleshooting guide

---

**Version:** v0.1.0
**Last Updated:** 2025-11-17
**Status:** 📋 PLANNED

This completes the comprehensive Agent Framework architecture documentation with all 7 sub-modules, LLM decision-making engine, integration examples, and implementation checklist.
