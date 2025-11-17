# Agent Framework

**Status:** 🔄 Phase 1 Complete (30%)
**Last Updated:** 2025-11-17

---

## Overview

LLM-powered autonomous agent framework for creative decision-making and task execution.

**Phase 1 Implementation (COMPLETE):**
- ✅ Core types and data structures
- ✅ Thinking Module (intent understanding, task decomposition)
- ✅ RAG Usage Module (knowledge retrieval)
- ✅ Agent Orchestrator (main workflow)

**Future Phases:**
- 📋 Reasoning Module (ReAct, CoT, ToT)
- 📋 Tool Calling Module
- 📋 Function Calling Module
- 📋 Multi-Step Reasoning Module
- 📋 Web Search Module

---

## Quick Start

### Basic Usage

```python
import asyncio
from scripts.agent import Agent

async def main():
    async with Agent() as agent:
        # Ask a question
        response = await agent.process(
            "Tell me about Luca's personality"
        )

        print(response.content)
        print(f"Confidence: {response.confidence}")

asyncio.run(main())
```

### With Context

```python
async with Agent() as agent:
    response = await agent.process(
        "Generate an image of Luca running on the beach",
        context={"style": "pixar_3d", "quality": "high"}
    )

    print(response.content)

    # View reasoning trace
    for thought in response.reasoning_trace.thoughts:
        print(f"[{thought.thought_type}] {thought.content}")
```

---

## Components

### 1. Core Types (`core/types.py`)

Base data structures:
- `Task` - Represents a task
- `Thought` - Single reasoning thought
- `AgentState` - Agent state and conversation history
- `AgentResponse` - Agent's response
- `ReasoningTrace` - Complete reasoning record

### 2. Thinking Module (`thinking/thinking_module.py`)

Intent understanding and task decomposition:
- `understand_intent()` - Analyze user request
- `decompose_task()` - Break down complex tasks
- `reflect_on_result()` - Reflect on results

### 3. RAG Usage Module (`rag_usage/rag_module.py`)

Knowledge retrieval:
- `retrieve_character_knowledge()` - Get character info
- `retrieve_style_guide()` - Get style info
- `retrieve_technical_parameters()` - Get technical params
- `answer_question()` - Q&A with sources

### 4. Agent Orchestrator (`agent.py`)

Main agent coordinator:
- `process()` - Main entry point
- Coordinates all modules
- Manages conversation state
- Tracks reasoning trace

---

## Architecture

```
User Request
     ↓
[1] Thinking Module
    - Understand intent
    - Extract entities
    - Assess confidence
     ↓
[2] RAG Module
    - Retrieve character knowledge
    - Retrieve style guides
    - Retrieve technical params
     ↓
[3] Generate Response
    - Build context-aware prompt
    - Query LLM
    - Format response
     ↓
[4] Reflection (optional)
    - Reflect on quality
    - Suggest improvements
     ↓
Agent Response
```

---

## Configuration

See `configs/agent/agent_config.yaml`

```yaml
llm:
  model: "qwen-14b"
  temperature: 0.7

rag:
  enabled: true
  top_k: 5

reasoning:
  default_strategy: "direct"
  enable_reflection: true
```

---

## Phase Roadmap

### ✅ Phase 1 (COMPLETE)
- Core types and infrastructure
- Thinking Module
- RAG Usage Module
- Basic agent orchestrator
- Intent understanding
- Knowledge retrieval
- Simple reasoning trace

### 📋 Phase 2 (Planned)
- Reasoning Module (ReAct, CoT, ToT)
- Tool Calling Module
- Function Calling Module
- Multi-step workflows
- Quality-driven iteration

### 📋 Phase 3 (Planned)
- Web Search Module
- Advanced reasoning strategies
- Parallel task execution
- Self-improvement mechanisms
- Performance optimization

---

## Integration

### With RAG System (Module 5)

```python
from scripts.rag import KnowledgeBase
from scripts.agent import Agent

async with KnowledgeBase() as kb:
    async with Agent(knowledge_base=kb) as agent:
        # Agent has access to full knowledge base
        response = await agent.process("...")
```

### With Image Generation (Module 2)

```python
# Phase 2: Agent will call image generation tools
async with Agent() as agent:
    response = await agent.process(
        "Generate an image of Luca"
    )
    # Agent will plan and execute image generation
```

---

## Files

```
scripts/agent/
├── __init__.py
├── agent.py (Agent orchestrator, 350 lines)
├── core/
│   ├── __init__.py
│   └── types.py (Core types, 350 lines)
├── thinking/
│   ├── __init__.py
│   └── thinking_module.py (Thinking, 450 lines)
├── rag_usage/
│   ├── __init__.py
│   └── rag_module.py (RAG usage, 300 lines)
├── reasoning/ (Phase 2)
├── tools/ (Phase 2)
├── functions/ (Phase 2)
├── multi_step/ (Phase 2)
└── README.md

Total Phase 1: ~1,450 lines
```

---

## Next Steps

1. Implement Reasoning Module (ReAct, CoT, ToT)
2. Implement Tool Calling Module
3. Implement Function Calling Module
4. Add comprehensive testing
5. Complete documentation

---

**Version:** v0.1.0 (Phase 1)
**Status:** 🔄 Phase 1 Complete (30% of full framework)
