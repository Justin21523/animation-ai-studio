# Agent Framework Documentation

**Module 6: Agent Framework (Phase 1 + Phase 2 Complete)**

Advanced LLM-driven agent system for autonomous task execution with multiple reasoning strategies and tool integration.

---

## Overview

The Agent Framework is a comprehensive system that enables autonomous task execution through:

- **Intelligent decision-making** using advanced reasoning strategies  
- **Knowledge retrieval** via RAG system integration
- **Tool orchestration** for complex multi-step workflows
- **Quality-driven iteration** with automatic retry logic
- **Transparent reasoning** with complete execution traces

### Key Components

**Phase 1: Core Infrastructure**
- Thinking Module: Intent understanding, task decomposition, reflection
- RAG Usage Module: Knowledge retrieval and context management

**Phase 2: Advanced Reasoning & Execution**
- Reasoning Module: ReAct, Chain-of-Thought, Tree-of-Thoughts
- Tool Calling Module: LLM-powered tool selection and execution  
- Function Calling Module: Type-safe function calling with auto-schema
- Multi-Step Module: Stateful workflow execution with quality checks

---

## Quick Start

### Basic Usage

\`\`\`python
from scripts.agent import Agent

async with Agent() as agent:
    response = await agent.process(
        "Tell me about Luca's personality"
    )
    print(response.content)
\`\`\`

### Advanced Usage (Phase 2)

\`\`\`python
from scripts.agent import Agent, AgentConfig, ReasoningStrategy

config = AgentConfig(
    enable_tool_calling=True,
    enable_multi_step=True,
    default_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
)

async with Agent(config=config) as agent:
    response = await agent.process_advanced(
        "Generate an animated scene of Luca on the beach",
        strategy=ReasoningStrategy.TREE_OF_THOUGHTS
    )
\`\`\`

---

## Module Details

See individual module documentation:

- `scripts/agent/thinking/` - Intent understanding and task decomposition
- `scripts/agent/rag_usage/` - Knowledge retrieval integration  
- `scripts/agent/reasoning/` - Advanced reasoning strategies
- `scripts/agent/tools/` - Tool calling and registry
- `scripts/agent/functions/` - Function calling with type safety
- `scripts/agent/multi_step/` - Multi-step workflow execution

---

## Testing

Run comprehensive test suite:

\`\`\`bash
python scripts/agent/test_agent_phase2.py
\`\`\`

---

## Version

**v1.0.0** - Phase 1 + Phase 2 Complete (2025-11-17)

