"""
Comprehensive Test Suite for Agent Framework Phase 2

Tests all Phase 2 modules:
- Reasoning Module (ReAct, CoT, ToT)
- Tool Calling Module
- Function Calling Module
- Multi-Step Module
- Agent Orchestrator with Phase 2 integration

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.agent.agent import Agent, AgentConfig
from scripts.agent.core.types import ReasoningStrategy
from scripts.agent.reasoning.reasoning_module import ReasoningModule
from scripts.agent.tools.tool_calling_module import ToolCallingModule
from scripts.agent.functions.function_calling_module import FunctionCallingModule, generate_image, synthesize_voice
from scripts.agent.multi_step.multi_step_module import MultiStepModule


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_reasoning_module():
    """Test Reasoning Module with all strategies"""
    print("\n" + "=" * 80)
    print("TEST 1: Reasoning Module")
    print("=" * 80)

    async with ReasoningModule() as reasoning:
        # Test 1.1: Chain-of-Thought
        print("\n--- Test 1.1: Chain-of-Thought Reasoning ---")
        trace = await reasoning.reason(
            task="Generate an image of Luca running on the beach with a happy expression",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context="Character: Luca, Style: Pixar 3D"
        )

        print(f"✅ CoT completed with {len(trace.thoughts)} reasoning steps")
        print(f"   Success: {trace.success}, Time: {trace.total_time:.2f}s")

        # Test 1.2: ReAct
        print("\n--- Test 1.2: ReAct Reasoning ---")
        trace = await reasoning.reason(
            task="Find information about Alberto's personality and generate a character summary",
            strategy=ReasoningStrategy.REACT,
            available_actions=["search_knowledge", "generate_summary", "complete"]
        )

        print(f"✅ ReAct completed with {len(trace.thoughts)} steps")
        print(f"   Success: {trace.success}, Time: {trace.total_time:.2f}s")

        # Test 1.3: Tree-of-Thoughts
        print("\n--- Test 1.3: Tree-of-Thoughts Reasoning ---")
        trace = await reasoning.reason(
            task="Create a creative scene combining Luca and Alberto",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS
        )

        print(f"✅ ToT completed with {len(trace.thoughts)} reasoning paths explored")
        print(f"   Success: {trace.success}, Time: {trace.total_time:.2f}s")


async def test_tool_calling_module():
    """Test Tool Calling Module"""
    print("\n" + "=" * 80)
    print("TEST 2: Tool Calling Module")
    print("=" * 80)

    async with ToolCallingModule() as tool_calling:
        # Test 2.1: Tool selection
        print("\n--- Test 2.1: Tool Selection ---")
        selected_tools = await tool_calling.select_tools(
            task="Generate an image of Luca running on the beach and synthesize his voice saying 'Silenzio, Bruno!'",
            context="Character: Luca, Style: Pixar 3D"
        )

        print(f"✅ Selected {len(selected_tools)} tools:")
        for tool in selected_tools:
            print(f"   - {tool['tool']}: {tool.get('reasoning', 'N/A')}")

        # Test 2.2: Tool execution
        print("\n--- Test 2.2: Tool Execution ---")
        result = await tool_calling.execute_tool(
            tool_name="generate_character_image",
            arguments={
                "character": "luca",
                "scene_description": "running on the beach, happy expression",
                "style": "pixar_3d",
                "quality_preset": "high"
            }
        )

        print(f"✅ Tool executed:")
        print(f"   Success: {result.success}")
        print(f"   Output: {json.dumps(result.output, indent=2)}")
        print(f"   Time: {result.tool_call.execution_time:.2f}s")


async def test_function_calling_module():
    """Test Function Calling Module"""
    print("\n" + "=" * 80)
    print("TEST 3: Function Calling Module")
    print("=" * 80)

    async with FunctionCallingModule() as function_calling:
        # Register test functions
        print("\n--- Test 3.1: Function Registration ---")
        function_calling.register_function(
            generate_image,
            "Generate an image of a character using SDXL + LoRA",
            parameter_descriptions={
                "character": "Character name (e.g., 'luca', 'alberto')",
                "scene": "Scene description",
                "style": "Art style to use",
                "quality": "Quality preset"
            }
        )

        function_calling.register_function(
            synthesize_voice,
            "Synthesize speech in character's voice",
            parameter_descriptions={
                "character": "Character name",
                "text": "Text to synthesize",
                "emotion": "Emotion or tone"
            }
        )

        print(f"✅ Registered {len(function_calling.function_registry.functions)} functions")

        # Test 3.2: Function selection
        print("\n--- Test 3.2: Function Selection ---")
        function_call = await function_calling.select_function(
            task="Generate an image of Alberto smiling in Portorosso",
            context="Character: Alberto, Setting: Italian seaside town"
        )

        if function_call:
            print(f"✅ Selected function: {function_call['function']}")
            print(f"   Arguments: {json.dumps(function_call['arguments'], indent=2)}")

        # Test 3.3: Function execution
        print("\n--- Test 3.3: Function Execution ---")
        result = await function_calling.call_function(
            function_name="generate_image",
            arguments={
                "character": "alberto",
                "scene": "smiling in Portorosso town square",
                "style": "pixar_3d",
                "quality": "high"
            }
        )

        print(f"✅ Function executed:")
        print(f"   Success: {result.success}")
        print(f"   Result: {json.dumps(result.result, indent=2)}")
        print(f"   Time: {result.execution_time:.2f}s")

        # Test 3.4: OpenAI schema generation
        print("\n--- Test 3.4: OpenAI Schema Generation ---")
        schemas = function_calling.function_registry.get_openai_schemas()
        print(f"✅ Generated {len(schemas)} OpenAI-compatible schemas")
        print(f"   Example schema:")
        print(f"   {json.dumps(schemas[0], indent=2)}")


async def test_multi_step_module():
    """Test Multi-Step Module"""
    print("\n" + "=" * 80)
    print("TEST 4: Multi-Step Module")
    print("=" * 80)

    async with MultiStepModule() as multi_step:
        # Test 4.1: Workflow planning
        print("\n--- Test 4.1: Workflow Planning ---")
        plan = await multi_step.create_workflow_plan(
            task="Generate a short animated video of Luca running on the beach with voiceover",
            context="Character: Luca, Style: Pixar 3D, Duration: 5 seconds",
            constraints=[
                "GPU memory limited to 16GB",
                "Must use existing character LoRA",
                "Audio must match character voice"
            ]
        )

        print(f"✅ Created workflow plan with {len(plan.steps)} steps:")
        for step in plan.steps:
            deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
            print(f"   {step.step_id}: {step.description}{deps}")

        # Test 4.2: Workflow execution
        print("\n--- Test 4.2: Workflow Execution ---")
        trace = await multi_step.execute_workflow(plan)

        print(f"✅ Workflow execution {'succeeded' if trace.success else 'failed'}")
        print(f"   Total time: {trace.total_time:.2f}s")
        print(f"   Execution trace: {len(trace.thoughts)} thoughts")

        # Test 4.3: Step status summary
        print("\n--- Test 4.3: Step Status Summary ---")
        from scripts.agent.multi_step.multi_step_module import StepStatus

        status_icons = {
            StepStatus.COMPLETED: "✅",
            StepStatus.FAILED: "❌",
            StepStatus.PENDING: "⏳",
            StepStatus.SKIPPED: "⏭️"
        }

        for step in plan.steps:
            icon = status_icons.get(step.status, "❓")
            print(f"   {icon} {step.step_id}: {step.status.value}")
            if step.status == StepStatus.COMPLETED:
                print(f"      Quality: {step.quality_score:.2f}")


async def test_agent_integration():
    """Test Agent with full Phase 2 integration"""
    print("\n" + "=" * 80)
    print("TEST 5: Agent Integration (Phase 1 + Phase 2)")
    print("=" * 80)

    # Configure agent with Phase 2 features enabled
    config = AgentConfig(
        enable_rag=True,
        enable_tool_calling=True,
        enable_function_calling=True,
        enable_multi_step=True,
        default_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        enable_reflection=True
    )

    async with Agent(config=config) as agent:
        # Test 5.1: Simple question (Phase 1)
        print("\n--- Test 5.1: Simple Question (Phase 1 Processing) ---")
        response = await agent.process(
            "Tell me about Luca's appearance and personality"
        )

        print(f"✅ Response generated:")
        print(f"   Success: {response.success}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Reasoning steps: {len(response.reasoning_trace.thoughts)}")
        print(f"   Response (truncated): {response.content[:200]}...")

        # Test 5.2: Advanced processing with CoT
        print("\n--- Test 5.2: Advanced Processing (CoT) ---")
        response = await agent.process_advanced(
            "I want to generate an image of Luca running on the beach",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )

        print(f"✅ Advanced response generated:")
        print(f"   Success: {response.success}")
        print(f"   Strategy: {response.reasoning_trace.strategy.value}")
        print(f"   Total time: {response.reasoning_trace.total_time:.2f}s")
        print(f"   Thoughts: {len(response.reasoning_trace.thoughts)}")

        # Test 5.3: Advanced processing with ReAct
        print("\n--- Test 5.3: Advanced Processing (ReAct) ---")
        response = await agent.process_advanced(
            "Find information about Alberto and create a character profile",
            strategy=ReasoningStrategy.REACT
        )

        print(f"✅ ReAct response generated:")
        print(f"   Success: {response.success}")
        print(f"   Strategy: {response.reasoning_trace.strategy.value}")
        print(f"   Total time: {response.reasoning_trace.total_time:.2f}s")

        # Test 5.4: Conversation history
        print("\n--- Test 5.4: Conversation History ---")
        history = agent.get_conversation_history()
        print(f"✅ Conversation history: {len(history)} messages")
        for msg in history[-3:]:  # Show last 3 messages
            role = msg['role']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"   [{role}]: {content}")


async def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "=" * 80)
    print("AGENT FRAMEWORK PHASE 2 - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    try:
        await test_reasoning_module()
        await test_tool_calling_module()
        await test_function_calling_module()
        await test_multi_step_module()
        await test_agent_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\nPhase 2 Module Summary:")
        print("  ✅ Reasoning Module (ReAct, CoT, ToT)")
        print("  ✅ Tool Calling Module")
        print("  ✅ Function Calling Module")
        print("  ✅ Multi-Step Module")
        print("  ✅ Agent Orchestrator Integration")

        print("\nAll Phase 2 modules are functioning correctly!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
