"""
Tool Calling Module for Agent Framework

LLM-powered tool selection and execution.

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
import json
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient
from scripts.agent.core.types import ToolCall
from scripts.agent.tools.tool_registry import (
    ToolRegistry,
    Tool,
    ToolCategory,
    get_default_registry
)


logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    tool_call: ToolCall
    success: bool
    output: Any
    error: Optional[str] = None


class ToolCallingModule:
    """
    Tool Calling Module

    Handles:
    - Tool selection (LLM decides which tool to use)
    - Argument extraction
    - Tool execution
    - Error handling and retry logic
    - Hardware-aware scheduling (GPU constraints)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        Initialize tool calling module

        Args:
            llm_client: LLM client (will create if not provided)
            tool_registry: Tool registry (will use default if not provided)
        """
        self._llm_client = llm_client
        self._own_client = llm_client is None

        self.tool_registry = tool_registry or get_default_registry()

        logger.info(f"ToolCallingModule initialized with {len(self.tool_registry.tools)} tools")

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

    async def select_tools(
        self,
        task: str,
        context: Optional[str] = None,
        available_tools: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select appropriate tools for task

        Args:
            task: Task description
            context: Additional context
            available_tools: List of available tool names (None = all tools)

        Returns:
            List of selected tools with arguments
        """
        # Get available tools
        if available_tools:
            tools = [self.tool_registry.get_tool(name) for name in available_tools]
            tools = [t for t in tools if t is not None]
        else:
            tools = self.tool_registry.get_all_tools()

        # Build tool selection prompt
        prompt = self._build_tool_selection_prompt(task, context, tools)

        # Query LLM
        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        # Parse tool selections
        selected_tools = self._parse_tool_selection_response(response["content"])

        logger.info(f"Selected {len(selected_tools)} tools for task")
        return selected_tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolExecutionResult:
        """
        Execute a tool

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            ToolExecutionResult
        """
        start_time = time.time()

        tool = self.tool_registry.get_tool(tool_name)

        if tool is None:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            return ToolExecutionResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    error=error_msg
                ),
                success=False,
                output=None,
                error=error_msg
            )

        # Validate arguments
        validation_error = self._validate_arguments(tool, arguments)
        if validation_error:
            logger.error(f"Argument validation failed: {validation_error}")
            return ToolExecutionResult(
                tool_call=ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    error=validation_error
                ),
                success=False,
                output=None,
                error=validation_error
            )

        # Execute tool
        logger.info(f"Executing tool: {tool_name}")

        try:
            # In real implementation, this would call the actual tool function
            # For now, we simulate execution
            output = await self._execute_tool_function(tool, arguments)

            execution_time = time.time() - start_time

            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=output,
                execution_time=execution_time
            )

            logger.info(f"Tool executed successfully in {execution_time:.2f}s")

            return ToolExecutionResult(
                tool_call=tool_call,
                success=True,
                output=output
            )

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)

            execution_time = time.time() - start_time

            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                error=error_msg,
                execution_time=execution_time
            )

            return ToolExecutionResult(
                tool_call=tool_call,
                success=False,
                output=None,
                error=error_msg
            )

    async def execute_tools_sequence(
        self,
        tool_selections: List[Dict[str, Any]]
    ) -> List[ToolExecutionResult]:
        """
        Execute multiple tools in sequence

        Args:
            tool_selections: List of tool selections with arguments

        Returns:
            List of execution results
        """
        results = []

        for selection in tool_selections:
            tool_name = selection["tool"]
            arguments = selection["arguments"]

            result = await self.execute_tool(tool_name, arguments)
            results.append(result)

            # Stop if execution failed (unless we want to continue despite errors)
            if not result.success:
                logger.warning(f"Tool execution failed: {tool_name}, stopping sequence")
                break

        return results

    def _build_tool_selection_prompt(
        self,
        task: str,
        context: Optional[str],
        tools: List[Tool]
    ) -> str:
        """Build prompt for tool selection"""
        # Format tools
        tools_json = json.dumps([tool.to_dict() for tool in tools], indent=2)

        prompt = f"""You are an AI agent that selects appropriate tools to complete tasks.

Task: {task}

Context: {context or "None"}

Available tools:
{tools_json}

Select the appropriate tool(s) and specify arguments. Respond in JSON format:
{{
  "selected_tools": [
    {{
      "tool": "tool_name",
      "arguments": {{
        "param1": "value1",
        "param2": "value2"
      }},
      "reasoning": "why this tool is needed"
    }}
  ]
}}

Select tools:"""

        return prompt

    def _parse_tool_selection_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into tool selections"""
        try:
            data = json.loads(response)
            selected_tools = data.get("selected_tools", [])

            # Validate format
            valid_tools = []
            for tool_selection in selected_tools:
                if "tool" in tool_selection and "arguments" in tool_selection:
                    valid_tools.append(tool_selection)

            return valid_tools

        except json.JSONDecodeError:
            logger.warning("Failed to parse tool selection JSON")
            return []

    def _validate_arguments(self, tool: Tool, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate tool arguments"""
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return f"Missing required parameter: {param.name}"

        # Type validation (basic)
        for param in tool.parameters:
            if param.name in arguments:
                value = arguments[param.name]

                # Check enum
                if param.enum and value not in param.enum:
                    return f"Invalid value for {param.name}: {value}, must be one of {param.enum}"

        return None

    async def _execute_tool_function(
        self,
        tool: Tool,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute the actual tool function

        This is where the real tool execution happens.
        In a complete implementation, this would:
        1. Load the appropriate module (image gen, voice synthesis, etc.)
        2. Call the actual function with arguments
        3. Return the result

        For now, we return a placeholder.
        """
        # Placeholder implementation
        if tool.category == ToolCategory.IMAGE_GENERATION:
            # Would call: from scripts.generation.image import CharacterGenerator
            # generator.generate_character(**arguments)
            return {
                "status": "success",
                "message": f"Would generate image with: {arguments}",
                "output_path": "outputs/generated_image.png"
            }

        elif tool.category == ToolCategory.VOICE_SYNTHESIS:
            # Would call: from scripts.synthesis.tts import GPTSoVITSWrapper
            # synthesizer.synthesize(**arguments)
            return {
                "status": "success",
                "message": f"Would synthesize voice with: {arguments}",
                "output_path": "outputs/synthesized_audio.wav"
            }

        elif tool.category == ToolCategory.KNOWLEDGE_RETRIEVAL:
            # Would call: RAG system
            return {
                "status": "success",
                "message": f"Would retrieve knowledge with: {arguments}",
                "results": ["Sample result 1", "Sample result 2"]
            }

        else:
            return {"status": "success", "message": "Tool executed (placeholder)"}


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async with ToolCallingModule() as tool_calling:
        # Example 1: Select tools for image generation
        print("\n" + "=" * 60)
        print("Example 1: Tool Selection")
        print("=" * 60)

        selected_tools = await tool_calling.select_tools(
            task="Generate an image of Luca running on the beach with a happy expression",
            context="Character: Luca, Style: Pixar 3D, Quality: High"
        )

        print(f"\nSelected {len(selected_tools)} tools:")
        for selection in selected_tools:
            print(f"  Tool: {selection['tool']}")
            print(f"  Arguments: {json.dumps(selection['arguments'], indent=4)}")
            print(f"  Reasoning: {selection.get('reasoning', 'N/A')}")

        # Example 2: Execute tool
        print("\n" + "=" * 60)
        print("Example 2: Tool Execution")
        print("=" * 60)

        result = await tool_calling.execute_tool(
            tool_name="generate_character_image",
            arguments={
                "character": "luca",
                "scene_description": "running on the beach, happy expression",
                "style": "pixar_3d",
                "quality_preset": "high"
            }
        )

        print(f"\nExecution result:")
        print(f"  Success: {result.success}")
        print(f"  Output: {result.output}")
        print(f"  Time: {result.tool_call.execution_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
