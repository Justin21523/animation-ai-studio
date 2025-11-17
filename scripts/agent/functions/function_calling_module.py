"""
Function Calling Module for Agent Framework

Type-safe function calling with automatic schema generation.
Compatible with OpenAI-style function calling API.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import inspect
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, get_type_hints
from dataclasses import dataclass, field
import time
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient
from scripts.agent.core.types import ToolCall


logger = logging.getLogger(__name__)


@dataclass
class FunctionParameter:
    """Function parameter definition"""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class Function:
    """
    Function definition with schema

    Auto-generates JSON schema from Python function signature.
    """
    name: str
    description: str
    function: Callable
    parameters: List[FunctionParameter] = field(default_factory=list)

    @classmethod
    def from_callable(
        cls,
        func: Callable,
        description: str,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ) -> "Function":
        """
        Create Function from Python callable with auto-schema generation

        Args:
            func: Python function or async function
            description: Function description
            parameter_descriptions: Optional descriptions for parameters

        Returns:
            Function instance
        """
        param_descs = parameter_descriptions or {}

        # Get function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters = []

        for param_name, param in sig.parameters.items():
            # Skip self/cls
            if param_name in ["self", "cls"]:
                continue

            # Get type
            param_type = type_hints.get(param_name, Any)
            type_str = cls._python_type_to_json_type(param_type)

            # Check if required
            required = param.default == inspect.Parameter.empty

            # Get default value
            default = None if required else param.default

            # Get description
            description = param_descs.get(param_name, f"Parameter {param_name}")

            parameters.append(FunctionParameter(
                name=param_name,
                type=type_str,
                description=description,
                required=required,
                default=default
            ))

        return cls(
            name=func.__name__,
            description=description,
            function=func,
            parameters=parameters
        )

    @staticmethod
    def _python_type_to_json_type(python_type) -> str:
        """Convert Python type to JSON schema type"""
        if python_type == str:
            return "string"
        elif python_type in [int, float]:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == list:
            return "array"
        elif python_type == dict:
            return "object"
        else:
            # Check if it's a generic type (List[str], Dict[str, Any], etc.)
            type_str = str(python_type)
            if "list" in type_str.lower():
                return "array"
            elif "dict" in type_str.lower():
                return "object"
            else:
                return "string"  # Default fallback

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling schema

        Returns:
            OpenAI-compatible function schema
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }

            if param.enum:
                properties[param.name]["enum"] = param.enum

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


@dataclass
class FunctionCallResult:
    """Result of function call"""
    function_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


class FunctionRegistry:
    """
    Function Registry

    Manages callable functions with automatic schema generation.
    """

    def __init__(self):
        self.functions: Dict[str, Function] = {}
        logger.info("FunctionRegistry initialized")

    def register(
        self,
        func: Callable,
        description: str,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Register a function with automatic schema generation

        Args:
            func: Python function to register
            description: Function description
            parameter_descriptions: Optional parameter descriptions
        """
        function = Function.from_callable(func, description, parameter_descriptions)
        self.functions[function.name] = function
        logger.info(f"Registered function: {function.name}")

    def register_function(self, function: Function):
        """Register a Function instance directly"""
        self.functions[function.name] = function
        logger.info(f"Registered function: {function.name}")

    def get_function(self, name: str) -> Optional[Function]:
        """Get function by name"""
        return self.functions.get(name)

    def get_all_functions(self) -> List[Function]:
        """Get all registered functions"""
        return list(self.functions.values())

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Get all functions as OpenAI schemas"""
        return [func.to_openai_schema() for func in self.functions.values()]


class FunctionCallingModule:
    """
    Function Calling Module

    Provides type-safe function calling with LLM decision-making.
    Compatible with OpenAI function calling API.

    Key features:
    - Automatic schema generation from Python functions
    - Type validation and conversion
    - OpenAI-compatible interface
    - Async/sync function support
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        function_registry: Optional[FunctionRegistry] = None
    ):
        """
        Initialize function calling module

        Args:
            llm_client: LLM client (will create if not provided)
            function_registry: Function registry (will create if not provided)
        """
        self._llm_client = llm_client
        self._own_client = llm_client is None

        self.function_registry = function_registry or FunctionRegistry()

        logger.info(f"FunctionCallingModule initialized with {len(self.function_registry.functions)} functions")

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

    def register_function(
        self,
        func: Callable,
        description: str,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Register a function

        Args:
            func: Python function
            description: Function description
            parameter_descriptions: Parameter descriptions
        """
        self.function_registry.register(func, description, parameter_descriptions)

    async def select_function(
        self,
        task: str,
        context: Optional[str] = None,
        available_functions: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select appropriate function for task using LLM

        Args:
            task: Task description
            context: Additional context
            available_functions: List of available function names (None = all)

        Returns:
            Function call specification or None
        """
        # Get available functions
        if available_functions:
            functions = [self.function_registry.get_function(name) for name in available_functions]
            functions = [f for f in functions if f is not None]
        else:
            functions = self.function_registry.get_all_functions()

        if not functions:
            logger.warning("No functions available")
            return None

        # Build function selection prompt
        prompt = self._build_function_selection_prompt(task, context, functions)

        # Query LLM
        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        # Parse function call
        function_call = self._parse_function_call_response(response["content"])

        if function_call:
            logger.info(f"Selected function: {function_call['function']}")

        return function_call

    async def call_function(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> FunctionCallResult:
        """
        Call a function with arguments

        Args:
            function_name: Name of function to call
            arguments: Function arguments

        Returns:
            FunctionCallResult
        """
        start_time = time.time()

        function = self.function_registry.get_function(function_name)

        if function is None:
            error_msg = f"Function '{function_name}' not found"
            logger.error(error_msg)
            return FunctionCallResult(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=error_msg
            )

        # Validate and convert arguments
        try:
            validated_args = self._validate_and_convert_arguments(function, arguments)
        except Exception as e:
            error_msg = f"Argument validation failed: {str(e)}"
            logger.error(error_msg)
            return FunctionCallResult(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=error_msg
            )

        # Execute function
        logger.info(f"Calling function: {function_name}")

        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(function.function):
                result = await function.function(**validated_args)
            else:
                result = function.function(**validated_args)

            execution_time = time.time() - start_time

            logger.info(f"Function executed successfully in {execution_time:.2f}s")

            return FunctionCallResult(
                function_name=function_name,
                arguments=validated_args,
                result=result,
                success=True,
                execution_time=execution_time
            )

        except Exception as e:
            error_msg = f"Function execution failed: {str(e)}"
            logger.error(error_msg)

            execution_time = time.time() - start_time

            return FunctionCallResult(
                function_name=function_name,
                arguments=validated_args,
                result=None,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )

    def _build_function_selection_prompt(
        self,
        task: str,
        context: Optional[str],
        functions: List[Function]
    ) -> str:
        """Build function selection prompt"""
        # Format functions as OpenAI schemas
        functions_json = json.dumps(
            [func.to_openai_schema() for func in functions],
            indent=2
        )

        prompt = f"""You are an AI agent that selects appropriate functions to complete tasks.

Task: {task}

Context: {context or "None"}

Available functions:
{functions_json}

Select the most appropriate function and specify arguments. Respond in JSON format:
{{
  "function": "function_name",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }},
  "reasoning": "why this function is appropriate"
}}

If no function is appropriate, respond with: {{"function": null, "reasoning": "..."}}

Select function:"""

        return prompt

    def _parse_function_call_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into function call"""
        try:
            data = json.loads(response)

            if data.get("function") is None:
                logger.info("LLM decided no function is appropriate")
                return None

            if "function" in data and "arguments" in data:
                return {
                    "function": data["function"],
                    "arguments": data["arguments"],
                    "reasoning": data.get("reasoning", "")
                }

            return None

        except json.JSONDecodeError:
            logger.warning("Failed to parse function call JSON")
            return None

    def _validate_and_convert_arguments(
        self,
        function: Function,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and convert arguments to correct types

        Args:
            function: Function definition
            arguments: Raw arguments

        Returns:
            Validated and converted arguments

        Raises:
            ValueError: If validation fails
        """
        validated = {}

        # Check required parameters
        for param in function.parameters:
            if param.required and param.name not in arguments:
                raise ValueError(f"Missing required parameter: {param.name}")

        # Validate and convert each argument
        for param in function.parameters:
            if param.name not in arguments:
                # Use default value if available
                if param.default is not None:
                    validated[param.name] = param.default
                continue

            value = arguments[param.name]

            # Type conversion
            if param.type == "number":
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"Cannot convert '{value}' to number for parameter {param.name}")

            elif param.type == "boolean":
                if isinstance(value, str):
                    value = value.lower() in ["true", "1", "yes"]

            # Enum validation
            if param.enum and value not in param.enum:
                raise ValueError(f"Invalid value for {param.name}: {value}, must be one of {param.enum}")

            validated[param.name] = value

        return validated


# Example functions for testing

async def generate_image(
    character: str,
    scene: str,
    style: str = "pixar_3d",
    quality: str = "high"
) -> Dict[str, Any]:
    """
    Generate an image of a character

    Args:
        character: Character name
        scene: Scene description
        style: Style to use
        quality: Quality preset

    Returns:
        Generation result
    """
    # Placeholder implementation
    return {
        "status": "success",
        "message": f"Generated image of {character} in {scene} with {style} style",
        "output_path": f"outputs/generated_{character}.png",
        "quality": quality
    }


async def synthesize_voice(
    character: str,
    text: str,
    emotion: str = "neutral"
) -> Dict[str, Any]:
    """
    Synthesize character voice

    Args:
        character: Character name
        text: Text to synthesize
        emotion: Emotion/tone

    Returns:
        Synthesis result
    """
    # Placeholder implementation
    return {
        "status": "success",
        "message": f"Synthesized voice for {character} with {emotion} emotion",
        "output_path": f"outputs/synthesized_{character}.wav",
        "text": text
    }


def search_knowledge(
    query: str,
    category: str = "all"
) -> Dict[str, Any]:
    """
    Search knowledge base

    Args:
        query: Search query
        category: Knowledge category

    Returns:
        Search results
    """
    # Placeholder implementation
    return {
        "status": "success",
        "query": query,
        "category": category,
        "results": [
            {"title": "Result 1", "content": "..."},
            {"title": "Result 2", "content": "..."}
        ]
    }


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async with FunctionCallingModule() as function_calling:
        # Register functions with auto-schema generation
        print("\n" + "=" * 60)
        print("Registering Functions")
        print("=" * 60)

        function_calling.register_function(
            generate_image,
            "Generate an image of a character using SDXL + LoRA",
            parameter_descriptions={
                "character": "Character name (e.g., 'luca', 'alberto')",
                "scene": "Description of the scene",
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

        function_calling.register_function(
            search_knowledge,
            "Search the knowledge base",
            parameter_descriptions={
                "query": "Search query",
                "category": "Knowledge category to search in"
            }
        )

        print(f"\nRegistered {len(function_calling.function_registry.functions)} functions")

        # Example 1: Function selection
        print("\n" + "=" * 60)
        print("Example 1: Function Selection")
        print("=" * 60)

        function_call = await function_calling.select_function(
            task="Generate an image of Luca running on the beach with a happy expression",
            context="Character: Luca, Style: Pixar 3D"
        )

        if function_call:
            print(f"\nSelected function: {function_call['function']}")
            print(f"Arguments: {json.dumps(function_call['arguments'], indent=2)}")
            print(f"Reasoning: {function_call.get('reasoning', 'N/A')}")

        # Example 2: Function execution
        print("\n" + "=" * 60)
        print("Example 2: Function Execution")
        print("=" * 60)

        result = await function_calling.call_function(
            function_name="generate_image",
            arguments={
                "character": "luca",
                "scene": "running on the beach, happy expression",
                "style": "pixar_3d",
                "quality": "high"
            }
        )

        print(f"\nExecution result:")
        print(f"  Success: {result.success}")
        print(f"  Result: {result.result}")
        print(f"  Time: {result.execution_time:.2f}s")

        # Example 3: OpenAI schema generation
        print("\n" + "=" * 60)
        print("Example 3: OpenAI Schema Generation")
        print("=" * 60)

        schemas = function_calling.function_registry.get_openai_schemas()
        print(f"\nGenerated {len(schemas)} OpenAI-compatible schemas:")
        print(json.dumps(schemas[0], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
