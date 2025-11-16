"""
Unified LLM Client for Animation AI Studio
Connects to self-hosted LLM backend (vLLM services)

Hardware: RTX 5080 16GB VRAM
Models: Qwen2.5-VL-7B, Qwen2.5-14B, Qwen2.5-Coder-7B
"""

import httpx
import json
import base64
from typing import List, Dict, Optional, Union, AsyncIterator
from pathlib import Path
from loguru import logger
import asyncio


class LLMClient:
    """
    Unified LLM client for all AI tasks

    Connects to vLLM services via FastAPI Gateway
    Handles model selection and switching automatically
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7000/v1",
        timeout: float = 300.0
    ):
        """
        Initialize LLM client

        Args:
            base_url: Gateway URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"✅ LLM Client initialized: {base_url}")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
        logger.info("✅ LLM Client closed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    # =========================================================================
    # Core API Methods
    # =========================================================================

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Union[str, List]]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ) -> Dict:
        """
        Send chat completion request

        Args:
            model: Model name (qwen-vl-7b, qwen-14b, qwen-coder-7b)
            messages: Conversation history
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Stream response (not yet implemented)
            **kwargs: Additional parameters

        Returns:
            Chat completion response
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"❌ Request failed: {e}")
            raise

    async def health_check(self) -> Dict:
        """
        Check gateway and service health

        Returns:
            Health status for all services
        """
        try:
            response = await self.client.get(
                f"{self.base_url.replace('/v1', '')}/health"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            raise

    async def list_models(self) -> List[Dict]:
        """
        List available models

        Returns:
            List of model information
        """
        try:
            response = await self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            raise

    # =========================================================================
    # High-Level Task Methods
    # =========================================================================

    async def understand_creative_intent(
        self,
        user_request: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Use reasoning model to understand creative intent

        Args:
            user_request: User's creative request
            context: Optional context information

        Returns:
            Structured creative intent analysis
        """
        system_prompt = """You are a creative director AI specializing in animation content creation.
Analyze creative requests and provide structured guidance."""

        user_prompt = f"""Analyze this creative request in detail:

{user_request}

Provide a JSON response with:
1. core_goal: Main creative objective
2. style_mood: Desired artistic style and emotional tone
3. target_audience: Who this is for
4. success_criteria: How to measure success
5. technical_challenges: Potential difficulties
6. suggested_approach: Step-by-step plan"""

        if context:
            user_prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.chat(
            model="qwen-14b",  # Use reasoning model
            messages=messages,
            temperature=0.3,  # Lower temperature for structured output
            max_tokens=2048
        )

        # Parse JSON from response
        content = response['choices'][0]['message']['content']
        try:
            # Try to extract JSON from markdown code block if present
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content

            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("⚠️ Failed to parse JSON, returning raw response")
            return {"raw_response": content}

    async def analyze_video_content(
        self,
        video_frames: List[str],
        analysis_focus: str,
        max_frames: int = 10
    ) -> Dict:
        """
        Analyze video content using vision model

        Args:
            video_frames: List of base64-encoded frames
            analysis_focus: What to focus on in analysis
            max_frames: Maximum frames to analyze

        Returns:
            Video analysis results
        """
        # Limit frames to avoid token overflow
        frames = video_frames[:max_frames]

        # Build multimodal content
        content = [
            {
                "type": "text",
                "text": f"Analyze this video focusing on: {analysis_focus}\n\nProvide detailed observations."
            }
        ]

        # Add image frames
        for i, frame in enumerate(frames):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })

        messages = [{"role": "user", "content": content}]

        response = await self.chat(
            model="qwen-vl-7b",  # Use vision model
            messages=messages,
            temperature=0.2,  # Low temperature for accurate analysis
            max_tokens=2048
        )

        return {
            "analysis": response['choices'][0]['message']['content'],
            "frames_analyzed": len(frames),
            "focus": analysis_focus
        }

    async def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str
    ) -> str:
        """
        Analyze a single image

        Args:
            image_path: Path to image file
            prompt: Analysis prompt

        Returns:
            Analysis text
        """
        # Load and encode image
        image_path = Path(image_path)
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }]

        response = await self.chat(
            model="qwen-vl-7b",
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )

        return response['choices'][0]['message']['content']

    async def generate_code(
        self,
        task_description: str,
        language: str = "python",
        context: Optional[str] = None
    ) -> str:
        """
        Generate code using coder model

        Args:
            task_description: What code to generate
            language: Programming language
            context: Optional code context

        Returns:
            Generated code
        """
        system_prompt = f"""You are an expert {language} programmer specializing in AI tools and automation.
Generate clean, efficient, well-documented code."""

        user_prompt = task_description
        if context:
            user_prompt = f"Context:\n{context}\n\nTask:\n{task_description}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.chat(
            model="qwen-coder-7b",  # Use code model
            messages=messages,
            temperature=0.1,  # Very low for code generation
            max_tokens=4096
        )

        return response['choices'][0]['message']['content']

    async def explain_code(
        self,
        code: str,
        language: str = "python"
    ) -> str:
        """
        Explain code functionality

        Args:
            code: Code to explain
            language: Programming language

        Returns:
            Code explanation
        """
        messages = [{
            "role": "user",
            "content": f"Explain this {language} code in detail:\n\n```{language}\n{code}\n```"
        }]

        response = await self.chat(
            model="qwen-coder-7b",
            messages=messages,
            temperature=0.3,
            max_tokens=2048
        )

        return response['choices'][0]['message']['content']

    async def chat_interactive(
        self,
        model: str = "qwen-14b",
        system_prompt: Optional[str] = None
    ):
        """
        Interactive chat session

        Args:
            model: Model to use
            system_prompt: Optional system prompt
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        print(f"🤖 Interactive chat with {model}")
        print("Type 'exit' to quit, 'clear' to reset conversation\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'clear':
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    print("🔄 Conversation cleared\n")
                    continue
                elif not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                response = await self.chat(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048
                )

                assistant_message = response['choices'][0]['message']['content']
                messages.append({"role": "assistant", "content": assistant_message})

                print(f"\n🤖 {model}: {assistant_message}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                print(f"\n⚠️ Error: {e}\n")


# =============================================================================
# Usage Examples
# =============================================================================

async def example_usage():
    """Example usage of LLM client"""

    async with LLMClient() as client:
        # 1. Health check
        print("\n=== Health Check ===")
        health = await client.health_check()
        print(json.dumps(health, indent=2))

        # 2. List models
        print("\n=== Available Models ===")
        models = await client.list_models()
        for model in models:
            print(f"- {model['id']}: {model['type']}")

        # 3. Creative intent analysis
        print("\n=== Creative Intent Analysis ===")
        intent = await client.understand_creative_intent(
            "Create a funny parody video of Luca's ocean scene with exaggerated expressions"
        )
        print(json.dumps(intent, indent=2))

        # 4. Code generation
        print("\n=== Code Generation ===")
        code = await client.generate_code(
            "Write a function to apply slow-motion effect to video using MoviePy"
        )
        print(code)


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
