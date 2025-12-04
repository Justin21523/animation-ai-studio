"""
LLM Client Module

Provides unified interface for interacting with LLM backend.
"""

from scripts.core.llm_client.llm_client import LLMClient
# from scripts.core.llm_client.utils import create_message, parse_response  # Functions not yet implemented

__all__ = [
    "LLMClient",
    # "create_message",
    # "parse_response",
]

__version__ = "1.0.0"
