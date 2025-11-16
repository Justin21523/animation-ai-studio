"""
Pydantic models for LLM Gateway API
Request/Response schemas compatible with OpenAI API format
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal, Union


class Message(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]  # Support multimodal content


class ChatRequest(BaseModel):
    """Chat completion request"""
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    stream: bool = False
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class ChatChoice(BaseModel):
    """Single chat completion choice"""
    index: int
    message: Message
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[UsageInfo] = None


class EmbeddingRequest(BaseModel):
    """Embedding request"""
    model: str
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    """Single embedding"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gateway: str
    services: Dict[str, str]


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    type: str
    capabilities: List[str]


class ErrorResponse(BaseModel):
    """Error response"""
    error: Dict[str, Any]
