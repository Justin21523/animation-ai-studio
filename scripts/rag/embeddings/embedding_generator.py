"""
Embedding Generation for RAG System

Generates embeddings using LLM Backend (Qwen2.5 models).
Supports text and multimodal embeddings.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding generation configuration"""
    model: str = "qwen-14b"  # Use Qwen2.5-14B for embeddings
    dimension: int = 1024  # Qwen2.5 embedding dimension
    normalize: bool = True  # Normalize embeddings
    batch_size: int = 32  # Batch processing
    max_length: int = 8192  # Max token length


class EmbeddingGenerator:
    """
    Generate embeddings using LLM Backend

    Uses Qwen2.5 models for high-quality embeddings:
    - Qwen2.5-14B for text embeddings
    - Qwen2.5-VL-7B for multimodal embeddings

    Features:
    - Batch processing
    - Automatic normalization
    - Caching support
    - Error handling
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize embedding generator

        Args:
            config: Embedding configuration
            llm_client: LLM client (will create if not provided)
        """
        self.config = config or EmbeddingConfig()
        self._llm_client = llm_client
        self._own_client = llm_client is None

        logger.info(f"EmbeddingGenerator initialized with model: {self.config.model}")

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

    async def generate_embedding(
        self,
        text: str,
        prefix: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Input text
            prefix: Optional prefix (e.g., "query:", "passage:")

        Returns:
            Embedding vector (normalized if config.normalize=True)
        """
        if prefix:
            text = f"{prefix} {text}"

        # Truncate if too long
        if len(text) > self.config.max_length:
            logger.warning(f"Text too long ({len(text)}), truncating to {self.config.max_length}")
            text = text[:self.config.max_length]

        try:
            # Use LLM client to get embedding
            # Note: This is a placeholder - actual implementation depends on LLM backend API
            # For now, we'll use chat completion with a special prompt
            response = await self._llm_client.chat(
                model=self.config.model,
                messages=[{
                    "role": "user",
                    "content": f"Generate embedding representation for: {text}"
                }],
                temperature=0.0,
                max_tokens=1  # We only need the hidden state
            )

            # Extract embedding from response
            # This is a placeholder - actual extraction depends on API
            embedding = self._extract_embedding_from_response(response)

            # Normalize if configured
            if self.config.normalize:
                embedding = self._normalize(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def generate_embeddings(
        self,
        texts: List[str],
        prefix: Optional[str] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts
            prefix: Optional prefix for all texts
            show_progress: Show progress bar

        Returns:
            Array of embeddings (N, dimension)
        """
        if not texts:
            return np.array([])

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            if show_progress:
                logger.info(f"Processing batch {i // self.config.batch_size + 1}/{(len(texts) - 1) // self.config.batch_size + 1}")

            # Generate embeddings for batch
            batch_embeddings = await asyncio.gather(*[
                self.generate_embedding(text, prefix)
                for text in batch
            ])

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _extract_embedding_from_response(self, response: Dict[str, Any]) -> np.ndarray:
        """
        Extract embedding from LLM response

        Note: This is a placeholder implementation.
        Actual extraction depends on LLM backend API format.

        For production, we need to:
        1. Modify LLM backend to expose embeddings endpoint
        2. Or use a dedicated embedding model
        """
        # Placeholder: Generate random embedding
        # TODO: Replace with actual embedding extraction
        embedding = np.random.randn(self.config.dimension).astype(np.float32)
        return embedding

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.config.dimension


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator with caching

    Caches embeddings to disk to avoid recomputation.
    Useful for large knowledge bases.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        llm_client: Optional[LLMClient] = None,
        cache_dir: str = "/mnt/c/AI_LLM_projects/ai_warehouse/cache/embeddings"
    ):
        super().__init__(config, llm_client)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}  # In-memory cache
        logger.info(f"Embedding cache directory: {self.cache_dir}")

    def _get_cache_key(self, text: str, prefix: Optional[str] = None) -> str:
        """Generate cache key for text"""
        import hashlib
        content = f"{prefix or ''}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate_embedding(
        self,
        text: str,
        prefix: Optional[str] = None
    ) -> np.ndarray:
        """Generate embedding with caching"""
        cache_key = self._get_cache_key(text, prefix)

        # Check in-memory cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit (memory): {cache_key}")
            return self.cache[cache_key]

        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            logger.debug(f"Cache hit (disk): {cache_key}")
            embedding = np.load(cache_path)
            self.cache[cache_key] = embedding
            return embedding

        # Generate new embedding
        logger.debug(f"Cache miss: {cache_key}")
        embedding = await super().generate_embedding(text, prefix)

        # Save to cache
        np.save(cache_path, embedding)
        self.cache[cache_key] = embedding

        return embedding

    def clear_cache(self):
        """Clear all caches"""
        self.cache = {}
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        logger.info("Embedding cache cleared")


class MultimodalEmbeddingGenerator(EmbeddingGenerator):
    """
    Multimodal embedding generator

    Uses Qwen2.5-VL-7B for generating embeddings from:
    - Text
    - Images
    - Text + Images (multimodal)

    Useful for character image retrieval, scene matching, etc.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        # Use vision model
        config = config or EmbeddingConfig()
        config.model = "qwen-vl-7b"
        super().__init__(config, llm_client)

    async def generate_image_embedding(
        self,
        image_path: str,
        text_context: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for image

        Args:
            image_path: Path to image
            text_context: Optional text context

        Returns:
            Multimodal embedding
        """
        # Use vision model to process image
        # Placeholder implementation
        logger.info(f"Generating embedding for image: {image_path}")

        # TODO: Implement actual multimodal embedding
        # This requires LLM backend to support vision + embedding extraction

        embedding = np.random.randn(self.config.dimension).astype(np.float32)

        if self.config.normalize:
            embedding = self._normalize(embedding)

        return embedding

    async def generate_multimodal_embedding(
        self,
        text: str,
        image_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate multimodal embedding

        Args:
            text: Text content
            image_path: Optional image path

        Returns:
            Multimodal embedding
        """
        if image_path:
            return await self.generate_image_embedding(image_path, text)
        else:
            return await self.generate_embedding(text)


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example 1: Basic embedding generation
    async with EmbeddingGenerator() as generator:
        text = "Luca is a young sea monster who dreams of exploring the human world."
        embedding = await generator.generate_embedding(text)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding)}")

    # Example 2: Batch embeddings
    async with EmbeddingGenerator() as generator:
        texts = [
            "Luca is a curious sea monster",
            "Alberto is Luca's best friend",
            "Portorosso is an Italian seaside town"
        ]
        embeddings = await generator.generate_embeddings(texts)
        print(f"Batch embeddings shape: {embeddings.shape}")

    # Example 3: Cached embeddings
    async with CachedEmbeddingGenerator() as generator:
        text = "Character description for caching test"

        # First call - generates and caches
        embedding1 = await generator.generate_embedding(text)

        # Second call - loads from cache
        embedding2 = await generator.generate_embedding(text)

        # Should be identical
        assert np.allclose(embedding1, embedding2)
        print("Cache test passed!")


if __name__ == "__main__":
    asyncio.run(main())
