"""
Embedding Generation for RAG System

Unified, reproducible embeddings using Sentence Transformers (CPU by default).

This replaces the previous placeholder implementation that attempted to extract
embeddings via the LLM backend (and fell back to random vectors).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


logger = logging.getLogger(__name__)

def _resolve_default_model(model: str) -> str:
    default_hf = "sentence-transformers/all-MiniLM-L6-v2"
    if str(model) != default_hf:
        return str(model)

    candidates = [
        Path("/mnt/c/ai_models/sentence_transformers/all-MiniLM-L6-v2"),
        Path("ai_models/sentence_transformers/all-MiniLM-L6-v2"),
        Path("models/sentence_transformers/all-MiniLM-L6-v2"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return default_hf


@dataclass
class EmbeddingConfig:
    """Embedding generation configuration"""

    # Sentence-Transformers model name or local path
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Execution
    device: str = "cpu"
    batch_size: int = 32

    # Normalization (cosine similarity friendly)
    normalize: bool = True

    # Optional max length (tokens). SentenceTransformer uses model.max_seq_length.
    max_length: int = 512

    # Backwards-compatible field (will be inferred if None)
    dimension: Optional[int] = None


class EmbeddingGenerator:
    """
    Sentence-Transformers embedding generator (async-friendly wrapper).

    Notes:
    - Uses CPU by default for stability and reproducibility.
    - Model loading happens in __aenter__ to keep imports lightweight.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model: Optional[Any] = None
        self._embedding_dim: Optional[int] = None

    async def __aenter__(self):
        if self._model is not None:
            return self

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from e

        model_name = _resolve_default_model(self.config.model)
        logger.info(f"Loading embedding model: {model_name} (device={self.config.device})")
        self._model = SentenceTransformer(model_name, device=self.config.device)

        # Apply max length if supported.
        try:
            self._model.max_seq_length = int(self.config.max_length)
        except Exception:
            pass

        self._embedding_dim = int(self._model.get_sentence_embedding_dimension())
        if self.config.dimension is None:
            self.config.dimension = self._embedding_dim

        logger.info(f"Embedding model ready: dim={self._embedding_dim}, batch_size={self.config.batch_size}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Best-effort cleanup; keeping the model loaded across calls is usually desirable.
        self._model = None

    async def generate_embedding(self, text: str, prefix: Optional[str] = None) -> np.ndarray:
        """Generate embedding for a single text"""
        if self._model is None:
            await self.__aenter__()

        if prefix:
            text = f"{prefix} {text}"

        def _encode_one() -> np.ndarray:
            embedding = self._model.encode(
                text,
                batch_size=1,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding.astype(np.float32, copy=False)

        return await asyncio.to_thread(_encode_one)

    async def generate_embeddings(
        self,
        texts: List[str],
        prefix: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self._model is None:
            await self.__aenter__()

        if not texts:
            return np.array([], dtype=np.float32)

        if prefix:
            texts = [f"{prefix} {t}" for t in texts]

        def _encode_many() -> np.ndarray:
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32, copy=False)

        return await asyncio.to_thread(_encode_many)

    def get_dimension(self) -> int:
        """Get embedding dimension (after model is loaded)"""
        if self.config.dimension is not None:
            return int(self.config.dimension)
        if self._embedding_dim is not None:
            return int(self._embedding_dim)
        raise RuntimeError("Embedding dimension not available until the model is loaded")


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator with disk cache.

    Caches embeddings to disk to avoid recomputation for large knowledge bases.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        cache_dir: str = "outputs/rag/embedding_cache",
    ):
        super().__init__(config=config)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, np.ndarray] = {}
        logger.info(f"Embedding cache directory: {self.cache_dir}")

    def _get_cache_key(self, text: str, prefix: Optional[str] = None) -> str:
        import hashlib

        content = f"{prefix or ''}:{text}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def generate_embedding(self, text: str, prefix: Optional[str] = None) -> np.ndarray:
        cache_key = self._get_cache_key(text, prefix)

        # Memory cache
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached

        # Disk cache
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            embedding = np.load(cache_path)
            self._memory_cache[cache_key] = embedding
            return embedding

        embedding = await super().generate_embedding(text, prefix)

        np.save(cache_path, embedding)
        self._memory_cache[cache_key] = embedding
        return embedding

    def clear_cache(self):
        self._memory_cache = {}
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        logger.info("Embedding cache cleared")
