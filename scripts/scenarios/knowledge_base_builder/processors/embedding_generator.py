"""
Embedding Generator

CPU-only embedding generation using Sentence Transformers.

Features:
- Sentence Transformers models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- Batch processing for efficiency
- CPU-only execution (no GPU required)
- Caching and progress tracking
- Normalized embeddings (L2 norm)

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from ..common import (
    Document,
    DocumentChunk,
    EmbeddingConfig
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    CPU-only embedding generator using Sentence Transformers

    Features:
    - Multiple Sentence Transformers models
    - Batch processing with configurable batch size
    - CPU-only execution
    - Normalized embeddings (cosine similarity)
    - Progress tracking
    - Memory efficient (streaming)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator

        Args:
            config: Embedding configuration (default: EmbeddingConfig with defaults)
        """
        self.config = config or EmbeddingConfig()

        # Import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Load model
        logger.info(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name, device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"EmbeddingGenerator initialized: model={self.config.model_name}, "
                   f"dim={self.embedding_dim}, batch_size={self.config.batch_size}")

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for documents

        Args:
            documents: List of documents

        Returns:
            Documents with embeddings attached
        """
        if not documents:
            logger.warning("No documents to embed")
            return documents

        # Extract content
        texts = [doc.content for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self._encode_batch(texts)

        # Attach embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        logger.info(f"Embedded {len(documents)} documents")

        return documents

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks

        Args:
            chunks: List of document chunks

        Returns:
            Chunks with embeddings attached
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return chunks

        # Extract content
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self._encode_batch(texts)

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        logger.info(f"Embedded {len(chunks)} chunks")

        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False
        )

        return embedding

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode texts in batches

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Convert to list of arrays
        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings[i] for i in range(len(embeddings))]

        return embeddings

    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension

        Returns:
            Embedding vector dimension
        """
        return self.embedding_dim


class EmbeddingCache:
    """
    Simple disk-based embedding cache

    Features:
    - Save/load embeddings to/from disk
    - Content hash-based caching
    - Numpy-based storage
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize embedding cache

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EmbeddingCache initialized: {self.cache_dir}")

    def get(self, content_hash: str) -> Optional[np.ndarray]:
        """
        Get cached embedding

        Args:
            content_hash: Hash of content

        Returns:
            Cached embedding or None
        """
        cache_file = self.cache_dir / f"{content_hash}.npy"

        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding {content_hash}: {e}")
                return None

        return None

    def set(self, content_hash: str, embedding: np.ndarray):
        """
        Cache embedding

        Args:
            content_hash: Hash of content
            embedding: Embedding vector
        """
        cache_file = self.cache_dir / f"{content_hash}.npy"

        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding {content_hash}: {e}")

    def clear(self):
        """Clear all cached embeddings"""
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")

        logger.info("Embedding cache cleared")

    def size(self) -> int:
        """
        Get cache size

        Returns:
            Number of cached embeddings
        """
        return len(list(self.cache_dir.glob("*.npy")))
