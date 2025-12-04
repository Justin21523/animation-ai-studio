"""
Vector Index

FAISS-based vector indexing and k-NN search for semantic similarity.

Features:
- Multiple FAISS index types (FLAT_L2, FLAT_IP, IVF_FLAT, HNSW)
- Fast k-NN search
- Incremental updates
- Persistent storage (index + metadata)
- CPU-only execution (faiss-cpu)

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..common import (
    DocumentChunk,
    SearchResult,
    VectorIndexType,
    VectorIndexConfig
)

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    FAISS-based vector index for semantic search

    Features:
    - Multiple FAISS index types
    - CPU-only execution
    - k-NN search
    - Incremental updates
    - Persistent storage
    """

    def __init__(
        self,
        config: Optional[VectorIndexConfig] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize vector index

        Args:
            config: Vector index configuration
            embedding_dim: Embedding dimension (required for new index)
        """
        self.config = config or VectorIndexConfig()

        # Import FAISS (CPU-only)
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. "
                "Install with: pip install faiss-cpu"
            )

        # Initialize index
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
            self.index = self._create_index(embedding_dim)
        else:
            self.embedding_dim = None
            self.index = None

        # Metadata storage (maps vector ID to chunk metadata)
        self.id_to_chunk: Dict[int, DocumentChunk] = {}
        self.next_id = 0

        logger.info(f"VectorIndex initialized: type={self.config.index_type.value}, "
                   f"dim={embedding_dim}")

    def build(self, chunks: List[DocumentChunk]):
        """
        Build index from document chunks

        Args:
            chunks: List of document chunks with embeddings
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        # Validate embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        if not chunks_with_embeddings:
            raise ValueError("No chunks have embeddings")

        # Get embedding dimension
        if self.embedding_dim is None:
            self.embedding_dim = chunks_with_embeddings[0].embedding.shape[0]
            self.index = self._create_index(self.embedding_dim)

        # Extract embeddings
        embeddings = np.array([c.embedding for c in chunks_with_embeddings], dtype=np.float32)

        # Add to index
        logger.info(f"Building index with {len(embeddings)} vectors...")

        # Train index if needed (IVF)
        if self.config.index_type in [VectorIndexType.IVF_FLAT]:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add vectors
        ids = np.arange(self.next_id, self.next_id + len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        # Store metadata
        for i, chunk in enumerate(chunks_with_embeddings):
            self.id_to_chunk[self.next_id + i] = chunk

        self.next_id += len(embeddings)

        logger.info(f"Index built: {self.index.ntotal} vectors")

    def add(self, chunks: List[DocumentChunk]):
        """
        Add chunks to existing index (incremental update)

        Args:
            chunks: List of document chunks with embeddings
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build() first.")

        # Validate embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        if not chunks_with_embeddings:
            logger.warning("No chunks with embeddings to add")
            return

        # Extract embeddings
        embeddings = np.array([c.embedding for c in chunks_with_embeddings], dtype=np.float32)

        # Add to index
        ids = np.arange(self.next_id, self.next_id + len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        # Store metadata
        for i, chunk in enumerate(chunks_with_embeddings):
            self.id_to_chunk[self.next_id + i] = chunk

        self.next_id += len(embeddings)

        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score (optional)

        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")

        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            chunk = self.id_to_chunk.get(int(idx))
            if chunk is None:
                logger.warning(f"Chunk not found for index {idx}")
                continue

            # Filter by minimum score
            if min_score is not None and score < min_score:
                continue

            result = SearchResult(
                chunk=chunk,
                score=float(score),
                rank=len(results) + 1
            )
            results.append(result)

        logger.debug(f"Search returned {len(results)} results")

        return results

    def save(self, output_dir: Path):
        """
        Save index and metadata to disk

        Args:
            output_dir: Directory to save index
        """
        if self.index is None:
            raise RuntimeError("Index not initialized")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = output_dir / "index.faiss"
        self.faiss.write_index(self.index, str(index_file))

        # Save metadata
        metadata = {
            "id_to_chunk": self.id_to_chunk,
            "next_id": self.next_id,
            "embedding_dim": self.embedding_dim,
            "config": self.config
        }
        metadata_file = output_dir / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Index saved to {output_dir}")

    def load(self, index_dir: Path):
        """
        Load index and metadata from disk

        Args:
            index_dir: Directory containing saved index
        """
        index_dir = Path(index_dir)

        # Load FAISS index
        index_file = index_dir / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        self.index = self.faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = index_dir / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

        self.id_to_chunk = metadata["id_to_chunk"]
        self.next_id = metadata["next_id"]
        self.embedding_dim = metadata["embedding_dim"]
        self.config = metadata["config"]

        logger.info(f"Index loaded from {index_dir}: {self.index.ntotal} vectors")

    def _create_index(self, embedding_dim: int):
        """
        Create FAISS index based on configuration

        Args:
            embedding_dim: Embedding dimension

        Returns:
            FAISS index
        """
        if self.config.index_type == VectorIndexType.FLAT_L2:
            # Exact search with L2 distance
            index = self.faiss.IndexFlatL2(embedding_dim)
            index = self.faiss.IndexIDMap(index)

        elif self.config.index_type == VectorIndexType.FLAT_IP:
            # Exact search with inner product (cosine similarity with normalized vectors)
            index = self.faiss.IndexFlatIP(embedding_dim)
            index = self.faiss.IndexIDMap(index)

        elif self.config.index_type == VectorIndexType.IVF_FLAT:
            # Inverted file index (approximate search, faster for large datasets)
            n_clusters = min(self.config.n_clusters, 100)  # Reasonable default
            quantizer = self.faiss.IndexFlatL2(embedding_dim)
            index = self.faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
            index = self.faiss.IndexIDMap(index)

        elif self.config.index_type == VectorIndexType.HNSW:
            # Hierarchical Navigable Small World (approximate search, very fast)
            index = self.faiss.IndexHNSWFlat(embedding_dim, 32)
            index = self.faiss.IndexIDMap(index)

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        logger.info(f"Created {self.config.index_type.value} index with dim={embedding_dim}")

        return index

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics

        Returns:
            Dictionary of statistics
        """
        if self.index is None:
            return {
                "total_vectors": 0,
                "embedding_dim": None,
                "index_type": self.config.index_type.value
            }

        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.config.index_type.value,
            "next_id": self.next_id,
            "metadata_count": len(self.id_to_chunk)
        }
