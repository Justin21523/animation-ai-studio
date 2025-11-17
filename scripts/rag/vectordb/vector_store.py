"""
Vector Store Management for RAG System

Manages vector database for efficient similarity search.
Supports multiple backends: FAISS, Chroma, Milvus.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Supported vector store types"""
    FAISS = "faiss"
    CHROMA = "chroma"
    MILVUS = "milvus"


@dataclass
class SearchResult:
    """Single search result"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    embedding: Optional[np.ndarray] = None


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    store_type: VectorStoreType
    persist_dir: str
    dimension: int = 1024  # Qwen2.5 embedding dimension
    metric: str = "cosine"  # cosine, l2, ip
    index_type: str = "Flat"  # Flat, IVF, HNSW

    # FAISS-specific
    nlist: int = 100  # For IVF index
    nprobe: int = 10  # Search clusters

    # Chroma-specific
    collection_name: str = "animation_ai_knowledge"

    # Performance
    batch_size: int = 100
    enable_gpu: bool = False


class BaseVectorStore:
    """Base class for vector stores"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.persist_dir = Path(config.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def add_documents(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        contents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to vector store"""
        raise NotImplementedError

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        raise NotImplementedError

    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs"""
        raise NotImplementedError

    def get_document(self, doc_id: str) -> Optional[SearchResult]:
        """Get document by ID"""
        raise NotImplementedError

    def count(self) -> int:
        """Get total document count"""
        raise NotImplementedError

    def save(self) -> None:
        """Persist vector store to disk"""
        raise NotImplementedError

    def load(self) -> None:
        """Load vector store from disk"""
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store

    Pros:
    - Extremely fast search
    - Low memory footprint
    - GPU support
    - Good for large-scale (millions of vectors)

    Cons:
    - Requires separate metadata storage
    - No built-in filtering
    """

    def __init__(self, config: VectorStoreConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        super().__init__(config)

        self.dimension = config.dimension
        self.index = None
        self.doc_ids = []
        self.metadata_store = {}  # doc_id -> {content, metadata}

        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.config.index_type == "Flat":
            # Exact search (brute force)
            if self.config.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
            elif self.config.metric == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported metric: {self.config.metric}")

        elif self.config.index_type == "IVF":
            # Inverted file index (faster but approximate)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.config.nlist,
                faiss.METRIC_L2
            )
            # Need to train IVF index
            self.index.nprobe = self.config.nprobe

        elif self.config.index_type == "HNSW":
            # Hierarchical navigable small world (fast + accurate)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        # GPU support
        if self.config.enable_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to use GPU: {e}")

        logger.info(f"Initialized FAISS index: {self.config.index_type}")

    def add_documents(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        contents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to FAISS index"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dimension}")

        # Normalize embeddings for cosine similarity
        if self.config.metric == "cosine":
            faiss.normalize_L2(embeddings)

        # Train IVF index if needed
        if self.config.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add to FAISS
        self.index.add(embeddings)

        # Store metadata separately
        for doc_id, content, meta in zip(doc_ids, contents, metadata):
            self.doc_ids.append(doc_id)
            self.metadata_store[doc_id] = {
                "content": content,
                "metadata": meta
            }

        logger.info(f"Added {len(doc_ids)} documents to FAISS")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search FAISS index"""
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No result
                continue

            doc_id = self.doc_ids[idx]
            doc_data = self.metadata_store[doc_id]

            # Apply filters if provided
            if filters:
                match = all(
                    doc_data["metadata"].get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue

            results.append(SearchResult(
                doc_id=doc_id,
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                score=float(score)
            ))

        return results

    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents (requires rebuild)"""
        # FAISS doesn't support deletion, need to rebuild
        remaining_indices = [
            i for i, doc_id in enumerate(self.doc_ids)
            if doc_id not in doc_ids
        ]

        if not remaining_indices:
            self._initialize_index()
            self.doc_ids = []
            self.metadata_store = {}
            return

        # Rebuild index with remaining documents
        logger.warning("FAISS deletion requires index rebuild")
        # TODO: Implement rebuild

    def get_document(self, doc_id: str) -> Optional[SearchResult]:
        """Get document by ID"""
        if doc_id not in self.metadata_store:
            return None

        doc_data = self.metadata_store[doc_id]
        return SearchResult(
            doc_id=doc_id,
            content=doc_data["content"],
            metadata=doc_data["metadata"],
            score=1.0
        )

    def count(self) -> int:
        """Get total count"""
        return self.index.ntotal

    def save(self) -> None:
        """Save FAISS index and metadata"""
        index_path = self.persist_dir / "faiss.index"
        metadata_path = self.persist_dir / "metadata.json"

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "doc_ids": self.doc_ids,
                "metadata_store": self.metadata_store
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index to {self.persist_dir}")

    def load(self) -> None:
        """Load FAISS index and metadata"""
        index_path = self.persist_dir / "faiss.index"
        metadata_path = self.persist_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            logger.warning("No saved index found")
            return

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.doc_ids = data["doc_ids"]
            self.metadata_store = data["metadata_store"]

        logger.info(f"Loaded FAISS index from {self.persist_dir}")


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-based vector store

    Pros:
    - Built-in metadata filtering
    - Easy to use
    - Persistent storage
    - Good for small-to-medium scale

    Cons:
    - Slower than FAISS for large datasets
    - Higher memory usage
    """

    def __init__(self, config: VectorStoreConfig):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")

        super().__init__(config)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"dimension": config.dimension}
        )

        logger.info(f"Initialized ChromaDB collection: {config.collection_name}")

    def add_documents(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        contents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents to ChromaDB"""
        # Convert numpy to list
        embeddings_list = embeddings.tolist()

        # Add to collection
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings_list,
            documents=contents,
            metadatas=metadata
        )

        logger.info(f"Added {len(doc_ids)} documents to ChromaDB")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search ChromaDB"""
        # Convert numpy to list
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_list = query_embedding.tolist()

        # Build where clause for filtering
        where_clause = filters if filters else None

        # Query
        results = self.collection.query(
            query_embeddings=query_list,
            n_results=top_k,
            where=where_clause
        )

        # Build SearchResults
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                doc_id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1.0 - results['distances'][0][i]  # Convert distance to similarity
            ))

        return search_results

    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from ChromaDB"""
        self.collection.delete(ids=doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB")

    def get_document(self, doc_id: str) -> Optional[SearchResult]:
        """Get document by ID"""
        results = self.collection.get(ids=[doc_id])

        if not results['ids']:
            return None

        return SearchResult(
            doc_id=results['ids'][0],
            content=results['documents'][0],
            metadata=results['metadatas'][0],
            score=1.0
        )

    def count(self) -> int:
        """Get total count"""
        return self.collection.count()

    def save(self) -> None:
        """ChromaDB auto-persists"""
        logger.info("ChromaDB auto-persists to disk")

    def load(self) -> None:
        """ChromaDB auto-loads"""
        logger.info("ChromaDB auto-loads from disk")


class VectorStoreFactory:
    """Factory for creating vector stores"""

    @staticmethod
    def create(config: VectorStoreConfig) -> BaseVectorStore:
        """Create vector store based on config"""
        if config.store_type == VectorStoreType.FAISS:
            return FAISSVectorStore(config)
        elif config.store_type == VectorStoreType.CHROMA:
            return ChromaVectorStore(config)
        elif config.store_type == VectorStoreType.MILVUS:
            raise NotImplementedError("Milvus not implemented yet")
        else:
            raise ValueError(f"Unknown store type: {config.store_type}")
