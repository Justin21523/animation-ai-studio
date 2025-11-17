"""
RAG (Retrieval-Augmented Generation) System

Provides semantic search and context retrieval for LLM.

Components:
- Vector Store: FAISS, ChromaDB for similarity search
- Embedding Generator: Uses LLM backend for embeddings
- Document Processor: Multi-format document ingestion
- Retrieval Engine: Semantic search with reranking
- Knowledge Base: High-level management interface

Author: Animation AI Studio
Date: 2025-11-17
"""

from scripts.rag.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from scripts.rag.vectordb.vector_store import (
    VectorStoreType,
    VectorStoreConfig,
    VectorStoreFactory,
    SearchResult
)
from scripts.rag.embeddings.embedding_generator import (
    EmbeddingGenerator,
    CachedEmbeddingGenerator,
    EmbeddingConfig
)
from scripts.rag.documents.document_processor import (
    DocumentProcessor,
    Document,
    DocumentType,
    ChunkingConfig
)
from scripts.rag.retrieval.retrieval_engine import (
    RetrievalEngine,
    RetrievalConfig,
    RetrievalResult
)

__all__ = [
    # Main interface
    "KnowledgeBase",
    "KnowledgeBaseConfig",

    # Vector store
    "VectorStoreType",
    "VectorStoreConfig",
    "VectorStoreFactory",
    "SearchResult",

    # Embeddings
    "EmbeddingGenerator",
    "CachedEmbeddingGenerator",
    "EmbeddingConfig",

    # Documents
    "DocumentProcessor",
    "Document",
    "DocumentType",
    "ChunkingConfig",

    # Retrieval
    "RetrievalEngine",
    "RetrievalConfig",
    "RetrievalResult",
]
