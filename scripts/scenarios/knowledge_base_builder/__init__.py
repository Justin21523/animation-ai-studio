"""
Knowledge Base Builder Scenario

CPU-only knowledge base construction and semantic search using sentence transformers and FAISS.

This scenario provides:
- Document parsing (Markdown, PDF, code, text, JSON, HTML, DOCX)
- Intelligent text chunking with semantic boundaries
- CPU-based embedding generation (Sentence Transformers)
- FAISS vector search (multiple index types)
- Incremental updates and querying
- Integration with Orchestration Layer and Safety System

Author: Animation AI Studio
Date: 2025-12-03
Version: 1.0.0
"""

from .common import (
    # Enums
    DocumentType,
    ProcessingStatus,
    ChunkingStrategy,
    VectorIndexType,

    # Dataclasses
    Document,
    DocumentChunk,
    SearchResult,
    KnowledgeBaseStats,
    ChunkingConfig,
    EmbeddingConfig,
    VectorIndexConfig,
    BuildResult,
    QueryResult,

    # Helper functions
    generate_document_id,
    generate_chunk_id
)

__version__ = "1.0.0"

__all__ = [
    # Enums
    "DocumentType",
    "ProcessingStatus",
    "ChunkingStrategy",
    "VectorIndexType",

    # Dataclasses
    "Document",
    "DocumentChunk",
    "SearchResult",
    "KnowledgeBaseStats",
    "ChunkingConfig",
    "EmbeddingConfig",
    "VectorIndexConfig",
    "BuildResult",
    "QueryResult",

    # Helper functions
    "generate_document_id",
    "generate_chunk_id"
]
