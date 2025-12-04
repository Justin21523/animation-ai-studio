"""
Knowledge Base Builder - Common Data Structures

Core data structures and enumerations for knowledge base construction and querying.
Provides document types, processing status, and comprehensive data models for
documents, chunks, and knowledge base statistics.

Author: Animation AI Studio
Date: 2025-12-03
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class DocumentType(Enum):
    """Supported document types for knowledge base construction"""
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    CODE = "code"
    JSON = "json"
    HTML = "html"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChunkingStrategy(Enum):
    """Text chunking strategies"""
    FIXED_SIZE = "fixed_size"  # Fixed token count
    SEMANTIC = "semantic"  # Paragraph/section boundaries
    SENTENCE = "sentence"  # Sentence-based chunks
    SLIDING_WINDOW = "sliding_window"  # Overlapping windows


class VectorIndexType(Enum):
    """FAISS vector index types"""
    FLAT_L2 = "flatl2"  # Exact search, L2 distance
    FLAT_IP = "flatip"  # Exact search, inner product
    IVF_FLAT = "ivfflat"  # Inverted file index with flat quantizer
    HNSW = "hnsw"  # Hierarchical Navigable Small World graphs


@dataclass
class Document:
    """
    Represents a single document in the knowledge base

    Attributes:
        id: Unique document identifier (hash-based)
        path: Original file path
        doc_type: Document type classification
        content: Full document text content
        metadata: Additional document metadata (title, author, tags, etc.)
        embedding: Optional document-level embedding
        chunk_ids: List of chunk IDs belonging to this document
        created_at: Document creation timestamp
        modified_at: Document modification timestamp
        file_size_bytes: Original file size
        processing_status: Current processing status
    """
    id: str
    path: Path
    doc_type: DocumentType
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_ids: List[str] = field(default_factory=list)
    created_at: Optional[float] = None
    modified_at: Optional[float] = None
    file_size_bytes: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING


@dataclass
class DocumentChunk:
    """
    Represents a text chunk from a document

    Attributes:
        id: Unique chunk identifier
        document_id: Parent document ID
        content: Chunk text content
        chunk_index: Position in document (0-indexed)
        embedding: Vector embedding for semantic search
        metadata: Chunk metadata (source, section, page, etc.)
        token_count: Number of tokens in chunk
        char_count: Number of characters in chunk
        start_pos: Starting character position in document
        end_pos: Ending character position in document
    """
    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    char_count: int = 0
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class SearchResult:
    """
    Represents a single search result

    Attributes:
        chunk_id: Matching chunk ID
        document_id: Source document ID
        content: Chunk text content
        score: Similarity/relevance score
        distance: Vector distance metric
        metadata: Result metadata
        rank: Result ranking (1-indexed)
    """
    chunk_id: str
    document_id: str
    content: str
    score: float
    distance: float
    metadata: Dict[str, Any]
    rank: int = 0


@dataclass
class KnowledgeBaseStats:
    """
    Knowledge base statistics and metrics

    Attributes:
        total_documents: Total number of documents indexed
        total_chunks: Total number of text chunks
        total_tokens: Total token count across all chunks
        doc_type_counts: Document count by type
        index_size_mb: FAISS index size in megabytes
        metadata_size_mb: Metadata storage size in megabytes
        embedding_dimension: Vector embedding dimensionality
        avg_chunk_size: Average chunk size in tokens
        avg_chunks_per_doc: Average chunks per document
        last_updated: Last index update timestamp
    """
    total_documents: int
    total_chunks: int
    total_tokens: int
    doc_type_counts: Dict[DocumentType, int]
    index_size_mb: float
    metadata_size_mb: float
    embedding_dimension: int
    avg_chunk_size: float = 0.0
    avg_chunks_per_doc: float = 0.0
    last_updated: Optional[float] = None


@dataclass
class ChunkingConfig:
    """
    Configuration for text chunking

    Attributes:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size (tokens or characters)
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_size: Minimum chunk size threshold
        max_chunk_size: Maximum chunk size threshold
        respect_boundaries: Respect semantic boundaries (paragraphs, sections)
        preserve_formatting: Preserve markdown/code formatting
    """
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    respect_boundaries: bool = True
    preserve_formatting: bool = True


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation

    Attributes:
        model_name: Sentence transformer model name
        device: Compute device (cpu, cuda)
        batch_size: Batch size for embedding generation
        normalize_embeddings: L2-normalize embeddings
        show_progress: Show progress bar during generation
        max_seq_length: Maximum sequence length for model
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = True
    max_seq_length: int = 512


@dataclass
class VectorIndexConfig:
    """
    Configuration for FAISS vector index

    Attributes:
        index_type: Type of FAISS index to use
        metric: Distance metric (l2, cosine, inner_product)
        nlist: Number of clusters for IVF index
        nprobe: Number of clusters to search (IVF)
        ef_construction: HNSW construction parameter
        ef_search: HNSW search parameter
        use_gpu: Use GPU for index operations (if available)
    """
    index_type: VectorIndexType = VectorIndexType.FLAT_L2
    metric: str = "l2"
    nlist: int = 100  # For IVF index
    nprobe: int = 10  # For IVF search
    ef_construction: int = 200  # For HNSW
    ef_search: int = 64  # For HNSW
    use_gpu: bool = False


@dataclass
class BuildResult:
    """
    Knowledge base build result

    Attributes:
        success: Build success status
        stats: Knowledge base statistics
        documents_processed: Number of documents processed
        documents_failed: Number of documents that failed
        chunks_generated: Number of chunks generated
        build_time_seconds: Total build time
        errors: List of error messages
        warnings: List of warning messages
    """
    success: bool
    stats: KnowledgeBaseStats
    documents_processed: int
    documents_failed: int
    chunks_generated: int
    build_time_seconds: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """
    Knowledge base query result

    Attributes:
        query: Original query text
        results: List of search results
        query_embedding: Query vector embedding
        search_time_ms: Search time in milliseconds
        total_results: Total number of results found
        returned_results: Number of results returned (after limit)
    """
    query: str
    results: List[SearchResult]
    query_embedding: Optional[np.ndarray] = None
    search_time_ms: float = 0.0
    total_results: int = 0
    returned_results: int = 0


# Helper function for generating document IDs
def generate_document_id(file_path: Path) -> str:
    """
    Generate unique document ID from file path

    Args:
        file_path: Path to document file

    Returns:
        SHA256 hash-based document ID
    """
    import hashlib
    path_str = str(file_path.resolve())
    return hashlib.sha256(path_str.encode()).hexdigest()[:16]


# Helper function for generating chunk IDs
def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate unique chunk ID

    Args:
        document_id: Parent document ID
        chunk_index: Chunk position in document

    Returns:
        Unique chunk identifier
    """
    return f"{document_id}_chunk_{chunk_index:04d}"
