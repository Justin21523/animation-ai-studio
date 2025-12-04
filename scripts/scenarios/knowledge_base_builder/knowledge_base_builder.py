"""
Knowledge Base Builder

Main orchestrator for building and querying knowledge bases.

Features:
- Document loading and parsing
- Text chunking with semantic boundaries
- Embedding generation (CPU-only)
- Vector indexing with FAISS
- Semantic search
- Incremental updates
- Statistics and reporting

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .common import (
    Document,
    DocumentChunk,
    SearchResult,
    BuildResult,
    QueryResult,
    KnowledgeBaseStats,
    ChunkingConfig,
    EmbeddingConfig,
    VectorIndexConfig,
    ProcessingStatus
)
from .analyzers import DocumentLoader
from .processors import TextChunker, EmbeddingGenerator, VectorIndex

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """
    Main knowledge base builder orchestrator

    Features:
    - End-to-end pipeline: load → chunk → embed → index
    - Incremental updates
    - Semantic search
    - Statistics tracking
    - Persistent storage
    """

    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        vector_config: Optional[VectorIndexConfig] = None
    ):
        """
        Initialize knowledge base builder

        Args:
            chunking_config: Text chunking configuration
            embedding_config: Embedding generation configuration
            vector_config: Vector index configuration
        """
        # Initialize components
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunking_config)
        self.embedder = EmbeddingGenerator(embedding_config)
        self.index = VectorIndex(vector_config, self.embedder.get_embedding_dim())

        # Configurations
        self.chunking_config = chunking_config or ChunkingConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_config = vector_config or VectorIndexConfig()

        # State tracking
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, DocumentChunk] = {}

        logger.info("KnowledgeBaseBuilder initialized")

    def build(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "**/*",
        skip_errors: bool = True
    ) -> BuildResult:
        """
        Build knowledge base from directory

        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save knowledge base
            pattern: Glob pattern for files
            skip_errors: Skip files that fail to process

        Returns:
            BuildResult with statistics
        """
        start_time = time.time()

        logger.info(f"Building knowledge base from {input_dir}")

        # 1. Load documents
        logger.info("Step 1/4: Loading documents...")
        documents = self.loader.batch_load(input_dir, pattern, skip_errors=skip_errors)
        self.documents = {doc.id: doc for doc in documents}

        if not documents:
            raise ValueError(f"No documents found in {input_dir}")

        logger.info(f"Loaded {len(documents)} documents")

        # 2. Chunk documents
        logger.info("Step 2/4: Chunking documents...")
        chunk_results = self.chunker.batch_chunk(documents)

        all_chunks = []
        for doc_id, chunks in chunk_results.items():
            all_chunks.extend(chunks)
            # Update document chunk IDs
            if doc_id in self.documents:
                self.documents[doc_id].chunk_ids = [c.id for c in chunks]

        self.chunks = {chunk.id: chunk for chunk in all_chunks}

        logger.info(f"Created {len(all_chunks)} chunks")

        # 3. Generate embeddings
        logger.info("Step 3/4: Generating embeddings...")
        all_chunks = self.embedder.embed_chunks(all_chunks)

        # Update chunks dict
        self.chunks = {chunk.id: chunk for chunk in all_chunks}

        logger.info(f"Generated {len(all_chunks)} embeddings")

        # 4. Build index
        logger.info("Step 4/4: Building vector index...")
        self.index.build(all_chunks)

        logger.info(f"Index built: {self.index.index.ntotal} vectors")

        # 5. Save knowledge base
        logger.info("Saving knowledge base...")
        self.save(output_dir)

        # Calculate statistics
        elapsed_time = time.time() - start_time

        stats = KnowledgeBaseStats(
            total_documents=len(self.documents),
            total_chunks=len(self.chunks),
            total_embeddings=self.index.index.ntotal,
            embedding_dim=self.index.embedding_dim,
            index_type=self.vector_config.index_type.value,
            chunking_strategy=self.chunking_config.strategy.value,
            embedding_model=self.embedding_config.model_name,
            build_time_seconds=elapsed_time
        )

        result = BuildResult(
            success=True,
            stats=stats,
            output_dir=output_dir,
            errors=[]
        )

        logger.info(f"Knowledge base built successfully in {elapsed_time:.2f}s")

        return result

    def add_documents(
        self,
        input_dir: Path,
        pattern: str = "**/*",
        skip_errors: bool = True
    ) -> BuildResult:
        """
        Add documents to existing knowledge base (incremental update)

        Args:
            input_dir: Directory containing new documents
            pattern: Glob pattern for files
            skip_errors: Skip files that fail to process

        Returns:
            BuildResult with statistics
        """
        start_time = time.time()

        logger.info(f"Adding documents from {input_dir}")

        # 1. Load documents
        documents = self.loader.batch_load(input_dir, pattern, skip_errors=skip_errors)

        if not documents:
            logger.warning(f"No documents found in {input_dir}")
            return BuildResult(
                success=True,
                stats=self.get_stats(),
                output_dir=None,
                errors=[]
            )

        # 2. Chunk documents
        chunk_results = self.chunker.batch_chunk(documents)

        all_chunks = []
        for doc_id, chunks in chunk_results.items():
            all_chunks.extend(chunks)
            # Update document
            if doc_id in self.documents:
                self.documents[doc_id].chunk_ids.extend([c.id for c in chunks])
            else:
                self.documents[doc_id] = next(d for d in documents if d.id == doc_id)
                self.documents[doc_id].chunk_ids = [c.id for c in chunks]

        # 3. Generate embeddings
        all_chunks = self.embedder.embed_chunks(all_chunks)

        # Update chunks
        for chunk in all_chunks:
            self.chunks[chunk.id] = chunk

        # 4. Add to index
        self.index.add(all_chunks)

        elapsed_time = time.time() - start_time

        logger.info(f"Added {len(documents)} documents ({len(all_chunks)} chunks) in {elapsed_time:.2f}s")

        stats = self.get_stats()

        result = BuildResult(
            success=True,
            stats=stats,
            output_dir=None,
            errors=[]
        )

        return result

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> QueryResult:
        """
        Query knowledge base

        Args:
            query_text: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            QueryResult with search results
        """
        start_time = time.time()

        logger.info(f"Querying: '{query_text}'")

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query_text)

        # Search index
        results = self.index.search(query_embedding, top_k, min_score)

        elapsed_time = time.time() - start_time

        logger.info(f"Query returned {len(results)} results in {elapsed_time:.3f}s")

        return QueryResult(
            query=query_text,
            results=results,
            query_time_seconds=elapsed_time
        )

    def save(self, output_dir: Path):
        """
        Save knowledge base to disk

        Args:
            output_dir: Directory to save knowledge base
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        self.index.save(output_dir)

        # Save documents
        documents_file = output_dir / "documents.json"
        with open(documents_file, "w", encoding="utf-8") as f:
            documents_data = {
                doc_id: {
                    "id": doc.id,
                    "path": str(doc.path),
                    "doc_type": doc.doc_type.value,
                    "metadata": doc.metadata,
                    "chunk_ids": doc.chunk_ids,
                    "file_size_bytes": doc.file_size_bytes,
                    "processing_status": doc.processing_status.value
                }
                for doc_id, doc in self.documents.items()
            }
            json.dump(documents_data, f, indent=2)

        # Save chunks (without embeddings to save space)
        chunks_file = output_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            chunks_data = {
                chunk_id: {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "char_count": chunk.char_count,
                    "metadata": chunk.metadata
                }
                for chunk_id, chunk in self.chunks.items()
            }
            json.dump(chunks_data, f, indent=2)

        # Save stats
        stats = self.get_stats()
        stats_file = output_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
                "total_embeddings": stats.total_embeddings,
                "embedding_dim": stats.embedding_dim,
                "index_type": stats.index_type,
                "chunking_strategy": stats.chunking_strategy,
                "embedding_model": stats.embedding_model,
                "build_time_seconds": stats.build_time_seconds
            }, f, indent=2)

        logger.info(f"Knowledge base saved to {output_dir}")

    def load(self, kb_dir: Path):
        """
        Load knowledge base from disk

        Args:
            kb_dir: Directory containing saved knowledge base
        """
        kb_dir = Path(kb_dir)

        # Load index
        self.index.load(kb_dir)

        # Load documents
        documents_file = kb_dir / "documents.json"
        with open(documents_file, "r", encoding="utf-8") as f:
            documents_data = json.load(f)

        # Reconstruct documents (without content to save memory)
        self.documents = {
            doc_id: Document(
                id=data["id"],
                path=Path(data["path"]),
                doc_type=data["doc_type"],
                content="",  # Not stored
                metadata=data["metadata"],
                chunk_ids=data["chunk_ids"],
                file_size_bytes=data["file_size_bytes"],
                processing_status=ProcessingStatus(data["processing_status"])
            )
            for doc_id, data in documents_data.items()
        }

        # Load chunks
        chunks_file = kb_dir / "chunks.json"
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = {
            chunk_id: DocumentChunk(
                id=data["id"],
                document_id=data["document_id"],
                content=data["content"],
                chunk_index=data["chunk_index"],
                token_count=data["token_count"],
                char_count=data["char_count"],
                metadata=data["metadata"]
            )
            for chunk_id, data in chunks_data.items()
        }

        logger.info(f"Knowledge base loaded from {kb_dir}")

    def get_stats(self) -> KnowledgeBaseStats:
        """
        Get knowledge base statistics

        Returns:
            KnowledgeBaseStats object
        """
        return KnowledgeBaseStats(
            total_documents=len(self.documents),
            total_chunks=len(self.chunks),
            total_embeddings=self.index.index.ntotal if self.index.index else 0,
            embedding_dim=self.index.embedding_dim,
            index_type=self.vector_config.index_type.value,
            chunking_strategy=self.chunking_config.strategy.value,
            embedding_model=self.embedding_config.model_name,
            build_time_seconds=0.0
        )
