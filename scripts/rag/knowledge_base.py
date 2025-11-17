"""
Knowledge Base Manager for RAG System

High-level interface for managing knowledge base:
- Document ingestion
- Vector indexing
- Retrieval
- Updates and maintenance

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag.vectordb.vector_store import (
    VectorStoreConfig,
    VectorStoreType,
    VectorStoreFactory,
    BaseVectorStore
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
from scripts.core.llm_client import LLMClient


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base"""
    # Paths
    persist_dir: str = "/mnt/c/AI_LLM_projects/ai_warehouse/rag/knowledge_base"
    cache_dir: str = "/mnt/c/AI_LLM_projects/ai_warehouse/cache/embeddings"

    # Vector store
    vector_store_type: str = "faiss"  # faiss, chroma
    embedding_dimension: int = 1024

    # Embedding
    embedding_model: str = "qwen-14b"
    use_cached_embeddings: bool = True

    # Document processing
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    default_top_k: int = 5
    similarity_threshold: float = 0.7

    # Maintenance
    auto_save: bool = True
    save_interval: int = 100  # Save every N documents


class KnowledgeBase:
    """
    Knowledge Base Manager

    Provides high-level interface for:
    - Adding documents from various sources
    - Semantic search and retrieval
    - Knowledge base maintenance
    - Integration with LLM

    Example:
        async with KnowledgeBase() as kb:
            # Add documents
            kb.add_documents_from_directory("data/films/luca")

            # Search
            results = await kb.search("Tell me about Luca's character")

            # Use with LLM
            context = kb.get_context_for_llm(results)
    """

    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        """
        Initialize knowledge base

        Args:
            config: Knowledge base configuration
        """
        self.config = config or KnowledgeBaseConfig()

        # Create directories
        Path(self.config.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.vector_store: Optional[BaseVectorStore] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.retrieval_engine: Optional[RetrievalEngine] = None
        self.llm_client: Optional[LLMClient] = None

        # Stats
        self.stats = {
            "total_documents": 0,
            "last_updated": None,
            "document_types": {}
        }

        self._initialized = False
        self._document_counter = 0

        logger.info(f"KnowledgeBase created with config: {self.config.persist_dir}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return

        logger.info("Initializing Knowledge Base...")

        # Initialize vector store
        vector_config = VectorStoreConfig(
            store_type=VectorStoreType(self.config.vector_store_type),
            persist_dir=str(Path(self.config.persist_dir) / "vector_store"),
            dimension=self.config.embedding_dimension
        )
        self.vector_store = VectorStoreFactory.create(vector_config)

        # Load existing index if available
        try:
            self.vector_store.load()
            logger.info("Loaded existing vector store")
        except Exception as e:
            logger.info("No existing vector store found, will create new")

        # Initialize embedding generator
        embedding_config = EmbeddingConfig(
            model=self.config.embedding_model,
            dimension=self.config.embedding_dimension
        )

        if self.config.use_cached_embeddings:
            self.embedding_generator = CachedEmbeddingGenerator(
                config=embedding_config,
                cache_dir=self.config.cache_dir
            )
        else:
            self.embedding_generator = EmbeddingGenerator(config=embedding_config)

        await self.embedding_generator.__aenter__()

        # Initialize document processor
        chunking_config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.document_processor = DocumentProcessor(chunking_config)

        # Initialize retrieval engine
        retrieval_config = RetrievalConfig(
            top_k=self.config.default_top_k,
            similarity_threshold=self.config.similarity_threshold
        )
        self.retrieval_engine = RetrievalEngine(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            config=retrieval_config
        )

        # Initialize LLM client
        self.llm_client = LLMClient()
        await self.llm_client.__aenter__()

        # Load stats
        await self._load_stats()

        self._initialized = True
        logger.info("Knowledge Base initialized successfully")

    async def cleanup(self):
        """Cleanup resources"""
        if not self._initialized:
            return

        # Save vector store
        if self.vector_store and self.config.auto_save:
            self.vector_store.save()

        # Save stats
        await self._save_stats()

        # Cleanup components
        if self.embedding_generator:
            await self.embedding_generator.__aexit__(None, None, None)

        if self.llm_client:
            await self.llm_client.__aexit__(None, None, None)

        logger.info("Knowledge Base cleanup complete")

    async def add_document(
        self,
        document: Document,
        save_immediately: bool = False
    ) -> None:
        """
        Add single document to knowledge base

        Args:
            document: Document to add
            save_immediately: Save vector store immediately
        """
        if not self._initialized:
            await self.initialize()

        # Generate embedding
        embedding = await self.embedding_generator.generate_embedding(
            document.content,
            prefix="passage:"
        )

        # Add to vector store
        self.vector_store.add_documents(
            doc_ids=[document.doc_id],
            embeddings=embedding.reshape(1, -1),
            contents=[document.content],
            metadata=[document.metadata]
        )

        # Update stats
        self.stats["total_documents"] += 1
        self.stats["last_updated"] = datetime.now().isoformat()

        doc_type_key = document.doc_type.value
        self.stats["document_types"][doc_type_key] = \
            self.stats["document_types"].get(doc_type_key, 0) + 1

        self._document_counter += 1

        # Auto-save if configured
        if self.config.auto_save and self._document_counter >= self.config.save_interval:
            self.vector_store.save()
            await self._save_stats()
            self._document_counter = 0

        if save_immediately:
            self.vector_store.save()

        logger.debug(f"Added document: {document.doc_id}")

    async def add_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> None:
        """
        Add multiple documents

        Args:
            documents: List of documents
            show_progress: Show progress
        """
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents...")

        for i, doc in enumerate(documents):
            await self.add_document(doc, save_immediately=False)

            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(documents)}")

        # Save at the end
        self.vector_store.save()
        await self._save_stats()

        logger.info(f"Successfully added {len(documents)} documents")

    async def add_document_from_file(
        self,
        file_path: Union[str, Path],
        doc_type: Optional[DocumentType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add document from file

        Args:
            file_path: Path to file
            doc_type: Document type (auto-detect if None)
            metadata: Additional metadata

        Returns:
            Number of documents added
        """
        if not self._initialized:
            await self.initialize()

        # Process file
        documents = self.document_processor.process_file(
            file_path,
            doc_type,
            metadata
        )

        # Add documents
        await self.add_documents(documents, show_progress=False)

        return len(documents)

    async def add_documents_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "**/*",
        doc_type: Optional[DocumentType] = None,
        recursive: bool = True
    ) -> int:
        """
        Add all documents from directory

        Args:
            directory: Directory path
            pattern: File pattern (glob)
            doc_type: Document type for all files
            recursive: Search recursively

        Returns:
            Total number of documents added
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        logger.info(f"Scanning directory: {directory}")

        # Find all matching files
        if recursive:
            files = list(directory.glob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter for text-based files
        supported_extensions = {'.txt', '.md', '.json', '.yaml', '.yml'}
        files = [f for f in files if f.suffix.lower() in supported_extensions]

        logger.info(f"Found {len(files)} files to process")

        total_docs = 0
        for file_path in files:
            try:
                count = await self.add_document_from_file(file_path, doc_type)
                total_docs += count
                logger.info(f"Processed {file_path}: {count} documents")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        logger.info(f"Added {total_docs} documents from {len(files)} files")
        return total_docs

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Search knowledge base

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters

        Returns:
            Retrieval result
        """
        if not self._initialized:
            await self.initialize()

        return await self.retrieval_engine.retrieve(query, top_k, filters)

    async def get_context_for_llm(
        self,
        query: str,
        max_tokens: int = 4000,
        top_k: Optional[int] = None
    ) -> str:
        """
        Get formatted context for LLM

        Args:
            query: Query
            max_tokens: Maximum context length
            top_k: Number of documents to retrieve

        Returns:
            Formatted context string
        """
        # Retrieve relevant documents
        results = await self.search(query, top_k)

        if not results.documents:
            return ""

        # Format context
        context_parts = []
        total_length = 0

        for i, doc in enumerate(results.documents, 1):
            # Format document
            doc_text = f"[Document {i}]\n{doc.content}\n"

            # Check length (rough estimate: 4 chars per token)
            if total_length + len(doc_text) > max_tokens * 4:
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n".join(context_parts)

    async def answer_question(
        self,
        question: str,
        model: str = "qwen-14b",
        max_tokens: int = 1000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer question using RAG

        Args:
            question: Question to answer
            model: LLM model to use
            max_tokens: Max response tokens
            include_sources: Include source documents

        Returns:
            Answer with sources
        """
        if not self._initialized:
            await self.initialize()

        # Get context
        context = await self.get_context_for_llm(question, max_tokens=2000)

        if not context:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": "low"
            }

        # Build prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer: """

        # Query LLM
        response = await self.llm_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )

        answer = response.get("content", "")

        # Get sources
        sources = []
        if include_sources:
            results = await self.search(question, top_k=3)
            sources = [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in results.documents
            ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high" if sources else "low"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            **self.stats,
            "vector_store_size": self.vector_store.count() if self.vector_store else 0
        }

    async def _save_stats(self):
        """Save statistics to disk"""
        stats_path = Path(self.config.persist_dir) / "stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

    async def _load_stats(self):
        """Load statistics from disk"""
        stats_path = Path(self.config.persist_dir) / "stats.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            logger.info(f"Loaded stats: {self.stats['total_documents']} documents")


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create knowledge base
    async with KnowledgeBase() as kb:
        # Add sample documents
        sample_docs = [
            Document(
                doc_id="luca_001",
                content="Luca is a young sea monster living off the Italian Riviera.",
                doc_type=DocumentType.CHARACTER_PROFILE,
                metadata={"character": "luca", "film": "luca"}
            ),
            Document(
                doc_id="alberto_001",
                content="Alberto is Luca's best friend, a fellow sea monster.",
                doc_type=DocumentType.CHARACTER_PROFILE,
                metadata={"character": "alberto", "film": "luca"}
            )
        ]

        await kb.add_documents(sample_docs)

        # Search
        results = await kb.search("Tell me about Luca")
        print(f"\nSearch results: {len(results.documents)}")
        for doc in results.documents:
            print(f"- {doc.content[:100]}... (score: {doc.score:.3f})")

        # Answer question
        answer_result = await kb.answer_question("Who is Luca's best friend?")
        print(f"\nQuestion: Who is Luca's best friend?")
        print(f"Answer: {answer_result['answer']}")
        print(f"Sources: {len(answer_result['sources'])}")

        # Stats
        print(f"\nStats: {kb.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
