"""
Retrieval Engine for RAG System

High-level retrieval interface combining vector search,
reranking, and relevance filtering.

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

from scripts.rag.vectordb.vector_store import (
    VectorStoreConfig,
    VectorStoreType,
    VectorStoreFactory,
    SearchResult,
    BaseVectorStore
)
from scripts.rag.embeddings.embedding_generator import (
    EmbeddingGenerator,
    CachedEmbeddingGenerator
)
from scripts.rag.documents.document_processor import Document


logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval engine"""
    # Vector search
    top_k: int = 10  # Initial retrieval count
    similarity_threshold: float = 0.7  # Minimum similarity score

    # Reranking
    enable_reranking: bool = True
    rerank_top_k: int = 5  # Final result count after reranking

    # Filtering
    enable_metadata_filter: bool = True
    filter_criteria: Optional[Dict[str, Any]] = None

    # Query enhancement
    enable_query_expansion: bool = False
    expansion_terms: int = 3

    # Context
    include_context: bool = True  # Include neighboring chunks
    context_window: int = 1  # Chunks before/after


@dataclass
class RetrievalResult:
    """Retrieval result with metadata"""
    documents: List[SearchResult]
    query: str
    query_embedding: Optional[np.ndarray] = None
    retrieval_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.retrieval_stats is None:
            self.retrieval_stats = {
                "total_retrieved": len(self.documents),
                "avg_score": np.mean([d.score for d in self.documents]) if self.documents else 0.0,
                "max_score": max([d.score for d in self.documents]) if self.documents else 0.0
            }


class RetrievalEngine:
    """
    Retrieval engine for RAG system

    Features:
    - Semantic vector search
    - Metadata filtering
    - Result reranking
    - Query expansion
    - Context inclusion
    - Hybrid search (dense + sparse)
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_generator: EmbeddingGenerator,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize retrieval engine

        Args:
            vector_store: Vector store for similarity search
            embedding_generator: Embedding generator
            config: Retrieval configuration
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or RetrievalConfig()

        logger.info("RetrievalEngine initialized")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_embeddings: bool = False
    ) -> RetrievalResult:
        """
        Retrieve relevant documents

        Args:
            query: Query string
            top_k: Number of results (overrides config)
            filters: Metadata filters
            return_embeddings: Include embeddings in results

        Returns:
            RetrievalResult with documents
        """
        top_k = top_k or self.config.top_k

        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_embedding(
            query,
            prefix="query:"
        )

        # Apply metadata filters if configured
        if self.config.enable_metadata_filter and filters is None:
            filters = self.config.filter_criteria

        # Vector search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        # Filter by similarity threshold
        results = [
            r for r in results
            if r.score >= self.config.similarity_threshold
        ]

        # Rerank if enabled
        if self.config.enable_reranking and len(results) > self.config.rerank_top_k:
            results = await self._rerank_results(query, results)
            results = results[:self.config.rerank_top_k]

        # Include context if enabled
        if self.config.include_context:
            results = await self._include_context(results)

        # Add embeddings if requested
        if return_embeddings:
            for result in results:
                result.embedding = query_embedding

        return RetrievalResult(
            documents=results,
            query=query,
            query_embedding=query_embedding if return_embeddings else None
        )

    async def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve for multiple queries

        Args:
            queries: List of queries
            top_k: Number of results per query
            filters: Metadata filters

        Returns:
            List of RetrievalResults
        """
        results = await asyncio.gather(*[
            self.retrieve(query, top_k, filters)
            for query in queries
        ])

        return results

    async def retrieve_with_feedback(
        self,
        query: str,
        positive_doc_ids: Optional[List[str]] = None,
        negative_doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve with relevance feedback

        Args:
            query: Query string
            positive_doc_ids: IDs of relevant documents
            negative_doc_ids: IDs of irrelevant documents
            top_k: Number of results

        Returns:
            RetrievalResult with refined results
        """
        # Generate base query embedding
        query_embedding = await self.embedding_generator.generate_embedding(
            query,
            prefix="query:"
        )

        # Modify embedding based on feedback
        if positive_doc_ids or negative_doc_ids:
            query_embedding = await self._apply_feedback(
                query_embedding,
                positive_doc_ids or [],
                negative_doc_ids or []
            )

        # Search with modified embedding
        top_k = top_k or self.config.top_k
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Filter by threshold
        results = [
            r for r in results
            if r.score >= self.config.similarity_threshold
        ]

        return RetrievalResult(
            documents=results,
            query=query,
            query_embedding=query_embedding
        )

    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder or LLM

        This provides more accurate relevance scoring than
        pure vector similarity.
        """
        # Placeholder: Use LLM to score relevance
        # In production, could use:
        # 1. Cross-encoder model (e.g., ms-marco)
        # 2. LLM-based scoring
        # 3. BM25 + semantic hybrid

        logger.debug(f"Reranking {len(results)} results")

        # For now, keep original order
        # TODO: Implement actual reranking
        return results

    async def _apply_feedback(
        self,
        query_embedding: np.ndarray,
        positive_doc_ids: List[str],
        negative_doc_ids: List[str]
    ) -> np.ndarray:
        """
        Apply relevance feedback using Rocchio algorithm

        Modified embedding = α*query + β*avg(positive) - γ*avg(negative)
        """
        alpha, beta, gamma = 1.0, 0.75, 0.25

        modified = alpha * query_embedding

        # Add positive examples
        if positive_doc_ids:
            positive_embeddings = []
            for doc_id in positive_doc_ids:
                doc = self.vector_store.get_document(doc_id)
                if doc and doc.embedding is not None:
                    positive_embeddings.append(doc.embedding)

            if positive_embeddings:
                avg_positive = np.mean(positive_embeddings, axis=0)
                modified += beta * avg_positive

        # Subtract negative examples
        if negative_doc_ids:
            negative_embeddings = []
            for doc_id in negative_doc_ids:
                doc = self.vector_store.get_document(doc_id)
                if doc and doc.embedding is not None:
                    negative_embeddings.append(doc.embedding)

            if negative_embeddings:
                avg_negative = np.mean(negative_embeddings, axis=0)
                modified -= gamma * avg_negative

        # Normalize
        norm = np.linalg.norm(modified)
        if norm > 0:
            modified = modified / norm

        return modified

    async def _include_context(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Include context (neighboring chunks) for each result

        This helps provide more complete information by including
        chunks before and after each retrieved chunk.
        """
        if self.config.context_window == 0:
            return results

        # Placeholder: Get neighboring chunks
        # TODO: Implement context retrieval based on chunk_index metadata

        logger.debug(f"Including context for {len(results)} results")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "vector_store_size": self.vector_store.count(),
            "config": {
                "top_k": self.config.top_k,
                "similarity_threshold": self.config.similarity_threshold,
                "enable_reranking": self.config.enable_reranking
            }
        }


class HybridRetrievalEngine(RetrievalEngine):
    """
    Hybrid retrieval combining dense and sparse retrieval

    Combines:
    - Dense retrieval (semantic embeddings)
    - Sparse retrieval (BM25, TF-IDF)

    Better recall and precision than either alone.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_generator: EmbeddingGenerator,
        config: Optional[RetrievalConfig] = None,
        sparse_weight: float = 0.3
    ):
        super().__init__(vector_store, embedding_generator, config)
        self.sparse_weight = sparse_weight
        self.dense_weight = 1.0 - sparse_weight

        logger.info(f"HybridRetrievalEngine: dense={self.dense_weight}, sparse={self.sparse_weight}")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_embeddings: bool = False
    ) -> RetrievalResult:
        """Hybrid retrieval combining dense and sparse"""
        top_k = top_k or self.config.top_k

        # Dense retrieval (semantic)
        dense_results = await super().retrieve(
            query,
            top_k=top_k * 2,  # Get more candidates
            filters=filters,
            return_embeddings=False
        )

        # Sparse retrieval (keyword-based)
        sparse_results = await self._sparse_retrieve(query, top_k * 2, filters)

        # Combine results with weighted scoring
        combined = self._combine_results(
            dense_results.documents,
            sparse_results,
            query
        )

        # Take top_k
        combined = combined[:top_k]

        return RetrievalResult(
            documents=combined,
            query=query,
            retrieval_stats={
                "dense_count": len(dense_results.documents),
                "sparse_count": len(sparse_results),
                "combined_count": len(combined)
            }
        )

    async def _sparse_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Sparse retrieval using keyword matching

        Placeholder for BM25/TF-IDF implementation
        """
        # TODO: Implement BM25 or TF-IDF
        logger.debug("Sparse retrieval not implemented yet")
        return []

    def _combine_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """
        Combine dense and sparse results

        Uses reciprocal rank fusion (RRF)
        """
        # Build score map
        scores = {}

        # Add dense scores
        for rank, result in enumerate(dense_results):
            doc_id = result.doc_id
            rrf_score = 1.0 / (60 + rank + 1)  # RRF with k=60
            scores[doc_id] = scores.get(doc_id, 0) + self.dense_weight * rrf_score

        # Add sparse scores
        for rank, result in enumerate(sparse_results):
            doc_id = result.doc_id
            rrf_score = 1.0 / (60 + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + self.sparse_weight * rrf_score

        # Get all unique documents
        doc_map = {}
        for result in dense_results + sparse_results:
            if result.doc_id not in doc_map:
                doc_map[result.doc_id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        combined = []
        for doc_id in sorted_ids:
            result = doc_map[doc_id]
            result.score = scores[doc_id]  # Update with combined score
            combined.append(result)

        return combined


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup (placeholder)
    from scripts.rag.vectordb.vector_store import VectorStoreConfig, VectorStoreType
    from scripts.rag.embeddings.embedding_generator import EmbeddingGenerator

    # Create vector store
    vector_config = VectorStoreConfig(
        store_type=VectorStoreType.FAISS,
        persist_dir="/tmp/test_vector_store",
        dimension=1024
    )
    from scripts.rag.vectordb.vector_store import VectorStoreFactory
    vector_store = VectorStoreFactory.create(vector_config)

    # Create embedding generator
    async with EmbeddingGenerator() as embedding_gen:
        # Create retrieval engine
        engine = RetrievalEngine(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            config=RetrievalConfig(top_k=5)
        )

        # Example retrieval
        query = "Tell me about Luca's character"
        results = await engine.retrieve(query)

        print(f"\nQuery: {query}")
        print(f"Retrieved {len(results.documents)} documents")
        print(f"Stats: {results.retrieval_stats}")


if __name__ == "__main__":
    asyncio.run(main())
