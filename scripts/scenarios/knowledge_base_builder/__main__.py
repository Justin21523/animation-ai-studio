"""
Knowledge Base Builder CLI

Command-line interface for knowledge base operations.

Commands:
- build: Build new knowledge base from documents
- add: Add documents to existing knowledge base
- query: Query knowledge base
- stats: Show knowledge base statistics

Author: Animation AI Studio
Date: 2025-12-03
"""

import argparse
import logging
import sys
from pathlib import Path

from .knowledge_base_builder import KnowledgeBaseBuilder
from .common import (
    ChunkingConfig,
    EmbeddingConfig,
    VectorIndexConfig,
    ChunkingStrategy,
    VectorIndexType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_build(args):
    """Build knowledge base"""
    logger.info("Building knowledge base...")

    # Create configurations
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy(args.chunking_strategy),
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap
    )

    embedding_config = EmbeddingConfig(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        normalize=True
    )

    vector_config = VectorIndexConfig(
        index_type=VectorIndexType(args.index_type),
        n_clusters=args.n_clusters
    )

    # Build knowledge base
    builder = KnowledgeBaseBuilder(chunking_config, embedding_config, vector_config)

    result = builder.build(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        pattern=args.pattern,
        skip_errors=not args.strict
    )

    # Print results
    print("\n=== Build Complete ===")
    print(f"Documents: {result.stats.total_documents}")
    print(f"Chunks: {result.stats.total_chunks}")
    print(f"Embeddings: {result.stats.total_embeddings}")
    print(f"Embedding Dim: {result.stats.embedding_dim}")
    print(f"Index Type: {result.stats.index_type}")
    print(f"Build Time: {result.stats.build_time_seconds:.2f}s")
    print(f"Output: {args.output_dir}")


def cmd_add(args):
    """Add documents to existing knowledge base"""
    logger.info("Adding documents to knowledge base...")

    # Load existing KB
    builder = KnowledgeBaseBuilder()
    builder.load(Path(args.kb_dir))

    # Add documents
    result = builder.add_documents(
        input_dir=Path(args.input_dir),
        pattern=args.pattern,
        skip_errors=not args.strict
    )

    # Save updated KB
    builder.save(Path(args.kb_dir))

    # Print results
    print("\n=== Add Complete ===")
    print(f"Total Documents: {result.stats.total_documents}")
    print(f"Total Chunks: {result.stats.total_chunks}")
    print(f"Total Embeddings: {result.stats.total_embeddings}")


def cmd_query(args):
    """Query knowledge base"""
    logger.info("Querying knowledge base...")

    # Load KB
    builder = KnowledgeBaseBuilder()
    builder.load(Path(args.kb_dir))

    # Query
    result = builder.query(
        query_text=args.query,
        top_k=args.top_k,
        min_score=args.min_score
    )

    # Print results
    print(f"\n=== Query: '{result.query}' ===")
    print(f"Found {len(result.results)} results in {result.query_time_seconds:.3f}s\n")

    for i, search_result in enumerate(result.results, 1):
        chunk = search_result.chunk
        print(f"[{i}] Score: {search_result.score:.4f}")
        print(f"    Document: {chunk.metadata.get('doc_path', 'unknown')}")
        print(f"    Chunk {chunk.chunk_index}: {chunk.content[:200]}...")
        print()


def cmd_stats(args):
    """Show knowledge base statistics"""
    logger.info("Loading knowledge base statistics...")

    # Load KB
    builder = KnowledgeBaseBuilder()
    builder.load(Path(args.kb_dir))

    # Get stats
    stats = builder.get_stats()

    # Print stats
    print("\n=== Knowledge Base Statistics ===")
    print(f"Documents: {stats.total_documents}")
    print(f"Chunks: {stats.total_chunks}")
    print(f"Embeddings: {stats.total_embeddings}")
    print(f"Embedding Dim: {stats.embedding_dim}")
    print(f"Index Type: {stats.index_type}")
    print(f"Chunking Strategy: {stats.chunking_strategy}")
    print(f"Embedding Model: {stats.embedding_model}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Knowledge Base Builder - CPU-only semantic search system"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build new knowledge base")
    build_parser.add_argument("--input-dir", type=str, required=True,
                             help="Directory containing documents")
    build_parser.add_argument("--output-dir", type=str, required=True,
                             help="Directory to save knowledge base")
    build_parser.add_argument("--pattern", type=str, default="**/*",
                             help="Glob pattern for files (default: **/*)")
    build_parser.add_argument("--chunking-strategy", type=str,
                             choices=["fixed_size", "semantic", "sentence", "sliding_window"],
                             default="semantic",
                             help="Text chunking strategy (default: semantic)")
    build_parser.add_argument("--chunk-size", type=int, default=512,
                             help="Chunk size in tokens (default: 512)")
    build_parser.add_argument("--chunk-overlap", type=int, default=128,
                             help="Chunk overlap in tokens (default: 128)")
    build_parser.add_argument("--embedding-model", type=str,
                             default="sentence-transformers/all-MiniLM-L6-v2",
                             help="Sentence Transformers model")
    build_parser.add_argument("--index-type", type=str,
                             choices=["flat_l2", "flat_ip", "ivf_flat", "hnsw"],
                             default="flat_ip",
                             help="FAISS index type (default: flat_ip)")
    build_parser.add_argument("--n-clusters", type=int, default=100,
                             help="Number of clusters for IVF index (default: 100)")
    build_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size for embedding generation (default: 32)")
    build_parser.add_argument("--strict", action="store_true",
                             help="Fail on any error (default: skip errors)")
    build_parser.set_defaults(func=cmd_build)

    # Add command
    add_parser = subparsers.add_parser("add", help="Add documents to existing KB")
    add_parser.add_argument("--kb-dir", type=str, required=True,
                           help="Knowledge base directory")
    add_parser.add_argument("--input-dir", type=str, required=True,
                           help="Directory containing new documents")
    add_parser.add_argument("--pattern", type=str, default="**/*",
                           help="Glob pattern for files (default: **/*)")
    add_parser.add_argument("--strict", action="store_true",
                           help="Fail on any error (default: skip errors)")
    add_parser.set_defaults(func=cmd_add)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("--kb-dir", type=str, required=True,
                             help="Knowledge base directory")
    query_parser.add_argument("--query", type=str, required=True,
                             help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5,
                             help="Number of results to return (default: 5)")
    query_parser.add_argument("--min-score", type=float, default=None,
                             help="Minimum similarity score")
    query_parser.set_defaults(func=cmd_query)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show KB statistics")
    stats_parser.add_argument("--kb-dir", type=str, required=True,
                             help="Knowledge base directory")
    stats_parser.set_defaults(func=cmd_stats)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
