"""
Knowledge Base Builder

Builds vector databases for RAG (Retrieval Augmented Generation) from various data sources.
All processing is CPU-only using local embeddings and vector stores.

Features:
  - Document ingestion (text, markdown, PDF, images with OCR)
  - Text chunking with overlap for context preservation
  - CPU-based embeddings (sentence-transformers)
  - Vector database creation (FAISS, ChromaDB)
  - Metadata extraction and indexing
  - Incremental updates support
  - JSON manifest generation

Usage:
  python scripts/automation/scenarios/knowledge_base_builder.py \
    --input-dir /path/to/documents/ \
    --output-dir /path/to/knowledge_base/ \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --chunk-size 512 \
    --vector-db faiss

Author: Animation AI Studio Team
Last Modified: 2025-12-02
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure
from scripts.core.safety import (
    enforce_cpu_only,
    verify_no_gpu_usage,
    MemoryMonitor,
    RuntimeMonitor,
    run_preflight,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Document:
    """Document with metadata for knowledge base."""
    doc_id: str
    content: str
    source_path: str
    doc_type: str
    metadata: Dict[str, Any]
    chunk_ids: List[str]
    timestamp: str


@dataclass
class Chunk:
    """Text chunk for embedding."""
    chunk_id: str
    doc_id: str
    content: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]


# ============================================================================
# Document Loaders
# ============================================================================

def load_text_file(file_path: Path) -> str:
    """Load plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def load_markdown_file(file_path: Path) -> str:
    """Load markdown file."""
    return load_text_file(file_path)


def load_pdf_file(file_path: Path) -> str:
    """
    Load PDF file using PyPDF2 (CPU-only).

    Falls back to plain text extraction if PyPDF2 not available.
    """
    try:
        import PyPDF2

        text_content = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1} from {file_path}: {e}")
                    continue

        return "\n\n".join(text_content)

    except ImportError:
        logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
        return f"[PDF content not extracted - PyPDF2 not available: {file_path}]"
    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        return f"[PDF extraction failed: {e}]"


def load_image_with_ocr(file_path: Path) -> str:
    """
    Load image and extract text using OCR (pytesseract).

    CPU-only OCR for extracting text from images.
    """
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)

        return f"[OCR from {file_path.name}]\n{text}"

    except ImportError:
        logger.warning("pytesseract not installed. Install with: pip install pytesseract")
        return f"[Image content not extracted - OCR not available: {file_path}]"
    except Exception as e:
        logger.error(f"Failed to OCR image {file_path}: {e}")
        return f"[OCR failed: {e}]"


def load_document(file_path: Path) -> Tuple[str, str]:
    """
    Load document based on file type.

    Args:
        file_path: Path to document

    Returns:
        Tuple of (content, doc_type)
    """
    ext = file_path.suffix.lower()

    loaders = {
        '.txt': (load_text_file, 'text'),
        '.md': (load_markdown_file, 'markdown'),
        '.markdown': (load_markdown_file, 'markdown'),
        '.pdf': (load_pdf_file, 'pdf'),
        '.jpg': (load_image_with_ocr, 'image'),
        '.jpeg': (load_image_with_ocr, 'image'),
        '.png': (load_image_with_ocr, 'image'),
    }

    if ext in loaders:
        loader_func, doc_type = loaders[ext]
        content = loader_func(file_path)
        return content, doc_type
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return f"[Unsupported file type: {ext}]", 'unknown'


# ============================================================================
# Text Chunking
# ============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    separator: str = "\n\n"
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        separator: Preferred split points (paragraphs, sentences)

    Returns:
        List of (chunk_text, start_idx, end_idx)
    """
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks = []

    # Split on separator first
    segments = text.split(separator)

    current_chunk = []
    current_length = 0
    current_start = 0

    for segment in segments:
        segment_length = len(segment) + len(separator)

        if current_length + segment_length <= chunk_size:
            # Add to current chunk
            current_chunk.append(segment)
            current_length += segment_length
        else:
            # Save current chunk
            if current_chunk:
                chunk_text = separator.join(current_chunk)
                chunks.append((chunk_text, current_start, current_start + len(chunk_text)))

                # Start new chunk with overlap
                overlap_text = separator.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                current_start = current_start + len(chunk_text) - len(overlap_text)
                current_chunk = [overlap_text, segment]
                current_length = len(overlap_text) + segment_length
            else:
                # Segment too large, split by character
                for i in range(0, len(segment), chunk_size - chunk_overlap):
                    sub_chunk = segment[i:i + chunk_size]
                    chunks.append((sub_chunk, current_start + i, current_start + i + len(sub_chunk)))
                current_chunk = []
                current_length = 0
                current_start += len(segment)

    # Add final chunk
    if current_chunk:
        chunk_text = separator.join(current_chunk)
        chunks.append((chunk_text, current_start, current_start + len(chunk_text)))

    return chunks


# ============================================================================
# Embedding Generator
# ============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers (CPU-only).
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        try:
            from sentence_transformers import SentenceTransformer

            # Force CPU device
            self.model = SentenceTransformer(model_name, device='cpu')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"  Model loaded (dim={self.embedding_dim})")

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True):
        """
        Generate embeddings for list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings


# ============================================================================
# Vector Database
# ============================================================================

class VectorDatabase:
    """
    Vector database wrapper supporting FAISS and ChromaDB.
    """

    def __init__(
        self,
        db_type: str = 'faiss',
        embedding_dim: int = 768,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize vector database.

        Args:
            db_type: 'faiss' or 'chromadb'
            embedding_dim: Dimension of embeddings
            output_dir: Directory to save database
        """
        self.db_type = db_type
        self.embedding_dim = embedding_dim
        self.output_dir = output_dir

        if db_type == 'faiss':
            self._init_faiss()
        elif db_type == 'chromadb':
            self._init_chromadb()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            import numpy as np

            # Use CPU-only FAISS index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata_store = []

            logger.info(f"FAISS index initialized (dim={self.embedding_dim})")

        except ImportError:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")

    def _init_chromadb(self):
        """Initialize ChromaDB."""
        try:
            import chromadb

            if self.output_dir:
                # Persistent storage
                self.client = chromadb.PersistentClient(path=str(self.output_dir / 'chromadb'))
            else:
                # In-memory
                self.client = chromadb.Client()

            self.collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"embedding_dim": self.embedding_dim}
            )

            logger.info("ChromaDB initialized")

        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")

    def add_vectors(
        self,
        embeddings,
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add vectors to database.

        Args:
            embeddings: Numpy array of embeddings
            chunk_ids: List of chunk IDs
            metadatas: List of metadata dicts
        """
        if self.db_type == 'faiss':
            import numpy as np

            # Add to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))

            # Store metadata
            for chunk_id, metadata in zip(chunk_ids, metadatas):
                self.metadata_store.append({
                    'chunk_id': chunk_id,
                    **metadata
                })

        elif self.db_type == 'chromadb':
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )

    def save(self, output_dir: Path):
        """Save database to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.db_type == 'faiss':
            import faiss

            # Save FAISS index
            index_path = output_dir / 'faiss.index'
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_store, f, indent=2)

            logger.info(f"FAISS database saved to {output_dir}")

        elif self.db_type == 'chromadb':
            # ChromaDB auto-saves if persistent
            logger.info(f"ChromaDB saved to {output_dir / 'chromadb'}")


# ============================================================================
# Knowledge Base Builder
# ============================================================================

def build_knowledge_base(
    input_dir: Path,
    output_dir: Path,
    embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    vector_db: str = 'faiss',
    batch_size: int = 32,
    memory_monitor: Optional[MemoryMonitor] = None,
) -> Dict[str, Any]:
    """
    Build knowledge base from documents.

    Args:
        input_dir: Directory containing documents
        output_dir: Directory to save knowledge base
        embedding_model: Sentence-transformers model name
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap between chunks
        vector_db: Vector database type ('faiss' or 'chromadb')
        batch_size: Batch size for embedding
        memory_monitor: Optional memory monitor

    Returns:
        Build report dict
    """
    logger.info("=" * 80)
    logger.info("KNOWLEDGE BASE BUILDER")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Chunk size: {chunk_size} (overlap: {chunk_overlap})")
    logger.info(f"Vector DB: {vector_db}")

    # Find all documents
    supported_extensions = ('.txt', '.md', '.markdown', '.pdf', '.jpg', '.jpeg', '.png')
    document_files = []
    for ext in supported_extensions:
        document_files.extend(input_dir.glob(f"**/*{ext}"))

    logger.info(f"Found {len(document_files)} documents")

    if len(document_files) == 0:
        logger.warning("No documents found")
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'no_documents',
            'total_documents': 0,
        }

    # Initialize components
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    vector_database = VectorDatabase(
        db_type=vector_db,
        embedding_dim=embedding_generator.embedding_dim,
        output_dir=output_dir
    )

    # Process documents
    all_documents = []
    all_chunks = []
    chunk_texts = []

    for i, doc_path in enumerate(document_files):
        try:
            # Memory safety check
            if memory_monitor:
                is_safe, level, info = memory_monitor.check_safety()
                if not is_safe:
                    logger.warning(f"Memory level {level} - stopping early")
                    break

            logger.info(f"[{i+1}/{len(document_files)}] Processing: {doc_path.name}")

            # Load document
            content, doc_type = load_document(doc_path)

            if not content.strip():
                logger.warning(f"  Empty document, skipping")
                continue

            # Generate document ID
            doc_id = hashlib.md5(str(doc_path).encode()).hexdigest()[:16]

            # Chunk document
            chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            logger.info(f"  Created {len(chunks)} chunks")

            # Create chunk objects
            doc_chunks = []
            for chunk_idx, (text_content, start_idx, end_idx) in enumerate(chunks):
                chunk_id = f"{doc_id}_{chunk_idx}"

                chunk_obj = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=text_content,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    metadata={
                        'source_path': str(doc_path),
                        'doc_type': doc_type,
                        'chunk_index': chunk_idx,
                    }
                )

                doc_chunks.append(chunk_obj)
                all_chunks.append(chunk_obj)
                chunk_texts.append(text_content)

            # Create document object
            document = Document(
                doc_id=doc_id,
                content=content,
                source_path=str(doc_path),
                doc_type=doc_type,
                metadata={
                    'file_size_bytes': doc_path.stat().st_size,
                    'file_name': doc_path.name,
                },
                chunk_ids=[c.chunk_id for c in doc_chunks],
                timestamp=datetime.now().isoformat()
            )

            all_documents.append(document)

        except Exception as e:
            logger.error(f"Failed to process {doc_path}: {e}")
            continue

    logger.info(f"\nGenerated {len(all_chunks)} chunks from {len(all_documents)} documents")

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_generator.embed_texts(chunk_texts, batch_size=batch_size)

    # Add to vector database
    logger.info("Adding to vector database...")
    vector_database.add_vectors(
        embeddings=embeddings,
        chunk_ids=[c.chunk_id for c in all_chunks],
        metadatas=[c.metadata for c in all_chunks]
    )

    # Save database
    logger.info("Saving vector database...")
    vector_database.save(output_dir)

    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'embedding_model': embedding_model,
        'embedding_dim': embedding_generator.embedding_dim,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'vector_db': vector_db,
        'total_documents': len(all_documents),
        'total_chunks': len(all_chunks),
        'documents': [asdict(doc) for doc in all_documents],
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Knowledge base built: {output_dir}")
    logger.info(f"  Documents: {len(all_documents)}")
    logger.info(f"  Chunks: {len(all_chunks)}")
    logger.info(f"  Embeddings: {embeddings.shape}")

    return manifest


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build knowledge base for RAG from documents'
    )

    # Input/output
    parser.add_argument('--input-dir', type=Path, required=True,
                       help='Directory containing documents')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for knowledge base')

    # Embedding
    parser.add_argument('--embedding-model', type=str,
                       default='sentence-transformers/all-mpnet-base-v2',
                       help='Sentence-transformers model name')

    # Chunking
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Chunk size in characters (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=128,
                       help='Overlap between chunks (default: 128)')

    # Vector DB
    parser.add_argument('--vector-db', type=str, default='faiss',
                       choices=['faiss', 'chromadb'],
                       help='Vector database type (default: faiss)')

    # Processing
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding (default: 32)')

    # Safety
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight safety checks (not recommended)')

    args = parser.parse_args()

    # Enforce CPU-only
    enforce_cpu_only()

    # Run preflight checks
    if not args.skip_preflight:
        logger.info("Running preflight checks...")
        try:
            run_preflight(strict=True)
        except Exception as e:
            logger.warning(f"Preflight checks failed: {e}")
            logger.warning("Continuing anyway (use --skip-preflight to suppress this)")

    # Create memory monitor
    memory_monitor = MemoryMonitor()

    # Start runtime monitoring
    with RuntimeMonitor(check_interval=30.0) as monitor:
        # Build knowledge base
        build_knowledge_base(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vector_db=args.vector_db,
            batch_size=args.batch_size,
            memory_monitor=memory_monitor,
        )


if __name__ == '__main__':
    main()
