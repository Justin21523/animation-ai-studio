"""
Text Chunker

Intelligent text chunking with semantic boundaries, token counting, and overlap.

Features:
- Multiple chunking strategies (fixed-size, semantic, sentence, sliding-window)
- Respect markdown headers and paragraph boundaries
- Token counting and validation
- Configurable chunk size, overlap, and boundaries
- Metadata extraction (position, token count, chunk index)

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..common import (
    Document,
    DocumentChunk,
    ChunkingStrategy,
    ChunkingConfig,
    generate_chunk_id
)

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Intelligent text chunker with semantic boundaries

    Features:
    - Multiple chunking strategies
    - Token-aware chunking
    - Semantic boundary respect (paragraphs, sentences, headers)
    - Sliding window with overlap
    - Metadata extraction (position, token count)
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize text chunker

        Args:
            config: Chunking configuration (default: ChunkingConfig with defaults)
        """
        self.config = config or ChunkingConfig()

        # Simple token estimation (whitespace-based)
        # More accurate: use tiktoken/transformers tokenizer
        self.token_estimator = self._simple_token_estimator

        logger.info(f"TextChunker initialized: strategy={self.config.strategy.value}, "
                   f"chunk_size={self.config.chunk_size}, overlap={self.config.overlap}")

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk document into smaller pieces

        Args:
            document: Document to chunk

        Returns:
            List of DocumentChunk objects
        """
        content = document.content

        if not content or len(content.strip()) == 0:
            logger.warning(f"Empty document: {document.id}")
            return []

        # Select chunking strategy
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(content)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(content)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_sentence(content)
        elif self.config.strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self._chunk_sliding_window(content)
        else:
            logger.warning(f"Unknown strategy {self.config.strategy}, using fixed-size")
            chunks = self._chunk_fixed_size(content)

        # Convert to DocumentChunk objects
        document_chunks = []
        for idx, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            # Token counting
            token_count = self.token_estimator(chunk_text)
            char_count = len(chunk_text)

            # Create chunk
            chunk = DocumentChunk(
                id=generate_chunk_id(document.id, idx),
                document_id=document.id,
                content=chunk_text,
                chunk_index=idx,
                token_count=token_count,
                char_count=char_count,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata={
                    "doc_type": document.doc_type.value,
                    "doc_path": str(document.path),
                    "strategy": self.config.strategy.value
                }
            )
            document_chunks.append(chunk)

        logger.info(f"Chunked document {document.id}: {len(document_chunks)} chunks")

        return document_chunks

    def batch_chunk(self, documents: List[Document]) -> Dict[str, List[DocumentChunk]]:
        """
        Batch chunk multiple documents

        Args:
            documents: List of documents

        Returns:
            Dictionary mapping document IDs to chunk lists
        """
        results = {}

        for document in documents:
            try:
                chunks = self.chunk_document(document)
                results[document.id] = chunks
            except Exception as e:
                logger.error(f"Failed to chunk document {document.id}: {e}")
                results[document.id] = []

        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(f"Batch chunked {len(documents)} documents: {total_chunks} total chunks")

        return results

    def _chunk_fixed_size(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Fixed-size chunking with token-based boundaries

        Args:
            content: Text content

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        chunks = []
        words = content.split()

        current_chunk = []
        current_tokens = 0
        start_pos = 0
        char_offset = 0

        for word in words:
            word_tokens = self.token_estimator(word)

            if current_tokens + word_tokens > self.config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                end_pos = char_offset
                chunks.append((chunk_text, start_pos, end_pos))

                # Overlap: keep last N tokens
                if self.config.overlap > 0:
                    overlap_size = min(self.config.overlap, len(current_chunk))
                    current_chunk = current_chunk[-overlap_size:]
                    current_tokens = sum(self.token_estimator(w) for w in current_chunk)
                    start_pos = end_pos - len(" ".join(current_chunk))
                else:
                    current_chunk = []
                    current_tokens = 0
                    start_pos = char_offset

            current_chunk.append(word)
            current_tokens += word_tokens
            char_offset += len(word) + 1  # +1 for space

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, start_pos, len(content)))

        return chunks

    def _chunk_semantic(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Semantic chunking respecting paragraph boundaries

        Args:
            content: Text content

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # Split by paragraphs (double newline or markdown headers)
        paragraphs = re.split(r'\n\s*\n|^#+\s+', content, flags=re.MULTILINE)

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_estimator(para)

            if current_tokens + para_tokens > self.config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = "\n\n".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))

                # No overlap in semantic mode (paragraph boundaries)
                current_chunk = []
                current_tokens = 0
                start_pos = end_pos + 2  # +2 for double newline

            current_chunk.append(para)
            current_tokens += para_tokens

        # Final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))

        return chunks

    def _chunk_sentence(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Sentence-based chunking

        Args:
            content: Text content

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        # Simple sentence splitting (periods, exclamation marks, question marks)
        sentences = re.split(r'(?<=[.!?])\s+', content)

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self.token_estimator(sentence)

            if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))

                # Overlap: keep last N sentences
                if self.config.overlap > 0:
                    overlap_count = min(1, len(current_chunk))  # Keep 1 sentence overlap
                    current_chunk = current_chunk[-overlap_count:]
                    current_tokens = sum(self.token_estimator(s) for s in current_chunk)
                    start_pos = end_pos - len(" ".join(current_chunk))
                else:
                    current_chunk = []
                    current_tokens = 0
                    start_pos = end_pos + 1

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, start_pos, start_pos + len(chunk_text)))

        return chunks

    def _chunk_sliding_window(self, content: str) -> List[Tuple[str, int, int]]:
        """
        Sliding window chunking with fixed overlap

        Args:
            content: Text content

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        words = content.split()
        chunks = []

        # Calculate step size (chunk_size - overlap)
        step_size = max(1, self.config.chunk_size - self.config.overlap)

        start_idx = 0
        char_offset = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.config.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            start_pos = char_offset
            end_pos = char_offset + len(chunk_text)

            chunks.append((chunk_text, start_pos, end_pos))

            # Move window
            start_idx += step_size
            char_offset += sum(len(w) + 1 for w in chunk_words[:step_size])

            # Stop if we've reached the end
            if end_idx >= len(words):
                break

        return chunks

    def _simple_token_estimator(self, text: str) -> int:
        """
        Simple token estimation (whitespace-based)

        For production, replace with tiktoken or transformers tokenizer

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~1.3 tokens per word for English
        words = len(text.split())
        return int(words * 1.3)
