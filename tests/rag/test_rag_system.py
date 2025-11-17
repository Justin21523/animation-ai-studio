"""
Unit Tests for RAG System

Tests all RAG components:
- Vector Store
- Embedding Generator
- Document Processor
- Retrieval Engine
- Knowledge Base

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag import (
    VectorStoreType,
    VectorStoreConfig,
    VectorStoreFactory,
    EmbeddingGenerator,
    EmbeddingConfig,
    DocumentProcessor,
    Document,
    DocumentType,
    ChunkingConfig,
    RetrievalEngine,
    RetrievalConfig,
    KnowledgeBase,
    KnowledgeBaseConfig
)


# Fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(
            doc_id="doc_001",
            content="Luca is a young sea monster living off the Italian Riviera.",
            doc_type=DocumentType.CHARACTER_PROFILE,
            metadata={"character": "luca", "film": "luca"}
        ),
        Document(
            doc_id="doc_002",
            content="Alberto is Luca's best friend, a fellow sea monster.",
            doc_type=DocumentType.CHARACTER_PROFILE,
            metadata={"character": "alberto", "film": "luca"}
        ),
        Document(
            doc_id="doc_003",
            content="Portorosso is a beautiful Italian coastal town.",
            doc_type=DocumentType.SCENE_DESCRIPTION,
            metadata={"location": "portorosso", "film": "luca"}
        )
    ]


# Vector Store Tests
class TestVectorStore:
    """Tests for vector store implementations"""

    def test_faiss_creation(self, temp_dir):
        """Test FAISS vector store creation"""
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            persist_dir=temp_dir,
            dimension=128
        )
        store = VectorStoreFactory.create(config)
        assert store is not None
        assert store.count() == 0

    def test_faiss_add_and_search(self, temp_dir):
        """Test adding and searching in FAISS"""
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            persist_dir=temp_dir,
            dimension=128
        )
        store = VectorStoreFactory.create(config)

        # Add documents
        embeddings = np.random.randn(3, 128).astype(np.float32)
        doc_ids = ["doc_1", "doc_2", "doc_3"]
        contents = ["Content 1", "Content 2", "Content 3"]
        metadata = [
            {"type": "test"},
            {"type": "test"},
            {"type": "test"}
        ]

        store.add_documents(doc_ids, embeddings, contents, metadata)

        assert store.count() == 3

        # Search
        query_embedding = embeddings[0:1]  # Use first embedding as query
        results = store.search(query_embedding, top_k=2)

        assert len(results) > 0
        assert results[0].doc_id in doc_ids

    def test_faiss_save_and_load(self, temp_dir):
        """Test FAISS persistence"""
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            persist_dir=temp_dir,
            dimension=128
        )

        # Create and populate store
        store1 = VectorStoreFactory.create(config)
        embeddings = np.random.randn(2, 128).astype(np.float32)
        store1.add_documents(
            ["doc_1", "doc_2"],
            embeddings,
            ["Content 1", "Content 2"],
            [{"type": "test"}, {"type": "test"}]
        )
        store1.save()

        # Load in new instance
        store2 = VectorStoreFactory.create(config)
        store2.load()

        assert store2.count() == 2


# Document Processor Tests
class TestDocumentProcessor:
    """Tests for document processor"""

    def test_process_text(self, temp_dir):
        """Test processing plain text"""
        processor = DocumentProcessor()

        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test document for processing."
        test_file.write_text(test_content, encoding='utf-8')

        docs = processor.process_file(test_file, DocumentType.TEXT)

        assert len(docs) > 0
        assert docs[0].doc_type == DocumentType.TEXT
        assert test_content in docs[0].content

    def test_process_json(self, temp_dir):
        """Test processing JSON"""
        import json

        processor = DocumentProcessor()

        # Create test file
        test_file = Path(temp_dir) / "test.json"
        test_data = {
            "character": "Luca",
            "description": "Young sea monster"
        }
        test_file.write_text(json.dumps(test_data), encoding='utf-8')

        docs = processor.process_file(test_file, DocumentType.JSON)

        assert len(docs) > 0
        assert docs[0].doc_type == DocumentType.JSON

    def test_text_chunking(self):
        """Test text chunking"""
        processor = DocumentProcessor(
            ChunkingConfig(chunk_size=50, chunk_overlap=10)
        )

        long_text = "A" * 200  # 200 character text
        chunks = processor._chunk_text(long_text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)

    def test_markdown_sections(self):
        """Test markdown section splitting"""
        processor = DocumentProcessor()

        markdown_content = """
# Section 1
Content for section 1

## Section 2
Content for section 2
"""
        sections = processor._split_markdown_sections(markdown_content)

        assert len(sections) >= 2


# Embedding Generator Tests (with mocking)
class TestEmbeddingGenerator:
    """Tests for embedding generator"""

    @pytest.mark.asyncio
    async def test_embedding_dimension(self):
        """Test embedding dimension"""
        config = EmbeddingConfig(dimension=1024)
        # Note: This will use placeholder implementation
        async with EmbeddingGenerator(config=config) as generator:
            assert generator.get_dimension() == 1024

    @pytest.mark.asyncio
    async def test_embedding_normalization(self):
        """Test embedding normalization"""
        config = EmbeddingConfig(dimension=128, normalize=True)

        async with EmbeddingGenerator(config=config) as generator:
            embedding = await generator.generate_embedding("test text")

            # Check normalization (unit length)
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01  # Close to 1.0


# Retrieval Engine Tests
class TestRetrievalEngine:
    """Tests for retrieval engine"""

    @pytest.mark.asyncio
    async def test_basic_retrieval(self, temp_dir):
        """Test basic retrieval"""
        # Setup vector store
        vector_config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            persist_dir=temp_dir,
            dimension=128
        )
        vector_store = VectorStoreFactory.create(vector_config)

        # Add documents
        embeddings = np.random.randn(3, 128).astype(np.float32)
        vector_store.add_documents(
            ["doc_1", "doc_2", "doc_3"],
            embeddings,
            ["Luca content", "Alberto content", "Portorosso content"],
            [{"type": "char"}, {"type": "char"}, {"type": "location"}]
        )

        # Setup retrieval engine
        embedding_config = EmbeddingConfig(dimension=128)
        async with EmbeddingGenerator(config=embedding_config) as embedding_gen:
            retrieval_config = RetrievalConfig(top_k=2)
            engine = RetrievalEngine(vector_store, embedding_gen, retrieval_config)

            # Test retrieval
            results = await engine.retrieve("Luca")

            assert len(results.documents) > 0


# Knowledge Base Tests
class TestKnowledgeBase:
    """Tests for knowledge base"""

    @pytest.mark.asyncio
    async def test_kb_initialization(self, temp_dir):
        """Test knowledge base initialization"""
        config = KnowledgeBaseConfig(
            persist_dir=temp_dir,
            vector_store_type="faiss",
            embedding_dimension=128
        )

        async with KnowledgeBase(config=config) as kb:
            assert kb._initialized
            assert kb.vector_store is not None

    @pytest.mark.asyncio
    async def test_add_documents(self, temp_dir, sample_documents):
        """Test adding documents"""
        config = KnowledgeBaseConfig(
            persist_dir=temp_dir,
            vector_store_type="faiss",
            embedding_dimension=128
        )

        async with KnowledgeBase(config=config) as kb:
            await kb.add_documents(sample_documents)

            stats = kb.get_stats()
            assert stats["total_documents"] == len(sample_documents)

    @pytest.mark.asyncio
    async def test_search(self, temp_dir, sample_documents):
        """Test search functionality"""
        config = KnowledgeBaseConfig(
            persist_dir=temp_dir,
            vector_store_type="faiss",
            embedding_dimension=128,
            default_top_k=2
        )

        async with KnowledgeBase(config=config) as kb:
            await kb.add_documents(sample_documents)

            results = await kb.search("Luca character")

            assert len(results.documents) > 0

    @pytest.mark.asyncio
    async def test_get_context_for_llm(self, temp_dir, sample_documents):
        """Test context generation for LLM"""
        config = KnowledgeBaseConfig(
            persist_dir=temp_dir,
            vector_store_type="faiss",
            embedding_dimension=128
        )

        async with KnowledgeBase(config=config) as kb:
            await kb.add_documents(sample_documents)

            context = await kb.get_context_for_llm("Who is Luca?")

            assert isinstance(context, str)
            # Context should contain relevant information


# Integration Tests
class TestRAGIntegration:
    """Integration tests for complete RAG pipeline"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, temp_dir):
        """Test complete RAG workflow"""
        # 1. Create knowledge base
        config = KnowledgeBaseConfig(
            persist_dir=temp_dir,
            vector_store_type="faiss",
            embedding_dimension=128
        )

        async with KnowledgeBase(config=config) as kb:
            # 2. Add documents from file
            test_file = Path(temp_dir) / "test_character.txt"
            test_file.write_text(
                "Luca is a curious young sea monster who dreams of adventure.",
                encoding='utf-8'
            )

            count = await kb.add_document_from_file(test_file)
            assert count > 0

            # 3. Search
            results = await kb.search("sea monster adventure")
            assert len(results.documents) > 0

            # 4. Get stats
            stats = kb.get_stats()
            assert stats["total_documents"] > 0


def test_config_validation():
    """Test configuration validation"""
    config = KnowledgeBaseConfig()

    assert config.vector_store_type in ["faiss", "chroma"]
    assert config.embedding_dimension > 0
    assert config.chunk_size > 0


def test_document_metadata():
    """Test document metadata handling"""
    doc = Document(
        doc_id="test_001",
        content="Test content",
        doc_type=DocumentType.TEXT,
        metadata={"key": "value"}
    )

    # Test to_dict
    doc_dict = doc.to_dict()
    assert "doc_id" in doc_dict
    assert "metadata" in doc_dict
    assert doc_dict["metadata"]["key"] == "value"

    # Test from_dict
    doc2 = Document.from_dict(doc_dict)
    assert doc2.doc_id == doc.doc_id
    assert doc2.metadata["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
