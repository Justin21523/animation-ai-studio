"""
Document Processing for RAG System

Processes various document types for knowledge base ingestion.
Supports text, JSON, YAML, Markdown, and custom formats.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    CHARACTER_PROFILE = "character_profile"
    SCENE_DESCRIPTION = "scene_description"
    STYLE_GUIDE = "style_guide"
    FILM_METADATA = "film_metadata"


@dataclass
class Document:
    """
    Document representation

    Core unit for RAG system.
    Contains content, metadata, and embeddings.
    """
    doc_id: str
    content: str
    doc_type: DocumentType
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields
    embedding: Optional[Any] = None
    source_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Hierarchical structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Quality metrics
    quality_score: float = 1.0
    relevance_tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize timestamps"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "doc_type": self.doc_type.value,
            "metadata": self.metadata,
            "source_path": self.source_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "quality_score": self.quality_score,
            "relevance_tags": self.relevance_tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary"""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            doc_type=DocumentType(data["doc_type"]),
            metadata=data.get("metadata", {}),
            source_path=data.get("source_path"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            quality_score=data.get("quality_score", 1.0),
            relevance_tags=data.get("relevance_tags", [])
        )


@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    chunk_size: int = 512  # Target chunk size (characters)
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    respect_sentences: bool = True  # Don't split sentences
    respect_paragraphs: bool = True  # Preserve paragraph boundaries


class DocumentProcessor:
    """
    Process documents for RAG ingestion

    Features:
    - Multiple format support
    - Intelligent chunking
    - Metadata extraction
    - Quality validation
    - Hierarchical document structure
    """

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.chunking_config = chunking_config or ChunkingConfig()
        logger.info("DocumentProcessor initialized")

    def process_file(
        self,
        file_path: Union[str, Path],
        doc_type: Optional[DocumentType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process file into documents

        Args:
            file_path: Path to file
            doc_type: Document type (auto-detect if None)
            metadata: Additional metadata

        Returns:
            List of processed documents
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect document type
        if doc_type is None:
            doc_type = self._detect_type(file_path)

        # Read file content
        content = self._read_file(file_path)

        # Process based on type
        if doc_type == DocumentType.TEXT:
            docs = self._process_text(content, file_path, metadata)
        elif doc_type == DocumentType.JSON:
            docs = self._process_json(content, file_path, metadata)
        elif doc_type == DocumentType.YAML:
            docs = self._process_yaml(content, file_path, metadata)
        elif doc_type == DocumentType.MARKDOWN:
            docs = self._process_markdown(content, file_path, metadata)
        elif doc_type == DocumentType.CHARACTER_PROFILE:
            docs = self._process_character_profile(content, file_path, metadata)
        elif doc_type == DocumentType.SCENE_DESCRIPTION:
            docs = self._process_scene_description(content, file_path, metadata)
        elif doc_type == DocumentType.STYLE_GUIDE:
            docs = self._process_style_guide(content, file_path, metadata)
        elif doc_type == DocumentType.FILM_METADATA:
            docs = self._process_film_metadata(content, file_path, metadata)
        else:
            docs = self._process_text(content, file_path, metadata)

        logger.info(f"Processed {file_path}: {len(docs)} documents")
        return docs

    def _detect_type(self, file_path: Path) -> DocumentType:
        """Auto-detect document type from file"""
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return DocumentType.JSON
        elif suffix in [".yaml", ".yml"]:
            return DocumentType.YAML
        elif suffix in [".md", ".markdown"]:
            return DocumentType.MARKDOWN
        elif suffix == ".txt":
            # Check filename for hints
            name = file_path.stem.lower()
            if "character" in name:
                return DocumentType.CHARACTER_PROFILE
            elif "scene" in name:
                return DocumentType.SCENE_DESCRIPTION
            elif "style" in name:
                return DocumentType.STYLE_GUIDE
            return DocumentType.TEXT
        else:
            return DocumentType.TEXT

    def _read_file(self, file_path: Path) -> str:
        """Read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    def _generate_doc_id(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        if metadata and "id" in metadata:
            return f"{metadata['id']}_{content_hash}"
        return content_hash

    def _process_text(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process plain text document"""
        chunks = self._chunk_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = self._generate_doc_id(chunk, {"source": str(source_path), "chunk": i})

            doc = Document(
                doc_id=doc_id,
                content=chunk,
                doc_type=DocumentType.TEXT,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_file": str(source_path)
                },
                source_path=str(source_path)
            )
            documents.append(doc)

        return documents

    def _process_json(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process JSON document"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []

        # If data is a list, process each item
        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                doc_content = json.dumps(item, ensure_ascii=False, indent=2)
                doc_id = self._generate_doc_id(doc_content, {"source": str(source_path), "index": i})

                doc = Document(
                    doc_id=doc_id,
                    content=doc_content,
                    doc_type=DocumentType.JSON,
                    metadata={
                        **(metadata or {}),
                        "index": i,
                        "total_items": len(data),
                        "source_file": str(source_path),
                        **({k: v for k, v in item.items() if isinstance(v, (str, int, float, bool))})
                    },
                    source_path=str(source_path)
                )
                documents.append(doc)
            return documents

        # Single object
        doc_content = json.dumps(data, ensure_ascii=False, indent=2)
        doc_id = self._generate_doc_id(doc_content, {"source": str(source_path)})

        return [Document(
            doc_id=doc_id,
            content=doc_content,
            doc_type=DocumentType.JSON,
            metadata={
                **(metadata or {}),
                "source_file": str(source_path),
                **({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))})
            },
            source_path=str(source_path)
        )]

    def _process_yaml(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process YAML document"""
        if not YAML_AVAILABLE:
            logger.warning("YAML not available, processing as text")
            return self._process_text(content, source_path, metadata)

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML: {e}")
            return []

        # Convert to JSON-like processing
        json_content = json.dumps(data, ensure_ascii=False, indent=2)
        return self._process_json(json_content, source_path, metadata)

    def _process_markdown(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process Markdown document"""
        # Split by headers for better chunking
        sections = self._split_markdown_sections(content)

        documents = []
        for i, (title, section_content) in enumerate(sections):
            # Chunk each section if too long
            chunks = self._chunk_text(section_content)

            for j, chunk in enumerate(chunks):
                doc_id = self._generate_doc_id(
                    chunk,
                    {"source": str(source_path), "section": i, "chunk": j}
                )

                doc = Document(
                    doc_id=doc_id,
                    content=chunk,
                    doc_type=DocumentType.MARKDOWN,
                    metadata={
                        **(metadata or {}),
                        "section_title": title,
                        "section_index": i,
                        "chunk_index": j,
                        "source_file": str(source_path)
                    },
                    source_path=str(source_path)
                )
                documents.append(doc)

        return documents

    def _process_character_profile(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process character profile document"""
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            character_name = data.get("name", "Unknown")
            character_description = data.get("description", content)
        except json.JSONDecodeError:
            # Plain text character profile
            character_name = source_path.stem
            character_description = content

        doc_id = self._generate_doc_id(character_description, {"character": character_name})

        return [Document(
            doc_id=doc_id,
            content=character_description,
            doc_type=DocumentType.CHARACTER_PROFILE,
            metadata={
                **(metadata or {}),
                "character_name": character_name,
                "source_file": str(source_path)
            },
            source_path=str(source_path),
            relevance_tags=["character", character_name.lower()]
        )]

    def _process_scene_description(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process scene description document"""
        doc_id = self._generate_doc_id(content, {"source": str(source_path)})

        return [Document(
            doc_id=doc_id,
            content=content,
            doc_type=DocumentType.SCENE_DESCRIPTION,
            metadata={
                **(metadata or {}),
                "source_file": str(source_path)
            },
            source_path=str(source_path),
            relevance_tags=["scene", "description"]
        )]

    def _process_style_guide(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process style guide document"""
        chunks = self._chunk_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = self._generate_doc_id(chunk, {"source": str(source_path), "chunk": i})

            doc = Document(
                doc_id=doc_id,
                content=chunk,
                doc_type=DocumentType.STYLE_GUIDE,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "source_file": str(source_path)
                },
                source_path=str(source_path),
                relevance_tags=["style", "guide"]
            )
            documents.append(doc)

        return documents

    def _process_film_metadata(
        self,
        content: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process film metadata document"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._process_text(content, source_path, metadata)

        film_name = data.get("film", "Unknown")
        doc_content = json.dumps(data, ensure_ascii=False, indent=2)
        doc_id = self._generate_doc_id(doc_content, {"film": film_name})

        return [Document(
            doc_id=doc_id,
            content=doc_content,
            doc_type=DocumentType.FILM_METADATA,
            metadata={
                **(metadata or {}),
                "film_name": film_name,
                "source_file": str(source_path),
                **({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))})
            },
            source_path=str(source_path),
            relevance_tags=["film", film_name.lower(), "metadata"]
        )]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces

        Respects sentence and paragraph boundaries.
        """
        if len(text) <= self.chunking_config.chunk_size:
            return [text]

        chunks = []

        if self.chunking_config.respect_paragraphs:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunking_config.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"

            if current_chunk:
                chunks.append(current_chunk.strip())

        else:
            # Simple character-based chunking with overlap
            for i in range(0, len(text), self.chunking_config.chunk_size - self.chunking_config.chunk_overlap):
                chunk = text[i:i + self.chunking_config.chunk_size]
                if len(chunk) >= self.chunking_config.min_chunk_size:
                    chunks.append(chunk)

        return chunks

    def _split_markdown_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split markdown by sections (headers)"""
        import re

        sections = []
        current_title = "Introduction"
        current_content = []

        for line in content.split('\n'):
            # Check for header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))

                # Start new section
                current_title = header_match.group(2)
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))

        return sections


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    processor = DocumentProcessor()

    # Example: Process text file
    text_content = """
    Luca is a young sea monster living off the Italian Riviera.
    He dreams of exploring the human world above the surface.

    Alberto is Luca's best friend, a fellow sea monster.
    Together they discover the town of Portorosso.
    """

    # Create temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text_content)
        temp_path = f.name

    docs = processor.process_file(temp_path, DocumentType.TEXT)

    for doc in docs:
        print(f"\nDocument ID: {doc.doc_id}")
        print(f"Type: {doc.doc_type.value}")
        print(f"Content length: {len(doc.content)}")
        print(f"Metadata: {doc.metadata}")

    # Cleanup
    os.unlink(temp_path)


if __name__ == "__main__":
    main()
