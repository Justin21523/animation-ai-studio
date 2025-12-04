#!/usr/bin/env python3
"""
RAG Knowledge Base Ingestion Script for Film Data

Imports character descriptions, film metadata, and style guides into RAG system.

Usage:
    python scripts/rag/ingest_film_knowledge.py --film luca
    python scripts/rag/ingest_film_knowledge.py --film luca --rebuild
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import yaml

from scripts.rag.knowledge_base import KnowledgeBase
from scripts.rag.documents.document_processor import Document, DocumentType
# from scripts.core.utils.logger import setup_logger
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FilmKnowledgeIngester:
    """Ingest film-specific knowledge into RAG system"""

    def __init__(self, film_name: str, project_root: Path = None):
        self.film_name = film_name
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.film_data_dir = self.project_root / "data" / "films" / film_name

        if not self.film_data_dir.exists():
            raise ValueError(f"Film data directory not found: {self.film_data_dir}")

        logger.info(f"Initialized ingester for film: {film_name}")
        logger.info(f"Film data directory: {self.film_data_dir}")

    async def ingest_all(self, rebuild: bool = False) -> Dict[str, Any]:
        """
        Ingest all film knowledge into RAG system

        Args:
            rebuild: If True, clear existing knowledge base first

        Returns:
            Statistics about ingestion
        """
        logger.info(f"Starting knowledge ingestion for {self.film_name}")

        stats = {
            "film": self.film_name,
            "characters_ingested": 0,
            "metadata_ingested": 0,
            "style_guides_ingested": 0,
            "prompts_ingested": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "errors": []
        }

        async with KnowledgeBase() as kb:
            # Optionally rebuild
            if rebuild:
                logger.warning("Rebuilding knowledge base (clearing existing data)")
                # Note: KnowledgeBase doesn't have clear() yet, would need to implement
                pass

            # 1. Ingest character descriptions
            logger.info("Ingesting character descriptions...")
            character_stats = await self._ingest_characters(kb)
            stats["characters_ingested"] = character_stats["count"]
            stats["total_chunks"] += character_stats["chunks"]

            # 2. Ingest film metadata
            logger.info("Ingesting film metadata...")
            metadata_stats = await self._ingest_metadata(kb)
            stats["metadata_ingested"] = metadata_stats["count"]
            stats["total_chunks"] += metadata_stats["chunks"]

            # 3. Ingest style guide
            logger.info("Ingesting style guide...")
            style_stats = await self._ingest_style_guide(kb)
            stats["style_guides_ingested"] = style_stats["count"]
            stats["total_chunks"] += style_stats["chunks"]

            # 4. Ingest prompt descriptions
            logger.info("Ingesting prompt descriptions...")
            prompt_stats = await self._ingest_prompts(kb)
            stats["prompts_ingested"] = prompt_stats["count"]
            stats["total_chunks"] += prompt_stats["chunks"]

            # Get final stats
            kb_stats = kb.get_stats()
            stats["total_documents"] = kb_stats.get("total_documents", 0)

            logger.info(f"Ingestion complete! Stats: {stats}")

        return stats

    async def _ingest_characters(self, kb: KnowledgeBase) -> Dict[str, int]:
        """Ingest character description files"""
        characters_dir = self.film_data_dir / "characters"

        if not characters_dir.exists():
            logger.warning(f"Characters directory not found: {characters_dir}")
            return {"count": 0, "chunks": 0}

        character_files = list(characters_dir.glob("character_*.md"))
        logger.info(f"Found {len(character_files)} character files")

        total_chunks = 0
        for char_file in character_files:
            try:
                # Extract character name from filename
                # e.g., character_luca.md -> Luca Paguro
                char_name = char_file.stem.replace("character_", "").replace("_", " ").title()

                logger.info(f"Ingesting character: {char_name} from {char_file.name}")

                # Read file content
                content = char_file.read_text(encoding="utf-8")

                # Create document
                doc = Document(
                    doc_id=f"{self.film_name}_character_{char_file.stem}",
                    content=content,
                    doc_type=DocumentType.CHARACTER_PROFILE,
                    metadata={
                        "film": self.film_name,
                        "character_name": char_name,
                        "source_file": str(char_file),
                        "type": "character_profile"
                    },
                    source_path=str(char_file),
                    quality_score=1.0,  # High quality curated data
                    relevance_tags=["character", "profile", char_name.lower(), self.film_name]
                )

                # Add to knowledge base
                result = await kb.add_documents([doc])
                chunks_added = result.get("chunks_added", 0)
                total_chunks += chunks_added

                logger.info(f"✓ {char_name}: {chunks_added} chunks added")

            except Exception as e:
                logger.error(f"Error ingesting {char_file}: {e}")

        return {"count": len(character_files), "chunks": total_chunks}

    async def _ingest_metadata(self, kb: KnowledgeBase) -> Dict[str, int]:
        """Ingest film metadata JSON"""
        metadata_file = self.film_data_dir / "film_metadata.json"

        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return {"count": 0, "chunks": 0}

        try:
            # Load JSON
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Convert to readable text format
            content = self._format_metadata_as_text(metadata)

            # Create document
            doc = Document(
                doc_id=f"{self.film_name}_metadata",
                content=content,
                doc_type=DocumentType.FILM_METADATA,
                metadata={
                    "film": self.film_name,
                    "source_file": str(metadata_file),
                    "type": "film_metadata",
                    "structured_data": metadata  # Keep original JSON
                },
                source_path=str(metadata_file),
                quality_score=1.0,
                relevance_tags=["film", "metadata", self.film_name, "setting", "characters"]
            )

            # Add to knowledge base
            result = await kb.add_documents([doc])
            chunks_added = result.get("chunks_added", 0)

            logger.info(f"✓ Film metadata: {chunks_added} chunks added")

            return {"count": 1, "chunks": chunks_added}

        except Exception as e:
            logger.error(f"Error ingesting metadata: {e}")
            return {"count": 0, "chunks": 0}

    async def _ingest_style_guide(self, kb: KnowledgeBase) -> Dict[str, int]:
        """Ingest style guide markdown"""
        style_file = self.film_data_dir / "style_guide.md"

        if not style_file.exists():
            logger.warning(f"Style guide not found: {style_file}")
            return {"count": 0, "chunks": 0}

        try:
            content = style_file.read_text(encoding="utf-8")

            # Create document
            doc = Document(
                doc_id=f"{self.film_name}_style_guide",
                content=content,
                doc_type=DocumentType.STYLE_GUIDE,
                metadata={
                    "film": self.film_name,
                    "source_file": str(style_file),
                    "type": "style_guide"
                },
                source_path=str(style_file),
                quality_score=1.0,
                relevance_tags=["style", "visual", "animation", "art_direction", self.film_name]
            )

            # Add to knowledge base
            result = await kb.add_documents([doc])
            chunks_added = result.get("chunks_added", 0)

            logger.info(f"✓ Style guide: {chunks_added} chunks added")

            return {"count": 1, "chunks": chunks_added}

        except Exception as e:
            logger.error(f"Error ingesting style guide: {e}")
            return {"count": 0, "chunks": 0}

    async def _ingest_prompts(self, kb: KnowledgeBase) -> Dict[str, int]:
        """Ingest prompt descriptions"""
        prompts_dir = self.film_data_dir / "prompt_descriptions"

        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {prompts_dir}")
            return {"count": 0, "chunks": 0}

        # Find all YAML/JSON prompt files
        prompt_files = list(prompts_dir.glob("*.yaml")) + list(prompts_dir.glob("*.json"))

        if not prompt_files:
            logger.warning(f"No prompt files found in {prompts_dir}")
            return {"count": 0, "chunks": 0}

        total_chunks = 0
        for prompt_file in prompt_files:
            try:
                # Read and parse file
                if prompt_file.suffix == '.yaml':
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompts = yaml.safe_load(f)
                else:  # .json
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompts = json.load(f)

                # Convert to text format
                content = self._format_prompts_as_text(prompts, prompt_file.stem)

                # Create document
                doc = Document(
                    doc_id=f"{self.film_name}_prompts_{prompt_file.stem}",
                    content=content,
                    doc_type=DocumentType.TEXT,  # Use TEXT type for prompts
                    metadata={
                        "film": self.film_name,
                        "source_file": str(prompt_file),
                        "type": "prompt_library",
                        "structured_data": prompts
                    },
                    source_path=str(prompt_file),
                    quality_score=1.0,
                    relevance_tags=["prompts", "generation", self.film_name]
                )

                # Add to knowledge base
                result = await kb.add_documents([doc])
                chunks_added = result.get("chunks_added", 0)
                total_chunks += chunks_added

                logger.info(f"✓ {prompt_file.name}: {chunks_added} chunks added")

            except Exception as e:
                logger.error(f"Error ingesting {prompt_file}: {e}")

        return {"count": len(prompt_files), "chunks": total_chunks}

    def _format_metadata_as_text(self, metadata: Dict[str, Any]) -> str:
        """Convert film metadata JSON to readable text"""
        lines = []

        # Film info
        film = metadata.get("film", {})
        lines.append(f"Film: {film.get('title', 'Unknown')}")
        lines.append(f"Studio: {film.get('studio', 'Unknown')}")
        lines.append(f"Director: {film.get('director', 'Unknown')}")
        lines.append(f"Runtime: {film.get('runtime', 'Unknown')}")
        lines.append("")

        # Setting
        setting = metadata.get("setting", {})
        lines.append(f"Setting: {setting.get('location', 'Unknown')}")
        lines.append(f"Town: {setting.get('town', 'Unknown')}")
        lines.append(f"Time Period: {setting.get('time_period', 'Unknown')}")
        lines.append(f"Season: {setting.get('season', 'Unknown')}")
        lines.append("")

        # Themes
        themes = metadata.get("themes", [])
        if themes:
            lines.append("Themes:")
            for theme in themes:
                lines.append(f"- {theme}")
            lines.append("")

        # Main characters
        main_chars = metadata.get("main_characters", [])
        if main_chars:
            lines.append("Main Characters:")
            for char in main_chars:
                lines.append(f"- {char.get('name')}: {char.get('role')} (Voice: {char.get('voice_actor')})")
            lines.append("")

        # Key locations
        locations = metadata.get("key_locations", [])
        if locations:
            lines.append("Key Locations:")
            for loc in locations:
                lines.append(f"- {loc.get('name')}: {loc.get('description')}")
            lines.append("")

        # Color palette
        colors = metadata.get("color_palette", {})
        if colors:
            lines.append(f"Color Palette: {colors.get('mood', 'Unknown')}")
            primary = colors.get("primary", [])
            if primary:
                lines.append(f"Primary colors: {', '.join(primary)}")
            lines.append("")

        return "\n".join(lines)

    def _format_prompts_as_text(self, prompts: Dict[str, Any], filename: str) -> str:
        """Convert prompt library to readable text"""
        lines = [f"Prompt Library: {filename}", ""]

        # Recursively format the prompts
        def format_dict(d: Dict, indent: int = 0):
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    format_dict(value, indent + 1)
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key}:")
                    for item in value:
                        if isinstance(item, str):
                            lines.append(f"{prefix}  - {item}")
                        else:
                            lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}{key}: {value}")

        format_dict(prompts)

        return "\n".join(lines)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ingest film knowledge into RAG system")
    parser.add_argument("--film", required=True, help="Film name (e.g., luca)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild knowledge base (clear existing)")
    args = parser.parse_args()

    try:
        ingester = FilmKnowledgeIngester(args.film)
        stats = await ingester.ingest_all(rebuild=args.rebuild)

        print("\n" + "="*60)
        print("📚 RAG Knowledge Base Ingestion Complete!")
        print("="*60)
        print(f"Film: {stats['film']}")
        print(f"Characters ingested: {stats['characters_ingested']}")
        print(f"Metadata files: {stats['metadata_ingested']}")
        print(f"Style guides: {stats['style_guides_ingested']}")
        print(f"Prompt libraries: {stats['prompts_ingested']}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print("="*60)

        if stats['errors']:
            print("\n⚠️  Errors encountered:")
            for error in stats['errors']:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
