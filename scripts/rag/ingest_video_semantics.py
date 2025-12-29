#!/usr/bin/env python3
"""
Ingest Video Semantic Index into the RAG KnowledgeBase (P3).

This ingests the JSON documents emitted by:
  scripts/processing/video_semantics/video_semantic_builder.py

It is intentionally incremental and reproducible:
- Generates deterministic doc_ids per (film, video_id, doc_kind, item_id)
- Can skip existing doc_ids to avoid duplicate vectors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.rag.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from scripts.rag.documents.document_processor import Document, DocumentType


logger = logging.getLogger(__name__)


def _normalize_tag(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def _iter_documents_dirs(index_root: Path, video_id: Optional[str]) -> Iterable[Path]:
    if video_id:
        docs_dir = index_root / video_id / "documents"
        if docs_dir.exists():
            yield docs_dir
        return

    if not index_root.exists():
        return

    for child in sorted(index_root.iterdir()):
        if not child.is_dir():
            continue
        docs_dir = child / "documents"
        if docs_dir.exists():
            yield docs_dir


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _doc_identity(item: Dict[str, Any], fallback_index: int) -> Tuple[str, str]:
    """
    Return (doc_kind, id_suffix) used for deterministic doc_id generation.
    """
    doc_kind = str(item.get("doc_kind") or "unknown")

    if doc_kind == "video_scene":
        return doc_kind, f"scene_{item.get('scene_id', fallback_index)}"
    if doc_kind == "video_shot":
        return doc_kind, f"shot_{item.get('shot_id', fallback_index)}"
    if doc_kind == "dialogue":
        return doc_kind, f"line_{item.get('line_index', fallback_index)}"
    if doc_kind == "character_track":
        character = item.get("character") or item.get("character_id") or fallback_index
        return doc_kind, f"character_{character}"

    return doc_kind, f"item_{fallback_index}"


def _item_to_document(
    *,
    item: Dict[str, Any],
    source_path: Path,
    film: Optional[str],
    video_id: str,
    fallback_index: int,
) -> Document:
    doc_kind, id_suffix = _doc_identity(item, fallback_index)
    film_norm = _normalize_tag(item.get("film")) or film
    video_id_norm = str(item.get("video_id") or video_id)

    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        # Prefer explicit dialogue text if present.
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            content = text.strip()
        else:
            content = json.dumps(item, ensure_ascii=False)

    doc_id = f"{film_norm or 'unknown'}:{video_id_norm}:{doc_kind}:{id_suffix}"

    metadata: Dict[str, Any] = {
        "doc_kind": doc_kind,
        "film": film_norm,
        "video_id": video_id_norm,
        "source_file": str(source_path),
    }

    # Promote scalar fields to metadata so filters work (FAISS filters are equality-only).
    for k, v in item.items():
        if k in {"content", "text"}:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            metadata[k] = v

    # Keep a few useful list fields as metadata (not filterable but actionable for agents).
    for k in ("reference_timestamps", "reference_frame_paths"):
        v = item.get(k)
        if isinstance(v, list):
            metadata[k] = v

    return Document(
        doc_id=doc_id,
        content=str(content),
        doc_type=DocumentType.JSON,
        metadata=metadata,
        source_path=str(source_path),
        quality_score=1.0,
        relevance_tags=[t for t in [doc_kind, film_norm, video_id_norm] if t],
    )


async def ingest_video_semantics(
    *,
    index_root: Path,
    film: Optional[str],
    video_id: Optional[str],
    kb_persist_dir: Optional[str],
    kb_cache_dir: Optional[str],
    skip_existing: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    film_norm = _normalize_tag(film)

    kb_config = KnowledgeBaseConfig()
    if kb_persist_dir:
        kb_config.persist_dir = kb_persist_dir
    if kb_cache_dir:
        kb_config.cache_dir = kb_cache_dir

    stats: Dict[str, Any] = {
        "index_root": str(index_root),
        "film": film_norm,
        "video_id": video_id,
        "files_scanned": 0,
        "items_loaded": 0,
        "documents_prepared": 0,
        "documents_skipped_existing": 0,
        "documents_added": 0,
    }

    documents: List[Document] = []

    async with KnowledgeBase(config=kb_config) as kb:
        for docs_dir in _iter_documents_dirs(index_root, video_id):
            for json_path in sorted(docs_dir.glob("*.json")):
                items = _load_json_list(json_path)
                if not items:
                    continue
                stats["files_scanned"] += 1
                stats["items_loaded"] += len(items)

                inferred_video_id = docs_dir.parent.name
                for i, item in enumerate(items):
                    doc = _item_to_document(
                        item=item,
                        source_path=json_path,
                        film=film_norm,
                        video_id=inferred_video_id,
                        fallback_index=i,
                    )
                    stats["documents_prepared"] += 1

                    if skip_existing:
                        existing = kb.vector_store.get_document(doc.doc_id) if kb.vector_store else None
                        if existing is not None:
                            stats["documents_skipped_existing"] += 1
                            continue

                    documents.append(doc)

        if dry_run:
            return {**stats, "dry_run": True}

        if documents:
            await kb.add_documents(documents, show_progress=True)

        stats["documents_added"] = len(documents)
        return {**stats, "dry_run": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest video semantic index docs into KnowledgeBase")
    parser.add_argument("--film", help="Film name (e.g., luca); used for metadata + default paths")
    parser.add_argument("--video-id", help="Only ingest one video_id (directory name under video_index/)")
    parser.add_argument(
        "--index-root",
        help="Root directory containing video_index/<video_id>/documents/*.json",
    )
    parser.add_argument("--kb-persist-dir", help="Override KnowledgeBase persist dir (default: outputs/rag/knowledge_base)")
    parser.add_argument("--kb-cache-dir", help="Override embedding cache dir (default: outputs/rag/embedding_cache)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip existing doc_ids")
    parser.add_argument("--dry-run", action="store_true", help="Scan & report without ingesting")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    film_norm = _normalize_tag(args.film)
    if args.index_root:
        index_root = Path(args.index_root)
    else:
        if film_norm:
            index_root = Path("data/films") / film_norm / "rag" / "video_index"
        else:
            index_root = Path("outputs/rag/video_index")

    result = asyncio.run(
        ingest_video_semantics(
            index_root=index_root,
            film=film_norm,
            video_id=args.video_id,
            kb_persist_dir=args.kb_persist_dir,
            kb_cache_dir=args.kb_cache_dir,
            skip_existing=not args.no_skip_existing,
            dry_run=bool(args.dry_run),
        )
    )

    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

