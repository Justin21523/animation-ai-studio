"""
Knowledge Retrieval Tools for Agent Framework

Thin wrappers that connect agent tool calls to the RAG subsystem.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


async def search_character_knowledge(
    character: str,
    aspect: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search knowledge base for character information.

    Tool name: search_character_knowledge
    """
    from scripts.rag import KnowledgeBase

    query = f"{character} {aspect}".strip() if aspect else f"{character} character"
    filters = {"character": character.lower()}

    async with KnowledgeBase() as kb:
        results = await kb.search(query=query, top_k=5, filters=filters)
        context = await kb.get_context_for_llm(query=query, max_tokens=1200)

    return {
        "success": True,
        "query": query,
        "filters": filters,
        "results": [
            {"content": doc.content, "metadata": doc.metadata, "score": doc.score}
            for doc in results.documents
        ],
        "context": context,
        "stats": results.retrieval_stats,
    }


async def search_style_guide(style_name: str) -> Dict[str, Any]:
    """
    Search knowledge base for style guide information.

    Tool name: search_style_guide
    """
    from scripts.rag import KnowledgeBase

    query = f"{style_name} style guide visual characteristics"
    filters = {"style": style_name.lower()}

    async with KnowledgeBase() as kb:
        results = await kb.search(query=query, top_k=3, filters=filters)
        context = await kb.get_context_for_llm(query=query, max_tokens=900)

    return {
        "success": True,
        "query": query,
        "filters": filters,
        "results": [
            {"content": doc.content, "metadata": doc.metadata, "score": doc.score}
            for doc in results.documents
        ],
        "context": context,
        "stats": results.retrieval_stats,
    }


async def search_technical_parameters(task_type: str) -> Dict[str, Any]:
    """
    Search for technical generation parameters.

    Tool name: search_technical_parameters
    """
    from scripts.rag import KnowledgeBase

    query = f"{task_type} technical parameters settings configuration"

    async with KnowledgeBase() as kb:
        results = await kb.search(query=query, top_k=3, filters=None)
        context = await kb.get_context_for_llm(query=query, max_tokens=900)

    return {
        "success": True,
        "query": query,
        "results": [
            {"content": doc.content, "metadata": doc.metadata, "score": doc.score}
            for doc in results.documents
        ],
        "context": context,
        "stats": results.retrieval_stats,
    }


async def answer_question(question: str) -> Dict[str, Any]:
    """
    Answer a question using the knowledge base (RAG).

    Tool name: answer_question
    """
    from scripts.rag import KnowledgeBase

    async with KnowledgeBase() as kb:
        answer = await kb.answer_question(
            question=question,
            include_sources=True,
        )

    return {"success": True, **answer}


async def search_video_dialogue(
    query: str,
    character: Optional[str] = None,
    film: Optional[str] = None,
    video_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Search indexed video dialogue segments (P3).

    Tool name: search_video_dialogue
    """
    from scripts.rag import KnowledgeBase

    filters: Dict[str, Any] = {"doc_kind": "dialogue"}
    if film:
        filters["film"] = film.lower()
    if video_id:
        filters["video_id"] = video_id
    if character:
        filters["character"] = character.lower()

    async with KnowledgeBase() as kb:
        results = await kb.search(query=query, top_k=int(top_k), filters=filters)

    # Optional time-window post-filter (vector store only supports equality filters)
    if start_time is not None or end_time is not None:
        start_t = float(start_time) if start_time is not None else float("-inf")
        end_t = float(end_time) if end_time is not None else float("inf")
        filtered = []
        for doc in results.documents:
            seg_start = doc.metadata.get("start_time")
            seg_end = doc.metadata.get("end_time")
            try:
                seg_start_f = float(seg_start)
                seg_end_f = float(seg_end)
            except Exception:
                continue
            if seg_end_f > start_t and seg_start_f < end_t:
                filtered.append(doc)
        results.documents = filtered

    return {
        "success": True,
        "query": query,
        "filters": filters,
        "results": [
            {"content": doc.content, "metadata": doc.metadata, "score": doc.score}
            for doc in results.documents
        ],
        "stats": results.retrieval_stats,
    }


async def search_video_shots(
    query: str,
    film: Optional[str] = None,
    video_id: Optional[str] = None,
    camera_style: Optional[str] = None,
    dominant_movement: Optional[str] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Search indexed video shots by cinematic language (P3).

    Tool name: search_video_shots
    """
    from scripts.rag import KnowledgeBase

    filters: Dict[str, Any] = {"doc_kind": "video_shot"}
    if film:
        filters["film"] = film.lower()
    if video_id:
        filters["video_id"] = video_id
    if camera_style:
        filters["camera_style"] = camera_style.lower()
    if dominant_movement:
        filters["dominant_movement"] = dominant_movement.lower()

    async with KnowledgeBase() as kb:
        results = await kb.search(query=query, top_k=int(top_k), filters=filters)

    return {
        "success": True,
        "query": query,
        "filters": filters,
        "results": [
            {"content": doc.content, "metadata": doc.metadata, "score": doc.score}
            for doc in results.documents
        ],
        "stats": results.retrieval_stats,
    }


async def search_character_reference_tracks(
    character: str,
    film: Optional[str] = None,
    video_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Search indexed character presence tracks (P3).

    Tool name: search_character_reference_tracks
    """
    from scripts.rag import KnowledgeBase

    filters: Dict[str, Any] = {"doc_kind": "character_track", "character": character.lower()}
    if film:
        filters["film"] = film.lower()
    if video_id:
        filters["video_id"] = video_id

    query = f"character track reference {character}"

    async with KnowledgeBase() as kb:
        results = await kb.search(query=query, top_k=int(top_k), filters=filters)

    return {
        "success": True,
        "query": query,
        "filters": filters,
        "results": [
            {"content": doc.content, "metadata": doc.metadata, "score": doc.score}
            for doc in results.documents
        ],
        "stats": results.retrieval_stats,
    }
