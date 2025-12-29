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

