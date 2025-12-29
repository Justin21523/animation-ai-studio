import sys
from pathlib import Path

import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.asyncio
async def test_answer_question_extracts_openai_content(monkeypatch) -> None:
    from scripts.rag.knowledge_base import KnowledgeBase
    from scripts.rag.retrieval.retrieval_engine import RetrievalResult
    from scripts.rag.vectordb.vector_store import SearchResult

    class DummyLLM:
        async def chat(self, *args, **kwargs):
            return {"choices": [{"message": {"content": "ANSWER"}}]}

    kb = KnowledgeBase()
    kb._initialized = True
    kb.llm_client = DummyLLM()

    async def fake_get_context_for_llm(*args, **kwargs) -> str:
        return "[Document 1]\nSome context\n"

    async def fake_search(query: str, top_k=None, filters=None):
        docs = [
            SearchResult(
                doc_id="doc_1",
                content="Some context",
                metadata={"source_file": "dummy"},
                score=0.9,
            )
        ]
        return RetrievalResult(documents=docs, query=query)

    monkeypatch.setattr(kb, "get_context_for_llm", fake_get_context_for_llm)
    monkeypatch.setattr(kb, "search", fake_search)

    result = await kb.answer_question("What is this?", include_sources=True)
    assert result["answer"] == "ANSWER"
    assert len(result["sources"]) == 1

