"""
RAG Usage Module for Agent Framework

Integrates with RAG System (Module 5) for knowledge retrieval.

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag import KnowledgeBase, KnowledgeBaseConfig
from scripts.agent.core.types import Thought


logger = logging.getLogger(__name__)


@dataclass
class RAGRetrievalResult:
    """Result from RAG retrieval"""
    query: str
    documents: List[Dict[str, Any]]
    context: str
    confidence: float
    metadata: Dict[str, Any]


class RAGUsageModule:
    """
    RAG Usage Module

    Provides interface to RAG System for:
    - Character knowledge retrieval
    - Style guide retrieval
    - Technical parameter retrieval
    - Past generation history

    Integrates with Module 5: RAG System
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[KnowledgeBaseConfig] = None
    ):
        """
        Initialize RAG usage module

        Args:
            knowledge_base: KnowledgeBase instance (will create if not provided)
            config: Knowledge base configuration
        """
        self.knowledge_base = knowledge_base
        self._own_kb = knowledge_base is None
        self.config = config or KnowledgeBaseConfig()

        logger.info("RAGUsageModule initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_kb:
            self.knowledge_base = KnowledgeBase(config=self.config)
            await self.knowledge_base.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._own_kb and self.knowledge_base:
            await self.knowledge_base.__aexit__(exc_type, exc_val, exc_tb)

    async def retrieve_character_knowledge(
        self,
        character_name: str,
        aspect: Optional[str] = None
    ) -> RAGRetrievalResult:
        """
        Retrieve character knowledge

        Args:
            character_name: Name of character
            aspect: Specific aspect (appearance, personality, voice, etc.)

        Returns:
            RAGRetrievalResult with character information
        """
        # Build query
        if aspect:
            query = f"{character_name} {aspect}"
        else:
            query = f"{character_name} character profile description"

        # Retrieve from knowledge base
        results = await self.knowledge_base.search(
            query=query,
            top_k=5,
            filters={"character": character_name.lower()}
        )

        # Get formatted context
        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=1000
        )

        logger.info(f"Retrieved character knowledge for {character_name}: {len(results.documents)} docs")

        return RAGRetrievalResult(
            query=query,
            documents=[
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in results.documents
            ],
            context=context,
            confidence=results.retrieval_stats.get("avg_score", 0.7),
            metadata={
                "character": character_name,
                "aspect": aspect,
                "total_docs": len(results.documents)
            }
        )

    async def retrieve_style_guide(
        self,
        style_name: str
    ) -> RAGRetrievalResult:
        """
        Retrieve style guide

        Args:
            style_name: Name of style (e.g., "pixar_3d", "italian_summer")

        Returns:
            RAGRetrievalResult with style information
        """
        query = f"{style_name} style guide visual characteristics"

        results = await self.knowledge_base.search(
            query=query,
            top_k=3,
            filters={"style": style_name.lower()}
        )

        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=800
        )

        logger.info(f"Retrieved style guide for {style_name}")

        return RAGRetrievalResult(
            query=query,
            documents=[
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in results.documents
            ],
            context=context,
            confidence=results.retrieval_stats.get("avg_score", 0.7),
            metadata={
                "style": style_name,
                "total_docs": len(results.documents)
            }
        )

    async def retrieve_technical_parameters(
        self,
        task_type: str
    ) -> RAGRetrievalResult:
        """
        Retrieve technical parameters

        Args:
            task_type: Type of task (image_generation, voice_synthesis, etc.)

        Returns:
            RAGRetrievalResult with technical parameters
        """
        query = f"{task_type} technical parameters settings configuration"

        results = await self.knowledge_base.search(
            query=query,
            top_k=3
        )

        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=800
        )

        logger.info(f"Retrieved technical parameters for {task_type}")

        return RAGRetrievalResult(
            query=query,
            documents=[
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in results.documents
            ],
            context=context,
            confidence=results.retrieval_stats.get("avg_score", 0.7),
            metadata={
                "task_type": task_type,
                "total_docs": len(results.documents)
            }
        )

    async def retrieve_context_for_task(
        self,
        task_description: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGRetrievalResult:
        """
        Retrieve general context for task

        Args:
            task_description: Description of the task
            filters: Optional metadata filters

        Returns:
            RAGRetrievalResult with relevant context
        """
        results = await self.knowledge_base.search(
            query=task_description,
            top_k=5,
            filters=filters
        )

        context = await self.knowledge_base.get_context_for_llm(
            query=task_description,
            max_tokens=1500
        )

        logger.info(f"Retrieved context for task: {len(results.documents)} docs")

        return RAGRetrievalResult(
            query=task_description,
            documents=[
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in results.documents
            ],
            context=context,
            confidence=results.retrieval_stats.get("avg_score", 0.7),
            metadata={
                "filters": filters,
                "total_docs": len(results.documents)
            }
        )

    async def answer_question(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer question using RAG

        Args:
            question: Question to answer
            filters: Optional metadata filters

        Returns:
            Answer string
        """
        result = await self.knowledge_base.answer_question(
            question=question,
            include_sources=True
        )

        logger.info(f"Answered question with confidence: {result['confidence']}")
        return result["answer"]

    def create_retrieval_thought(
        self,
        retrieval_result: RAGRetrievalResult
    ) -> Thought:
        """
        Create a Thought from retrieval result

        Args:
            retrieval_result: RAG retrieval result

        Returns:
            Thought representing the retrieval
        """
        content = f"Retrieved {len(retrieval_result.documents)} relevant documents for '{retrieval_result.query}'"

        return Thought(
            content=content,
            thought_type="observation",
            confidence=retrieval_result.confidence,
            metadata={
                "query": retrieval_result.query,
                "num_docs": len(retrieval_result.documents),
                **retrieval_result.metadata
            }
        )


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async with RAGUsageModule() as rag:
        # Example 1: Retrieve character knowledge
        char_result = await rag.retrieve_character_knowledge(
            character_name="luca",
            aspect="appearance"
        )

        print(f"\nCharacter Knowledge:")
        print(f"  Query: {char_result.query}")
        print(f"  Documents: {len(char_result.documents)}")
        print(f"  Confidence: {char_result.confidence:.2f}")
        print(f"  Context preview: {char_result.context[:200]}...")

        # Example 2: Retrieve style guide
        style_result = await rag.retrieve_style_guide("pixar_3d")

        print(f"\nStyle Guide:")
        print(f"  Query: {style_result.query}")
        print(f"  Documents: {len(style_result.documents)}")

        # Example 3: Answer question
        answer = await rag.answer_question("Who is Luca's best friend?")

        print(f"\nQuestion Answering:")
        print(f"  Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
