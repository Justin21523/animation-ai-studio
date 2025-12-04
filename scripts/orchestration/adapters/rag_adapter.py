"""
RAG System Adapter

Adapter for integrating the RAG System with the orchestration layer.
Wraps the existing RAG module to provide standardized Task/TaskResult interface.

Author: Animation AI Studio
Date: 2025-12-02
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.orchestration.module_registry import (
    ModuleAdapter,
    ModuleType,
    ModuleStatus,
    ModuleCapabilities,
    HealthCheckResult,
    Task,
    TaskResult
)
from scripts.rag import KnowledgeBase, KnowledgeBaseConfig

logger = logging.getLogger(__name__)


class RAGAdapter(ModuleAdapter):
    """
    Adapter for RAG System (3,335 LOC)

    Wraps the existing RAG System to provide standardized interface for:
    - Knowledge base search
    - Document retrieval
    - Context generation for LLMs
    - Question answering
    - Character/style/technical knowledge queries

    Task Types Supported:
    - "search": Search knowledge base
    - "retrieve": Retrieve documents
    - "context": Generate context for LLM
    - "answer": Answer question using RAG
    - "character": Character knowledge query
    - "style": Style guide query
    - "technical": Technical parameters query

    Task Parameters:
    - query (str): Search query
    - top_k (int, optional): Number of results (default: 5)
    - filters (dict, optional): Metadata filters
    - character (str, optional): Character name for character queries
    - aspect (str, optional): Specific aspect (appearance, personality, etc.)
    - style (str, optional): Style name for style queries
    - task_type (str, optional): Task type for technical queries
    - max_tokens (int, optional): Max tokens for context generation
    - include_sources (bool, optional): Include source documents in answer

    Example:
        adapter = RAGAdapter(config=KnowledgeBaseConfig(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            vector_db="faiss"
        ))

        await adapter.initialize()

        # Search knowledge base
        task = Task(
            task_id="1",
            task_type="search",
            parameters={
                "query": "Luca's appearance",
                "top_k": 5,
                "filters": {"character": "luca"}
            }
        )

        result = await adapter.execute(task)
        print(result.output["documents"])

        # Answer question
        task = Task(
            task_id="2",
            task_type="answer",
            parameters={
                "query": "Who is Luca's best friend?",
                "include_sources": True
            }
        )

        result = await adapter.execute(task)
        print(result.output["answer"])
    """

    def __init__(
        self,
        config: Optional[KnowledgeBaseConfig] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        Initialize RAG Adapter

        Args:
            config: Knowledge base configuration
            knowledge_base: Optional existing knowledge base instance
        """
        super().__init__(
            module_name="rag",
            module_type=ModuleType.RAG
        )

        self.config = config or KnowledgeBaseConfig()
        self.knowledge_base = knowledge_base
        self._own_kb = knowledge_base is None

        # Statistics
        self.total_searches = 0
        self.total_retrievals = 0
        self.total_answers = 0
        self.total_time = 0.0

        logger.info("RAGAdapter created")

    async def initialize(self) -> bool:
        """
        Initialize RAG System

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing RAG System...")

            if self._own_kb:
                # Create and initialize knowledge base
                self.knowledge_base = KnowledgeBase(config=self.config)
                await self.knowledge_base.__aenter__()

            self._initialized = True
            logger.info("RAG System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {e}", exc_info=True)
            return False

    async def execute(self, task: Task) -> TaskResult:
        """
        Execute task using RAG System

        Args:
            task: Task to execute

        Returns:
            TaskResult with RAG results
        """
        if not self._initialized:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error="RAG System not initialized"
            )

        start_time = time.time()

        try:
            # Route to appropriate handler based on task type
            if task.task_type == "search":
                result = await self._handle_search(task)
            elif task.task_type == "retrieve":
                result = await self._handle_retrieve(task)
            elif task.task_type == "context":
                result = await self._handle_context(task)
            elif task.task_type == "answer":
                result = await self._handle_answer(task)
            elif task.task_type == "character":
                result = await self._handle_character(task)
            elif task.task_type == "style":
                result = await self._handle_style(task)
            elif task.task_type == "technical":
                result = await self._handle_technical(task)
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")

            execution_time = time.time() - start_time
            self.total_time += execution_time
            result.execution_time = execution_time

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.total_time += execution_time

            logger.error(f"RAG execution failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def _handle_search(self, task: Task) -> TaskResult:
        """Handle search task"""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        top_k = task.parameters.get("top_k", 5)
        filters = task.parameters.get("filters")

        self.total_searches += 1

        # Search knowledge base
        results = await self.knowledge_base.search(
            query=query,
            top_k=top_k,
            filters=filters
        )

        documents = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score
            }
            for doc in results.documents
        ]

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "documents": documents,
                "total_results": len(documents),
                "stats": results.retrieval_stats
            },
            metadata={
                "query": query,
                "top_k": top_k,
                "filters": filters
            }
        )

    async def _handle_retrieve(self, task: Task) -> TaskResult:
        """Handle retrieve task"""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        top_k = task.parameters.get("top_k", 5)

        self.total_retrievals += 1

        # Retrieve documents
        results = await self.knowledge_base.search(
            query=query,
            top_k=top_k
        )

        documents = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score
            }
            for doc in results.documents
        ]

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "documents": documents,
                "count": len(documents)
            }
        )

    async def _handle_context(self, task: Task) -> TaskResult:
        """Handle context generation task"""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        max_tokens = task.parameters.get("max_tokens", 1500)

        # Generate context for LLM
        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=max_tokens
        )

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "context": context,
                "token_count": len(context.split())
            }
        )

    async def _handle_answer(self, task: Task) -> TaskResult:
        """Handle question answering task"""
        query = task.parameters.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        include_sources = task.parameters.get("include_sources", True)

        self.total_answers += 1

        # Answer question
        result = await self.knowledge_base.answer_question(
            question=query,
            include_sources=include_sources
        )

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "answer": result["answer"],
                "confidence": result.get("confidence", 0.7),
                "sources": result.get("sources", []) if include_sources else []
            }
        )

    async def _handle_character(self, task: Task) -> TaskResult:
        """Handle character knowledge query"""
        character = task.parameters.get("character")
        if not character:
            raise ValueError("Missing required parameter: character")

        aspect = task.parameters.get("aspect")

        # Build query
        if aspect:
            query = f"{character} {aspect}"
        else:
            query = f"{character} character profile description"

        # Search with character filter
        results = await self.knowledge_base.search(
            query=query,
            top_k=5,
            filters={"character": character.lower()}
        )

        # Get formatted context
        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=1000
        )

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "character": character,
                "aspect": aspect,
                "context": context,
                "documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score
                    }
                    for doc in results.documents
                ]
            }
        )

    async def _handle_style(self, task: Task) -> TaskResult:
        """Handle style guide query"""
        style = task.parameters.get("style")
        if not style:
            raise ValueError("Missing required parameter: style")

        query = f"{style} style guide visual characteristics"

        # Search with style filter
        results = await self.knowledge_base.search(
            query=query,
            top_k=3,
            filters={"style": style.lower()}
        )

        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=800
        )

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "style": style,
                "context": context,
                "documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score
                    }
                    for doc in results.documents
                ]
            }
        )

    async def _handle_technical(self, task: Task) -> TaskResult:
        """Handle technical parameters query"""
        task_type = task.parameters.get("task_type")
        if not task_type:
            raise ValueError("Missing required parameter: task_type")

        query = f"{task_type} technical parameters settings configuration"

        results = await self.knowledge_base.search(
            query=query,
            top_k=3
        )

        context = await self.knowledge_base.get_context_for_llm(
            query=query,
            max_tokens=800
        )

        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "task_type": task_type,
                "context": context,
                "documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score
                    }
                    for doc in results.documents
                ]
            }
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Check RAG System health

        Returns:
            HealthCheckResult with status and details
        """
        try:
            if not self._initialized:
                return HealthCheckResult(
                    module_name=self.module_name,
                    status=ModuleStatus.UNHEALTHY,
                    message="RAG System not initialized"
                )

            # Check knowledge base is available
            if not self.knowledge_base:
                return HealthCheckResult(
                    module_name=self.module_name,
                    status=ModuleStatus.UNHEALTHY,
                    message="Knowledge base not available"
                )

            # Basic health check
            status = ModuleStatus.HEALTHY
            message = "RAG System fully functional"

            return HealthCheckResult(
                module_name=self.module_name,
                status=status,
                message=message,
                details={
                    "total_searches": self.total_searches,
                    "total_retrievals": self.total_retrievals,
                    "total_answers": self.total_answers,
                    "avg_time": (
                        self.total_time / (self.total_searches + self.total_retrievals + self.total_answers)
                        if (self.total_searches + self.total_retrievals + self.total_answers) > 0
                        else 0.0
                    )
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return HealthCheckResult(
                module_name=self.module_name,
                status=ModuleStatus.UNKNOWN,
                message=f"Health check error: {str(e)}"
            )

    def get_capabilities(self) -> ModuleCapabilities:
        """
        Get RAG System capabilities

        Returns:
            ModuleCapabilities describing RAG features
        """
        return ModuleCapabilities(
            module_name=self.module_name,
            module_type=self.module_type,
            supported_operations=[
                "search",       # Knowledge base search
                "retrieve",     # Document retrieval
                "context",      # Context generation
                "answer",       # Question answering
                "character",    # Character queries
                "style",        # Style guide queries
                "technical"     # Technical parameters
            ],
            requires_gpu=False,  # CPU-only embeddings and vector search
            max_concurrent_tasks=10,  # RAG is lightweight, can handle many concurrent
            estimated_memory_mb=1024,  # ~1GB for embeddings + FAISS index
            description=(
                "RAG System for knowledge retrieval and question answering. "
                "Supports character knowledge, style guides, and technical parameters. "
                "Uses sentence-transformers for embeddings and FAISS for vector search. "
                "CPU-only operation."
            )
        )

    async def cleanup(self):
        """Cleanup RAG System resources"""
        try:
            if self._own_kb and self.knowledge_base:
                await self.knowledge_base.__aexit__(None, None, None)
                logger.info("RAG System cleaned up")
        except Exception as e:
            logger.error(f"Error during RAG cleanup: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get adapter statistics

        Returns:
            Statistics dictionary
        """
        total_ops = self.total_searches + self.total_retrievals + self.total_answers
        avg_time = self.total_time / total_ops if total_ops > 0 else 0.0

        return {
            "total_searches": self.total_searches,
            "total_retrievals": self.total_retrievals,
            "total_answers": self.total_answers,
            "total_operations": total_ops,
            "total_time": self.total_time,
            "avg_time": avg_time
        }

    def __repr__(self) -> str:
        total_ops = self.total_searches + self.total_retrievals + self.total_answers
        return (
            f"RAGAdapter(initialized={self._initialized}, "
            f"operations={total_ops})"
        )
