#!/usr/bin/env python3
"""
Test RAG retrieval functionality with film knowledge

Usage:
    python scripts/rag/test_rag_retrieval.py
    python scripts/rag/test_rag_retrieval.py --interactive
"""

import asyncio
import argparse
from typing import List, Dict, Any

from scripts.rag.knowledge_base import KnowledgeBase
from scripts.core.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGTester:
    """Test RAG retrieval with various queries"""

    def __init__(self):
        self.test_queries = [
            # Character queries
            {
                "query": "Tell me about Luca's personality and character traits",
                "expected_topics": ["curious", "timid", "brave", "kind-hearted"],
                "category": "Character"
            },
            {
                "query": "What does Alberto look like in human form?",
                "expected_topics": ["green vest", "curly hair", "italian boy"],
                "category": "Character"
            },
            {
                "query": "Describe Giulia's appearance and personality",
                "expected_topics": ["red hair", "freckles", "energetic"],
                "category": "Character"
            },

            # Scene and setting queries
            {
                "query": "Describe the visual style and color palette of Portorosso",
                "expected_topics": ["italian", "coastal", "colorful", "summer"],
                "category": "Scene/Setting"
            },
            {
                "query": "What are the key locations in the film?",
                "expected_topics": ["Portorosso", "tower", "underwater", "plaza"],
                "category": "Scene/Setting"
            },

            # Style and technical queries
            {
                "query": "What animation style and lighting should I use for Luca characters?",
                "expected_topics": ["pixar", "3d animation", "smooth shading", "mediterranean"],
                "category": "Style/Technical"
            },
            {
                "query": "What color palette should I use for Luca-style images?",
                "expected_topics": ["warm", "blue", "gold", "summer"],
                "category": "Style/Technical"
            },

            # Relationship queries
            {
                "query": "What is the relationship between Luca and Alberto?",
                "expected_topics": ["best friend", "brother", "mentor", "silenzio bruno"],
                "category": "Relationships"
            },

            # Prompts and generation
            {
                "query": "Give me a good prompt for generating an image of Luca",
                "expected_topics": ["striped shirt", "wavy hair", "pixar", "3d animated"],
                "category": "Generation"
            },
        ]

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all predefined test queries"""
        logger.info("Starting RAG retrieval tests...")

        results = {
            "total_queries": len(self.test_queries),
            "successful": 0,
            "failed": 0,
            "tests": []
        }

        async with KnowledgeBase() as kb:
            # Get KB stats first
            stats = kb.get_stats()
            logger.info(f"Knowledge Base Stats: {stats}")

            if stats.get("total_documents", 0) == 0:
                logger.error("Knowledge base is empty! Run ingestion first.")
                return results

            for i, test in enumerate(self.test_queries, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Test {i}/{len(self.test_queries)}: {test['category']}")
                logger.info(f"Query: {test['query']}")
                logger.info(f"{'='*60}")

                try:
                    # Retrieve relevant documents
                    retrieval_result = await kb.search(
                        query=test['query'],
                        top_k=5,
                        filters={}
                    )

                    # Check if we got results
                    num_results = len(retrieval_result.documents)
                    logger.info(f"Retrieved {num_results} documents")

                    if num_results == 0:
                        logger.warning("No documents retrieved!")
                        results["failed"] += 1
                        results["tests"].append({
                            "query": test['query'],
                            "category": test['category'],
                            "status": "FAILED",
                            "reason": "No results"
                        })
                        continue

                    # Display top results
                    for j, doc in enumerate(retrieval_result.documents[:3], 1):
                        logger.info(f"\n  Result {j} (score: {doc.score:.4f}):")
                        logger.info(f"  Source: {doc.metadata.get('source_file', 'Unknown')}")
                        logger.info(f"  Type: {doc.metadata.get('type', 'Unknown')}")
                        logger.info(f"  Preview: {doc.content[:200]}...")

                    # Check if expected topics are covered
                    all_content = " ".join([doc.content.lower() for doc in retrieval_result.documents[:3]])
                    topics_found = [topic for topic in test['expected_topics']
                                    if topic.lower() in all_content]

                    coverage = len(topics_found) / len(test['expected_topics'])

                    if coverage >= 0.5:  # At least 50% topics found
                        logger.info(f"✓ Test PASSED (coverage: {coverage:.1%})")
                        logger.info(f"  Topics found: {topics_found}")
                        results["successful"] += 1
                        results["tests"].append({
                            "query": test['query'],
                            "category": test['category'],
                            "status": "PASSED",
                            "coverage": coverage,
                            "topics_found": topics_found
                        })
                    else:
                        logger.warning(f"✗ Test FAILED (coverage: {coverage:.1%})")
                        logger.warning(f"  Expected: {test['expected_topics']}")
                        logger.warning(f"  Found: {topics_found}")
                        results["failed"] += 1
                        results["tests"].append({
                            "query": test['query'],
                            "category": test['category'],
                            "status": "FAILED",
                            "reason": "Low coverage",
                            "coverage": coverage
                        })

                except Exception as e:
                    logger.error(f"Error during test: {e}")
                    results["failed"] += 1
                    results["tests"].append({
                        "query": test['query'],
                        "category": test['category'],
                        "status": "ERROR",
                        "error": str(e)
                    })

        return results

    async def interactive_mode(self):
        """Interactive query mode"""
        logger.info("Starting interactive RAG query mode...")
        logger.info("Type 'quit' or 'exit' to stop")

        async with KnowledgeBase() as kb:
            # Check KB status
            stats = kb.get_stats()
            print(f"\n📚 Knowledge Base Status:")
            print(f"  Total documents: {stats.get('total_documents', 0)}")
            print(f"  Films: luca")
            print()

            if stats.get("total_documents", 0) == 0:
                print("⚠️  Knowledge base is empty! Run ingestion first:")
                print("  python scripts/rag/ingest_film_knowledge.py --film luca")
                return

            while True:
                try:
                    # Get user query
                    query = input("\n🔍 Enter your query: ").strip()

                    if query.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break

                    if not query:
                        continue

                    # Search
                    print(f"\nSearching for: '{query}'...")
                    result = await kb.search(query, top_k=3)

                    if not result.documents:
                        print("No results found.")
                        continue

                    # Display results
                    print(f"\n{'='*60}")
                    print(f"Found {len(result.documents)} results:")
                    print(f"{'='*60}")

                    for i, doc in enumerate(result.documents, 1):
                        print(f"\n📄 Result {i} (relevance: {doc.score:.4f})")
                        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
                        print(f"Type: {doc.metadata.get('type', 'Unknown')}")
                        print(f"\nContent preview:")
                        print(f"{doc.content[:400]}...")
                        print(f"\n{'-'*60}")

                    # Option to get Q&A answer
                    use_qa = input("\n💬 Generate answer with LLM? (y/n): ").strip().lower()
                    if use_qa == 'y':
                        try:
                            answer = await kb.answer_question(query, include_sources=True)
                            print(f"\n🤖 LLM Answer:")
                            print(f"{answer['answer']}")
                            if answer.get('sources'):
                                print(f"\nSources:")
                                for src in answer['sources']:
                                    print(f"  - {src}")
                        except Exception as e:
                            print(f"Error generating answer: {e}")

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test RAG retrieval functionality")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive query mode")
    args = parser.parse_args()

    tester = RAGTester()

    if args.interactive:
        await tester.interactive_mode()
    else:
        # Run automated tests
        results = await tester.run_all_tests()

        # Print summary
        print("\n" + "="*60)
        print("📊 RAG Retrieval Test Summary")
        print("="*60)
        print(f"Total queries: {results['total_queries']}")
        print(f"Successful: {results['successful']} ✓")
        print(f"Failed: {results['failed']} ✗")
        print(f"Success rate: {results['successful']/results['total_queries']*100:.1f}%")
        print("="*60)

        # Category breakdown
        categories = {}
        for test in results['tests']:
            cat = test['category']
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if test['status'] == "PASSED":
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        print("\nBy Category:")
        for cat, counts in categories.items():
            total = counts['passed'] + counts['failed']
            rate = counts['passed'] / total * 100 if total > 0 else 0
            print(f"  {cat}: {counts['passed']}/{total} ({rate:.0f}%)")

        print("\nTip: Run with --interactive for manual testing")


if __name__ == "__main__":
    asyncio.run(main())
