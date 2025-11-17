"""
Web Search Module for Agent Framework

Provides real-time web search capabilities for agents:
- Brave Search API (primary, requires API key)
- DuckDuckGo Search (fallback, no API key needed)
- Google Custom Search (optional)
- LLM-powered result ranking and synthesis

Author: Animation AI Studio
Date: 2025-11-17
"""

import os
import sys
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.llm_client import LLMClient


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single web search result"""
    title: str
    url: str
    snippet: str
    source: str  # "brave", "duckduckgo", "google"
    relevance_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Complete search response with multiple results"""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time: float = 0.0
    source: str = ""
    success: bool = True
    error: Optional[str] = None


class BraveSearchClient:
    """
    Brave Search API Client

    Brave Search provides privacy-respecting web search.
    Requires API key from: https://brave.com/search/api/

    Pricing (as of 2024):
    - Free tier: 2,000 queries/month
    - Pro: $5/month for 20,000 queries
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Brave Search client

        Args:
            api_key: Brave Search API key (or set BRAVE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search(
        self,
        query: str,
        count: int = 10,
        safesearch: str = "moderate"
    ) -> SearchResponse:
        """
        Search using Brave Search API

        Args:
            query: Search query
            count: Number of results (max 20)
            safesearch: Safe search level ("off", "moderate", "strict")

        Returns:
            SearchResponse
        """
        if not self.api_key:
            return SearchResponse(
                query=query,
                success=False,
                error="Brave API key not provided"
            )

        import time
        start_time = time.time()

        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }

            params = {
                "q": query,
                "count": min(count, 20),
                "safesearch": safesearch
            }

            async with self.session.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return SearchResponse(
                        query=query,
                        success=False,
                        error=f"Brave API error {response.status}: {error_text}"
                    )

                data = await response.json()

                # Parse results
                results = []
                web_results = data.get("web", {}).get("results", [])

                for item in web_results:
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                        source="brave",
                        metadata={
                            "age": item.get("age"),
                            "language": item.get("language"),
                            "family_friendly": item.get("family_friendly")
                        }
                    ))

                search_time = time.time() - start_time

                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time,
                    source="brave",
                    success=True
                )

        except asyncio.TimeoutError:
            return SearchResponse(
                query=query,
                success=False,
                error="Brave Search timeout"
            )
        except Exception as e:
            logger.error(f"Brave Search error: {e}")
            return SearchResponse(
                query=query,
                success=False,
                error=str(e)
            )


class DuckDuckGoSearchClient:
    """
    DuckDuckGo Search Client (No API key needed)

    Uses DuckDuckGo's HTML search interface.
    Free, privacy-respecting, no rate limits.
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search(
        self,
        query: str,
        count: int = 10,
        region: str = "wt-wt"
    ) -> SearchResponse:
        """
        Search using DuckDuckGo

        Args:
            query: Search query
            count: Number of results
            region: Region code (e.g., "us-en", "wt-wt" for worldwide)

        Returns:
            SearchResponse
        """
        import time
        start_time = time.time()

        try:
            # Use DuckDuckGo Lite (simpler HTML parsing)
            url = "https://lite.duckduckgo.com/lite/"

            params = {
                "q": query,
                "kl": region
            }

            async with self.session.post(
                url,
                data=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    return SearchResponse(
                        query=query,
                        success=False,
                        error=f"DuckDuckGo error {response.status}"
                    )

                html = await response.text()

                # Simple HTML parsing for results
                results = self._parse_ddg_results(html, count)

                search_time = time.time() - start_time

                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time,
                    source="duckduckgo",
                    success=True
                )

        except Exception as e:
            logger.error(f"DuckDuckGo Search error: {e}")
            return SearchResponse(
                query=query,
                success=False,
                error=str(e)
            )

    def _parse_ddg_results(self, html: str, count: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML results"""
        results = []

        try:
            # Simple regex-based parsing (in production, use BeautifulSoup)
            import re

            # Find result blocks
            pattern = r'<a rel="nofollow" href="([^"]+)">([^<]+)</a>.*?<td class="result-snippet">([^<]+)</td>'
            matches = re.findall(pattern, html, re.DOTALL)

            for url, title, snippet in matches[:count]:
                # Clean up snippet
                snippet = snippet.strip()

                results.append(SearchResult(
                    title=title.strip(),
                    url=url.strip(),
                    snippet=snippet,
                    source="duckduckgo"
                ))

        except Exception as e:
            logger.warning(f"Error parsing DuckDuckGo results: {e}")

        return results


class WebSearchModule:
    """
    Web Search Module

    Provides intelligent web search capabilities for agents:
    - Tries Brave Search first (if API key available)
    - Falls back to DuckDuckGo if Brave fails
    - LLM-powered result ranking
    - Result synthesis and summarization

    Usage:
        async with WebSearchModule() as search:
            result = await search.search("Pixar animation techniques")
            summary = await search.synthesize_results(result)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        brave_api_key: Optional[str] = None,
        prefer_brave: bool = True
    ):
        """
        Initialize web search module

        Args:
            llm_client: LLM client for result ranking/synthesis
            brave_api_key: Brave Search API key
            prefer_brave: Prefer Brave over DuckDuckGo
        """
        self._llm_client = llm_client
        self._own_client = llm_client is None

        self.prefer_brave = prefer_brave
        self.brave_client = BraveSearchClient(api_key=brave_api_key)
        self.ddg_client = DuckDuckGoSearchClient()

        logger.info("WebSearchModule initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_client:
            self._llm_client = LLMClient()
            await self._llm_client.__aenter__()

        await self.brave_client.__aenter__()
        await self.ddg_client.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.brave_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.ddg_client.__aexit__(exc_type, exc_val, exc_tb)

        if self._own_client and self._llm_client:
            await self._llm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def search(
        self,
        query: str,
        count: int = 10,
        rank_results: bool = True
    ) -> SearchResponse:
        """
        Search the web with automatic fallback

        Args:
            query: Search query
            count: Number of results
            rank_results: Whether to rank results using LLM

        Returns:
            SearchResponse
        """
        logger.info(f"Web search: {query[:50]}...")

        response = None

        # Try Brave first if preferred and API key available
        if self.prefer_brave and self.brave_client.api_key:
            logger.info("Trying Brave Search...")
            response = await self.brave_client.search(query, count)

            if not response.success:
                logger.warning(f"Brave Search failed: {response.error}")
                response = None

        # Fallback to DuckDuckGo
        if response is None:
            logger.info("Using DuckDuckGo Search...")
            response = await self.ddg_client.search(query, count)

        # Rank results using LLM
        if response.success and rank_results and response.results:
            response.results = await self._rank_results(query, response.results)

        logger.info(f"Web search completed: {len(response.results)} results from {response.source}")

        return response

    async def synthesize_results(
        self,
        search_response: SearchResponse,
        max_results: int = 5
    ) -> str:
        """
        Synthesize search results into a coherent summary

        Args:
            search_response: Search response with results
            max_results: Maximum number of results to include

        Returns:
            Synthesized summary
        """
        if not search_response.results:
            return "No search results found."

        # Build synthesis prompt
        results_text = "\n\n".join([
            f"Result {i+1}:\nTitle: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
            for i, r in enumerate(search_response.results[:max_results])
        ])

        prompt = f"""Synthesize the following web search results into a coherent summary.

Query: {search_response.query}

Search Results:
{results_text}

Provide a comprehensive summary that:
1. Answers the query using information from the results
2. Cites sources when possible
3. Highlights key points
4. Is concise but informative (2-3 paragraphs)

Summary:"""

        response = await self._llm_client.chat(
            model="qwen-14b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800
        )

        return response["content"]

    async def _rank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rank search results by relevance using LLM

        Args:
            query: Original search query
            results: List of search results

        Returns:
            Ranked list of results
        """
        if len(results) <= 1:
            return results

        # Build ranking prompt
        results_text = "\n".join([
            f"{i+1}. {r.title} - {r.snippet[:100]}"
            for i, r in enumerate(results)
        ])

        prompt = f"""Rank the following search results by relevance to the query.

Query: {query}

Search Results:
{results_text}

Rank the results from most to least relevant. Respond with JSON:
{{
  "rankings": [1, 3, 2, ...]  // Indices in order of relevance
}}

Rankings:"""

        try:
            response = await self._llm_client.chat(
                model="qwen-14b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            # Parse rankings
            data = json.loads(response["content"])
            rankings = data.get("rankings", list(range(1, len(results) + 1)))

            # Reorder results
            ranked_results = []
            for rank_idx in rankings:
                if 1 <= rank_idx <= len(results):
                    result = results[rank_idx - 1]
                    result.relevance_score = 1.0 - (len(ranked_results) / len(results))
                    ranked_results.append(result)

            # Add any missing results
            for result in results:
                if result not in ranked_results:
                    result.relevance_score = 0.0
                    ranked_results.append(result)

            return ranked_results

        except Exception as e:
            logger.warning(f"Failed to rank results: {e}, using original order")
            return results


async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async with WebSearchModule() as search:
        # Example 1: Basic search
        print("\n" + "=" * 60)
        print("Example 1: Web Search")
        print("=" * 60)

        response = await search.search(
            query="Pixar animation rendering techniques",
            count=5
        )

        print(f"\nSearch completed:")
        print(f"  Source: {response.source}")
        print(f"  Results: {response.total_results}")
        print(f"  Time: {response.search_time:.2f}s")

        for i, result in enumerate(response.results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet[:150]}...")
            print(f"   Relevance: {result.relevance_score:.2f}")

        # Example 2: Synthesize results
        print("\n" + "=" * 60)
        print("Example 2: Result Synthesis")
        print("=" * 60)

        summary = await search.synthesize_results(response, max_results=3)
        print(f"\nSynthesized Summary:")
        print(summary)


if __name__ == "__main__":
    asyncio.run(main())
