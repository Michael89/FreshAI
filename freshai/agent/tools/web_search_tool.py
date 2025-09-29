"""Web search tool for agents."""
import logging
import httpx
import json
from typing import Dict, Any, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Tool for searching the web using various search APIs."""

    def __init__(
        self,
        search_engine: str = "duckduckgo",
        max_results: int = 5,
        timeout: int = 10
    ):
        """Initialize the web search tool.

        Args:
            search_engine: Search engine to use ("duckduckgo", "searx")
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
        """
        self.search_engine = search_engine
        self.max_results = max_results
        self.timeout = timeout

        # Search engine configurations
        self.engines = {
            "duckduckgo": {
                "url": "https://html.duckduckgo.com/html/",
                "parser": self._parse_duckduckgo
            },
            "searx": {
                "url": "https://searx.be/search",
                "parser": self._parse_searx
            }
        }

    async def search(self, query: str) -> str:
        """Perform a web search.

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        logger.info(f"Searching web for: {query}")
        logger.debug(f"Using search engine: {self.search_engine}")

        if self.search_engine not in self.engines:
            return f"Error: Unknown search engine '{self.search_engine}'"

        engine_config = self.engines[self.search_engine]

        try:
            # Perform search based on engine
            if self.search_engine == "duckduckgo":
                results = await self._search_duckduckgo(query)
            elif self.search_engine == "searx":
                results = await self._search_searx(query)
            else:
                results = []

            # Format results
            if not results:
                return "No results found."

            formatted = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results[:self.max_results], 1):
                formatted += f"{i}. **{result['title']}**\n"
                formatted += f"   URL: {result['url']}\n"
                if result.get('snippet'):
                    formatted += f"   {result['snippet']}\n"
                formatted += "\n"

            logger.debug(f"Found {len(results)} results")
            return formatted

        except Exception as e:
            error = f"Search failed: {str(e)}"
            logger.error(error)
            return f"Error: {error}"

    async def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search using DuckDuckGo HTML interface.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        try:
            # Use DuckDuckGo's instant answer API
            api_url = "https://api.duckduckgo.com/"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    api_url,
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    }
                )

                if response.status_code != 200:
                    logger.error(f"DuckDuckGo API returned status {response.status_code}")
                    return []

                data = response.json()
                results = []

                # Parse instant answer
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "Summary"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("AbstractText", "")
                    })

                # Parse related topics
                for topic in data.get("RelatedTopics", [])[:self.max_results - 1]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text", "").split(" - ")[0][:100],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })

                # If no instant answers, try HTML scraping as fallback
                if not results:
                    logger.debug("No instant answers, trying HTML search")
                    return await self._search_duckduckgo_html(query)

                return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _search_duckduckgo_html(self, query: str) -> List[Dict[str, str]]:
        """Fallback DuckDuckGo search using HTML interface.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True
            ) as client:
                response = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                if response.status_code != 200:
                    return []

                # Simple HTML parsing (without BeautifulSoup)
                html = response.text
                results = []

                # Look for result snippets
                import re
                # Pattern to find search results
                pattern = r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>'
                matches = re.findall(pattern, html, re.DOTALL)

                for url, title, snippet in matches[:self.max_results]:
                    results.append({
                        "title": title.strip(),
                        "url": url,
                        "snippet": snippet.strip()
                    })

                return results

        except Exception as e:
            logger.error(f"DuckDuckGo HTML search error: {e}")
            return []

    async def _search_searx(self, query: str) -> List[Dict[str, str]]:
        """Search using Searx instance.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.engines["searx"]["url"],
                    params={
                        "q": query,
                        "format": "json",
                        "categories": "general"
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Searx returned status {response.status_code}")
                    return []

                data = response.json()
                results = []

                for result in data.get("results", [])[:self.max_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", "")
                    })

                return results

        except Exception as e:
            logger.error(f"Searx search error: {e}")
            return []

    def _parse_duckduckgo(self, html: str) -> List[Dict[str, str]]:
        """Parse DuckDuckGo HTML results (legacy)."""
        # This is handled in the search methods above
        return []

    def _parse_searx(self, data: Dict) -> List[Dict[str, str]]:
        """Parse Searx JSON results (legacy)."""
        # This is handled in the search methods above
        return []

    async def __call__(self, query: str) -> str:
        """Allow the tool to be called as a function."""
        return await self.search(query)


class SimpleWebSearchTool:
    """Simplified synchronous web search tool."""

    def __init__(self, max_results: int = 5):
        """Initialize the simple web search tool.

        Args:
            max_results: Maximum number of results to return
        """
        self.max_results = max_results

    def search(self, query: str) -> str:
        """Perform a simple web search (synchronous).

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        logger.info(f"Performing simple web search for: {query}")

        try:
            # Use DuckDuckGo instant answer API (synchronous)
            import requests

            response = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1"
                },
                timeout=10
            )

            if response.status_code != 200:
                return "Search failed: Unable to connect to search service"

            data = response.json()
            results = []

            # Get abstract if available
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
                if data.get("AbstractURL"):
                    results.append(f"Source: {data['AbstractURL']}")

            # Get related topics
            for i, topic in enumerate(data.get("RelatedTopics", [])[:self.max_results], 1):
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"\n{i}. {topic['Text']}")
                    if topic.get("FirstURL"):
                        results.append(f"   Link: {topic['FirstURL']}")

            if results:
                return "\n".join(results)
            else:
                return f"No specific results found for '{query}'. Try a different search term."

        except Exception as e:
            error = f"Search error: {str(e)}"
            logger.error(error)
            return error

    def __call__(self, query: str) -> str:
        """Allow the tool to be called as a function."""
        return self.search(query)


def create_web_search_tool(
    search_engine: str = "duckduckgo",
    max_results: int = 5,
    async_mode: bool = False
) -> Dict[str, Any]:
    """Create a web search tool specification for agents.

    Args:
        search_engine: Search engine to use
        max_results: Maximum number of results
        async_mode: Whether to use async version

    Returns:
        Tool specification with function and metadata
    """
    if async_mode:
        tool = WebSearchTool(search_engine, max_results)
    else:
        tool = SimpleWebSearchTool(max_results)

    return {
        "name": "web_search",
        "description": "Search the web for information. Use this to find current information, facts, news, etc.",
        "function": tool,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }