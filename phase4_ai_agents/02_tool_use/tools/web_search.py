"""
Web Search Tool - Search the web for information.

Supports mock mode (for learning) and real Tavily API integration.
"""

import os

from schemas.tool import ToolDefinition, ToolParameter, ToolResult
from tools.base import BaseTool


class WebSearchTool(BaseTool):
    """
    Tool to search the web for information.

    By default, uses mock results for learning.
    Set use_mock=False and provide api_key to use real search.

    Usage by agent:
        ACTION: web_search(query="Python asyncio tutorial", max_results=3)
        OBSERVATION: [{"title": "...", "url": "...", "snippet": "..."}, ...]
    """

    def __init__(self, api_key: str | None = None, use_mock: bool = True):
        """
        Initialize web search tool.

        Args:
            api_key: API key for real search (Tavily)
            use_mock: If True, use mock results (default for learning)
        """
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._use_mock = use_mock

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Search the web for information. Returns a list of search results with titles, URLs, and snippets.",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type="string",
                    description="The search query",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    param_type="integer",
                    description="Maximum number of results to return (default: 5)",
                    required=False,
                ),
            ],
        )

    def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        """Search the web for the given query."""
        if self._use_mock:
            return self._mock_search(query, max_results)
        return self._real_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> ToolResult:
        """Return mock search results for learning/testing."""
        query_lower = query.lower()

        if "python" in query_lower:
            results = [
                {"title": "Python Official Documentation", "url": "https://docs.python.org/3/", "snippet": "Welcome to Python 3 documentation."},
                {"title": "Real Python Tutorials", "url": "https://realpython.com/", "snippet": "Learn Python programming with step-by-step tutorials."},
                {"title": "Python Tutorial - W3Schools", "url": "https://www.w3schools.com/python/", "snippet": "Python is a popular programming language."},
            ]
        elif "weather" in query_lower:
            results = [
                {"title": "Weather.com", "url": "https://weather.com/", "snippet": "Get the latest weather news and forecasts."},
                {"title": "National Weather Service", "url": "https://www.weather.gov/", "snippet": "Official weather forecasts and warnings."},
            ]
        elif "ai" in query_lower or "artificial intelligence" in query_lower:
            results = [
                {"title": "What is AI? - IBM", "url": "https://www.ibm.com/topics/artificial-intelligence", "snippet": "AI is the simulation of human intelligence by machines."},
                {"title": "MIT Technology Review - AI", "url": "https://www.technologyreview.com/topic/artificial-intelligence/", "snippet": "The latest AI news and breakthroughs."},
                {"title": "OpenAI", "url": "https://openai.com/", "snippet": "OpenAI is an AI research and deployment company."},
            ]
        else:
            results = [
                {"title": f"Search Results for: {query}", "url": f"https://example.com/search?q={query.replace(' ', '+')}", "snippet": f"Mock result for '{query}'."},
                {"title": f"Wikipedia - {query.title()}", "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}", "snippet": f"Wikipedia article about {query}."},
            ]

        return ToolResult.ok(results[:max_results])

    def _real_search(self, query: str, max_results: int) -> ToolResult:
        """Perform real web search using Tavily API."""
        if not self._api_key:
            return ToolResult.fail("No API key. Set TAVILY_API_KEY or use use_mock=True.")

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self._api_key)
            response = client.search(query=query, max_results=max_results, search_depth="basic")

            results = [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in response.get("results", [])
            ]
            return ToolResult.ok(results)
        except ImportError:
            return ToolResult.fail("tavily-python not installed. Run: pip install tavily-python")
        except Exception as e:
            return ToolResult.fail(f"Search error: {str(e)}")
