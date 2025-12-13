"""
HTTP Tool - Make HTTP API calls.

Allows agents to fetch data from REST APIs.
"""

import json

from schemas.tool import ToolDefinition, ToolParameter, ToolResult
from tools.base import BaseTool


class HttpGetTool(BaseTool):
    """
    Tool to make HTTP GET requests.

    Usage by agent:
        ACTION: http_get(url="https://api.github.com/users/octocat")
        OBSERVATION: {"login": "octocat", "id": 583231, ...}
    """

    def __init__(self, timeout_seconds: int = 30, allowed_domains: list[str] | None = None, use_mock: bool = False):
        """
        Initialize HTTP GET tool.

        Args:
            timeout_seconds: Request timeout
            allowed_domains: If set, only allow requests to these domains
            use_mock: If True, return mock responses for learning
        """
        self._timeout_seconds = timeout_seconds
        self._allowed_domains = allowed_domains
        self._use_mock = use_mock

    @property
    def name(self) -> str:
        return "http_get"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Make an HTTP GET request to a URL. Returns the response body as text or JSON.",
            parameters=[
                ToolParameter(
                    name="url",
                    param_type="string",
                    description="The URL to request (must start with http:// or https://)",
                    required=True,
                ),
            ],
        )

    def execute(self, url: str, **kwargs) -> ToolResult:
        """Make HTTP GET request."""
        if not url.startswith(("http://", "https://")):
            return ToolResult.fail("Invalid URL: must start with http:// or https://")

        if self._allowed_domains is not None:
            domain = url.split("//")[1].split("/")[0].split(":")[0]
            if domain not in self._allowed_domains:
                return ToolResult.fail(f"Domain not allowed: {domain}")

        if self._use_mock:
            return self._mock_request(url)
        return self._real_request(url)

    def _mock_request(self, url: str) -> ToolResult:
        """Return mock response for learning/testing."""
        url_lower = url.lower()

        if "github.com" in url_lower and "/users/" in url_lower:
            return ToolResult.ok({"login": "mock_user", "id": 12345, "type": "User", "name": "Mock User", "public_repos": 42})
        elif "jsonplaceholder" in url_lower and "/posts/" in url_lower:
            return ToolResult.ok({"id": 1, "title": "Mock Post Title", "body": "Mock post body.", "userId": 1})
        elif "weather" in url_lower:
            return ToolResult.ok({"temperature": 22, "conditions": "sunny", "humidity": 65})
        else:
            return ToolResult.ok({"message": "Mock HTTP response", "url": url})

    def _real_request(self, url: str) -> ToolResult:
        """Make real HTTP GET request."""
        try:
            try:
                import httpx
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.get(url)
                    if response.status_code >= 400:
                        return ToolResult.fail(f"HTTP {response.status_code}: {response.text[:200]}")
                    try:
                        return ToolResult.ok(response.json())
                    except json.JSONDecodeError:
                        return ToolResult.ok(response.text)
            except ImportError:
                import urllib.request
                import urllib.error
                request = urllib.request.Request(url, headers={"User-Agent": "Python-Agent-Tool/1.0"})
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                    content = response.read().decode("utf-8")
                    try:
                        return ToolResult.ok(json.loads(content))
                    except json.JSONDecodeError:
                        return ToolResult.ok(content)
        except Exception as e:
            return ToolResult.fail(f"Request error: {str(e)}")
