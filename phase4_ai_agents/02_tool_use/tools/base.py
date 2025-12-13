"""
Base Tool - Abstract base class for all tools.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# enable imports from parent package
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from schemas.tool import ToolDefinition, ToolResult


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Every tool must:
    1. Have a name (unique identifier)
    2. Provide a definition (describes parameters)
    3. Implement execute() (does the actual work)

    To create a new tool:
        class MyTool(BaseTool):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(...)

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult.ok(result)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Full definition describing the tool and its parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def __call__(self, **kwargs: Any) -> ToolResult:
        """Allow tool to be called like a function: tool(arg=value)."""
        return self.execute(**kwargs)


class ToolRegistry:
    """
    Registry that holds all available tools.

    Like a service registry - the agent uses this to look up tools by name.

    Usage:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WebSearchTool())

        tool = registry.get("read_file")
        result = registry.execute("read_file", path="/tmp/test.txt")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name, or None if not found."""
        return self._tools.get(name)

    def get_all_definitions(self) -> list[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.definition for tool in self._tools.values()]

    def get_openai_tools(self) -> list[dict]:
        """Get all tool definitions in OpenAI format."""
        return [tool.definition.to_openai_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self.get(tool_name)

        if tool is None:
            return ToolResult.fail(f"Tool not found: {tool_name}")

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult.fail(f"Tool execution error: {str(e)}")

    def list_tools(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
