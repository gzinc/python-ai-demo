"""
ToolRegistry - Manages collection of available tools

Similar to a Spring @ComponentScan or Service Registry in Java.
Keeps track of all registered tools and provides them in API-specific formats.
"""

from typing import Any, Optional

from models import Tool


class ToolRegistry:
    """
    registry for managing multiple tools

    Usage:
        registry = ToolRegistry()
        registry.register(weather_tool)
        registry.register(calculator_tool)

        # get all tools for API
        openai_tools = registry.to_openai_format()
        anthropic_tools = registry.to_anthropic_format()

        # execute a tool by name
        result = registry.execute("get_weather", {"location": "Tokyo"})
    """

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """register a tool"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """list all registered tool names"""
        return list(self.tools.keys())

    def to_openai_format(self) -> list[dict]:
        """get all tools in OpenAI format"""
        return [tool.to_openai_schema() for tool in self.tools.values()]

    def to_anthropic_format(self) -> list[dict]:
        """get all tools in Anthropic format"""
        return [tool.to_anthropic_schema() for tool in self.tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        """
        execute a tool by name with given arguments

        Raises:
            ValueError: if tool not found or has no function
        """
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"tool not found: {name}")
        if not tool.function:
            raise ValueError(f"tool has no function: {name}")
        return tool.function(**args)
