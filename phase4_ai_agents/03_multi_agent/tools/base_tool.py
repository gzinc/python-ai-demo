"""
Tool Infrastructure - Base classes for tool-based communication.

Copied from Module 2 for self-containment. This module provides:
- ToolParameter: describes a single parameter
- ToolDefinition: complete tool specification for LLM
- ToolResult: standardized execution result
- BaseTool: abstract base class for all tools
- ToolRegistry: container for managing tools

These classes enable the Agent-as-Tool pattern where specialist agents
can be registered and called just like regular tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Data Classes (defined first - no dependencies)
# =============================================================================


@dataclass
class ToolParameter:
    """
    Describes a single parameter that a tool accepts.

    Attributes:
        name: parameter name (e.g., "task", "query")
        param_type: data type ("string", "integer", "boolean")
        description: what this parameter does
        required: whether the parameter must be provided
    """

    name: str
    param_type: str
    description: str
    required: bool = True


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool that an agent can use.

    This metadata tells the LLM what the tool does and how to call it.

    Attributes:
        name: unique tool identifier
        description: what the tool does (shown to LLM)
        parameters: list of parameters the tool accepts
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """convert to OpenAI function calling format"""
        properties = {}
        required_params = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.required:
                required_params.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }


@dataclass
class ToolResult:
    """
    Result from executing a tool.

    Standardized format for success/failure handling.

    Attributes:
        success: whether the tool executed without errors
        data: the result data (any type)
        error: error message if failed
    """

    success: bool
    data: str | dict | list | None = None
    error: str | None = None

    @classmethod
    def ok(cls, data: str | dict | list) -> "ToolResult":
        """factory method for successful result"""
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        """factory method for failed result"""
        return cls(success=False, data=None, error=error)

    def to_observation(self) -> str:
        """convert result to string for agent observation"""
        if self.success:
            if isinstance(self.data, (dict, list)):
                import json

                return json.dumps(self.data, indent=2)
            return str(self.data)
        return f"ERROR: {self.error}"


# =============================================================================
# Base Classes (depend on data classes above)
# =============================================================================


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Every tool must:
    1. Have a name (unique identifier)
    2. Provide a definition (describes parameters)
    3. Implement execute() (does the actual work)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """unique identifier for this tool"""
        pass

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """full definition describing the tool and its parameters"""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """execute the tool with given arguments"""
        pass

    def __call__(self, **kwargs: Any) -> ToolResult:
        """allow tool to be called like a function"""
        return self.execute(**kwargs)


class ToolRegistry:
    """
    Registry that holds all available tools (including specialist agents).

    The orchestrator uses this to look up and execute specialists by name.

    Usage:
        registry = ToolRegistry()
        registry.register(ResearchAgent(profile))
        registry.register(AnalysisAgent(profile))

        result = registry.execute("research_agent", task="find info about X")
    """

    def __init__(self):
        """initialize empty registry"""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """register a tool in the registry"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """get a tool by name, or None if not found"""
        return self._tools.get(name)

    def get_all_definitions(self) -> list[ToolDefinition]:
        """get definitions for all registered tools"""
        return [tool.definition for tool in self._tools.values()]

    def get_openai_tools(self) -> list[dict]:
        """get all tool definitions in OpenAI format"""
        return [tool.definition.to_openai_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """execute a tool by name with given arguments"""
        tool = self.get(tool_name)

        if tool is None:
            return ToolResult.fail(f"Tool not found: {tool_name}")

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult.fail(f"Tool execution error: {str(e)}")

    def list_tools(self) -> list[str]:
        """get list of all registered tool names"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """return number of registered tools"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """check if a tool is registered"""
        return name in self._tools