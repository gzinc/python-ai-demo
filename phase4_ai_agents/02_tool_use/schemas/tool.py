"""
Tool Schemas - Data structures for tool definitions and results.
"""

from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """
    Describes a single parameter that a tool accepts.

    Attributes:
        name: Parameter name (e.g., "city", "query", "file_path")
        param_type: Data type ("string", "integer", "boolean", "number")
        description: What this parameter does
        required: Whether the parameter must be provided
        enum_values: Optional list of allowed values
    """
    name: str
    param_type: str
    description: str
    required: bool = True
    enum_values: list[str] | None = None


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool that an agent can use.

    This metadata tells the LLM what the tool does and how to call it.

    Attributes:
        name: Unique tool identifier (e.g., "web_search", "read_file")
        description: What the tool does (shown to LLM)
        parameters: List of parameters the tool accepts
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required_params = []

        for param in self.parameters:
            prop_def = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum_values is not None:
                prop_def["enum"] = param.enum_values

            properties[param.name] = prop_def

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
        success: Whether the tool executed without errors
        data: The result data (any type - string, dict, list, etc.)
        error: Error message if failed, None if success
    """
    success: bool
    data: str | dict | list | None = None
    error: str | None = None

    @classmethod
    def ok(cls, data: str | dict | list) -> "ToolResult":
        """Factory method for successful result."""
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        """Factory method for failed result."""
        return cls(success=False, data=None, error=error)

    def to_observation(self) -> str:
        """Convert result to string for agent observation."""
        if self.success:
            if isinstance(self.data, (dict, list)):
                import json
                return json.dumps(self.data, indent=2)
            return str(self.data)
        return f"ERROR: {self.error}"
