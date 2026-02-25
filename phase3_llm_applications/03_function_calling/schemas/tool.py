"""
Tool and ToolParameter dataclasses

These define the schema that tells the LLM what functions are available.
Think of it like a Java interface definition - describes what exists.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """
    single parameter definition for a tool

    Similar to a method parameter in Java:
        public String getWeather(
            @Required String location,  // ToolParameter(name="location", required=True)
            String unit = "celsius"     // ToolParameter(name="unit", required=False, default="celsius")
        )
    """
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None  # for constrained choices
    default: Any | None = None


@dataclass
class Tool:
    """
    complete tool definition

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                         Tool                                 │
    │                                                              │
    │  name: "get_weather"         ← identifier for LLM           │
    │  description: "Get current   ← helps LLM decide when to use │
    │               weather..."                                    │
    │  parameters: [...]           ← what args it accepts         │
    │  function: get_weather_fn    ← actual Python function       │
    │                                                              │
    │  to_openai_schema() → OpenAI format                         │
    │  to_anthropic_schema() → Anthropic format                   │
    └─────────────────────────────────────────────────────────────┘
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    function: Callable | None = None  # the actual function to execute

    def to_openai_schema(self) -> dict:
        """
        convert to OpenAI tool format

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        """
        convert to Anthropic tool format

        Anthropic format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
