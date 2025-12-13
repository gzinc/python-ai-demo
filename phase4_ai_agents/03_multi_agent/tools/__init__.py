"""
Tool Infrastructure for Multi-Agent Module.

Contains base classes and utilities for tool-based agent communication.
Copied from Module 2 for self-containment.
"""

from .base_tool import (
    ToolParameter,
    ToolDefinition,
    ToolResult,
    BaseTool,
    ToolRegistry,
)

__all__ = [
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "BaseTool",
    "ToolRegistry",
]
