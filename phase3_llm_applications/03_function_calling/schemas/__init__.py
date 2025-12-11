"""
Models package - Data classes for function calling

Like Java POJOs/DTOs - clean separation of data structures.
"""

from .tool import Tool, ToolParameter
from .tool_result import ToolResult

__all__ = ["Tool", "ToolParameter", "ToolResult"]
