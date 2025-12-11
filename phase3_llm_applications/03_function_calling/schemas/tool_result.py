"""
ToolResult dataclass - Result of executing a tool

Similar to a Response/Result wrapper in Java:
    public class ToolResult<T> {
        private boolean success;
        private T data;
        private String error;
        private String toolCallId;
    }
"""

from dataclasses import dataclass
from typing import Any, Optional
import json


@dataclass
class ToolResult:
    """
    result of tool execution

    Structure:
    ┌─────────────────────────────────────────────────────────────┐
    │                       ToolResult                            │
    │                                                             │
    │  success: True/False    ← did execution succeed?            │
    │  data: {...}            ← result if success                 │
    │  error: "..."           ← error message if failed           │
    │  tool_call_id: "..."    ← for matching with LLM request     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_openai_message(self) -> dict:
        """format as OpenAI tool result message"""
        content = json.dumps(self.data) if self.success else json.dumps({"error": self.error})
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": content,
        }

    def to_anthropic_content(self) -> dict:
        """format as Anthropic tool result content block"""
        content = self.data if self.success else {"error": self.error}
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": json.dumps(content) if isinstance(content, dict) else str(content),
        }
