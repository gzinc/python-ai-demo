"""
Agent Action and Result - represents steps and outcomes in the agent loop.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentAction:
    """
    Single agent action (one iteration of the ReAct loop).

    Contains:
    - thought: reasoning before taking action
    - tool_name: which tool to call
    - tool_args: arguments for the tool
    - observation: result after execution
    - is_final: True if this completes the task
    """

    thought: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None
    is_final: bool = False

    @property
    def action_str(self) -> str:
        """Format action as string for display."""
        if self.is_final:
            answer = self.tool_args.get("answer", "")
            return f"finish(answer={answer!r})"
        if self.tool_name:
            args_str = ", ".join(f'{k}="{v}"' for k, v in self.tool_args.items())
            return f"{self.tool_name}({args_str})"
        return "no action"

    def __repr__(self) -> str:
        thought_preview = self.thought[:50] if len(self.thought) > 50 else self.thought
        return f"AgentAction(thought={thought_preview!r}..., action={self.action_str})"


@dataclass
class AgentResult:
    """
    Final result of agent execution.

    Contains the answer, execution history, and status.
    """

    answer: str
    actions: list[AgentAction] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def total_thoughts(self) -> int:
        """Count reasoning steps."""
        return len([action for action in self.actions if action.thought])

    @property
    def total_tool_calls(self) -> int:
        """Count tool executions (excluding finish)."""
        return len([action for action in self.actions if action.tool_name and not action.is_final])
