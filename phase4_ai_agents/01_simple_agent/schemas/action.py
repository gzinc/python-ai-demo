"""
Agent Action - represents one step in the agent loop
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentAction:
    """
    single agent action (one iteration of the loop)

    Structure:
    ┌─────────────────────────────────────────────────┐
    │                  AgentAction                    │
    │                                                 │
    │  thought: "I need to check the weather..."      │
    │  tool_name: "get_weather"                       │
    │  tool_args: {"city": "Tokyo"}                   │
    │  observation: "{'temp': 18, 'rainy': true}"     │
    │  is_final: False                                │
    └─────────────────────────────────────────────────┘
    """

    thought: str  # reasoning before action
    tool_name: Optional[str] = None  # tool to call (None if finish)
    tool_args: dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None  # result after execution
    is_final: bool = False  # True if this is finish() action

    @property
    def action_str(self) -> str:
        """format action as string for display"""
        if self.is_final:
            answer = self.tool_args.get("answer", "")
            return f"finish(answer={answer!r})"
        if self.tool_name:
            args_str = ", ".join(f'{k}="{v}"' for k, v in self.tool_args.items())
            return f"{self.tool_name}({args_str})"
        return "no action"

    def __repr__(self) -> str:
        return (
            f"AgentAction(thought={self.thought[:50]!r}..., "
            f"action={self.action_str}, is_final={self.is_final})"
        )


@dataclass
class AgentResult:
    """
    final result of agent execution

    Contains the full history of actions taken and the final answer.
    """

    answer: str  # final answer to the task
    actions: list[AgentAction] = field(default_factory=list)  # history
    iterations: int = 0  # how many loops
    success: bool = True  # completed successfully
    error: Optional[str] = None  # error message if failed

    @property
    def total_thoughts(self) -> int:
        """count reasoning steps"""
        return len([action for action in self.actions if action.thought])

    @property
    def total_tool_calls(self) -> int:
        """count tool executions"""
        return len([action for action in self.actions if action.tool_name and not action.is_final])
