"""
Base Specialist - Foundation for specialist agents.

Key Pattern: Agent-as-Tool
    Specialist agents extend BaseTool, allowing the orchestrator
    to treat agents the same way it treats regular tools.
    This reuses all existing tool infrastructure from tools/base.py.
"""

import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any

# enable imports from parent package
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from schemas import AgentProfile
from tools import BaseTool, ToolDefinition, ToolParameter, ToolResult


class BaseSpecialist(BaseTool):
    """
    Base class for specialist agents that can be called by the orchestrator.

    Key Insight: Agent-as-Tool Pattern
        By extending BaseTool, specialists can be registered in ToolRegistry
        and called exactly like regular tools. The orchestrator doesn't need
        to know if it's calling a simple function or another agent.

    Subclasses must implement:
        - _execute_mock(task): return mock response for testing
        - _execute_real(task): run actual agent logic with LLM

    Attributes:
        profile: defines this agent's role and capabilities
        use_mock: if True, return mock responses without LLM calls
        verbose: if True, print execution details
    """

    def __init__(
        self,
        profile: AgentProfile,
        use_mock: bool = True,
        verbose: bool = True,
    ):
        self.profile = profile
        self.use_mock = use_mock
        self.verbose = verbose

    @property
    def name(self) -> str:
        """unique identifier from profile"""
        return self.profile.name

    @property
    def definition(self) -> ToolDefinition:
        """tool definition describing this specialist's capabilities"""
        return ToolDefinition(
            name=self.name,
            description=f"{self.profile.role}. Capabilities: {self.profile.capability_description()}",
            parameters=[
                ToolParameter(
                    name="task",
                    param_type="string",
                    description="The task to delegate to this specialist agent",
                    required=True,
                ),
            ],
        )

    def execute(self, task: str, **kwargs: Any) -> ToolResult:
        """
        Execute this specialist agent with a task.

        Args:
            task: the subtask to perform
            **kwargs: additional arguments (ignored)

        Returns:
            ToolResult with the specialist's response
        """
        if self.verbose:
            print(f"\n  📋 [{self.name}] received task: {task[:60]}...")

        try:
            if self.use_mock:
                result = self._execute_mock(task)
            else:
                result = self._execute_real(task)

            if self.verbose:
                print(f"  ✅ [{self.name}] completed: {result[:60]}...")

            return ToolResult.ok(result)

        except Exception as e:
            error_msg = f"Specialist error: {str(e)}"
            if self.verbose:
                print(f"  ❌ [{self.name}] failed: {error_msg}")
            return ToolResult.fail(error_msg)

    @abstractmethod
    def _execute_mock(self, task: str) -> str:
        """
        Return mock response for testing without LLM.

        Subclasses should return realistic but fake responses
        that demonstrate what this specialist would produce.
        """
        pass

    def _execute_real(self, task: str) -> str:
        """
        Execute with real LLM call.

        Default implementation falls back to mock.
        Subclasses can override to add actual agent logic.
        """
        # default: fall back to mock if not overridden
        return self._execute_mock(task)
