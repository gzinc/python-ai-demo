"""
Delegation result schemas for tracking multi-agent execution.

DelegationResult tracks a single agent delegation.
TeamResult tracks the complete orchestration outcome.
"""

from dataclasses import dataclass, field


@dataclass
class DelegationResult:
    """
    Result of delegating a task to a specialist agent.

    Tracks what was asked, who handled it, and what they returned.
    Used by the orchestrator to build context for next decisions.

    Attributes:
        agent_name: which specialist handled this task
        task: the subtask that was delegated
        result: the specialist's response
        success: whether the delegation succeeded
        error: error message if delegation failed
    """

    agent_name: str
    task: str
    result: str
    success: bool = True
    error: str = ""

    def to_observation(self) -> str:
        """format as observation string for orchestrator prompt"""
        if self.success:
            return f"[{self.agent_name}] completed task: {self.result}"
        else:
            return f"[{self.agent_name}] failed: {self.error}"


@dataclass
class TeamResult:
    """
    Final result of multi-agent orchestration.

    Contains the synthesized answer plus a trace of all delegations,
    useful for understanding how agents collaborated.

    Attributes:
        answer: the final synthesized response
        delegations: list of all agent delegations made
        iterations: number of orchestrator loop iterations
        success: whether the task was completed successfully
        error: error message if orchestration failed
    """

    answer: str
    delegations: list[DelegationResult] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: str = ""

    def delegation_summary(self) -> str:
        """summarize which agents were used and how many times"""
        if not self.delegations:
            return "no delegations made"

        agent_counts: dict[str, int] = {}
        for delegation in self.delegations:
            agent_counts[delegation.agent_name] = agent_counts.get(delegation.agent_name, 0) + 1

        parts = [f"{name}: {count}x" for name, count in agent_counts.items()]
        return ", ".join(parts)

    def trace(self) -> str:
        """format full delegation trace for debugging"""
        lines = [f"=== Team Result (iterations={self.iterations}) ==="]
        lines.append(f"success: {self.success}")

        if self.delegations:
            lines.append("\n--- Delegation Trace ---")
            for i, d in enumerate(self.delegations, 1):
                lines.append(f"{i}. [{d.agent_name}] task: {d.task[:50]}...")
                lines.append(f"   result: {d.result[:100]}...")

        lines.append(f"\n--- Final Answer ---\n{self.answer}")
        return "\n".join(lines)
