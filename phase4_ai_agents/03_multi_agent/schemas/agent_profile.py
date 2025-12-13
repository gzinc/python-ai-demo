"""
Agent profile and team configuration schemas.

AgentProfile defines a specialist agent's identity and capabilities.
TeamConfig configures the multi-agent orchestration settings.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentProfile:
    """
    Defines a specialist agent's identity and capabilities.

    The profile tells the orchestrator what this agent can do,
    so it knows when to delegate tasks to this specialist.

    Attributes:
        name: unique identifier (e.g., "research_agent")
        role: human-readable role description (e.g., "Research Specialist")
        capabilities: list of things this agent can do
        system_prompt: instructions for how this agent should behave
    """

    name: str
    role: str
    capabilities: list[str]
    system_prompt: str = ""

    def __post_init__(self) -> None:
        """validate profile fields"""
        if not self.name:
            raise ValueError("agent name cannot be empty")
        if not self.role:
            raise ValueError("agent role cannot be empty")
        if not self.capabilities:
            raise ValueError("agent must have at least one capability")

    def capability_description(self) -> str:
        """format capabilities as a readable string"""
        return ", ".join(self.capabilities)


@dataclass
class TeamConfig:
    """
    Configuration for multi-agent orchestration.

    Controls how the orchestrator manages specialist agents,
    including model selection and execution limits.

    Attributes:
        orchestrator_model: LLM model for the orchestrator
        specialist_model: LLM model for specialist agents
        max_delegations: maximum number of agent calls per task
        max_iterations: maximum orchestrator loop iterations
        verbose: whether to print detailed execution logs
        provider: LLM provider (openai or anthropic)
    """

    orchestrator_model: str = "gpt-4o-mini"
    specialist_model: str = "gpt-4o-mini"
    max_delegations: int = 10
    max_iterations: int = 15
    verbose: bool = True
    provider: Literal["openai", "anthropic"] = "openai"

    def __post_init__(self) -> None:
        """validate configuration bounds"""
        if self.max_delegations < 1:
            raise ValueError("max_delegations must be at least 1")
        if self.max_delegations > 50:
            raise ValueError("max_delegations cannot exceed 50")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_iterations > 100:
            raise ValueError("max_iterations cannot exceed 100")
