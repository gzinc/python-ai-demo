"""
Agent Config - configuration for agent behavior.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AgentConfig:
    """
    Configuration for ReActAgent.

    Settings control LLM provider, iteration limits, and output verbosity.
    """

    max_iterations: int = 10
    model: str | None = None  # None = use provider default
    provider: Literal["openai", "anthropic"] = "openai"
    temperature: float = 0.0
    verbose: bool = True
    timeout_seconds: int = 120
    max_retries: int = 2

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.max_iterations > 50:
            raise ValueError("max_iterations capped at 50 for safety")

        # set default model based on provider
        if self.model is None:
            self.model = "gpt-4o-mini" if self.provider == "openai" else "claude-3-haiku-20240307"
