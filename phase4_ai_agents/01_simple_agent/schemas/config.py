"""
Agent Config - configuration for agent behavior
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentConfig:
    """
    configuration for ReActAgent

    Settings:
    ┌─────────────────────────────────────────────────┐
    │              AgentConfig                        │
    │                                                 │
    │  max_iterations: 10      ← safety limit         │
    │  model: "gpt-4o-mini"    ← LLM to use           │
    │  provider: "openai"      ← API provider         │
    │  temperature: 0.0        ← deterministic        │
    │  verbose: True           ← show thinking        │
    └─────────────────────────────────────────────────┘
    """

    max_iterations: int = 10  # prevent infinite loops
    model: str = "gpt-4o-mini"  # LLM model
    provider: Literal["openai", "anthropic"] = "openai"
    temperature: float = 0.0  # deterministic for reliability
    verbose: bool = True  # print thoughts/actions

    # timeout settings
    timeout_seconds: int = 120  # max total time

    # retry settings
    max_retries: int = 2  # retries on parse errors

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.max_iterations > 50:
            raise ValueError("max_iterations capped at 50 for safety")
