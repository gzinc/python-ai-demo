"""
Multi-Agent schemas for agent profiles and delegation results.
"""

from .agent_profile import AgentProfile, TeamConfig
from .delegation import DelegationResult, TeamResult

__all__ = [
    "AgentProfile",
    "TeamConfig",
    "DelegationResult",
    "TeamResult",
]
