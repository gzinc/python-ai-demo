"""
Agent State - enum for agent lifecycle states.
"""

from enum import Enum


class AgentState(Enum):
    """
    Agent lifecycle states.

    State transitions:
    PENDING → RUNNING → THINKING → ACTING → OBSERVING → (loop or FINISHED/TIMEOUT)
    """

    PENDING = "pending"
    RUNNING = "running"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    TIMEOUT = "timeout"
    ERROR = "error"
