"""
Agent State - enum for agent lifecycle states
"""

from enum import Enum


class AgentState(Enum):
    """
    agent lifecycle states

    State transitions:
    PENDING → RUNNING → THINKING → ACTING → OBSERVING → (loop or FINISHED/TIMEOUT)
    """

    PENDING = "pending"  # not started
    RUNNING = "running"  # actively processing
    THINKING = "thinking"  # generating reasoning
    ACTING = "acting"  # executing tool
    OBSERVING = "observing"  # processing result
    FINISHED = "finished"  # task complete
    TIMEOUT = "timeout"  # max iterations reached
    ERROR = "error"  # unrecoverable error
