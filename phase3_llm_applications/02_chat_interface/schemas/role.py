"""
Role enum for Chat Interface

Defines the standard message roles in chat conversations.
"""

from enum import Enum


class Role(str, Enum):
    """message roles in chat"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
