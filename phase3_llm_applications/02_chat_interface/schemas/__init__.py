"""
Chat Interface Data Models

Module Structure:
- role.py    → Role enum (SYSTEM, USER, ASSISTANT)
- message.py → Message dataclass

Usage:
    from models import Role, Message
"""

from .role import Role
from .message import Message

__all__ = ["Role", "Message"]
