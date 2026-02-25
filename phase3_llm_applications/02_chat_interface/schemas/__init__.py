"""
Chat Interface Data Models

Module Structure:
- role.py    → Role enum (SYSTEM, USER, ASSISTANT)
- message.py → Message dataclass

Usage:
    from models import Role, Message
"""

from .message import Message
from .role import Role

__all__ = ["Role", "Message"]
