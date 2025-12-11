"""
Message model for Chat Interface

Represents a single chat message with role and content.
"""

from dataclasses import dataclass

from .role import Role


@dataclass
class Message:
    """single chat message"""
    role: Role
    content: str

    def to_dict(self) -> dict:
        """convert to API format"""
        return {"role": self.role.value, "content": self.content}

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f"Message({self.role.value}: '{preview}...')"
