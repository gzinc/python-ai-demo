"""
Tool Configuration - Settings for tool behavior.
"""

from dataclasses import dataclass


@dataclass
class ToolConfig:
    """
    Configuration settings for tool execution.

    Attributes:
        timeout_seconds: Max time to wait for tool execution
        max_retries: How many times to retry on failure
        enable_logging: Whether to log tool calls
        temp_directory: Where to store temporary files
    """
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_logging: bool = True
    temp_directory: str = "/tmp/agent_tools"
