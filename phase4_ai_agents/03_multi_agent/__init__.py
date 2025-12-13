"""
Module 3: Multi-Agent Systems

Hierarchical orchestration where one coordinator delegates to specialist agents.

Key Pattern: Agent-as-Tool
    Specialists are wrapped as tools, allowing the orchestrator
    to call them using the existing ToolRegistry infrastructure.

Module Structure:
    schemas/  - Data classes (AgentProfile, TeamConfig, etc.)
    tools/    - Tool infrastructure (BaseTool, ToolRegistry, etc.)
    agents/   - Specialist agents (Research, Analysis, Writer)
"""

from .schemas import AgentProfile, TeamConfig, DelegationResult, TeamResult
from .tools import (
    BaseTool,
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from .agents import (
    BaseSpecialist,
    ResearchAgent,
    AnalysisAgent,
    WriterAgent,
)
from .orchestrator import MultiAgentOrchestrator

__all__ = [
    # schemas
    "AgentProfile",
    "TeamConfig",
    "DelegationResult",
    "TeamResult",
    # tool infrastructure (from tools/)
    "BaseTool",
    "ToolRegistry",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    # agents (from agents/)
    "BaseSpecialist",
    "ResearchAgent",
    "AnalysisAgent",
    "WriterAgent",
    # orchestrator
    "MultiAgentOrchestrator",
]
