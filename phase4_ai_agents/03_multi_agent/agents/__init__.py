"""
Multi-Agent specialist agents.

Each specialist is wrapped as a tool so the orchestrator can delegate to them.
Tool infrastructure is in tools/ - agents only contain specialist logic.
"""

import sys
from pathlib import Path

# enable imports from parent package
_module_dir = Path(__file__).parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# tool infrastructure from tools/
from tools import (
    BaseTool,
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

# specialist agents
from .base_specialist import BaseSpecialist
from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from .writer_agent import WriterAgent

__all__ = [
    # tool infrastructure (from tools/)
    "BaseTool",
    "ToolRegistry",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    # specialist agents
    "BaseSpecialist",
    "ResearchAgent",
    "AnalysisAgent",
    "WriterAgent",
]
