# Module 3: Multi-Agent Systems

Hierarchical orchestration where one coordinator delegates to specialist agents.

## Learning Objectives

After this module, you'll understand:
- How to build multi-agent systems with hierarchical coordination
- The Agent-as-Tool pattern for reusing existing infrastructure
- How orchestrators break down tasks and delegate to specialists
- Communication patterns between agents

## Key Concept: Agent-as-Tool

The central insight of this module: **agents can be wrapped as tools**.

This allows the orchestrator to treat specialist agents exactly like regular tools, reusing all existing tool infrastructure (ToolRegistry, ToolResult, etc.).

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                       │
│  "Break down task, delegate to specialists, synthesize"     │
├─────────────────────────────────────────────────────────────┤
│                      ToolRegistry                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ ResearchAgent│  │AnalysisAgent │  │ WriterAgent  │       │
│  │   (as Tool)  │  │   (as Tool)  │  │   (as Tool)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                   ┌──────────────┐                          │
│                   │   BaseTool   │  ← Same interface!       │
│                   └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Why Multi-Agent?

Single agents have limitations:
- Can't be expert at everything
- Limited context for complex tasks
- No specialization benefits

Multi-agent systems enable:
- **Specialization**: Each agent focuses on what it does best
- **Parallel work**: Independent subtasks can run concurrently
- **Better quality**: Specialists produce better results in their domain
- **Modularity**: Easy to add/remove specialists

## Module Architecture

```
03_multi_agent/
├── README.md               # This file
├── __init__.py            # Package exports
├── orchestrator.py        # MultiAgentOrchestrator
├── examples.py            # Demonstrations
├── schemas/               # Data classes
│   ├── __init__.py
│   ├── agent_profile.py   # AgentProfile, TeamConfig
│   └── delegation.py      # DelegationResult, TeamResult
├── tools/                 # Tool infrastructure (from Module 2)
│   ├── __init__.py
│   └── base_tool.py       # BaseTool, ToolRegistry, ToolResult
└── agents/                # Specialist agents
    ├── __init__.py
    ├── base_specialist.py # BaseSpecialist (extends BaseTool)
    ├── research_agent.py  # Research specialist
    ├── analysis_agent.py  # Analysis specialist
    └── writer_agent.py    # Writing specialist
```

## The Orchestration Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION FLOW                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  User: "Research AI trends and write a summary report"             │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ORCHESTRATOR                                                │   │
│  │                                                             │   │
│  │ THOUGHT: I need to research AI trends first                 │   │
│  │ ACTION: research_agent(task="Research AI trends in 2024")   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ RESEARCH AGENT                                              │   │
│  │ Returns: Key findings about AI trends, LLMs, etc.           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ORCHESTRATOR                                                │   │
│  │                                                             │   │
│  │ THOUGHT: Good research. Now I need to analyze the findings  │   │
│  │ ACTION: analysis_agent(task="Analyze AI trend patterns")    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ANALYSIS AGENT                                              │   │
│  │ Returns: Trend analysis, patterns, insights                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ORCHESTRATOR                                                │   │
│  │                                                             │   │
│  │ THOUGHT: Have research + analysis. Time to write report     │   │
│  │ ACTION: writer_agent(task="Write summary report")           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ WRITER AGENT                                                │   │
│  │ Returns: Formatted summary report                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ORCHESTRATOR                                                │   │
│  │                                                             │   │
│  │ THOUGHT: Task complete. I have all the information needed   │   │
│  │ ACTION: finish(answer="Here is your AI trends report...")   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# run all examples (mock mode - no API key needed)
cd phase4_ai_agents/03_multi_agent
uv run python examples.py

# run with real LLM (needs API key)
OPENAI_API_KEY=your-key uv run python examples.py
```

## Usage Examples

### Basic Orchestration

```python
from orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(use_mock=True)
result = orchestrator.run("Research Python frameworks and recommend one")

print(result.answer)
print(f"Delegations: {result.delegation_summary()}")
```

### Using Specialists Directly

```python
from agents import ResearchAgent, AnalysisAgent, WriterAgent

research = ResearchAgent(use_mock=True)
result = research.execute(task="Research cloud computing trends")
print(result.data)
```

### Custom Configuration

```python
from orchestrator import MultiAgentOrchestrator
from schemas import TeamConfig

config = TeamConfig(
    max_delegations=5,      # limit specialist calls
    max_iterations=10,      # limit orchestrator loops
    verbose=True,           # show detailed output
)

orchestrator = MultiAgentOrchestrator(config=config, use_mock=True)
```

## The Specialist Agents

| Agent | Role | Capabilities |
|-------|------|--------------|
| `ResearchAgent` | Research Specialist | Web search, data gathering, fact finding |
| `AnalysisAgent` | Analysis Specialist | Pattern recognition, comparison, evaluation |
| `WriterAgent` | Writing Specialist | Summarization, report writing, formatting |

## Creating Custom Specialists

```python
from schemas import AgentProfile
from agents import BaseSpecialist

class CodeReviewAgent(BaseSpecialist):
    def __init__(self, use_mock: bool = True):
        profile = AgentProfile(
            name="code_review_agent",
            role="Code Review Specialist",
            capabilities=["code analysis", "security review", "best practices"],
            system_prompt="You review code for quality and security issues.",
        )
        super().__init__(profile=profile, use_mock=use_mock)

    def _execute_mock(self, task: str) -> str:
        return f"Code review complete for: {task}"
```

## Java Equivalents

| Python | Java |
|--------|------|
| `BaseSpecialist` | `AgentService` interface |
| `MultiAgentOrchestrator` | `OrchestratorService` |
| `AgentProfile` | `AgentConfiguration` POJO |
| `TeamConfig` | `TeamConfiguration` |
| `TeamResult` | `OrchestratorResult` |
| `DelegationResult` | `DelegationRecord` |

## Key Patterns

### Agent-as-Tool Pattern

```python
class ResearchAgent(BaseSpecialist):  # extends BaseTool
    @property
    def name(self) -> str:
        return "research_agent"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Research specialist for finding information",
            parameters=[ToolParameter(name="task", ...)]
        )

    def execute(self, task: str, **kwargs) -> ToolResult:
        # agent logic here
        return ToolResult.ok(result)
```

### Registry Pattern

```python
registry = ToolRegistry()
registry.register(ResearchAgent())
registry.register(AnalysisAgent())

# orchestrator calls agents like tools
result = registry.execute("research_agent", task="...")
```

## Orchestration Patterns

### Hierarchical (This Module)

```
         Orchestrator
        /     |     \
    Research Analysis Writer
```

One coordinator makes all decisions, delegates to specialists.

### Peer-to-Peer (Not Implemented)

```
    Agent A ←→ Agent B
       ↑         ↑
       └────┬────┘
            ↓
         Agent C
```

Agents communicate directly with each other.

### Pipeline (Special Case)

```
    Research → Analysis → Writer
```

Fixed sequence of agents, each feeding into the next.

## Exercises

1. **Add a Specialist**: Create a `FactCheckerAgent` that verifies information
2. **Custom Pipeline**: Build a code review pipeline: `Analyze → Security Check → Report`
3. **Error Handling**: Improve how the orchestrator handles specialist failures
4. **Parallel Execution**: Modify orchestrator to call multiple specialists in parallel

## Key Takeaways

1. **Agent-as-Tool** - Wrap agents as tools to reuse existing infrastructure
2. **Specialization** - Different agents for different capabilities
3. **Orchestration** - A coordinator decides who does what
4. **Traceability** - Track all delegations for debugging and auditing
5. **Modularity** - Easy to add/remove specialists without changing orchestrator

## Further Reading

- [Phase 4 Overview](../README.md) - Complete agents phase
- [CONCEPTS.md](../CONCEPTS.md) - Agent fundamentals
- [Module 1: Simple Agent](../01_simple_agent/) - Basic ReAct pattern
- [Module 2: Tool Use](../02_tool_use/) - Tool infrastructure this module builds on
