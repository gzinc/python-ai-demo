# Phase 4: AI Agents

Build autonomous AI agents that reason, act, and accomplish goals.

## What You'll Learn

- The ReAct pattern (Reasoning + Acting)
- Building tools for agents to use
- Agent decision loops and state management
- Multi-agent collaboration and orchestration

## Prerequisites

- Phase 3: Function Calling (foundation for tool use)
- Understanding of LLM APIs

## Modules

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [01_simple_agent](01_simple_agent/) | ReAct Pattern | Agent loop, reasoning, observations |
| [02_tool_use](02_tool_use/) | Real Tools | Tool interface, registry, file/web tools |
| [03_multi_agent](03_multi_agent/) | Multi-Agent | Hierarchical orchestration, specialist agents |

## Core Concepts

See [CONCEPTS.md](CONCEPTS.md) for comprehensive documentation on:
- What agents are and how they differ from regular LLMs
- The relationship between functions, tools, and agents
- The ReAct pattern in depth
- Static code vs semantic understanding

## Quick Start

```bash
# Module 1: Simple Agent (ReAct pattern)
cd phase4_ai_agents/01_simple_agent
uv run python examples.py

# Module 2: Tool Use (real tools)
cd phase4_ai_agents/02_tool_use
uv run python examples.py

# Module 3: Multi-Agent (orchestration)
cd phase4_ai_agents/03_multi_agent
uv run python examples.py

# With API key for full demos
OPENAI_API_KEY=your-key uv run python examples.py
```

## The Big Picture

```
Phase 3: Function Calling          Phase 4: Agents
┌──────────────────────┐          ┌──────────────────────────────┐
│  LLM picks ONE tool  │    →     │  LLM picks tools in a LOOP   │
│  You execute it      │          │  Agent executes and continues │
│  Done                │          │  Until goal is achieved       │
└──────────────────────┘          └──────────────────────────────┘
```

## Module Progression

```
01_simple_agent/     → Learn the ReAct loop pattern
        ↓
02_tool_use/         → Add real tools (files, web, HTTP)
        ↓
03_multi_agent/      → Multiple agents working together
```