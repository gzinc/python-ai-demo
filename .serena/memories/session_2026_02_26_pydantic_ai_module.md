# Session: 2026-02-26 — Phase 7: Pydantic AI Module Added

## What Was Done

Added Pydantic AI as Phase 7 Module 4, renumbering Framework Comparison to Module 5.

## Files Created

- `phase7_frameworks/04_pydantic_ai/__init__.py`
- `phase7_frameworks/04_pydantic_ai/examples.py` — 7 demos
- `phase7_frameworks/04_pydantic_ai/README.md`

## Files Modified

- `phase7_frameworks/04_framework_comparison/` → `phase7_frameworks/05_framework_comparison/` (git mv)
- `phase7_frameworks/05_framework_comparison/README.md` — added Pydantic AI to comparison table, decision tree, and 2 new scenarios
- `phase7_frameworks/README.md` — added module 4 row, updated comparison table with Pydantic AI column
- `.serena/memories/learning_progress.md` — added Pydantic AI section

## Package Installed

`pydantic-ai` added to pyproject.toml via `uv add pydantic-ai`

## Key API Facts (Pydantic AI)

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

# Agent constructor
agent = Agent(
    "openai:gpt-4o-mini",    # or TestModel() for no-API-key testing
    output_type=MyModel,      # NOT result_type
    deps_type=MyDeps,
    instructions="..."
)

# Result attribute: .output (not .data — though .data is alias)
result = agent.run_sync("prompt")
print(result.output)          # typed as output_type
print(result.usage().requests)

# Multi-turn
second = agent.run_sync("follow-up", message_history=first.all_messages())

# Tool decorators
@agent.tool_plain             # no RunContext, pure function
def my_tool(x: str) -> str: ...

@agent.tool                   # first arg is RunContext[DepsType]
def my_tool(ctx: RunContext[Deps], x: str) -> str:
    return ctx.deps.service.call(x)

# Testing
with production_agent.override(model=TestModel()):
    result = production_agent.run_sync("prompt")
```

## Critical Discovery: TestModel Actually Calls Tools

TestModel is NOT a stub — it calls registered tools and returns their output:
- With no tools: returns `'success (no tool calls)'`
- With tools: actually calls them and returns `'{"tool_name":"tool_result"}'`
- Demo 4 showed: `{"roll_die":"2","current_time":"00:19"}`
- Demo 6 showed: `call_log = ["get_temperature('a')"]` (called with dummy param 'a')

## Demo Overview (examples.py)

| # | Demo | Runnable? |
|---|------|-----------|
| 1 | Philosophy | display |
| 2 | Core API | ✅ TestModel |
| 3 | Structured Outputs | display |
| 4 | Tool Use | ✅ TestModel |
| 5 | Dependency Injection | display |
| 6 | Testing | ✅ TestModel |
| 7 | Framework Comparison | display |

## Pydantic AI Positioning

- Best for: typed structured output, FastAPI backends, testable agents, data extraction
- Not for: complex multi-agent graphs (LangGraph), document-heavy RAG (LlamaIndex), LangSmith monitoring (LangChain)
- Key differentiator vs LangChain: `result.output` is your Pydantic model (typed), not a string

## Pending Before Commit

User is reviewing `examples.py` and `05_framework_comparison/README.md` before committing.

## Next Steps (Pre-existing)

- LangGraph Demo 4: Sequential Multi-Node Graph (01_state_basics.py)
- LangChain Module 1: 04_memory/, 05_rag/ practical files
- LangChain agents Demo 6-7
