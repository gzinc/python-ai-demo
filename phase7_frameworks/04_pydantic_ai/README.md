# Module 4: Pydantic AI — Type-Safe Agent Framework

## Learning Objectives

By the end of this module, you will:
- Understand how Pydantic AI differs from LangChain, LangGraph, and LlamaIndex
- Build agents with typed structured outputs using Pydantic models
- Register tools with `@agent.tool_plain` and `@agent.tool`
- Inject dependencies via `deps_type` for testable agents
- Write fast, deterministic unit tests using `TestModel`
- Make informed decisions about when Pydantic AI is the right choice

## Prerequisites

- [Phase 7 Module 1: LangChain](../01_langchain_basics/) — framework comparison context
- No API key needed for Demos 2, 4, 6 (TestModel)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for structured output demos

## The Core Idea

```
LangChain:
  chain.invoke("prompt") → Any
                           ↑
                    no type safety

Pydantic AI:
  agent.run_sync("prompt") → AgentRunResult[MyModel]
                              ↑
                     result.output: MyModel (typed!)
                     IDE autocomplete ✅  mypy validated ✅
```

Pydantic AI treats the LLM as a function with a typed return value.
Instead of parsing strings into dicts, you define a Pydantic model
and get an instance back — validated, typed, ready to use.

## Key Concepts

### 1. Agent

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    topic: str
    key_points: list[str]
    confidence: float

agent = Agent(
    "openai:gpt-4o-mini",   # model string
    output_type=Summary,     # Pydantic model → typed output
    deps_type=MyDeps,        # dependency type (optional)
    instructions="..."       # system prompt
)

result = agent.run_sync("Summarize quantum computing")
print(result.output.topic)       # str — typed, IDE autocomplete
print(result.output.key_points)  # list[str] — Pydantic validated
```

### 2. Two Tool Decorators

```
@agent.tool_plain              @agent.tool
─────────────────              ───────────
Pure function                  First arg: RunContext
No dep access                  Access ctx.deps, ctx.model

def my_tool(x: str) -> str:    def my_tool(ctx: RunContext[Deps], x: str) -> str:
    return do_something(x)         return ctx.deps.service.call(x)
```

Schema generated automatically from type hints — no JSON boilerplate.

### 3. Dependency Injection

```
Define deps ──────────────────────────────────────────────┐
                                                           │
@dataclass                                                 │
class AppDeps:                                             │
    db: Database                                           │
    http: httpx.AsyncClient                                │
    api_key: str                                           │
                                                           │
Agent(model, deps_type=AppDeps)                            │
                                                           │
@agent.tool                                                │
def search(ctx: RunContext[AppDeps], q: str) -> str:  ←────┘
    return ctx.deps.db.search(q)       # injected

# Production                    # Tests
deps = AppDeps(real_db, ...)    deps = AppDeps(FakeDB(), ...)
agent.run_sync("...", deps=deps) agent.run_sync("...", deps=deps)
```

Swap dependencies at call time → no global state → easily testable.

### 4. Model Strings (Provider Switching)

| String | Provider |
|--------|----------|
| `'openai:gpt-4o-mini'` | OpenAI |
| `'anthropic:claude-3-5-haiku-latest'` | Anthropic |
| `'google-gla:gemini-2.0-flash'` | Google |
| `'ollama:llama3.2'` | Local Ollama (no API key!) |
| `TestModel()` | No API key, deterministic |

Switch provider by changing one string — no code changes.

### 5. TestModel for Unit Tests

```python
from pydantic_ai.models.test import TestModel

# TestModel runs locally, no network, deterministic output
with my_agent.override(model=TestModel()):
    result = my_agent.run_sync("any prompt")
    assert isinstance(result.output, str)  # validates wiring

# TestModel actually calls tools and returns their output:
# {"tool_name": "tool_result"}
```

This makes agent unit tests fast (milliseconds), free, and
usable in CI without API credentials.

## Files in This Module

| File | Purpose |
|------|---------|
| [examples.py](examples.py) | 7 demos — Demos 2, 4, 6 runnable without API key |

## Running Examples

```bash
# interactive menu
uv run python -m phase7_frameworks.04_pydantic_ai.examples

# run demos 2, 4, 6 (TestModel — no API key)
uv run python -c "
from phase7_frameworks.04_pydantic_ai.examples import demo_core_api, demo_tool_use, demo_testing
demo_core_api()
demo_tool_use()
demo_testing()
"
```

## Demo Overview

| # | Demo | Runnable? | What You Learn |
|---|------|-----------|----------------|
| 1 | Philosophy | display | Pydantic AI vs LangChain vs LlamaIndex |
| 2 | Core API | ✅ TestModel | Agent, run_sync, result.output, usage |
| 3 | Structured Outputs | display | output_type=MyModel, nested Pydantic |
| 4 | Tool Use | ✅ TestModel | @tool_plain, @tool, auto schema |
| 5 | Dependency Injection | display | deps_type, RunContext, swapping deps |
| 6 | Testing | ✅ TestModel | agent.override(), pytest patterns |
| 7 | Framework Comparison | display | vs LangChain vs raw API |

## Framework Decision: When to Use Pydantic AI

```
New LLM task:
├── Structured typed output needed?
│   └── Yes → Pydantic AI (output_type=MyModel)
│
├── Complex multi-agent graph (branching, cycles)?
│   └── Yes → LangGraph
│
├── Document-heavy RAG at scale?
│   └── Yes → LlamaIndex
│
├── Need LangSmith / large ecosystem?
│   └── Yes → LangChain
│
├── Team uses Pydantic + FastAPI?
│   └── Yes → Pydantic AI (familiar patterns, Depends-like DI)
│
└── Simple one-shot LLM call?
    └── Raw API (no framework overhead)
```

## Pydantic AI vs LangChain Side-by-Side

| Concern | LangChain | Pydantic AI |
|---------|-----------|-------------|
| Output type | `Any` / `str` | `MyModel` (typed) |
| Tool schema | manual or `@tool` | auto from type hints |
| Testability | hard (mock LLM) | easy (TestModel) |
| Dependency injection | global vars | `deps_type` |
| Learning curve | high | low (if you know Pydantic) |
| Ecosystem | very large | growing |
| Magic factor | high | low |

## Key Takeaways

1. **Type safety is the differentiator** — `result.output` is your Pydantic model, not a string
2. **TestModel makes agents testable** — no API key, deterministic, fast
3. **Dependency injection enables clean architecture** — swap prod/test services
4. **Minimal magic** — you can read and understand what happens at every step
5. **Know Pydantic? You're 80% there** — same validation, same models

## Connection to Other Modules

```
Phase 3: Function Calling (manual JSON schema)
    ↓
Phase 7 Module 1: LangChain (@tool decorator, less boilerplate)
    ↓
Phase 7 Module 4: Pydantic AI (typed output + auto schema + testable)
    ↓
Phase 7 Module 5: Framework Comparison (when to choose which)
```

## Next Steps

- → [Module 5: Framework Comparison](../05_framework_comparison/) — decision framework across all frameworks
- Build a typed agent for a real use case (code reviewer, document summarizer)
- Explore [pydantic-ai docs](https://ai.pydantic.dev/) for streaming and multi-agent patterns
