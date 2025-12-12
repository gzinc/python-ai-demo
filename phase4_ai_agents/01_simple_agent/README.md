# Simple Agent - ReAct Pattern Implementation

Build autonomous agents that reason, act, and observe in a loop.

## Learning Objectives

- Understand the ReAct (Reason-Act) pattern
- Build an agent loop with observation feedback
- Implement task planning and execution
- Handle agent termination conditions

## Key Concept: From Function Calling to Agents

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FUNCTION CALLING vs AGENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FUNCTION CALLING (Phase 3):                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                               │
│  │  User    │───►│   LLM    │───►│  Tool    │───► Done                      │
│  │  Query   │    │ Decides  │    │ Execute  │                               │
│  └──────────┘    └──────────┘    └──────────┘                               │
│                                                                             │
│  Single decision, single execution, done.                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AGENT (Phase 4):                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  User    │───►│  REASON  │───►│   ACT    │───►│ OBSERVE  │───┐           │
│  │  Task    │    │  (Think) │    │  (Tool)  │    │ (Result) │   │           │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │           │
│                       ▲                                         │           │
│                       └─────────────────────────────────────────┘           │
│                              Loop until task complete                       │
│                                                                             │
│  Multiple reasoning steps, multiple actions, iterative refinement.          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The ReAct Pattern

ReAct = **Re**ason + **Act**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ReAct AGENT LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User: "What's the weather in Tokyo and should I bring an umbrella?"        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ ITERATION 1                                                          │   │
│  │                                                                      │   │
│  │ THOUGHT: I need to check Tokyo's weather first.                      │   │
│  │          Let me call the weather API.                                │   │
│  │                                                                      │   │
│  │ ACTION: get_weather(city="Tokyo")                                    │   │
│  │                                                                      │   │
│  │ OBSERVATION: {"temp": 18, "conditions": "rainy", "humidity": 85}     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ ITERATION 2                                                          │   │
│  │                                                                      │   │
│  │ THOUGHT: Weather shows rainy conditions with 85% humidity.           │   │
│  │          I have enough information to answer.                        │   │
│  │                                                                      │   │
│  │ ACTION: finish(answer="Tokyo is 18°C and rainy. Yes, bring an        │   │
│  │                        umbrella - it's currently raining with        │   │
│  │                        85% humidity.")                               │   │
│  │                                                                      │   │
│  │ OBSERVATION: Task complete                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENT STATE MACHINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌──────────────┐                                         │
│                    │   PENDING    │ ◄─── Initial state                      │
│                    └──────┬───────┘                                         │
│                           │ start()                                         │
│                           ▼                                                 │
│                    ┌──────────────┐                                         │
│        ┌──────────►│   RUNNING    │◄──────────┐                             │
│        │           └──────┬───────┘           │                             │
│        │                  │                   │                             │
│        │     ┌────────────┴────────────┐      │                             │
│        │     ▼                         ▼      │                             │
│        │ ┌────────┐               ┌────────┐  │                             │
│        │ │THINKING│               │ ACTING │  │                             │
│        │ └───┬────┘               └───┬────┘  │                             │
│        │     │                        │       │                             │
│        │     └────────────┬───────────┘       │                             │
│        │                  ▼                   │                             │
│        │           ┌──────────────┐           │                             │
│        │           │  OBSERVING   │───────────┘                             │
│        │           └──────┬───────┘  (continue loop)                        │
│        │                  │                                                 │
│        │   ┌──────────────┴──────────────┐                                  │
│        │   ▼                             ▼                                  │
│   ┌────────────┐                  ┌──────────────┐                          │
│   │ max_iters  │                  │   FINISHED   │ ◄─── finish() called     │
│   │  reached   │                  └──────────────┘                          │
│   └─────┬──────┘                                                            │
│         ▼                                                                   │
│   ┌──────────────┐                                                          │
│   │   TIMEOUT    │ ◄─── Safety exit                                         │
│   └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         01_simple_agent/                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         schemas/                                    │    │
│  │                      (Data Classes)                                 │    │
│  │  • AgentState - current agent status and history                    │    │
│  │  • AgentAction - thought + action + observation                     │    │
│  │  • AgentConfig - max iterations, model, tools                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        agent.py                                     │    │
│  │                    (ReActAgent class)                               │    │
│  │  • run(task) - main agent loop                                      │    │
│  │  • think() - generate reasoning                                     │    │
│  │  • act() - execute tool or finish                                   │    │
│  │  • observe() - process tool result                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       examples.py                                   │    │
│  │                     (Demo Functions)                                │    │
│  │  • demo_simple_agent() - basic weather query                        │    │
│  │  • demo_multi_step() - complex multi-tool task                      │    │
│  │  • demo_planning() - task decomposition                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files in This Module

| File | Purpose |
|------|---------|
| [schemas/](schemas/) | AgentState, AgentAction, AgentConfig dataclasses |
| [agent.py](agent.py) | ReActAgent - core agent loop implementation |
| [examples.py](examples.py) | Interactive demos showing agent behavior |

## Key Differences from Function Calling

| Aspect | Function Calling | Agent |
|--------|------------------|-------|
| Iterations | Single | Multiple (loop) |
| State | Stateless | Maintains history |
| Decision | "Which tool?" | "What to do next?" |
| Termination | After tool call | When task complete |
| Complexity | Simple queries | Multi-step tasks |

## The Prompt That Makes It Work

```python
REACT_SYSTEM_PROMPT = """
You are an AI agent that solves tasks step by step.

For each step, you must respond in this exact format:

THOUGHT: [Your reasoning about what to do next]
ACTION: [tool_name(param="value")] OR finish(answer="final answer")

Available tools:
{tool_descriptions}

Rules:
1. Always start with THOUGHT explaining your reasoning
2. Then provide exactly ONE action
3. Use finish() when you have enough information to answer
4. Never make up information - use tools to get facts
"""
```

## Running the Examples

```bash
# run all demos
uv run python phase4_ai_agents/01_simple_agent/examples.py

# requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
```

## Exercises

1. **Basic Agent**: Run the weather agent and observe the thought process
2. **Add a Tool**: Add a "search" tool and have the agent use it
3. **Multi-Step Task**: Ask the agent to compare weather in two cities
4. **Timeout Handling**: See what happens when max_iterations is reached

## Next Steps

After this module:
- **02_tool_use**: More complex tools (web search, file operations)
- **03_multi_agent**: Multiple agents collaborating

## Key Insight

> The agent is just a **loop around function calling** with added reasoning.
> The LLM's job changes from "pick a tool" to "think about the task, then pick a tool or finish."
