# AI Agents: Core Concepts

Understanding what agents are, how they differ from regular LLMs, and the relationship between agents, tools, and functions.

---

## Table of Contents

- [What is an Agent?](#what-is-an-agent)
- [Regular LLM vs Agent](#regular-llm-vs-agent)
- [Tools, Functions, and Their Relationship](#tools-functions-and-their-relationship)
- [The ReAct Pattern](#the-react-pattern)
- [Static Code vs Semantic Understanding](#static-code-vs-semantic-understanding)
- [Why This Matters](#why-this-matters)
- [Connection to Other Phases](#connection-to-other-phases)

---

## What is an Agent?

**Simple definition:** An agent is an LLM that can take actions, not just talk.

A regular chatbot generates text responses. An agent goes further - it can:
- Decide what actions to take
- Execute those actions (via tools)
- Observe the results
- Continue until the task is complete

Think of it like the difference between:
- **Advisor**: "Here's how you could check the weather..."
- **Assistant**: *Actually checks the weather and tells you the result*

### The Name "Agent"

The term comes from the idea of acting on someone's behalf - like a travel agent or real estate agent. You give the agent a goal, and it figures out the steps to achieve it.

---

## Regular LLM vs Agent

| Aspect | Regular LLM | Agent |
|--------|-------------|-------|
| **Response** | One-shot answer | Iterative until done |
| **Capabilities** | Text generation only | Text + actions (tools) |
| **Flow** | Ask → Answer | Ask → Think → Act → Observe → Repeat → Answer |
| **Autonomy** | None | Decides what to do next |
| **State** | Stateless (per request) | Maintains goal and progress |

### Example: "What's the weather in Tel Aviv?"

**Regular LLM:**
> "I don't have access to real-time weather data. You can check weather.com or..."

**Agent:**
1. *Thinks:* "I need current weather data for Tel Aviv"
2. *Acts:* Calls `weather_api(city="Tel Aviv")`
3. *Observes:* `{"temp": 22, "conditions": "sunny"}`
4. *Answers:* "It's 22°C and sunny in Tel Aviv right now!"

The agent didn't just know about the weather - it **went and checked**.

---

## Tools, Functions, and Their Relationship

This is where terminology can get confusing. Let's clarify:

### What is a Function?

A function is just code that does something:

```python
def search_web(query: str) -> list[dict]:
    """Search the web and return results."""
    # ... implementation
    return results
```

### What is a Tool?

A tool is a function **wrapped with metadata** so an LLM can understand it:

```python
Tool(
    name="web_search",
    description="Search the web for information. Use when you need current data.",
    parameters=[
        Parameter(name="query", type="string", description="The search query")
    ],
    function=search_web  # The actual function to call
)
```

### The Key Insight

**Tool = Function + Description**

The description is what allows the LLM to:
1. **Know when** to use the tool (matches user intent to tool purpose)
2. **Know how** to use it (understands what each parameter means)

### What is an Agent?

**Agent = LLM + Tools + Decision Loop**

```
┌───────────────────────────────────────────┐
│                 AGENT                     │
│  ┌─────────┐     ┌─────────────────────┐  │
│  │   LLM   │───▶ │   Decision Loop     │  │
│  │ (Brain) │     │ (Think/Act/Observe) │  │
│  └─────────┘     └─────────────────────┘  │
│        │                                  │
│        ▼                                  │
│  ┌─────────────────────────────────────┐  │
│  │              TOOLS                  │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐         │  │
│  │  │search│ │ read │ │ api  │  ...    │  │
│  │  │ web  │ │ file │ │ call │         │  │
│  │  └──────┘ └──────┘ └──────┘         │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
```

### Summary Table

| Concept | What It Is | Example |
|---------|------------|---------|
| **Function** | Code that does something | `def search(q): ...` |
| **Tool** | Function + metadata for LLM | Function + name + description + params |
| **Agent** | LLM + tools + reasoning loop | Autonomous task executor |

---

## The ReAct Pattern

ReAct stands for **Reasoning and Acting**. It's the core pattern that makes agents work.

> 📄 **Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)

### The Loop

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│    ┌─────────┐      ┌─────────┐      ┌──────────┐    │
│    │  THINK  │─────▶│   ACT   │─────▶│ OBSERVE  │    │
│    │         │      │         │      │          │    │
│    │ "What   │      │ Call a  │      │ See what │    │
│    │ should  │      │ tool    │      │ happened │    │
│    │ I do?"  │      │         │      │          │    │
│    └─────────┘      └─────────┘      └──────────┘    │
│         ▲                                  │         │
│         │                                  │         │
│         └──────────────────────────────────┘         │
│                                                      │
│                   Repeat until done                  │
│                         │                            │
│                         ▼                            │
│                  ┌────────────┐                      │
│                  │   ANSWER   │                      │
│                  └────────────┘                      │
└──────────────────────────────────────────────────────┘
```

### Example Trace

**Task:** "Find Python tutorials and save the top 3 to a file"

```
THOUGHT: I need to search for Python tutorials first.
ACTION: web_search(query="Python tutorials 2024")
OBSERVATION: [{"title": "Real Python", "url": "..."}, {"title": "Python.org", ...}, ...]

THOUGHT: Good, I have results. Now I need to format and save the top 3.
ACTION: write_file(path="tutorials.txt", content="1. Real Python\n2. Python.org\n3. ...")
OBSERVATION: File written successfully: tutorials.txt (156 bytes)

THOUGHT: Task complete. I searched for tutorials and saved the top 3 to a file.
ANSWER: I found Python tutorials and saved the top 3 to tutorials.txt:
        1. Real Python - https://realpython.com
        2. Python.org Tutorial - https://python.org/tutorial
        3. W3Schools Python - https://w3schools.com/python
```

### Why ReAct Works

1. **Explicit reasoning** - The LLM explains its thinking (easier to debug)
2. **Grounded actions** - Actions are based on reasoning, not random
3. **Iterative refinement** - Can adjust based on observations
4. **Transparent** - You can see exactly what the agent did and why

### Other Agent Patterns

ReAct is the most common, but not the only approach:

| Pattern | How It Works | Trade-off |
|---------|--------------|-----------|
| **ReAct** | Think → Act → Observe (interleaved) | Flexible, but can be slow |
| **Plan-and-Execute** | Plan all steps first → Execute all | Faster, but less adaptive |
| **Reflexion** | ReAct + self-critique after failures | Better learning, more tokens |
| **Tree of Thoughts** | Explore multiple reasoning paths | Better for complex problems |

We focus on ReAct because it's the foundation - other patterns build on similar ideas.

---

## Static Code vs Semantic Understanding

This is perhaps the most important conceptual difference.

### Static Code: Compile-Time Decisions

In traditional programming, the developer decides everything at write-time:

```python
# Developer decided: call this function with these exact arguments
result = weather_api.get_temperature(
    city="Tel Aviv",
    units="celsius"
)
```

The code is fixed. It will always:
- Call `get_temperature`
- Pass "Tel Aviv" as the city
- Use celsius

### Agent: Runtime Semantic Understanding

With an agent, the LLM understands the **meaning** and decides at runtime:

```
User: "Is it cold in Tel Aviv?"

LLM reasoning:
- "cold" → user wants temperature → I should use weather_tool
- "Tel Aviv" → that's a city → maps to the 'city' parameter
- User is asking about comfort → probably wants celsius
- Generate: weather_tool(city="Tel Aviv", units="celsius")
```

### The Key Differences

| Aspect | Static Code | Agent (LLM) |
|--------|-------------|-------------|
| **When decided** | Compile/write time | Runtime (per request) |
| **Who decides** | Developer | LLM |
| **Based on** | Code logic | Semantic understanding |
| **Argument binding** | By position/name | By meaning/intent |
| **Flexibility** | Fixed behavior | Adapts to context |

### Why Tool Descriptions Matter

In static code, function documentation is for humans. In agents, **tool descriptions are for the LLM**:

```python
# This description IS the interface for the LLM
Tool(
    name="send_email",
    description="""
    Send an email to a recipient. Use this when the user wants to
    communicate with someone via email. Do NOT use for internal notes.
    """,
    parameters=[
        Parameter(
            name="to",
            description="Email address of the recipient (must be valid email format)"
        ),
        Parameter(
            name="subject",
            description="Email subject line - keep it concise and descriptive"
        ),
        Parameter(
            name="body",
            description="The email content - can be formal or informal based on context"
        )
    ]
)
```

Good descriptions help the LLM:
- Know **when** to use the tool
- Understand **what** each parameter means
- Make **appropriate** decisions about values

### Analogy: Navigation

| Static Code | Agent |
|-------------|-------|
| GPS: "In 200m, turn left onto Main St" | Human: "Oh, there's a coffee shop! Let's stop." |
| Follows pre-programmed route | Understands context, makes decisions |
| Can't adapt to new information | Responds to real-time observations |

---

## Why This Matters

Understanding these concepts helps you:

### 1. Design Better Tools

Knowing that the LLM reads descriptions to decide:
- Write clear, specific descriptions
- Include when to use AND when not to use
- Be explicit about parameter expectations

### 2. Debug Agent Behavior

When an agent does something unexpected:
- Check the tool descriptions - is the intent clear?
- Look at the reasoning trace - what did it think?
- Consider if observations gave enough information

### 3. Understand Limitations

Agents are powerful but:
- Only as good as their tools
- Can misunderstand vague descriptions
- May take unexpected paths to goals
- Need guardrails for safety

### 4. See the Bigger Picture

The evolution:
```
Functions → APIs → Function Calling → Agents → Multi-Agent Systems
    │          │           │              │              │
    └──────────┴───────────┴──────────────┴──────────────┘
                Building blocks for AI systems
```

---

## Connection to Other Phases

### Phase 3: Function Calling

Function calling is the **foundation** for agents:

| Function Calling (Phase 3) | Agents (Phase 4) |
|---------------------------|------------------|
| LLM picks ONE function | LLM picks functions in a LOOP |
| You execute it | Agent executes and continues |
| Single action | Multi-step workflows |
| Response ends | Continues until goal met |

Think of it as:
- **Function calling** = Teaching LLM to use a single tool
- **Agent** = Giving LLM autonomy to use tools as needed

### Module Progression in Phase 4

```
01_simple_agent/     → Basic ReAct loop, understanding the pattern
02_tool_use/         → Real tools, practical capabilities
03_multi_agent/      → Multiple agents working together
```

Each module builds on the previous, adding more capability and complexity.

---

## Summary

| Concept | One-Line Summary |
|---------|------------------|
| **Agent** | LLM that can take actions, not just talk |
| **Tool** | Function + description so LLM knows how to use it |
| **ReAct** | Think → Act → Observe → Repeat until done |
| **Semantic binding** | LLM understands meaning, not just code |
| **Key difference** | Agents decide at runtime what to do; static code is fixed |

---

## Further Reading

- [Module 01: Simple Agent](01_simple_agent/) - ReAct pattern implementation
- [Module 02: Tool Use](02_tool_use/) - Practical tool system
- [Phase 3: Function Calling](../phase3_llm_applications/03_function_calling/) - Foundation for tool use
