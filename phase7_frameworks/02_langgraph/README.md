# Module 2: LangGraph

**Purpose**: Build stateful multi-agent systems using graph-based orchestration

---

## Learning Objectives

- Understand graph-based agent orchestration
- Build state machines for complex workflows
- Implement multi-agent collaboration with LangGraph
- Add human-in-the-loop approvals
- Visualize agent execution flow

---

## What is LangGraph?

**LangGraph** = State machines + Graphs + Agents

```
Traditional Agent (Phase 4):
  while not done:
      thought = think()
      action = act(thought)
      observation = observe(action)

LangGraph:
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Planner │ --> │ Executor│ --> │ Reviewer│
  └─────────┘     └─────────┘     └─────────┘
       │                                 │
       └─────── (retry loop) ←───────────┘
```

---

## Key Concepts

### 1. StateGraph
**What it is**: A graph where nodes process/modify shared state

```python
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list[str]
    next_step: str

graph = StateGraph(AgentState)
graph.add_node("planner", plan_step)
graph.add_node("executor", execute_step)
graph.add_edge("planner", "executor")
graph.set_entry_point("planner")
```

### 2. Conditional Edges
**What it is**: Routing based on state

```python
def should_continue(state):
    if state["done"]:
        return "finish"
    else:
        return "continue"

graph.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue": "planner",
        "finish": END
    }
)
```

### 3. Human-in-the-Loop
**What it is**: Pause execution for human approval

```python
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint import MemorySaver

# Enable checkpoints for human intervention
memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["executor"])

# Run until interrupt
result = app.invoke(input, config={"configurable": {"thread_id": "1"}})

# Human approves, continue
result = app.invoke(None, config={"configurable": {"thread_id": "1"}})
```

---

## Module Structure

```
02_langgraph/
├── README.md                     # This file
├── 01_state_basics.py            # StateGraph, nodes, edges (foundation)
├── 02_conditional_routing.py     # Conditional edges, routing logic
├── 03_multi_agent.py             # Specialist agents collaborating
├── 04_human_in_loop.py           # Checkpoints, interrupts, approvals
├── 05_graph_visualization.py     # Render graphs with mermaid/graphviz
└── 06_migration_from_phase4.py   # Your multi-agent → LangGraph
```

**Learning Order**: Files are numbered to show progressive complexity
- **01**: Foundation (state management)
- **02**: Patterns (routing, branching)
- **03**: Applications (multi-agent)
- **04**: Production (human approval)
- **05**: Tools (visualization, debugging)
- **06**: Context (comparison to Phase 4)

---

## Comparison to Your Phase 4 Code

### Your Multi-Agent System (Phase 4)
```python
class MultiAgentOrchestrator:
    def __init__(self, specialists):
        self.specialists = specialists

    def orchestrate(self, task):
        # Manual routing logic
        if "research" in task:
            return self.specialists["researcher"].run(task)
        elif "analysis" in task:
            return self.specialists["analyst"].run(task)
        # etc...
```

**Challenges**:
- Hard to visualize
- No built-in retries
- Manual state management
- Difficult to add human approval

### LangGraph Version
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("researcher", research_agent)
graph.add_node("analyst", analysis_agent)
graph.add_node("writer", writer_agent)

def route_next(state):
    if needs_research(state):
        return "researcher"
    elif needs_analysis(state):
        return "analyst"
    else:
        return "writer"

graph.add_conditional_edges("researcher", route_next)
graph.add_edge("analyst", "writer")
graph.set_entry_point("researcher")

app = graph.compile()
```

**Benefits**:
- Visual graph representation
- Built-in cycle detection
- Easy state inspection
- Simple human-in-loop

---

## Common Patterns

### Pattern 1: Plan-Execute-Reflect

```python
from langgraph.graph import StateGraph, END

class PlanExecuteState(TypedDict):
    plan: str
    execution_result: str
    reflection: str

graph = StateGraph(PlanExecuteState)
graph.add_node("planner", create_plan)
graph.add_node("executor", execute_plan)
graph.add_node("reflector", reflect_on_result)

def should_replan(state):
    if state["reflection"] == "success":
        return END
    return "planner"

graph.add_conditional_edges("reflector", should_replan, {"planner": "planner", END: END})
graph.set_entry_point("planner")
```

### Pattern 2: Hierarchical Multi-Agent

```python
# Orchestrator delegates to specialists
graph.add_node("orchestrator", orchestrator_agent)
graph.add_node("specialist_1", specialist_1_agent)
graph.add_node("specialist_2", specialist_2_agent)

def route_to_specialist(state):
    task_type = state["task_type"]
    return f"specialist_{task_type}"

graph.add_conditional_edges("orchestrator", route_to_specialist)
graph.add_edge("specialist_1", "orchestrator")  # return to orchestrator
graph.add_edge("specialist_2", "orchestrator")
```

### Pattern 3: Human-in-the-Loop Approval

```python
from langgraph.checkpoint import MemorySaver

graph.add_node("generate_draft", draft_generator)
graph.add_node("human_review", human_reviewer)  # interrupt here
graph.add_node("publish", publisher)

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])

# 1. Generate draft
result = app.invoke(input, config={"configurable": {"thread_id": "1"}})

# 2. Human reviews, provides feedback
feedback = get_human_feedback()

# 3. Continue with feedback
result = app.invoke({"feedback": feedback}, config={"configurable": {"thread_id": "1"}})
```

---

## When to Use LangGraph

### ✅ Use LangGraph
- Multi-agent collaboration needed
- Complex branching logic (if-then-else routing)
- Human approval required
- Want to visualize workflow
- Cyclical workflows (retry loops)

### ❌ Skip LangGraph
- Simple linear chains
- Single agent with tools (use LangChain)
- Performance critical (graph overhead)
- Graph mental model doesn't fit

---

## Exercises

### Exercise 1: Refactor Phase 4 Multi-Agent
Convert your Phase 4 `MultiAgentOrchestrator` to LangGraph.

### Exercise 2: Add Conditional Routing
Implement smart routing based on task complexity.

### Exercise 3: Human-in-the-Loop
Add approval step before final output.

### Exercise 4: Graph Visualization
Render your agent graph using mermaid.

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [Graph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Human-in-the-Loop Guide](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)

---

## Next Steps

After this module:
1. Build multi-agent research system with LangGraph
2. Add visualization to understand flow
3. Implement human approval for critical decisions
4. Compare to your Phase 4 manual orchestration
5. Move to Module 3 (LlamaIndex) for RAG comparison
