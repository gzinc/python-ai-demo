"""
LangGraph Conditional Routing - Dynamic Path Selection

This module teaches conditional edges and dynamic routing in LangGraph:
- Conditional edges for if/else routing
- Decision functions that examine state
- Loops with termination conditions
- Multi-way branching
- Agent loops with conditional stopping

No API key required - uses mock logic for demonstrations.

Run with: uv run python -m phase7_frameworks.02_langgraph.conditional_routing
"""

from inspect import cleandoc
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section


# region Utility Functions
# endregion


# region Demo 1: Simple Conditional Routing
def demo_simple_conditional() -> None:
    """
    demonstrate basic conditional edge (if/else routing)

    Concept: Conditional Edges
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Conditional Edge Pattern                        │
    │                                                                  │
    │  Purpose: Route to different nodes based on state                │
    │                                                                  │
    │  Pattern:                                                        │
    │     def router(state) -> str:                                    │
    │         if condition:                                            │
    │             return "path_a"                                      │
    │         else:                                                    │
    │             return "path_b"                                      │
    │                                                                  │
    │     graph.add_conditional_edges(                                 │
    │         "decision_node",                                         │
    │         router,                                                  │
    │         {                                                        │
    │             "path_a": "node_a",                                  │
    │             "path_b": "node_b"                                   │
    │         }                                                        │
    │     )                                                            │
    │                                                                  │
    │  Graph Structure:                                                │
    │              ┌─→ A (if condition)                                │
    │     START → Router                                               │
    │              └─→ B (else)                                        │
    │                                                                  │
    │  Key Insight: Router function examines state, returns string     │
    │  String maps to node name in routing dictionary                  │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Simple Conditional Routing")

    class RouteState(TypedDict):
        """state with routing decision"""
        value: int
        path: str

    def router(state: RouteState) -> Literal["high", "low"]:
        """decide routing based on value"""
        if state["value"] > 10:
            print(f"   Router: value={state['value']} > 10 → 'high' path")
            return "high"
        else:
            print(f"   Router: value={state['value']} <= 10 → 'low' path")
            return "low"

    def high_path(state: RouteState) -> dict:
        """process high values"""
        print("   → High Path: Applying expensive processing")
        return {"path": "high"}

    def low_path(state: RouteState) -> dict:
        """process low values"""
        print("   → Low Path: Applying cheap processing")
        return {"path": "low"}

    # build graph with conditional routing
    graph = StateGraph(RouteState)
    graph.add_node("high", high_path)
    graph.add_node("low", low_path)

    # add conditional edges from START
    # router function examines state and returns "high" or "low"
    # this string maps to the node name in the routing dictionary
    graph.add_conditional_edges(
        START,
        router,
        {
            "high": "high",
            "low": "low"
        }
    )

    graph.add_edge("high", END)
    graph.add_edge("low", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure:
                   ┌─→ high (value > 10)
          START → Router
                   └─→ low (value <= 10)
    """))

    # test with different values
    print("\nTest 1: value=15")
    result1 = app.invoke({"value": 15, "path": ""})
    print(f"  Result: path='{result1['path']}'")

    print("\nTest 2: value=5")
    result2 = app.invoke({"value": 5, "path": ""})
    print(f"  Result: path='{result2['path']}'")

    print("\n✅ Key Takeaway: Router function + conditional_edges = dynamic routing")


# endregion


# region Demo 2: Multi-Way Branching
def demo_multi_way_routing() -> None:
    """
    demonstrate routing to multiple destinations (>2 paths)

    Concept: Multi-Way Routing
    ┌──────────────────────────────────────────────────────────────────┐
    │                   Multi-Way Branch Pattern                       │
    │                                                                  │
    │  Router Returns String → Maps to Node:                           │
    │                                                                  │
    │              ┌─→ "urgent" → urgent_handler                       │
    │              │                                                   │
    │     START → Router ─→ "normal" → normal_handler                  │
    │              │                                                   │
    │              └─→ "low" → low_priority_handler                    │
    │                                                                  │
    │  Router Function:                                                │
    │     def route_by_priority(state):                                │
    │         if state["priority"] == "urgent":                        │
    │             return "urgent"                                      │
    │         elif state["priority"] == "normal":                      │
    │             return "normal"                                      │
    │         else:                                                    │
    │             return "low"                                         │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Priority-based routing                                        │
    │  • Task type classification                                      │
    │  • Multi-specialist agent routing                                │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: Multi-Way Routing (3+ Paths)")

    class TaskState(TypedDict):
        """state with task classification"""
        task_type: str
        result: str

    def classify_task(state: TaskState) -> Literal["research", "analysis", "writing"]:
        """classify task and route appropriately"""
        task = state["task_type"]
        print(f"   Classifier: task_type='{task}'")

        if "research" in task:
            print("   → Routing to: research specialist")
            return "research"
        elif "analysis" in task or "analyze" in task:
            print("   → Routing to: analysis specialist")
            return "analysis"
        else:
            print("   → Routing to: writing specialist")
            return "writing"

    def research_specialist(state: TaskState) -> dict:
        """handle research tasks"""
        print("   🔍 Research Specialist: Gathering information...")
        return {"result": "Research completed with 10 sources"}

    def analysis_specialist(state: TaskState) -> dict:
        """handle analysis tasks"""
        print("   📊 Analysis Specialist: Analyzing data...")
        return {"result": "Analysis completed with insights"}

    def writing_specialist(state: TaskState) -> dict:
        """handle writing tasks"""
        print("   ✍️  Writing Specialist: Creating content...")
        return {"result": "Document written"}

    # build graph
    graph = StateGraph(TaskState)
    graph.add_node("research", research_specialist)
    graph.add_node("analysis", analysis_specialist)
    graph.add_node("writing", writing_specialist)

    graph.add_conditional_edges(
        START,
        classify_task,
        {
            "research": "research",
            "analysis": "analysis",
            "writing": "writing"
        }
    )

    graph.add_edge("research", END)
    graph.add_edge("analysis", END)
    graph.add_edge("writing", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure:
                      ┌─→ research (research tasks)
                      │
          START → Classifier ─→ analysis (analysis tasks)
                      │
                      └─→ writing (other tasks)
    """))

    # test different task types
    for task in ["do research on AI", "analyze sales data", "write report"]:
        print(f"\nTask: '{task}'")
        result = app.invoke({"task_type": task, "result": ""})
        print(f"  Result: {result['result']}")

    print("\n✅ Key Takeaway: One router can route to 3+ different nodes")


# endregion


# region Demo 3: Loop with Termination
def demo_loop_with_termination() -> None:
    """
    demonstrate conditional loop (continue or end)

    Concept: Loop Patterns with Termination
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Loop with Exit Condition                        │
    │                                                                  │
    │  Pattern:                                                        │
    │     START → process → should_continue?                           │
    │                           │           │                          │
    │                           ├─ continue ─┘ (loop back)             │
    │                           │                                      │
    │                           └─ finish → END                        │
    │                                                                  │
    │  Decision Function:                                              │
    │     def should_continue(state):                                  │
    │         if state["iterations"] < state["max_iterations"]:        │
    │             return "continue"                                    │
    │         return "finish"                                          │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Agent loops (iterate until solved)                            │
    │  • Retry loops (try until success)                               │
    │  • Refinement loops (improve until good enough)                  │
    │  • Batch processing (process items one by one)                   │
    │                                                                  │
    │  Safety: Always have max_iterations to prevent infinite loops!   │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Loop with Termination Condition")

    class LoopState(TypedDict):
        """state for iterative process"""
        value: int
        iterations: int
        max_iterations: int
        history: list[int]

    def process_step(state: LoopState) -> dict:
        """process one iteration"""
        new_value = state["value"] + 5
        new_iter = state["iterations"] + 1
        print(f"   Iteration {new_iter}: value={state['value']} → {new_value}")

        return {
            "value": new_value,
            "iterations": new_iter,
            "history": state["history"] + [new_value]
        }

    def should_continue(state: LoopState) -> Literal["continue", "finish"]:
        """decide whether to continue or finish"""
        if state["iterations"] < state["max_iterations"]:
            print(f"   Decision: {state['iterations']}/{state['max_iterations']} → continue")
            return "continue"
        else:
            print(f"   Decision: {state['iterations']}/{state['max_iterations']} → finish")
            return "finish"

    # build graph with loop
    graph = StateGraph(LoopState)
    graph.add_node("process", process_step)

    graph.add_edge(START, "process")

    graph.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",  # loop back to process
            "finish": END
        }
    )

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure (Loop):
          START → process → should_continue?
                       ↑         │         │
                       └─continue┘         └─finish→ END
    """))

    print("\nExecution (max_iterations=3):")
    result = app.invoke({
        "value": 0,
        "iterations": 0,
        "max_iterations": 3,
        "history": []
    })

    print("\n" + cleandoc(f"""
        Final State:
          value: {result['value']}
          iterations: {result['iterations']}
          history: {result['history']}
    """))

    print("\n" + cleandoc("""
        ✅ Key Takeaway: Conditional edge can loop back to same node
        ⚠️  Safety: Always have max_iterations to prevent infinite loops!
    """))


# endregion


# region Demo 4: Agent Loop with ReAct Pattern
def demo_agent_react_loop() -> None:
    """
    demonstrate agent ReAct loop with conditional stopping

    Concept: ReAct Agent Pattern with Conditional Termination
    ┌──────────────────────────────────────────────────────────────────┐
    │                    ReAct Agent Loop                              │
    │                                                                  │
    │  Pattern (Your Phase 4 Agent in LangGraph):                      │
    │                                                                  │
    │     START → think → act → should_continue?                       │
    │                  ↑            │          │                       │
    │                  └─ continue ─┘          └─ finish → END         │
    │                                                                  │
    │  State Schema:                                                   │
    │     - messages: list  # conversation history                     │
    │     - iterations: int # safety counter                           │
    │     - task_complete: bool  # termination flag                    │
    │                                                                  │
    │  Decision Logic:                                                 │
    │     def should_continue(state):                                  │
    │         if state["task_complete"]:                               │
    │             return "finish"                                      │
    │         if state["iterations"] >= max_iter:                      │
    │             return "finish"  # safety                            │
    │         return "continue"                                        │
    │                                                                  │
    │  This is how Phase 4 agents work in LangGraph!                   │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Agent ReAct Loop (Phase 4 Pattern)")

    class AgentState(TypedDict):
        """agent state with termination"""
        messages: Annotated[list[str], add_messages]
        iterations: int
        max_iterations: int
        task_complete: bool

    def think(state: AgentState) -> dict:
        """agent thinking step"""
        thought = f"Thought {state['iterations'] + 1}: Working on the problem..."
        print(f"   🤔 {thought}")

        # simulate task completion after 2 iterations
        complete = state["iterations"] >= 1
        if complete:
            thought += " (Solution found!)"

        return {
            "messages": [thought],
            "task_complete": complete
        }

    def act(state: AgentState) -> dict:
        """agent action step"""
        action = f"Action {state['iterations'] + 1}: Taking action..."
        print(f"   🎯 {action}")

        return {
            "messages": [action],
            "iterations": state["iterations"] + 1
        }

    def should_continue(state: AgentState) -> Literal["continue", "finish"]:
        """decide whether to continue agent loop"""
        if state["task_complete"]:
            print("   Decision: Task complete → finish")
            return "finish"
        elif state["iterations"] >= state["max_iterations"]:
            print("   Decision: Max iterations reached → finish")
            return "finish"
        else:
            print("   Decision: Continue working → continue")
            return "continue"

    # build agent graph
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)

    graph.add_edge(START, "think")
    graph.add_edge("think", "act")

    graph.add_conditional_edges(
        "act",
        should_continue,
        {
            "continue": "think",  # loop back to think
            "finish": END
        }
    )

    app = graph.compile()

    print("\n" + cleandoc("""
        Agent Graph Structure (ReAct Loop):
          START → think → act → should_continue?
                    ↑            │          │
                    └─ continue ─┘          └─ finish → END
    """))

    print("\nAgent Execution:")
    result = app.invoke({
        "messages": ["User: Solve this problem"],
        "iterations": 0,
        "max_iterations": 5,
        "task_complete": False
    })

    print("\n" + cleandoc(f"""
        Final State:
          iterations: {result['iterations']}
          task_complete: {result['task_complete']}
          total messages: {len(result['messages'])}
    """))

    print("\nAll messages:")
    for msg in result["messages"]:
        print(f"  • {msg}")

    print("\n" + cleandoc("""
        ✅ Key Takeaway: This is your Phase 4 ReAct agent in LangGraph!
        📝 Advantage: Declarative graph structure vs imperative while loop
    """))


# endregion


# region Demo 5: Error Handling with Retry
def demo_error_retry_pattern() -> None:
    """
    demonstrate error handling with conditional retry

    Concept: Retry Pattern with Conditional Logic
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Error Retry Pattern                           │
    │                                                                  │
    │  Pattern:                                                        │
    │     START → attempt → check_result?                              │
    │                  ↑         │        │                            │
    │                  └─ retry ─┘        └─ success → END             │
    │                                                                  │
    │  Decision Logic:                                                 │
    │     def check_result(state):                                     │
    │         if state["success"]:                                     │
    │             return "success"                                     │
    │         elif state["retries"] < max_retries:                     │
    │             return "retry"                                       │
    │         else:                                                    │
    │             return "failed"  # give up                           │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • API call retry logic                                          │
    │  • LLM generation with validation + regeneration                 │
    │  • Tool execution with fallback attempts                         │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Error Handling with Retry")

    class RetryState(TypedDict):
        """state for retry logic"""
        attempt: int
        max_attempts: int
        success: bool
        result: str

    def attempt_operation(state: RetryState) -> dict:
        """simulate operation that might fail"""
        attempt = state["attempt"] + 1
        print(f"   Attempt {attempt}/{state['max_attempts']}...")

        # simulate: succeed on 3rd attempt
        success = attempt >= 3

        if success:
            print("   ✅ Success!")
            return {
                "attempt": attempt,
                "success": True,
                "result": "Operation completed successfully"
            }
        else:
            print("   ❌ Failed, will retry...")
            return {
                "attempt": attempt,
                "success": False,
                "result": ""
            }

    def check_result(state: RetryState) -> Literal["success", "retry", "failed"]:
        """decide next action based on result"""
        if state["success"]:
            return "success"
        elif state["attempt"] < state["max_attempts"]:
            print("   Decision: Retry")
            return "retry"
        else:
            print("   Decision: Max attempts reached, giving up")
            return "failed"

    # build retry graph
    graph = StateGraph(RetryState)
    graph.add_node("attempt", attempt_operation)

    graph.add_edge(START, "attempt")

    graph.add_conditional_edges(
        "attempt",
        check_result,
        {
            "success": END,
            "retry": "attempt",  # loop back
            "failed": END
        }
    )

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure (Retry Pattern):
          START → attempt → check_result?
                       ↑         │     │
                       └─ retry ─┘     └─ success/failed → END
    """))

    print("\nExecution:")
    result = app.invoke({
        "attempt": 0,
        "max_attempts": 5,
        "success": False,
        "result": ""
    })

    print("\n" + cleandoc(f"""
        Final State:
          attempt: {result['attempt']}
          success: {result['success']}
          result: {result['result']}
    """))

    print("\n✅ Key Takeaway: Conditional routing enables retry/fallback patterns")


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Simple Conditional", "simple conditional", demo_simple_conditional),
    Demo("2", "Multi-Way Routing", "multi-way routing", demo_multi_way_routing),
    Demo("3", "Loop with Termination", "loop with termination", demo_loop_with_termination),
    Demo("4", "Agent ReAct Loop", "agent react loop", demo_agent_react_loop),
    Demo("5", "Error Retry Pattern", "error retry pattern", demo_error_retry_pattern),
]

# endregion

def main() -> None:
    """run all conditional routing demonstrations"""
    print(cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║                 LANGGRAPH CONDITIONAL ROUTING                      ║
        ║                                                                    ║
        ║  Key Concepts:                                                     ║
        ║  • conditional_edges = dynamic routing based on state              ║
        ║  • Router function returns string → maps to node                   ║
        ║  • Can route to multiple destinations (3+ paths)                   ║
        ║  • Can loop back to same node (with safety limits)                 ║
        ║  • Foundation for agent loops and retry logic                      ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝
    """))

    
    runner = MenuRunner(DEMOS, title="LangGraph Conditional Routing")
    runner.run()


if __name__ == "__main__":
    main()


# endregion
