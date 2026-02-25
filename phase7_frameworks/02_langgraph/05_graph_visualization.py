"""
LangGraph Graph Visualization - Mermaid and Graphviz Rendering

This module teaches graph visualization in LangGraph:
- Mermaid diagram generation
- ASCII diagram generation
- Visualizing graph structure
- Understanding graph flow
- Debugging with visualizations

No API key required - focuses on graph structure visualization.

Run with: uv run python -m phase7_frameworks.02_langgraph.graph_visualization
"""

from inspect import cleandoc
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section

# region Utility Functions
# endregion


# region Demo 1: Simple Linear Graph
def demo_simple_linear() -> None:
    """
    visualize simple linear graph

    Concept: Graph Visualization
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Why Visualize Graphs?                           │
    │                                                                  │
    │  Benefits:                                                       │
    │  • Understand graph structure at a glance                        │
    │  • Debug routing logic visually                                  │
    │  • Document agent workflows                                      │
    │  • Communicate design to stakeholders                            │
    │  • Verify graph correctness before execution                     │
    │                                                                  │
    │  Visualization Methods:                                          │
    │  1. Mermaid (web-based, interactive)                             │
    │  2. ASCII (terminal, simple)                                     │
    │  3. Graphviz (PDF/PNG, publication-quality)                      │
    │                                                                  │
    │  LangGraph API:                                                  │
    │     graph = StateGraph(...)                                      │
    │     app = graph.compile()                                        │
    │     mermaid = app.get_graph().draw_mermaid()                     │
    │     ascii_art = app.get_graph().draw_ascii()                     │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Simple Linear Graph Visualization")

    class SimpleState(TypedDict):
        """minimal state"""
        value: int

    def step_a(state: SimpleState) -> dict:
        """first step"""
        return {"value": state["value"] + 1}

    def step_b(state: SimpleState) -> dict:
        """second step"""
        return {"value": state["value"] * 2}

    def step_c(state: SimpleState) -> dict:
        """third step"""
        return {"value": state["value"] - 3}

    # build simple graph
    graph = StateGraph(SimpleState)
    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("step_c", step_c)

    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "step_c")
    graph.add_edge("step_c", END)

    app = graph.compile()

    print("\nGraph Structure: Linear Chain")
    print("  START → step_a → step_b → step_c → END")

    # get ASCII visualization
    try:
        ascii_diagram = app.get_graph().draw_ascii()
        print("\nASCII Visualization:")
        print(ascii_diagram)
    except Exception:
        print("\nASCII visualization not available")

    # get Mermaid diagram
    try:
        mermaid = app.get_graph().draw_mermaid()
        print(cleandoc(            """

                Mermaid Diagram:
                mermaid
                💡 Copy the Mermaid code above to https://mermaid.live to view it!
            """))
    except Exception as e:
        print(f"\nMermaid visualization error: {e}")

    print("\n✅ Key Takeaway: Visualization reveals graph structure instantly")


# endregion


# region Demo 2: Conditional Routing Graph
def demo_conditional_graph() -> None:
    """
    visualize graph with conditional edges

    Concept: Visualizing Conditional Logic
    ┌──────────────────────────────────────────────────────────────────┐
    │               Conditional Edge Visualization                     │
    │                                                                  │
    │  What to Look For:                                               │
    │  • Diamond shapes = decision points                              │
    │  • Multiple outgoing edges = conditional routing                 │
    │  • Edge labels = routing conditions                              │
    │  • Parallel paths = alternative execution flows                  │
    │                                                                  │
    │  Visual Debugging:                                               │
    │  • Missing edge? Graph incomplete                                │
    │  • Unexpected path? Router logic issue                           │
    │  • Cycles? Check termination conditions                          │
    │  • Dead ends? Missing END connections                            │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: Conditional Routing Visualization")

    class RouteState(TypedDict):
        """state with routing"""
        value: int
        path: str

    def router(state: RouteState) -> Literal["high", "low"]:
        """conditional router"""
        return "high" if state["value"] > 10 else "low"

    def high_path(state: RouteState) -> dict:
        """high value path"""
        return {"path": "high"}

    def low_path(state: RouteState) -> dict:
        """low value path"""
        return {"path": "low"}

    # build conditional graph
    graph = StateGraph(RouteState)
    graph.add_node("high", high_path)
    graph.add_node("low", low_path)

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
            Graph Structure: Conditional Routing
                          ┌─→ high (value > 10)
              START → router
                          └─→ low (value <= 10)
        """))

    # ASCII visualization
    try:
        ascii_diagram = app.get_graph().draw_ascii()
        print("\nASCII Visualization:")
        print(ascii_diagram)
    except Exception:
        print("\nASCII visualization not available")

    # Mermaid diagram
    try:
        mermaid = app.get_graph().draw_mermaid()
        print(cleandoc(            """

                Mermaid Diagram:
                mermaid
                💡 Notice the conditional edges in the diagram!
            """))
    except Exception as e:
        print(f"\nMermaid visualization error: {e}")

    print("\n✅ Key Takeaway: Conditional edges show as branches in visualization")


# endregion


# region Demo 3: Loop Pattern Graph
def demo_loop_graph() -> None:
    """
    visualize graph with loop (cycle)

    Concept: Visualizing Loops
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Loop Visualization                            │
    │                                                                  │
    │  What to Look For:                                               │
    │  • Back edges = loops (edge pointing to earlier node)            │
    │  • Multiple paths from decision node                             │
    │  • One path loops back, one path exits                           │
    │                                                                  │
    │  Visual Debugging:                                               │
    │  • Infinite loop? Check exit condition edge                      │
    │  • Not looping? Router might always return exit                  │
    │  • Wrong loop target? Edge points to wrong node                  │
    │                                                                  │
    │  Safety Check:                                                   │
    │  • Every loop MUST have an exit path                             │
    │  • Visualize to verify exit condition exists                     │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Loop Pattern Visualization")

    class LoopState(TypedDict):
        """state with iteration"""
        counter: int
        max_iterations: int

    def process(state: LoopState) -> dict:
        """process step"""
        return {"counter": state["counter"] + 1}

    def should_continue(state: LoopState) -> Literal["continue", "finish"]:
        """loop condition"""
        if state["counter"] < state["max_iterations"]:
            return "continue"
        return "finish"

    # build loop graph
    graph = StateGraph(LoopState)
    graph.add_node("process", process)

    graph.add_edge(START, "process")

    graph.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",  # loop back
            "finish": END
        }
    )

    app = graph.compile()

    print("\n" + cleandoc("""
            Graph Structure: Loop with Exit
              START → process → should_continue?
                           ↑         │         │
                           └─continue┘         └─finish→ END
        """))

    # ASCII visualization
    try:
        ascii_diagram = app.get_graph().draw_ascii()
        print("\nASCII Visualization:")
        print(ascii_diagram)
    except Exception:
        print("\nASCII visualization not available")

    # Mermaid diagram
    try:
        mermaid = app.get_graph().draw_mermaid()
        print(cleandoc(            """

                Mermaid Diagram:
                mermaid
                💡 Notice the loop (back edge) in the diagram!
            """))
    except Exception as e:
        print(f"\nMermaid visualization error: {e}")

    print("\n✅ Key Takeaway: Loops appear as back edges in visualization")


# endregion


# region Demo 4: Multi-Agent Graph
def demo_multi_agent_graph() -> None:
    """
    visualize complex multi-agent graph

    Concept: Complex Graph Visualization
    ┌──────────────────────────────────────────────────────────────────┐
    │              Multi-Agent Graph Structure                         │
    │                                                                  │
    │  Complexity Patterns:                                            │
    │  • Multiple decision points                                      │
    │  • Multiple loops                                                │
    │  • Parallel branches                                             │
    │  • Nested routing logic                                          │
    │                                                                  │
    │  Visual Benefits:                                                │
    │  • See entire workflow at once                                   │
    │  • Identify bottlenecks (many edges converging)                  │
    │  • Spot unused nodes (no incoming edges)                         │
    │  • Verify all paths lead to END                                  │
    │                                                                  │
    │  Documentation:                                                  │
    │  • Include visualizations in design docs                         │
    │  • Show stakeholders agent workflows                             │
    │  • Use for code review discussions                               │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Multi-Agent Graph Visualization")

    class AgentState(TypedDict):
        """multi-agent state"""
        task: str
        specialist: str
        result: str

    def router(state: AgentState) -> Literal["research", "code", "analysis"]:
        """route to specialist"""
        task = state["task"].lower()
        if "research" in task:
            return "research"
        elif "code" in task:
            return "code"
        else:
            return "analysis"

    def research_agent(state: AgentState) -> dict:
        """research specialist"""
        return {"specialist": "research", "result": "Research done"}

    def code_agent(state: AgentState) -> dict:
        """code specialist"""
        return {"specialist": "code", "result": "Code written"}

    def analysis_agent(state: AgentState) -> dict:
        """analysis specialist"""
        return {"specialist": "analysis", "result": "Analysis complete"}

    # build multi-agent graph
    graph = StateGraph(AgentState)
    graph.add_node("research", research_agent)
    graph.add_node("code", code_agent)
    graph.add_node("analysis", analysis_agent)

    graph.add_conditional_edges(
        START,
        router,
        {
            "research": "research",
            "code": "code",
            "analysis": "analysis"
        }
    )

    graph.add_edge("research", END)
    graph.add_edge("code", END)
    graph.add_edge("analysis", END)

    app = graph.compile()

    print("\n" + cleandoc("""
            Graph Structure: Router + Specialists
                             ┌─→ research_agent
                             │
              START → router ─→ code_agent
                             │
                             └─→ analysis_agent → END
        """))

    # ASCII visualization
    try:
        ascii_diagram = app.get_graph().draw_ascii()
        print("\nASCII Visualization:")
        print(ascii_diagram)
    except Exception:
        print("\nASCII visualization not available")

    # Mermaid diagram
    try:
        mermaid = app.get_graph().draw_mermaid()
        print(cleandoc(            """

                Mermaid Diagram:
                mermaid
                💡 This diagram shows the complete agent routing logic!
            """))
    except Exception as e:
        print(f"\nMermaid visualization error: {e}")

    print("\n✅ Key Takeaway: Visualization essential for complex multi-agent systems")


# endregion


# region Demo 5: Practical Visualization Tips
def demo_visualization_tips() -> None:
    """
    practical tips for graph visualization

    Tips for Effective Visualization
    ┌──────────────────────────────────────────────────────────────────┐
    │                 Graph Design Best Practices                      │
    │                                                                  │
    │  1. Descriptive Node Names:                                      │
    │     ✅ "validate_user_input"                                      │
    │     ❌ "node1"                                                     │
    │                                                                  │
    │  2. Clear Routing Conditions:                                    │
    │     ✅ return "high_priority" or "low_priority"                   │
    │     ❌ return "a" or "b"                                          │
    │                                                                  │
    │  3. Logical Grouping:                                            │
    │     • Group related nodes together                               │
    │     • Keep decision points visible                               │
    │     • Minimize edge crossings                                    │
    │                                                                  │
    │  4. Documentation:                                               │
    │     • Add docstrings to nodes                                    │
    │     • Comment complex routing logic                              │
    │     • Include state schema in docs                               │
    │                                                                  │
    │  5. Validation:                                                  │
    │     • Visualize before running                                   │
    │     • Check for dead ends                                        │
    │     • Verify loop exit conditions                                │
    │     • Ensure all paths reach END                                 │
    │                                                                  │
    │  Tools:                                                          │
    │  • https://mermaid.live → Interactive Mermaid viewer             │
    │  • LangGraph Studio → Visual graph builder                       │
    │  • Graphviz → Publication-quality diagrams                       │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Visualization Best Practices")

    print("\n" + cleandoc("""
        Best Practices for LangGraph Visualization:

        1. DESCRIPTIVE NODE NAMES
           ✅ Good: "validate_input", "process_payment", "send_email"
           ❌ Bad: "node1", "step_a", "func"

           Why: Clear names make graphs self-documenting

        2. MEANINGFUL ROUTING STRINGS
           ✅ Good: return "approved" or "rejected"
           ❌ Bad: return "a" or "b"

           Why: Routing logic should be obvious in diagram

        3. MINIMIZE COMPLEXITY
           • Keep graphs under 10 nodes when possible
           • Break complex workflows into sub-graphs
           • Use clear conditional logic

        4. VERIFY BEFORE RUNNING
           • Every node should connect to something
           • All paths should eventually reach END
           • Loops should have exit conditions
           • No orphaned nodes (no incoming edges)

        5. DOCUMENTATION INTEGRATION
           • Include Mermaid diagrams in README
           • Show stakeholders visual workflows
           • Use for code review discussions
           • Version control graph diagrams

        COMMON VISUALIZATION ISSUES:

        Issue: "Graph looks messy"
        → Solution: Simplify routing, reduce nodes, use sub-graphs

        Issue: "Can't see routing logic"
        → Solution: Use descriptive routing strings, not "a"/"b"

        Issue: "Visualization doesn't match code"
        → Solution: Generate diagram from compiled graph, not manually

        TOOLS TO USE:

        • mermaid.live: Paste Mermaid code for interactive viewing
        • LangGraph Studio: Visual graph builder and debugger
        • Graphviz: Generate PDF/PNG for documentation
        • draw_ascii(): Quick terminal visualization

        EXAMPLE WORKFLOW:

        1. Design graph on paper or whiteboard
        2. Implement in code with descriptive names
        3. Generate visualization: app.get_graph().draw_mermaid()
        4. View in mermaid.live to verify structure
        5. Test execution and fix issues
        6. Include diagram in documentation
        7. Update diagram when graph changes
    """))

    print("\n✅ Key Takeaway: Good visualization = better understanding = fewer bugs")
    print("💡 Always visualize graphs before deploying to production!")


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Simple Linear", "simple linear", demo_simple_linear),
    Demo("2", "Conditional Routing", "conditional routing", demo_conditional_graph),
    Demo("3", "Loop Pattern", "loop pattern", demo_loop_graph),
    Demo("4", "Multi-Agent", "multi-agent", demo_multi_agent_graph),
    Demo("5", "Best Practices", "best practices", demo_visualization_tips),
]

# endregion

def main() -> None:
    """run all visualization demonstrations"""
    print(cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║               LANGGRAPH GRAPH VISUALIZATION                        ║
        ║                                                                    ║
        ║  Key Concepts:                                                     ║
        ║  • app.get_graph().draw_mermaid() → Mermaid diagram                ║
        ║  • app.get_graph().draw_ascii() → Terminal visualization           ║
        ║  • Visualize BEFORE running to catch errors                        ║
        ║  • Use descriptive names for self-documenting graphs               ║
        ║  • Include diagrams in documentation                               ║
        ║                                                                    ║
        ║  Benefits:                                                         ║
        ║  • Understand complex workflows instantly                          ║
        ║  • Debug routing logic visually                                    ║
        ║  • Communicate designs to stakeholders                             ║
        ║  • Verify correctness before execution                             ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝
    """))


    runner = MenuRunner(DEMOS, title="LangGraph Visualization")
    runner.run()


if __name__ == "__main__":
    main()


# endregion
