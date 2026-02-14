"""
LangGraph State Basics - StateGraph Fundamentals

This module teaches the core concepts of LangGraph's StateGraph:
- State definition with TypedDict
- Creating graphs with nodes and edges
- State immutability and updates
- Graph compilation and execution

No API key required - uses mock LLM for demonstrations.

Run with: uv run python -m phase7_frameworks.02_langgraph.state_basics
"""

from inspect import cleandoc
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section


# region Utility Functions

def print_state(state: dict, label: str = "State") -> None:
    """print state contents with nice formatting"""
    print(f"\n{label}:")
    for key, value in state.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"  {key}: [{len(value)} items]")
        else:
            print(f"  {key}: {value}")


# endregion


# region Demo 1: Simple State and Graph
def demo_simple_state_graph() -> None:
    """
    demonstrate basic StateGraph with simple state

    Concept: StateGraph Core Architecture
    ┌──────────────────────────────────────────────────────────────────┐
    │                    StateGraph Fundamentals                       │
    │                                                                  │
    │  State (TypedDict):                                              │
    │     class MyState(TypedDict):                                    │
    │         messages: list[str]                                      │
    │         counter: int                                             │
    │                                                                  │
    │  Nodes (Functions):                                              │
    │     def node_a(state: MyState) -> MyState:                       │
    │         # process state, return updates                          │
    │         return {"counter": state["counter"] + 1}                 │
    │                                                                  │
    │  Graph Structure:                                                │
    │     START → node_a → node_b → END                                │
    │                                                                  │
    │  Execution Flow:                                                 │
    │     1. Initial state created                                     │
    │     2. Each node receives state, returns updates                 │
    │     3. Updates merged into state (immutably)                     │
    │     4. Next node receives updated state                          │
    │     5. Repeat until END reached                                  │
    │                                                                  │
    │  Key Insight: Nodes don't modify state directly!                 │
    │  They return updates, graph merges them.                         │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Simple State and Graph")

    # step 1: define state structure
    class SimpleState(TypedDict):
        """state schema for simple counter graph"""
        counter: int
        message: str

    print("\n" + cleandoc("""
        1. State Schema Defined:
           class SimpleState(TypedDict):
               counter: int
               message: str
    """))

    # step 2: define node functions
    def increment_node(state: SimpleState) -> dict:
        """node that increments counter"""
        print(f"   → increment_node: counter={state['counter']} → {state['counter'] + 1}")
        return {"counter": state["counter"] + 1}

    def message_node(state: SimpleState) -> dict:
        """node that updates message"""
        msg = f"Count is now {state['counter']}"
        print(f"   → message_node: setting message='{msg}'")
        return {"message": msg}

    print("\n" + cleandoc("""
        2. Node Functions Defined:
           • increment_node: counter + 1
           • message_node: sets message based on counter
    """))

    # step 3: build graph
    graph = StateGraph(SimpleState)
    graph.add_node("increment", increment_node)
    graph.add_node("set_message", message_node)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", "set_message")
    graph.add_edge("set_message", END)

    print("\n" + cleandoc("""
        3. Graph Structure Built:
           START → increment → set_message → END
    """))

    # step 4: compile graph
    app = graph.compile()
    print("\n4. Graph Compiled (ready to run)")

    # step 5: run graph
    print("\n5. Graph Execution:")
    initial_state = {"counter": 0, "message": ""}
    print_state(initial_state, "Initial State")

    result = app.invoke(initial_state)
    print_state(result, "Final State")

    print("\n✅ Key Takeaway: Nodes return updates, graph merges them automatically")


# endregion


# region Demo 2: State Updates and Immutability
def demo_state_updates() -> None:
    """
    demonstrate how state updates work (immutability)

    Concept: State Update Mechanics
    ┌──────────────────────────────────────────────────────────────────┐
    │                    State Update Flow                             │
    │                                                                  │
    │  Initial State:          Node Returns:        Updated State:     │
    │  {"count": 0,            {"count": 1}         {"count": 1,       │
    │   "name": "Alice"}                             "name": "Alice"}  │
    │                                                                  │
    │  How Updates Work:                                               │
    │  1. Node receives COPY of state                                  │
    │  2. Node returns dictionary with UPDATES only                    │
    │  3. Graph MERGES updates into state (shallow merge)              │
    │  4. Original state unchanged (immutable)                         │
    │                                                                  │
    │  Update Rules:                                                   │
    │  • Return full values, not deltas                                │
    │  • Only return fields you want to change                         │
    │  • Missing fields = no change                                    │
    │  • None values = set to None (not delete)                        │
    │                                                                  │
    │  Example:                                                        │
    │     state = {"count": 5, "name": "Bob"}                          │
    │     update = {"count": 10}  # name unchanged                     │
    │     result = {"count": 10, "name": "Bob"}                        │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: State Updates and Immutability")

    class CounterState(TypedDict):
        """state with multiple fields"""
        count: int
        name: str
        status: str

    def update_count(state: CounterState) -> dict:
        """update only count field"""
        print(f"   Node 1: Updating count: {state['count']} → {state['count'] + 10}")
        return {"count": state["count"] + 10}

    def update_status(state: CounterState) -> dict:
        """update only status field"""
        print(f"   Node 2: Updating status → 'processed'")
        return {"status": "processed"}

    # build graph
    graph = StateGraph(CounterState)
    graph.add_node("update_count", update_count)
    graph.add_node("update_status", update_status)
    graph.add_edge(START, "update_count")
    graph.add_edge("update_count", "update_status")
    graph.add_edge("update_status", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph: START → update_count → update_status → END

        Execution:
    """))

    initial = {"count": 0, "name": "Alice", "status": "pending"}
    print_state(initial, "Initial")

    result = app.invoke(initial)
    print_state(result, "Final")

    print("\n✅ Key Takeaway: Each node returns partial updates, name field unchanged")


# endregion


# region Demo 3: Message State Pattern
def demo_message_state() -> None:
    """
    demonstrate message accumulation pattern with add_messages

    Concept: Message State Pattern (LangGraph Convention)
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Message Accumulation Pattern                    │
    │                                                                  │
    │  Problem: Chat history needs to accumulate messages              │
    │                                                                  │
    │  Wrong Approach (Replace):                                       │
    │     Node 1 returns: {"messages": ["Hello"]}                      │
    │     Node 2 returns: {"messages": ["How are you?"]}               │
    │     Result: {"messages": ["How are you?"]}  ❌ Lost "Hello"      │
    │                                                                  │
    │  Right Approach (Accumulate):                                    │
    │     from langgraph.graph.message import add_messages             │
    │                                                                  │
    │     class State(TypedDict):                                      │
    │         messages: Annotated[list, add_messages]                  │
    │                                                                  │
    │     Node 1 returns: {"messages": ["Hello"]}                      │
    │     Node 2 returns: {"messages": ["How are you?"]}               │
    │     Result: {"messages": ["Hello", "How are you?"]} ✅           │
    │                                                                  │
    │  add_messages Function:                                          │
    │  • Appends new messages to existing list                         │
    │  • Handles message deduplication                                 │
    │  • Standard pattern for chat/agent workflows                     │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Message State Pattern (add_messages)")

    # define state with message accumulation
    class ChatState(TypedDict):
        """state with accumulating messages"""
        messages: Annotated[list[str], add_messages]
        turn: int

    def user_turn(state: ChatState) -> dict:
        """simulate user message"""
        msg = f"User message {state['turn']}"
        print(f"   User: {msg}")
        return {"messages": [msg], "turn": state["turn"] + 1}

    def assistant_turn(state: ChatState) -> dict:
        """simulate assistant response"""
        msg = f"Assistant reply {state['turn']}"
        print(f"   Assistant: {msg}")
        return {"messages": [msg]}

    # build graph
    graph = StateGraph(ChatState)
    graph.add_node("user", user_turn)
    graph.add_node("assistant", assistant_turn)
    graph.add_edge(START, "user")
    graph.add_edge("user", "assistant")
    graph.add_edge("assistant", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph: START → user → assistant → END

        Execution (Watch messages accumulate):
    """))

    initial = {"messages": [], "turn": 1}
    result = app.invoke(initial)

    print("\n" + cleandoc(f"""
        Final State:
          messages: {result['messages']}
          turn: {result['turn']}
    """))

    print("\n✅ Key Takeaway: Annotated[list, add_messages] accumulates, not replaces")


# endregion


# region Demo 4: Multi-Path Graph
def demo_multi_path_graph() -> None:
    """
    demonstrate graph with sequential multi-node processing

    Concept: Graph Topology and Sequential Execution
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Sequential Graph Structure                      │
    │                                                                  │
    │  Linear Graph (Sequential):                                      │
    │     START → A → B → C → END                                      │
    │                                                                  │
    │  State flows through each node in order                          │
    │  Each node processes and updates state                           │
    │  Final state is cumulative result of all nodes                   │
    │                                                                  │
    │  Note: For branching/conditional, see conditional_routing.py     │
    │  • Conditional edges enable if/else routing                      │
    │  • Fan-out requires reducer functions (e.g., operator.add)       │
    │  • This demo shows simple sequential processing                  │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Sequential Multi-Node Graph")

    class ProcessState(TypedDict):
        """state for processing pipeline"""
        value: int
        path_taken: list[str]

    def node_a(state: ProcessState) -> dict:
        """step 1: multiply"""
        print("   → Node A: Multiply by 2")
        return {
            "value": state["value"] * 2,
            "path_taken": state["path_taken"] + ["A"]
        }

    def node_b(state: ProcessState) -> dict:
        """step 2: add"""
        print("   → Node B: Add 10")
        return {
            "value": state["value"] + 10,
            "path_taken": state["path_taken"] + ["B"]
        }

    def node_c(state: ProcessState) -> dict:
        """step 3: subtract"""
        print("   → Node C: Subtract 3")
        return {
            "value": state["value"] - 3,
            "path_taken": state["path_taken"] + ["C"]
        }

    def node_d(state: ProcessState) -> dict:
        """step 4: final processing"""
        print("   → Node D: Add 5")
        return {
            "value": state["value"] + 5,
            "path_taken": state["path_taken"] + ["D"]
        }

    # build sequential graph
    graph = StateGraph(ProcessState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("c", node_c)
    graph.add_node("d", node_d)

    # sequential structure: A → B → C → D
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_edge("d", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure (Sequential):
          START → A → B → C → D → END
    """))

    print("\nExecution (value=5):")
    result = app.invoke({"value": 5, "path_taken": []})
    print("\n" + cleandoc(f"""
          Final value: {result['value']}
          Path: {' → '.join(result['path_taken'])}
          Calculation: (5 * 2) + 10 - 3 + 5 = {result['value']}
    """))

    print("\n" + cleandoc("""
        ✅ Key Takeaway: Sequential nodes process state in order
        📝 Note: For branching/conditional paths, see conditional_routing.py
    """))


# endregion


# region Demo 5: Stateful Counter with Loops
def demo_stateful_counter() -> None:
    """
    demonstrate stateful operations across multiple invocations

    Concept: State Persistence and Multi-Invocation
    ┌──────────────────────────────────────────────────────────────────┐
    │                State Across Multiple Runs                        │
    │                                                                  │
    │  Problem: How to maintain state across graph executions?         │
    │                                                                  │
    │  Solution: Pass previous result as input to next run             │
    │                                                                  │
    │  Example:                                                        │
    │     state = {"count": 0}                                         │
    │                                                                  │
    │     # run 1                                                      │
    │     result1 = app.invoke(state)        # {"count": 1}            │
    │                                                                  │
    │     # run 2 (reuse previous state)                               │
    │     result2 = app.invoke(result1)      # {"count": 2}            │
    │                                                                  │
    │     # run 3                                                      │
    │     result3 = app.invoke(result2)      # {"count": 3}            │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Iterative refinement (generate → review → improve)            │
    │  • Multi-turn conversations (preserve history)                   │
    │  • Workflow with human approval (pause → resume)                 │
    │  • Batch processing (process items one by one)                   │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Stateful Counter (Multiple Invocations)")

    class CountState(TypedDict):
        """state with counter"""
        count: int
        history: list[int]

    def increment(state: CountState) -> dict:
        """increment counter and record in history"""
        new_count = state["count"] + 1
        print(f"   Incrementing: {state['count']} → {new_count}")
        return {
            "count": new_count,
            "history": state["history"] + [new_count]
        }

    # build graph
    graph = StateGraph(CountState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph: START → increment → END

        Multiple Invocations (State Carries Over):
    """))

    # run 1
    state = {"count": 0, "history": []}
    print("\nRun 1:")
    state = app.invoke(state)
    print(f"  count={state['count']}, history={state['history']}")

    # run 2 (reuse previous state)
    print("\nRun 2 (reusing previous state):")
    state = app.invoke(state)
    print(f"  count={state['count']}, history={state['history']}")

    # run 3
    print("\nRun 3:")
    state = app.invoke(state)
    print(f"  count={state['count']}, history={state['history']}")

    print("\n✅ Key Takeaway: Pass result as input to next run to maintain state")


# endregion


# region Demo 6: Complete Example - Simple Agent Loop
def demo_simple_agent_loop() -> None:
    """
    demonstrate complete example: simple agent with state

    Concept: Agent State Pattern
    ┌──────────────────────────────────────────────────────────────────┐
    │                     Agent State Pattern                          │
    │                                                                  │
    │  Agent State Typically Contains:                                 │
    │  • messages: Annotated[list, add_messages]  # conversation       │
    │  • iterations: int                           # loop counter      │
    │  • next_action: str                          # routing decision  │
    │                                                                  │
    │  Agent Loop Pattern:                                             │
    │     START → think → act → observe → (loop or end)                │
    │                                                                  │
    │  State Flow:                                                     │
    │     1. think: add thought to messages                            │
    │     2. act: add action to messages                               │
    │     3. observe: add observation to messages                      │
    │     4. decide: continue loop or finish                           │
    │                                                                  │
    │  This is the foundation for ReAct agents (Phase 4 revisited)     │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 6: Simple Agent Loop (Foundation for ReAct)")

    class AgentState(TypedDict):
        """agent state with messages and metadata"""
        messages: Annotated[list[str], add_messages]
        iterations: int
        max_iterations: int

    def think(state: AgentState) -> dict:
        """agent thinking step"""
        thought = f"Thought {state['iterations']}: Analyzing the situation..."
        print(f"   🤔 {thought}")
        return {"messages": [thought]}

    def act(state: AgentState) -> dict:
        """agent action step"""
        action = f"Action {state['iterations']}: Taking action..."
        print(f"   🎯 {action}")
        return {
            "messages": [action],
            "iterations": state["iterations"] + 1
        }

    # build graph
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_edge(START, "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Agent Loop: START → think → act → END

        Execution:
    """))

    initial = {
        "messages": ["User: Solve the problem"],
        "iterations": 1,
        "max_iterations": 3
    }

    result = app.invoke(initial)

    print("\n" + cleandoc(f"""
        Final State:
          iterations: {result['iterations']}
          message count: {len(result['messages'])}
    """))
    print("\nAll messages:")
    for msg in result["messages"]:
        print(f"  • {msg}")

    print("\n" + cleandoc("""
        ✅ Key Takeaway: Agent pattern = state + loop + messages
        📝 Next: Add conditional routing to make agent decide when to stop!
    """))


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Simple State and Graph", "simple state and graph", demo_simple_state_graph),
    Demo("2", "State Updates", "state updates", demo_state_updates),
    Demo("3", "Message State Pattern", "message state pattern", demo_message_state),
    Demo("4", "Sequential Multi-Node Graph", "sequential multi-node graph", demo_multi_path_graph),
    Demo("5", "Stateful Counter", "stateful counter", demo_stateful_counter),
    Demo("6", "Simple Agent Loop", "simple agent loop", demo_simple_agent_loop),
]

# endregion

def main() -> None:
    """run all state basics demonstrations"""
    print(cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║                   LANGGRAPH STATE BASICS                           ║
        ║                                                                    ║
        ║  Key Concepts:                                                     ║
        ║  • StateGraph = nodes + edges + state schema                       ║
        ║  • Nodes return updates (dicts), not full state                    ║
        ║  • Graph merges updates automatically (shallow merge)              ║
        ║  • add_messages accumulates instead of replacing                   ║
        ║  • State flows through graph, gets updated at each node            ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝
    """))

    
    runner = MenuRunner(DEMOS, title="LangGraph State Basics")
    runner.run()


if __name__ == "__main__":
    main()


# endregion
