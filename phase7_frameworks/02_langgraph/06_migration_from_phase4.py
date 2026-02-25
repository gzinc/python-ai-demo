"""
Migration Guide: Phase 4 → LangGraph

This module shows how to convert Phase 4 imperative multi-agent code
to LangGraph's declarative graph-based approach.

We'll convert these Phase 4 patterns:
1. MultiAgentOrchestrator → Router Pattern
2. AgentExecutor while loop → ReAct Loop Pattern
3. Sequential agents → Sequential Handoff Pattern
4. Supervisor coordination → Supervisor Pattern

No API key required - focuses on patterns and structure.

Run with: uv run python -m phase7_frameworks.02_langgraph.migration_from_phase4
"""

from inspect import cleandoc
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section

# region Utility Functions
# endregion


# region Pattern 1: MultiAgentOrchestrator → Router Pattern
def pattern_1_router_migration() -> None:
    """
    migrate Phase 4 MultiAgentOrchestrator to LangGraph router pattern

    Migration: Imperative Routing → Declarative Graph
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Phase 4 → LangGraph                             │
    │                                                                  │
    │  Phase 4 (Imperative):                                           │
    │     class MultiAgentOrchestrator:                                │
    │         def route_query(self, query: str):                       │
    │             if "research" in query:                              │
    │                 return self.research_agent.execute(query)        │
    │             elif "code" in query:                                │
    │                 return self.code_agent.execute(query)            │
    │             else:                                                │
    │                 return self.general_agent.execute(query)         │
    │                                                                  │
    │  LangGraph (Declarative):                                        │
    │     def router(state):                                           │
    │         if "research" in state["query"]:                         │
    │             return "research"                                    │
    │         elif "code" in state["query"]:                           │
    │             return "code"                                        │
    │         else:                                                    │
    │             return "general"                                     │
    │                                                                  │
    │     graph.add_conditional_edges(START, router, {...})            │
    │                                                                  │
    │  Benefits:                                                       │
    │  • Visual graph representation                                   │
    │  • Built-in state management                                     │
    │  • Easy to add/remove specialists                                │
    │  • Automatic checkpointing                                       │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Pattern 1: MultiAgentOrchestrator → Router")

    print("\n" + cleandoc("""
        Phase 4 Code (Imperative):

        class MultiAgentOrchestrator:
            def __init__(self):
                self.research_agent = ResearchAgent()
                self.code_agent = CodeAgent()
                self.general_agent = GeneralAgent()

            def route_query(self, query: str) -> str:
                # imperative if/else routing
                if "research" in query.lower():
                    return self.research_agent.execute(query)
                elif "code" in query.lower():
                    return self.code_agent.execute(query)
                else:
                    return self.general_agent.execute(query)

        # usage
        orchestrator = MultiAgentOrchestrator()
        result = orchestrator.route_query("do research on AI")
    """))

    print("\n" + "─" * 70)
    print("LangGraph Code (Declarative):")
    print("─" * 70)

    class RouterState(TypedDict):
        """state schema"""
        query: str
        specialist: str
        result: str

    def router(state: RouterState) -> Literal["research", "code", "general"]:
        """declarative routing logic"""
        query = state["query"].lower()
        if "research" in query:
            print(f"   Router: '{state['query']}' → research specialist")
            return "research"
        elif "code" in query:
            print(f"   Router: '{state['query']}' → code specialist")
            return "code"
        else:
            print(f"   Router: '{state['query']}' → general specialist")
            return "general"

    def research_agent(state: RouterState) -> dict:
        """research specialist"""
        print("   🔍 Research Agent: Searching...")
        return {"specialist": "research", "result": "Research complete"}

    def code_agent(state: RouterState) -> dict:
        """code specialist"""
        print("   💻 Code Agent: Generating code...")
        return {"specialist": "code", "result": "Code generated"}

    def general_agent(state: RouterState) -> dict:
        """general specialist"""
        print("   🤖 General Agent: Processing...")
        return {"specialist": "general", "result": "Query answered"}

    # build graph
    graph = StateGraph(RouterState)
    graph.add_node("research", research_agent)
    graph.add_node("code", code_agent)
    graph.add_node("general", general_agent)

    graph.add_conditional_edges(
        START,
        router,
        {
            "research": "research",
            "code": "code",
            "general": "general"
        }
    )

    graph.add_edge("research", END)
    graph.add_edge("code", END)
    graph.add_edge("general", END)

    app = graph.compile()

    print("\n" + cleandoc("""
            # LangGraph implementation
            # (code shown above)
            # usage
            app = graph.compile()
        """))

    print("\nExecution:")
    result = app.invoke({"query": "do research on AI", "specialist": "", "result": ""})
    print(f"  Result: {result['result']} (by {result['specialist']})")

    print("\n" + cleandoc("""
            ✅ Benefits of LangGraph:
              • Declarative graph structure (easier to understand)
              • Visual representation with draw_mermaid()
              • Built-in state management
              • Easy to add new specialists
              • Automatic checkpointing available
        """))


# endregion


# region Pattern 2: Agent Loop → ReAct Pattern
def pattern_2_agent_loop_migration() -> None:
    """
    migrate Phase 4 agent execution loop to LangGraph ReAct pattern

    Migration: While Loop → Conditional Edges
    ┌──────────────────────────────────────────────────────────────────┐
    │             Phase 4 Loop → LangGraph Graph                       │
    │                                                                  │
    │  Phase 4 (Imperative):                                           │
    │     def execute_agent(query):                                    │
    │         iterations = 0                                           │
    │         while iterations < max_iterations:                       │
    │             thought = agent.think(state)                         │
    │             action = agent.act(state)                            │
    │             if task_complete:                                    │
    │                 break                                            │
    │             iterations += 1                                      │
    │         return result                                            │
    │                                                                  │
    │  LangGraph (Declarative):                                        │
    │     graph.add_edge("think", "act")                               │
    │     graph.add_conditional_edges(                                 │
    │         "act",                                                   │
    │         should_continue,  # router function                      │
    │         {"continue": "think", "finish": END}                     │
    │     )                                                            │
    │                                                                  │
    │  Benefits:                                                       │
    │  • No manual iteration management                                │
    │  • Visual loop structure                                         │
    │  • Can interrupt/resume easily                                   │
    │  • Automatic state tracking                                      │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Pattern 2: Agent Loop → ReAct Pattern")

    print("\n" + cleandoc("""
        Phase 4 Code (Imperative Loop):

        def execute_agent(query: str, max_iterations: int = 5) -> str:
            state = {"query": query, "complete": False}
            iterations = 0

            # imperative while loop
            while iterations < max_iterations:
                # think step
                thought = agent.think(state)
                print(f"Thought {iterations + 1}: {thought}")

                # act step
                action = agent.act(state)
                print(f"Action {iterations + 1}: {action}")

                # check completion
                if state["complete"]:
                    break

                iterations += 1

            return state["result"]

        # usage
        result = execute_agent("solve problem")
    """))

    print("\n" + "─" * 70)
    print("LangGraph Code (Declarative Graph):")
    print("─" * 70)

    class AgentState(TypedDict):
        """agent state"""
        messages: Annotated[list[str], add_messages]
        iterations: int
        max_iterations: int
        task_complete: bool

    def think(state: AgentState) -> dict:
        """thinking step"""
        thought = f"Thought {state['iterations'] + 1}: Analyzing..."
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
        """action step"""
        action = f"Action {state['iterations'] + 1}: Executing..."
        print(f"   🎯 {action}")

        return {
            "messages": [action],
            "iterations": state["iterations"] + 1
        }

    def should_continue(state: AgentState) -> Literal["continue", "finish"]:
        """loop condition (declarative)"""
        if state["task_complete"]:
            print("   Decision: Task complete → finish")
            return "finish"
        elif state["iterations"] >= state["max_iterations"]:
            print("   Decision: Max iterations → finish")
            return "finish"
        else:
            print("   Decision: Continue → loop")
            return "continue"

    # build ReAct graph
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)

    graph.add_edge(START, "think")
    graph.add_edge("think", "act")

    # conditional loop
    graph.add_conditional_edges(
        "act",
        should_continue,
        {
            "continue": "think",  # loop back
            "finish": END
        }
    )

    app = graph.compile()

    print("\n" + cleandoc("""
            # LangGraph implementation
            # (code shown above)
            # usage
            app = graph.compile()
        """))

    print("\nExecution:")
    result = app.invoke({
        "messages": ["User: Solve this problem"],
        "iterations": 0,
        "max_iterations": 5,
        "task_complete": False
    })

    print(f"\n  Final: {result['iterations']} iterations, complete={result['task_complete']}")

    print("\n" + cleandoc("""
            ✅ Benefits of LangGraph:
              • No manual loop management
              • Visual loop structure in graph
              • Can interrupt/resume execution
              • Checkpoint between iterations
              • Easier to debug and understand
        """))


# endregion


# region Pattern 3: Sequential Agents → Sequential Handoff
def pattern_3_sequential_migration() -> None:
    """
    migrate Phase 4 sequential agent execution to LangGraph handoff pattern

    Migration: Sequential Calls → Sequential Edges
    ┌──────────────────────────────────────────────────────────────────┐
    │          Phase 4 Sequential → LangGraph Chain                    │
    │                                                                  │
    │  Phase 4 (Imperative):                                           │
    │     def process_pipeline(data):                                  │
    │         # step 1                                                 │
    │         result1 = agent1.process(data)                           │
    │         # step 2                                                 │
    │         result2 = agent2.process(result1)                        │
    │         # step 3                                                 │
    │         result3 = agent3.process(result2)                        │
    │         return result3                                           │
    │                                                                  │
    │  LangGraph (Declarative):                                        │
    │     graph.add_edge(START, "agent1")                              │
    │     graph.add_edge("agent1", "agent2")                           │
    │     graph.add_edge("agent2", "agent3")                           │
    │     graph.add_edge("agent3", END)                                │
    │                                                                  │
    │  Benefits:                                                       │
    │  • Clear pipeline visualization                                  │
    │  • Easy to add/remove/reorder steps                              │
    │  • Automatic state passing                                       │
    │  • Can checkpoint between steps                                  │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Pattern 3: Sequential Agents → Sequential Handoff")

    print("\n" + cleandoc("""\n        Phase 4 Code (Imperative Sequential):")
        def process_content_pipeline(topic: str) -> str:
            # step 1: outline
            outline = outliner_agent.create_outline(topic)
            print(f"Step 1: Created outline")

            # step 2: draft
            draft = writer_agent.write_draft(outline)
            print(f"Step 2: Wrote draft")

            # step 3: edit
            final = editor_agent.edit(draft)
            print(f"Step 3: Edited and finalized")

            return final

        # usage
        result = process_content_pipeline("AI Agents")
    """))

    print("\n" + cleandoc("""
            " + "─" * 7
            LangGraph Code (Declarative Chain):
            "─" * 70
        """))

    class ContentState(TypedDict):
        """content pipeline state"""
        topic: str
        outline: str
        draft: str
        final: str

    def outliner(state: ContentState) -> dict:
        """step 1: create outline"""
        outline = f"Outline for '{state['topic']}':\\n1. Intro\\n2. Body\\n3. Conclusion"
        print("   📝 Step 1: Created outline")
        return {"outline": outline}

    def writer(state: ContentState) -> dict:
        """step 2: write draft"""
        draft = f"Draft: {state['topic']}... (based on outline)"
        print("   ✍️  Step 2: Wrote draft")
        return {"draft": draft}

    def editor(state: ContentState) -> dict:
        """step 3: edit and finalize"""
        final = f"Final: Polished article about {state['topic']}"
        print("   ✨ Step 3: Edited and finalized")
        return {"final": final}

    # build sequential chain
    graph = StateGraph(ContentState)
    graph.add_node("outliner", outliner)
    graph.add_node("writer", writer)
    graph.add_node("editor", editor)

    # sequential edges
    graph.add_edge(START, "outliner")
    graph.add_edge("outliner", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    app = graph.compile()

    print("\n" + cleandoc("""
            # LangGraph implementation
            # (code shown above)
            # usage
            app = graph.compile()
        """))

    print("\nExecution:")
    result = app.invoke({
        "topic": "AI Agents",
        "outline": "",
        "draft": "",
        "final": ""
    })

    print(f"\n  Final output: {result['final']}")

    print("\n" + cleandoc("""
            ✅ Benefits of LangGraph:
              • Clear pipeline visualization
              • Easy to reorder/add/remove steps
              • Automatic state threading
              • Can interrupt between steps for review
              • Visual workflow documentation
        """))


# endregion


# region Pattern 4: Supervisor → Supervisor Pattern
def pattern_4_supervisor_migration() -> None:
    """
    migrate Phase 4 supervisor coordination to LangGraph supervisor pattern

    Migration: Coordinator Class → Supervisor Graph
    ┌──────────────────────────────────────────────────────────────────┐
    │         Phase 4 Supervisor → LangGraph Supervisor                │
    │                                                                  │
    │  Phase 4 (Imperative):                                           │
    │     class AgentSupervisor:                                       │
    │         def coordinate(self, task):                              │
    │             workers_done = []                                    │
    │             while not task_complete:                             │
    │                 # decide next worker                             │
    │                 worker = self.select_worker(workers_done)        │
    │                 # execute worker                                 │
    │                 result = worker.execute(task)                    │
    │                 workers_done.append(worker)                      │
    │             return result                                        │
    │                                                                  │
    │  LangGraph (Declarative):                                        │
    │     graph.add_edge(START, "supervisor")                          │
    │     graph.add_conditional_edges(                                 │
    │         "supervisor",                                            │
    │         route_decision,                                          │
    │         {"worker1": "worker1", "finish": END}                    │
    │     )                                                            │
    │     graph.add_edge("worker1", "supervisor")  # loop              │
    │                                                                  │
    │  Benefits:                                                       │
    │  • Visual coordination flow                                      │
    │  • Easy to add workers                                           │
    │  • Automatic state management                                    │
    │  • Can interrupt for approval                                    │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Pattern 4: Supervisor → Supervisor Pattern")

    print("\n" + cleandoc("""
        Phase 4 Code (Imperative Supervisor):

        class AgentSupervisor:
            def __init__(self):
                self.workers = {
                    "collect": DataCollector(),
                    "analyze": DataAnalyzer()
                }

            def coordinate(self, task: str) -> str:
                workers_done = []
                task_complete = False

                # imperative coordination loop
                while not task_complete:
                    # decide next worker
                    if "collect" not in workers_done:
                        worker = self.workers["collect"]
                        result = worker.execute(task)
                        workers_done.append("collect")
                    elif "analyze" not in workers_done:
                        worker = self.workers["analyze"]
                        result = worker.execute(task)
                        workers_done.append("analyze")
                        task_complete = True

                return result

        # usage
        supervisor = AgentSupervisor()
        result = supervisor.coordinate("analyze data")
    """))

    print("\n" + cleandoc("""
            " + "─" * 7
            LangGraph Code (Declarative Supervisor):
            "─" * 70
        """))

    class SupervisorState(TypedDict):
        """supervisor state"""
        task: str
        workers_done: list[str]
        task_complete: bool
        next_worker: Literal["collect", "analyze", "finish"]

    def supervisor(state: SupervisorState) -> dict:
        """supervisor coordinates workers"""
        done = state["workers_done"]
        print(f"   👔 Supervisor: Workers done = {done}")

        if state["task_complete"]:
            print("   → Decision: All work complete")
            return {"next_worker": "finish"}
        elif "collect" not in done:
            print("   → Decision: Assign to Data Collector")
            return {"next_worker": "collect"}
        elif "analyze" not in done:
            print("   → Decision: Assign to Data Analyzer")
            return {"next_worker": "analyze"}
        else:
            print("   → Decision: Task complete")
            return {"next_worker": "finish"}

    def route_decision(state: SupervisorState) -> Literal["collect", "analyze", "finish"]:
        """read supervisor's decision"""
        return state["next_worker"]

    def data_collector(state: SupervisorState) -> dict:
        """worker: collect data"""
        print("   📊 Data Collector: Collecting...")
        return {"workers_done": state["workers_done"] + ["collect"]}

    def data_analyzer(state: SupervisorState) -> dict:
        """worker: analyze data"""
        print("   🔍 Data Analyzer: Analyzing...")
        return {
            "workers_done": state["workers_done"] + ["analyze"],
            "task_complete": True
        }

    # build supervisor graph
    checkpointer = MemorySaver()
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("collect", data_collector)
    graph.add_node("analyze", data_analyzer)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "collect": "collect",
            "analyze": "analyze",
            "finish": END
        }
    )

    # workers report back to supervisor
    graph.add_edge("collect", "supervisor")
    graph.add_edge("analyze", "supervisor")

    app = graph.compile(checkpointer=checkpointer)

    print("\n" + cleandoc("""
            # LangGraph implementation
            # (code shown above)
            # usage
            app = graph.compile(checkpointer=checkpointer)
        """))

    print("\nExecution:")
    config = {"configurable": {"thread_id": "supervisor_session"}}
    result = app.invoke({
        "task": "analyze customer data",
        "workers_done": [],
        "task_complete": False,
        "next_worker": "collect"
    }, config)

    print(f"\n  Workers completed: {result['workers_done']}")
    print(f"  Task complete: {result['task_complete']}")

    print("\n" + cleandoc("""
            ✅ Benefits of LangGraph:
              • Visual coordination flow
              • Easy to add/remove workers
              • Automatic state tracking
              • Can interrupt for approval
              • Checkpointing built-in
        """))


# endregion


# region Migration Summary
def migration_summary() -> None:
    """summary of migration benefits and patterns"""
    print_section("Migration Summary: Why LangGraph?")

    print("\n" + cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║                  PHASE 4 → LANGGRAPH BENEFITS                      ║
        ║                                                                    ║
        ║  1. DECLARATIVE vs IMPERATIVE                                      ║
        ║     Phase 4: if/else, while loops, manual state                    ║
        ║     LangGraph: Graphs, edges, automatic state                      ║
        ║                                                                    ║
        ║  2. VISUAL REPRESENTATION                                          ║
        ║     Phase 4: Code is the only documentation                        ║
        ║     LangGraph: draw_mermaid() shows flow instantly                 ║
        ║                                                                    ║
        ║  3. STATE MANAGEMENT                                               ║
        ║     Phase 4: Manual state passing and updates                      ║
        ║     LangGraph: Automatic state threading and merging               ║
        ║                                                                    ║
        ║  4. CHECKPOINTING                                                  ║
        ║     Phase 4: Custom persistence layer required                     ║
        ║     LangGraph: Built-in checkpointing with MemorySaver             ║
        ║                                                                    ║
        ║  5. INTERRUPTS                                                     ║
        ║     Phase 4: Complex pause/resume logic                            ║
        ║     LangGraph: interrupt_before/after built-in                     ║
        ║                                                                    ║
        ║  6. DEBUGGING                                                      ║
        ║     Phase 4: Print statements and debugger                         ║
        ║     LangGraph: Visual flow + state inspection                      ║
        ║                                                                    ║
        ║  7. MAINTAINABILITY                                                ║
        ║     Phase 4: Complex class hierarchies                             ║
        ║     LangGraph: Simple nodes + edges                                ║
        ║                                                                    ║
        ║  8. COLLABORATION                                                  ║
        ║     Phase 4: Need to read code to understand                       ║
        ║     LangGraph: Share Mermaid diagram                               ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝

        MIGRATION PATTERNS SUMMARY:

        ┌──────────────────────────────────────────────────────────────────┐
        │ Phase 4 Pattern         │ LangGraph Pattern                      │
        ├──────────────────────────────────────────────────────────────────┤
        │ MultiAgentOrchestrator  │ Router + Conditional Edges             │
        │ Agent while loop        │ ReAct Loop + Conditional Edges         │
        │ Sequential execution    │ Sequential Edges                       │
        │ Supervisor coordination │ Supervisor + Worker Nodes              │
        │ Manual state passing    │ Automatic state threading              │
        │ Custom checkpointing    │ Built-in MemorySaver                   │
        │ Pause/resume logic      │ interrupt_before/after                 │
        └──────────────────────────────────────────────────────────────────┘

        STEP-BY-STEP MIGRATION GUIDE:

        1. IDENTIFY PATTERN
           → Look at your Phase 4 code structure
           → Match to one of the 4 patterns above

        2. DEFINE STATE SCHEMA
           → Convert class attributes to TypedDict
           → Identify fields that need accumulation (use add_messages)

        3. CONVERT TO NODES
           → Each agent method → Node function
           → Each node returns dict of state updates

        4. ADD EDGES
           → Sequential flow → add_edge()
           → Conditional flow → add_conditional_edges()
           → Loops → conditional edge back to same node

        5. COMPILE AND TEST
           → app = graph.compile()
           → Test with same inputs as Phase 4
           → Verify same outputs

        6. ADD PRODUCTION FEATURES
           → Add checkpointer for persistence
           → Add interrupts for approvals
           → Visualize with draw_mermaid()

        7. OPTIMIZE
           → Add error handling nodes
           → Add logging and monitoring
           → Add retry logic with conditional edges

        COMMON MIGRATION CHALLENGES:

        Challenge: "My Phase 4 code has complex state"
        Solution: Use TypedDict with all fields, use add_messages for lists

        Challenge: "Need to pause for approval"
        Solution: Use interrupt_before/after, much easier than Phase 4

        Challenge: "Agent loop is complicated"
        Solution: Use conditional_edges with loop back, cleaner than while

        Challenge: "Multiple agents coordinate"
        Solution: Supervisor pattern or router pattern

        WHEN TO MIGRATE:

        ✅ Migrate When:
        • Adding new features to Phase 4 agents
        • Need visual workflow representation
        • Need human-in-the-loop approvals
        • Want easier testing and debugging
        • Scaling to more agents

        ⏳ Wait When:
        • Phase 4 code is working and won't change
        • No new requirements coming
        • Team unfamiliar with LangGraph

        LEARNING PATH:

        1. ✅ Complete all LangGraph demos (this module)
        2. ✅ Pick ONE Phase 4 agent to migrate
        3. ✅ Start with simplest pattern (sequential or router)
        4. ✅ Test thoroughly against Phase 4 outputs
        5. ✅ Add production features (checkpointing, interrupts)
        6. ✅ Visualize and document
        7. ✅ Migrate remaining agents

        RESOURCES:

        • LangGraph Docs: https://python.langchain.com/docs/langgraph
        • This Module: Complete examples in previous files
        • Mermaid Live: https://mermaid.live
        • Phase 4 Code: /phase4_ai_agents/
    """))

    print("\n✅ You now have all the tools to migrate Phase 4 → LangGraph!")
    print("🚀 Start with your simplest agent and work your way up!")


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Router Pattern Migration", "router pattern migration", pattern_1_router_migration),
    Demo("2", "Agent Loop Migration", "agent loop migration", pattern_2_agent_loop_migration),
    Demo("3", "Sequential Migration", "sequential migration", pattern_3_sequential_migration),
    Demo("4", "Supervisor Migration", "supervisor migration", pattern_4_supervisor_migration),
    Demo("5", "Migration Summary", "migration summary", migration_summary),
]

# endregion

def main() -> None:
    """run all migration demonstrations"""
    print("\n" + cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║          MIGRATION GUIDE: PHASE 4 → LANGGRAPH                      ║
        ║                                                                    ║
        ║  Converting Your Phase 4 Agents:                                   ║
        ║  1. MultiAgentOrchestrator → Router Pattern                        ║
        ║  2. Agent Loop              → ReAct Pattern                        ║
        ║  3. Sequential Agents       → Sequential Handoff                   ║
        ║  4. Supervisor              → Supervisor Pattern                   ║
        ║                                                                    ║
        ║  What You'll Learn:                                                ║
        ║  • Side-by-side Phase 4 vs LangGraph code                          ║
        ║  • Benefits of declarative graphs                                  ║
        ║  • Step-by-step migration process                                  ║
        ║  • Common patterns and their conversions                           ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝
    """))


    runner = MenuRunner(DEMOS, title="LangGraph Migration from Phase 4")
    runner.run()


if __name__ == "__main__":
    main()


# endregion
