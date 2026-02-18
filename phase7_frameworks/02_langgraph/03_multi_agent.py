"""
LangGraph Multi-Agent Collaboration - Specialist Agent Patterns

This module teaches building multi-agent systems with specialized agents:
- Router pattern (delegate to specialists)
- Supervisor pattern (central coordinator)
- Sequential handoff (agent chain)
- Parallel specialists (concurrent execution)
- Aggregation pattern (combine results)

No API key required - uses mock logic for demonstrations.

Run with: uv run python -m phase7_frameworks.02_langgraph.multi_agent
"""

from inspect import cleandoc
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section


# region Utility Functions
# endregion


# region Demo 1: Router Pattern (Delegate to Specialists)
def demo_router_pattern() -> None:
    """
    demonstrate router pattern with specialist delegation

    Concept: Router + Specialist Pattern
    ┌──────────────────────────────────────────────────────────────────┐
    │                Router + Specialist Delegation                    │
    │                                                                  │
    │  Pattern:                                                        │
    │               ┌─→ research_agent (research queries)              │
    │               │                                                  │
    │     START → router ─→ math_agent (math queries)                  │
    │               │                                                  │
    │               └─→ code_agent (coding queries)                    │
    │                                                                  │
    │  Router Logic:                                                   │
    │     def route_to_specialist(state):                              │
    │         query = state["query"]                                   │
    │         if "research" in query:                                  │
    │             return "research_agent"                              │
    │         elif "math" in query or "calculate" in query:            │
    │             return "math_agent"                                  │
    │         else:                                                    │
    │             return "code_agent"                                  │
    │                                                                  │
    │  Benefits:                                                       │
    │  • Each agent specialized for specific task type                 │
    │  • Router intelligently delegates based on query                 │
    │  • Easy to add new specialists (just update router)              │
    │  • Similar to Phase 4 MultiAgentOrchestrator routing             │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Router Pattern - Delegate to Specialists")

    class RouterState(TypedDict):
        """state for router + specialists"""
        query: str
        specialist: str
        result: str

    def router(state: RouterState) -> Literal["research", "math", "code"]:
        """route query to appropriate specialist"""
        query = state["query"].lower()
        print(f"\n   🧭 Router analyzing: '{state['query']}'")

        if "research" in query or "find" in query:
            print("   → Routing to: Research Specialist")
            return "research"
        elif "math" in query or "calculate" in query or "+" in query:
            print("   → Routing to: Math Specialist")
            return "math"
        else:
            print("   → Routing to: Code Specialist")
            return "code"

    def research_agent(state: RouterState) -> dict:
        """specialized research agent"""
        print("   🔍 Research Agent: Searching knowledge base...")
        return {
            "specialist": "research",
            "result": "Research complete: Found 5 relevant papers on AI"
        }

    def math_agent(state: RouterState) -> dict:
        """specialized math agent"""
        print("   🧮 Math Agent: Performing calculations...")
        return {
            "specialist": "math",
            "result": "Math complete: Result is 42"
        }

    def code_agent(state: RouterState) -> dict:
        """specialized coding agent"""
        print("   💻 Code Agent: Generating code...")
        return {
            "specialist": "code",
            "result": "Code complete: Generated Python function"
        }

    # build router graph
    graph = StateGraph(RouterState)
    graph.add_node("research", research_agent)
    graph.add_node("math", math_agent)
    graph.add_node("code", code_agent)

    # router decides which specialist to use
    graph.add_conditional_edges(
        START,
        router,
        {
            "research": "research",
            "math": "math",
            "code": "code"
        }
    )

    graph.add_edge("research", END)
    graph.add_edge("math", END)
    graph.add_edge("code", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure:
                        ┌─→ research_agent
                        │
          START → router ─→ math_agent
                        │
                        └─→ code_agent → END
    """))

    # test different query types
    test_queries = [
        "do research on AI",
        "calculate 2 + 2",
        "write a function"
    ]

    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Query: '{query}'")
        result = app.invoke({"query": query, "specialist": "", "result": ""})
        print(f"\n   Result: {result['result']}")
        print(f"   (Handled by: {result['specialist']} specialist)")

    print("\n✅ Key Takeaway: Router pattern = Phase 4 orchestration in LangGraph")


# endregion


# region Demo 2: Supervisor Pattern (Central Coordinator)
def demo_supervisor_pattern() -> None:
    """
    demonstrate supervisor pattern with coordination

    Concept: Supervisor + Worker Pattern
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Supervisor Coordination                         │
    │                                                                  │
    │  Pattern:                                                        │
    │                      ┌─→ worker_1 ─┐                             │
    │                      │             │                             │
    │     START → supervisor ─→ worker_2 ─→ supervisor → END           │
    │                      │             │       ↑                     │
    │                      └─→ worker_3 ─┘       │                     │
    │                           (or)             │                     │
    │                      supervisor ←──────────┘                     │
    │                           (loop until done)                      │
    │                                                                  │
    │  Supervisor Logic:                                               │
    │     def supervisor(state):                                       │
    │         if state["task_complete"]:                               │
    │             return "finish"                                      │
    │         # analyze state, decide next worker                      │
    │         return next_worker_name                                  │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Complex tasks requiring multiple steps                        │
    │  • Dynamic worker selection based on progress                    │
    │  • Quality control (supervisor validates worker output)          │
    │  • Hierarchical agent systems                                    │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: Supervisor Pattern - Central Coordinator")

    class SupervisorState(TypedDict):
        """state for supervisor + workers"""
        task: str
        workers_completed: list[str]
        task_complete: bool
        results: list[str]
        next_worker: Literal["worker_a", "worker_b", "finish"]  # supervisor's routing decision

    def supervisor(state: SupervisorState) -> dict:
        """supervisor coordinates workers"""
        completed = state["workers_completed"]
        print(f"\n   👔 Supervisor: Completed workers = {completed}")

        if state["task_complete"]:
            print("   → Decision: All work done, finishing")
            return {"next_worker": "finish"}

        # decide next worker based on what's been completed
        if "worker_a" not in completed:
            print("   → Decision: Assign to Worker A (data collection)")
            return {"next_worker": "worker_a"}
        elif "worker_b" not in completed:
            print("   → Decision: Assign to Worker B (analysis)")
            return {"next_worker": "worker_b"}
        else:
            print("   → Decision: Task complete!")
            return {"next_worker": "finish"}

    def route_supervisor(state: SupervisorState) -> Literal["worker_a", "worker_b", "finish"]:
        """read supervisor's decision and route accordingly"""
        return state["next_worker"]

    def worker_a(state: SupervisorState) -> dict:
        """worker A - data collection"""
        print("   👷 Worker A: Collecting data...")
        return {
            "workers_completed": state["workers_completed"] + ["worker_a"],
            "results": state["results"] + ["Data collected: 100 records"]
        }

    def worker_b(state: SupervisorState) -> dict:
        """worker B - analysis"""
        print("   👷 Worker B: Analyzing data...")
        return {
            "workers_completed": state["workers_completed"] + ["worker_b"],
            "results": state["results"] + ["Analysis complete: Found patterns"],
            "task_complete": True
        }

    # build supervisor graph
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("worker_a", worker_a)
    graph.add_node("worker_b", worker_b)

    graph.add_edge(START, "supervisor")

    # supervisor decides what to do next
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,  # router reads next_worker from state
        {
            "worker_a": "worker_a",
            "worker_b": "worker_b",
            "finish": END
        }
    )

    # workers report back to supervisor
    graph.add_edge("worker_a", "supervisor")
    graph.add_edge("worker_b", "supervisor")

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure:
                             ┌─→ worker_a ─┐
          START → supervisor ─→ worker_b ─→ supervisor → END
                      ↑                          │
                      └──────────────────────────┘
                           (loop until done)
    """))

    print("\nExecution:")
    result = app.invoke({
        "task": "analyze customer data",
        "workers_completed": [],
        "task_complete": False,
        "results": [],
        "next_worker": "worker_a"  # initial value (will be updated by supervisor)
    })

    print(f"\n✅ Final Results:")
    for i, res in enumerate(result["results"], 1):
        print(f"   {i}. {res}")

    print("\n✅ Key Takeaway: Supervisor coordinates multiple workers dynamically")


# endregion


# region Demo 3: Sequential Handoff (Agent Chain)
def demo_sequential_handoff() -> None:
    """
    demonstrate sequential handoff between agents

    Concept: Agent Chain with Handoffs
    ┌──────────────────────────────────────────────────────────────────┐
    │                     Sequential Handoff                           │
    │                                                                  │
    │  Pattern:                                                        │
    │     START → agent_1 → agent_2 → agent_3 → END                    │
    │                                                                  │
    │  Each agent:                                                     │
    │  1. Receives state from previous agent                           │
    │  2. Performs specialized work                                    │
    │  3. Updates state for next agent                                 │
    │  4. Hands off to next agent                                      │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Pipeline processing (data → analysis → report)                │
    │  • Content creation (outline → draft → edit → publish)           │
    │  • Software workflow (design → code → test → deploy)             │
    │  • Each stage adds value to previous work                        │
    │                                                                  │
    │  State Accumulation:                                             │
    │  • Each agent adds to state (doesn't replace)                    │
    │  • Final state contains all agents' contributions                │
    │  • Use add_messages for conversation history                     │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Sequential Handoff - Agent Chain")

    class ChainState(TypedDict):
        """state passed through agent chain"""
        topic: str
        messages: Annotated[list[str], add_messages]
        outline: str
        draft: str
        final: str

    def outliner(state: ChainState) -> dict:
        """agent 1: create outline"""
        print("\n   📝 Outliner Agent: Creating outline...")
        outline = f"Outline for '{state['topic']}':\n   1. Intro\n   2. Body\n   3. Conclusion"
        print(f"   {outline}")
        return {
            "outline": outline,
            "messages": ["Outliner: Outline created"]
        }

    def drafter(state: ChainState) -> dict:
        """agent 2: write draft"""
        print("\n   ✍️  Drafter Agent: Writing draft based on outline...")
        draft = f"Draft: Introduction to {state['topic']}..."
        print(f"   {draft}")
        return {
            "draft": draft,
            "messages": ["Drafter: Draft written"]
        }

    def editor(state: ChainState) -> dict:
        """agent 3: edit and finalize"""
        print("\n   ✨ Editor Agent: Polishing draft...")
        final = f"Final: Polished article about {state['topic']}"
        print(f"   {final}")
        return {
            "final": final,
            "messages": ["Editor: Final version ready"]
        }

    # build sequential chain
    graph = StateGraph(ChainState)
    graph.add_node("outliner", outliner)
    graph.add_node("drafter", drafter)
    graph.add_node("editor", editor)

    # sequential handoffs
    graph.add_edge(START, "outliner")
    graph.add_edge("outliner", "drafter")
    graph.add_edge("drafter", "editor")
    graph.add_edge("editor", END)

    app = graph.compile()

    print("\n" + cleandoc("""
        Graph Structure (Sequential):
          START → outliner → drafter → editor → END
    """))

    print("\nExecution:")
    result = app.invoke({
        "topic": "AI Agents",
        "messages": [],
        "outline": "",
        "draft": "",
        "final": ""
    })

    print("\n" + cleandoc(f"""
        ✅ Final State:
           Topic: {result['topic']}
           Outline: {result['outline'][:50]}...
           Draft: {result['draft']}
           Final: {result['final']}
    """))

    print("\n   Message History:")
    for msg in result["messages"]:
        print(f"     • {msg}")

    print("\n✅ Key Takeaway: Sequential handoff = pipeline with state accumulation")


# endregion


# region Demo 4: Parallel Specialists (Concurrent Execution)
def demo_parallel_specialists() -> None:
    """
    demonstrate parallel specialist execution

    Concept: Parallel Specialist Pattern
    ┌──────────────────────────────────────────────────────────────────┐
    │                   Parallel Specialists                           │
    │                                                                  │
    │  Pattern:                                                        │
    │              ┌─→ specialist_a ─┐                                 │
    │              │                 │                                 │
    │     START → fork ─→ specialist_b ─→ aggregator → END             │
    │              │                 │                                 │
    │              └─→ specialist_c ─┘                                 │
    │                                                                  │
    │  Important:                                                      │
    │  • Each specialist works on DIFFERENT state fields!              │
    │  • Or use reducer (operator.add) for same field                  │
    │  • Aggregator combines results from all specialists              │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Multi-perspective analysis                                    │
    │  • Consensus building (multiple opinions)                        │
    │  • Parallel research (different sources)                         │
    │  • Independent evaluations                                       │
    │                                                                  │
    │  Note: This requires Send API (LangGraph 0.2+)                   │
    │  For now, we show sequential pattern (see state_basics.py note)  │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Parallel Specialists (Concept)")

    print(cleandoc("""
        Parallel Specialist Pattern:

                     ┌─→ specialist_a (security check) ─┐
                     │                                  │
          START → fork ─→ specialist_b (performance) ─→ aggregator → END
                     │                                  │
                     └─→ specialist_c (code quality) ───┘

        How It Works:
        • Fork node splits work to multiple specialists
        • Each specialist analyzes from their perspective
        • Specialists work on DIFFERENT state fields to avoid conflicts
        • Aggregator combines all specialist results

        Example State Schema:
            class ParallelState(TypedDict):
                code: str
                security_score: int     # specialist_a writes here
                performance_score: int  # specialist_b writes here
                quality_score: int      # specialist_c writes here
                final_report: str       # aggregator writes here

        Why Different Fields?
        • Avoids "Can receive only one value per step" error
        • Each specialist has dedicated output field
        • Aggregator reads all fields, produces combined result

        Sequential Implementation (This Demo):
        For this learning demo, we show sequential pattern:
            START → specialist_a → specialist_b → specialist_c → aggregator → END

        This teaches the concept without requiring Send API (advanced).
        In production, use Send() for true parallel execution.
    """))

    class AnalysisState(TypedDict):
        """state for parallel analysis"""
        code: str
        security_score: int
        performance_score: int
        quality_score: int
        final_report: str

    def security_specialist(state: AnalysisState) -> dict:
        """analyze security"""
        print("\n   🛡️  Security Specialist: Checking for vulnerabilities...")
        return {"security_score": 85}

    def performance_specialist(state: AnalysisState) -> dict:
        """analyze performance"""
        print("   ⚡ Performance Specialist: Analyzing efficiency...")
        return {"performance_score": 90}

    def quality_specialist(state: AnalysisState) -> dict:
        """analyze code quality"""
        print("   ✨ Quality Specialist: Reviewing code structure...")
        return {"quality_score": 88}

    def aggregator(state: AnalysisState) -> dict:
        """combine all specialist results"""
        print("\n   📊 Aggregator: Combining specialist results...")
        avg_score = (
            state["security_score"] +
            state["performance_score"] +
            state["quality_score"]
        ) / 3

        report = cleandoc(f"""
            Code Analysis Report:
            • Security: {state['security_score']}/100
            • Performance: {state['performance_score']}/100
            • Quality: {state['quality_score']}/100
            • Overall: {avg_score:.1f}/100
        """)

        return {"final_report": report}

    # build sequential analysis graph (simulating parallel)
    graph = StateGraph(AnalysisState)
    graph.add_node("security", security_specialist)
    graph.add_node("performance", performance_specialist)
    graph.add_node("quality", quality_specialist)
    graph.add_node("aggregator", aggregator)

    # sequential execution (each specialist updates different field)
    graph.add_edge(START, "security")
    graph.add_edge("security", "performance")
    graph.add_edge("performance", "quality")
    graph.add_edge("quality", "aggregator")
    graph.add_edge("aggregator", END)

    app = graph.compile()

    print("\n" + "─" * 70)
    print("Execution (Sequential simulation of parallel):")
    result = app.invoke({
        "code": "def hello(): return 'world'",
        "security_score": 0,
        "performance_score": 0,
        "quality_score": 0,
        "final_report": ""
    })

    print(f"\n✅ {result['final_report']}")

    print("\n✅ Key Takeaway: Parallel specialists work on different state fields")
    print("📝 Advanced: Use Send API for true parallel execution (LangGraph 0.2+)")


# endregion


# region Demo 5: Aggregation Pattern (Combine Results)
def demo_aggregation_pattern() -> None:
    """
    demonstrate result aggregation from multiple agents

    Concept: Fan-Out / Fan-In Pattern
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Aggregation Pattern                           │
    │                                                                  │
    │  Pattern:                                                        │
    │     START → gather → agent_1 → ┐                                 │
    │                   → agent_2 → ├─→ combine → END                  │
    │                   → agent_3 → ┘                                  │
    │                                                                  │
    │  Three Phases:                                                   │
    │  1. Gather: Collect information needed for specialists           │
    │  2. Specialize: Each agent contributes unique perspective        │
    │  3. Combine: Aggregator synthesizes all contributions            │
    │                                                                  │
    │  Aggregation Strategies:                                         │
    │  • Voting: Majority consensus                                    │
    │  • Averaging: Mean/median of numeric results                     │
    │  • Concatenation: Append all results                             │
    │  • Selection: Pick best result (quality criteria)                │
    │  • Synthesis: Create new insight from all inputs                 │
    │                                                                  │
    │  Use Cases:                                                      │
    │  • Multi-model consensus (ask 3 LLMs, combine answers)           │
    │  • Ensemble predictions (ML models voting)                       │
    │  • Multi-source research (combine findings)                      │
    │  • Collaborative decision-making                                 │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Aggregation Pattern - Combine Results")

    class AggregationState(TypedDict):
        """state for aggregation"""
        question: str
        agent_responses: list[str]
        confidence_scores: list[int]
        final_answer: str

    def agent_1(state: AggregationState) -> dict:
        """agent 1's perspective"""
        print("\n   🤖 Agent 1: Analyzing from perspective A...")
        return {
            "agent_responses": state["agent_responses"] + ["Answer from Agent 1: Yes"],
            "confidence_scores": state["confidence_scores"] + [85]
        }

    def agent_2(state: AggregationState) -> dict:
        """agent 2's perspective"""
        print("   🤖 Agent 2: Analyzing from perspective B...")
        return {
            "agent_responses": state["agent_responses"] + ["Answer from Agent 2: Yes"],
            "confidence_scores": state["confidence_scores"] + [92]
        }

    def agent_3(state: AggregationState) -> dict:
        """agent 3's perspective"""
        print("   🤖 Agent 3: Analyzing from perspective C...")
        return {
            "agent_responses": state["agent_responses"] + ["Answer from Agent 3: No"],
            "confidence_scores": state["confidence_scores"] + [78]
        }

    def aggregator(state: AggregationState) -> dict:
        """combine agent results using voting + confidence"""
        print("\n   📊 Aggregator: Combining agent responses...")

        # count votes
        yes_votes = sum(1 for agent_response in state["agent_responses"] if "Yes" in agent_response)
        no_votes = len(state["agent_responses"]) - yes_votes

        # calculate average confidence
        avg_confidence = sum(state["confidence_scores"]) / len(state["confidence_scores"])

        # determine consensus
        consensus = "Yes" if yes_votes > no_votes else "No"

        final = cleandoc(f"""
            Consensus Decision: {consensus}
            Votes: Yes={yes_votes}, No={no_votes}
            Average Confidence: {avg_confidence:.1f}%

            Individual Responses:
        """)

        for i, (resp, conf) in enumerate(zip(state["agent_responses"], state["confidence_scores"]), 1):
            final += f"\n            {i}. {resp} (confidence: {conf}%)"

        return {"final_answer": final}

    # build aggregation graph
    graph = StateGraph(AggregationState)
    graph.add_node("agent_1", agent_1)
    graph.add_node("agent_2", agent_2)
    graph.add_node("agent_3", agent_3)
    graph.add_node("aggregator", aggregator)

    # sequential (simulating parallel)
    graph.add_edge(START, "agent_1")
    graph.add_edge("agent_1", "agent_2")
    graph.add_edge("agent_2", "agent_3")
    graph.add_edge("agent_3", "aggregator")
    graph.add_edge("aggregator", END)

    app = graph.compile()

    print("\nGraph Structure:")
    print("  START → agent_1 → agent_2 → agent_3 → aggregator → END")
    print("                                            ↑")
    print("                              (combines all responses)")

    print("\nExecution:")
    result = app.invoke({
        "question": "Should we proceed with this approach?",
        "agent_responses": [],
        "confidence_scores": [],
        "final_answer": ""
    })

    print(f"\n✅ {result['final_answer']}")

    print("\n✅ Key Takeaway: Aggregation combines multiple perspectives into consensus")


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Router Pattern", "router pattern", demo_router_pattern),
    Demo("2", "Supervisor Pattern", "supervisor pattern", demo_supervisor_pattern),
    Demo("3", "Sequential Handoff", "sequential handoff", demo_sequential_handoff),
    Demo("4", "Parallel Specialists", "parallel specialists", demo_parallel_specialists),
    Demo("5", "Aggregation Pattern", "aggregation pattern", demo_aggregation_pattern),
]

# endregion

def main() -> None:
    """run all multi-agent demonstrations"""
    print(cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║               LANGGRAPH MULTI-AGENT COLLABORATION                  ║
        ║                                                                    ║
        ║  Key Concepts:                                                     ║
        ║  • Router delegates to specialized agents                          ║
        ║  • Supervisor coordinates dynamic workflows                        ║
        ║  • Sequential chains build on previous work                        ║
        ║  • Parallel specialists work on different fields                   ║
        ║  • Aggregation combines multiple perspectives                      ║
        ║                                                                    ║
        ║  Phase 4 Connection:                                               ║
        ║  • Router = MultiAgentOrchestrator routing logic                   ║
        ║  • Supervisor = AgentOrchestrator coordination                     ║
        ║  • Patterns = Your Phase 4 agents in declarative graphs!           ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝
    """))

    
    runner = MenuRunner(DEMOS, title="LangGraph Multi-Agent")
    runner.run()


if __name__ == "__main__":
    main()


# endregion
