"""
LangGraph Human-in-the-Loop - Checkpoints and Approval Workflows

This module teaches human-in-the-loop patterns in LangGraph:
- Checkpoints (save state at specific points)
- Interrupts (pause execution for approval)
- Resume execution after approval
- Conditional interrupts (only when needed)
- Update state during interruption

These patterns are CRITICAL for production agents that:
- Execute sensitive actions (database writes, API calls, payments)
- Need human oversight for quality control
- Require approval before proceeding

No API key required - uses mock logic for demonstrations.

Run with: uv run python -m phase7_frameworks.02_langgraph.human_in_loop
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


# region Demo 1: Basic Checkpoint Pattern
def demo_basic_checkpoint() -> None:
    """
    demonstrate basic checkpoint pattern

    Concept: Checkpoints Save State
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Checkpoint Pattern                            │
    │                                                                  │
    │  Without Checkpoints:                                            │
    │     Each invocation starts fresh, no memory                      │
    │                                                                  │
    │  With Checkpoints:                                               │
    │     State persists between invocations                           │
    │     Can resume from any checkpoint                               │
    │     Complete execution history available                         │
    │                                                                  │
    │  Setup:                                                          │
    │     checkpointer = MemorySaver()  # in-memory storage            │
    │     app = graph.compile(checkpointer=checkpointer)               │
    │                                                                  │
    │  Usage:                                                          │
    │     config = {"configurable": {"thread_id": "session_1"}}        │
    │     app.invoke(input, config)  # saves checkpoint                │
    │     app.invoke(input, config)  # resumes from checkpoint         │
    │                                                                  │
    │  Key Insight:                                                    │
    │     thread_id = session identifier (like user session)           │
    │     Different threads = independent execution histories          │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Basic Checkpoint Pattern")

    class CounterState(TypedDict):
        """state with counter"""
        counter: int
        history: list[str]

    def increment(state: CounterState) -> dict:
        """increment counter"""
        new_count = state["counter"] + 1
        print(f"   Incrementing: {state['counter']} → {new_count}")
        return {
            "counter": new_count,
            "history": state["history"] + [f"Step {new_count}"]
        }

    # build graph with checkpoint
    checkpointer = MemorySaver()  # in-memory checkpoint storage
    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)

    app = graph.compile(checkpointer=checkpointer)

    print("\n" + cleandoc("""
            Graph with Checkpoints:
              START → increment → END
                       (checkpoint saved after each step)
        """))

    # session configuration
    config = {"configurable": {"thread_id": "counter_session"}}

    print("\nExecution 1 (fresh start):")
    result1 = app.invoke(
        {"counter": 0, "history": []},
        config
    )
    print(f"  Result: counter={result1['counter']}, history={result1['history']}")

    print("\nExecution 2 (resumes from checkpoint):")
    result2 = app.invoke(
        {"counter": result1["counter"], "history": result1["history"]},
        config
    )
    print(f"  Result: counter={result2['counter']}, history={result2['history']}")

    print("\nExecution 3 (continues from checkpoint):")
    result3 = app.invoke(
        {"counter": result2["counter"], "history": result2["history"]},
        config
    )
    print(f"  Result: counter={result3['counter']}, history={result3['history']}")

    print("\n✅ Key Takeaway: Checkpoints preserve state between invocations")
    print("💡 Use Cases: Chat history, multi-turn conversations, stateful agents")


# endregion


# region Demo 2: Interrupt Before Node
def demo_interrupt_before() -> None:
    """
    demonstrate interrupt before node execution

    Concept: Interrupt Before Action
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Interrupt Pattern                             │
    │                                                                  │
    │  Pattern:                                                        │
    │     START → prepare → [INTERRUPT] → execute → END                │
    │                                                                  │
    │  Setup:                                                          │
    │     app = graph.compile(                                         │
    │         checkpointer=MemorySaver(),                              │
    │         interrupt_before=["execute"]  # pause before execute     │
    │     )                                                            │
    │                                                                  │
    │  Execution:                                                      │
    │     1. invoke() → runs prepare → pauses before execute           │
    │     2. Human reviews state                                       │
    │     3. invoke(None) → resumes → executes → ends                  │
    │                                                                  │
    │  Use Cases:                                                      │
    │     • Approve database writes                                    │
    │     • Review API call parameters                                 │
    │     • Confirm payment amount                                     │
    │     • Quality check before publishing                            │
    │                                                                  │
    │  Key Insight:                                                    │
    │     invoke(None) = resume from checkpoint                        │
    │     No new input needed, uses saved state                        │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: Interrupt Before Node Execution")

    class ActionState(TypedDict):
        """state for action approval"""
        action: str
        approved: bool
        result: str

    def prepare(state: ActionState) -> dict:
        """prepare action"""
        print(f"\n   📋 Preparing action: '{state['action']}'")
        return {"action": state["action"]}

    def execute(state: ActionState) -> dict:
        """execute action (only after approval)"""
        print(f"   ✅ Executing approved action: '{state['action']}'")
        return {
            "result": f"Action '{state['action']}' completed successfully"
        }

    # build graph with interrupt
    checkpointer = MemorySaver()
    graph = StateGraph(ActionState)
    graph.add_node("prepare", prepare)
    graph.add_node("execute", execute)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "execute")
    graph.add_edge("execute", END)

    # compile with interrupt BEFORE execute node
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"]  # pause here for approval
    )

    print("\n" + cleandoc("""
            Graph with Interrupt:
              START → prepare → [INTERRUPT] → execute → END
                                  ↑
                             (pause for approval)
        """))

    config = {"configurable": {"thread_id": "approval_session"}}

    print("\n" + cleandoc("""
            " + "=" * 7
            Step 1: Initial invocation (will pause at interrupt)
            "=" * 70
        """))

    # first invocation - will pause at interrupt
    result1 = app.invoke(
        {"action": "delete database records", "approved": False, "result": ""},
        config
    )

    print("\n" + cleandoc(f"""
            \n   Paused State:
                 action: '{result1['action']}'
                 result: '{result1['result']}'
        """))

    print("\n" + cleandoc("""
               🤔 Human Review:
                  → Action prepared but NOT executed yet
                  → State saved at checkpoint
                  → Waiting for approval...
        """))

    print("\n" + cleandoc("""
            " + "=" * 7
            Step 2: Resume execution (after approval)
            "=" * 70
        """))

    # resume with None (uses checkpoint state)
    print("\n   👍 Human: Approved! Resuming execution...")
    result2 = app.invoke(None, config)

    print("\n" + cleandoc(f"""
            \n   Final State:
                 action: '{result2['action']}'
                 result: '{result2['result']}'
        """))

    print("\n✅ Key Takeaway: Interrupt before allows human approval before actions")
    print("⚠️  Critical: Use for sensitive operations (DB writes, payments, etc.)")


# endregion


# region Demo 3: Interrupt After Node
def demo_interrupt_after() -> None:
    """
    demonstrate interrupt after node execution

    Concept: Interrupt After for Review
    ┌──────────────────────────────────────────────────────────────────┐
    │                 Interrupt After Pattern                          │
    │                                                                  │
    │  Pattern:                                                        │
    │     START → generate → [INTERRUPT] → publish → END               │
    │                                                                  │
    │  Setup:                                                          │
    │     app = graph.compile(                                         │
    │         checkpointer=MemorySaver(),                              │
    │         interrupt_after=["generate"]  # pause after generate     │
    │     )                                                            │
    │                                                                  │
    │  Use Cases:                                                      │
    │     • Review LLM output before using                             │
    │     • Quality check generated content                            │
    │     • Validate analysis results                                  │
    │     • Approve recommendations                                    │
    │                                                                  │
    │  Difference from interrupt_before:                               │
    │     • AFTER: Node executes, then pause for review                │
    │     • BEFORE: Pause, then execute after approval                 │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Interrupt After Node Execution")

    class ContentState(TypedDict):
        """state for content review"""
        topic: str
        draft: str
        published: bool

    def generate(state: ContentState) -> dict:
        """generate content"""
        draft = f"AI-generated article about {state['topic']}...\n(contains important insights)"
        print(f"\n   ✍️  Generated draft for topic: '{state['topic']}'")
        return {"draft": draft}

    def publish(state: ContentState) -> dict:
        """publish content (only after review)"""
        print(f"   📢 Publishing content: {state['draft'][:50]}...")
        return {"published": True}

    # build graph with interrupt after
    checkpointer = MemorySaver()
    graph = StateGraph(ContentState)
    graph.add_node("generate", generate)
    graph.add_node("publish", publish)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "publish")
    graph.add_edge("publish", END)

    # compile with interrupt AFTER generate node
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["generate"]  # pause here for review
    )

    print("\n" + cleandoc("""
            Graph with Interrupt:
              START → generate → [INTERRUPT] → publish → END
                                     ↑
                                (pause for review)
        """))

    config = {"configurable": {"thread_id": "review_session"}}

    print("\n" + cleandoc("""
            " + "=" * 7
            Step 1: Generate content (will pause after generation)
            "=" * 70
        """))

    result1 = app.invoke(
        {"topic": "AI Agents", "draft": "", "published": False},
        config
    )

    print("\n" + cleandoc(f"""
            \n   Paused State (after generation):
                 topic: '{result1['topic']}'
                 draft: '{result1['draft']}'
                 published: {result1['published']}
        """))

    print("\n" + cleandoc("""
               🤔 Human Review:
                  → Content generated successfully
                  → Draft ready for review
                  → NOT published yet
        """))

    print("\n" + cleandoc("""
            " + "=" * 7
            Step 2: Continue to publish (after review)
            "=" * 70
        """))

    print("\n   👍 Human: Content looks good! Publishing...")
    result2 = app.invoke(None, config)

    print("\n" + cleandoc(f"""
            \n   Final State:
                 topic: '{result2['topic']}'
                 draft: '{result2['draft'][:50]}...'
                 published: {result2['published']}
        """))

    print("\n✅ Key Takeaway: Interrupt after allows review before proceeding")
    print("💡 Use Cases: LLM output review, quality control, validation")


# endregion


# region Demo 4: Conditional Interrupt
def demo_conditional_interrupt() -> None:
    """
    demonstrate conditional interrupt (only when needed)

    Concept: Smart Interrupts
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Conditional Interrupt                           │
    │                                                                  │
    │  Pattern:                                                        │
    │     def should_interrupt(state):                                 │
    │         if state["needs_approval"]:                              │
    │             return "approve"  # interrupt here                   │
    │         return "execute"  # skip interrupt                       │
    │                                                                  │
    │  Use Cases:                                                      │
    │     • Only interrupt for high-value transactions                 │
    │     • Skip approval for trusted users                            │
    │     • Conditional quality checks                                 │
    │     • Risk-based approval workflows                              │
    │                                                                  │
    │  Implementation:                                                 │
    │     Use conditional_edges + interrupt_before on approve node     │
    │     Router decides whether to go to approve or skip it           │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Conditional Interrupt (Smart Approval)")

    class TransactionState(TypedDict):
        """state for transaction approval"""
        amount: int
        threshold: int
        approved: bool
        result: str
        needs_approval: bool

    def check_amount(state: TransactionState) -> dict:
        """check if amount exceeds threshold"""
        needs_approval = state["amount"] > state["threshold"]
        print(f"\n   💰 Transaction: ${state['amount']}")
        print(f"   📊 Threshold: ${state['threshold']}")

        if needs_approval:
            print("   ⚠️  Exceeds threshold → Needs approval")
        else:
            print("   ✅ Within threshold → Auto-approve")

        return {"needs_approval": needs_approval}

    def router(state: TransactionState) -> Literal["approve", "execute"]:
        """route to approval or direct execution"""
        if state["needs_approval"]:
            return "approve"
        return "execute"

    def approve(state: TransactionState) -> dict:
        """approval node (will interrupt here)"""
        print("\n   ⏸️  Paused for approval...")
        return {"approved": True}

    def execute(state: TransactionState) -> dict:
        """execute transaction"""
        print(f"\n   ✅ Executing transaction: ${state['amount']}")
        return {"result": f"Transaction for ${state['amount']} completed"}

    # build graph with conditional interrupt
    checkpointer = MemorySaver()
    graph = StateGraph(TransactionState)
    graph.add_node("check", check_amount)
    graph.add_node("approve", approve)
    graph.add_node("execute", execute)

    graph.add_edge(START, "check")

    # conditional routing
    graph.add_conditional_edges(
        "check",
        router,
        {
            "approve": "approve",
            "execute": "execute"
        }
    )

    graph.add_edge("approve", "execute")
    graph.add_edge("execute", END)

    # only interrupt at approve node
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approve"]  # only pause if going to approve
    )

    print("\n" + cleandoc("""
            Graph with Conditional Interrupt:
                             ┌─→ approve (needs approval) → [INTERRUPT]
              START → check ─┤
                             └─→ execute (auto-approve) → END
        """))

    # test 1: small transaction (no approval needed)
    print("\n" + cleandoc("""
            " + "=" * 7
            Test 1: Small Transaction ($50, threshold=$100)
            "=" * 70
        """))

    config1 = {"configurable": {"thread_id": "tx_small"}}
    result1 = app.invoke(
        {
            "amount": 50,
            "threshold": 100,
            "approved": False,
            "result": "",
            "needs_approval": False
        },
        config1
    )

    print(f"\n   Result: {result1['result']}")
    print("   ✅ No interrupt (auto-approved)")

    # test 2: large transaction (needs approval)
    print("\n" + cleandoc("""
            " + "=" * 7
            Test 2: Large Transaction ($150, threshold=$100)
            "=" * 70
        """))

    config2 = {"configurable": {"thread_id": "tx_large"}}
    result2 = app.invoke(
        {
            "amount": 150,
            "threshold": 100,
            "approved": False,
            "result": "",
            "needs_approval": False
        },
        config2
    )

    print("\n" + cleandoc(f"""
            \n   Paused State:
                 amount: ${result2['amount']}
                 needs_approval: {result2['needs_approval']}
                 result: '{result2['result']}'
        """))

    print("\n   🤔 Human: Review needed for large transaction...")
    print("   👍 Human: Approved! Resuming...")

    result3 = app.invoke(None, config2)

    print(f"\n   Final Result: {result3['result']}")

    print("\n✅ Key Takeaway: Conditional interrupts only pause when needed")
    print("💡 Use Cases: Risk-based approval, threshold-based reviews")


# endregion


# region Demo 5: Update State During Interrupt
def demo_update_during_interrupt() -> None:
    """
    demonstrate updating state during interrupt

    Concept: Modify State While Paused
    ┌──────────────────────────────────────────────────────────────────┐
    │                Update State During Interrupt                     │
    │                                                                  │
    │  Pattern:                                                        │
    │     1. invoke() → pauses at interrupt                            │
    │     2. Human reviews and modifies state                          │
    │     3. invoke(modified_state) → resumes with new values          │
    │                                                                  │
    │  Use Cases:                                                      │
    │     • Edit LLM output before proceeding                          │
    │     • Adjust parameters after review                             │
    │     • Fix errors before continuing                               │
    │     • Add missing information                                    │
    │                                                                  │
    │  Key Difference:                                                 │
    │     • invoke(None) → resume with same state                      │
    │     • invoke(new_state) → resume with modified state             │
    └──────────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Update State During Interrupt")

    class EmailState(TypedDict):
        """state for email editing"""
        recipient: str
        subject: str
        body: str
        sent: bool

    def draft(state: EmailState) -> dict:
        """draft email"""
        subject = f"Update on {state['recipient']} project"
        body = f"Hello {state['recipient']},\nThis is an automated message..."
        print(f"\n   📝 Drafted email to: {state['recipient']}")
        return {"subject": subject, "body": body}

    def send(state: EmailState) -> dict:
        """send email"""
        print(cleandoc(            f"""

                \n   📧 Sending email:
                      To: {state['recipient']}
                      Subject: {state['subject']}
                      Body: {state['body'][:50]}...
            """))
        return {"sent": True}

    # build graph
    checkpointer = MemorySaver()
    graph = StateGraph(EmailState)
    graph.add_node("draft", draft)
    graph.add_node("send", send)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "send")
    graph.add_edge("send", END)

    # interrupt after draft for editing
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["draft"]
    )

    print("\n" + cleandoc("""
            Graph with Interrupt:
              START → draft → [INTERRUPT] → send → END
                                  ↑
                             (edit before sending)
        """))

    config = {"configurable": {"thread_id": "email_session"}}

    print("\n" + cleandoc("""
            " + "=" * 7
            Step 1: Draft email (will pause for editing)
            "=" * 70
        """))

    result1 = app.invoke(
        {"recipient": "Alice", "subject": "", "body": "", "sent": False},
        config
    )

    print("\n" + cleandoc(f"""
            \n   Paused State (draft ready):
                 recipient: {result1['recipient']}
                 subject: {result1['subject']}
                 body: {result1['body']}
        """))

    print("\n" + cleandoc("""
               🤔 Human Review:
                  → Subject could be more specific
                  → Body needs personalization
                  → Editing before sending...
        """))

    print("\n" + cleandoc("""
            " + "=" * 7
            Step 2: Resume with edited state
            "=" * 70
        """))

    # resume with MODIFIED state
    edited_state = {
        "recipient": result1["recipient"],
        "subject": "Q4 Project Status Update",  # edited
        "body": f"Hi {result1['recipient']},\nGreat work on the project! Here's the update...",  # edited
        "sent": False
    }

    print("\n" + cleandoc(f"""
               ✏️  Human edited:
                  subject: '{edited_state['subject']}'
                  body: '{edited_state['body'][:50]}...'
        """))

    result2 = app.invoke(edited_state, config)

    print(f"\n   Final State:")
    print(f"     sent: {result2['sent']}")

    print("\n✅ Key Takeaway: Can update state during interrupt before resuming")
    print("💡 Use Cases: Edit LLM output, fix parameters, add missing data")


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Basic Checkpoints", "basic checkpoints", demo_basic_checkpoint),
    Demo("2", "Interrupt Before", "interrupt before", demo_interrupt_before),
    Demo("3", "Interrupt After", "interrupt after", demo_interrupt_after),
    Demo("4", "Conditional Interrupt", "conditional interrupt", demo_conditional_interrupt),
    Demo("5", "Update During Interrupt", "update during interrupt", demo_update_during_interrupt),
]

# endregion

def main() -> None:
    """run all human-in-the-loop demonstrations"""
    print(cleandoc("""
        ╔════════════════════════════════════════════════════════════════════╗
        ║                                                                    ║
        ║             LANGGRAPH HUMAN-IN-THE-LOOP PATTERNS                   ║
        ║                                                                    ║
        ║  Key Concepts:                                                     ║
        ║  • Checkpoints save state between invocations                      ║
        ║  • Interrupts pause execution for human review                     ║
        ║  • thread_id identifies independent sessions                       ║
        ║  • invoke(None) resumes from checkpoint                            ║
        ║  • invoke(new_state) resumes with modifications                    ║
        ║                                                                    ║
        ║  Production Patterns:                                              ║
        ║  • Database writes → interrupt_before for approval                 ║
        ║  • LLM output → interrupt_after for review                         ║
        ║  • High-value transactions → conditional interrupts                ║
        ║  • Content publishing → edit during interrupt                      ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝
    """))

    
    runner = MenuRunner(DEMOS, title="LangGraph Human-in-the-Loop")
    runner.run()


if __name__ == "__main__":
    main()


# endregion
