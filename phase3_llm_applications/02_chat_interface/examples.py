"""
Chat Interface Examples - Interactive demos and usage patterns

This module demonstrates various ways to build chat interfaces:
1. Basic chat loop with memory
2. Different memory strategies comparison
3. Streaming chat responses
4. Reference resolution across turns

Module Structure (Java-like separation):
- engine.py      → ChatEngine (orchestrator)
- chat_memory.py → ChatMemory, SummarizingMemory (memory strategies)
- streaming.py   → Streaming utilities
- examples.py    → Demo functions (this file)

Run with: uv run python phase3_llm_applications/02_chat_interface/examples.py
"""

from dotenv import load_dotenv

from chat_memory import ChatMemory
from streaming import stream_to_console, _simulate_stream
from engine import ChatEngine

load_dotenv()


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# EXAMPLE 1: BASIC CHAT
# ─────────────────────────────────────────────────────────────


def example_basic_chat():
    """
    demonstrate basic chat with memory

    Shows how conversation context is maintained:
    ┌─────────────────────────────────────────────────────────────┐
    │  Turn 1: "What is Python?"                                  │
    │  Turn 2: "What are its main features?"                     │
    │           ↑ LLM knows "its" = Python                       │
    │  Turn 3: "Show me an example"                              │
    │           ↑ LLM knows context from previous turns          │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Example 1: Basic Chat with Memory")

    engine = ChatEngine(
        system_prompt="You are a helpful coding assistant. Be concise.",
        streaming=True,
    )

    conversations = [
        "What is Python?",
        "What are its main features?",  # "its" refers to Python
        "Show me a simple example",      # "example" of Python features
    ]

    for user_msg in conversations:
        print(f"\n   You: {user_msg}")
        print("   Assistant: ", end="")
        engine.chat(user_msg)

    print(f"\n   --- Conversation Stats ---")
    print(f"   Turns: {engine.memory.get_turn_count()}")
    print(f"   Messages in memory: {len(engine.memory)}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 2: MEMORY STRATEGIES
# ─────────────────────────────────────────────────────────────


def example_memory_strategies():
    """compare different memory management strategies"""
    print_section("Example 2: Memory Strategies Comparison")

    # create memories with different strategies
    full_memory = ChatMemory(strategy="full", system_prompt="Assistant")
    window_memory = ChatMemory(strategy="sliding_window", max_messages=4, system_prompt="Assistant")
    budget_memory = ChatMemory(strategy="token_budget", max_tokens=200, system_prompt="Assistant")

    # simulate conversation
    messages = [
        ("user", "Question 1: What is AI?"),
        ("assistant", "AI is artificial intelligence, the simulation of human intelligence."),
        ("user", "Question 2: What is ML?"),
        ("assistant", "ML is machine learning, a subset of AI that learns from data."),
        ("user", "Question 3: What is DL?"),
        ("assistant", "DL is deep learning, using neural networks with many layers."),
        ("user", "Question 4: What is NLP?"),
        ("assistant", "NLP is natural language processing, enabling computers to understand text."),
    ]

    for role, content in messages:
        if role == "user":
            full_memory.add_user_message(content)
            window_memory.add_user_message(content)
            budget_memory.add_user_message(content)
        else:
            full_memory.add_assistant_message(content)
            window_memory.add_assistant_message(content)
            budget_memory.add_assistant_message(content)

    print("\n   After 4 conversation turns:")
    print(f"\n   Full Strategy:")
    print(f"      Messages: {len(full_memory)}")
    print(f"      Tokens: ~{full_memory._estimate_tokens()}")

    print(f"\n   Sliding Window (max 4):")
    print(f"      Messages: {len(window_memory)}")
    print(f"      Kept: {[m.content[:20]+'...' for m in window_memory.messages]}")

    print(f"\n   Token Budget (max 200):")
    print(f"      Messages: {len(budget_memory)}")
    print(f"      Tokens: ~{budget_memory._estimate_tokens()}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 3: STREAMING CHAT
# ─────────────────────────────────────────────────────────────


def example_streaming_chat():
    """demonstrate streaming vs non-streaming"""
    print_section("Example 3: Streaming Chat")

    print("\n   Streaming enabled - response appears token by token:")
    print("   ", end="")

    # simulate streaming response
    text = "This response streams word by word, giving immediate feedback to the user. Much better UX than waiting for the complete response!"
    stream_to_console(_simulate_stream(text))

    print("\n   Non-streaming - would wait then show all at once")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 4: REFERENCE RESOLUTION
# ─────────────────────────────────────────────────────────────


def example_reference_resolution():
    """
    demonstrate how LLM resolves references using context

    Key insight: References work because the FULL conversation
    is sent to the LLM every time!
    """
    print_section("Example 4: Reference Resolution")

    memory = ChatMemory(system_prompt="You are a helpful assistant.")

    # simulate conversation with references
    conversation = [
        ("user", "Tell me about Tesla stock"),
        ("assistant", "Tesla (TSLA) is an electric vehicle company. The stock is known for its volatility."),
        ("user", "What about its competitors?"),  # "its" = Tesla
        ("assistant", "Tesla's main competitors include Rivian, Lucid, and traditional automakers like Ford and GM."),
        ("user", "Compare them"),  # "them" = Tesla and competitors
        ("assistant", "Tesla leads in EV market share, while Rivian focuses on trucks and Lucid on luxury."),
        ("user", "Which one is the safest investment?"),  # "which one" = among compared
    ]

    print("\n   Conversation with references:")
    for role, content in conversation:
        if role == "user":
            memory.add_user_message(content)
            print(f"\n   User: {content}")

            # show what gets sent to API
            if "its" in content or "them" in content or "one" in content:
                print(f"         ↑ Reference resolved using conversation history!")
        else:
            memory.add_assistant_message(content)
            preview = content[:60] + "..." if len(content) > 60 else content
            print(f"   Assistant: {preview}")

    print(f"\n   --- How References Work ---")
    print(f"   Messages sent to API: {len(memory.get_messages())}")
    print("   The LLM sees ALL previous messages, so it knows:")
    print("   • 'its' refers to Tesla")
    print("   • 'them' refers to Tesla and competitors")
    print("   • 'which one' refers to the compared companies")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 5: INTERACTIVE CHAT (REPL)
# ─────────────────────────────────────────────────────────────


def example_interactive():
    """
    interactive chat loop

    This shows what a real chat interface looks like
    """
    print_section("Example 5: Interactive Chat Demo")

    print("\n   Sample chat session:")

    # simulate interactive session
    sample_exchanges = [
        ("Hi, I'm learning Python", "Hello! That's great. Python is an excellent language to learn. What aspect would you like to focus on?"),
        ("Functions", "Functions in Python are defined using the 'def' keyword. They help organize code into reusable blocks."),
        ("Show me", "Here's a simple example:\n\n   def greet(name):\n       return f'Hello, {name}!'\n\n   print(greet('World'))"),
    ]

    for user_msg, assistant_msg in sample_exchanges:
        print(f"\n   You: {user_msg}")
        print(f"   Assistant: {assistant_msg[:100]}{'...' if len(assistant_msg) > 100 else ''}")

    print("\n   [In a real app, this would be an interactive loop]")
    print("   [Type 'quit' to exit, 'reset' to clear history]")


# ─────────────────────────────────────────────────────────────
# INTERACTIVE CHAT LOOP (run separately)
# ─────────────────────────────────────────────────────────────


def run_interactive_chat():
    """
    run actual interactive chat loop

    Usage: uv run python -c "from examples import run_interactive_chat; run_interactive_chat()"
    """
    print("\n" + "=" * 60)
    print("  Interactive Chat")
    print("=" * 60)
    print("\nCommands: 'quit' to exit, 'reset' to clear history")
    print("-" * 60)

    engine = ChatEngine(
        system_prompt="You are a helpful assistant. Be concise but thorough.",
        streaming=True,
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                engine.reset()
                print("[Conversation cleared]")
                continue

            print("Assistant: ", end="")
            engine.chat(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Chat Interface Examples - Multi-turn Chat Patterns")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Basic Chat", "conversation with context memory"),
        ("2", "Memory Strategies", "compare full, window, budget strategies"),
        ("3", "Streaming Chat", "streaming vs non-streaming responses"),
        ("4", "Reference Resolution", "how LLM resolves references using context"),
        ("5", "Interactive Demo", "sample chat session walkthrough"),
        ("6", "Interactive Chat (REPL)", "actual interactive chat loop"),
    ]

    for num, name, desc in demos:
        print(f"   [{num}] {name}")
        print(f"      {desc}")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 70)


def run_selected_demos(selections: str) -> bool:
    """run selected demos based on user input"""
    selections = selections.strip().lower()

    if selections == 'q':
        return False

    demo_map = {
        '1': ('Basic Chat', example_basic_chat),
        '2': ('Memory Strategies', example_memory_strategies),
        '3': ('Streaming Chat', example_streaming_chat),
        '4': ('Reference Resolution', example_reference_resolution),
        '5': ('Interactive Demo', example_interactive),
        '6': ('Interactive Chat (REPL)', run_interactive_chat),
    }

    if selections == 'a':
        demos_to_run = list(demo_map.keys())
    else:
        demos_to_run = [s.strip() for s in selections.replace(',', ' ').split() if s.strip() in demo_map]

    if not demos_to_run:
        print("\n⚠️  invalid selection. please enter demo numbers, 'a' for all, or 'q' to quit")
        return True

    print(f"\n🚀 Running {len(demos_to_run)} demo(s)...\n")

    for demo_num in demos_to_run:
        name, func = demo_map[demo_num]
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n⚠️  demo interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ error in demo: {e}")
            continue

    print("\n✅ selected demos complete!")
    return True


def main():
    """run demonstrations with interactive menu"""
    while True:
        show_menu()

        try:
            selection = input("\n🎯 select demo(s) (e.g., '1', '1,3', or 'a' for all): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 goodbye!")
            break

        if not run_selected_demos(selection):
            print("\n👋 goodbye!")
            break

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 goodbye!")
            break


if __name__ == "__main__":
    main()
