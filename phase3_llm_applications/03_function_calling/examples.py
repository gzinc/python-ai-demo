"""
Function Calling Examples - Practical demos

This file contains ONLY the demo/example functions.
All implementation classes are in their own files:
- schemas/        → Tool, ToolParameter, ToolResult
- registry.py    → ToolRegistry
- executor.py    → ToolExecutor
- common_tools.py → create_*_tool(), implementations
- engine.py      → FunctionCallingEngine

Run with: uv run python phase3_llm_applications/03_function_calling/examples.py
"""

from inspect import cleandoc
from dotenv import load_dotenv

from schemas import Tool, ToolParameter
from engine import FunctionCallingEngine
from common_tools import (
    create_weather_tool,
    create_calculator_tool,
    create_search_tool,
    get_weather,
    calculate,
    validate_calculate_args,
)

load_dotenv()


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# EXAMPLES
# ─────────────────────────────────────────────────────────────


def example_basic_function_call():
    """demonstrate basic function calling"""
    print_section("Example 1: Basic Function Calling")

    engine = FunctionCallingEngine()

    # register weather tool
    engine.register_tool(
        tool=create_weather_tool(),
        function=get_weather,
    )

    print("\nRegistered tools:", engine.registry.list_tools())

    print("\n--- Conversation ---")
    print("User: What's the weather in Tokyo?")
    response = engine.chat("What's the weather in Tokyo?")
    print(f"Assistant: {response}")


def example_multi_tool():
    """demonstrate multiple tools"""
    print_section("Example 2: Multiple Tools")

    engine = FunctionCallingEngine()

    # register multiple tools
    engine.register_tool(create_weather_tool(), get_weather)
    engine.register_tool(create_calculator_tool(), calculate, validate_calculate_args)

    print("\nRegistered tools:", engine.registry.list_tools())

    print("\n--- Query requiring multiple tools ---")
    print("User: What's the weather in Tokyo and London? Also, what's 22 minus 15?")
    response = engine.chat(
        "What's the weather in Tokyo and London? Also, what's 22 minus 15?"
    )
    print(f"Assistant: {response}")


def example_conversation_with_tools():
    """demonstrate multi-turn conversation with tools"""
    print_section("Example 3: Conversation with Tools")

    engine = FunctionCallingEngine()
    engine.register_tool(create_weather_tool(), get_weather)
    engine.register_tool(create_calculator_tool(), calculate, validate_calculate_args)

    conversations = [
        "What's the weather like in Paris?",
        "How about in New York?",
        "What's the temperature difference?",
    ]

    print("\n--- Multi-turn conversation ---")
    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = engine.chat(user_msg)
        print(f"Assistant: {response}")


def example_tool_choice():
    """demonstrate LLM choosing NOT to use tools"""
    print_section("Example 4: Tool Choice (LLM Decides)")

    engine = FunctionCallingEngine()
    engine.register_tool(create_weather_tool(), get_weather)

    queries = [
        "What's the weather in Tokyo?",  # should use tool
        "What is the capital of France?",  # should NOT use tool
        "Tell me a joke",  # should NOT use tool
    ]

    print("\n--- LLM decides when to use tools ---")
    for query in queries:
        engine.reset()
        print(f"\nUser: {query}")
        response = engine.chat(query)
        print(f"Assistant: {response[:150]}{'...' if len(response) > 150 else ''}")


def example_error_handling():
    """demonstrate handling tool errors"""
    print_section("Example 5: Error Handling")

    # create a tool that can fail
    def risky_function(value: int) -> dict:
        if value < 0:
            raise ValueError("value must be non-negative")
        return {"result": value * 2}

    risky_tool = Tool(
        name="risky_operation",
        description="A function that doubles a value but fails on negative numbers",
        parameters=[
            ToolParameter(
                name="value",
                type="number",
                description="A non-negative integer",
                required=True,
            )
        ]
    )

    engine = FunctionCallingEngine()
    engine.register_tool(risky_tool, risky_function)

    print("\n--- Error handling demo ---")

    # manually test the executor
    print("\nDirect executor test:")
    result1 = engine.executor.execute("risky_operation", {"value": 5})
    print(f"  value=5: success={result1.success}, data={result1.data}")

    result2 = engine.executor.execute("risky_operation", {"value": -1})
    print(f"  value=-1: success={result2.success}, error={result2.error}")


def example_agent_loop_pattern():
    """
    demonstrate the agent loop pattern

    This is the foundation for AI agents in Phase 4!
    """
    print_section("Example 6: Agent Loop Pattern (Preview)")

    print(cleandoc("""
        The Agent Loop:
        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
        │  def agent_loop(task):                                      │
        │      messages = [{"role": "user", "content": task}]         │
        │                                                             │
        │      while True:                                            │
        │          response = llm.chat(messages, tools=tools)         │
        │                                                             │
        │          if response.has_tool_calls:                        │
        │              # LLM wants to use a tool                      │
        │              for tool_call in response.tool_calls:          │
        │                  result = execute(tool_call)                │
        │                  messages.append(result)                    │
        │              # Continue loop for more processing            │
        │                                                             │
        │          else:                                              │
        │              # LLM is done - return final answer            │
        │              return response.content                        │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘

        This pattern is implemented in FunctionCallingEngine.chat()
        and will be expanded in Phase 4: AI Agents!
    """))

    # demonstrate with actual engine
    engine = FunctionCallingEngine()
    engine.register_tool(create_weather_tool(), get_weather)
    engine.register_tool(create_calculator_tool(), calculate)

    print("Demonstration:")
    print("User: Compare the weather in Tokyo and London, then calculate the difference")
    response = engine.chat(
        "Compare the weather in Tokyo and London, then calculate the temperature difference."
    )
    print(f"Assistant: {response}")


def example_schema_inspection():
    """show what tool schemas look like"""
    print_section("Example 7: Tool Schema Inspection")

    tools = [
        create_weather_tool(),
        create_calculator_tool(),
        create_search_tool(),
    ]

    print("\nTool schemas sent to LLM:")
    for tool in tools:
        schema = tool.to_openai_schema()
        print(f"\n--- {tool.name} ---")
        print(f"Description: {tool.description[:60]}...")
        print(f"Parameters: {[p.name for p in tool.parameters]}")
        print(f"Required: {schema['function']['parameters'].get('required', [])}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Function Calling Examples - LLM Tool Integration")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Basic Function Call", "single tool usage with weather API"),
        ("2", "Multiple Tools", "LLM using multiple tools in one query"),
        ("3", "Conversation with Tools", "multi-turn chat with tool access"),
        ("4", "Tool Choice", "LLM deciding when to use tools"),
        ("5", "Error Handling", "handling tool failures gracefully"),
        ("6", "Agent Loop Pattern", "foundation for AI agents"),
        ("7", "Schema Inspection", "viewing tool schemas sent to LLM"),
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
        '1': ('Basic Function Call', example_basic_function_call),
        '2': ('Multiple Tools', example_multi_tool),
        '3': ('Conversation with Tools', example_conversation_with_tools),
        '4': ('Tool Choice', example_tool_choice),
        '5': ('Error Handling', example_error_handling),
        '6': ('Agent Loop Pattern', example_agent_loop_pattern),
        '7': ('Schema Inspection', example_schema_inspection),
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
