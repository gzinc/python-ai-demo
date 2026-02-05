"""
Simple Agent Examples - Demonstrating ReAct pattern

This module shows different ways to use the ReActAgent:
1. Simple single-tool queries
2. Multi-step reasoning tasks
3. Complex multi-tool tasks

Run with: uv run python phase4_ai_agents/01_simple_agent/examples.py
"""

import json
import os
from dotenv import load_dotenv

from agent import ReActAgent
from schemas import AgentConfig

load_dotenv()


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# MOCK TOOLS (for demos without real APIs)
# ─────────────────────────────────────────────────────────────


def get_weather(city: str) -> str:
    """
    mock weather API

    In production, this would call a real weather API.
    """
    weather_data = {
        "tokyo": {"temp_c": 18, "conditions": "rainy", "humidity": 85, "wind_kph": 12},
        "paris": {"temp_c": 22, "conditions": "sunny", "humidity": 45, "wind_kph": 8},
        "new york": {"temp_c": 15, "conditions": "cloudy", "humidity": 60, "wind_kph": 20},
        "london": {"temp_c": 12, "conditions": "foggy", "humidity": 90, "wind_kph": 5},
        "sydney": {"temp_c": 28, "conditions": "sunny", "humidity": 55, "wind_kph": 15},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return json.dumps(weather_data[city_lower])
    return json.dumps({"error": f"Weather data not available for {city}"})


def calculate(expression: str) -> str:
    """
    safe calculator tool

    Only allows basic math operations.
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters in expression"})
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_knowledge(query: str) -> str:
    """
    mock knowledge search

    In production, this would search a RAG system or knowledge base.
    """
    knowledge_base = {
        "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
        "tokyo": "Tokyo is the capital of Japan with a population of about 14 million. Known for technology and culture.",
        "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. Built in 1889, it's 330 meters tall.",
    }
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return json.dumps({"found": True, "content": value})
    return json.dumps({"found": False, "content": "No relevant information found"})


def convert_temperature(value: str, from_unit: str, to_unit: str) -> str:
    """
    temperature conversion tool
    """
    try:
        temp = float(value)
        if from_unit.lower() == "c" and to_unit.lower() == "f":
            result = (temp * 9 / 5) + 32
            return json.dumps({"result": round(result, 1), "unit": "F"})
        elif from_unit.lower() == "f" and to_unit.lower() == "c":
            result = (temp - 32) * 5 / 9
            return json.dumps({"result": round(result, 1), "unit": "C"})
        else:
            return json.dumps({"error": "Unsupported conversion"})
    except ValueError:
        return json.dumps({"error": "Invalid temperature value"})


# ─────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────

TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
    "convert_temperature": convert_temperature,
}

TOOL_DESCRIPTIONS = {
    "get_weather": 'get_weather(city="city name") - Get current weather for a city',
    "calculate": 'calculate(expression="2+2") - Calculate a math expression',
    "search_knowledge": 'search_knowledge(query="topic") - Search knowledge base for information',
    "convert_temperature": 'convert_temperature(value="20", from_unit="C", to_unit="F") - Convert temperature',
}


# ─────────────────────────────────────────────────────────────
# region Demo Functions

def demo_simple_query():
    """
    Demo 1: Simple single-tool query

    Agent uses one tool to answer a straightforward question.
    """
    print_section("Demo 1: Simple Weather Query")

    config = AgentConfig(max_iterations=5, verbose=True)
    agent = ReActAgent(
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        config=config,
    )

    result = agent.run("What's the weather like in Paris right now?")

    print(f"\n📊 Summary:")
    print(f"   Answer: {result.answer}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Tool calls: {result.total_tool_calls}")


def demo_multi_step_reasoning():
    """
    Demo 2: Multi-step reasoning

    Agent needs to use multiple tools and reason about results.
    """
    print_section("Demo 2: Multi-Step Reasoning")

    config = AgentConfig(max_iterations=10, verbose=True)
    agent = ReActAgent(
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        config=config,
    )

    result = agent.run(
        "What's the temperature in Tokyo in Fahrenheit? "
        "Also tell me if I should bring an umbrella."
    )

    print(f"\n📊 Summary:")
    print(f"   Answer: {result.answer}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Tool calls: {result.total_tool_calls}")


def demo_knowledge_and_calculation():
    """
    Demo 3: Combining knowledge search with calculation

    Agent needs to find information and do math.
    """
    print_section("Demo 3: Knowledge + Calculation")

    config = AgentConfig(max_iterations=10, verbose=True)
    agent = ReActAgent(
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        config=config,
    )

    result = agent.run(
        "How tall is the Eiffel Tower? "
        "If I stacked 3 of them, how tall would that be?"
    )

    print(f"\n📊 Summary:")
    print(f"   Answer: {result.answer}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Tool calls: {result.total_tool_calls}")


def demo_compare_cities():
    """
    Demo 4: Comparing data from multiple API calls

    Agent needs to gather data from multiple sources and compare.
    """
    print_section("Demo 4: Compare Multiple Cities")

    config = AgentConfig(max_iterations=10, verbose=True)
    agent = ReActAgent(
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        config=config,
    )

    result = agent.run(
        "Compare the weather in Tokyo and Sydney. "
        "Which city is warmer and by how many degrees?"
    )

    print(f"\n📊 Summary:")
    print(f"   Answer: {result.answer}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Tool calls: {result.total_tool_calls}")


def demo_action_history():
    """
    Demo 5: Examining the action history

    Shows how to inspect what the agent did.
    """
    print_section("Demo 5: Action History Analysis")

    config = AgentConfig(max_iterations=5, verbose=False)  # quiet mode
    agent = ReActAgent(
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        config=config,
    )

    result = agent.run("What's 15% of 200? And what's the weather in London?")

    print(f"\n📜 Full Action History:")
    print("-" * 40)
    for i, action in enumerate(result.actions, 1):
        print(f"\nStep {i}:")
        print(f"  💭 Thought: {action.thought[:100]}...")
        print(f"  🎯 Action: {action.action_str}")
        if action.observation:
            print(f"  👁️  Observation: {action.observation[:100]}...")

    print(f"\n📊 Summary:")
    print(f"   Final answer: {result.answer}")
    print(f"   Total iterations: {result.iterations}")
    print(f"   Total thoughts: {result.total_thoughts}")
    print(f"   Total tool calls: {result.total_tool_calls}")


def demo_timeout_handling():
    """
    Demo 6: What happens when agent hits max iterations

    Shows timeout behavior with a complex task.
    """
    print_section("Demo 6: Timeout Handling")

    # set very low max iterations to trigger timeout
    config = AgentConfig(max_iterations=2, verbose=True)
    agent = ReActAgent(
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        config=config,
    )

    result = agent.run(
        "Get weather for Tokyo, Paris, New York, London, and Sydney. "
        "Compare them all and tell me which is best for a beach vacation."
    )

    print(f"\n📊 Result:")
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error}")
    print(f"   Partial answer: {result.answer}")


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  ReAct Agent Demos - Reasoning and Acting Pattern")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Simple Query", "single-tool weather query"),
        ("2", "Multi-Step Reasoning", "multiple tools with reasoning"),
        ("3", "Knowledge + Calculation", "combining search and math"),
        ("4", "Compare Cities", "multiple API calls with comparison"),
        ("5", "Action History", "examining agent decision trace"),
        ("6", "Timeout Handling", "behavior when hitting max iterations"),
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
        '1': ('Simple Query', demo_simple_query),
        '2': ('Multi-Step Reasoning', demo_multi_step_reasoning),
        '3': ('Knowledge + Calculation', demo_knowledge_and_calculation),
        '4': ('Compare Cities', demo_compare_cities),
        '5': ('Action History', demo_action_history),
        '6': ('Timeout Handling', demo_timeout_handling),
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
    # check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  No API key found!")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        print("   Example: OPENAI_API_KEY=sk-...")
        return

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

# endregion


if __name__ == "__main__":
    main()
