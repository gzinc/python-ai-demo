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

from common.demo_menu import Demo, MenuRunner

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


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Basic Agent", "simple agent with tools", example_basic_agent),
    Demo("2", "ReAct Agent", "reasoning and action loop", example_react_agent),
    Demo("3", "Agent with Memory", "stateful agent execution", example_agent_with_memory),
    Demo("4", "Multi-Step Planning", "complex task breakdown", example_multi_step_planning),
]

# endregion


def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Examples")
    runner.run()
if __name__ == "__main__":
    main()
