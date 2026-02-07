"""
Agents & Tools - Conceptual Demonstrations

This module explains agent and tool concepts without requiring API keys.
Demonstrates patterns, architectures, and decision-making flows.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.concepts
"""

from inspect import cleandoc

from common.demo_menu import Demo, MenuRunner


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


# region Agent Architecture Concepts


def demo_agent_vs_chain() -> None:
    """demonstrate conceptual difference between agents and chains"""
    print_section("Agent vs Chain: Conceptual Difference")

    print(cleandoc("""
        CHAIN: Fixed sequence (deterministic flow)
        ┌────────────────────────────────────────────────────────┐
        │  User Input                                            │
        │      ↓                                                 │
        │  Step 1: Summarize (always happens)                    │
        │      ↓                                                 │
        │  Step 2: Translate (always happens)                    │
        │      ↓                                                 │
        │  Step 3: Format (always happens)                       │
        │      ↓                                                 │
        │  Final Output                                          │
        │                                                        │
        │  ✅ Predictable, fast                                  │
        │  ❌ Not adaptive to input                              │
        └────────────────────────────────────────────────────────┘

        AGENT: Dynamic reasoning (adaptive flow)
        ┌────────────────────────────────────────────────────────┐
        │  User: "What's weather in Tokyo? Convert 25°C to F"    │
        │      ↓                                                 │
        │  ┌──────────────────────────────────┐                  │
        │  │ Agent Reasoning Loop:            │                  │
        │  │                                  │                  │
        │  │ Thought: Need weather data       │                  │
        │  │ Action: weather_tool("Tokyo")    │                  │
        │  │ Observation: 25°C, sunny         │                  │
        │  │                                  │                  │
        │  │ Thought: Need temperature conv   │                  │
        │  │ Action: calculator("25*9/5+32")  │                  │
        │  │ Observation: 77                  │                  │
        │  │                                  │                  │
        │  │ Thought: Have all information    │                  │
        │  │ Action: Final Answer             │                  │
        │  └──────────────────────────────────┘                  │
        │      ↓                                                 │
        │  "Tokyo: 25°C (77°F), sunny"                           │
        │                                                        │
        │  ✅ Adaptive to input complexity                       │
        │  ❌ Slower, less predictable                           │
        └────────────────────────────────────────────────────────┘

        Key Insight: Agents decide WHICH tools to use and WHEN.
        Chains always execute the same sequence.
    """))


def demo_react_pattern() -> None:
    """demonstrate ReAct (Reasoning + Acting) pattern"""
    print_section("ReAct Pattern: How Agents Think")

    print(cleandoc("""
        ReAct = Reason (think) + Act (use tools)

        Example: "Who won the 2023 FIFA World Cup?"

        ┌────────────────────────────────────────────────────────┐
        │  Iteration 1:                                          │
        │                                                        │
        │  Thought: I need current sports information. My        │
        │           training data ends in 2023, so I should      │
        │           search the web for recent results.           │
        │                                                        │
        │  Action: web_search                                    │
        │  Action Input: "2023 FIFA World Cup winner"            │
        │                                                        │
        │  Observation: "Argentina won the 2023 FIFA World Cup,  │
        │               defeating France 4-2 on penalties after  │
        │               a 3-3 draw"                              │
        ├────────────────────────────────────────────────────────┤
        │  Iteration 2:                                          │
        │                                                        │
        │  Thought: I have the answer from the search results.   │
        │           Argentina won the 2023 World Cup.            │
        │                                                        │
        │  Action: Final Answer                                  │
        │  Action Input: "Argentina won the 2023 FIFA World Cup, │
        │                defeating France in a dramatic final"   │
        └────────────────────────────────────────────────────────┘

        Pattern Components:
        1. Thought: LLM reasoning about what to do next
        2. Action: Tool to use (or "Final Answer" to finish)
        3. Action Input: Parameters for the tool
        4. Observation: Result from tool execution
        5. Repeat until "Final Answer"

        Why "ReAct"?
        - Reasoning: LLM thinks before acting
        - Acting: LLM uses tools to get information
        - Combines symbolic reasoning with external actions
    """))


def demo_tool_selection() -> None:
    """demonstrate how agents select tools"""
    print_section("Tool Selection: How LLM Chooses Tools")

    print(cleandoc("""
        Agent has access to multiple tools:

        Available Tools:
        1. calculator: "evaluate mathematical expressions"
        2. web_search: "search the web for current information"
        3. weather: "get current weather for a city"
        4. wikipedia: "look up facts and historical information"

        ┌────────────────────────────────────────────────────────┐
        │  Query: "What's 25 * 48 + 100?"                        │
        │                                                        │
        │  LLM Decision Process:                                 │
        │    • Is this math? YES                                 │
        │    • calculator: "evaluate mathematical expressions"   │
        │    • ✓ MATCH - description mentions math               │
        │                                                        │
        │  Selected: calculator("25 * 48 + 100")                 │
        │  → Result: 1300                                        │
        └────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────────┐
        │  Query: "What's the weather in Tokyo?"                 │
        │                                                        │
        │  LLM Decision Process:                                 │
        │    • Need current weather? YES                         │
        │    • weather: "get current weather for a city"         │
        │    • ✓ MATCH - description mentions weather            │
        │                                                        │
        │  Selected: weather("Tokyo")                            │
        │  → Result: "25°C, sunny"                               │
        └────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────────┐
        │  Query: "Who invented the telephone?"                  │
        │                                                        │
        │  LLM Decision Process:                                 │
        │    • Historical fact? YES                              │
        │    • wikipedia: "look up facts and historical info"    │
        │    • ✓ MATCH - historical question                     │
        │                                                        │
        │  Selected: wikipedia("telephone inventor")             │
        │  → Result: "Alexander Graham Bell"                     │
        └────────────────────────────────────────────────────────┘

        Key Insight: Tool descriptions are CRITICAL
        - LLM reads tool descriptions to make decisions
        - Better descriptions → better tool selection
        - Vague descriptions confuse the agent
    """))


def demo_tool_description_quality() -> None:
    """demonstrate impact of tool description quality"""
    print_section("Tool Description Quality Impact")

    print(cleandoc("""
        BAD Tool Descriptions (Vague, confusing):
        ┌────────────────────────────────────────────────────────┐
        │  Tool 1:                                               │
        │    name: "process"                                     │
        │    description: "process data"                         │
        │    ❌ What kind of processing?                         │
        │    ❌ What data formats?                               │
        │                                                        │
        │  Tool 2:                                               │
        │    name: "fetch"                                       │
        │    description: "fetch information"                    │
        │    ❌ From where?                                      │
        │    ❌ What kind of information?                        │
        │                                                        │
        │  Result: Agent confused, selects wrong tools           │
        └────────────────────────────────────────────────────────┘

        GOOD Tool Descriptions (Specific, actionable):
        ┌────────────────────────────────────────────────────────┐
        │  Tool 1:                                               │
        │    name: "json_parser"                                 │
        │    description: "parse JSON strings into Python dicts. │
        │                 handles nested objects and arrays.     │
        │                 useful for API responses"              │
        │    ✓ Clear purpose                                     │
        │    ✓ Specific input/output                             │
        │    ✓ Use case hints                                    │
        │                                                        │
        │  Tool 2:                                               │
        │    name: "web_search"                                  │
        │    description: "search the web for current info using │
        │                 DuckDuckGo. returns top 5 results.     │
        │                 best for recent events and news"       │
        │    ✓ Data source clear                                 │
        │    ✓ Result format specified                           │
        │    ✓ Best use cases mentioned                          │
        │                                                        │
        │  Result: Agent makes correct tool choices              │
        └────────────────────────────────────────────────────────┘

        Best Practices:
        1. Describe WHAT the tool does (specific action)
        2. Mention WHEN to use it (use cases)
        3. Specify INPUT format if constrained
        4. Indicate OUTPUT format if relevant
        5. Keep description under 100 words
    """))


def demo_agent_memory_integration() -> None:
    """demonstrate how agents integrate with memory"""
    print_section("Agent + Memory Integration")

    print(cleandoc("""
        Agent WITHOUT Memory (stateless):
        ┌────────────────────────────────────────────────────────┐
        │  Turn 1:                                               │
        │    User: "My name is Alice"                            │
        │    Agent: "Nice to meet you, Alice!"                   │
        │                                                        │
        │  Turn 2:                                               │
        │    User: "What's my name?"                             │
        │    Agent: "I don't know your name"                     │
        │    ❌ Lost context from Turn 1                         │
        └────────────────────────────────────────────────────────┘

        Agent WITH Memory (stateful):
        ┌────────────────────────────────────────────────────────┐
        │  Memory Store: []                                      │
        │                                                        │
        │  Turn 1:                                               │
        │    User: "My name is Alice"                            │
        │    Agent: "Nice to meet you, Alice!"                   │
        │    Memory: [User: "My name is Alice",                  │
        │             AI: "Nice to meet you, Alice!"]            │
        │                                                        │
        │  Turn 2:                                               │
        │    User: "What's my name?"                             │
        │    Memory → Agent: [previous conversation]             │
        │    Agent: "Your name is Alice"                         │
        │    ✓ Context preserved                                 │
        └────────────────────────────────────────────────────────┘

        Memory Flow with Tools:
        ┌────────────────────────────────────────────────────────┐
        │  Turn 1:                                               │
        │    User: "Search for Python tutorials"                 │
        │    Agent → web_search("Python tutorials")              │
        │    Result: [tutorial links]                            │
        │    Memory: Stores query + results                      │
        │                                                        │
        │  Turn 2:                                               │
        │    User: "Which one is best for beginners?"            │
        │    Memory → Agent: [previous search results]           │
        │    Agent: Analyzes previous results in context         │
        │    Response: "The Real Python tutorial is best..."     │
        │    ✓ Builds on previous tool usage                     │
        └────────────────────────────────────────────────────────┘

        Key Insight: Memory + Tools = Powerful conversations
        - Agent remembers what tools it used
        - Agent remembers tool results
        - User can reference previous actions
    """))


def demo_error_handling_strategies() -> None:
    """demonstrate agent error handling approaches"""
    print_section("Agent Error Handling Strategies")

    print(cleandoc("""
        Strategy 1: Graceful Degradation
        ┌────────────────────────────────────────────────────────┐
        │  Tool: weather_api                                     │
        │                                                        │
        │  Execution:                                            │
        │    weather_api("Tokyo")                                │
        │    → API Error: Rate limit exceeded                    │
        │                                                        │
        │  Tool Response:                                        │
        │    "Error: Weather API rate limit exceeded.            │
        │     Please try again in 1 minute or use a              │
        │     different city"                                    │
        │                                                        │
        │  Agent Reaction:                                       │
        │    Thought: API failed, need alternative approach      │
        │    Action: web_search("Tokyo weather")                 │
        │    ✓ Adapts to failure                                 │
        └────────────────────────────────────────────────────────┘

        Strategy 2: Retry with Backoff
        ┌────────────────────────────────────────────────────────┐
        │  Attempt 1: Call API → Timeout                         │
        │  Wait: 1 second                                        │
        │                                                        │
        │  Attempt 2: Call API → Timeout                         │
        │  Wait: 2 seconds                                       │
        │                                                        │
        │  Attempt 3: Call API → Success!                        │
        │  ✓ Transient error recovered                           │
        └────────────────────────────────────────────────────────┘

        Strategy 3: Fallback Tools
        ┌────────────────────────────────────────────────────────┐
        │  Primary Tool: paid_search_api (high quality)          │
        │    → Error: API key invalid                            │
        │                                                        │
        │  Fallback Tool: free_search (lower quality)            │
        │    → Success: Returns basic results                    │
        │    ✓ Still provides value                              │
        └────────────────────────────────────────────────────────┘

        Strategy 4: Max Iterations Protection
        ┌────────────────────────────────────────────────────────┐
        │  Iteration 1: web_search("topic")                      │
        │  Iteration 2: wikipedia("topic")                       │
        │  Iteration 3: calculator("...")                        │
        │  Iteration 4: web_search("more specific")              │
        │  Iteration 5: Final Answer                             │
        │  → Max iterations (5) reached, forced to conclude      │
        │  ✓ Prevents infinite loops                             │
        └────────────────────────────────────────────────────────┘

        Best Practices:
        1. Set max_iterations (prevent infinite loops)
        2. Set max_execution_time (timeout protection)
        3. Return helpful error messages from tools
        4. Enable handle_parsing_errors=True
        5. Use try-except in tool implementations
    """))


def demo_agent_decision_tree() -> None:
    """demonstrate agent decision-making as a tree"""
    print_section("Agent Decision Tree")

    print(cleandoc("""
        Query: "What's the capital of France and its population?"

        ┌─────────────────────────────────────────────────────────┐
        │                        START                            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Thought: Need two pieces of info │            │
        │        │  1. Capital of France             │            │
        │        │  2. Population of that capital    │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Action: wikipedia                │            │
        │        │  Input: "France capital"          │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Observation: "Paris"             │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Thought: Got capital (Paris).    │            │
        │        │  Now need population.             │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Action: web_search               │            │
        │        │  Input: "Paris population 2024"   │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Observation: "2.1 million"       │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Thought: Have both pieces.       │            │
        │        │  Ready to answer.                 │            │
        │        └─────────────────┬─────────────────┘            │
        │                          │                              │
        │        ┌─────────────────┴─────────────────┐            │
        │        │  Action: Final Answer             │            │
        │        │  Input: "Paris, population 2.1M"  │            │
        │        └───────────────────────────────────┘            │
        └─────────────────────────────────────────────────────────┘

        Key Observations:
        1. Agent breaks complex queries into sub-tasks
        2. Each tool result informs next decision
        3. Decision tree adapts based on observations
        4. No fixed sequence - dynamic reasoning
    """))


# endregion



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Agent vs Chain", "agent vs chain", demo_agent_vs_chain),
    Demo("2", "ReAct Pattern", "react pattern", demo_react_pattern),
    Demo("3", "Tool Selection", "tool selection", demo_tool_selection),
    Demo("4", "Tool Description Quality", "tool description quality", demo_tool_description_quality),
    Demo("5", "Agent Memory Integration", "agent memory integration", demo_agent_memory_integration),
    Demo("6", "Error Handling", "error handling", demo_error_handling_strategies),
    Demo("7", "Agent Decision Tree", "agent decision tree", demo_agent_decision_tree),
]

# endregion

def main() -> None:
    """run demonstrations with interactive menu"""
    print("\n" + "=" * 70)
    print("  Agents & Tools - Conceptual Understanding")
    print("  No API key required - demonstrates patterns only")
    print("=" * 70)

    
    runner = MenuRunner(DEMOS, title="TODO: Add title")
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
