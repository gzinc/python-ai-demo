"""
Agents & Tools - Practical Demonstrations (LangChain 1.0+ / LangGraph)

This module provides hands-on examples of LangChain agents and tools using
the modern LangGraph API (LangChain 1.0+).

Requires OPENAI_API_KEY in .env file.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.practical
"""

import os
from datetime import datetime
from inspect import cleandoc

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

# load environment variables
load_dotenv()


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def check_api_key() -> tuple[bool, str]:
    """check if OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, cleandoc("""
            ❌ OPENAI_API_KEY not found in environment variables

            To run these examples, set your API key:
              export OPENAI_API_KEY='your-key-here'

            Or create a .env file:
              OPENAI_API_KEY=your-key-here
        """)
    return True, f"✓ OpenAI API key configured ({api_key[:8]}...)"


# region Demo Functions


def demo_basic_tool_creation() -> None:
    """
    demonstrate basic tool creation with @tool decorator

    Basic Tool Creation Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         @tool Decorator: Simple Function → LangChain Tool   │
    │                                                             │
    │  Python Function:                                           │
    │     def calculator(expression: str) -> float:               │
    │         '''evaluate math expressions'''                     │
    │         return eval(expression)                             │
    │                                                             │
    │  Decorator Application:                                     │
    │     @tool                                                   │
    │     def calculator(expression: str) -> float:               │
    │         '''evaluate mathematical expressions'''             │
    │         return eval(expression)                             │
    │         ↓                                                   │
    │     Automatic Extraction:                                   │
    │       • name: "calculator" (from function name)             │
    │       • description: "evaluate mathematical expressions"    │
    │       • args_schema: {"expression": str} (from signature)   │
    │                                                             │
    │  Usage in Agent (LangGraph):                                │
    │     tools = [calculator]                                    │
    │     agent = create_agent(model=llm, tools=tools)            │
    │         │                                                   │
    │         ▼                                                   │
    │     Agent can now use calculator tool automatically!        │
    │                                                             │
    │  Tool Execution Flow:                                       │
    │     User Query → Agent → Tool Selection → Tool Execution    │
    │                    ↓                          ↓             │
    │                  "2+2"                   calculator("2+2")  │
    │                    ↓                          ↓             │
    │                Final Answer  ←  Tool Result (4.0)           │
    │                                                             │
    │  ✅ Benefit: Zero boilerplate (decorator handles metadata)  │
    │  ✅ Benefit: Type hints → automatic schema validation       │
    │  ⚠️  Caution: eval() is dangerous for untrusted input       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Basic Tool Creation with @tool")

    # define calculator tool
    @tool
    def calculator(expression: str) -> str:
        """evaluate mathematical expressions like '2+2' or '25*48+100'"""
        try:
            result = eval(expression)
            return str(float(result))
        except Exception as e:
            return f"Error: {str(e)}"

    print("1. Tool created with @tool decorator")
    print("\n2. Tool metadata (automatically extracted):")
    print(f"   • name: {calculator.name}")
    print(f"   • description: {calculator.description}")
    print(f"   • args: {list(calculator.args_schema.model_fields.keys())}")

    print("\n3. Direct tool execution (without agent):")
    result = calculator.invoke({"expression": "2 + 2"})
    print(f"   calculator('2 + 2') = {result}")

    print("\n4. How @tool works:")
    print("   • Function name → tool name")
    print("   • Docstring → tool description")
    print("   • Type hints → argument schema")


def demo_create_agent() -> None:
    """
    demonstrate create_agent basics with single tool (ReAct pattern)

    ReAct Agent Creation Pattern (LangChain 1.0+ / LangGraph):
    ┌─────────────────────────────────────────────────────────────┐
    │     create_agent: Model + Tools → CompiledGraph             │
    │                                                             │
    │  Step 1: Define Tools                                       │
    │     @tool                                                   │
    │     def get_time() -> str:                                  │
    │         ""get current time""                                │
    │         return datetime.now().strftime("%H:%M")             │
    │                                                             │
    │  Step 2: Create Agent (LangGraph simplified API)            │
    │     agent = create_agent(                                   │
    │         model=ChatOpenAI(model="gpt-4o-mini"),              │
    │         tools=[get_time]                                    │
    │     )                                                       │
    │     # Returns CompiledStateGraph, ready to use!             │
    │                                                             │
    │  Step 3: Execute Query                                      │
    │     result = agent.invoke({                                 │
    │         "messages": [HumanMessage(content="What time?")]    │
    │     })                                                      │
    │         ▼                                                   │
    │     Agent ReAct Loop:                                       │
    │       Thought: "I need the current time"                    │
    │       Action: get_time()                                    │
    │       Observation: "14:30"                                  │
    │       Thought: "I have the answer"                          │
    │       Final Answer: "It is 14:30"                           │
    │                                                             │
    │  Tool → Agent → LLM → Tool → Agent → Answer                 │
    │                                                             │
    │  ✅ Benefit: Simpler API (no AgentExecutor needed)          │
    │  ✅ Benefit: LLM decides when to use tools                  │
    │  ✅ Benefit: Built-in streaming and async support           │
    │  ⚠️  Caution: LLM calls cost money (watch iterations)       │
    │                                                             │
    │  ℹ️  Note: LangChain 1.0+ uses LangGraph for agents         │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: create_agent Basics - ReAct Pattern (LangChain 1.0+)")

    # define time tool
    @tool
    def get_current_time() -> str:
        """get the current time in HH:MM format"""
        return datetime.now().strftime("%H:%M")

    print("1. Tool metadata (automatically extracted):")
    print(f"   • name: {get_current_time.name}")
    print(f"   • description: {get_current_time.description}")

    # create llm and tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_current_time]

    # create agent (LangGraph simplified API)
    agent = create_agent(model=llm, tools=tools)

    print("\n2. Agent created (CompiledStateGraph)")
    print(f"   Type: {type(agent).__name__}")

    print("\n3. Asking agent: 'What time is it?'")
    print("   (Agent will use ReAct pattern to answer)")

    # invoke with messages format
    result = agent.invoke({"messages": [HumanMessage(content="What time is it?")]})

    # extract final answer from messages
    final_message = result["messages"][-1]
    print(f"\n4. Agent result: {final_message.content}")

    print("\n5. ReAct pattern flow:")
    print("   Thought → Action → Observation → Thought → Answer")

    print("\n6. LangChain 1.0+ / LangGraph simplifications:")
    print("   • No AgentExecutor needed")
    print("   • Direct invoke() on agent")
    print("   • Messages-based interface")
    print("   • Built-in tool calling")


def demo_multi_tool_agent() -> None:
    """
    demonstrate agent with multiple tools

    Multi-Tool Agent Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │            Agent with Multiple Tool Options                 │
    │                                                             │
    │  Tools Available:                                           │
    │     1. calculator(expression: str) → float                  │
    │     2. string_reverse(text: str) → str                      │
    │     3. string_upper(text: str) → str                        │
    │                                                             │
    │  Agent Decision Making:                                     │
    │     User: "What is 25 * 48 + 100?"                          │
    │       ↓                                                     │
    │     Agent analyzes query → identifies math task             │
    │       ↓                                                     │
    │     Selects calculator tool (not string tools)              │
    │       ↓                                                     │
    │     Executes: calculator("25*48+100")                       │
    │       ↓                                                     │
    │     Returns: 1300.0                                         │
    │                                                             │
    │  Tool Selection Criteria:                                   │
    │     • Tool name relevance                                   │
    │     • Tool description match                                │
    │     • Query context analysis                                │
    │     • Previous interaction history                          │
    │                                                             │
    │  ✅ Benefit: Agent chooses appropriate tool                 │
    │  ✅ Benefit: Multiple capabilities in one agent             │
    │  ✅ Benefit: Extensible (add more tools easily)             │
    │  ⚠️  Caution: Too many tools can confuse the agent          │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Multi-Tool Agent")

    # define multiple tools
    @tool
    def calculator(expression: str) -> str:
        """evaluate mathematical expressions"""
        try:
            return str(float(eval(expression)))
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def string_reverse(text: str) -> str:
        """reverse a string"""
        return text[::-1]

    @tool
    def string_upper(text: str) -> str:
        """convert string to uppercase"""
        return text.upper()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, string_reverse, string_upper]

    agent = create_agent(model=llm, tools=tools)

    print("\n1. Agent has 3 tools:")
    for t in tools:
        print(f"   • {t.name}: {t.description}")

    # test different queries
    test_queries = [
        "What is 25 * 48 + 100?",
        "Reverse the string 'hello'",
        "Convert 'python' to uppercase",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i+1}. Query: {query}")
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        final_answer = result["messages"][-1].content
        print(f"   Answer: {final_answer}")

    print("\n5. Tool selection insights:")
    print("   • Agent analyzes query semantics")
    print("   • Matches query to tool description")
    print("   • Executes most relevant tool")


def demo_custom_tool_class() -> None:
    """
    demonstrate custom tool using BaseTool class

    Custom Tool Class Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         BaseTool: Stateful Tools with Advanced Features     │
    │                                                             │
    │  @tool Decorator (Simple):                                  │
    │     @tool                                                   │
    │     def search(query: str) -> str:                          │
    │         return api_call(query)                              │
    │                                                             │
    │  ✅ Good for: Stateless functions                           │
    │  ❌ Limitations: No state, no complex initialization        │
    │                                                             │
    │  BaseTool Class (Advanced):                                 │
    │     class WeatherTool(BaseTool):                            │
    │         name: str = "weather"                               │
    │         description: str = "get weather"                    │
    │         api_key: str = Field(...)  # state!                 │
    │                                                             │
    │         def _run(self, city: str) -> str:                   │
    │             # use self.api_key                              │
    │             return fetch_weather(city, self.api_key)        │
    │                                                             │
    │  ✅ Good for: Stateful tools, API clients, DB connections   │
    │  ✅ Good for: Async support, error handling, retries        │
    │  ✅ Good for: Complex initialization logic                  │
    │                                                             │
    │  When to use each:                                          │
    │     @tool → Simple stateless functions                      │
    │     BaseTool → Stateful tools with configuration            │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Custom Tool Class with BaseTool")

    # define custom tool class
    class WeatherTool(BaseTool):
        name: str = "weather"
        description: str = "get current weather for a city. useful for weather queries"
        api_key: str = Field(default="demo-api-key", description="API key for weather service")

        def _run(self, city: str) -> str:
            """synchronous execution"""
            # simulate weather API call
            weather_data = {
                "tokyo": "25°C, sunny",
                "london": "15°C, rainy",
                "paris": "20°C, cloudy",
            }
            weather = weather_data.get(city.lower(), "Weather data not available")
            return f"{city}: {weather}"

        async def _arun(self, city: str) -> str:
            """async execution"""
            return self._run(city)

    print("1. Custom tool class created:")
    weather_tool = WeatherTool()
    print(f"   • name: {weather_tool.name}")
    print(f"   • description: {weather_tool.description}")
    print(f"   • api_key: {weather_tool.api_key}")

    print("\n2. Direct tool execution:")
    result = weather_tool._run("Tokyo")
    print(f"   {result}")

    print("\n3. Using custom tool with agent:")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(model=llm, tools=[weather_tool])

    query = "What's the weather in Paris?"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    final_answer = result["messages"][-1].content
    print(f"   Query: {query}")
    print(f"   Answer: {final_answer}")

    print("\n4. BaseTool advantages:")
    print("   • State management (api_key)")
    print("   • Sync and async support")
    print("   • Complex initialization")
    print("   • Error handling hooks")


def demo_error_handling() -> None:
    """
    demonstrate error handling in agent tools

    Error Handling Strategies:
    ┌─────────────────────────────────────────────────────────────┐
    │              Tool Error Handling Patterns                   │
    │                                                             │
    │  Strategy 1: Graceful Degradation                           │
    │     @tool                                                   │
    │     def search(query: str) -> str:                          │
    │         try:                                                │
    │             return api_call(query)                          │
    │         except APIError:                                    │
    │             return "Search unavailable"                     │
    │                                                             │
    │  Strategy 2: Error Reporting                                │
    │     @tool                                                   │
    │     def search(query: str) -> str:                          │
    │         try:                                                │
    │             return api_call(query)                          │
    │         except APIError as e:                               │
    │             return f"Error: {str(e)}"                       │
    │                                                             │
    │  Strategy 3: Fallback Tools                                 │
    │     tools = [primary_search, backup_search]                 │
    │     # Agent tries primary, falls back to backup             │
    │                                                             │
    │  Strategy 4: Validation                                     │
    │     @tool                                                   │
    │     def divide(a: float, b: float) -> str:                  │
    │         if b == 0:                                          │
    │             return "Cannot divide by zero"                  │
    │         return str(a / b)                                   │
    │                                                             │
    │  ✅ Benefit: Agent continues despite tool failures          │
    │  ✅ Benefit: Clear error messages guide agent               │
    │  ⚠️  Caution: Don't hide critical errors                    │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Error Handling in Tools")

    # tool with error handling
    @tool
    def safe_divide(numerator: float, denominator: float) -> str:
        """divide two numbers with error handling"""
        if denominator == 0:
            return "Error: Cannot divide by zero"
        try:
            result = numerator / denominator
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    print("1. Tool with error handling:")
    print(f"   • {safe_divide.name}: {safe_divide.description}")

    print("\n2. Testing error cases:")
    test_cases = [
        (10, 2, "Valid division"),
        (10, 0, "Division by zero"),
        (5, 0.5, "Valid division with float"),
    ]

    for num, den, desc in test_cases:
        result = safe_divide.invoke({"numerator": num, "denominator": den})
        print(f"   • {desc}: {num}/{den} = {result}")

    print("\n3. Using with agent:")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(model=llm, tools=[safe_divide])

    queries = [
        "What is 100 divided by 5?",
        "What is 10 divided by 0?",
    ]

    for query in queries:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        final_answer = result["messages"][-1].content
        print(f"\n   Query: {query}")
        print(f"   Answer: {final_answer}")

    print("\n4. Error handling best practices:")
    print("   • Validate inputs before execution")
    print("   • Return descriptive error messages")
    print("   • Don't crash the agent")
    print("   • Log errors for debugging")


def demo_web_search_agent() -> None:
    """
    demonstrate agent with web search capability

    Web Search Agent Pattern:
    ┌───────────────────────────────────────────────────────────────┐
    │         Agent with Real-World Information Access              │
    │                                                               │
    │  Pre-built Tool: DuckDuckGoSearchRun                          │
    │     from langchain_community.tools import DuckDuckGoSearchRun │
    │     search = DuckDuckGoSearchRun()                            │
    │                                                               │
    │  Agent Flow:                                                  │
    │     User: "What's the latest news about Python?"              │
    │       ↓                                                       │
    │     Agent: "I need current information"                       │
    │       ↓                                                       │
    │     Action: search("Python news 2024")                        │
    │       ↓                                                       │
    │     Observation: [search results]                             │
    │       ↓                                                       │
    │     Agent synthesizes answer from results                     │
    │                                                               │
    │  Use Cases:                                                   │
    │     • Current events and news                                 │
    │     • Real-time information                                   │
    │     • Fact-checking                                           │
    │     • Research and discovery                                  │
    │                                                               │
    │  ✅ Benefit: Access to current information                    │
    │  ✅ Benefit: Grounded in real data                            │
    │  ⚠️  Caution: Search results may be noisy                     │
    │  ⚠️  Caution: API rate limits apply                           │
    └───────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 6: Web Search Agent")

    print("1. Creating web search tool...")
    search = DuckDuckGoSearchRun()
    print(f"   • Tool: {search.name}")
    print(f"   • Description: {search.description}")

    print("\n2. Creating agent with search capability...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(model=llm, tools=[search])

    print("\n3. Testing search queries:")
    queries = [
        "What is LangChain?",
        "What is the capital of France?",
    ]

    for query in queries:
        print(f"\n   Query: {query}")
        try:
            result = agent.invoke({"messages": [HumanMessage(content=query)]})
            final_answer = result["messages"][-1].content
            # truncate long answers
            if len(final_answer) > 200:
                final_answer = final_answer[:200] + "..."
            print(f"   Answer: {final_answer}")
        except Exception as e:
            print(f"   Error: {str(e)}")

    print("\n4. Web search agent insights:")
    print("   • Provides real-time information")
    print("   • Complements LLM's training data")
    print("   • Useful for current events")
    print("   • Rate limits may apply")


def demo_phase4_comparison() -> None:
    """
    compare Phase 4 custom agents with LangChain agents

    Phase 4 vs LangChain Comparison:
    ┌─────────────────────────────────────────────────────────────┐
    │        Custom ReActAgent (Phase 4)                          │
    │                                                             │
    │  class ReActAgent:                                          │
    │      def __init__(self, registry: ToolRegistry):            │
    │          self.registry = registry                           │
    │          self.max_iterations = 5                            │
    │                                                             │
    │      def run(self, task: str) -> str:                       │
    │          for i in range(self.max_iterations):               │
    │              # 1. Generate thought and action               │
    │              response = self.llm.generate(prompt)           │
    │              thought, action, action_input = self.parse()   │
    │                                                             │
    │              # 2. Execute tool                              │
    │              tool = self.registry.get(action)               │
    │              observation = tool.execute(action_input)       │
    │                                                             │
    │              # 3. Check if done                             │
    │              if action == "Final Answer":                   │
    │                  return observation                         │
    │          return "Max iterations reached"                    │
    │                                                             │
    │  ✅ Learning: Full control, understand internals            │
    │  ❌ Production: Manual error handling, no streaming         │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │        LangChain / LangGraph Agent                          │
    │                                                             │
    │  @tool                                                      │
    │  def calculator(expr: str) -> str:                          │
    │      return str(eval(expr))                                 │
    │                                                             │
    │  agent = create_agent(                                      │
    │      model=ChatOpenAI(model="gpt-4o-mini"),                 │
    │      tools=[calculator]                                     │
    │  )                                                          │
    │  result = agent.invoke({                                    │
    │      "messages": [HumanMessage(content="2+2")]              │
    │  })                                                         │
    │                                                             │
    │  ✅ Production: Built-in features, robust, tested           │
    │  ✅ Concise: Less boilerplate code                          │
    │  ❌ Learning: Internal logic abstracted                     │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                When to Use Each                             │
    │                                                             │
    │  Phase 4 Custom:                                            │
    │    • Learning agent fundamentals                            │
    │    • Understanding ReAct pattern                            │
    │    • Prototyping new patterns                               │
    │    • Educational purposes                                   │
    │                                                             │
    │  LangChain / LangGraph:                                     │
    │    • Production applications                                │
    │    • Rapid development                                      │
    │    • Need advanced features                                 │
    │    • Standard use cases                                     │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 7: Phase 4 vs LangChain Comparison")

    print(cleandoc("""
        Your Phase 4 ReActAgent (Custom Implementation):
        ================================================

        class ReActAgent:
            def __init__(self, registry: ToolRegistry):
                self.registry = registry
                self.max_iterations = 5

            def run(self, task: str) -> str:
                for i in range(self.max_iterations):
                    # 1. Generate thought and action
                    response = self.llm.generate(prompt)
                    thought, action, action_input = self.parse(response)

                    # 2. Execute tool
                    tool = self.registry.get(action)
                    observation = tool.execute(action_input)

                    # 3. Check if done
                    if action == "Final Answer":
                        return observation

                return "Max iterations reached"

        ✅ Full control: Every step is explicit
        ✅ Learning: Understand agent internals
        ✅ Customization: Easy to modify logic
        ❌ Manual work: Error handling, retries, memory
        ❌ Missing: Streaming, async, advanced features
    """))

    print(cleandoc("""

        LangChain / LangGraph Agent (Framework):
        =========================================

        @tool
        def calculator(expr: str) -> str:
            return str(eval(expr))

        agent = create_agent(
            model=ChatOpenAI(model="gpt-4o-mini"),
            tools=[calculator]
        )
        result = agent.invoke({
            "messages": [HumanMessage(content="Calculate 2+2")]
        })

        ✅ Concise: ~10 lines of code
        ✅ Robust: Built-in error handling, retries
        ✅ Features: Streaming, async, memory, callbacks
        ✅ Production: Battle-tested, maintained
        ❌ Abstraction: Internal logic hidden
        ❌ Less control: Framework makes decisions
    """))

    print("\n\nWhen to Use Each:")
    print("-" * 70)
    print("\nPhase 4 Custom Agent:")
    print("  • Learning how agents work internally")
    print("  • Understanding ReAct pattern step-by-step")
    print("  • Prototyping new agent patterns")
    print("  • Educational and research purposes")

    print("\nLangChain / LangGraph Agent:")
    print("  • Production applications")
    print("  • Rapid development and deployment")
    print("  • Standard use cases and patterns")
    print("  • Need advanced features out-of-the-box")

    print("\n\nRecommendation:")
    print("-" * 70)
    print("  • Learn with Phase 4 custom implementation")
    print("  • Build with LangChain / LangGraph for production")
    print("  • Understand both to make informed choices")


# endregion


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Agents & Tools - Practical Examples")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Basic Tool Creation", "simple @tool decorator pattern"),
        ("2", "Create Agent", "LangGraph agent with single tool"),
        ("3", "Multi-Tool Agent", "agent with multiple capabilities"),
        ("4", "Custom Tool Class", "advanced tool with BaseTool"),
        ("5", "Error Handling", "graceful tool failure recovery"),
        ("6", "Web Search Agent", "real-world search integration"),
        ("7", "Phase 4 Comparison", "custom vs LangChain agents"),
    ]

    for num, name, desc in demos:
        api_marker = "🔑"
        print(f"  {api_marker} [{num}] {name}")
        print(f"      {desc}")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 70)


def run_selected_demos(selections: str) -> bool:
    """run selected demos based on user input"""
    selections = selections.lower().strip()

    if selections == 'q':
        return False

    demo_map = {
        '1': ("Basic Tool Creation", demo_basic_tool_creation),
        '2': ("Create Agent", demo_create_agent),
        '3': ("Multi-Tool Agent", demo_multi_tool_agent),
        '4': ("Custom Tool Class", demo_custom_tool_class),
        '5': ("Error Handling", demo_error_handling),
        '6': ("Web Search Agent", demo_web_search_agent),
        '7': ("Phase 4 Comparison", demo_phase4_comparison),
    }

    if selections == 'a':
        # run all demos
        for name, demo_func in demo_map.values():
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # parse comma-separated selections
        selected = [s.strip() for s in selections.split(',')]
        for sel in selected:
            if sel in demo_map:
                name, demo_func = demo_map[sel]
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Error in {name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Invalid selection: {sel}")

    return True


def main() -> None:
    """run demonstrations with interactive menu"""
    # check api key
    api_key_ok, message = check_api_key()
    if not api_key_ok:
        print(message)
        return

    print(message)
    print("\n" + "=" * 70)
    print("  Agents & Tools - Practical Examples")
    print("  Using LangChain 1.0+ / LangGraph API")
    print("=" * 70)

    try:
        while True:
            show_menu()
            selection = input("\nSelect demos to run (comma-separated) or 'a' for all: ").strip()

            if not selection:
                continue

            if not run_selected_demos(selection):
                break

            print("\n" + "=" * 70)
            print("  Demos complete!")
            print("=" * 70)

            # pause before showing menu again
            try:
                input("\n⏸️  Press Enter to continue...")
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

        print("\n" + "=" * 70)
        print("  Thanks for exploring agents and tools!")
        print("\nNext steps:")
        print("  • Try modifying the examples")
        print("  • Create your own tools")
        print("  • Build a multi-agent system")
        print("  • Explore exercises in README.md")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
