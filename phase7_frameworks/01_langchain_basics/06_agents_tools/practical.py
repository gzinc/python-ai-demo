"""
Agents & Tools - Practical Demonstrations

This module provides hands-on examples of LangChain agents and tools.
Requires OPENAI_API_KEY in .env file.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.practical
"""

import os
from inspect import cleandoc
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    │  Usage in Agent:                                            │
    │     tools = [calculator]                                    │
    │     agent = create_react_agent(llm, tools, prompt)          │
    │         │                                                   │
    │         ▼                                                   │
    │     Agent can now use calculator tool automatically!        │
    │                                                             │
    │  Tool Execution Flow:                                       │
    │     User: "What's 25 * 48 + 100?"                           │
    │         ↓                                                   │
    │     Agent: Thought: This is math                            │
    │            Action: calculator                               │
    │            Action Input: "25 * 48 + 100"                    │
    │         ↓                                                   │
    │     Tool: calculator("25 * 48 + 100")                       │
    │         ↓                                                   │
    │     Result: 1300.0                                          │
    │         ↓                                                   │
    │     Agent: Final Answer: "The result is 1300"               │
    │                                                             │
    │  ✅ Benefit: Zero boilerplate (decorator handles metadata)  │
    │  ✅ Benefit: Type hints → automatic schema validation       │
    │  ✅ Benefit: Docstring → tool description                   │
    │  ⚠️  Caution: eval() is dangerous for untrusted input       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Basic Tool Creation with @tool")

    # define simple calculator tool
    @tool
    def calculator(expression: str) -> float:
        """evaluate mathematical expressions like '2+2' or '25*48+100'"""
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"

    print("\n1. Tool created with @tool decorator")
    print(f"   Name: {calculator.name}")
    print(f"   Description: {calculator.description}")
    print(f"   Args: {calculator.args}")

    print("\n2. Test tool directly:")
    result = calculator.invoke({"expression": "25 * 48 + 100"})
    print(f"   calculator('25 * 48 + 100') = {result}")

    print("\n3. Tool metadata (auto-extracted):")
    print(f"   • Function name → tool name")
    print(f"   • Docstring → tool description")
    print(f"   • Type hints → argument schema")


def demo_create_react_agent() -> None:
    """
    demonstrate create_react_agent basics with single tool

    ReAct Agent Creation Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │     create_react_agent: LLM + Tools + Prompt → Agent        │
    │                                                             │
    │  Step 1: Define Tools                                       │
    │     @tool                                                   │
    │     def get_time() -> str:                                  │
    │         ""get current time""                                │
    │         return datetime.now().strftime("%H:%M:%S")          │
    │                                                             │
    │  Step 2: Create ReAct Prompt                                │
    │     prompt = ChatPromptTemplate([                           │
    │         ("system", "You are helpful assistant.              │
    │                     Tools: {tools}                          │
    │                     Tool names: {tool_names}"),             │
    │         ("user", "{input}"),                                │
    │         ("assistant", "{agent_scratchpad}")                 │
    │     ])                                                      │
    │                                                             │
    │  Step 3: Create Agent                                       │
    │     agent = create_react_agent(                             │
    │         llm=ChatOpenAI(model="gpt-4o-mini"),                │
    │         tools=[get_time],                                   │
    │         prompt=prompt                                       │
    │     )                                                       │
    │                                                             │
    │  Step 4: Wrap in Executor                                   │
    │     executor = AgentExecutor(                               │
    │         agent=agent,                                        │
    │         tools=[get_time],                                   │
    │         verbose=True  # show reasoning steps                │
    │     )                                                       │
    │                                                             │
    │  Execution Flow:                                            │
    │     User: "What time is it?"                                │
    │         ↓                                                   │
    │     ┌───────────────────────────────────┐                   │
    │     │ Agent Reasoning Loop:             │                   │
    │     │                                   │                   │
    │     │ Thought: Need current time        │                   │
    │     │ Action: get_time                  │                   │
    │     │ Action Input: (no params)         │                   │
    │     │     ↓                             │                   │
    │     │ Observation: "14:30:45"           │                   │
    │     │     ↓                             │                   │
    │     │ Thought: I have the time          │                   │
    │     │ Action: Final Answer              │                   │
    │     │ Action Input: "It's 14:30:45"     │                   │
    │     └───────────────────────────────────┘                   │
    │         ↓                                                   │
    │     Output: "It's 14:30:45"                                 │
    │                                                             │
    │  ✅ Benefit: Built-in ReAct reasoning loop                  │
    │  ✅ Benefit: Automatic tool selection and execution         │
    │  ✅ Benefit: Verbose mode shows agent thinking              │
    │  ⚠️  Caution: LLM calls cost money (watch iterations)       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: create_react_agent Basics")

    from datetime import datetime

    # define time tool
    @tool
    def get_current_time() -> str:
        """get the current time in HH:MM:SS format"""
        return datetime.now().strftime("%H:%M:%S")

    # create react prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", cleandoc("""
            You are a helpful assistant with access to tools.

            Available tools:
            {tools}

            Tool names: {tool_names}

            Use the ReAct format:
            Thought: think about what to do
            Action: tool to use (one of {tool_names})
            Action Input: input for the tool
            Observation: result from the tool
            ... repeat as needed ...
            Thought: I now know the answer
            Final Answer: the final response
        """)),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    # create llm and tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_current_time]

    # create agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # create executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # show reasoning steps
        max_iterations=3,
    )

    print("\n1. Agent created with:")
    print(f"   LLM: gpt-4o-mini")
    print(f"   Tools: {[t.name for t in tools]}")

    print("\n2. Asking: 'What time is it?'")
    result = executor.invoke({"input": "What time is it?"})
    print(f"\n3. Final answer: {result['output']}")


def demo_multi_tool_agent() -> None:
    """
    demonstrate agent with multiple tools

    Multi-Tool Agent Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │    Multi-Tool Agent: LLM Selects from Tool Arsenal          │
    │                                                             │
    │  Tool Arsenal:                                              │
    │     1. calculator: "evaluate math expressions"              │
    │     2. string_reverse: "reverse text strings"               │
    │     3. string_upper: "convert text to uppercase"            │
    │                                                             │
    │  Complex Query: "Reverse 'hello' and uppercase it,          │
    │                  then calculate 5*5"                        │
    │                                                             │
    │  Agent Reasoning:                                           │
    │     ┌─────────────────────────────────────┐                 │
    │     │ Iteration 1:                        │                 │
    │     │   Thought: Need to reverse "hello"  │                 │
    │     │   Action: string_reverse            │                 │
    │     │   Input: "hello"                    │                 │
    │     │   Observation: "olleh"              │                 │
    │     ├─────────────────────────────────────┤                 │
    │     │ Iteration 2:                        │                 │
    │     │   Thought: Now uppercase "olleh"    │                 │
    │     │   Action: string_upper              │                 │
    │     │   Input: "olleh"                    │                 │
    │     │   Observation: "OLLEH"              │                 │
    │     ├─────────────────────────────────────┤                 │
    │     │ Iteration 3:                        │                 │
    │     │   Thought: Now calculate 5*5        │                 │
    │     │   Action: calculator                │                 │
    │     │   Input: "5*5"                      │                 │
    │     │   Observation: 25.0                 │                 │
    │     ├─────────────────────────────────────┤                 │
    │     │ Iteration 4:                        │                 │
    │     │   Thought: Have all results         │                 │
    │     │   Action: Final Answer              │                 │
    │     │   Input: "OLLEH and 25"             │                 │
    │     └─────────────────────────────────────┘                 │
    │                                                             │
    │  Tool Selection Strategy:                                   │
    │     • LLM reads ALL tool descriptions                       │
    │     • Matches task to best tool                             │
    │     • Executes sequentially as needed                       │
    │     • Chains results together                               │
    │                                                             │
    │  ✅ Benefit: Agent handles complex multi-step tasks         │
    │  ✅ Benefit: Automatic tool sequencing                      │
    │  ✅ Benefit: No hardcoded workflow required                 │
    │  ⚠️  Caution: More tools = more tokens in prompt            │
    │  ⚠️  Caution: Similar tools can confuse LLM selection       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Multi-Tool Agent")

    # define multiple tools
    @tool
    def calculator(expression: str) -> float:
        """evaluate mathematical expressions"""
        return eval(expression)

    @tool
    def string_reverse(text: str) -> str:
        """reverse a string. useful for text manipulation"""
        return text[::-1]

    @tool
    def string_upper(text: str) -> str:
        """convert text to uppercase letters"""
        return text.upper()

    # create agent with multiple tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, string_reverse, string_upper]

    prompt = ChatPromptTemplate.from_messages([
        ("system", cleandoc("""
            You are a helpful assistant with access to tools.
            Tools: {tools}
            Tool names: {tool_names}

            Use ReAct format to solve tasks step by step.
        """)),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

    print("\n1. Agent has 3 tools:")
    for t in tools:
        print(f"   • {t.name}: {t.description}")

    print("\n2. Complex task: 'Reverse hello, uppercase it, then calculate 5*5'")
    result = executor.invoke({
        "input": "Reverse the word 'hello', convert it to uppercase, and calculate 5 * 5"
    })
    print(f"\n3. Result: {result['output']}")
    print("\n4. Notice: Agent selected appropriate tools in sequence")


def demo_agent_with_memory() -> None:
    """
    demonstrate agent with conversational memory

    Agent + Memory Integration Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      Agent with Memory: Stateful Conversations + Tools      │
    │                                                             │
    │  Turn 1: "My favorite number is 42"                         │
    │     Memory Store: [] (empty)                                │
    │         ↓                                                   │
    │     Agent: "I'll remember that your favorite number is 42!" │
    │         ↓                                                   │
    │     Memory Store: [                                         │
    │         User: "My favorite number is 42",                   │
    │         AI: "I'll remember..."                              │
    │     ]                                                       │
    │                                                             │
    │  Turn 2: "Calculate my favorite number * 2"                 │
    │     Memory → Agent: [previous conversation]                 │
    │         ↓                                                   │
    │     Agent Reasoning:                                        │
    │       Thought: User's favorite is 42 (from memory)          │
    │       Thought: Need to calculate 42 * 2                     │
    │       Action: calculator                                    │
    │       Input: "42 * 2"                                       │
    │         ↓                                                   │
    │     Tool: calculator("42 * 2") → 84.0                       │
    │         ↓                                                   │
    │     Agent: Final Answer: "84"                               │
    │         ↓                                                   │
    │     Memory Store: [                                         │
    │         User: "My favorite number is 42",                   │
    │         AI: "I'll remember...",                             │
    │         User: "Calculate my favorite * 2",                  │
    │         AI: "84"                                            │
    │     ]                                                       │
    │                                                             │
    │  Key Flow:                                                  │
    │     1. Memory provides context to agent                     │
    │     2. Agent uses context + tools to solve task             │
    │     3. Results stored back in memory                        │
    │     4. Next turn has full conversational context            │
    │                                                             │
    │  ✅ Benefit: Agent remembers preferences and past actions   │
    │  ✅ Benefit: Natural multi-turn conversations with tools    │
    │  ✅ Benefit: Can reference previous tool results            │
    │  ⚠️  Caution: Memory grows unbounded (use window/summary)   │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Agent with Memory")

    # define calculator tool
    @tool
    def calculator(expression: str) -> float:
        """evaluate mathematical expressions"""
        return eval(expression)

    # create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", cleandoc("""
            You are a helpful assistant with access to tools.
            Tools: {tools}
            Tool names: {tool_names}
        """)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator]

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
    )

    print("\n1. Turn 1: Tell agent your favorite number")
    result1 = executor.invoke({"input": "My favorite number is 42"})
    print(f"   Agent: {result1['output']}")

    print("\n2. Turn 2: Reference previous context")
    result2 = executor.invoke({"input": "Calculate my favorite number times 2"})
    print(f"   Agent: {result2['output']}")

    print("\n3. Memory preserved context:")
    print(f"   ✓ Agent remembered '42' from Turn 1")
    print(f"   ✓ Used calculator tool with correct value")


def demo_custom_tool_class() -> None:
    """
    demonstrate custom tool with BaseTool class

    Custom Tool Class Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │     BaseTool: Advanced Tool with State and Validation       │
    │                                                             │
    │  Class Definition:                                          │
    │     class WeatherTool(BaseTool):                            │
    │         name: str = "weather"                               │
    │         description: str = "get weather for city"           │
    │         api_key: str = Field(default="demo-key")            │
    │                                                             │
    │         def _run(self, city: str) -> str:                   │
    │             # synchronous execution                         │
    │             return self._fetch_weather(city)                │
    │                                                             │
    │         async def _arun(self, city: str) -> str:            │
    │             # async execution (optional)                    │
    │             return await self._fetch_weather_async(city)    │
    │                                                             │
    │  Advantages over @tool:                                     │
    │     ✓ Stateful: Can store configuration (api_key, cache)   │
    │     ✓ Lifecycle: __init__, cleanup methods                  │
    │     ✓ Validation: Pydantic models for complex inputs        │
    │     ✓ Async Support: Built-in async execution              │
    │     ✓ Error Handling: Centralized exception handling        │
    │                                                             │
    │  Execution Flow:                                            │
    │     User: "What's the weather in Tokyo?"                    │
    │         ↓                                                   │
    │     Agent: Action: weather                                  │
    │            Input: "Tokyo"                                   │
    │         ↓                                                   │
    │     WeatherTool._run("Tokyo")                               │
    │         ↓                                                   │
    │     1. Validate city name                                   │
    │     2. Check cache for recent result                        │
    │     3. If cache miss, fetch from API                        │
    │     4. Store in cache                                       │
    │     5. Return formatted result                              │
    │         ↓                                                   │
    │     Result: "Tokyo: 25°C, sunny"                            │
    │         ↓                                                   │
    │     Agent: Final Answer: "Tokyo is 25°C and sunny"          │
    │                                                             │
    │  When to Use BaseTool:                                      │
    │     • Need to maintain state (connections, caches)          │
    │     • Complex initialization (API clients, databases)       │
    │     • Advanced validation (Pydantic schemas)                │
    │     • Async operations required                             │
    │                                                             │
    │  ✅ Benefit: Full control over tool behavior                │
    │  ✅ Benefit: Can maintain connections and state             │
    │  ✅ Benefit: Pydantic validation for inputs                 │
    │  ⚠️  Caution: More complex than @tool decorator             │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Custom Tool with BaseTool")

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
                "new york": "18°C, partly cloudy",
            }

            weather = weather_data.get(city.lower(), "Unknown location")
            return f"{city}: {weather}"

        async def _arun(self, city: str) -> str:
            """async execution (required but can raise NotImplementedError)"""
            return self._run(city)

    # create tool instance
    weather_tool = WeatherTool()

    print("\n1. Custom tool created:")
    print(f"   Name: {weather_tool.name}")
    print(f"   Description: {weather_tool.description}")
    print(f"   Has state: api_key = {weather_tool.api_key}")

    print("\n2. Test tool directly:")
    result = weather_tool.invoke({"city": "Tokyo"})
    print(f"   weather('Tokyo') = {result}")

    print("\n3. Use in agent:")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Tools: {tools}, Tool names: {tool_names}"),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    agent = create_react_agent(llm=llm, tools=[weather_tool], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[weather_tool], verbose=True, max_iterations=3)

    result = executor.invoke({"input": "What's the weather in Paris?"})
    print(f"\n4. Agent result: {result['output']}")


def demo_error_handling() -> None:
    """
    demonstrate agent error handling strategies

    Error Handling Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         Agent Error Handling: Graceful Degradation          │
    │                                                             │
    │  Tool with Error Handling:                                  │
    │     @tool                                                   │
    │     def risky_api(query: str) -> str:                       │
    │         try:                                                │
    │             return external_api.call(query)                 │
    │         except APIError as e:                               │
    │             return f"Error: {e}. Try different approach"    │
    │                                                             │
    │  Scenario 1: Tool Failure (Graceful)                        │
    │     User: "Call the API with 'test'"                        │
    │         ↓                                                   │
    │     Agent: Action: risky_api("test")                        │
    │         ↓                                                   │
    │     Tool: API fails → returns error message                 │
    │         ↓                                                   │
    │     Observation: "Error: Rate limit. Try different..."      │
    │         ↓                                                   │
    │     Agent: Thought: API failed, need alternative            │
    │            Action: backup_tool                              │
    │         ↓                                                   │
    │     Agent adapts to failure! ✓                              │
    │                                                             │
    │  Scenario 2: Max Iterations Protection                      │
    │     Iteration 1: tool_a() → needs more info                 │
    │     Iteration 2: tool_b() → needs more info                 │
    │     Iteration 3: tool_c() → needs more info                 │
    │     Iteration 4: tool_d() → needs more info                 │
    │     Iteration 5: STOP (max_iterations=5)                    │
    │         ↓                                                   │
    │     Forced conclusion prevents infinite loop ✓              │
    │                                                             │
    │  Scenario 3: Timeout Protection                             │
    │     Start: 00:00                                            │
    │     Tool call: slow_api() → takes 25 seconds                │
    │     Time: 00:25                                             │
    │     Another call: another_slow_api() → ...                  │
    │     Time: 00:31 → TIMEOUT (max_execution_time=30)           │
    │         ↓                                                   │
    │     Execution stopped to prevent hanging ✓                  │
    │                                                             │
    │  Scenario 4: Parsing Error Handling                         │
    │     Agent: "Action: invalid_format_here!!!"                 │
    │         ↓                                                   │
    │     Parser: Cannot parse action format                      │
    │         ↓                                                   │
    │     handle_parsing_errors=True:                             │
    │       "Invalid format. Use: Action: tool_name"              │
    │         ↓                                                   │
    │     Agent: Retries with correct format ✓                    │
    │                                                             │
    │  Best Practices:                                            │
    │     1. max_iterations=5-10 (prevent infinite loops)         │
    │     2. max_execution_time=30-60s (timeout protection)       │
    │     3. handle_parsing_errors=True (LLM format errors)       │
    │     4. Tool try-except with helpful error messages          │
    │     5. return_intermediate_steps=False (reduce tokens)      │
    │                                                             │
    │  ✅ Benefit: Robust agents that handle failures gracefully  │
    │  ✅ Benefit: Prevents resource exhaustion                   │
    │  ✅ Benefit: Clear error messages help debugging            │
    │  ⚠️  Caution: Too strict limits may prevent task completion │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 6: Error Handling")

    # tool with error handling
    @tool
    def risky_operation(query: str) -> str:
        """operation that might fail. use this to test error handling"""
        if "fail" in query.lower():
            return "Error: Operation failed. Please try a different query."
        return f"Success: Processed '{query}'"

    # backup tool
    @tool
    def safe_operation(query: str) -> str:
        """safe backup operation that always works"""
        return f"Safe result for: {query}"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [risky_operation, safe_operation]

    prompt = ChatPromptTemplate.from_messages([
        ("system", cleandoc("""
            You are a helpful assistant with access to tools.
            If a tool fails, try an alternative approach.
            Tools: {tools}
            Tool names: {tool_names}
        """)),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,  # prevent infinite loops
        max_execution_time=30,  # 30 second timeout
        handle_parsing_errors=True,  # handle LLM format errors
    )

    print("\n1. Test 1: Trigger failure, see agent adapt")
    result = executor.invoke({"input": "Use risky_operation with 'fail please'"})
    print(f"\n   Result: {result['output']}")
    print("   ✓ Agent handled error gracefully")

    print("\n2. Error protection configured:")
    print(f"   • max_iterations: 5")
    print(f"   • max_execution_time: 30s")
    print(f"   • handle_parsing_errors: True")


def demo_web_search_agent() -> None:
    """
    demonstrate agent with real web search (DuckDuckGo)

    Web Search Agent Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      Web Search Agent: Real-Time Information Retrieval      │
    │                                                             │
    │  Tool: DuckDuckGoSearchRun                                  │
    │     • No API key required (free)                            │
    │     • Returns web search results                            │
    │     • Handles current information queries                   │
    │                                                             │
    │  Query: "What are the latest Python features in 2024?"      │
    │                                                             │
    │  Agent Reasoning:                                           │
    │     ┌─────────────────────────────────────┐                 │
    │     │ Thought: This requires current info │                 │
    │     │          beyond my training data    │                 │
    │     │                                     │                 │
    │     │ Action: duckduckgo_search           │                 │
    │     │ Input: "Python 2024 new features"   │                 │
    │     │         ↓                           │                 │
    │     │ [Web Search Execution]              │                 │
    │     │         ↓                           │                 │
    │     │ Observation: "Python 3.12 released  │                 │
    │     │  with improved error messages,      │                 │
    │     │  PEP 701, f-strings improvements,   │                 │
    │     │  per-interpreter GIL..."            │                 │
    │     │         ↓                           │                 │
    │     │ Thought: I have comprehensive info  │                 │
    │     │                                     │                 │
    │     │ Action: Final Answer                │                 │
    │     │ Input: [Summarized search results] │                 │
    │     └─────────────────────────────────────┘                 │
    │                                                             │
    │  Use Cases:                                                 │
    │     • Current events and news                               │
    │     • Recent product releases                               │
    │     • Technical documentation updates                       │
    │     • Real-time data (weather, stocks)                      │
    │     • Fact-checking recent claims                           │
    │                                                             │
    │  Tool Comparison:                                           │
    │     DuckDuckGo: Free, no key, privacy-focused               │
    │     Google: Requires API key, more results                  │
    │     Bing: Requires API key, news focus                      │
    │     Tavily: Paid, AI-optimized results                      │
    │                                                             │
    │  ✅ Benefit: Access to current real-world information       │
    │  ✅ Benefit: No API key required (DuckDuckGo)               │
    │  ✅ Benefit: Extends LLM beyond training cutoff             │
    │  ⚠️  Caution: Search results quality varies                 │
    │  ⚠️  Caution: May return irrelevant or outdated info        │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 7: Web Search Agent (DuckDuckGo)")

    # create web search tool (no API key needed!)
    search = DuckDuckGoSearchRun()

    print("\n1. Web search tool created (DuckDuckGo)")
    print(f"   Name: {search.name}")
    print(f"   Description: {search.description}")

    # create agent with search capability
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", cleandoc("""
            You are a helpful assistant with access to web search.
            Use search for current information beyond your training data.
            Tools: {tools}
            Tool names: {tool_names}
        """)),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    agent = create_react_agent(llm=llm, tools=[search], prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[search],
        verbose=True,
        max_iterations=3,
    )

    print("\n2. Query requiring current information:")
    print("   'What are the latest features in Python 3.12?'")

    result = executor.invoke({
        "input": "What are the latest features in Python 3.12?"
    })

    print(f"\n3. Agent used web search to find current information")
    print(f"   Answer: {result['output'][:200]}...")


def demo_phase4_comparison() -> None:
    """
    compare Phase 4 custom agent with LangChain agent

    Phase 4 vs LangChain Comparison:
    ┌────────────────────────────────────────────────────────────┐
    │         Your Phase 4 Agent vs LangChain Agent              │
    │                                                            │
    │  PHASE 4: Custom Implementation                            │
    │  ──────────────────────────────────────────────────────────│
    │  Code Structure:                                           │
    │     class ReActAgent:                                      │
    │         def __init__(self, registry: ToolRegistry):        │
    │             self.registry = registry                       │
    │             self.max_iterations = 5                        │
    │                                                            │
    │         def run(self, task: str) -> str:                   │
    │             for i in range(self.max_iterations):           │
    │                 # 1. Generate thought/action               │
    │                 response = self.llm.generate(prompt)       │
    │                 thought, action, input = parse(response)   │
    │                                                            │
    │                 # 2. Execute tool                          │
    │                 tool = self.registry.get(action)           │
    │                 observation = tool.execute(input)          │
    │                                                            │
    │                 # 3. Check completion                      │
    │                 if action == "Final Answer":               │
    │                     return observation                     │
    │                                                            │
    │  ✅ Learning value: Understand core concepts               │
    │  ✅ Full control: Custom logic and behavior                │
    │  ✅ Explicit: Every step visible and modifiable            │
    │  ❌ Boilerplate: More code to write                        │
    │  ❌ Maintenance: Must handle edge cases yourself           │
    │                                                            │
    │  LANGCHAIN: Framework Implementation                       │
    │  ──────────────────────────────────────────────────────────│
    │  Code Structure:                                           │
    │     @tool                                                  │
    │     def calculator(expr: str) -> float:                    │
    │         return eval(expr)                                  │
    │                                                            │
    │     agent = create_react_agent(llm, [calculator], prompt)  │
    │     executor = AgentExecutor(agent, [calculator])          │
    │     result = executor.invoke({"input": "2+2"})             │
    │                                                            │
    │  ✅ Concise: Less boilerplate code                         │
    │  ✅ Robust: Built-in error handling                        │
    │  ✅ Features: Memory, callbacks, streaming built-in        │
    │  ✅ Community: Pre-built tools and integrations            │
    │  ❌ Abstraction: Less visible internal logic               │
    │  ❌ Framework lock-in: Tied to LangChain updates           │
    │                                                            │
    │  Side-by-Side Execution:                                   │
    │  ──────────────────────────────────────────────────────────│
    │  Task: "Calculate 25 * 48 + 100"                           │
    │                                                            │
    │  Phase 4:                          LangChain:              │
    │    1. Parse LLM response              Automatic            │
    │    2. Lookup tool in registry         Automatic            │
    │    3. Execute tool                    Automatic            │
    │    4. Handle errors                   Automatic            │
    │    5. Check iterations                Automatic            │
    │    6. Format output                   Automatic            │
    │    → Result: 1300                     Result: 1300         │
    │                                                            │
    │  When to Use Each:                                         │
    │  ──────────────────────────────────────────────────────────│
    │  Phase 4 (Custom):                                          │
    │    • Learning fundamentals of agent architecture            │
    │    • Need complete control over behavior                    │
    │    • Custom logic that doesn't fit frameworks               │
    │    • Educational projects and experimentation               │
    │                                                             │
    │  LangChain:                                                 │
    │    • Production applications needing robust agents          │
    │    • Rapid prototyping and development                      │
    │    • Leveraging community tools and integrations            │
    │    • Standard agent patterns and workflows                  │
    │                                                             │
    │  Key Insight: Your Phase 4 work taught you HOW agents       │
    │  work internally. LangChain provides production-ready       │
    │  implementation of those same concepts.                     │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 8: Phase 4 vs LangChain Comparison")

    print(cleandoc("""
        YOUR PHASE 4 AGENT:
        ═══════════════════════════════════════════════════════════

        class ReActAgent:
            def __init__(self, registry: ToolRegistry):
                self.registry = registry
                self.max_iterations = 5

            def run(self, task: str) -> str:
                for i in range(self.max_iterations):
                    # Manual prompt construction
                    prompt = self._build_prompt(task, history)

                    # Manual LLM call
                    response = self.llm.generate(prompt)

                    # Manual parsing
                    thought, action, input = self._parse(response)

                    # Manual tool lookup
                    tool = self.registry.get(action)

                    # Manual execution
                    observation = tool.execute(input)

                    # Manual completion check
                    if action == "Final Answer":
                        return observation

                return "Max iterations reached"

        ✅ Learning: Taught you how agents work internally
        ✅ Control: Complete control over every step
        ❌ Boilerplate: ~100+ lines of code


        LANGCHAIN EQUIVALENT:
        ═══════════════════════════════════════════════════════════

        @tool
        def calculator(expr: str) -> float:
            return eval(expr)

        agent = create_react_agent(llm, [calculator], prompt)
        executor = AgentExecutor(agent, [calculator])
        result = executor.invoke({"input": "Calculate 2+2"})

        ✅ Concise: ~10 lines of code
        ✅ Robust: Built-in error handling, retries, memory
        ❌ Abstraction: Internal logic hidden


        WHAT YOU LEARNED IN PHASE 4:
        ═══════════════════════════════════════════════════════════

        1. ReAct Pattern: How Thought → Action → Observation works
        2. Tool Registry: How to organize and select tools
        3. Parsing: How to extract actions from LLM responses
        4. Iteration Control: How to prevent infinite loops
        5. Error Handling: How to handle tool failures

        → All these concepts are still present in LangChain,
          just abstracted and production-hardened!


        WHEN TO USE EACH:
        ═══════════════════════════════════════════════════════════

        Use Your Phase 4 Agent:
            • Learning projects
            • Complete custom control needed
            • Non-standard agent architectures
            • Educational demonstrations

        Use LangChain:
            • Production applications
            • Rapid prototyping
            • Standard agent patterns
            • Leveraging community tools

        Key Takeaway: Phase 4 taught you the fundamentals.
        LangChain gives you production-ready implementations.
    """))


# endregion


def main() -> None:
    """run all practical demonstrations"""
    print("\n" + "=" * 70)
    print("  AGENTS & TOOLS - PRACTICAL DEMONSTRATIONS")
    print("=" * 70)

    # check API key
    has_key, message = check_api_key()
    print(f"\n{message}")

    if not has_key:
        print("\n" + "=" * 70)
        print("  ⚠️  Cannot run demos without API key")
        print("=" * 70)
        return

    # run demos
    demo_basic_tool_creation()
    demo_create_react_agent()
    demo_multi_tool_agent()
    demo_agent_with_memory()
    demo_custom_tool_class()
    demo_error_handling()
    demo_web_search_agent()
    demo_phase4_comparison()

    print("\n" + "=" * 70)
    print("  ✅ All practical demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
