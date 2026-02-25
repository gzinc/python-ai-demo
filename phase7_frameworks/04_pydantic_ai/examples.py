"""
Module: Pydantic AI - Type-Safe Agent Framework

Build typed AI agents with dependency injection and clean testing.
Philosophy: explicit types > hidden magic | testable > black box

Demos 2, 4, 6: runnable with TestModel (no API key needed)
Demos 1, 3, 5, 7: conceptual code patterns (require real API key)

Run with: uv run python -m phase7_frameworks.04_pydantic_ai.examples
"""

import random
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section


# region Helper Functions
def print_code(label: str, code: str) -> None:
    """print code block with label"""
    print(f"\n┌─ {label} {'─' * (70 - len(label))}┐")
    print(code)
    print(f"└{'─' * 78}┘")
# endregion


# region 1. Philosophy
def demo_philosophy() -> None:
    """compare Pydantic AI philosophy with other frameworks"""
    print_section("1. What is Pydantic AI?")

    print("\n📦 ORIGIN:")
    print("   Created by Samuel Colvin (creator of Pydantic itself)")
    print("   Released late 2024 — agent framework built around Python's type system")

    print("\n🎯 PHILOSOPHY:")
    print("""
        LangChain:   high-level abstractions, huge ecosystem, lots of magic
        LlamaIndex:  document-centric, RAG-optimized, storage abstractions
        LangGraph:   stateful graphs, explicit state machines
        Pydantic AI: minimal magic, type-safe, dependency injection, testable
    """)

    langchain_code = """
        # LangChain: output is Any at runtime
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser

        chain = ChatOpenAI() | StrOutputParser()
        result = chain.invoke("Hello")   # result: str (or AIMessage? depends on chain)
        # no type checking possible — IDE shows 'Any'
    """

    pydantic_code = """
        # Pydantic AI: output is typed, validated, IDE-friendly
        from pydantic import BaseModel
        from pydantic_ai import Agent

        class Greeting(BaseModel):
            message: str
            language: str

        agent = Agent("openai:gpt-4o-mini", output_type=Greeting)
        result = agent.run_sync("Say hello in French")
        print(result.output.message)    # str — mypy validates, IDE autocompletes
        print(result.output.language)   # str — Pydantic validates at runtime
    """

    print_code("LANGCHAIN: untyped chain output", langchain_code)
    print_code("PYDANTIC AI: typed agent output", pydantic_code)

    print("\n💡 WHY THIS MATTERS:")
    print("   ✅ IDE autocomplete on result.output (knows the exact type)")
    print("   ✅ mypy catches type errors at dev time, not runtime")
    print("   ✅ Pydantic validation: if LLM returns wrong shape → ValidationError")
    print("   ✅ You already know Pydantic → lower learning curve than LangChain")
# endregion


# region 2. Core API
def demo_core_api() -> None:
    """demonstrate Agent, run_sync, and result structure with TestModel"""
    print_section("2. Core API: Agent, run_sync, Result")

    print("\n📋 AGENT CONSTRUCTOR:")
    print("""
        Agent(
            model,              # 'openai:gpt-4o', 'anthropic:claude-3-5-haiku-latest', etc.
            output_type=str,    # default: str | Pydantic model for structured output
            deps_type=None,     # dependency type (see Demo 5)
            instructions='...'  # system prompt
        )
    """)

    print("\n🚀 RUNNING WITH TestModel (no API key needed):")
    model = TestModel()
    agent = Agent(model, instructions="You are a helpful assistant.")
    result = agent.run_sync("What is Pydantic AI?")

    print(f"   result.output  = {result.output!r}")
    print(f"   output type    = {type(result.output).__name__}")
    usage = result.usage()
    print(f"   usage.requests = {usage.requests}")

    print("\n📋 EXECUTION METHODS:")
    print("""
        # Synchronous (scripts, notebooks)
        result = agent.run_sync("prompt")
        print(result.output)

        # Asynchronous (FastAPI, async apps)
        result = await agent.run("prompt")
        print(result.output)

        # Streaming (token by token)
        async with agent.run_stream("prompt") as response:
            async for text in response.stream_text():
                print(text, end="", flush=True)

        # Multi-turn (pass previous messages)
        first  = agent.run_sync("What is Paris?")
        second = agent.run_sync("And its population?",
                                message_history=first.all_messages())
    """)

    print("\n💡 SWAP PROVIDER IN ONE LINE:")
    print("   'openai:gpt-4o-mini'               → OpenAI")
    print("   'anthropic:claude-3-5-haiku-latest' → Anthropic")
    print("   'google-gla:gemini-2.0-flash'       → Google")
    print("   'ollama:llama3.2'                   → Local Ollama (no API key!)")
# endregion


# region 3. Structured Outputs
def demo_structured_outputs() -> None:
    """show type-safe structured outputs using Pydantic models"""
    print_section("3. Structured Outputs")

    print("\n📦 DEFINE OUTPUT TYPE AS PYDANTIC MODEL:")
    review_code = """
        from pydantic import BaseModel
        from pydantic_ai import Agent

        class CodeReview(BaseModel):
            severity: str          # 'low', 'medium', 'high', 'critical'
            issue: str             # what is wrong
            fix: str               # how to fix it
            confidence: float      # 0.0 - 1.0

        agent = Agent(
            "anthropic:claude-3-5-haiku-latest",
            output_type=CodeReview,
            instructions="You are a code reviewer. Return structured analysis."
        )

        result = agent.run_sync("Review: for i in range(len(items)):")
        review = result.output       # type: CodeReview (mypy knows this!)
        print(review.severity)       # 'low'
        print(review.issue)          # 'use enumerate() instead'
        print(review.fix)            # 'for i, item in enumerate(items):'
        print(review.confidence)     # 0.92
    """
    print_code("STRUCTURED OUTPUT: typed CodeReview", review_code)

    print("\n📦 NESTED MODELS (complex structured responses):")
    nested_code = """
        from pydantic import BaseModel, Field
        from typing import Literal

        class Finding(BaseModel):
            file: str
            line: int
            severity: Literal['low', 'medium', 'high', 'critical']
            description: str

        class SecurityAudit(BaseModel):
            score: int = Field(ge=0, le=100)
            findings: list[Finding]
            summary: str

        agent = Agent("openai:gpt-4o", output_type=SecurityAudit)
        audit = agent.run_sync(code_to_review).output

        # fully typed — IDE autocomplete, mypy validated, Pydantic validated
        for finding in audit.findings:
            if finding.severity == 'critical':
                send_alert(finding.file, finding.line)
    """
    print_code("NESTED MODELS: complex structured audit", nested_code)

    print("\n💡 VS LANGCHAIN:")
    print("   LangChain: StrOutputParser() → str → json.loads() → dict → manual access")
    print("   Pydantic AI: output_type=MyModel → result.output.field (typed, validated)")
# endregion


# region 4. Tool Use
def demo_tool_use() -> None:
    """show @agent.tool_plain and @agent.tool with TestModel"""
    print_section("4. Tool Use")

    print("\n📋 TWO DECORATOR STYLES:")
    print("""
        @agent.tool_plain          # no RunContext — pure function, no dep access
        def get_time() -> str:
            \"\"\"Return current time.\"\"\"
            return datetime.now().strftime("%H:%M")

        @agent.tool                # first arg is RunContext — access deps, metadata
        def get_user(ctx: RunContext[Deps]) -> str:
            \"\"\"Get current user name.\"\"\"
            return ctx.deps.user_name
    """)

    print("\n🚀 RUNNABLE DEMO (TestModel — validates tool registration):")
    model = TestModel()
    agent = Agent(model, instructions="Help the user with their requests.")

    @agent.tool_plain
    def roll_die() -> str:
        """Roll a six-sided die and return the result."""
        return str(random.randint(1, 6))

    @agent.tool_plain
    def current_time() -> str:
        """Return the current time."""
        return datetime.now().strftime("%H:%M")

    result = agent.run_sync("Roll a die and tell me the time.")
    print(f"   result.output = {result.output!r}")
    print("   (TestModel validates tool setup — real model would call the tools)")

    print("\n📋 TYPED PARAMETERS (auto-schema from type hints):")
    param_code = """
        @agent.tool_plain
        def search_docs(query: str, limit: int = 5) -> list[str]:
            \"\"\"Search documentation. Returns up to `limit` results.\"\"\"
            return vector_db.search(query, top_k=limit)

        # Pydantic AI auto-generates JSON schema from type hints
        # No need to write schema manually (unlike raw OpenAI function calling)
    """
    print_code("TYPED PARAMETERS: schema generated from hints", param_code)

    print("\n💡 VS RAW OPENAI FUNCTION CALLING:")
    print("   Raw OpenAI: write JSON schema manually → verbose boilerplate")
    print("   Pydantic AI: annotated type hints → schema generated automatically")
# endregion


# region 5. Dependency Injection
def demo_dependency_injection() -> None:
    """show deps_type pattern for injectable, testable services"""
    print_section("5. Dependency Injection")

    print("\n🎯 THE PROBLEM:")
    print("   Tools need external services (database, API client, config)")
    print("   Global variables = hidden coupling, hard to test")
    print("   Pydantic AI solution: inject deps at call time, swap in tests")

    deps_code = """
        from dataclasses import dataclass
        from pydantic_ai import Agent, RunContext
        import httpx

        @dataclass
        class AppDeps:
            db_url: str
            http_client: httpx.AsyncClient
            api_key: str

        agent = Agent(
            "openai:gpt-4o-mini",
            deps_type=AppDeps,
            instructions="You are a research assistant."
        )

        @agent.tool
        async def web_search(ctx: RunContext[AppDeps], query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            response = await ctx.deps.http_client.get(
                "https://api.search.com/search",
                params={"q": query},
                headers={"Authorization": f"Bearer {ctx.deps.api_key}"}
            )
            return response.text

        @agent.tool
        def db_lookup(ctx: RunContext[AppDeps], user_id: str) -> dict:
            \"\"\"Look up user in database.\"\"\"
            conn = create_connection(ctx.deps.db_url)
            return conn.query(f"SELECT * FROM users WHERE id = {user_id!r}")
    """
    print_code("DEPENDENCY INJECTION: inject services via deps", deps_code)

    swap_code = """
        # Production: real services
        async with httpx.AsyncClient() as client:
            prod_deps = AppDeps(
                db_url="postgresql://prod-server/db",
                http_client=client,
                api_key=os.environ["SEARCH_API_KEY"]
            )
            result = await agent.run("Research quantum computing", deps=prod_deps)

        # Tests: fake services, no real network calls, no real DB
        test_deps = AppDeps(
            db_url="sqlite:///:memory:",
            http_client=FakeHttpClient(responses={"quantum": "quantum facts..."}),
            api_key="test-key"
        )
        result = agent.run_sync("Research quantum computing", deps=test_deps)
    """
    print_code("SWAPPING DEPS: production vs test", swap_code)

    print("\n💡 WHY THIS MATTERS:")
    print("   ✅ Test without network calls (inject FakeHttpClient)")
    print("   ✅ Test without real database (inject in-memory SQLite)")
    print("   ✅ No global state — every run can use different deps")
    print("   ✅ Same pattern as FastAPI Depends() — familiar to Python devs")
# endregion


# region 6. Testing with TestModel
def demo_testing() -> None:
    """demonstrate TestModel and agent.override() for unit testing"""
    print_section("6. Testing Agents with TestModel")

    print("\n🎯 THE TESTING PROBLEM:")
    print("   Real API calls in tests: slow, costs money, non-deterministic")
    print("   Pydantic AI: TestModel — fast, free, deterministic")

    print("\n🚀 RUNNABLE DEMO — actual test pattern with agent.override():")

    # this simulates a 'production' agent defined elsewhere in codebase
    production_agent = Agent(
        "openai:gpt-4o-mini",    # real model — NOT called during test
        instructions="You are a helpful weather assistant."
    )

    call_log: list[str] = []

    @production_agent.tool_plain
    def get_temperature(city: str) -> float:
        """Get current temperature in Celsius for a city."""
        call_log.append(f"get_temperature({city!r})")
        return 22.5

    print("   [running with TestModel via agent.override() — no API key used]")
    test_model = TestModel()
    with production_agent.override(model=test_model):
        result = production_agent.run_sync("What is the temperature in London?")
        print(f"   result.output  = {result.output!r}")
        print(f"   call_log       = {call_log}")
        print(f"   usage.requests = {result.usage().requests}")

    print("\n📋 IN A REAL PYTEST FILE:")
    test_code = """
        import pytest
        from pydantic_ai.models.test import TestModel

        def test_weather_agent_wiring():
            \"\"\"verify agent tools are registered and agent runs without error\"\"\"
            with weather_agent.override(model=TestModel()):
                result = weather_agent.run_sync("Weather in Tokyo?")
                assert isinstance(result.output, str)
                assert result.usage().requests == 1

        def test_weather_agent_with_fake_deps():
            \"\"\"verify tool injection works with fake deps\"\"\"
            fake_deps = WeatherDeps(
                api_key="test-key",
                http_client=FakeHttpClient()
            )
            with weather_agent.override(model=TestModel()):
                result = weather_agent.run_sync("Weather?", deps=fake_deps)
                assert result.output is not None
    """
    print_code("PYTEST: testing with TestModel and fake deps", test_code)

    print("\n💡 TESTING PHILOSOPHY:")
    print("   ✅ TestModel: validates agent wiring (tools registered, deps injected)")
    print("   ✅ Fast: hundreds of unit tests run in seconds")
    print("   ✅ Deterministic: same input → same output, no flakiness")
    print("   ✅ Real model in integration tests only (separate CI job)")
# endregion


# region 7. Framework Comparison
def demo_comparison() -> None:
    """compare Pydantic AI vs LangChain vs raw API for the same task"""
    print_section("7. Framework Comparison: Same Task, Three Ways")

    print("\n📋 TASK: build an agent that searches docs and returns a structured summary")

    raw_code = """
        # Raw OpenAI: full control, most verbose, no type safety
        import json
        from openai import OpenAI

        client = OpenAI()
        tools = [{
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search documentation",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]
        # manual agent loop: call API, handle tool_calls, call API again...
        result = json.loads(final_response)    # hope it's valid JSON with right keys
        summary = result["summary"]            # dict access — no type safety
    """

    langchain_code = """
        # LangChain: less boilerplate, but output is untyped str
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.tools import tool

        @tool
        def search_docs(query: str) -> str:
            \"\"\"Search documentation.\"\"\"
            return do_search(query)

        agent = create_react_agent(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            tools=[search_docs],
            prompt=hub.pull("hwchase17/react")      # needs hub pull!
        )
        executor = AgentExecutor(agent=agent, tools=[search_docs])
        result = executor.invoke({"input": "Find Python async patterns"})
        summary = result["output"]    # str — parse manually to get structure
    """

    pydantic_code = """
        # Pydantic AI: type-safe, testable, minimal boilerplate
        from pydantic import BaseModel
        from pydantic_ai import Agent

        class DocSummary(BaseModel):
            topic: str
            key_points: list[str]
            confidence: float

        agent = Agent(
            "openai:gpt-4o-mini",
            output_type=DocSummary,
            instructions="Search docs and summarize findings."
        )

        @agent.tool_plain
        def search_docs(query: str) -> str:
            \"\"\"Search documentation.\"\"\"
            return do_search(query)

        result = agent.run_sync("Find Python async patterns")
        summary = result.output            # type: DocSummary (typed!)
        print(summary.key_points)          # list[str], IDE autocomplete ✅
        print(summary.confidence)          # float, Pydantic-validated ✅
    """

    print_code("RAW API: manual schema + loop + JSON parsing", raw_code)
    print_code("LANGCHAIN: less boilerplate, untyped output", langchain_code)
    print_code("PYDANTIC AI: typed output, no schema boilerplate", pydantic_code)

    print("\n📊 DECISION MATRIX:")
    print("                   Raw API    LangChain   Pydantic AI")
    print("   Type safety       ❌           ❌           ✅")
    print("   Testability       🟡           🟡           ✅")
    print("   Boilerplate       high         medium       low")
    print("   Ecosystem         ✅           ✅✅          🟡 (newer)")
    print("   Learning curve    low          high         low")
    print("   Magic factor      none         high         low")

    print("\n💡 CHOOSE PYDANTIC AI WHEN:")
    print("   ✅ Structured/typed output is important")
    print("   ✅ Testability is a priority (FastAPI-style apps)")
    print("   ✅ Team already uses Pydantic")
    print("   ✅ Want simplicity without LangChain's abstraction overhead")

    print("\n💡 STICK WITH LANGCHAIN/LANGGRAPH WHEN:")
    print("   ✅ Need LangSmith monitoring and tracing")
    print("   ✅ Complex multi-agent graphs (LangGraph)")
    print("   ✅ Need the large integration ecosystem")
    print("   ✅ Document-heavy RAG at scale (LlamaIndex)")
# endregion


# region Demo Menu Configuration
DEMOS = [
    Demo("1", "Philosophy",           "Pydantic AI in the framework landscape",    demo_philosophy),
    Demo("2", "Core API",             "Agent, run_sync, result (live TestModel)",  demo_core_api),
    Demo("3", "Structured Outputs",   "output_type for typed responses",           demo_structured_outputs),
    Demo("4", "Tool Use",             "@agent.tool_plain and @agent.tool",         demo_tool_use),
    Demo("5", "Dependency Injection", "deps_type for injectable, testable agents", demo_dependency_injection),
    Demo("6", "Testing",              "TestModel and agent.override()",            demo_testing),
    Demo("7", "Framework Comparison", "Pydantic AI vs LangChain vs raw API",      demo_comparison),
]
# endregion


# region Main
def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Pydantic AI Demo")
    runner.run()


if __name__ == "__main__":
    main()
# endregion
