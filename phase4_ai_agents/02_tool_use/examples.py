"""
Tool Use Examples - Demonstrates real tool implementations.

This file shows how to use the tool system:
1. Individual tools (file, web, http)
2. Tool registry
3. Integration with agent

Run with:
    cd phase4_ai_agents/02_tool_use
    uv run python examples.py
"""

import os
import sys
import tempfile

from common.demo_menu import Demo, MenuRunner

# allow running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas.tool import ToolParameter, ToolDefinition, ToolResult
from tools.base import BaseTool, ToolRegistry
from tools.file_tools import ReadFileTool, WriteFileTool, ListDirectoryTool
from tools.web_search import WebSearchTool
from tools.http_tool import HttpGetTool
from common.util.utils import print_section



# ==============================================================================
#   Example 1: Tool Schema Basics
# ==============================================================================

def example_tool_schemas():
    """
    Understand the building blocks of tools.

    Every tool has:
    1. Parameters (what inputs it accepts)
    2. Definition (metadata for the LLM)
    3. Result (success/failure + data)
    """
    print_section("Example 1: Tool Schema Basics")

    # --- ToolParameter ---
    # describes a single parameter
    print("\n--- ToolParameter ---")

    param = ToolParameter(
        name="query",
        param_type="string",
        description="the search query to execute",
        required=True,
    )
    print(f"Parameter: {param.name}")
    print(f"  Type: {param.param_type}")
    print(f"  Required: {param.required}")

    # with enum values (like Java enum)
    format_param = ToolParameter(
        name="format",
        param_type="string",
        description="output format",
        required=False,
        enum_values=["json", "text", "html"],
    )
    print(f"\nParameter with enum: {format_param.name}")
    print(f"  Allowed values: {format_param.enum_values}")

    # --- ToolDefinition ---
    # complete tool metadata
    print("\n--- ToolDefinition ---")

    definition = ToolDefinition(
        name="example_search",
        description="search for information on a topic",
        parameters=[param, format_param],
    )
    print(f"Tool: {definition.name}")
    print(f"  Description: {definition.description}")
    print(f"  Parameters: {[p.name for p in definition.parameters]}")

    # convert to OpenAI format
    openai_schema = definition.to_openai_schema()
    print(f"\nOpenAI schema:")
    print(f"  {openai_schema}")

    # --- ToolResult ---
    # standardized result type
    print("\n--- ToolResult ---")

    # success result (factory method pattern)
    success = ToolResult.ok({"answer": "Python is awesome!"})
    print(f"Success result: {success}")
    print(f"  As observation: {success.to_observation()}")

    # failure result
    failure = ToolResult.fail("file not found")
    print(f"\nFailure result: {failure}")
    print(f"  As observation: {failure.to_observation()}")


# ==============================================================================
#   Example 2: File Tools
# ==============================================================================

def example_file_tools():
    """
    Demonstrate file operation tools.

    These are REAL tools - they actually read/write to the filesystem.
    """
    print_section("Example 2: File Tools")

    # create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temp directory: {temp_dir}")

        # --- WriteFileTool ---
        print("\n--- WriteFileTool ---")
        write_tool = WriteFileTool(allowed_directories=[temp_dir])

        result = write_tool.execute(
            path=f"{temp_dir}/test.txt",
            content="Hello from the agent!\nThis is line 2.",
        )
        print(f"Write result: {result.to_observation()}")

        # --- ReadFileTool ---
        print("\n--- ReadFileTool ---")
        read_tool = ReadFileTool(allowed_directories=[temp_dir])

        result = read_tool.execute(path=f"{temp_dir}/test.txt")
        print(f"Read result:\n{result.to_observation()}")

        # try to read non-existent file
        result = read_tool.execute(path=f"{temp_dir}/nonexistent.txt")
        print(f"\nRead non-existent: {result.to_observation()}")

        # --- ListDirectoryTool ---
        print("\n--- ListDirectoryTool ---")

        # create some more files
        write_tool.execute(path=f"{temp_dir}/notes.md", content="# Notes")
        os.makedirs(f"{temp_dir}/subdir", exist_ok=True)

        list_tool = ListDirectoryTool(allowed_directories=[temp_dir])
        result = list_tool.execute(path=temp_dir)
        print(f"Directory contents: {result.to_observation()}")

        # --- Security: Directory Restriction ---
        print("\n--- Security: Directory Restriction ---")
        restricted_tool = ReadFileTool(allowed_directories=[temp_dir])

        # try to read outside allowed directory
        result = restricted_tool.execute(path="/etc/passwd")
        print(f"Restricted access: {result.to_observation()}")


# ==============================================================================
#   Example 3: Web Search Tool
# ==============================================================================

def example_web_search():
    """
    Demonstrate web search tool.

    Uses mock results by default (no API key needed).
    """
    print_section("Example 3: Web Search Tool")

    # create tool with mock mode
    search_tool = WebSearchTool(use_mock=True)

    print(f"Tool: {search_tool.name}")
    print(f"Definition: {search_tool.definition.description}")

    # search for Python
    print("\n--- Search: Python tutorials ---")
    result = search_tool.execute(query="Python tutorials", max_results=2)
    print(f"Results:\n{result.to_observation()}")

    # search for AI
    print("\n--- Search: artificial intelligence ---")
    result = search_tool.execute(query="artificial intelligence", max_results=2)
    print(f"Results:\n{result.to_observation()}")

    # search for something else
    print("\n--- Search: weather forecast ---")
    result = search_tool.execute(query="weather forecast")
    print(f"Results:\n{result.to_observation()}")


# ==============================================================================
#   Example 4: HTTP Tool
# ==============================================================================

def example_http_tool():
    """
    Demonstrate HTTP GET tool.

    Uses mock results by default (no actual HTTP calls).
    """
    print_section("Example 4: HTTP Tool")

    # create tool with mock mode
    http_tool = HttpGetTool(use_mock=True)

    print(f"Tool: {http_tool.name}")

    # mock GitHub API call
    print("\n--- GET GitHub API ---")
    result = http_tool.execute(url="https://api.github.com/users/octocat")
    print(f"Response:\n{result.to_observation()}")

    # mock JSONPlaceholder API
    print("\n--- GET JSONPlaceholder ---")
    result = http_tool.execute(url="https://jsonplaceholder.typicode.com/posts/1")
    print(f"Response:\n{result.to_observation()}")

    # invalid URL
    print("\n--- Invalid URL ---")
    result = http_tool.execute(url="not-a-valid-url")
    print(f"Response: {result.to_observation()}")


# ==============================================================================
#   Example 5: Tool Registry
# ==============================================================================

def example_tool_registry():
    """
    Demonstrate the tool registry pattern.

    The registry is like a service container or DI container.
    """
    print_section("Example 5: Tool Registry")

    # create empty registry
    registry = ToolRegistry()
    print(f"Empty registry: {len(registry)} tools")

    # register tools
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(WebSearchTool(use_mock=True))
    registry.register(HttpGetTool(use_mock=True))

    print(f"After registration: {len(registry)} tools")
    print(f"Tool names: {registry.list_tools()}")

    # check if tool exists
    print(f"\n'read_file' in registry: {'read_file' in registry}")
    print(f"'unknown_tool' in registry: {'unknown_tool' in registry}")

    # get tool by name
    print("\n--- Get tool by name ---")
    tool = registry.get("web_search")
    if tool:
        print(f"Got tool: {tool.name}")
        print(f"  Description: {tool.definition.description}")

    # execute tool through registry
    print("\n--- Execute through registry ---")
    result = registry.execute("web_search", query="Python", max_results=1)
    print(f"Result: {result.to_observation()}")

    # execute unknown tool
    print("\n--- Execute unknown tool ---")
    result = registry.execute("unknown_tool", foo="bar")
    print(f"Result: {result.to_observation()}")

    # get OpenAI format tools
    print("\n--- OpenAI format ---")
    openai_tools = registry.get_openai_tools()
    print(f"OpenAI tools count: {len(openai_tools)}")
    for tool_schema in openai_tools:
        print(f"  - {tool_schema['function']['name']}")


# ==============================================================================
#   Example 6: Creating Custom Tools
# ==============================================================================

def example_custom_tool():
    """
    Show how to create your own custom tool.

    Just extend BaseTool and implement name, definition, execute.
    """
    print_section("Example 6: Creating Custom Tools")

    # define a custom tool
    class CalculatorTool(BaseTool):
        """Simple calculator tool for demonstration."""

        @property
        def name(self) -> str:
            return "calculator"

        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name=self.name,
                description="perform basic math operations: add, subtract, multiply, divide",
                parameters=[
                    ToolParameter(
                        name="operation",
                        param_type="string",
                        description="math operation to perform",
                        required=True,
                        enum_values=["add", "subtract", "multiply", "divide"],
                    ),
                    ToolParameter(
                        name="a",
                        param_type="number",
                        description="first number",
                        required=True,
                    ),
                    ToolParameter(
                        name="b",
                        param_type="number",
                        description="second number",
                        required=True,
                    ),
                ],
            )

        def execute(self, operation: str, a: float, b: float, **kwargs) -> ToolResult:
            """Perform the calculation."""
            try:
                if operation == "add":
                    result = a + b
                elif operation == "subtract":
                    result = a - b
                elif operation == "multiply":
                    result = a * b
                elif operation == "divide":
                    if b == 0:
                        return ToolResult.fail("cannot divide by zero")
                    result = a / b
                else:
                    return ToolResult.fail(f"unknown operation: {operation}")

                return ToolResult.ok(f"{a} {operation} {b} = {result}")

            except Exception as e:
                return ToolResult.fail(str(e))

    # use the custom tool
    calc = CalculatorTool()

    print(f"Tool: {calc.name}")
    print(f"Parameters: {[p.name for p in calc.definition.parameters]}")

    # test calculations
    print("\n--- Calculations ---")
    print(calc.execute(operation="add", a=10, b=5).to_observation())
    print(calc.execute(operation="multiply", a=7, b=6).to_observation())
    print(calc.execute(operation="divide", a=100, b=4).to_observation())
    print(calc.execute(operation="divide", a=1, b=0).to_observation())

    # add to registry
    print("\n--- Add to registry ---")
    registry = ToolRegistry()
    registry.register(calc)
    result = registry.execute("calculator", operation="subtract", a=100, b=37)
    print(f"Via registry: {result.to_observation()}")


# ==============================================================================
#   Main
# ==============================================================================

# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Calculator Tool", "basic math operations", example_calculator_tool),
    Demo("2", "Web Search Tool", "information retrieval", example_web_search_tool),
    Demo("3", "Custom Tool Creation", "build your own tools", example_custom_tool),
    Demo("4", "Tool Composition", "combine multiple tools", example_tool_composition),
]

# endregion


def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Examples")
    runner.run()
if __name__ == "__main__":
    main()
