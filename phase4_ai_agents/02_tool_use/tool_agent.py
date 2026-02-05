"""
Tool Agent - Agent that uses real tools.

This module integrates the tool system with ReActAgent to create an agent that can:
- Search the web
- Read and write files
- Make HTTP API calls

Run with:
    cd phase4_ai_agents/02_tool_use
    uv run python tool_agent.py
"""

import os
import sys

# allow running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.base import ToolRegistry
from tools.file_tools import ReadFileTool, WriteFileTool, ListDirectoryTool
from tools.web_search import WebSearchTool
from tools.http_tool import HttpGetTool
from agent import ReActAgent, AgentConfig


def create_tool_registry(
    allowed_directories: list[str] | None = None,
    use_mock: bool = True,
) -> ToolRegistry:
    """
    Create a registry with all available tools.

    Args:
        allowed_directories: Directories the file tools can access
        use_mock: Whether to use mock responses for web/http tools

    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()

    # register file tools
    registry.register(ReadFileTool(allowed_directories=allowed_directories))
    registry.register(WriteFileTool(allowed_directories=allowed_directories))
    registry.register(ListDirectoryTool(allowed_directories=allowed_directories))

    # register web tools
    registry.register(WebSearchTool(use_mock=use_mock))
    registry.register(HttpGetTool(use_mock=use_mock))

    return registry


def build_tool_descriptions(registry: ToolRegistry) -> dict[str, str]:
    """
    Build tool descriptions for the agent prompt.

    Args:
        registry: Tool registry with registered tools

    Returns:
        Dict mapping tool name to formatted description
    """
    descriptions = {}

    for definition in registry.get_all_definitions():
        param_parts = []
        for param in definition.parameters:
            required_marker = "" if param.required else " (optional)"
            param_parts.append(f"    - {param.name}: {param.description}{required_marker}")

        param_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
        descriptions[definition.name] = f"{definition.description}\n  Parameters:\n{param_str}"

    return descriptions


class ToolAgent:
    """
    Agent equipped with real tools.

    Combines ReActAgent with the tool system for practical AI agent capabilities.

    Usage:
        agent = ToolAgent()
        result = agent.run("Search for Python tutorials and summarize the top 3 results")
    """

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        use_mock: bool = True,
        provider: str = "openai",
        model: str | None = None,
    ):
        """
        Initialize tool agent.

        Args:
            allowed_directories: Directories file tools can access
            use_mock: Whether to use mock web/http responses
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (defaults to provider's default)
        """
        self._registry = create_tool_registry(
            allowed_directories=allowed_directories,
            use_mock=use_mock,
        )

        tool_descriptions = build_tool_descriptions(self._registry)

        # create tool functions dict for ReActAgent
        tools = {}
        for tool_name in self._registry.list_tools():
            def make_tool_func(name: str):
                def tool_func(**kwargs) -> str:
                    result = self._registry.execute(name, **kwargs)
                    return result.to_observation()
                return tool_func
            tools[tool_name] = make_tool_func(tool_name)

        config = AgentConfig(
            provider=provider,
            model=model,
            max_iterations=10,
            temperature=0.0,
        )

        self._agent = ReActAgent(
            tools=tools,
            tool_descriptions=tool_descriptions,
            config=config,
        )

    @property
    def available_tools(self) -> list[str]:
        """List of available tool names."""
        return self._registry.list_tools()

    def run(self, task: str):
        """
        Run the agent on a task.

        Args:
            task: Task description for the agent

        Returns:
            AgentResult with answer, actions, and status
        """
        return self._agent.run(task)


# ==============================================================================
#   DEMO
# ==============================================================================

def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# region Demo Functions

def demo_tool_registry():
    """Demonstrate the tool registry."""
    print_section("Tool Registry Demo")

    registry = create_tool_registry(use_mock=True)

    print(f"Registered tools: {registry.list_tools()}")
    print(f"Total tools: {len(registry)}")

    print("\n--- Testing read_file tool ---")
    result = registry.execute("read_file", path="/nonexistent/file.txt")
    print(f"Result: {result}")

    print("\n--- Testing web_search tool ---")
    result = registry.execute("web_search", query="Python tutorials", max_results=2)
    print(f"Result: {result.to_observation()}")

    print("\n--- Testing http_get tool ---")
    result = registry.execute("http_get", url="https://api.github.com/users/test")
    print(f"Result: {result.to_observation()}")


def demo_tool_agent():
    """Demonstrate the full tool agent."""
    print_section("Tool Agent Demo")

    agent = ToolAgent(
        use_mock=True,
        provider="openai",
    )

    print(f"Available tools: {agent.available_tools}")

    task = "Search the web for Python asyncio tutorials and tell me the top result"
    print(f"\n--- Task: {task} ---")

    result = agent.run(task)

    print(f"\n--- Result ---")
    print(f"Answer: {result.answer}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {result.total_tool_calls}")


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Tool Agent - Agent with Real Tool Integration")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Tool Registry", "registry with all available tools"),
        ("2", "Tool Agent", "ReAct agent with real tool access"),
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
        '1': ('Tool Registry', demo_tool_registry),
        '2': ('Tool Agent', demo_tool_agent),
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

        # check API key for demo 2
        if demo_num == '2':
            if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
                print("\n⚠️  Skipping Tool Agent demo - no API key found")
                print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this demo")
                continue

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
            selection = input("\n🎯 select demo(s) (e.g., '1', '1,2', or 'a' for all): ").strip()
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
