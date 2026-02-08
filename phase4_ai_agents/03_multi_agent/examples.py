"""
Multi-Agent Examples - Demonstrations of hierarchical agent orchestration.

Shows different scenarios of how the orchestrator coordinates specialists:
1. Basic delegation - single specialist call
2. Research pipeline - research → analyze → write
3. Direct specialist use - calling specialists without orchestrator
4. Real API demo - using actual LLM (needs API key)

Run with: uv run python phase4_ai_agents/03_multi_agent/examples.py
"""

import os
import sys
from pathlib import Path

# enable imports from this package
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from schemas import AgentProfile, TeamConfig, TeamResult
from agents import (
    ToolRegistry,
    ResearchAgent,
    AnalysisAgent,
    WriterAgent,
)
from orchestrator import MultiAgentOrchestrator
from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section



def print_result(result: TeamResult) -> None:
    """print a team result nicely"""
    print(f"\n📝 Answer:\n{result.answer[:500]}...")
    print(f"\n📊 Stats:")
    print(f"   - Iterations: {result.iterations}")
    print(f"   - Delegations: {result.delegation_summary()}")
    print(f"   - Success: {'✅' if result.success else '❌'}")


# =============================================================================
# Demo 1: Basic Delegation
# =============================================================================


def demo_basic_delegation():
    """
    Basic multi-agent orchestration.

    Shows how the orchestrator automatically:
    1. Breaks down the task
    2. Delegates to appropriate specialists
    3. Synthesizes a final answer
    """
    print_section("Demo 1: Basic Delegation")

    print("\nScenario: Ask the team to research and summarize a topic")
    print("The orchestrator will decide which specialists to use.\n")

    orchestrator = MultiAgentOrchestrator(use_mock=True)
    result = orchestrator.run("What are the latest trends in AI development?")

    print_result(result)


# =============================================================================
# Demo 2: Research Pipeline
# =============================================================================


def demo_research_pipeline():
    """
    Research → Analyze → Write pipeline.

    A more complex task that naturally requires all three specialists
    in sequence.
    """
    print_section("Demo 2: Research Pipeline")

    print("\nScenario: Complex task requiring multiple specialists")
    print("Expected flow: Research → Analyze → Write\n")

    orchestrator = MultiAgentOrchestrator(
        config=TeamConfig(verbose=True, max_delegations=5),
        use_mock=True,
    )

    result = orchestrator.run(
        "Research Python web frameworks, analyze their pros and cons, "
        "and write a recommendation report for choosing one."
    )

    print_result(result)

    # show full delegation trace
    print("\n📋 Delegation Trace:")
    for i, d in enumerate(result.delegations, 1):
        status = "✅" if d.success else "❌"
        print(f"   {i}. [{d.agent_name}] {status}")
        print(f"      Task: {d.task[:60]}...")


# =============================================================================
# Demo 3: Direct Specialist Use
# =============================================================================


def demo_direct_specialists():
    """
    Using specialists directly without orchestrator.

    Sometimes you want to call specific specialists yourself,
    bypassing the orchestrator's decision-making.
    """
    print_section("Demo 3: Direct Specialist Use")

    print("\nScenario: Calling specialists directly (no orchestrator)")
    print("Useful when you know exactly which specialist you need.\n")

    # create specialists
    research = ResearchAgent(use_mock=True, verbose=True)
    analysis = AnalysisAgent(use_mock=True, verbose=True)
    writer = WriterAgent(use_mock=True, verbose=True)

    # call them directly
    print("\n--- Step 1: Research ---")
    research_result = research.execute(task="Research Python async programming")

    print("\n--- Step 2: Analyze ---")
    analysis_result = analysis.execute(task="Compare sync vs async programming patterns")

    print("\n--- Step 3: Write ---")
    writer_result = writer.execute(task="Write a summary of async programming benefits")

    print("\n📝 Final Output (from writer):")
    print(writer_result.data[:400] if writer_result.data else "No data")


# =============================================================================
# Demo 4: Custom Specialists
# =============================================================================


def demo_custom_specialist():
    """
    Creating a custom specialist with a specific role.

    Shows how to define a specialist with a custom profile
    for domain-specific tasks.
    """
    print_section("Demo 4: Custom Specialist")

    print("\nScenario: Creating a specialist with custom capabilities")

    # create a custom profile
    code_reviewer_profile = AgentProfile(
        name="code_review_agent",
        role="Code Review Specialist",
        capabilities=[
            "code analysis",
            "best practices review",
            "security checks",
            "performance suggestions",
        ],
        system_prompt="You are a code review specialist. Review code for quality, security, and performance.",
    )

    # create research agent with custom profile
    # (normally you'd create a CodeReviewAgent subclass)
    custom_agent = ResearchAgent(
        profile=code_reviewer_profile,
        use_mock=True,
        verbose=True,
    )

    print(f"\nCustom agent: {custom_agent.name}")
    print(f"Role: {custom_agent.profile.role}")
    print(f"Capabilities: {custom_agent.profile.capability_description()}")

    # use in a registry
    registry = ToolRegistry()
    registry.register(custom_agent)

    result = registry.execute("code_review_agent", task="Review this authentication code")
    print(f"\n📝 Result: {result.data[:200] if result.data else 'No data'}...")


# =============================================================================
# Demo 5: Team Configuration
# =============================================================================


def demo_team_config():
    """
    Configuring the orchestrator team.

    Shows how to customize team behavior with TeamConfig.
    """
    print_section("Demo 5: Team Configuration")

    print("\nScenario: Configuring orchestrator behavior")

    # strict configuration
    strict_config = TeamConfig(
        max_delegations=3,  # limit specialist calls
        max_iterations=5,  # limit orchestrator loops
        verbose=True,
    )

    print(f"\nConfiguration:")
    print(f"  - Max delegations: {strict_config.max_delegations}")
    print(f"  - Max iterations: {strict_config.max_iterations}")
    print(f"  - Provider: {strict_config.provider}")

    orchestrator = MultiAgentOrchestrator(
        config=strict_config,
        use_mock=True,
    )

    result = orchestrator.run("Quick analysis of cloud computing trends")

    print_result(result)


# =============================================================================
# Demo 6: Real API (Optional)
# =============================================================================


def demo_real_api():
    """
    Using real LLM API for orchestration.

    Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
    """
    print_section("Demo 6: Real API (Optional)")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("\n⚠️  No API key found. Skipping real API demo.")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable.")
        return

    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
    print(f"\nUsing {provider} API for real orchestration...")

    config = TeamConfig(
        provider=provider,
        orchestrator_model="gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307",
        max_delegations=3,
        verbose=True,
    )

    orchestrator = MultiAgentOrchestrator(
        config=config,
        use_mock=False,  # use real LLM!
    )

    result = orchestrator.run("What are three key benefits of using Python for AI development?")

    print_result(result)


# =============================================================================
# Main
# =============================================================================


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Two Agent System", "collaborative agents", example_two_agent_system),
    Demo("2", "Agent Orchestration", "coordinator pattern", example_agent_orchestration),
    Demo("3", "Hierarchical Agents", "manager-worker pattern", example_hierarchical_agents),
    Demo("4", "Agent Communication", "message passing between agents", example_agent_communication),
]

# endregion


def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Examples")
    runner.run()
if __name__ == "__main__":
    main()
