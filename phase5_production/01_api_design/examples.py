"""
Module 1 Examples - LLM-Specific API Patterns.

Demonstrates:
1. Semantic caching
2. SSE streaming
3. Cost tracking

Run with: uv run python -m phase5_production.01_api_design.examples
"""

import asyncio

from common.demo_menu import Demo, MenuRunner

from .semantic_cache import demo_semantic_cache
from .llm_streaming import main as streaming_demo_async
from .cost_tracker import demo_cost_tracking


def demo_streaming() -> None:
    """wrapper to run async streaming demo"""
    asyncio.run(streaming_demo_async())


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Semantic Caching", "cache similar prompts, not just identical", demo_semantic_cache),
    Demo("2", "SSE Streaming", "token-by-token output for better UX", demo_streaming),
    Demo("3", "Cost Tracking", "per-request token/cost monitoring", demo_cost_tracking),
]

# endregion


def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Module 1: LLM-Specific API Patterns")
    runner.run()

    print("\n" + "=" * 70)
    print("  Summary: LLM APIs need these patterns beyond standard REST")
    print("=" * 70)
    print("""
    1. SEMANTIC CACHING
       - Cache similar prompts, not just identical
       - Reduces costs significantly

    2. SSE STREAMING
       - Token-by-token output
       - Better UX for slow responses

    3. COST TRACKING
       - Per-request token/cost
       - Budget monitoring and alerts
    """)


if __name__ == "__main__":
    main()
