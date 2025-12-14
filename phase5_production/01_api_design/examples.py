"""
Module 1 Examples - LLM-Specific API Patterns.

Demonstrates:
1. Semantic caching
2. SSE streaming
3. Cost tracking

Run with: uv run python -m phase5_production.01_api_design.examples
"""

import asyncio

from .semantic_cache import demo_semantic_cache
from .llm_streaming import main as streaming_demo
from .cost_tracker import demo_cost_tracking


def print_section(title: str) -> None:
    """print section header"""
    print("\n")
    print("#" * 70)
    print(f"#  {title}")
    print("#" * 70)


async def main() -> None:
    """run all demos"""
    print("=" * 70)
    print("  Module 1: LLM-Specific API Patterns")
    print("=" * 70)

    # 1. semantic caching
    print_section("1. SEMANTIC CACHING")
    demo_semantic_cache()

    # 2. streaming
    print_section("2. SSE STREAMING")
    await streaming_demo()

    # 3. cost tracking
    print_section("3. COST TRACKING")
    demo_cost_tracking()

    print("\n")
    print("=" * 70)
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
    asyncio.run(main())
