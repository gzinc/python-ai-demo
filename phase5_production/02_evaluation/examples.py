"""
Module 2 Examples - Evaluation & Observability.

Demonstrates:
1. RAG evaluation metrics (relevance, groundedness, faithfulness)
2. LLM tracing with spans

Run with: uv run python -m phase5_production.02_evaluation.examples
"""

from common.demo_menu import Demo, MenuRunner

try:
    # when run as module: python -m phase5_production.02_evaluation.examples
    from .rag_metrics import demo_rag_evaluation
    from .llm_tracing import demo_llm_tracing
except ImportError:
    # when run directly: python examples.py (or PyCharm)
    from rag_metrics import demo_rag_evaluation
    from llm_tracing import demo_llm_tracing


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "RAG Evaluation Metrics", "context relevance, groundedness, faithfulness", demo_rag_evaluation),
    Demo("2", "LLM Tracing", "structured spans with timing and metadata", demo_llm_tracing),
]

# endregion


def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Module 2: Evaluation & Observability")
    runner.run()

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print("""
    1. RAG EVALUATION (Ragas-style)
       - Context Relevance: Did we retrieve the right docs?
       - Groundedness: Is the answer in the context?
       - Faithfulness: No hallucination?
       - Answer Relevance: Does it answer the question?

    2. LLM TRACING
       - Structured spans for each operation
       - Timing, tokens, metadata captured
       - Enables debugging and optimization

    Production tools:
       - Ragas, TruLens, Phoenix for evaluation
       - LangSmith, W&B, OpenTelemetry for tracing
    """)


if __name__ == "__main__":
    main()
