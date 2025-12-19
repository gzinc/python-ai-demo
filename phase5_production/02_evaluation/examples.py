"""
Module 2 Examples - Evaluation & Observability.

Demonstrates:
1. RAG evaluation metrics (relevance, groundedness, faithfulness)
2. LLM tracing with spans

Run with: uv run python -m phase5_production.02_evaluation.examples
"""

try:
    # when run as module: python -m phase5_production.02_evaluation.examples
    from .rag_metrics import demo_rag_evaluation
    from .llm_tracing import demo_llm_tracing
except ImportError:
    # when run directly: python examples.py (or PyCharm)
    from rag_metrics import demo_rag_evaluation
    from llm_tracing import demo_llm_tracing


def print_section(title: str) -> None:
    """print section header"""
    print("\n")
    print("#" * 70)
    print(f"#  {title}")
    print("#" * 70)


def main() -> None:
    """run all demos"""
    print("=" * 70)
    print("  Module 2: Evaluation & Observability")
    print("=" * 70)

    # 1. RAG metrics
    print_section("1. RAG EVALUATION METRICS")
    demo_rag_evaluation()

    # 2. LLM tracing
    print_section("2. LLM TRACING")
    demo_llm_tracing()

    print("\n")
    print("=" * 70)
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
