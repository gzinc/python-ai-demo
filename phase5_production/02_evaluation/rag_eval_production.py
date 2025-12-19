"""
Production RAG Evaluation Examples - Real API usage.

Shows all 3 evaluation configurations:
1. Embedding + LLM (recommended balance)
2. Cross-encoder + LLM (best accuracy, no embedding cost)
3. All LLM (most accurate, most expensive)

Run with: uv run python -m phase5_production.02_evaluation.rag_eval_production

Requires: OPENAI_API_KEY in .env file or environment
"""

import os
from inspect import cleandoc
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from .rag_metrics import (
        CrossEncoderScorer,
        EmbeddingScorer,
        LLMJudgeScorer,
        RAGEvaluator,
    )
except ImportError:
    from rag_metrics import (
        CrossEncoderScorer,
        EmbeddingScorer,
        LLMJudgeScorer,
        RAGEvaluator,
    )

# test data
TEST_CASES = [
    {
        "name": "Good RAG",
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Paris is the capital and largest city of France.",
        "answer": "The capital of France is Paris.",
    },
    {
        "name": "Hallucinated",
        "question": "When was Python created?",
        "context": "Python is a programming language created by Guido van Rossum. It was first released in 1991.",
        "answer": "Python was created in 1991 by Guido van Rossum while working at Google. It has over 50 million users.",
    },
    {
        "name": "Irrelevant Context",
        "question": "What is machine learning?",
        "context": "The weather in London is often rainy. Big Ben is a famous landmark.",
        "answer": "Machine learning is a type of artificial intelligence that learns from data.",
    },
]


def run_evaluation(evaluator: RAGEvaluator, name: str) -> None:
    """run evaluation on all test cases"""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print("=" * 60)

    for case in TEST_CASES:
        result = evaluator.evaluate(
            question=case["question"],
            context=case["context"],
            answer=case["answer"],
        )
        print(f"\n[{case['name']}]")
        print(f"Q: {case['question']}")
        print(f"A: {case['answer'][:60]}...")
        print(result)


def option1_embedding_plus_llm() -> None:
    """
    Option 1: Embedding + LLM (RECOMMENDED)

    - Context/Answer relevance: Embedding similarity (fast, ~$0.02/1K)
    - Groundedness/Faithfulness: LLM judge (accurate, ~$0.15/1K)

    Best for: Production systems needing balance of speed and accuracy
    """
    from openai import OpenAI

    print("\n" + "#" * 60)
    print("# Option 1: Embedding + LLM (Recommended)")
    print("#" * 60)

    client = OpenAI()

    evaluator = RAGEvaluator(
        embedding_scorer=EmbeddingScorer(client, model="text-embedding-3-small"),
        llm_scorer=LLMJudgeScorer(client, model="gpt-4o-mini"),
    )

    run_evaluation(evaluator, "Embedding + LLM Judge")

    print(cleandoc('''

        Cost breakdown per evaluation:
        - 2 embedding calls (~$0.00002)
        - 2 LLM calls (~$0.0003)
        - Total: ~$0.0003/eval or $0.30/1000 evals
    '''))


def option2_cross_encoder_plus_llm() -> None:
    """
    Option 2: Cross-encoder + LLM (BEST ACCURACY/COST)

    - Context/Answer relevance: Cross-encoder (local model, FREE)
    - Groundedness/Faithfulness: LLM judge (accurate)

    Best for: When you need high accuracy and have compute for local model
    Requires: pip install sentence-transformers
    """
    from openai import OpenAI

    print("\n" + "#" * 60)
    print("# Option 2: Cross-encoder + LLM (Best accuracy/cost)")
    print("#" * 60)

    client = OpenAI()

    evaluator = RAGEvaluator(
        cross_encoder_scorer=CrossEncoderScorer(),  # downloads ~80MB model on first use
        llm_scorer=LLMJudgeScorer(client, model="gpt-4o-mini"),
    )

    run_evaluation(evaluator, "Cross-encoder + LLM Judge")

    print(cleandoc('''

        Cost breakdown per evaluation:
        - 2 cross-encoder calls (FREE, ~50ms each)
        - 2 LLM calls (~$0.0003)
        - Total: ~$0.0003/eval or $0.30/1000 evals

        Note: First run downloads ms-marco-MiniLM model (~80MB)
    '''))


def option3_all_llm() -> None:
    """
    Option 3: All LLM (MOST ACCURATE)

    - All metrics use LLM judge
    - Most accurate but most expensive

    Best for: Critical evaluations where accuracy matters more than cost
    """
    from openai import OpenAI

    print("\n" + "#" * 60)
    print("# Option 3: All LLM (Most accurate)")
    print("#" * 60)

    client = OpenAI()

    # use same LLM for all metrics
    llm_scorer = LLMJudgeScorer(client, model="gpt-4o-mini")

    evaluator = RAGEvaluator(
        embedding_scorer=llm_scorer,  # LLM for relevance too
        llm_scorer=llm_scorer,
    )

    run_evaluation(evaluator, "All LLM Judge")

    print(cleandoc('''

        Cost breakdown per evaluation:
        - 4 LLM calls (~$0.0006)
        - Total: ~$0.0006/eval or $0.60/1000 evals

        For maximum accuracy, use gpt-4o instead of gpt-4o-mini:
        - ~$0.012/eval or $12/1000 evals
    '''))


def main() -> None:
    """run all options"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY='your-key'")
        return

    print("=" * 60)
    print("  Production RAG Evaluation Examples")
    print("=" * 60)

    # option 1: recommended
    option1_embedding_plus_llm()

    # option 2: best accuracy/cost (requires sentence-transformers)
    try:
        option2_cross_encoder_plus_llm()
    except ImportError as e:
        print(f"\n[Skipping Option 2: {e}]")
        print("Install with: pip install sentence-transformers")

    # option 3: most accurate
    option3_all_llm()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(cleandoc('''
        | Option | Relevance      | Hallucination  | Cost/1K  | Accuracy |
        |--------|----------------|----------------|----------|----------|
        | 1      | Embedding      | LLM            | $0.30    | Good     |
        | 2      | Cross-encoder  | LLM            | $0.30    | Better   |
        | 3      | LLM            | LLM            | $0.60    | Best     |

        Recommendation:
        - Start with Option 1 (Embedding + LLM)
        - Use Option 2 if you need better relevance scoring
        - Use Option 3 for critical evaluations only
    '''))


if __name__ == "__main__":
    main()
