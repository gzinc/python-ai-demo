"""
RAG Evaluation Metrics - measure retrieval and generation quality.

Three evaluation approaches with different trade-offs:
1. Embedding similarity - fast, cheap, good for semantic overlap
2. LLM-as-judge - best accuracy, expensive, needed for reasoning
3. Cross-encoder - best accuracy/cost ratio, requires model download

Recommended per metric:
- Context Relevance: Embedding (semantic overlap check)
- Answer Relevance: Embedding (semantic overlap check)
- Groundedness: LLM-as-judge (needs reasoning about claims)
- Faithfulness: LLM-as-judge (needs reasoning about hallucination)

Run with: uv run python -m phase5_production.02_evaluation.rag_metrics
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from inspect import cleandoc
from typing import Protocol


class MetricType(Enum):
    """RAG evaluation metric types"""
    CONTEXT_RELEVANCE = "context_relevance"
    GROUNDEDNESS = "groundedness"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"


# =============================================================================
# Evaluation Backends
# =============================================================================


class Scorer(ABC):
    """base class for evaluation scorers"""

    @abstractmethod
    def score(self, text_a: str, text_b: str, prompt: str | None = None) -> float:
        """score similarity or evaluate with prompt"""
        ...


class EmbeddingScorer(Scorer):
    """
    Embedding-based similarity scoring.

    Fast and cheap - good for context_relevance and answer_relevance.
    Uses cosine similarity between embeddings.

    Usage:
        from openai import OpenAI
        scorer = EmbeddingScorer(OpenAI())
        score = scorer.score("query", "document")
    """

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        """get embedding with caching"""
        if text not in self._cache:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            self._cache[text] = response.data[0].embedding
        return self._cache[text]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """compute cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def score(self, text_a: str, text_b: str, prompt: str | None = None) -> float:
        """compute embedding similarity (prompt ignored)"""
        emb_a = self._embed(text_a)
        emb_b = self._embed(text_b)
        return self._cosine_similarity(emb_a, emb_b)


class LLMJudgeScorer(Scorer):
    """
    LLM-as-judge scoring.

    Most accurate but expensive - required for groundedness and faithfulness.
    Uses structured prompts to get numeric scores.

    Usage:
        from openai import OpenAI
        scorer = LLMJudgeScorer(OpenAI())
        score = scorer.score(context, answer, prompt="Is the answer grounded?...")
    """

    def __init__(self, client, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def score(self, text_a: str, text_b: str, prompt: str | None = None) -> float:
        """get LLM judgment score"""
        if prompt is None:
            raise ValueError("LLMJudgeScorer requires a prompt")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip()

        # extract numeric score
        match = re.search(r'(\d*\.?\d+)', result)
        if match:
            score = float(match.group(1))
            return min(1.0, max(0.0, score))
        return 0.5


class CrossEncoderScorer(Scorer):
    """
    Cross-encoder model scoring.

    Best accuracy/cost ratio for relevance tasks.
    Requires sentence-transformers library and model download.

    Usage:
        scorer = CrossEncoderScorer()  # downloads model on first use
        score = scorer.score("query", "document")

    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """lazy load model"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "CrossEncoderScorer requires sentence-transformers: "
                    "pip install sentence-transformers"
                )
        return self._model

    def score(self, text_a: str, text_b: str, prompt: str | None = None) -> float:
        """compute cross-encoder relevance score (prompt ignored)"""
        model = self._get_model()
        scores = model.predict([(text_a, text_b)])
        # normalize to 0-1 (ms-marco outputs ~-10 to 10)
        raw_score = float(scores[0])
        return 1 / (1 + 2.718281828 ** (-raw_score))  # sigmoid


class HeuristicScorer(Scorer):
    """
    Fallback heuristic scoring when no API is available.

    Less accurate but works offline. Good for testing and demos.
    """

    def score(self, text_a: str, text_b: str, prompt: str | None = None) -> float:
        """keyword overlap heuristic"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what',
                     'who', 'where', 'when', 'how', 'why', 'of', 'in', 'to',
                     'and', 'or', 'but', 'for', 'with', 'at', 'by', 'from'}
        words_a -= stopwords
        words_b -= stopwords
        if not words_a:
            return 0.5
        overlap = len(words_a & words_b)
        return min(1.0, overlap / len(words_a))


# =============================================================================
# RAG Evaluation
# =============================================================================


@dataclass
class RAGEvaluation:
    """complete RAG evaluation result"""
    question: str
    context: str
    answer: str
    context_relevance: float
    groundedness: float
    faithfulness: float
    answer_relevance: float

    @property
    def overall_score(self) -> float:
        """weighted average of all metrics"""
        return (
            self.context_relevance * 0.25 +
            self.groundedness * 0.30 +
            self.faithfulness * 0.25 +
            self.answer_relevance * 0.20
        )

    @property
    def has_hallucination(self) -> bool:
        """detect likely hallucination"""
        return self.groundedness < 0.7 or self.faithfulness < 0.7

    def __str__(self) -> str:
        status = "⚠️ HALLUCINATION RISK" if self.has_hallucination else "✓ OK"
        return cleandoc(f'''
            RAG Evaluation [{status}]
            ├── Context Relevance: {self.context_relevance:.2f}
            ├── Groundedness:      {self.groundedness:.2f}
            ├── Faithfulness:      {self.faithfulness:.2f}
            ├── Answer Relevance:  {self.answer_relevance:.2f}
            └── Overall Score:     {self.overall_score:.2f}
        ''')


class RAGEvaluator:
    """
    Evaluate RAG pipeline quality using appropriate methods per metric.

    Recommended configuration:
    - Context/Answer Relevance: EmbeddingScorer (fast, semantic)
    - Groundedness/Faithfulness: LLMJudgeScorer (needs reasoning)

    Usage:
        from openai import OpenAI
        client = OpenAI()

        evaluator = RAGEvaluator(
            embedding_scorer=EmbeddingScorer(client),
            llm_scorer=LLMJudgeScorer(client),
        )
        result = evaluator.evaluate(question, context, answer)
    """

    def __init__(
        self,
        embedding_scorer: Scorer | None = None,
        llm_scorer: Scorer | None = None,
        cross_encoder_scorer: Scorer | None = None,
    ):
        # use heuristic fallback if no scorers provided
        self._heuristic = HeuristicScorer()
        self.embedding_scorer = embedding_scorer or self._heuristic
        self.llm_scorer = llm_scorer or self._heuristic
        self.cross_encoder_scorer = cross_encoder_scorer

    def evaluate_context_relevance(self, question: str, context: str) -> float:
        """
        Are the retrieved documents relevant to the question?

        Method: Embedding similarity (or cross-encoder if available)
        Why: Semantic overlap check doesn't need reasoning
        """
        if self.cross_encoder_scorer:
            return self.cross_encoder_scorer.score(question, context)
        return self.embedding_scorer.score(question, context)

    def evaluate_groundedness(self, context: str, answer: str) -> float:
        """
        Is each claim in the answer supported by the context?

        Method: LLM-as-judge (required)
        Why: Needs to reason about whether claims can be inferred
        """
        prompt = cleandoc(f'''
            Evaluate how well the answer is grounded in the context.

            Context: {context}

            Answer: {answer}

            For each claim in the answer, check if it can be verified from the context.

            Score from 0.0 to 1.0 where:
            - 0.0 = answer contains many claims not in context
            - 0.5 = some claims supported, some not
            - 1.0 = all claims can be traced to context

            Return ONLY a number between 0 and 1.
        ''')

        if isinstance(self.llm_scorer, HeuristicScorer):
            return self._heuristic_groundedness(context, answer)

        return self.llm_scorer.score(context, answer, prompt)

    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Does the answer avoid making claims not in context?

        Method: LLM-as-judge (required)
        Why: Needs to identify fabricated facts, numbers, names
        """
        prompt = cleandoc(f'''
            Check if the answer is faithful to the context (no hallucinations).

            Context: {context}

            Answer: {answer}

            Look for:
            - Made-up facts not in context
            - Numbers/statistics not mentioned in context
            - Names/entities not in context
            - Exaggerations or embellishments

            Score from 0.0 to 1.0 where:
            - 0.0 = significant hallucinations present
            - 0.5 = minor additions beyond context
            - 1.0 = completely faithful to context

            Return ONLY a number between 0 and 1.
        ''')

        if isinstance(self.llm_scorer, HeuristicScorer):
            return self._heuristic_faithfulness(context, answer)

        return self.llm_scorer.score(context, answer, prompt)

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Does the answer actually address the question?

        Method: Embedding similarity (or cross-encoder if available)
        Why: Semantic overlap check doesn't need reasoning
        """
        if self.cross_encoder_scorer:
            return self.cross_encoder_scorer.score(question, answer)
        return self.embedding_scorer.score(question, answer)

    def evaluate(self, question: str, context: str, answer: str) -> RAGEvaluation:
        """run all evaluations and return complete result"""
        return RAGEvaluation(
            question=question,
            context=context,
            answer=answer,
            context_relevance=self.evaluate_context_relevance(question, context),
            groundedness=self.evaluate_groundedness(context, answer),
            faithfulness=self.evaluate_faithfulness(context, answer),
            answer_relevance=self.evaluate_answer_relevance(question, answer),
        )

    # fallback heuristics for groundedness/faithfulness

    def _heuristic_groundedness(self, context: str, answer: str) -> float:
        """fallback: sentence-word overlap"""
        sentences = [s.strip() for s in answer.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if not sentences:
            return 1.0
        context_lower = context.lower()
        supported = 0
        for sentence in sentences:
            words = set(sentence.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'it'}
            if not words:
                supported += 1
                continue
            matches = sum(1 for w in words if w in context_lower)
            if matches / len(words) > 0.5:
                supported += 1
        return supported / len(sentences)

    def _heuristic_faithfulness(self, context: str, answer: str) -> float:
        """fallback: check for numbers/names not in context"""
        # check numbers
        ctx_nums = set(re.findall(r'\b\d+\b', context))
        ans_nums = set(re.findall(r'\b\d+\b', answer))
        num_score = 1.0
        if ans_nums:
            new_nums = ans_nums - ctx_nums
            num_score = 1.0 - (len(new_nums) / len(ans_nums))

        # check capitalized words (potential names)
        ctx_caps = set(re.findall(r'\b[A-Z][a-z]+\b', context))
        ans_caps = set(re.findall(r'\b[A-Z][a-z]+\b', answer))
        cap_score = 1.0
        if ans_caps:
            new_caps = ans_caps - ctx_caps
            tolerance = min(2, len(ans_caps) // 2)
            penalty = max(0, len(new_caps) - tolerance)
            cap_score = 1.0 - (penalty / len(ans_caps))

        return (num_score + cap_score) / 2


# =============================================================================
# Demo
# =============================================================================


def demo_rag_evaluation() -> None:
    """demonstrate RAG evaluation with different backends"""
    print("=" * 60)
    print("  RAG Evaluation Metrics Demo")
    print("=" * 60)

    print("\n[Using heuristic fallback - see production examples below]\n")

    evaluator = RAGEvaluator()  # no scorers = heuristics

    # test case 1: good RAG response
    print("1. GOOD RAG Response:\n")
    eval1 = evaluator.evaluate(
        question="What is the capital of France?",
        context="France is a country in Western Europe. Paris is the capital and largest city of France, with a population of over 2 million.",
        answer="The capital of France is Paris.",
    )
    print(f"   Q: {eval1.question}")
    print(f"   A: {eval1.answer}")
    print(f"\n{eval1}")

    # test case 2: hallucinated response
    print("\n" + "-" * 60)
    print("\n2. HALLUCINATED Response:\n")
    eval2 = evaluator.evaluate(
        question="When was Python created?",
        context="Python is a programming language created by Guido van Rossum. It was first released in 1991.",
        answer="Python was created in 1991 by Guido van Rossum while working at Google. It has over 50 million users worldwide.",
    )
    print(f"   Q: {eval2.question}")
    print(f"   A: {eval2.answer}")
    print(f"\n{eval2}")
    print("\n   Note: 'Google' and '50 million' not in context = hallucination")

    # test case 3: irrelevant context
    print("\n" + "-" * 60)
    print("\n3. IRRELEVANT Context:\n")
    eval3 = evaluator.evaluate(
        question="What is machine learning?",
        context="The weather in London is often rainy. The city has many famous landmarks including Big Ben.",
        answer="Machine learning is a type of artificial intelligence.",
    )
    print(f"   Q: {eval3.question}")
    print(f"   Context: {eval3.context[:50]}...")
    print(f"   A: {eval3.answer}")
    print(f"\n{eval3}")
    print("\n   Note: Low context relevance, answer not grounded")

    # production usage examples
    print("\n" + "=" * 60)
    print("  Production Usage Examples")
    print("=" * 60)

    example_embedding = cleandoc('''
        # Option 1: Embedding + LLM (recommended)
        from openai import OpenAI

        client = OpenAI()
        evaluator = RAGEvaluator(
            embedding_scorer=EmbeddingScorer(client),      # fast for relevance
            llm_scorer=LLMJudgeScorer(client),             # accurate for hallucination
        )
    ''')

    example_cross = cleandoc('''
        # Option 2: Cross-encoder + LLM (best accuracy, no embedding cost)
        from openai import OpenAI

        client = OpenAI()
        evaluator = RAGEvaluator(
            cross_encoder_scorer=CrossEncoderScorer(),     # local model, very accurate
            llm_scorer=LLMJudgeScorer(client),             # still need LLM for reasoning
        )
    ''')

    example_all_llm = cleandoc('''
        # Option 3: All LLM (most accurate, most expensive)
        from openai import OpenAI

        client = OpenAI()
        llm = LLMJudgeScorer(client, model="gpt-4o")
        evaluator = RAGEvaluator(
            embedding_scorer=llm,    # use LLM for everything
            llm_scorer=llm,
        )
    ''')

    print(f"\n{example_embedding}\n")
    print("-" * 40)
    print(f"\n{example_cross}\n")
    print("-" * 40)
    print(f"\n{example_all_llm}\n")

    print("=" * 60)
    print("  Method Selection Guide")
    print("=" * 60)
    print(cleandoc('''
        | Metric            | Recommended    | Why                           |
        |-------------------|----------------|-------------------------------|
        | context_relevance | Embedding      | semantic overlap, fast        |
        | answer_relevance  | Embedding      | semantic overlap, fast        |
        | groundedness      | LLM-as-judge   | needs to reason about claims  |
        | faithfulness      | LLM-as-judge   | needs to detect fabrication   |

        Cost comparison (per 1000 evaluations):
        - Heuristic:     $0 (but least accurate)
        - Cross-encoder: $0 (local model, ~50ms/eval)
        - Embedding:     ~$0.02 (text-embedding-3-small)
        - LLM-as-judge:  ~$0.15 (gpt-4o-mini) to $3 (gpt-4o)

        Tools: Ragas, TruLens, Phoenix (Arize), DeepEval
    '''))
    print("=" * 60)


if __name__ == "__main__":
    demo_rag_evaluation()
