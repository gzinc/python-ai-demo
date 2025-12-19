# Module 2: Evaluation & Observability

**This is the learning gap** - measuring LLM and RAG quality in production.

## What You'll Learn

- RAG evaluation metrics (relevance, groundedness, faithfulness)
- LLM call tracing with structured spans
- Detecting hallucinations programmatically

## Why This Matters

```
Traditional App Metrics          LLM App Metrics
┌─────────────────────────┐     ┌─────────────────────────────────┐
│ Response time           │     │ Response time + token latency   │
│ Error rate              │     │ Hallucination rate              │
│ Throughput              │     │ Context relevance score         │
│ Memory usage            │     │ Answer groundedness             │
└─────────────────────────┘     └─────────────────────────────────┘
```

## Key Concepts

### 1. RAG Evaluation Metrics (Ragas Framework)

Four core metrics for RAG quality:

| Metric | Question It Answers | Best Method |
|--------|---------------------|-------------|
| **Context Relevance** | Did we retrieve the right documents? | Embedding/Cross-encoder |
| **Groundedness** | Is the answer supported by the context? | LLM-as-judge |
| **Faithfulness** | Does the answer avoid making things up? | LLM-as-judge |
| **Answer Relevance** | Does the answer address the question? | Embedding/Cross-encoder |

### Evaluation Backends

| Backend | Cost | Speed | Use For |
|---------|------|-------|---------|
| **EmbeddingScorer** | ~$0.02/1K | Fast | Semantic similarity (relevance) |
| **LLMJudgeScorer** | ~$0.15-3/1K | Slow | Reasoning (hallucination) |
| **CrossEncoderScorer** | FREE | Fast | Best accuracy for relevance |

```
User Question: "What is the capital of France?"

Retrieved Context: "France is a country in Western Europe.
                    Paris is the capital and largest city."

LLM Answer: "The capital of France is Paris."

Evaluation:
├── Context Relevance: 1.0 (context mentions capital)
├── Groundedness: 1.0 (answer found in context)
├── Faithfulness: 1.0 (no hallucination)
└── Answer Relevance: 1.0 (directly answers question)
```

### 2. Hallucination Detection

Hallucination = answer contains claims not in the context.

```
Context: "Python was created by Guido van Rossum in 1991."

Good answer: "Python was created in 1991 by Guido van Rossum."
  → Groundedness: 1.0

Hallucinated answer: "Python was created in 1991 by Guido van Rossum
                      at Google."
  → Groundedness: 0.5 (Google claim not in context)
```

### 3. LLM Tracing

Every LLM call should capture:
- Input (prompt, context)
- Output (response, tokens)
- Timing (latency per component)
- Metadata (model, temperature, etc.)

```
Trace: RAG Query
├── Span: Embedding (45ms, 1 API call)
├── Span: Retrieval (12ms, 5 docs returned)
├── Span: Context Assembly (2ms)
└── Span: LLM Generation (1,850ms, 450 tokens)
    Total: 1,909ms
```

## File Structure

```
02_evaluation/
├── README.md               # this file
├── __init__.py
├── rag_metrics.py          # evaluation backends (Embedding, LLM, Cross-encoder)
├── llm_tracing.py          # structured tracing for LLM calls
├── examples.py             # basic demos (heuristic fallback)
└── rag_eval_production.py  # production examples with real API calls
```

## Running the Examples

```bash
# from project root
uv run python -m phase5_production.02_evaluation.examples
```

## Industry Tools

| Category | Tools |
|----------|-------|
| RAG Evaluation | Ragas, TruLens, Phoenix (Arize) |
| LLM Tracing | LangSmith, Weights & Biases, OpenTelemetry |
| Observability | Prometheus, Grafana, Datadog |

## Exercises

1. **Evaluate your RAG**: Run metrics on Phase 3 RAG system
2. **Add tracing to ac-agent**: Instrument LLM calls with spans
3. **Hallucination threshold**: Alert when groundedness < 0.7

## Next Module

→ [03_optimization](../03_optimization/): Caching, rate limiting, performance