# Prompt Compression

Production-grade prompt compression using Microsoft LLMLingua-2.

## Why Compress?

Tokens = cost. A 50% reduction in prompt tokens = 50% lower API bill.

## Approaches

| Approach | Quality | Speed | Cost |
|----------|---------|-------|------|
| **LLMLingua-2** | Best | Fast | FREE (local model) |
| Naive regex | Poor | Fastest | FREE |
| LLM summarization | Good | Slow | API cost |

## LLMLingua-2

Uses a BERT model trained specifically for compression - learns which tokens are **essential for meaning**, not just which are predictable.

```python
from token_optimization import LLMLingua2Compressor

compressor = LLMLingua2Compressor()

# compress to 50% of original tokens
compressed, stats = compressor.compress(prompt, rate=0.5)
print(f"Saved {stats.reduction_percent:.1f}% tokens")
```

### Compression Rates

| Rate | Meaning | Use Case |
|------|---------|----------|
| 0.7 | Keep 70% | Light compression, preserve nuance |
| 0.5 | Keep 50% | Balanced compression |
| 0.3 | Keep 30% | Aggressive, for very long contexts |

### First Run

Downloads ~400MB model from HuggingFace. Subsequent runs use cached model.

## Context Truncation

For RAG systems - keep most relevant chunks within token budget:

```python
from token_optimization import ContextTruncator

truncator = ContextTruncator(max_tokens=2000)
kept_chunks, stats = truncator.truncate_chunks(chunks, relevance_scores)
```

## Response Limits

Set appropriate `max_tokens` by task type:

```python
from token_optimization import ResponseLimiter

limiter = ResponseLimiter()
max_tokens = limiter.get_limit('summarization')  # 500
max_tokens = limiter.get_limit('classification')  # 50
```

## Run Demo

```bash
uv run python -m phase5_production.03_optimization.01_compression.token_optimization
```

## Key Classes

- `LLMLingua2Compressor` - Production compressor using Microsoft LLMLingua-2
- `NaiveCompressor` - Demo/fallback using regex (not for production)
- `ContextTruncator` - Keep top chunks within token budget
- `ResponseLimiter` - Task-appropriate max_tokens settings
