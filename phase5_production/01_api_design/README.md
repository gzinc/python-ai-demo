# Module 1: LLM-Specific API Patterns

Patterns unique to LLM applications that go beyond standard FastAPI.

## What's New Here

You already know FastAPI from ac-agent. This module covers:

| Pattern | What It Does | Why LLM-Specific |
|---------|--------------|------------------|
| **Semantic Caching** | Cache similar prompts, not just identical | Embeddings enable fuzzy matching |
| **SSE Streaming** | Token-by-token output | LLM responses take seconds |
| **Cost Tracking** | Per-request token/cost | LLM calls have variable cost |

## Key Concepts

### 1. Semantic Caching

Traditional cache: exact key match only.
Semantic cache: similarity-based matching.

```
User: "What is machine learning?"     → Cache miss, call LLM
User: "Explain machine learning"      → Cache HIT (similar enough)
User: "What is deep learning?"        → Cache miss (different topic)
```

Implementation:
```python
# embed the query
query_embedding = embed(query)

# search cache by similarity
cached = cache.search(query_embedding, threshold=0.95)
if cached:
    return cached.response

# cache miss - call LLM
response = await llm.generate(query)
cache.store(query_embedding, response)
```

### 2. SSE Streaming

LLM responses take 2-5 seconds. Streaming improves UX:

```
Without streaming:
[User waits 3 seconds...] → "Here is a complete response about..."

With streaming:
"Here" → "is" → "a" → "complete" → "response" → "about..."
```

### 3. Cost Tracking

Every LLM request has variable cost:

```python
@dataclass
class RequestCost:
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def cost_usd(self) -> float:
        # GPT-4: $30/1M input, $60/1M output
        return (self.prompt_tokens * 30 + self.completion_tokens * 60) / 1_000_000
```

## File Structure

```
01_api_design/
├── README.md           # this file
├── __init__.py
├── semantic_cache.py   # embedding-based caching
├── llm_streaming.py    # SSE streaming patterns
├── cost_tracker.py     # per-request cost tracking
└── examples.py         # runnable FastAPI demo
```

## Running the Examples

```bash
# from project root
uv run python -m phase5_production.01_api_design.examples
```

## Exercises

1. **Tune cache threshold**: Test 0.90 vs 0.95 vs 0.99 similarity
2. **Add cache TTL**: Expire entries after N hours
3. **Cost alerting**: Log warning when request exceeds $0.10

## Next Module

→ [02_evaluation](../02_evaluation/): RAG evaluation metrics, observability
