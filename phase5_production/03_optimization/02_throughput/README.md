# Throughput Optimization

Maximize requests per second while respecting API limits.

## Batching

Reduce API calls by batching operations.

### Embedding Batches

Instead of N separate calls, send 1 batch:

```python
from batching import EmbeddingBatcher

batcher = EmbeddingBatcher(client, batch_size=100)
embeddings, stats = batcher.embed_all(texts)
print(f"Processed {stats.total_items} in {stats.num_batches} batches")
```

**Cost savings**: 100 texts = 1 API call instead of 100.

### Parallel LLM Calls

Run independent LLM calls concurrently:

```python
from batching import ParallelExecutor

executor = ParallelExecutor(max_concurrent=5)
results, stats = executor.execute_parallel(llm_call, prompts)
print(f"Completed in {stats.elapsed_seconds:.1f}s")
```

## Rate Limiting

Prevent 429 errors by controlling request rate.

### Token Bucket

Classic algorithm - allows bursts within limits:

```python
from rate_limiter import TokenBucket

limiter = TokenBucket(rate=10, capacity=100)  # 10/sec, burst 100

if limiter.acquire():
    make_api_call()
```

### Sliding Window

Smooth rate limiting over time window:

```python
from rate_limiter import SlidingWindow

limiter = SlidingWindow(max_requests=100, window_seconds=60)

if limiter.acquire():
    make_api_call()
```

### Adaptive Rate Limiter

Adjusts rate based on API responses:

```python
from rate_limiter import AdaptiveRateLimiter

limiter = AdaptiveRateLimiter(initial_rate=10)

try:
    response = make_api_call()
    limiter.record_success()
except RateLimitError as e:
    limiter.record_rate_limit(e.retry_after)
```

## Run Demos

```bash
# batching demo
uv run python -m phase5_production.03_optimization.02_throughput.batching

# rate limiter demo
uv run python -m phase5_production.03_optimization.02_throughput.rate_limiter
```

## Key Classes

**Batching:**
- `EmbeddingBatcher` - Batch embedding API calls
- `ParallelExecutor` - Concurrent LLM calls
- `BatchScheduler` - Manage batch queue

**Rate Limiting:**
- `TokenBucket` - Classic burst-allowing limiter
- `SlidingWindow` - Smooth rate limiting
- `AdaptiveRateLimiter` - Self-adjusting based on responses
