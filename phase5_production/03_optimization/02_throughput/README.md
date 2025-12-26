# Throughput Optimization

Maximize requests per second while respecting API limits.

## Module Structure

```
02_throughput/
├── batching.py      # EmbeddingBatcher, ParallelExecutor, BatchScheduler
├── rate_limiter.py  # TokenBucket, SlidingWindow, AdaptiveRateLimiter
└── README.md
```

## Batching

Reduce API calls by batching operations.

### Embedding Batches

Instead of N separate calls, send 1 batch:

```python
from phase5_production.03_optimization.02_throughput.batching import EmbeddingBatcher

batcher = EmbeddingBatcher(client, batch_size=100)
embeddings, stats = batcher.embed_all(texts)
print(f"Processed {stats.total_items} in {stats.num_batches} batches")
```

**Key insight**: Batching reduces latency, not cost - same tokens, fewer round-trips.

### Parallel LLM Calls

Run independent LLM calls concurrently:

```python
from phase5_production.03_optimization.02_throughput.batching import ParallelExecutor

executor = ParallelExecutor(max_concurrent=5)
results, stats = executor.execute_parallel(llm_call, prompts)
print(f"Completed in {stats.elapsed_seconds:.1f}s")
```

## Rate Limiting

Prevent 429 errors by controlling request rate.

### Token Bucket

Classic algorithm - allows bursts within limits:

```python
from phase5_production.03_optimization.02_throughput.rate_limiter import TokenBucket

limiter = TokenBucket(rate=10, capacity=100)  # 10/sec, burst 100

if limiter.acquire():
    make_api_call()
```

### Sliding Window

Smooth rate limiting over time window:

```python
from phase5_production.03_optimization.02_throughput.rate_limiter import SlidingWindow

limiter = SlidingWindow(max_requests=100, window_seconds=60)

if limiter.acquire():
    make_api_call()
```

### Adaptive Rate Limiter

Adjusts rate based on API responses:

```python
from phase5_production.03_optimization.02_throughput.rate_limiter import AdaptiveRateLimiter

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

| File | Classes | Purpose |
|------|---------|---------|
| `batching.py` | `EmbeddingBatcher`, `ParallelExecutor`, `BatchScheduler` | Batch operations |
| `rate_limiter.py` | `TokenBucket`, `SlidingWindow`, `AdaptiveRateLimiter` | Rate control |
