# Module 3: LLM Optimization

Optimize LLM applications for cost, latency, and throughput.

## What You'll Learn

- Token optimization (prompt compression, context management)
- Batching strategies (embeddings, parallel calls)
- Rate limiting for LLM APIs
- Cost budgets and guardrails

## Why This Matters

```
Unoptimized LLM App              Optimized LLM App
┌─────────────────────────┐     ┌─────────────────────────────────┐
│ $50/day API costs       │     │ $15/day (70% reduction)         │
│ 3s average latency      │     │ 800ms (parallel + cache)        │
│ Rate limit errors       │     │ Smooth rate limiting            │
│ Context overflow errors │     │ Smart truncation                │
└─────────────────────────┘     └─────────────────────────────────┘
```

## Module Structure

```
03_optimization/
├── 01_compression/           # Token optimization
│   ├── schemas.py            # TokenStats dataclass
│   ├── compressors.py        # LLMLingua-2, NaiveCompressor
│   ├── truncation.py         # ContextTruncator
│   ├── response_limits.py    # ResponseLimiter
│   ├── examples.py           # Demo functions
│   └── README.md
│
├── 02_throughput/            # Speed optimization
│   ├── batching.py           # Embedding batches, parallel calls
│   ├── rate_limiter.py       # Token bucket, sliding window
│   └── README.md
│
├── 03_cost_control/          # Budget management
│   ├── cost_budget.py        # Per-request/user limits
│   └── README.md
│
└── README.md                 # This file
```

## Submodules

### [01_compression](01_compression/) - Token Optimization

Reduce tokens = reduce cost.

- **LLMLingua-2**: Production prompt compression (50-70% reduction)
- **Context truncation**: Keep relevant chunks within budget
- **Response limits**: Task-appropriate max_tokens

```bash
uv run python -m phase5_production.03_optimization.01_compression.examples
```

### [02_throughput](02_throughput/) - Speed Optimization

Maximize requests/second.

- **Batching**: 1 call instead of N for embeddings
- **Parallel execution**: Concurrent LLM calls
- **Rate limiting**: Token bucket, sliding window, adaptive

```bash
uv run python -m phase5_production.03_optimization.02_throughput.batching
uv run python -m phase5_production.03_optimization.02_throughput.rate_limiter
```

### [03_cost_control](03_cost_control/) - Budget Management

Prevent runaway costs.

- **Request budget**: Per-request cost limits
- **User budget**: Daily/monthly per-user limits
- **Cost guard**: Combined enforcement

```bash
uv run python -m phase5_production.03_optimization.03_cost_control.cost_budget
```

## Key Concepts

### Token Optimization

```
Before: 2,500 tokens (verbose prompt + full context)
After:  1,200 tokens (LLMLingua-2 compression + smart truncation)
Savings: 52%
```

### Batching

```
Embedding 100 documents:
├── Sequential: 100 API calls × 200ms = 20 seconds
└── Batched:    1 API call × 300ms = 0.3 seconds (66x faster)
```

### Rate Limiting

```
Token Bucket Algorithm:
├── Bucket: 10,000 tokens/minute capacity
├── Refill: 167 tokens/second
└── Request: Check bucket → deduct → proceed or wait
```

### Cost Budgets

```
Budget Configuration:
├── Per-request: max $0.05
├── Per-user/day: max $5.00
├── Per-app/month: max $500
└── Alert at 80% threshold
```

## Cost Optimization Cheatsheet

| Technique | Savings | Effort |
|-----------|---------|--------|
| LLMLingua-2 compression | 50-70% | Low |
| Semantic caching (Module 1) | 50-80% | Medium |
| Smaller model (gpt-4o-mini vs gpt-4o) | 90% | Low |
| Context truncation | 30-50% | Low |
| Batching embeddings | Time only | Low |
| Rate limiting | Prevents errors | Medium |

## Next Steps

After this module, you'll have completed Phase 5: Production!

Consider:
- Apply patterns to your projects
- Build a cost dashboard
- Set up alerting for budget thresholds
