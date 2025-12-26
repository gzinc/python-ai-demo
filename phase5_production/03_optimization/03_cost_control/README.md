# Cost Control

Budget management for LLM API usage.

## Module Structure

```
03_cost_control/
├── cost_budget.py  # RequestBudget, UserBudget, CostGuard, MODEL_PRICING
└── README.md
```

## Why Cost Control?

LLM APIs are pay-per-token. Without limits:
- Single runaway request can cost $$$
- Users can drain budget unexpectedly
- No visibility into spend until bill arrives

**Key insight**: Budget checks BEFORE the API call prevent surprises on your bill.

## Request Budget

Limit cost per individual request:

```python
from phase5_production.03_optimization.03_cost_control.cost_budget import RequestBudget

budget = RequestBudget(
    max_input_tokens=4000,
    max_output_tokens=1000,
    max_cost_usd=0.10
)

status = budget.check("gpt-4o", input_tokens=3500, output_tokens=500)
if status.allowed:
    make_request()
else:
    print(f"Blocked: {status.reason}")
```

## User Budget

Track and limit per-user spending:

```python
from phase5_production.03_optimization.03_cost_control.cost_budget import UserBudget

budget = UserBudget(
    daily_limit_usd=10.00,
    monthly_limit_usd=100.00,
    warning_threshold=0.8  # warn at 80%
)

status = budget.check_and_record(
    user_id="user_123",
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500
)

if status.warning:
    notify_user(f"You've used {status.percent_used:.0f}% of your daily budget")
```

## Cost Guard

Combined request + user budget enforcement:

```python
from phase5_production.03_optimization.03_cost_control.cost_budget import CostGuard

guard = CostGuard(
    request_budget=RequestBudget(...),
    user_budget=UserBudget(...)
)

status = guard.check_request(user_id, model, input_tokens)
if not status.allowed:
    return {"error": status.reason}

# make request...

guard.record(user_id, model, input_tokens, output_tokens)
```

## Model Pricing

Built-in pricing for common models:

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-5-haiku | $0.80 | $4.00 |

## Run Demo

```bash
uv run python -m phase5_production.03_optimization.03_cost_control.cost_budget
```

## Key Classes

| File | Classes | Purpose |
|------|---------|---------|
| `cost_budget.py` | `RequestBudget` | Per-request limits |
| `cost_budget.py` | `UserBudget` | Per-user daily/monthly limits |
| `cost_budget.py` | `CostGuard` | Combined enforcement |
| `cost_budget.py` | `MODEL_PRICING` | Token pricing dictionary |
