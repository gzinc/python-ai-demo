"""
Cost Tracker - per-request token and cost tracking for LLM APIs.

LLM calls have variable cost based on:
- Model (GPT-4 vs GPT-3.5 vs Claude)
- Input tokens (prompt length)
- Output tokens (response length)

Run with: uv run python -m phase5_production.01_api_design.cost_tracker
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Model(Enum):
    """supported models with pricing (USD per 1M tokens)"""
    GPT_4 = ("gpt-4", 30.0, 60.0)
    GPT_4_TURBO = ("gpt-4-turbo", 10.0, 30.0)
    GPT_35_TURBO = ("gpt-3.5-turbo", 0.50, 1.50)
    CLAUDE_3_OPUS = ("claude-3-opus", 15.0, 75.0)
    CLAUDE_3_SONNET = ("claude-3-sonnet", 3.0, 15.0)
    CLAUDE_35_SONNET = ("claude-3.5-sonnet", 3.0, 15.0)

    def __init__(self, model_id: str, input_price: float, output_price: float):
        self.model_id = model_id
        self.input_price = input_price  # per 1M tokens
        self.output_price = output_price  # per 1M tokens


@dataclass
class RequestCost:
    """cost breakdown for a single LLM request"""
    model: Model
    prompt_tokens: int
    completion_tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def input_cost(self) -> float:
        """cost for prompt tokens in USD"""
        return (self.prompt_tokens / 1_000_000) * self.model.input_price

    @property
    def output_cost(self) -> float:
        """cost for completion tokens in USD"""
        return (self.completion_tokens / 1_000_000) * self.model.output_price

    @property
    def total_cost(self) -> float:
        """total cost in USD"""
        return self.input_cost + self.output_cost

    def __str__(self) -> str:
        return (
            f"{self.model.model_id}: "
            f"{self.prompt_tokens}+{self.completion_tokens} tokens = "
            f"${self.total_cost:.6f}"
        )


class CostTracker:
    """
    Track LLM costs across requests.

    Usage:
        tracker = CostTracker()

        # after each LLM call
        cost = tracker.track(
            model=Model.GPT_4,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

        # check totals
        print(tracker.summary())
    """

    def __init__(self, budget_usd: float | None = None):
        self.budget_usd = budget_usd
        self._requests: list[RequestCost] = []

    def track(
        self,
        model: Model,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> RequestCost:
        """record a request and return cost details"""
        cost = RequestCost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._requests.append(cost)

        # budget warning
        if self.budget_usd and self.total_cost > self.budget_usd:
            print(f"⚠️  BUDGET EXCEEDED: ${self.total_cost:.4f} > ${self.budget_usd:.4f}")

        return cost

    @property
    def total_cost(self) -> float:
        """total cost across all requests"""
        return sum(r.total_cost for r in self._requests)

    @property
    def total_tokens(self) -> int:
        """total tokens across all requests"""
        return sum(r.total_tokens for r in self._requests)

    def summary(self) -> dict:
        """cost summary"""
        by_model: dict[str, float] = {}
        for r in self._requests:
            model_id = r.model.model_id
            by_model[model_id] = by_model.get(model_id, 0) + r.total_cost

        return {
            "requests": len(self._requests),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "budget_usd": self.budget_usd,
            "budget_remaining": round(self.budget_usd - self.total_cost, 6) if self.budget_usd else None,
        }

    def reset(self) -> None:
        """clear all tracked requests"""
        self._requests.clear()


def demo_cost_tracking() -> None:
    """demonstrate cost tracking"""
    print("=" * 60)
    print("  Cost Tracker Demo")
    print("=" * 60)

    tracker = CostTracker(budget_usd=0.10)

    # simulate requests
    requests = [
        (Model.GPT_4, 500, 200),
        (Model.GPT_4, 1000, 500),
        (Model.GPT_35_TURBO, 2000, 1000),
        (Model.CLAUDE_35_SONNET, 800, 300),
    ]

    print("\nTracking requests:\n")

    for model, prompt, completion in requests:
        cost = tracker.track(model, prompt, completion)
        print(f"  {cost}")

    print(f"\nSummary:")
    summary = tracker.summary()
    print(f"  Total requests: {summary['requests']}")
    print(f"  Total tokens: {summary['total_tokens']:,}")
    print(f"  Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"  By model: {summary['by_model']}")
    print(f"  Budget remaining: ${summary['budget_remaining']:.6f}")

    print("\n" + "-" * 60)
    print("\nModel pricing comparison (per 1M tokens):\n")
    print(f"  {'Model':<20} {'Input':>10} {'Output':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for model in Model:
        print(f"  {model.model_id:<20} ${model.input_price:>8.2f} ${model.output_price:>8.2f}")

    print("\n" + "=" * 60)
    print("  Key insight: Track costs per-request for budget control")
    print("=" * 60)


if __name__ == "__main__":
    demo_cost_tracking()
