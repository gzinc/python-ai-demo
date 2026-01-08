"""
Cost Budgets - prevent runaway LLM costs with hard limits.

Features:
1. Per-request limits - reject expensive requests
2. Per-user budgets - daily/monthly quotas
3. Alerts and notifications - warn before hitting limits

Run with: uv run python -m phase5_production.03_optimization.cost_budget
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from inspect import cleandoc


class BudgetResult(Enum):
    """result of budget check"""
    ALLOWED = "allowed"
    WARNING = "warning"
    REJECTED = "rejected"


@dataclass
class BudgetStatus:
    """status of a budget check"""
    result: BudgetResult
    used: float
    limit: float
    message: str = ""

    @property
    def remaining(self) -> float:
        return max(0, self.limit - self.used)

    @property
    def percent_used(self) -> float:
        if self.limit == 0:
            return 100.0
        return (self.used / self.limit) * 100

    def __str__(self) -> str:
        status_icon = {
            BudgetResult.ALLOWED: "✓",
            BudgetResult.WARNING: "⚠️",
            BudgetResult.REJECTED: "✗",
        }[self.result]

        return (
            f"{status_icon} ${self.used:.4f} / ${self.limit:.2f} "
            f"({self.percent_used:.1f}% used, ${self.remaining:.4f} remaining)"
        )


# approximate pricing (as of late 2024)
MODEL_PRICING = {
    # model: (input_per_1k, output_per_1k)
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "text-embedding-3-small": (0.00002, 0),
    "text-embedding-3-large": (0.00013, 0),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
) -> float:
    """estimate cost for a request"""
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    input_rate, output_rate = MODEL_PRICING[model]
    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)


@dataclass
class RequestBudget:
    """
    Per-request cost limit.

    Rejects requests that would exceed the limit.
    Useful for preventing accidental expensive requests.
    """

    max_cost: float = 0.10  # max $0.10 per request

    def check(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 500,
    ) -> BudgetStatus:
        """check if request is within budget"""
        cost = estimate_cost(model, estimated_input_tokens, estimated_output_tokens)

        if cost <= self.max_cost:
            return BudgetStatus(
                result=BudgetResult.ALLOWED,
                used=cost,
                limit=self.max_cost,
            )

        return BudgetStatus(
            result=BudgetResult.REJECTED,
            used=cost,
            limit=self.max_cost,
            message=f"Request would cost ${cost:.4f}, limit is ${self.max_cost:.2f}",
        )


@dataclass
class UserBudget:
    """
    Per-user budget with daily/monthly limits.

    Tracks usage and enforces quotas per user.
    """

    daily_limit: float = 5.00
    monthly_limit: float = 50.00
    warning_threshold: float = 0.8  # warn at 80%

    # usage tracking
    _daily_usage: dict[str, float] = field(default_factory=dict)
    _monthly_usage: dict[str, float] = field(default_factory=dict)
    _last_reset_day: int = -1
    _last_reset_month: int = -1

    def _reset_if_needed(self) -> None:
        """reset daily/monthly counters if period changed"""
        now = datetime.now(timezone.utc)

        if now.day != self._last_reset_day:
            self._daily_usage.clear()
            self._last_reset_day = now.day

        if now.month != self._last_reset_month:
            self._monthly_usage.clear()
            self._last_reset_month = now.month

    def check(self, user_id: str, cost: float) -> BudgetStatus:
        """check if user can spend this amount"""
        self._reset_if_needed()

        daily_used = self._daily_usage.get(user_id, 0)
        monthly_used = self._monthly_usage.get(user_id, 0)

        # check daily limit
        if daily_used + cost > self.daily_limit:
            return BudgetStatus(
                result=BudgetResult.REJECTED,
                used=daily_used,
                limit=self.daily_limit,
                message=f"Daily limit exceeded for user {user_id}",
            )

        # check monthly limit
        if monthly_used + cost > self.monthly_limit:
            return BudgetStatus(
                result=BudgetResult.REJECTED,
                used=monthly_used,
                limit=self.monthly_limit,
                message=f"Monthly limit exceeded for user {user_id}",
            )

        # check warning threshold
        daily_percent = (daily_used + cost) / self.daily_limit
        if daily_percent >= self.warning_threshold:
            return BudgetStatus(
                result=BudgetResult.WARNING,
                used=daily_used + cost,
                limit=self.daily_limit,
                message=f"Approaching daily limit ({daily_percent*100:.0f}%)",
            )

        return BudgetStatus(
            result=BudgetResult.ALLOWED,
            used=daily_used + cost,
            limit=self.daily_limit,
        )

    def record(self, user_id: str, cost: float) -> None:
        """record usage after successful request"""
        self._reset_if_needed()
        self._daily_usage[user_id] = self._daily_usage.get(user_id, 0) + cost
        self._monthly_usage[user_id] = self._monthly_usage.get(user_id, 0) + cost

    def get_usage(self, user_id: str) -> tuple[float, float]:
        """get (daily, monthly) usage for user"""
        self._reset_if_needed()
        return (
            self._daily_usage.get(user_id, 0),
            self._monthly_usage.get(user_id, 0),
        )


@dataclass
class CostGuard:
    """
    Combined cost protection with all budget checks.

    Usage:
        guard = CostGuard()
        status = guard.check_request(user_id, model, input_tokens)
        if status.result == BudgetResult.ALLOWED:
            response = call_llm(...)
            guard.record(user_id, model, input_tokens, output_tokens)
    """

    request_budget: RequestBudget = field(default_factory=RequestBudget)
    user_budget: UserBudget = field(default_factory=UserBudget)

    def check_request(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        estimated_output_tokens: int = 500,
    ) -> BudgetStatus:
        """check all budget constraints"""
        # 1. check request budget
        request_status = self.request_budget.check(
            model, input_tokens, estimated_output_tokens
        )
        if request_status.result == BudgetResult.REJECTED:
            return request_status

        # 2. check user budget
        cost = estimate_cost(model, input_tokens, estimated_output_tokens)
        user_status = self.user_budget.check(user_id, cost)

        return user_status

    def record(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """record usage after successful request, returns cost"""
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.user_budget.record(user_id, cost)
        return cost


# region Demo Functions

def demo_cost_budget() -> None:
    """demonstrate cost budget features"""
    print("=" * 60)
    print("  Cost Budget Demo")
    print("=" * 60)

    # 1. cost estimation
    print("\n1. COST ESTIMATION\n")

    examples = [
        ("gpt-4o-mini", 1000, 500, "typical chat"),
        ("gpt-4o", 1000, 500, "typical chat"),
        ("gpt-4o", 10000, 2000, "long document"),
        ("text-embedding-3-small", 1000, 0, "embedding"),
    ]

    for model, input_t, output_t, desc in examples:
        cost = estimate_cost(model, input_t, output_t)
        print(f"  {desc:20s} ({model}): ${cost:.4f}")

    # 2. request budget
    print("\n" + "-" * 60)
    print("\n2. REQUEST BUDGET\n")
    print("Config: max $0.05 per request")

    request_budget = RequestBudget(max_cost=0.05)

    requests = [
        ("Small chat", "gpt-4o-mini", 500, 200),
        ("Large chat", "gpt-4o", 5000, 2000),
        ("Embedding", "text-embedding-3-small", 1000, 0),
    ]

    for name, model, input_t, output_t in requests:
        status = request_budget.check(model, input_t, output_t)
        print(f"  {name}: {status}")

    # 3. user budget
    print("\n" + "-" * 60)
    print("\n3. USER BUDGET\n")
    print("Config: $1.00/day, $10.00/month, warn at 80%")

    user_budget = UserBudget(
        daily_limit=1.00,
        monthly_limit=10.00,
        warning_threshold=0.8,
    )

    user = "user_123"

    # simulate usage
    usages = [0.20, 0.30, 0.25, 0.15, 0.20]  # total = $1.10

    print(f"\n  Simulating requests for {user}:")
    for i, cost in enumerate(usages):
        status = user_budget.check(user, cost)
        print(f"  Request {i+1} (${cost:.2f}): {status}")

        if status.result != BudgetResult.REJECTED:
            user_budget.record(user, cost)

    # 4. combined guard
    print("\n" + "-" * 60)
    print("\n4. COST GUARD (COMBINED)\n")

    guard = CostGuard(
        request_budget=RequestBudget(max_cost=0.10),
        user_budget=UserBudget(daily_limit=2.00),
    )

    print("Simulating LLM workflow with cost protection:\n")

    code = cleandoc('''
        # check before request
        status = guard.check_request(
            user_id="user_456",
            model="gpt-4o-mini",
            input_tokens=1000,
        )

        if status.result == BudgetResult.REJECTED:
            return {"error": "Budget exceeded", "details": status.message}

        if status.result == BudgetResult.WARNING:
            log.warning(f"Budget warning: {status.message}")

        # make the LLM call
        response = client.chat.completions.create(...)

        # record actual usage
        guard.record(
            user_id="user_456",
            model="gpt-4o-mini",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    ''')
    print(code)

    print("\n" + "=" * 60)
    print("  Key Insight: Budget checks BEFORE the API call")
    print("  prevent surprises on your bill")
    print("=" * 60)

# endregion


if __name__ == "__main__":
    demo_cost_budget()
