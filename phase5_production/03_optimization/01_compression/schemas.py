"""
Token optimization schemas.

Run with: uv run python -m phase5_production.03_optimization.01_compression.schemas
"""

from dataclasses import dataclass


@dataclass
class TokenStats:
    """token usage statistics"""
    original: int
    optimized: int

    @property
    def saved(self) -> int:
        return self.original - self.optimized

    @property
    def reduction_percent(self) -> float:
        if self.original == 0:
            return 0.0
        return (self.saved / self.original) * 100

    def __str__(self) -> str:
        return (
            f"Tokens: {self.original} → {self.optimized} "
            f"(saved {self.saved}, {self.reduction_percent:.1f}% reduction)"
        )


if __name__ == "__main__":
    stats = TokenStats(original=1000, optimized=300)
    print(stats)
