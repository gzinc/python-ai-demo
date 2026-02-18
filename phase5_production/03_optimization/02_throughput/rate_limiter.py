"""
Rate Limiting - prevent API throttling with client-side limits.

Algorithms:
1. Token Bucket - smooth rate limiting with burst capacity
2. Sliding Window - strict per-window limits
3. Adaptive - adjust based on API responses

Run with: uv run python -m phase5_production.03_optimization.rate_limiter
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock


class RateLimitResult(Enum):
    """result of rate limit check"""
    ALLOWED = "allowed"
    WAIT = "wait"
    REJECTED = "rejected"


@dataclass
class RateLimitStatus:
    """status of rate limiter"""
    result: RateLimitResult
    wait_time_ms: float = 0
    tokens_remaining: float = 0
    message: str = ""

    def __str__(self) -> str:
        if self.result == RateLimitResult.ALLOWED:
            return f"✓ Allowed ({self.tokens_remaining:.0f} tokens remaining)"
        elif self.result == RateLimitResult.WAIT:
            return f"⏳ Wait {self.wait_time_ms:.0f}ms"
        else:
            return f"✗ Rejected: {self.message}"


class TokenBucket:
    """
    Token Bucket rate limiter.

    How it works:
    - Bucket holds up to `capacity` tokens
    - Tokens refill at `refill_rate` per second
    - Each request consumes tokens
    - If not enough tokens, wait or reject

    Good for:
    - Smooth rate limiting
    - Allowing bursts up to capacity
    - API rate limits (e.g., 10K tokens/min)
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        initial_tokens: float | None = None,
    ):
        """
        Args:
            capacity: maximum tokens in bucket
            refill_rate: tokens added per second
            initial_tokens: starting tokens (default: capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.monotonic()
        self._lock = Lock()

    def _refill(self) -> None:
        """refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, tokens: float = 1, wait: bool = True) -> RateLimitStatus:
        """
        Acquire tokens from bucket.

        Args:
            tokens: number of tokens to acquire
            wait: if True, wait for tokens; if False, reject immediately
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    tokens_remaining=self.tokens,
                )

            # not enough tokens
            if not wait:
                return RateLimitStatus(
                    result=RateLimitResult.REJECTED,
                    tokens_remaining=self.tokens,
                    message=f"Need {tokens}, have {self.tokens:.1f}",
                )

            # calculate wait time
            needed = tokens - self.tokens
            wait_seconds = needed / self.refill_rate

            return RateLimitStatus(
                result=RateLimitResult.WAIT,
                wait_time_ms=wait_seconds * 1000,
                tokens_remaining=self.tokens,
            )

    def wait_and_acquire(self, tokens: float = 1) -> RateLimitStatus:
        """acquire tokens, waiting if necessary"""
        status = self.acquire(tokens, wait=True)

        if status.result == RateLimitResult.WAIT:
            time.sleep(status.wait_time_ms / 1000)
            return self.acquire(tokens, wait=False)

        return status


@dataclass
class SlidingWindow:
    """
    Sliding Window rate limiter.

    How it works:
    - Track requests in current window
    - Reject if window limit exceeded
    - Window slides with time

    Good for:
    - Strict per-minute/per-hour limits
    - Simple implementation
    - When bursts should be prevented
    """

    limit: int
    window_seconds: float
    _requests: list[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def _clean_old_requests(self) -> None:
        """remove requests outside current window"""
        cutoff = time.monotonic() - self.window_seconds
        self._requests = [timestamp for timestamp in self._requests if timestamp > cutoff]

    def acquire(self) -> RateLimitStatus:
        """try to acquire a request slot"""
        with self._lock:
            self._clean_old_requests()

            if len(self._requests) < self.limit:
                self._requests.append(time.monotonic())
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    tokens_remaining=self.limit - len(self._requests),
                )

            # calculate when oldest request expires
            oldest = min(self._requests)
            wait_time = (oldest + self.window_seconds) - time.monotonic()

            return RateLimitStatus(
                result=RateLimitResult.WAIT,
                wait_time_ms=max(0, wait_time * 1000),
                tokens_remaining=0,
                message=f"Window full ({self.limit} requests)",
            )


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on API responses.

    How it works:
    - Start with conservative limits
    - If 429 errors, back off
    - If success streak, gradually increase
    - Respects Retry-After headers

    Good for:
    - Unknown or variable rate limits
    - Multi-tenant scenarios
    - Graceful degradation
    """

    def __init__(
        self,
        initial_rps: float = 10,
        min_rps: float = 1,
        max_rps: float = 100,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
    ):
        self.current_rps = initial_rps
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor

        self.success_streak = 0
        self.last_request = 0.0
        self._lock = Lock()

    def record_success(self) -> None:
        """record successful request"""
        with self._lock:
            self.success_streak += 1

            # gradually increase rate after 10 successes
            if self.success_streak >= 10:
                self.current_rps = min(
                    self.max_rps,
                    self.current_rps * self.recovery_factor
                )
                self.success_streak = 0

    def record_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        """record rate limit hit (429 error)"""
        with self._lock:
            self.success_streak = 0
            self.current_rps = max(
                self.min_rps,
                self.current_rps * self.backoff_factor
            )

            if retry_after_seconds:
                # sleep for retry-after duration
                time.sleep(retry_after_seconds)

    def acquire(self) -> RateLimitStatus:
        """acquire permission for a request"""
        with self._lock:
            now = time.monotonic()
            min_interval = 1.0 / self.current_rps

            if now - self.last_request >= min_interval:
                self.last_request = now
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    tokens_remaining=self.current_rps,
                )

            wait_time = min_interval - (now - self.last_request)
            return RateLimitStatus(
                result=RateLimitResult.WAIT,
                wait_time_ms=wait_time * 1000,
                tokens_remaining=self.current_rps,
            )

    @property
    def status(self) -> str:
        return f"Adaptive: {self.current_rps:.1f} RPS (streak: {self.success_streak})"


# region Demo Functions

def demo_rate_limiting() -> None:
    """demonstrate rate limiting strategies"""
    print("=" * 60)
    print("  Rate Limiting Demo")
    print("=" * 60)

    # 1. token bucket
    print("\n1. TOKEN BUCKET\n")
    print("Config: capacity=100 tokens, refill=10 tokens/sec")

    bucket = TokenBucket(capacity=100, refill_rate=10)

    # simulate requests
    requests = [
        ("Small request", 10),
        ("Medium request", 30),
        ("Large request", 50),
        ("Another large", 50),  # will need to wait
    ]

    for name, tokens in requests:
        status = bucket.acquire(tokens, wait=False)
        print(f"  {name} ({tokens} tokens): {status}")

    print("\n  Waiting 2 seconds for refill...")
    time.sleep(2)
    status = bucket.acquire(20, wait=False)
    print(f"  After wait (20 tokens): {status}")

    # 2. sliding window
    print("\n" + "-" * 60)
    print("\n2. SLIDING WINDOW\n")
    print("Config: 5 requests per 1 second window")

    window = SlidingWindow(limit=5, window_seconds=1.0)

    for i in range(7):
        status = window.acquire()
        print(f"  Request {i+1}: {status}")

    print("\n  Waiting 1 second for window to slide...")
    time.sleep(1)
    status = window.acquire()
    print(f"  After wait: {status}")

    # 3. adaptive
    print("\n" + "-" * 60)
    print("\n3. ADAPTIVE RATE LIMITER\n")
    print("Simulating success and rate limit scenarios")

    adaptive = AdaptiveRateLimiter(initial_rps=10)

    print(f"\n  Initial: {adaptive.status}")

    # simulate successes
    for _ in range(15):
        adaptive.record_success()
    print(f"  After 15 successes: {adaptive.status}")

    # simulate rate limit
    adaptive.record_rate_limit()
    print(f"  After rate limit hit: {adaptive.status}")

    # more successes
    for _ in range(20):
        adaptive.record_success()
    print(f"  After 20 more successes: {adaptive.status}")

    print("\n" + "=" * 60)
    print("  Key Insight: Client-side limits prevent 429 errors")
    print("  and make your app a good API citizen")
    print("=" * 60)

# endregion


if __name__ == "__main__":
    demo_rate_limiting()
