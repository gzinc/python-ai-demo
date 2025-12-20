"""
Batching Strategies - reduce latency through batch operations.

Techniques:
1. Embedding batches - single API call for multiple texts
2. Parallel LLM calls - concurrent requests for independent tasks
3. Batch scheduling - queue and process in batches

Run with: uv run python -m phase5_production.03_optimization.batching
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass
class BatchResult:
    """result of a batch operation"""
    items_processed: int
    total_time_ms: float
    avg_time_per_item_ms: float

    @property
    def throughput(self) -> float:
        """items per second"""
        if self.total_time_ms == 0:
            return 0
        return self.items_processed / (self.total_time_ms / 1000)

    def __str__(self) -> str:
        return (
            f"Processed {self.items_processed} items in {self.total_time_ms:.1f}ms "
            f"(avg {self.avg_time_per_item_ms:.1f}ms/item, {self.throughput:.1f} items/sec)"
        )


class EmbeddingBatcher:
    """
    Batch embedding requests for efficiency.

    OpenAI embedding API accepts up to 2048 texts per request.
    Batching reduces:
    - Network round-trips
    - Rate limit consumption
    - Total latency
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        max_batch_size: int = 100,
    ):
        """
        Args:
            embed_fn: function that takes list of texts, returns list of embeddings
            max_batch_size: maximum texts per batch
        """
        self.embed_fn = embed_fn
        self.max_batch_size = max_batch_size

    def embed_all(self, texts: list[str]) -> tuple[list[list[float]], BatchResult]:
        """embed all texts in optimal batches"""
        start = time.perf_counter()

        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            embeddings = self.embed_fn(batch)
            all_embeddings.extend(embeddings)

        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / len(texts) if texts else 0

        return all_embeddings, BatchResult(len(texts), elapsed_ms, avg_ms)


class ParallelExecutor:
    """
    Execute independent LLM calls in parallel.

    Use when:
    - Multiple independent questions
    - Document processing (each doc independent)
    - Multi-step pipelines with parallel stages
    """

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers

    def execute_parallel(
        self,
        fn: Callable,
        items: list,
    ) -> tuple[list, BatchResult]:
        """
        Execute function on all items in parallel.

        Args:
            fn: function to apply to each item
            items: list of items to process
        """
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(fn, items))

        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / len(items) if items else 0

        return results, BatchResult(len(items), elapsed_ms, avg_ms)

    async def execute_parallel_async(
        self,
        fn: Callable,
        items: list,
    ) -> tuple[list, BatchResult]:
        """async version for async functions"""
        start = time.perf_counter()

        tasks = [fn(item) for item in items]
        results = await asyncio.gather(*tasks)

        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / len(items) if items else 0

        return list(results), BatchResult(len(items), elapsed_ms, avg_ms)


class BatchScheduler:
    """
    Queue requests and process in batches.

    Useful for high-throughput scenarios where you can
    tolerate small delays for efficiency gains.
    """

    def __init__(
        self,
        process_fn: Callable[[list], list],
        batch_size: int = 10,
        max_wait_ms: float = 100,
    ):
        """
        Args:
            process_fn: function to process a batch of items
            batch_size: target batch size
            max_wait_ms: max time to wait for batch to fill
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: list = []
        self._results: dict = {}
        self._counter = 0

    def submit(self, item) -> int:
        """submit item for processing, returns ticket ID"""
        ticket = self._counter
        self._counter += 1
        self._queue.append((ticket, item))

        # process if batch is full
        if len(self._queue) >= self.batch_size:
            self._process_batch()

        return ticket

    def flush(self) -> None:
        """process any remaining items in queue"""
        if self._queue:
            self._process_batch()

    def get_result(self, ticket: int):
        """get result for ticket (call flush first if needed)"""
        return self._results.get(ticket)

    def _process_batch(self) -> None:
        """process current queue as a batch"""
        if not self._queue:
            return

        tickets, items = zip(*self._queue)
        results = self.process_fn(list(items))

        for ticket, result in zip(tickets, results):
            self._results[ticket] = result

        self._queue.clear()


def demo_batching() -> None:
    """demonstrate batching strategies"""
    print("=" * 60)
    print("  Batching Strategies Demo")
    print("=" * 60)

    # 1. embedding batching
    print("\n1. EMBEDDING BATCHING\n")

    def mock_embed_single(text: str) -> list[float]:
        """simulate single embedding call (slow)"""
        time.sleep(0.05)  # 50ms per call
        return [0.1, 0.2, 0.3]

    def mock_embed_batch(texts: list[str]) -> list[list[float]]:
        """simulate batch embedding call (fast)"""
        time.sleep(0.08)  # 80ms for entire batch
        return [[0.1, 0.2, 0.3] for _ in texts]

    texts = [f"Document {i}" for i in range(20)]

    # sequential (slow)
    print("Sequential (20 calls):")
    start = time.perf_counter()
    sequential_results = [mock_embed_single(t) for t in texts]
    sequential_ms = (time.perf_counter() - start) * 1000
    print(f"  Time: {sequential_ms:.1f}ms ({sequential_ms/len(texts):.1f}ms/item)")

    # batched (fast)
    print("\nBatched (1 call):")
    batcher = EmbeddingBatcher(mock_embed_batch, max_batch_size=100)
    batch_results, stats = batcher.embed_all(texts)
    print(f"  {stats}")
    print(f"  Speedup: {sequential_ms / stats.total_time_ms:.1f}x faster")

    # 2. parallel execution
    print("\n" + "-" * 60)
    print("\n2. PARALLEL LLM CALLS\n")

    def mock_llm_call(prompt: str) -> str:
        """simulate LLM call"""
        time.sleep(0.2)  # 200ms per call
        return f"Response to: {prompt[:20]}..."

    prompts = [f"Question {i}: What is...?" for i in range(5)]

    # sequential
    print("Sequential (5 calls):")
    start = time.perf_counter()
    seq_results = [mock_llm_call(p) for p in prompts]
    seq_ms = (time.perf_counter() - start) * 1000
    print(f"  Time: {seq_ms:.1f}ms")

    # parallel
    print("\nParallel (5 concurrent):")
    executor = ParallelExecutor(max_workers=5)
    par_results, stats = executor.execute_parallel(mock_llm_call, prompts)
    print(f"  {stats}")
    print(f"  Speedup: {seq_ms / stats.total_time_ms:.1f}x faster")

    # 3. batch scheduling
    print("\n" + "-" * 60)
    print("\n3. BATCH SCHEDULING\n")

    def process_batch(items: list) -> list:
        """process items in batch"""
        time.sleep(0.1)  # 100ms for batch
        return [f"Processed: {item}" for item in items]

    scheduler = BatchScheduler(process_batch, batch_size=5)

    print("Submitting 12 items (batch_size=5):")
    tickets = []
    for i in range(12):
        ticket = scheduler.submit(f"Item_{i}")
        tickets.append(ticket)
        print(f"  Submitted Item_{i} (ticket={ticket})")
        if (i + 1) % 5 == 0:
            print("  → Batch processed!")

    scheduler.flush()  # process remaining
    print("  → Final batch processed!")

    print("\nResults:")
    for ticket in tickets[:3]:
        result = scheduler.get_result(ticket)
        print(f"  Ticket {ticket}: {result}")
    print("  ...")

    print("\n" + "=" * 60)
    print("  Key Insight: Batching reduces latency, not cost")
    print("  (same tokens, fewer round-trips)")
    print("=" * 60)


if __name__ == "__main__":
    demo_batching()
