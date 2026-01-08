"""
Semantic Cache - cache LLM responses by embedding similarity.

Unlike exact-match caching, semantic caching finds similar queries:
- "What is ML?" and "Explain machine learning" → cache hit
- Reduces LLM costs significantly for similar queries

Run with: uv run python -m phase5_production.01_api_design.semantic_cache
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np


@dataclass
class CacheEntry:
    """single cache entry with embedding and response"""
    query: str
    embedding: np.ndarray
    response: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0


class SemanticCache:
    """
    LLM response cache using embedding similarity.

    Key insight: similar prompts often have similar answers.
    Cache hit when cosine similarity > threshold.

    Usage:
        cache = SemanticCache(threshold=0.95)

        # check cache
        cached = cache.get(query_embedding)
        if cached:
            return cached.response

        # cache miss - call LLM and store
        response = await llm.generate(query)
        cache.set(query, query_embedding, response)
    """

    def __init__(self, threshold: float = 0.95, max_entries: int = 1000):
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(self, query_embedding: np.ndarray) -> CacheEntry | None:
        """
        Find cached response by embedding similarity.

        Returns entry if similarity > threshold, else None.
        """
        best_match: CacheEntry | None = None
        best_score = 0.0

        for entry in self._entries:
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > self.threshold and score > best_score:
                best_match = entry
                best_score = score

        if best_match:
            best_match.hit_count += 1

        return best_match

    def set(self, query: str, embedding: np.ndarray, response: str) -> None:
        """store query-response pair in cache"""
        # evict oldest if at capacity
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)

        self._entries.append(CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
        ))

    def stats(self) -> dict:
        """cache statistics"""
        total_hits = sum(e.hit_count for e in self._entries)
        return {
            "entries": len(self._entries),
            "total_hits": total_hits,
            "threshold": self.threshold,
        }

    def clear(self) -> None:
        """clear all cache entries"""
        self._entries.clear()


class MockEmbedder:
    """
    Mock embedder for testing.

    In production: use OpenAI, sentence-transformers, etc.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """
        Generate deterministic mock embedding from text.

        Real implementation would call embedding API.
        """
        # deterministic hash-based embedding for testing
        np.random.seed(hash(text.lower().strip()) % (2**32))
        return np.random.randn(self.dim).astype(np.float32)


# region Demo Functions

def demo_semantic_cache() -> None:
    """demonstrate semantic caching behavior"""
    print("=" * 60)
    print("  Semantic Cache Demo")
    print("=" * 60)

    embedder = MockEmbedder()
    cache = SemanticCache(threshold=0.95)

    # simulate queries
    queries = [
        ("What is machine learning?", "ML is a subset of AI..."),
        ("Explain machine learning", None),  # should hit cache
        ("What is deep learning?", "DL uses neural networks..."),
        ("How does deep learning work?", None),  # might hit cache
        ("What is Python?", "Python is a programming language..."),
    ]

    print("\nProcessing queries:\n")

    for query, mock_response in queries:
        embedding = embedder.embed(query)
        cached = cache.get(embedding)

        if cached:
            print(f"  ✓ CACHE HIT: '{query}'")
            print(f"    → Matched: '{cached.query}'")
            print(f"    → Response: {cached.response[:50]}...")
        else:
            print(f"  ✗ CACHE MISS: '{query}'")
            if mock_response:
                cache.set(query, embedding, mock_response)
                print(f"    → Stored response")
        print()

    print(f"Cache stats: {cache.stats()}")

    print("\n" + "=" * 60)
    print("  Key insight: Similar queries hit cache, reducing LLM costs")
    print("=" * 60)

# endregion


if __name__ == "__main__":
    demo_semantic_cache()
