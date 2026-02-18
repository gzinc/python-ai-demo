"""
Context truncation - fit context into token limits.

Run with: uv run python -m phase5_production.03_optimization.01_compression.truncation
"""

from .schemas import TokenStats


class ContextTruncator:
    """
    Truncate context to fit token limits while preserving relevance.

    Strategies:
    - Keep first N chunks (recency)
    - Keep highest relevance chunks
    - Smart truncation with ellipsis
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def truncate_by_tokens(self, text: str) -> tuple[str, TokenStats]:
        """truncate text to fit token limit"""
        original_tokens = self._estimate_tokens(text)

        if original_tokens <= self.max_tokens:
            return text, TokenStats(original_tokens, original_tokens)

        # estimate chars needed
        target_chars = self.max_tokens * 4

        # truncate with ellipsis
        truncated = text[:target_chars - 3].rsplit(' ', 1)[0] + '...'
        optimized_tokens = self._estimate_tokens(truncated)

        return truncated, TokenStats(original_tokens, optimized_tokens)

    def truncate_chunks(
        self,
        chunks: list[str],
        scores: list[float] | None = None,
    ) -> tuple[list[str], TokenStats]:
        """
        Truncate list of chunks to fit token limit.

        Args:
            chunks: list of text chunks
            scores: optional relevance scores (higher = more relevant)
        """
        if not chunks:
            return [], TokenStats(0, 0)

        original_tokens = sum(self._estimate_tokens(c) for c in chunks)

        if original_tokens <= self.max_tokens:
            return chunks, TokenStats(original_tokens, original_tokens)

        # sort by relevance if scores provided
        if scores:
            indexed = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
            sorted_chunks = [chunk for chunk, _ in indexed]
        else:
            sorted_chunks = chunks

        # keep chunks until we hit the limit
        kept = []
        total_tokens = 0

        for chunk in sorted_chunks:
            chunk_tokens = self._estimate_tokens(chunk)
            if total_tokens + chunk_tokens > self.max_tokens:
                break
            kept.append(chunk)
            total_tokens += chunk_tokens

        return kept, TokenStats(original_tokens, total_tokens)

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4


if __name__ == "__main__":
    chunks = [
        "Python is a high-level programming language known for its simplicity.",
        "Guido van Rossum created Python in 1991 while at CWI in Netherlands.",
        "Python emphasizes code readability with significant whitespace.",
        "The language supports multiple programming paradigms.",
        "Python has a large standard library often called 'batteries included'.",
    ]
    scores = [0.9, 0.7, 0.6, 0.4, 0.8]

    print(f"Original: {len(chunks)} chunks")

    truncator = ContextTruncator(max_tokens=100)
    kept, stats = truncator.truncate_chunks(chunks, scores)

    print(f"Kept: {len(kept)} chunks")
    print(stats)
