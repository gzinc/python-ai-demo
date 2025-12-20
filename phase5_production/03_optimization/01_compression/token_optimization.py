"""
Token Optimization - reduce tokens to reduce cost.

Strategies:
1. Prompt compression - LLMLingua-2 (production) or naive regex (fallback)
2. Context truncation - keep most relevant chunks
3. Response limits - set appropriate max_tokens

Run with: uv run python -m phase5_production.03_optimization.token_optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import cleandoc


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


# =============================================================================
# Prompt Compression - Abstract Base
# =============================================================================


class PromptCompressor(ABC):
    """abstract base for prompt compressors"""

    @abstractmethod
    def compress(self, text: str, rate: float = 0.5) -> tuple[str, TokenStats]:
        """
        Compress prompt text.

        Args:
            text: prompt to compress
            rate: target compression rate (0.5 = keep 50% of tokens)

        Returns:
            (compressed_text, stats)
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """rough token estimate (~4 chars per token for English)"""
        return len(text) // 4


# =============================================================================
# LLMLingua-2 Compressor (Production)
# =============================================================================


class LLMLingua2Compressor(PromptCompressor):
    """
    Production prompt compressor using Microsoft LLMLingua-2.

    Uses a BERT model trained specifically for compression - learns which
    tokens are essential for meaning, not just which are predictable.

    Models:
    - llmlingua-2-bert-base-multilingual-cased-meetingbank (default, ~400MB)
    - llmlingua-2-xlm-roberta-large-meetingbank (larger, more accurate)

    First call downloads the model (~400MB). Subsequent calls use cache.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device: str = "cpu",  # use 'cuda' if you have GPU + torch with CUDA
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
        """
        from llmlingua import PromptCompressor as LLMCompressor

        self._compressor = LLMCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )

    def compress(self, text: str, rate: float = 0.5) -> tuple[str, TokenStats]:
        """
        Compress using LLMLingua-2.

        Args:
            text: prompt to compress
            rate: target compression rate (0.5 = keep 50%)

        Returns:
            (compressed_text, stats)
        """
        result = self._compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=["!", ".", "?", "\n", ":"],  # keep punctuation
            drop_consecutive=True,  # remove consecutive dropped tokens
        )

        return result["compressed_prompt"], TokenStats(
            original=result["origin_tokens"],
            optimized=result["compressed_tokens"],
        )


# =============================================================================
# Naive Compressor (Fallback/Demo)
# =============================================================================


class NaiveCompressor(PromptCompressor):
    """
    Simple regex-based compressor for demo/fallback.

    NOT for production - only removes obvious filler words and phrases.
    Use LLMLingua2Compressor for real compression.
    """

    import re

    FILLER_WORDS = {
        'please', 'kindly', 'just', 'simply', 'basically',
        'actually', 'really', 'very', 'quite', 'rather',
    }

    PHRASE_MAP = {
        'in order to': 'to',
        'make sure to': 'ensure',
        'due to the fact that': 'because',
        'in the event that': 'if',
        'at this point in time': 'now',
        'it is important to note that': 'note:',
    }

    def compress(self, text: str, rate: float = 0.5) -> tuple[str, TokenStats]:
        """compress using regex rules (ignores rate parameter)"""
        import re

        original_tokens = self._estimate_tokens(text)
        result = text

        # collapse whitespace
        result = re.sub(r'\n\s*\n', '\n\n', result)
        result = re.sub(r'[ \t]+', ' ', result)

        # replace verbose phrases
        for verbose, concise in self.PHRASE_MAP.items():
            result = re.sub(rf'\b{verbose}\b', concise, result, flags=re.IGNORECASE)

        # remove filler words
        for filler in self.FILLER_WORDS:
            result = re.sub(rf'\b{filler}\b\s*', '', result, flags=re.IGNORECASE)

        # cleanup
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()

        return result, TokenStats(original_tokens, self._estimate_tokens(result))


# =============================================================================
# Context Truncation
# =============================================================================


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
            indexed = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            sorted_chunks = [c for c, _ in indexed]
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


# =============================================================================
# Response Limits
# =============================================================================


class ResponseLimiter:
    """
    Set appropriate max_tokens for responses.

    Prevents over-generation and reduces costs.
    """

    TASK_LIMITS = {
        'classification': 50,
        'extraction': 200,
        'summarization': 500,
        'qa': 300,
        'generation': 1000,
        'code': 2000,
        'default': 500,
    }

    def get_limit(self, task_type: str) -> int:
        """get recommended max_tokens for task type"""
        return self.TASK_LIMITS.get(task_type, self.TASK_LIMITS['default'])

    def estimate_needed(self, prompt: str) -> int:
        """estimate tokens needed based on prompt characteristics"""
        prompt_tokens = len(prompt) // 4

        if any(word in prompt.lower() for word in ['summarize', 'brief', 'short']):
            return min(300, prompt_tokens)
        if any(word in prompt.lower() for word in ['explain', 'detail', 'elaborate']):
            return max(500, prompt_tokens * 2)
        if any(word in prompt.lower() for word in ['code', 'implement', 'write function']):
            return max(1000, prompt_tokens)

        return max(200, min(1000, prompt_tokens))


# =============================================================================
# Demo
# =============================================================================


def demo_naive_compression() -> None:
    """demo naive compression (no model download)"""
    print("\n2. NAIVE COMPRESSION (for comparison)\n")

    verbose_prompt = cleandoc('''
        Please kindly analyze the following text and make sure to identify
        the main themes. It is important to note that you should basically
        focus on the key points. In order to provide a good response, please
        be sure to consider all aspects of the text very carefully.

        Due to the fact that this is a complex topic, please take your time
        and provide a detailed analysis as well as a summary.
    ''')

    print(f"Original:\n{verbose_prompt}\n")

    compressor = NaiveCompressor()
    compressed, stats = compressor.compress(verbose_prompt)

    print(f"Compressed:\n{compressed}\n")
    print(f"Stats: {stats}")
    print("\n⚠️  Naive = demo only. Use LLMLingua2Compressor for production.")


def demo_llmlingua2() -> None:
    """demo LLMLingua-2 (downloads model on first run)"""
    print("\n1. LLMLINGUA-2 COMPRESSION (production)\n")

    prompt = cleandoc('''
        The user is asking about the weather forecast for tomorrow.
        Based on the available meteorological data from the national
        weather service and various satellite imagery analysis systems,
        it appears that there will likely be partly cloudy conditions
        with a high probability of scattered showers in the afternoon
        hours, particularly in the northern regions of the metropolitan
        area where temperatures are expected to reach approximately 72
        degrees Fahrenheit with humidity levels around 65 percent.
    ''')

    print(f"Original ({len(prompt)} chars):\n{prompt}\n")

    print("loading LLMLingua-2 model (first run downloads ~400MB)...")
    compressor = LLMLingua2Compressor()

    for rate in [0.7, 0.5, 0.3]:
        compressed, stats = compressor.compress(prompt, rate=rate)
        print(f"\nrate={rate}: {stats}")
        print(f"Result: {compressed}")


def demo_context_truncation() -> None:
    """demo context truncation"""
    print("\n3. CONTEXT TRUNCATION\n")

    # in real RAG: chunks come from vector DB search
    # scores are similarity scores from the embedding search
    chunks = [
        "Python is a high-level programming language known for its simplicity.",
        "Guido van Rossum created Python in 1991 while at CWI in Netherlands.",
        "Python emphasizes code readability with significant whitespace.",
        "The language supports multiple programming paradigms.",
        "Python has a large standard library often called 'batteries included'.",
    ]
    # simulated relevance scores (in production: from vector_db.query())
    scores = [0.9, 0.7, 0.6, 0.4, 0.8]

    print(f"Original: {len(chunks)} chunks")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"  [{i+1}] score={score:.1f}: {chunk[:50]}...")

    truncator = ContextTruncator(max_tokens=100)
    kept, stats = truncator.truncate_chunks(chunks, scores)

    print(f"\nKept (max 100 tokens): {len(kept)} chunks")
    for chunk in kept:
        print(f"  ✓ {chunk[:50]}...")
    print(f"\n{stats}")


def demo_response_limits() -> None:
    """demo response limits"""
    print("\n4. RESPONSE LIMITS\n")

    limiter = ResponseLimiter()

    for task in ['classification', 'summarization', 'code', 'qa']:
        print(f"  {task:20s} → max_tokens={limiter.get_limit(task)}")


def main() -> None:
    """run demos"""
    print("=" * 60)
    print("  Token Optimization Demo")
    print("=" * 60)

    # 1. LLMLingua-2 (production)
    # demo_llmlingua2()
    print("\n" + "-" * 60)

    # 2. naive (for comparison)
    demo_naive_compression()
    print("\n" + "-" * 60)

    # 3. context truncation
    demo_context_truncation()
    print("\n" + "-" * 60)

    # 4. response limits
    demo_response_limits()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
