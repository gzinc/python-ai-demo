"""
Prompt compressors - reduce token count while preserving meaning.

Compressors:
1. LLMLingua2Compressor - production, uses BERT model
2. NaiveCompressor - fallback/demo, regex-based

Run with: uv run python -m phase5_production.03_optimization.01_compression.compressors
"""

from abc import ABC, abstractmethod

from .schemas import TokenStats


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
        device: str = "cpu",
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
            force_tokens=["!", ".", "?", "\n", ":"],
            drop_consecutive=True,
        )

        return result["compressed_prompt"], TokenStats(
            original=result["origin_tokens"],
            optimized=result["compressed_tokens"],
        )


class NaiveCompressor(PromptCompressor):
    """
    Simple regex-based compressor for demo/fallback.

    NOT for production - only removes obvious filler words and phrases.
    Use LLMLingua2Compressor for real compression.
    """

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


if __name__ == "__main__":
    from inspect import cleandoc

    text = cleandoc('''
        Please kindly analyze the following text and make sure to identify
        the main themes. It is important to note that you should basically
        focus on the key points. In order to provide a good response, please
        be sure to consider all aspects of the text very carefully.
    ''')

    print("=" * 60)
    print("  Prompt Compressors Demo")
    print("=" * 60)

    print(f"\nOriginal ({len(text)} chars):\n{text}\n")

    # 1. naive compressor (fast, no model)
    print("-" * 60)
    print("1. NaiveCompressor (regex-based, demo only)")
    print("-" * 60)

    naive = NaiveCompressor()
    compressed, stats = naive.compress(text)

    print(f"Compressed:\n{compressed}\n")
    print(f"Stats: {stats}")

    # 2. LLMLingua-2 compressor (production)
    print("\n" + "-" * 60)
    print("2. LLMLingua2Compressor (production)")
    print("-" * 60)
    print("loading LLMLingua-2 model (first run downloads ~400MB)...")

    compressor = LLMLingua2Compressor()

    for rate in [0.7, 0.5, 0.3]:
        compressed, stats = compressor.compress(text, rate=rate)
        print(f"\nrate={rate}: {stats}")
        print(f"Result: {compressed}")
