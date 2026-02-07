"""
Token optimization examples and demos.

Run with: uv run python -m phase5_production.03_optimization.01_compression.examples
"""

from inspect import cleandoc

from .compressors import NaiveCompressor, LLMLingua2Compressor
from .truncation import ContextTruncator
from .response_limits import ResponseLimiter

from common.demo_menu import Demo, MenuRunner


def demo_naive_compression() -> None:
    """demo naive compression (no model download)"""
    print("\n1. NAIVE COMPRESSION (for comparison)\n")

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
    print("\n2. LLMLINGUA-2 COMPRESSION (production)\n")

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

    chunks = [
        "Python is a high-level programming language known for its simplicity.",
        "Guido van Rossum created Python in 1991 while at CWI in Netherlands.",
        "Python emphasizes code readability with significant whitespace.",
        "The language supports multiple programming paradigms.",
        "Python has a large standard library often called 'batteries included'.",
    ]
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



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Naive Compression", "naive compression", demo_naive_compression),
    Demo("2", "LLMLingua-2", "llmlingua-2", demo_llmlingua2),
    Demo("3", "Context Truncation", "context truncation", demo_context_truncation),
    Demo("4", "Response Limits", "response limits", demo_response_limits),
]

# endregion

def main() -> None:
    """run all demos"""
    print("=" * 60)
    print("  Token Optimization Demo")
    print("  Multiple strategies for token reduction")
    print("=" * 60)

    
    runner = MenuRunner(DEMOS, title="Response Compression - Examples")
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
