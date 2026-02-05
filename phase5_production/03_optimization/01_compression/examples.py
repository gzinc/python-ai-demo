"""
Token optimization examples and demos.

Run with: uv run python -m phase5_production.03_optimization.01_compression.examples
"""

from inspect import cleandoc

from .compressors import NaiveCompressor, LLMLingua2Compressor
from .truncation import ContextTruncator
from .response_limits import ResponseLimiter


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


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 60)
    print("  Token Optimization - Interactive Demos")
    print("=" * 60)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Naive Compression", "simple token reduction (demo only)"),
        ("2", "LLMLingua-2", "production compression (~400MB download)"),
        ("3", "Context Truncation", "score-based chunk selection"),
        ("4", "Response Limits", "task-specific token limits"),
    ]

    for num, name, desc in demos:
        print(f"    [{num}] {name}")
        print(f"        {desc}")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 60)


def run_selected_demos(selections: str) -> bool:
    """run selected demos based on user input"""
    selections = selections.lower().strip()

    if selections == 'q':
        return False

    demo_map = {
        '1': ("Naive Compression", demo_naive_compression),
        '2': ("LLMLingua-2", demo_llmlingua2),
        '3': ("Context Truncation", demo_context_truncation),
        '4': ("Response Limits", demo_response_limits),
    }

    if selections == 'a':
        # run all demos
        for name, demo_func in demo_map.values():
            if name == "LLMLingua-2":
                print("\n⚠️  Skipping LLMLingua-2 (requires ~400MB model download)")
                print("   Select demo '2' individually to run it")
                continue
            demo_func()
            print("\n" + "-" * 60)
    else:
        # parse comma-separated selections
        selected = [s.strip() for s in selections.split(',')]
        for sel in selected:
            if sel in demo_map:
                name, demo_func = demo_map[sel]
                demo_func()
                print("\n" + "-" * 60)
            else:
                print(f"⚠️  Invalid selection: {sel}")

    return True


def main() -> None:
    """run all demos"""
    print("=" * 60)
    print("  Token Optimization Demo")
    print("  Multiple strategies for token reduction")
    print("=" * 60)

    while True:
        show_menu()
        selection = input("\nSelect demos to run (comma-separated) or 'a' for all: ").strip()

        if not selection:
            continue

        if not run_selected_demos(selection):
            break

        print("\n" + "=" * 60)
        print("  ✅ Demos complete!")
        print("=" * 60)
        print("\n💡 Key Insight:")
        print("  Multiple strategies for token reduction:")
        print("  • Naive: Simple pattern matching (demo only)")
        print("  • LLMLingua-2: Smart semantic compression")
        print("  • Truncation: Score-based selection")
        print("  • Limits: Task-specific token budgets")

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

    print("\n" + "=" * 60)
    print("  Thanks for exploring Token Optimization!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
