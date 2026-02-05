"""
RAG Pipeline Examples - Interactive demos and usage patterns

This module demonstrates various ways to use the RAG system:
1. Basic usage - add documents, query
2. Different chunking strategies
3. Custom retrieval settings
4. Working with files

Run with: uv run python phase3_llm_applications/01_rag_system/examples.py
"""

from rag_pipeline import RAGPipeline
from schemas import Document
from chunking import chunk_document


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SAMPLE DOCUMENTS
# ─────────────────────────────────────────────────────────────

PYTHON_FUNCTIONS = """
Python Functions Guide

Functions in Python are defined using the def keyword. They allow you to
organize code into reusable blocks. A function can take parameters and
return values.

To define a function:
def function_name(parameters):
    # function body
    return result

Functions can have default parameter values, making some arguments optional.
You can also use *args for variable positional arguments and **kwargs for
variable keyword arguments.
"""

PYTHON_VARIABLES = """
Python Variables and Data Types

Variables in Python store data values. Unlike other languages, you don't
need to declare variable types - Python infers them automatically.

Common data types include:
- int: whole numbers (42)
- float: decimal numbers (3.14)
- str: text strings ("hello")
- bool: True or False
- list: ordered collections [1, 2, 3]
- dict: key-value pairs {"key": "value"}

Variables are created when you assign a value using the = operator.
"""

PYTHON_ERRORS = """
Python Error Handling

Python uses try-except blocks to handle errors gracefully. This prevents
your program from crashing when something unexpected happens.

Basic syntax:
try:
    # code that might raise an error
    risky_operation()
except SomeError as e:
    # handle the error
    print(f"Error occurred: {e}")
finally:
    # always runs, cleanup code
    cleanup()

Common exceptions include ValueError, TypeError, FileNotFoundError, and
KeyError. You can also create custom exceptions by inheriting from Exception.
"""


# ─────────────────────────────────────────────────────────────
# EXAMPLE 1: BASIC RAG USAGE
# ─────────────────────────────────────────────────────────────


def example_basic_rag():
    """
    basic RAG pipeline usage

    Flow:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Create  │───►│   Add    │───►│  Query   │
    │ Pipeline │    │   Docs   │    │          │
    └──────────┘    └──────────┘    └──────────┘
    """
    print_section("Example 1: Basic RAG Usage")

    import tempfile
    temp_dir = tempfile.mkdtemp()

    # create pipeline
    rag = RAGPipeline(
        collection_name="basic_demo",
        chunk_size=300,
        chunk_overlap=50,
        top_k=2,
        persist_directory=temp_dir,
    )

    # add documents
    print("\n1. Adding documents...")
    rag.add_text(PYTHON_FUNCTIONS, source="functions_guide.md")
    rag.add_text(PYTHON_VARIABLES, source="variables_guide.md")
    rag.add_text(PYTHON_ERRORS, source="error_handling.md")

    stats = rag.get_stats()
    print(f"   Indexed {stats['document_count']} chunks")

    # test retrieval
    print("\n2. Testing retrieval...")
    results = rag.retrieve("How do I define a function?")
    for r in results:
        print(f"   - [{r.source}] similarity: {r.similarity:.3f}")

    # full RAG query
    print("\n3. Full RAG query...")
    question = "How do I handle errors in Python?"
    print(f"   Q: {question}")

    answer, context = rag.query(question, return_context=True)
    print(f"\n   A: {answer[:200]}..." if len(answer) > 200 else f"\n   A: {answer}")

    # cleanup
    rag.clear()


# ─────────────────────────────────────────────────────────────
# EXAMPLE 2: CHUNKING STRATEGIES
# ─────────────────────────────────────────────────────────────


def example_chunking_strategies():
    """demonstrate different chunking approaches"""
    print_section("Example 2: Chunking Strategies")

    doc = Document(content=PYTHON_FUNCTIONS, source="functions.md")

    strategies = ["paragraph", "sentence", "fixed"]

    for strategy in strategies:
        print(f"\n   Strategy: {strategy.upper()}")
        chunks = chunk_document(
            doc,
            strategy=strategy,
            chunk_size=200,
            chunk_overlap=20,
        )
        print(f"   Chunks created: {len(chunks)}")
        for chunk in chunks:
            print(f"   - Chunk {chunk.chunk_index}: {len(chunk.content)} chars")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 3: RETRIEVAL TUNING
# ─────────────────────────────────────────────────────────────


def example_retrieval_tuning():
    """show how retrieval parameters affect results"""
    print_section("Example 3: Retrieval Tuning")

    import tempfile
    temp_dir = tempfile.mkdtemp()

    rag = RAGPipeline(
        collection_name="tuning_demo",
        chunk_size=200,
        chunk_overlap=30,
        persist_directory=temp_dir,
    )

    # add all documents
    rag.add_text(PYTHON_FUNCTIONS, source="functions.md")
    rag.add_text(PYTHON_VARIABLES, source="variables.md")
    rag.add_text(PYTHON_ERRORS, source="errors.md")

    query = "What is a variable?"

    print(f"\n   Query: '{query}'")

    # compare different top_k values
    for k in [1, 3, 5]:
        print(f"\n   top_k={k}:")
        results = rag.retrieve(query, top_k=k)
        for r in results:
            print(f"   - {r.source}: {r.similarity:.3f}")

    rag.clear()


# ─────────────────────────────────────────────────────────────
# EXAMPLE 4: INTERACTIVE DEMO
# ─────────────────────────────────────────────────────────────


def example_interactive():
    """interactive Q&A session"""
    print_section("Example 4: Interactive Demo")

    import tempfile
    temp_dir = tempfile.mkdtemp()

    rag = RAGPipeline(
        collection_name="interactive_demo",
        chunk_size=300,
        chunk_overlap=50,
        top_k=2,
        persist_directory=temp_dir,
    )

    # add documents
    rag.add_text(PYTHON_FUNCTIONS, source="functions.md")
    rag.add_text(PYTHON_VARIABLES, source="variables.md")
    rag.add_text(PYTHON_ERRORS, source="errors.md")

    print("\n   Sample questions you can ask:")
    sample_questions = [
        "How do I define a function?",
        "What data types are available in Python?",
        "How do I catch exceptions?",
        "What is *args used for?",
    ]

    for q in sample_questions:
        print(f"\n   Q: {q}")
        answer = rag.query(q)
        # truncate for display
        short_answer = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"   A: {short_answer}")

    rag.clear()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def show_menu() -> None:
    """display interactive menu of available demos"""
    print("\n" + "=" * 60)
    print("  RAG Pipeline Examples")
    print("=" * 60)
    print("\nAvailable demos:")
    print("  1. basic RAG usage - create, add, query")
    print("  2. chunking strategies comparison")
    print("  3. retrieval tuning (top_k adjustment)")
    print("  4. interactive Q&A session")
    print("\n  [a] Run all demos")
    print("  [q] Quit")
    print("=" * 60)


def run_selected_demos(selections: str) -> bool:
    """run selected demos based on user input"""
    demo_map = {
        "1": ("basic RAG", example_basic_rag),
        "2": ("chunking strategies", example_chunking_strategies),
        "3": ("retrieval tuning", example_retrieval_tuning),
        "4": ("interactive demo", example_interactive),
    }

    if selections.lower() == "q":
        return False

    if selections.lower() == "a":
        print("\n🚀 Running all demos...\n")
        for name, func in demo_map.values():
            func()
        print("\n✅ All demos completed!")
        return True

    # parse comma-separated selections
    selected = [s.strip() for s in selections.split(",")]
    valid_selections = [s for s in selected if s in demo_map]

    if not valid_selections:
        print("❌ Invalid selection. Please enter numbers (1-4), 'a' for all, or 'q' to quit.")
        return True

    for selection in valid_selections:
        name, func = demo_map[selection]
        print(f"\n▶️  Running: {name}")
        func()

    print("\n✅ Selected demos completed!")
    return True


def main():
    """interactive demo runner"""
    while True:
        show_menu()
        choice = input("\nSelect demos (e.g., '1', '1,3', or 'a' for all): ").strip()
        should_continue = run_selected_demos(choice)
        if not should_continue:
            print("\n👋 Goodbye!\n")
            break

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!\n")
            break


if __name__ == "__main__":
    main()
