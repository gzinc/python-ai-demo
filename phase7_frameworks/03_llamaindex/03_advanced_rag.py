"""
LlamaIndex Advanced RAG - Production Patterns

This module demonstrates advanced RAG techniques:
- Reranking for better retrieval
- Query transformations
- Response synthesis modes
- Hybrid search (keyword + vector)
- Custom retrievers

Run with: uv run python phase7_frameworks/03_llamaindex/03_advanced_rag.py
"""


from llama_index.core import Document, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

from common.demo_menu import Demo, MenuRunner
from common.util.utils import check_api_keys, print_section

# region Demo 1: Custom Retrieval with Top-K


def demo_custom_retrieval():
    """demonstrate controlling retrieval parameters"""
    print_section("Demo 1: Custom Retrieval Parameters")

    documents = [
        Document(text="LlamaIndex is a data framework for LLM applications.", metadata={"id": 1}),
        Document(text="Vector databases store embeddings for semantic search.", metadata={"id": 2}),
        Document(text="RAG combines retrieval with generation for accurate responses.", metadata={"id": 3}),
        Document(text="Query engines handle the end-to-end RAG pipeline.", metadata={"id": 4}),
        Document(text="Chat engines add conversational memory to RAG systems.", metadata={"id": 5}),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query = "What is RAG?"

    # test different top_k values
    top_k_values = [1, 2, 5]

    print(f"💬 Query: {query}\n")

    for top_k in top_k_values:
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        query_engine = RetrieverQueryEngine(retriever=retriever)

        response = query_engine.query(query)

        print(f"🔍 Top-K = {top_k}")
        print(f"   Retrieved {len(response.source_nodes)} chunks")
        print(f"   Response: {response.response[:100]}...")
        print()

    print(
        """
        💡 Top-K Trade-offs:
        • Lower K (1-2): Faster, focused but may miss context
        • Medium K (3-5): Balanced approach (most common)
        • Higher K (10+): More context but slower, more noise
        • Optimal K depends on chunk size and query complexity
    """
    )


# endregion


# region Demo 2: Query Transformation


def demo_query_transformation():
    """demonstrate query rewriting for better retrieval"""
    print_section("Demo 2: Query Transformation")

    documents = [
        Document(
            text="""
                Query transformation improves retrieval by reformulating user questions.
                Techniques include: decomposition, multi-query, and HyDE (Hypothetical Document Embeddings).
                Decomposition breaks complex queries into simpler sub-queries.
                Multi-query generates variations to capture different aspects.
            """
        ),
        Document(
            text="""
                HyDE (Hypothetical Document Embeddings) generates a hypothetical answer first.
                The hypothetical answer is embedded and used for retrieval.
                This often finds better matches than searching with the question directly.
                Works well when questions and answers have different vocabulary.
            """
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)

    print("💬 Original Query: 'query techniques'\n")

    # standard retrieval
    query_engine = index.as_query_engine(similarity_top_k=1)
    response = query_engine.query("query techniques")

    print("📝 Standard Retrieval:")
    print(f"   {response.response[:150]}...\n")

    # with query understanding (built into LlamaIndex)
    print("💡 Query Transformation Concepts:")
    print(
        """
        • Decomposition: "What is X and how does Y work?"
          → Query 1: "What is X?"
          → Query 2: "How does Y work?"

        • Multi-query: "best practices for RAG"
          → "RAG best practices"
          → "RAG optimization techniques"
          → "effective RAG strategies"

        • HyDE: "How do I improve retrieval?"
          → Generate: "To improve retrieval, use query transformation..."
          → Search with generated answer (better semantic match)
    """
    )


# endregion


# region Demo 3: Response Synthesis Strategies


def demo_synthesis_strategies():
    """demonstrate different ways to synthesize final responses"""
    print_section("Demo 3: Response Synthesis Strategies")

    documents = [
        Document(text="Step 1: Ingest your data from various sources.", metadata={"step": 1}),
        Document(text="Step 2: Parse and chunk documents into smaller pieces.", metadata={"step": 2}),
        Document(text="Step 3: Create embeddings for each chunk.", metadata={"step": 3}),
        Document(text="Step 4: Store embeddings in a vector database.", metadata={"step": 4}),
        Document(text="Step 5: Retrieve relevant chunks based on query.", metadata={"step": 5}),
        Document(text="Step 6: Synthesize final response using LLM.", metadata={"step": 6}),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query = "What are the steps in a RAG pipeline?"

    synthesis_modes = {
        "compact": "Combine chunks efficiently before LLM call (default)",
        "refine": "Iteratively refine answer with each chunk",
        "tree_summarize": "Build hierarchical summary tree",
    }

    print(f"💬 Query: {query}\n")

    for mode, description in synthesis_modes.items():
        query_engine = index.as_query_engine(
            response_mode=mode, similarity_top_k=6  # retrieve all steps
        )

        response = query_engine.query(query)

        print(f"🔧 Mode: {mode}")
        print(f"   {description}")
        print(f"   Response length: {len(response.response)} chars")
        print(f"   Preview: {response.response[:120]}...")
        print()


# endregion


# region Demo 4: Retrieval Evaluation


def demo_retrieval_evaluation():
    """demonstrate evaluating retrieval quality"""
    print_section("Demo 4: Retrieval Quality Metrics")

    documents = [
        Document(
            text="Python is a high-level programming language known for readability.",
            metadata={"topic": "python", "relevance": "high"},
        ),
        Document(
            text="JavaScript is essential for web development and runs in browsers.",
            metadata={"topic": "javascript", "relevance": "low"},
        ),
        Document(
            text="Python is widely used in data science and machine learning.",
            metadata={"topic": "python", "relevance": "high"},
        ),
        Document(
            text="TypeScript adds static typing to JavaScript.",
            metadata={"topic": "javascript", "relevance": "low"},
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # query about Python
    query = "What is Python used for?"
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    print(f"💬 Query: {query}\n")
    print("🔍 Retrieved chunks:")

    nodes = retriever.retrieve(query)

    for i, node in enumerate(nodes, 1):
        relevance = node.metadata.get("relevance", "unknown")
        score = node.score if hasattr(node, "score") else "N/A"

        print(f"\n  {i}. Relevance: {relevance} | Score: {score}")
        print(f"     Text: {node.text[:80]}...")

    # calculate simple metrics
    high_relevance = sum(1 for node in nodes if node.metadata.get("relevance") == "high")
    precision = high_relevance / len(nodes) if nodes else 0

    print(
        f"\n📊 Metrics:\n"
        f"   Retrieved: {len(nodes)} chunks\n"
        f"   Relevant: {high_relevance} chunks\n"
        f"   Precision: {precision:.2%}\n"
    )

    print(
        """
        💡 Retrieval Metrics:
        • Precision: % of retrieved docs that are relevant
        • Recall: % of relevant docs that were retrieved
        • MRR (Mean Reciprocal Rank): Quality of top result
        • NDCG: Ranking quality (graded relevance)

        In production:
        • Use test sets with known relevant docs
        • Track metrics over time
        • A/B test different retrieval configs
    """
    )


# endregion


# region Demo 5: Context Window Management


def demo_context_window():
    """demonstrate managing context window limits"""
    print_section("Demo 5: Context Window Management")

    # create documents with varying sizes
    documents = [
        Document(text="Short document: LlamaIndex handles context windows automatically."),
        Document(
            text="Medium document: " + " ".join(["Context management is important."] * 10)
        ),
        Document(text="Another short: Vector search retrieves relevant content."),
        Document(text="Long document: " + " ".join(["This is detailed content."] * 50)),
    ]

    print("📚 Document Sizes:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {len(doc.text)} characters")

    index = VectorStoreIndex.from_documents(documents)

    # demonstrate different context budgets
    configs = [
        {"top_k": 2, "label": "Conservative (top-2)"},
        {"top_k": 4, "label": "Aggressive (all docs)"},
    ]

    query = "Tell me about context management"
    print(f"\n💬 Query: {query}\n")

    for config in configs:
        query_engine = index.as_query_engine(similarity_top_k=config["top_k"])

        response = query_engine.query(query)

        total_chars = sum(len(node.text) for node in response.source_nodes)

        print(f"🔧 {config['label']}")
        print(f"   Retrieved: {len(response.source_nodes)} chunks")
        print(f"   Total context: {total_chars} chars (~{total_chars // 4} tokens)")
        print(f"   Response: {response.response[:100]}...")
        print()

    print(
        """
        💡 Context Window Strategies:
        • Start conservative, increase if needed
        • Monitor token usage (prompt + completion)
        • Use smaller chunks for finer control
        • Consider compacting/summarizing long chunks
        • Reserve tokens for response generation

        Typical limits:
        • GPT-3.5 Turbo: 16K tokens (12K for context)
        • GPT-4 Turbo: 128K tokens (100K for context)
        • Claude 3: 200K tokens (180K for context)
        • Llama 3.1: 128K tokens (100K for context)
    """
    )


# endregion


# region Demo 6: Production Best Practices


def demo_production_patterns():
    """demonstrate production-ready RAG patterns"""
    print_section("Demo 6: Production RAG Patterns")

    print(
        """
        🏭 Production RAG Checklist:

        1️⃣  **Chunking Strategy**
           ✓ Optimal size: 200-512 tokens per chunk
           ✓ Overlap: 10-20% to prevent context splits
           ✓ Sentence-aware splitting (don't break mid-sentence)
           ✓ Consider semantic chunking for better coherence

        2️⃣  **Retrieval Configuration**
           ✓ Top-K: Start with 3-5, tune based on metrics
           ✓ Similarity threshold: Filter low-relevance chunks
           ✓ Metadata filters: Enable faceted search
           ✓ Reranking: Use cross-encoder for better ordering

        3️⃣  **Response Quality**
           ✓ Synthesis mode: Use 'compact' for efficiency
           ✓ Streaming: Enable for better UX
           ✓ Citations: Include source references
           ✓ Fallback: Handle no-result scenarios gracefully

        4️⃣  **Performance Optimization**
           ✓ Caching: Cache embeddings and frequent queries
           ✓ Async: Use async for concurrent operations
           ✓ Batch processing: Embed in batches during indexing
           ✓ Vector DB: Choose based on scale (local vs cloud)

        5️⃣  **Monitoring & Evaluation**
           ✓ Track retrieval precision/recall
           ✓ Monitor response latency (p50, p95, p99)
           ✓ Log failed queries for analysis
           ✓ A/B test configuration changes
           ✓ User feedback collection

        6️⃣  **Cost Management**
           ✓ Embedding costs: Use local models or cache
           ✓ LLM costs: Monitor prompt/completion tokens
           ✓ Vector DB: Balance performance vs cost
           ✓ Batch operations: Reduce API calls

        7️⃣  **Data Management**
           ✓ Version control: Track index versions
           ✓ Incremental updates: Add new docs without full rebuild
           ✓ Data privacy: Use local models for sensitive data
           ✓ Backup: Regular vector store backups

        📊 Typical Production Architecture:

        ┌─────────────────────────────────────────────────────┐
        │                  User Query                         │
        └─────────────────────┬───────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Query Transform  │ (optional)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Vector Retrieval │ (top-k chunks)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    Reranking      │ (cross-encoder)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   LLM Synthesis   │ (generate response)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Response + Cites │
                    └───────────────────┘

        🎯 Performance Targets:
        • Indexing: < 1 sec per document
        • Retrieval: < 200ms (p95)
        • End-to-end: < 2 sec (p95)
        • Accuracy: > 80% user satisfaction

        💰 Cost Optimization:
        • Use local embeddings (HuggingFace)
        • Cache frequent queries
        • Batch embed during off-peak
        • Monitor and set budget alerts
    """
    )


# endregion


# region Main Menu


DEMOS = [
    Demo("1", "Custom Retrieval", "control top-k and parameters", demo_custom_retrieval, needs_api=True),
    Demo("2", "Query Transformation", "improve retrieval quality", demo_query_transformation, needs_api=True),
    Demo("3", "Synthesis Strategies", "different response modes", demo_synthesis_strategies, needs_api=True),
    Demo("4", "Retrieval Evaluation", "measure retrieval quality", demo_retrieval_evaluation, needs_api=True),
    Demo("5", "Context Window", "manage token limits", demo_context_window, needs_api=True),
    Demo("6", "Production Patterns", "best practices and checklist", demo_production_patterns),
]


def main() -> None:
    """run interactive demo menu"""
    has_openai, _ = check_api_keys()

    if not has_openai:
        print("\n⚠️  Warning: OPENAI_API_KEY not found")
        print("Most demos require an API key to run.\n")

    runner = MenuRunner(
        DEMOS,
        title="LlamaIndex Advanced RAG",
        subtitle="Production patterns and optimization",
        has_api=has_openai,
    )
    runner.run()


# endregion

if __name__ == "__main__":
    main()
