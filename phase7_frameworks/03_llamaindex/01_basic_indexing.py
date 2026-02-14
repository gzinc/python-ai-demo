"""
LlamaIndex Basic Indexing - Core Concepts

This module demonstrates fundamental LlamaIndex concepts:
- Document loading and indexing
- VectorStoreIndex (most common)
- Query engines
- Local vs cloud LLM setup

Run with: uv run python phase7_frameworks/03_llamaindex/01_basic_indexing.py
"""

from pathlib import Path
from inspect import cleandoc

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser

from common.demo_menu import Demo, MenuRunner
from common.util.utils import check_api_keys, print_section


# region Demo 1: In-Memory Index with Documents


def demo_in_memory_index():
    """demonstrate creating index from in-memory documents"""
    print_section("Demo 1: In-Memory Index")

    # create sample documents
    documents = [
        Document(
            text="LlamaIndex is a data framework for LLM applications. "
            "It helps you ingest, structure, and access private or domain-specific data.",
            metadata={"source": "intro", "category": "overview"},
        ),
        Document(
            text="VectorStoreIndex is the most common index type in LlamaIndex. "
            "It creates embeddings of your text and stores them for semantic search.",
            metadata={"source": "indexes", "category": "technical"},
        ),
        Document(
            text="Query engines in LlamaIndex handle the retrieval and synthesis process. "
            "They retrieve relevant chunks and use an LLM to generate responses.",
            metadata={"source": "querying", "category": "technical"},
        ),
    ]

    print(f"\n📄 Created {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc.text[:60]}... (metadata: {doc.metadata})")

    # create index (uses default LLM and embeddings from Settings)
    print("\n🔧 Building VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(documents)
    print("✅ Index created (in-memory)")

    # create query engine
    query_engine = index.as_query_engine(similarity_top_k=2)

    # query the index
    queries = [
        "What is LlamaIndex?",
        "How do query engines work?",
    ]

    print("\n💬 Querying the index:")
    for query in queries:
        print(f"\n  Q: {query}")
        response = query_engine.query(query)
        print(f"  A: {response.response}")

        # show source nodes
        print(f"  📚 Sources: {len(response.source_nodes)} chunks retrieved")


# endregion


# region Demo 2: Index from Directory


def demo_directory_index():
    """demonstrate loading documents from directory"""
    print_section("Demo 2: Index from Directory")

    # create sample directory with documents
    data_dir = Path("./temp_data")
    data_dir.mkdir(exist_ok=True)

    # create sample files
    files = {
        "rag_basics.txt": cleandoc("""
            Retrieval Augmented Generation (RAG) combines retrieval and generation.
            First, relevant documents are retrieved from a knowledge base.
            Then, an LLM uses those documents to generate accurate responses.
            This reduces hallucinations and grounds responses in your data.
        """),
        "embeddings.txt": cleandoc("""
            Embeddings are vector representations of text.
            Similar text has similar embeddings (cosine similarity).
            LlamaIndex uses embeddings for semantic search.
            Common models: OpenAI ada-002, HuggingFace BGE, Sentence Transformers.
        """),
        "chunking.txt": cleandoc("""
            Document chunking splits long texts into smaller pieces.
            Chunk size affects retrieval quality and context window usage.
            LlamaIndex supports various chunking strategies:
            - Simple fixed-size chunks
            - Sentence-aware chunking
            - Semantic chunking
        """),
    }

    for filename, content in files.items():
        (data_dir / filename).write_text(content)

    print(f"📁 Created sample directory: {data_dir}")
    print(f"   Files: {list(files.keys())}")

    # load documents from directory
    print("\n📚 Loading documents from directory...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"✅ Loaded {len(documents)} documents")

    for doc in documents:
        print(f"  - {doc.metadata.get('file_name', 'unknown')}: {len(doc.text)} chars")

    # create index
    print("\n🔧 Building index...")
    index = VectorStoreIndex.from_documents(documents)

    # query
    query_engine = index.as_query_engine()
    query = "What is RAG and why is it useful?"

    print(f"\n💬 Query: {query}")
    response = query_engine.query(query)
    print(f"📝 Response:\n{response.response}")

    # cleanup
    import shutil

    shutil.rmtree(data_dir)
    print("\n🧹 Cleaned up temp directory")


# endregion


# region Demo 3: Custom Node Parsing


def demo_node_parsing():
    """demonstrate custom chunk sizes and overlap"""
    print_section("Demo 3: Custom Node Parsing")

    long_document = Document(
        text=cleandoc("""
            LlamaIndex is a comprehensive data framework for LLM applications.
            It provides tools for data ingestion from various sources.
            The framework excels at structuring unstructured data.
            VectorStoreIndex creates semantic search capabilities.
            Query engines handle retrieval and response generation.
            Chat engines enable conversational interactions.
            The framework supports multiple LLM providers.
            Custom indices can be created for specific use cases.
            LlamaIndex integrates with vector databases.
            Production deployments benefit from caching strategies.
            The community actively contributes integrations.
            Documentation is comprehensive and well-maintained.
        """)
    )

    print("📄 Document length:", len(long_document.text), "characters")

    # parse with different chunk sizes
    chunk_configs = [
        {"chunk_size": 100, "chunk_overlap": 20, "label": "Small chunks"},
        {"chunk_size": 200, "chunk_overlap": 40, "label": "Medium chunks"},
        {"chunk_size": 500, "chunk_overlap": 50, "label": "Large chunks"},
    ]

    for config in chunk_configs:
        parser = SimpleNodeParser.from_defaults(
            chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
        )

        nodes = parser.get_nodes_from_documents([long_document])

        print(f"\n🔧 {config['label']} (size={config['chunk_size']}, overlap={config['chunk_overlap']}):")
        print(f"   Created {len(nodes)} chunks")

        for i, node in enumerate(nodes, 1):
            print(f"   Chunk {i}: {len(node.text)} chars - \"{node.text[:50]}...\"")

    print(
        cleandoc("""

        💡 Key Insights:
        • Smaller chunks = more granular retrieval, more chunks to search
        • Larger chunks = more context per chunk, fewer chunks
        • Overlap = prevents splitting related content
        • Optimal size depends on your use case (typically 200-512)
    """)
    )


# endregion


# region Demo 4: Metadata Filtering


def demo_metadata_filtering():
    """demonstrate filtering results by metadata"""
    print_section("Demo 4: Metadata Filtering")

    # create documents with rich metadata
    documents = [
        Document(
            text="Python is great for data science and machine learning.",
            metadata={"language": "python", "topic": "data_science", "level": "beginner"},
        ),
        Document(
            text="Advanced Python includes decorators, metaclasses, and async programming.",
            metadata={"language": "python", "topic": "advanced", "level": "expert"},
        ),
        Document(
            text="JavaScript is essential for web development and frontend applications.",
            metadata={"language": "javascript", "topic": "web_dev", "level": "beginner"},
        ),
        Document(
            text="React and Next.js are popular JavaScript frameworks for building UIs.",
            metadata={"language": "javascript", "topic": "web_dev", "level": "intermediate"},
        ),
        Document(
            text="TypeScript adds static typing to JavaScript for better code quality.",
            metadata={"language": "typescript", "topic": "web_dev", "level": "intermediate"},
        ),
    ]

    print(f"📚 Created {len(documents)} documents with metadata:")
    for doc in documents:
        print(f"  - {doc.metadata}")

    # create index
    index = VectorStoreIndex.from_documents(documents)

    # query with metadata filters
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

    filters_config = [
        {
            "label": "Python only",
            "filters": MetadataFilters(
                filters=[ExactMatchFilter(key="language", value="python")]
            ),
        },
        {
            "label": "Beginner level",
            "filters": MetadataFilters(
                filters=[ExactMatchFilter(key="level", value="beginner")]
            ),
        },
        {
            "label": "Web development",
            "filters": MetadataFilters(
                filters=[ExactMatchFilter(key="topic", value="web_dev")]
            ),
        },
    ]

    query = "What programming concepts should I learn?"

    print(f"\n💬 Query: {query}\n")

    for config in filters_config:
        query_engine = index.as_query_engine(
            similarity_top_k=3, filters=config["filters"]
        )

        response = query_engine.query(query)

        print(f"🔍 Filter: {config['label']}")
        print(f"   Response: {response.response[:120]}...")
        print(
            f"   Sources: {[node.metadata for node in response.source_nodes]}"
        )
        print()


# endregion


# region Demo 5: Response Modes


def demo_response_modes():
    """demonstrate different response synthesis modes"""
    print_section("Demo 5: Response Synthesis Modes")

    documents = [
        Document(text="LlamaIndex supports multiple response modes for different use cases."),
        Document(text="Compact mode combines chunks before sending to LLM, saving tokens."),
        Document(text="Refine mode iteratively refines the answer by processing chunks sequentially."),
        Document(text="Tree summarize creates a hierarchical summary of retrieved chunks."),
        Document(text="Simple concat mode joins all chunks with simple concatenation."),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query = "What response modes does LlamaIndex support?"

    response_modes = [
        ("compact", "Combine chunks before LLM call (default)"),
        ("refine", "Iteratively refine answer with each chunk"),
        ("tree_summarize", "Build hierarchical summary"),
        ("simple_summarize", "Simple concatenation"),
    ]

    print(f"💬 Query: {query}\n")

    for mode, description in response_modes:
        query_engine = index.as_query_engine(response_mode=mode)

        print(f"🔧 Mode: {mode}")
        print(f"   Description: {description}")

        response = query_engine.query(query)
        print(f"   Response: {response.response[:100]}...")
        print()


# endregion


# region Demo 6: Streaming Responses


def demo_streaming():
    """demonstrate streaming responses for real-time output"""
    print_section("Demo 6: Streaming Responses")

    documents = [
        Document(
            text=cleandoc("""
                Streaming responses in LlamaIndex allow you to display results as they're generated.
                This improves user experience by showing progress in real-time.
                Instead of waiting for the complete response, users see text appear gradually.
                Streaming is especially useful for long-form content generation.
                It reduces perceived latency and provides immediate feedback.
            """)
        )
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)

    query = "Explain the benefits of streaming responses"

    print(f"💬 Query: {query}\n")
    print("📝 Streaming response:")

    streaming_response = query_engine.query(query)

    # stream the response
    for text in streaming_response.response_gen:
        print(text, end="", flush=True)

    print("\n\n✅ Streaming complete")


# endregion


# region Main Menu


DEMOS = [
    Demo("1", "In-Memory Index", "create index from documents", demo_in_memory_index, needs_api=True),
    Demo("2", "Directory Index", "load documents from folder", demo_directory_index, needs_api=True),
    Demo("3", "Node Parsing", "custom chunk sizes and overlap", demo_node_parsing),
    Demo("4", "Metadata Filtering", "filter by metadata fields", demo_metadata_filtering, needs_api=True),
    Demo("5", "Response Modes", "different synthesis strategies", demo_response_modes, needs_api=True),
    Demo("6", "Streaming", "real-time response generation", demo_streaming, needs_api=True),
]


def main() -> None:
    """run interactive demo menu"""
    has_openai, _ = check_api_keys()

    if not has_openai:
        print("\n⚠️  Warning: OPENAI_API_KEY not found")
        print("Some demos will not work without it.")
        print("Set OPENAI_API_KEY in your .env file or environment.\n")

    runner = MenuRunner(
        DEMOS,
        title="LlamaIndex Basic Indexing",
        subtitle="Core concepts and patterns",
        has_api=has_openai,
    )
    runner.run()


# endregion

if __name__ == "__main__":
    main()