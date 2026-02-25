"""
RAG (Retrieval-Augmented Generation) - Practical Examples

This module demonstrates real RAG implementations with OpenAI and vector stores.
Requires OPENAI_API_KEY environment variable.

Note: Demos use persistent ChromaDB storage in ./chroma_rag_db/
First run creates embeddings via API. Subsequent runs load from disk.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.05_rag.practical
"""

import tempfile
from inspect import cleandoc
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.demo_menu import Demo, MenuRunner
from common.util.utils import check_api_keys, print_section, requires_openai


# region Helper Functions
def create_sample_documents() -> list[Document]:
    """create sample documents for RAG demonstrations"""
    docs = [
        Document(
            page_content=cleandoc(
                """
            Python is a high-level, interpreted programming language known for its
            simplicity and readability. It was created by Guido van Rossum and first
            released in 1991. Python emphasizes code readability with significant
            whitespace and supports multiple programming paradigms including procedural,
            object-oriented, and functional programming.
            """
            ),
            metadata={"source": "python_intro.txt", "topic": "programming"},
        ),
        Document(
            page_content=cleandoc(
                """
            Machine learning is a subset of artificial intelligence that enables
            computers to learn from data without being explicitly programmed. It uses
            algorithms that iteratively learn from data, allowing computers to find
            hidden insights. Machine learning models improve their performance over time
            as they are exposed to more data.
            """
            ),
            metadata={"source": "ml_basics.txt", "topic": "ai"},
        ),
        Document(
            page_content=cleandoc(
                """
            Natural Language Processing (NLP) is a branch of AI that helps computers
            understand, interpret, and manipulate human language. NLP draws from many
            disciplines including computer science and linguistics. Modern NLP uses
            machine learning, especially deep learning models like transformers, to
            achieve human-like language understanding.
            """
            ),
            metadata={"source": "nlp_guide.txt", "topic": "ai"},
        ),
        Document(
            page_content=cleandoc(
                """
            Vector databases are specialized databases designed to store and query
            high-dimensional vectors efficiently. They are essential for AI applications
            that use embeddings, such as semantic search and recommendation systems.
            Vector databases use specialized indexing techniques like HNSW (Hierarchical
            Navigable Small World) to enable fast similarity search.
            """
            ),
            metadata={"source": "vector_db.txt", "topic": "database"},
        ),
        Document(
            page_content=cleandoc(
                """
            Retrieval-Augmented Generation (RAG) is a technique that combines information
            retrieval with text generation. RAG first retrieves relevant documents from a
            knowledge base, then uses those documents as context for an LLM to generate
            accurate, grounded responses. This approach reduces hallucinations and enables
            LLMs to access current, domain-specific information.
            """
            ),
            metadata={"source": "rag_explained.txt", "topic": "ai"},
        ),
    ]
    return docs


def get_or_create_vectorstore(
    persist_directory: str = "./chroma_rag_db",
    collection_name: str = "phase7_rag_demos",
) -> Chroma:
    """
    get or create persistent vectorstore for RAG demos

    On first run: creates embeddings via API and persists to disk
    On subsequent runs: loads from disk (zero API calls)

    Args:
        persist_directory: directory for ChromaDB persistence
        collection_name: name for the collection

    Returns:
        Chroma vectorstore instance
    """

    #  If you want to use local embeddings in Phase 7 (no API key needed):

    # from langchain_community.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # try loading existing collection
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        # verify collection has data
        if vectorstore._collection.count() > 0:
            print(f"   ✓ loaded existing collection ({vectorstore._collection.count()} docs)")
            return vectorstore
    except Exception:
        pass  # collection doesn't exist yet

    # create new collection with embeddings
    print("   ⚡ creating new collection (one-time embedding API calls)...")
    documents = create_sample_documents()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"   ✓ created and persisted {len(documents)} documents")
    return vectorstore


# endregion


# region Demo 1: Basic RAG with ChromaDB
@requires_openai
def demo_basic_rag() -> None:
    """
    demonstrate basic RAG pipeline with ChromaDB

    Basic RAG Flow:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Basic RAG Pipeline                       │
    │                                                             │
    │  User Question                                              │
    │       │                                                     │
    │       ▼                                                     │
    │  ┌──────────────┐                                           │
    │  │  Embedding   │ (convert question to vector)              │
    │  └──────┬───────┘                                           │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌──────────────┐                                           │
    │  │  VectorDB    │ (similarity search)                       │
    │  │   Search     │                                           │
    │  └──────┬───────┘                                           │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌──────────────┐                                           │
    │  │  Retrieved   │ (top-k most similar docs)                 │
    │  │  Documents   │                                           │
    │  └──────┬───────┘                                           │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌──────────────┐                                           │
    │  │   Format     │ (combine docs into context)               │
    │  │   Context    │                                           │
    │  └──────┬───────┘                                           │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌──────────────┐                                           │
    │  │  LLM Prompt  │ (context + question)                      │
    │  └──────┬───────┘                                           │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌──────────────┐                                           │
    │  │     LLM      │ (generate grounded answer)                │
    │  └──────┬───────┘                                           │
    │         │                                                   │
    │         ▼                                                   │
    │     Answer                                                  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    This is the foundation of RAG: retrieve relevant context, then
    generate answers grounded in that context.
    """
    print_section("Basic RAG Pipeline")

    # initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # load persistent vector store
    print("\n💾 loading vector store...")
    vectorstore = get_or_create_vectorstore()

    # create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # create RAG prompt
    template = cleandoc(
        """
        answer the question based only on the following context:

        {context}

        question: {question}

        answer:
        """
    )
    prompt = ChatPromptTemplate.from_template(template)

    # create RAG chain using LCEL
    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # test with questions
    questions = [
        "What is Python and when was it created?",
        "How does machine learning work?",
        "What is RAG and why is it useful?",
    ]

    for question in questions:
        print(f"\n❓ question: {question}")
        response = rag_chain.invoke(question)
        print(f"💬 answer: {response}")


# endregion


# region Demo 2: Text Chunking Strategies
def demo_text_chunking() -> None:
    """
    demonstrate different text chunking approaches

    Text Chunking with Overlap:
    ┌─────────────────────────────────────────────────────────┐
    │              Original Document                          │
    │  ┌────────────────────────────────────────────┐         │
    │  │  The quick brown fox jumps over the lazy   │         │
    │  │  dog. The dog was sleeping under a tree.   │         │
    │  │  The tree provided shade from the sun.     │         │
    │  └────────────────────────────────────────────┘         │
    │                      │                                  │
    │                      ▼                                  │
    │         RecursiveCharacterTextSplitter                  │
    │         (chunk_size=50, overlap=10)                     │
    │                      │                                  │
    │           ┌──────────┴──────────┐                       │
    │           ▼                     ▼                       │
    │  ┌─────────────────┐   ┌─────────────────┐              │
    │  │ Chunk 1:        │   │ Chunk 2:        │              │
    │  │ "The quick...   │   │ "...lazy dog.   │              │
    │  │  lazy dog."     │   │  The dog was..."│              │
    │  └─────────────────┘   └─────────────────┘              │
    │         │  overlap          │                           │
    │         └───────┬───────────┘                           │
    │                 ▼                                       │
    │         "...lazy dog." (shared)                         │
    │                                                         │
    │  Benefits of overlap:                                   │
    │  • Preserves context across chunk boundaries            │
    │  • Prevents splitting sentences/concepts                │
    │  • Improves retrieval accuracy                          │
    └─────────────────────────────────────────────────────────┘
    """
    print_section("Text Chunking Strategies")

    sample_text = cleandoc(
        """
        Artificial intelligence (AI) is revolutionizing technology across industries.
        Machine learning, a subset of AI, enables computers to learn from data without
        explicit programming.

        Deep learning, which uses neural networks with multiple layers, has achieved
        remarkable success in computer vision and natural language processing. These
        models can process vast amounts of data to recognize patterns and make predictions.

        Natural Language Processing (NLP) focuses on the interaction between computers
        and human language. Modern NLP systems use transformer architectures like BERT
        and GPT to achieve human-like language understanding and generation capabilities.

        The field continues to evolve rapidly, with new breakthroughs in reasoning,
        multimodal understanding, and efficient model training techniques.
        """
    )

    print(f"📖 original text length: {len(sample_text)} characters")

    # strategy 1: recursive character splitter (recommended)
    print("\n✂️ strategy 1: recursive character splitter")
    splitter1 = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks1 = splitter1.split_text(sample_text)
    print(f"   created {len(chunks1)} chunks")
    for i, chunk in enumerate(chunks1[:2], 1):
        print(f"   chunk {i}: {chunk[:80]}...")

    # strategy 2: larger chunks for more context
    print("\n✂️ strategy 2: larger chunks (more context)")
    splitter2 = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    chunks2 = splitter2.split_text(sample_text)
    print(f"   created {len(chunks2)} chunks")
    for i, chunk in enumerate(chunks2, 1):
        print(f"   chunk {i} length: {len(chunk)} chars")

    print("\n📊 comparison:")
    print(f"   small chunks (200): {len(chunks1)} chunks, better precision")
    print(f"   large chunks (400): {len(chunks2)} chunks, more context")


# endregion


# region Demo 3: Similarity Search Methods
@requires_openai
def demo_similarity_search() -> None:
    """
    demonstrate different similarity search methods

    Similarity Search Methods:
    ┌──────────────────────────────────────────────────────────────┐
    │           Vector Similarity Search Methods                   │
    │                                                              │
    │  1. Basic Similarity Search (Cosine Distance):               │
    │     Query Vector                                             │
    │          │                                                   │
    │          ▼                                                   │
    │     ┌─────────┐                                              │
    │     │VectorDB │ → Compare with all vectors                   │
    │     └────┬────┘                                              │
    │          │                                                   │
    │          ▼                                                   │
    │     Return top-k most similar                                │
    │                                                              │
    │  2. Similarity Search with Scores:                           │
    │     Query Vector                                             │
    │          │                                                   │
    │          ▼                                                   │
    │     ┌─────────┐                                              │
    │     │VectorDB │ → Calculate similarity scores                │
    │     └────┬────┘                                              │
    │          │                                                   │
    │          ▼                                                   │
    │     [(Doc1, 0.95), (Doc2, 0.87), (Doc3, 0.82)]               │
    │                                                              │
    │  3. MMR (Max Marginal Relevance):                            │
    │     Query Vector                                             │
    │          │                                                   │
    │          ▼                                                   │
    │     ┌─────────┐                                              │
    │     │VectorDB │ → Fetch top-k candidates (fetch_k=4)         │
    │     └────┬────┘                                              │
    │          │                                                   │
    │          ▼                                                   │
    │     ┌─────────────────┐                                      │
    │     │  Diversity      │ → Balance relevance vs diversity     │
    │     │  Reranking      │                                      │
    │     └────┬────────────┘                                      │
    │          │                                                   │
    │          ▼                                                   │
    │     Return diverse top-k (k=2)                               │
    │                                                              │
    │  Trade-offs:                                                 │
    │  • Basic: Fastest, most relevant, but may be redundant       │
    │  • With Scores: Same as basic + confidence metric            │
    │  • MMR: Slower, but prevents redundant similar results       │
    └──────────────────────────────────────────────────────────────┘
    """
    print_section("Similarity Search Methods")

    # load persistent vector store
    vectorstore = get_or_create_vectorstore()

    query = "What is artificial intelligence and machine learning?"

    # method 1: basic similarity search
    print(f"🔍 query: '{query}'")
    print("\n📊 method 1: similarity search (top 2)")
    results = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content[:80]}...")
        print(f"      source: {doc.metadata.get('source', 'unknown')}")

    # method 2: similarity search with scores
    print("\n📊 method 2: similarity search with scores")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=2)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"   {i}. (score: {score:.4f}) {doc.page_content[:60]}...")

    # method 3: mmr (max marginal relevance) for diversity
    print("\n📊 method 3: MMR (max marginal relevance)")
    print("   (balances relevance with diversity)")
    mmr_results = vectorstore.max_marginal_relevance_search(query, k=2, fetch_k=4)
    for i, doc in enumerate(mmr_results, 1):
        print(f"   {i}. {doc.page_content[:80]}...")


# endregion


# region Demo 4: Metadata Filtering
@requires_openai
def demo_metadata_filtering() -> None:
    """
    demonstrate retrieval with metadata filters

    Metadata Filtering Flow:
    ┌──────────────────────────────────────────────────────────────┐
    │              Metadata Filtering in Vector Search             │
    │                                                              │
    │  Without Filter (search all documents):                      │
    │  ┌────────────────────────────────────────────┐              │
    │  │  VectorDB (all topics)                     │              │
    │  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐        │              │
    │  │  │ AI │ │ DB │ │ AI │ │Code│ │ AI │        │              │
    │  │  └────┘ └────┘ └────┘ └────┘ └────┘        │              │
    │  └────────────────────────────────────────────┘              │
    │           │                                                  │
    │           ▼                                                  │
    │  Query → Similarity Search → Top 3 (mixed topics)            │
    │                                                              │
    │  ─────────────────────────────────────────────               │
    │                                                              │
    │  With Filter (topic='ai' only):                              │
    │  ┌────────────────────────────────────────────┐              │
    │  │  VectorDB (filtered view)                  │              │
    │  │  ┌────┐          ┌────┐          ┌────┐    │              │
    │  │  │ AI │    ✓     │ AI │    ✓     │ AI │    │              │
    │  │  └────┘          └────┘          └────┘    │              │
    │  │         ┌────┐          ┌────┐             │              │
    │  │         │ DB │    ✗     │Code│    ✗        │              │
    │  │         └────┘          └────┘             │              │
    │  └────────────────────────────────────────────┘              │
    │           │                                                  │
    │           ▼                                                  │
    │  Query → Filter → Similarity Search → Top 3 (AI only)        │
    │                                                              │
    │  Benefits:                                                   │
    │  • Faster: Searches smaller subset                           │
    │  • Precise: Only relevant categories                         │
    │  • Flexible: Time-based, user-based, category filtering      │
    │  • Multi-tenant: Isolate data by tenant ID                   │
    └──────────────────────────────────────────────────────────────┘
    """
    print_section("Metadata Filtering")

    # load persistent vector store
    vectorstore = get_or_create_vectorstore()

    query = "Tell me about technology"

    # search without filter
    print(f"🔍 query: '{query}'")
    print("\n📊 without filter (all topics):")
    results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        topic = doc.metadata.get("topic", "unknown")
        print(f"   {i}. [{topic}] {doc.page_content[:60]}...")

    # search with metadata filter (only AI topic)
    print("\n📊 with filter (topic='ai' only):")
    filtered_results = vectorstore.similarity_search(query, k=3, filter={"topic": "ai"})
    for i, doc in enumerate(filtered_results, 1):
        topic = doc.metadata.get("topic", "unknown")
        print(f"   {i}. [{topic}] {doc.page_content[:60]}...")

    print("\n💡 metadata filtering:")
    print("   • narrows search scope before similarity calculation")
    print("   • useful for multi-tenant applications")
    print("   • enables time-based or category-based filtering")


# endregion


# region Demo 5: Document Loading from Files
def demo_document_loading() -> None:
    """
    demonstrate loading documents from files

    Document Loading Pipeline:
    ┌──────────────────────────────────────────────────────────────┐
    │              File → RAG Pipeline                             │
    │                                                              │
    │  Filesystem                                                  │
    │  ┌────────────┐  ┌────────────┐   ┌────────────┐             │
    │  │ file1.txt  │  │ file2.txt  │   │ file3.pdf  │             │
    │  └─────┬──────┘  └─────┬──────┘   └─────┬──────┘             │
    │        │               │                │                    │
    │        ▼               ▼                ▼                    │
    │  ┌─────────────────────────────────────────┐                 │
    │  │          Document Loaders               │                 │
    │  │  (TextLoader, PDFLoader, CSVLoader)     │                 │
    │  └────────────────┬────────────────────────┘                 │
    │                   │                                          │
    │                   ▼                                          │
    │  ┌─────────────────────────────────────────┐                 │
    │  │      Document Objects (with metadata)   │                 │
    │  │  [{content: "...", metadata: {...}}]    │                 │
    │  └────────────────┬────────────────────────┘                 │
    │                   │                                          │
    │                   ▼                                          │
    │  ┌─────────────────────────────────────────┐                 │
    │  │    RecursiveCharacterTextSplitter       │                 │
    │  │    (chunk_size=200, overlap=50)         │                 │
    │  └────────────────┬────────────────────────┘                 │
    │                   │                                          │
    │                   ▼                                          │
    │  ┌─────────────────────────────────────────┐                 │
    │  │     Chunks (preserves metadata)         │                 │
    │  │  [{content: "chunk1", metadata: {...}}] │                 │
    │  │  [{content: "chunk2", metadata: {...}}] │                 │
    │  └────────────────┬────────────────────────┘                 │
    │                   │                                          │
    │                   ▼                                          │
    │  ┌─────────────────────────────────────────┐                 │
    │  │         Embed & Store                   │                 │
    │  │        (VectorDB ready)                 │                 │
    │  └─────────────────────────────────────────┘                 │
    │                                                              │
    │  Key Points:                                                 │
    │  • Different loaders for different file types                │
    │  • Metadata preserved through pipeline                       │
    │  • Splitting maintains source attribution                    │
    │  • Ready for vectorDB ingestion                              │
    └──────────────────────────────────────────────────────────────┘
    """
    print_section("Document Loading from Files")

    # create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # create sample text files
        file1 = temp_path / "ai_overview.txt"
        file1.write_text(
            cleandoc(
                """
            Artificial Intelligence (AI) Overview

            AI is the simulation of human intelligence by machines. It encompasses various
            subfields including machine learning, natural language processing, computer
            vision, and robotics. Modern AI systems can perform complex tasks like image
            recognition, language translation, and decision-making.
            """
            )
        )

        file2 = temp_path / "ml_intro.txt"
        file2.write_text(
            cleandoc(
                """
            Machine Learning Introduction

            Machine learning is a method of data analysis that automates analytical model
            building. It uses algorithms that iteratively learn from data, allowing computers
            to find hidden insights without being explicitly programmed where to look.
            """
            )
        )

        print(f"📁 created temporary directory: {temp_dir}")
        print(f"   file 1: {file1.name}")
        print(f"   file 2: {file2.name}")

        # load documents
        print("\n📄 loading documents...")
        loader1 = TextLoader(str(file1))
        loader2 = TextLoader(str(file2))

        docs1 = loader1.load()
        docs2 = loader2.load()

        all_docs = docs1 + docs2

        print(f"   loaded {len(all_docs)} documents")
        for i, doc in enumerate(all_docs, 1):
            source = Path(doc.metadata["source"]).name
            print(f"   {i}. {source}: {len(doc.page_content)} characters")

        # split documents
        print("\n✂️ splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
        print(f"   created {len(chunks)} chunks from {len(all_docs)} documents")


# endregion


# region Demo 6: RAG with Custom Retriever
@requires_openai
def demo_custom_retriever() -> None:
    """
    demonstrate RAG with custom retriever configuration

    Custom Retriever Configuration:
    ┌──────────────────────────────────────────────────────────────┐
    │           Retriever Configuration Options                    │
    │                                                              │
    │  Configuration 1: Top-K Retrieval                            │
    │  ┌────────────────────────────────┐                          │
    │  │  retriever = vectorstore.      │                          │
    │  │    as_retriever(               │                          │
    │  │      search_kwargs={"k": 3}    │                          │
    │  │    )                           │                          │
    │  └────────────┬───────────────────┘                          │
    │               │                                              │
    │               ▼                                              │
    │  Query → VectorDB → Top 3 docs (by similarity)               │
    │                                                              │
    │  ──────────────────────────────────────────                  │
    │                                                              │
    │  Configuration 2: MMR (Diversity)                            │
    │  ┌────────────────────────────────┐                          │
    │  │  retriever = vectorstore.      │                          │
    │  │    as_retriever(               │                          │
    │  │      search_type="mmr",        │                          │
    │  │      search_kwargs={           │                          │
    │  │        "k": 2,                 │                          │
    │  │        "fetch_k": 4            │                          │
    │  │      }                         │                          │
    │  │    )                           │                          │
    │  └────────────┬───────────────────┘                          │
    │               │                                              │
    │               ▼                                              │
    │  Query → VectorDB → Fetch 4 → Diversity Rerank → Top 2       │
    │                                                              │
    │  ──────────────────────────────────────────                  │
    │                                                              │
    │  Configuration 3: Score Threshold                            │
    │  ┌────────────────────────────────┐                          │
    │  │  retriever = vectorstore.      │                          │
    │  │    as_retriever(               │                          │
    │  │      search_type=              │                          │
    │  │        "similarity_score_      │                          │
    │  │         threshold",            │                          │
    │  │      search_kwargs={           │                          │
    │  │        "score_threshold": 0.3, │                          │
    │  │        "k": 3                  │                          │
    │  │      }                         │                          │
    │  │    )                           │                          │
    │  └────────────┬───────────────────┘                          │
    │               │                                              │
    │               ▼                                              │
    │  Query → VectorDB → Filter by score > 0.3 → Return matches   │
    │                                                              │
    │  Key Parameters:                                             │
    │  • k: How many docs to retrieve                              │
    │  • search_type: similarity | mmr | similarity_score_threshold│
    │  • fetch_k: Candidates for MMR reranking                     │
    │  • score_threshold: Minimum similarity score                 │
    └──────────────────────────────────────────────────────────────┘
    """
    print_section("Custom Retriever Configuration")

    # load persistent vector store
    vectorstore = get_or_create_vectorstore()

    # configuration 1: retrieve more documents
    print("⚙️ configuration 1: retrieve top 3 documents")
    retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = "Explain AI and machine learning"

    docs = retriever1.invoke(query)
    print(f"   retrieved {len(docs)} documents")

    # configuration 2: use MMR for diversity
    print("\n⚙️ configuration 2: MMR retrieval (diversity)")
    retriever2 = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    docs = retriever2.invoke(query)
    print(f"   retrieved {len(docs)} diverse documents")

    # configuration 3: similarity threshold
    print("\n⚙️ configuration 3: similarity score threshold")
    retriever3 = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3},
    )

    docs = retriever3.invoke(query)
    print(f"   retrieved {len(docs)} documents above threshold")

    print("\n💡 retriever configurations:")
    print("   • k: number of documents to retrieve")
    print("   • search_type: similarity, mmr, or similarity_score_threshold")
    print("   • fetch_k: for MMR, candidates to fetch before reranking")
    print("   • score_threshold: minimum similarity score")


# endregion


# region Demo 7: Multi-Query RAG
@requires_openai
def demo_multi_query() -> None:
    """
    demonstrate generating multiple query perspectives for improved retrieval

    Multi-Query RAG Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Multi-Query RAG Pipeline                    │
    │                                                                 │
    │  Original Query                                                 │
    │       │                                                         │
    │       ▼                                                         │
    │  ┌─────────┐                                                    │
    │  │   LLM   │ (generate variations)                              │
    │  └────┬────┘                                                    │
    │       │                                                         │
    │       ▼                                                         │
    │  ┌─────────────────────────────────────┐                        │
    │  │  Q1  │  Q2  │  Q3  │  Q4   │  (variations)                   │
    │  └──┬───┴──┬───┴──┬───┴──┬────┘                                 │
    │     │      │      │      │                                      │
    │     ▼      ▼      ▼      ▼   (parallel search)                  │
    │  ┌──────────────────────────┐                                   │
    │  │      VectorDB Search     │                                   │
    │  └────────────┬─────────────┘                                   │
    │               │                                                 │
    │               ▼                                                 │
    │  ┌─────────────────────┐                                        │
    │  │  Deduplicate Docs   │                                        │
    │  └──────────┬──────────┘                                        │
    │             │                                                   │
    │             ▼                                                   │
    │  ┌─────────────────────┐                                        │
    │  │  Combine Context    │                                        │
    │  └──────────┬──────────┘                                        │
    │             │                                                   │
    │             ▼                                                   │
    │  ┌─────────────────────┐                                        │
    │  │   LLM Generation    │ (answer using context)                 │
    │  └──────────┬──────────┘                                        │
    │             │                                                   │
    │             ▼                                                   │
    │        Final Answer                                             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Key improvement: generates multiple perspectives of the query to improve
    retrieval recall and capture different aspects of the question.
    """
    print_section("Multi-Query RAG")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # load persistent vector store
    vectorstore = get_or_create_vectorstore()

    original_query = "What is AI?"

    # generate query variations using LLM
    print(f"🔍 original query: '{original_query}'")
    print("\n🔄 generating query variations...")

    query_prompt = ChatPromptTemplate.from_template(
        cleandoc(
            """
        you are an AI assistant that generates alternative search queries.
        given the original query, generate 3 different versions that capture the
        same intent but use different wording.

        original query: {question}

        alternative queries (one per line):
        """
        )
    )

    query_chain = query_prompt | llm | StrOutputParser()
    alternatives = query_chain.invoke({"question": original_query})

    queries = [original_query] + [q.strip("123. -") for q in alternatives.split("\n") if q.strip()]

    print("\n📋 query variations:")
    for i, q in enumerate(queries[:4], 1):
        print(f"   {i}. {q}")

    # retrieve documents for each query
    print("\n📚 retrieving documents for each variation...")
    all_docs = []
    seen_contents = set()

    for query in queries[:4]:
        docs = vectorstore.similarity_search(query, k=2)
        for doc in docs:
            content = doc.page_content[:100]
            if content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(content)

    print(f"   retrieved {len(all_docs)} unique documents (deduped)")

    print("\n💡 multi-query benefits:")
    print("   • captures different aspects of the question")
    print("   • improves recall (finds more relevant docs)")
    print("   • handles ambiguous or underspecified queries")

    # next step: generate answer using retrieved documents
    print("\n🤖 generating answer using retrieved documents...")

    # combine documents into context
    context = "\n\n".join([doc.page_content for doc in all_docs])

    # create RAG prompt
    rag_prompt = ChatPromptTemplate.from_template(
        cleandoc(
            """
        answer the question based only on the following context:

        {context}

        question: {question}

        answer:
        """
        )
    )

    # generate answer
    answer_chain = rag_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"context": context, "question": original_query})

    print("\n📝 final answer:")
    print(f"   {answer[:300]}{'...' if len(answer) > 300 else ''}")

    print("\n🎯 next steps after retrieval:")
    print("   1. combine retrieved docs into context")
    print("   2. create RAG prompt (context + question)")
    print("   3. generate answer using LLM")
    print("   4. (optional) add citations/source tracking")
    print("   5. (optional) implement streaming for long answers")


# endregion


# region Demo 8: RAG Chain Comparison
@requires_openai
def demo_chain_comparison() -> None:
    """
    compare different RAG chain approaches

    RAG Chain Types:
    ┌──────────────────────────────────────────────────────────────┐
    │                RAG Chain Architecture Patterns               │
    │                                                              │
    │  1. STUFF Chain (recommended for <4 docs):                   │
    │     Query → Retrieve → [All Docs] → LLM → Answer             │
    │                                                              │
    │  2. MAP_REDUCE Chain (large doc sets):                       │
    │     Query → Retrieve → [Doc1] → LLM → Summary1               │
    │                     ├─ [Doc2] → LLM → Summary2               │
    │                     └─ [Doc3] → LLM → Summary3               │
    │                           ↓                                  │
    │                   [All Summaries] → LLM → Answer             │
    │                                                              │
    │  3. REFINE Chain (iterative improvement):                    │
    │     Query → Retrieve → [Doc1] → LLM → Draft1                 │
    │                     ├─ [Draft1 + Doc2] → LLM → Draft2        │
    │                     └─ [Draft2 + Doc3] → LLM → Final         │
    │                                                              │
    │  4. MAP_RERANK Chain (score-based selection):                │
    │     Query → Retrieve → [Doc1] → LLM → (Answer1, Score1)      │
    │                     ├─ [Doc2] → LLM → (Answer2, Score2)      │
    │                     └─ [Doc3] → LLM → (Answer3, Score3)      │
    │                           ↓                                  │
    │                   Select Highest Score → Best Answer         │
    │                                                              │
    │  Trade-offs:                                                 │
    │  • Stuff: Fast, cheap, but limited by context window         │
    │  • Map_Reduce: Handles many docs, but more expensive         │
    │  • Refine: Good for iterative improvement, sequential        │
    │  • Map_Rerank: Best quality, but highest cost                │
    └──────────────────────────────────────────────────────────────┘
    """
    print_section("RAG Chain Comparison")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # load persistent vector store
    vectorstore = get_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    query = "What are the key concepts in AI?"

    # approach 1: basic stuff chain (most common)
    print("📊 approach 1: stuff chain (recommended)")
    print("   puts all docs in single prompt")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    prompt_template = cleandoc(
        """
        answer based on the following context:

        {context}

        question: {question}

        answer:
        """
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain1 = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response1 = chain1.invoke(query)
    print(f"   answer: {response1[:100]}...")

    # approach 2: with source citations
    print("\n📊 approach 2: with source citations")
    print("   includes metadata in response")

    def format_docs_with_sources(docs: list[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i}] source: {source}\n{doc.page_content}")
        return "\n\n".join(formatted)

    prompt_with_sources = ChatPromptTemplate.from_template(
        cleandoc(
            """
        answer based on the following sources:

        {context}

        question: {question}

        answer (cite sources using [1], [2] etc.):
        """
        )
    )

    chain2 = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt_with_sources
        | llm
        | StrOutputParser()
    )

    response2 = chain2.invoke(query)
    print(f"   answer: {response2[:100]}...")

    print("\n📋 chain type comparison:")
    print("   stuff: best for small doc sets (< 4 docs)")
    print("   map_reduce: for large doc sets, parallel processing")
    print("   refine: iterative refinement across docs")
    print("   map_rerank: scores each doc, picks best")


# endregion


# region Main Execution
# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Basic RAG Pipeline", "basic RAG pipeline", demo_basic_rag, needs_api=True),
    Demo("2", "Text Chunking Strategies", "text chunking strategies", demo_text_chunking),
    Demo("3", "Similarity Search Methods", "similarity search methods", demo_similarity_search, needs_api=True),
    Demo("4", "Metadata Filtering", "metadata filtering", demo_metadata_filtering, needs_api=True),
    Demo("5", "Document Loading from Files", "document loading from files", demo_document_loading),
    Demo("6", "Custom Retriever Configuration", "custom retriever configuration", demo_custom_retriever, needs_api=True),
    Demo("7", "Multi-Query RAG", "multi-query RAG", demo_multi_query, needs_api=True),
    Demo("8", "RAG Chain Comparison", "RAG chain comparison", demo_chain_comparison, needs_api=True),
]

# endregion


def main() -> None:
    """run RAG demonstrations with interactive menu"""
    has_openai, _ = check_api_keys()

    runner = MenuRunner(
        DEMOS,
        title="RAG (Retrieval-Augmented Generation) - Practical Examples",
        subtitle="Using OpenAI API for real RAG implementations",
        has_api=has_openai
    )
    runner.run()

    print("\n" + "=" * 70)
    print("  Thanks for exploring LangChain RAG!")
    print("  You now understand retrieval-augmented generation patterns")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    main()
# endregion
