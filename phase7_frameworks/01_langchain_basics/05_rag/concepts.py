"""
RAG (Retrieval-Augmented Generation) - Conceptual Demonstrations

This module demonstrates RAG concepts without requiring API keys.
Shows document processing, chunking, embedding concepts, and retrieval logic.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.05_rag.concepts
"""

from inspect import cleandoc
import math
from typing import Any

from common.util.utils import print_section

from common.demo_menu import Demo, MenuRunner


# region Mock Document Classes
class MockDocument:
    """simple document representation for demonstrations"""

    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(content='{self.page_content[:50]}...', metadata={self.metadata})"


class MockEmbedding:
    """simple embedding representation (mock vector)"""

    def __init__(self, vector: list[float], text: str):
        self.vector = vector
        self.text = text

    def __repr__(self) -> str:
        return f"Embedding(dims={len(self.vector)}, text='{self.text[:30]}...')"


# endregion


# region Demo 1: Document Loading Patterns
def demo_document_loading() -> None:
    """demonstrate document loading from different sources"""
    print_section("Document Loading Patterns")

    # simulated PDF document loading
    pdf_docs = [
        MockDocument(
            page_content="Introduction to AI and Machine Learning",
            metadata={"source": "ai_guide.pdf", "page": 0},
        ),
        MockDocument(
            page_content="Deep Learning fundamentals and neural networks",
            metadata={"source": "ai_guide.pdf", "page": 1},
        ),
        MockDocument(
            page_content="Natural Language Processing with transformers",
            metadata={"source": "ai_guide.pdf", "page": 2},
        ),
    ]

    print("📄 loaded PDF documents:")
    for doc in pdf_docs:
        print(f"  • page {doc.metadata['page']}: {doc.page_content[:50]}...")

    # simulated text file loading
    text_doc = MockDocument(
        page_content=cleandoc(
            """
        RAG (Retrieval-Augmented Generation) combines retrieval with generation.
        It retrieves relevant documents and uses them as context for LLM generation.
        This improves factual accuracy and reduces hallucinations.
        """
        ),
        metadata={"source": "rag_notes.txt"},
    )

    print(f"\n📝 loaded text document:")
    print(f"  source: {text_doc.metadata['source']}")
    print(f"  content: {text_doc.page_content[:80]}...")

    # simulated web scraping
    web_doc = MockDocument(
        page_content="Latest AI research shows improvements in reasoning capabilities",
        metadata={"source": "https://ai-news.com/article", "title": "AI Progress 2024"},
    )

    print(f"\n🌐 loaded web document:")
    print(f"  url: {web_doc.metadata['source']}")
    print(f"  title: {web_doc.metadata['title']}")


# endregion


# region Demo 2: Text Chunking Strategies
def demo_text_chunking() -> None:
    """demonstrate different text chunking strategies"""
    print_section("Text Chunking Strategies")

    sample_text = cleandoc(
        """
    Artificial Intelligence (AI) is transforming technology. Machine learning enables
    computers to learn from data. Deep learning uses neural networks with multiple layers.
    Natural Language Processing (NLP) helps computers understand human language.
    Large Language Models (LLMs) can generate human-like text. Retrieval-Augmented
    Generation (RAG) combines retrieval with generation for better accuracy.
    """
    )

    print("📖 original text:")
    print(f"{sample_text}\n")
    print(f"total length: {len(sample_text)} characters")

    # strategy 1: fixed size chunks with overlap
    chunk_size = 100
    chunk_overlap = 20

    chunks = []
    start = 0
    while start < len(sample_text):
        end = start + chunk_size
        chunks.append(sample_text[start:end])
        start = end - chunk_overlap  # overlap for context preservation

    print(f"\n✂️ chunking strategy: fixed size (size={chunk_size}, overlap={chunk_overlap})")
    for i, chunk in enumerate(chunks, 1):
        print(f"  chunk {i}: '{chunk.strip()}'")

    # strategy 2: sentence-based chunking
    sentences = sample_text.split(". ")
    sentence_chunks = [sentences[i: i + 2] for i in range(0, len(sentences), 2)]

    print(f"\n✂️ chunking strategy: sentence-based (2 sentences per chunk)")
    for i, chunk in enumerate(sentence_chunks, 1):
        chunk_text = ". ".join(chunk)
        print(f"  chunk {i}: '{chunk_text}'")


# endregion


# region Demo 3: Chunk Size Trade-offs
def demo_chunk_size_tradeoffs() -> None:
    """demonstrate trade-offs between different chunk sizes"""
    print_section("Chunk Size Trade-offs")

    document = "A" * 5000  # simulate 5000 character document

    configs = [
        {"size": 500, "overlap": 50},
        {"size": 1000, "overlap": 100},
        {"size": 2000, "overlap": 200},
    ]

    print("📊 chunk size comparison:")
    print(f"{'size':<10} {'overlap':<10} {'chunks':<10} {'tokens~':<12} {'context':<10}")
    print("-" * 60)

    for config in configs:
        chunk_size = config["size"]
        overlap = config["overlap"]

        # calculate number of chunks
        num_chunks = math.ceil(len(document) / (chunk_size - overlap))

        # estimate tokens (rough: 1 token ≈ 4 chars)
        tokens_per_chunk = chunk_size // 4
        total_tokens = num_chunks * tokens_per_chunk

        # context quality (higher overlap = better context preservation)
        context_score = "high" if overlap / chunk_size > 0.15 else "medium"

        print(
            f"{chunk_size:<10} {overlap:<10} {num_chunks:<10} {total_tokens:<12} {context_score:<10}"
        )

    print("\n💡 insights:")
    print("  • smaller chunks: better precision, more chunks to search")
    print("  • larger chunks: more context, fewer chunks, higher token cost per retrieval")
    print("  • overlap: preserves context at chunk boundaries")


# endregion


# region Demo 4: Mock Embedding Generation
def demo_mock_embeddings() -> None:
    """demonstrate embedding concept with mock vectors"""
    print_section("Embedding Generation (Mock)")

    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text",
        "Python is a programming language",
    ]

    # generate mock embeddings (in reality, these would be from a model)
    # we'll create simple vectors where similar texts have similar vectors
    embeddings = []

    # mock vectors: [ai_related, neural_related, language_related, tech_related]
    mock_vectors = [
        [0.9, 0.3, 0.2, 0.5],  # ML - high AI, some tech
        [0.8, 0.9, 0.1, 0.4],  # DL - high AI, high neural
        [0.5, 0.1, 0.9, 0.3],  # NLP - some AI, high language
        [0.2, 0.0, 0.4, 0.9],  # Python - low AI, high tech
    ]

    for text, vector in zip(texts, mock_vectors):
        embeddings.append(MockEmbedding(vector, text))

    print("🔢 mock embeddings generated:")
    for i, emb in enumerate(embeddings, 1):
        print(f"  {i}. '{emb.text}'")
        print(f"     vector: {emb.vector}")

    print("\n💡 in reality:")
    print("  • embeddings are high-dimensional (384-3072 dimensions)")
    print("  • generated by neural networks (OpenAI, HuggingFace)")
    print("  • semantically similar texts have similar vectors")


# endregion


# region Demo 5: Vector Similarity Calculation
def demo_vector_similarity() -> None:
    """demonstrate cosine similarity for vector search"""
    print_section("Vector Similarity (Cosine)")

    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2)

    # mock document embeddings
    docs = [
        ("AI and machine learning fundamentals", [0.9, 0.8, 0.2]),
        ("Deep neural network architectures", [0.8, 0.9, 0.1]),
        ("Natural language understanding", [0.3, 0.2, 0.9]),
        ("Python programming basics", [0.1, 0.1, 0.3]),
    ]

    query = "How does deep learning work?"
    query_vector = [0.7, 0.9, 0.2]  # mock query embedding

    print(f"🔍 query: '{query}'")
    print(f"query vector: {query_vector}")

    print("\n📊 similarity scores:")
    similarities = []
    for text, vector in docs:
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((text, similarity))
        print(f"  {similarity:.3f} - '{text}'")

    # sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🎯 top 2 most relevant documents:")
    for i, (text, score) in enumerate(similarities[:2], 1):
        print(f"  {i}. ({score:.3f}) '{text}'")


# endregion


# region Demo 6: Retrieval Logic Patterns
def demo_retrieval_logic() -> None:
    """demonstrate different retrieval strategies"""
    print_section("Retrieval Logic Patterns")

    # mock documents with embeddings and metadata
    docs = [
        {
            "text": "Machine learning enables computers to learn from data",
            "vector": [0.9, 0.7, 0.2],
            "date": "2024-01-15",
        },
        {
            "text": "Deep learning uses neural networks with multiple layers",
            "vector": [0.8, 0.9, 0.3],
            "date": "2024-01-10",
        },
        {
            "text": "Natural language processing analyzes human language",
            "vector": [0.3, 0.2, 0.8],
            "date": "2024-01-20",
        },
        {
            "text": "Computer vision enables image understanding",
            "vector": [0.4, 0.8, 0.1],
            "date": "2024-01-05",
        },
    ]

    query_vector = [0.7, 0.8, 0.2]  # query: "deep learning neural networks"

    # strategy 1: simple similarity search (top k)
    print("🔍 strategy 1: similarity search (k=2)")

    def cosine_sim(v1: list[float], v2: list[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2)) / (
                math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2))
        )

    results = [(doc, cosine_sim(query_vector, doc["vector"])) for doc in docs]
    results.sort(key=lambda x: x[1], reverse=True)

    for i, (doc, score) in enumerate(results[:2], 1):
        print(f"  {i}. ({score:.3f}) {doc['text']}")

    # strategy 2: mmr (max marginal relevance) - diversity
    print("\n🎯 strategy 2: MMR (diversity-aware)")
    print("  goal: balance relevance with diversity")

    selected = [results[0]]  # start with most relevant

    # for second doc, consider diversity from first
    for doc, score in results[1:]:
        diversity = 1 - cosine_sim(doc["vector"], selected[0][0]["vector"])
        mmr_score = 0.7 * score + 0.3 * diversity  # 70% relevance, 30% diversity
        doc["mmr_score"] = mmr_score

    # pick highest mmr score
    remaining = [(doc, score) for doc, score in results[1:]]
    remaining.sort(key=lambda x: x[0].get("mmr_score", 0), reverse=True)

    selected.append(remaining[0])

    for i, (doc, score) in enumerate(selected, 1):
        print(f"  {i}. {doc['text']}")

    # strategy 3: metadata filtering
    print("\n📅 strategy 3: metadata filtering (date >= 2024-01-15)")
    filtered = [
        (doc, cosine_sim(query_vector, doc["vector"]))
        for doc in docs
        if doc["date"] >= "2024-01-15"
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    for i, (doc, score) in enumerate(filtered, 1):
        print(f"  {i}. ({score:.3f}) {doc['text']} (date: {doc['date']})")


# endregion


# region Demo 7: RAG Pipeline Walkthrough
def demo_rag_pipeline() -> None:
    """walk through complete RAG pipeline with mock data"""
    print_section("RAG Pipeline Walkthrough (Mock)")

    # step 1: document loading
    print("📚 step 1: document loading")
    docs = [
        "Python is a high-level programming language. It emphasizes code readability.",
        "Machine learning is a subset of AI. It uses data to improve performance.",
        "RAG combines retrieval with generation. It improves factual accuracy.",
    ]
    print(f"  loaded {len(docs)} documents")

    # step 2: chunking
    print("\n✂️ step 2: text chunking")
    chunks = []
    for doc in docs:
        # split by sentence for simplicity
        chunks.extend([s.strip() for s in doc.split(".") if s.strip()])
    print(f"  created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"    {i}. '{chunk}'")

    # step 3: embedding
    print("\n🔢 step 3: embedding generation")
    # mock embeddings (in reality, these come from embedding model)
    embeddings = [
        [0.2, 0.1, 0.8],  # Python chunk
        [0.1, 0.2, 0.9],  # readability chunk
        [0.9, 0.8, 0.2],  # ML chunk
        [0.8, 0.9, 0.3],  # AI/data chunk
        [0.7, 0.6, 0.4],  # RAG chunk
        [0.6, 0.7, 0.3],  # accuracy chunk
    ]
    print(f"  generated embeddings for {len(embeddings)} chunks")

    # step 4: vector store (simulated)
    print("\n💾 step 4: vector store creation")
    vector_store = list(zip(chunks, embeddings))
    print(f"  stored {len(vector_store)} vectors in database")

    # step 5: query and retrieval
    print("\n🔍 step 5: query and retrieval")
    query = "What is machine learning?"
    query_vector = [0.85, 0.75, 0.25]  # mock query embedding

    print(f"  query: '{query}'")

    # calculate similarities
    def cosine_sim(v1: list[float], v2: list[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2)) / (
                math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2))
        )

    similarities = [(chunk, cosine_sim(query_vector, emb)) for chunk, emb in vector_store]
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  top 2 retrieved chunks:")
    context_chunks = similarities[:2]
    for i, (chunk, score) in enumerate(context_chunks, 1):
        print(f"    {i}. ({score:.3f}) '{chunk}'")

    # step 6: context assembly
    print("\n📝 step 6: context assembly")
    context = " ".join(chunk for chunk, _ in context_chunks)
    print(f"  context: '{context}'")

    # step 7: llm generation (simulated)
    print("\n🤖 step 7: LLM generation (simulated)")
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    print(f"  prompt sent to LLM:")
    print(f"  '{prompt}'")

    print("\n  simulated LLM response:")
    print(
        "  'Machine learning is a subset of AI that uses data to improve performance over time.'"
    )

    print("\n✅ rag pipeline complete!")


# endregion


# region Demo 8: RAG vs Traditional Generation
def demo_rag_vs_traditional() -> None:
    """compare RAG with traditional LLM generation"""
    print_section("RAG vs Traditional Generation")

    question = "What was the company's Q4 2023 revenue?"

    print(f"❓ question: '{question}'")

    # traditional llm (no context)
    print("\n🤖 traditional LLM (no context):")
    print("  response: 'I don't have access to specific company financial data.'")
    print("  or: 'Based on my training data, I cannot provide Q4 2023 revenue.'")
    print("  ⚠️ issue: no access to company-specific information")

    # rag approach
    print("\n📚 rag approach:")
    print("  step 1: retrieve relevant documents from company database")
    retrieved_docs = [
        "Q4 2023 financial report: Total revenue was $125.4M",
        "Year-over-year growth in Q4 2023 reached 18%",
    ]

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"    retrieved {i}: '{doc}'")

    print("\n  step 2: provide context to LLM")
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\n\nQuestion: {question}"
    print(f"    prompt: '{prompt[:80]}...'")

    print("\n  step 3: LLM generates answer with context")
    print("    response: 'According to the Q4 2023 financial report, ")
    print("    the company's revenue was $125.4M, with 18% year-over-year growth.'")
    print("    ✅ accurate, factual, cited answer")

    print("\n📊 comparison:")
    print("  traditional LLM:")
    print("    • limited to training data")
    print("    • may hallucinate or refuse to answer")
    print("    • no company-specific knowledge")

    print("\n  RAG:")
    print("    • retrieves current information")
    print("    • provides factual, cited answers")
    print("    • works with proprietary data")
    print("    • reduces hallucinations")


# endregion


# region Main Execution



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Document Loading", "document loading", demo_document_loading),
    Demo("2", "Text Chunking", "text chunking", demo_text_chunking),
    Demo("3", "Chunk Size Tradeoffs", "chunk size tradeoffs", demo_chunk_size_tradeoffs),
    Demo("4", "Mock Embeddings", "mock embeddings", demo_mock_embeddings),
    Demo("5", "Vector Similarity", "vector similarity", demo_vector_similarity),
    Demo("6", "Retrieval Logic", "retrieval logic", demo_retrieval_logic),
    Demo("7", "RAG Pipeline", "rag pipeline", demo_rag_pipeline),
    Demo("8", "RAG vs Traditional", "rag vs traditional", demo_rag_vs_traditional),
]

# endregion

def main() -> None:
    """run demonstrations with interactive menu"""
    print("\n" + "=" * 70)
    print("  RAG (Retrieval-Augmented Generation) - Conceptual Understanding")
    print("  No API key required - demonstrates patterns only")
    print("=" * 70)

    
    runner = MenuRunner(DEMOS, title="TODO: Add title")
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


# endregion
