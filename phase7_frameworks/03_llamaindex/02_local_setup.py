"""
LlamaIndex Local Setup - 100% Privacy-Friendly

This module demonstrates running LlamaIndex completely locally:
- Local LLM via Ollama (llama3.1)
- Local embeddings (HuggingFace)
- Local vector store (ChromaDB)
- NO API keys required!

Prerequisites:
1. Install Ollama: https://ollama.com
2. Pull model: ollama pull llama3.1
3. pip install llama-index-llms-ollama llama-index-embeddings-huggingface

Run with: uv run python phase7_frameworks/03_llamaindex/02_local_setup.py
"""

from pathlib import Path
from inspect import cleandoc

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section


# region Utility: Check Ollama


def check_ollama_available() -> bool:
    """check if Ollama is running and has required model"""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            has_llama = any("llama3" in name or "llama2" in name for name in model_names)
            return has_llama
        return False
    except Exception:
        return False


# endregion


# region Demo 1: Basic Local Setup


def demo_local_basic():
    """demonstrate basic local RAG with Ollama + HuggingFace"""
    print_section("Demo 1: Basic Local Setup")

    if not check_ollama_available():
        print(
            cleandoc("""
            ❌ Ollama not detected or no llama model found!

            Setup instructions:
            1. Install Ollama: https://ollama.com
            2. Run: ollama pull llama3.1
            3. Verify: ollama list

            Skipping demo...
        """)
        )
        return

    print("✅ Ollama detected with llama model\n")

    # configure local LLM
    from llama_index.llms.ollama import Ollama

    print("🔧 Configuring local LLM (Ollama)...")
    Settings.llm = Ollama(
        model="llama3.1:latest",
        request_timeout=120.0,
        temperature=0.1,
    )
    print("   Model: llama3.1")
    print("   Provider: Ollama (local)")

    # configure local embeddings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    print("\n🔧 Configuring local embeddings (HuggingFace)...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("   Model: BAAI/bge-small-en-v1.5")
    print("   Provider: HuggingFace (local)")
    print("   Note: First run will download model (~100MB)")

    # create sample documents
    documents = [
        Document(
            text=cleandoc("""
                Retrieval Augmented Generation (RAG) is a technique that combines
                retrieval and generation. It retrieves relevant documents from a
                knowledge base and uses them to generate accurate, grounded responses.
                This reduces hallucinations and improves factual accuracy.
            """)
        ),
        Document(
            text=cleandoc("""
                Local LLMs like Llama 3.1 can run entirely on your machine.
                Benefits include complete privacy, no API costs, and offline capability.
                Trade-offs include slower inference and lower quality compared to GPT-4.
                Good hardware (16GB+ RAM, GPU) improves performance significantly.
            """)
        ),
        Document(
            text=cleandoc("""
                Ollama makes running local LLMs easy. It handles model downloads,
                server management, and provides a simple API. Popular models include
                Llama 3.1, Mistral, and CodeLlama. Models run via optimized llama.cpp.
            """)
        ),
    ]

    print(f"\n📚 Creating index with {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(documents)
    print("✅ Index created (using local embeddings)")

    # query the index
    query_engine = index.as_query_engine(similarity_top_k=2)
    query = "What are the benefits of local LLMs?"

    print(f"\n💬 Query: {query}")
    print("⏳ Generating response (this may take 10-30 seconds)...\n")

    response = query_engine.query(query)

    print("📝 Response:")
    print(f"{response.response}\n")

    print(f"📚 Retrieved {len(response.source_nodes)} chunks")
    print("✅ Everything ran locally - no API calls!")


# endregion


# region Demo 2: Persistent Local Vector Store


def demo_local_persistent():
    """demonstrate persistent storage with ChromaDB"""
    print_section("Demo 2: Persistent Local Vector Store")

    if not check_ollama_available():
        print("❌ Ollama not available. Skipping demo...")
        return

    # configure local LLM and embeddings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # configure ChromaDB for persistence
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    db_path = "./temp_chroma_db"
    print(f"📁 Using ChromaDB at: {db_path}")

    # create/load chroma collection
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("local_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # check if collection already has data
    existing_count = chroma_collection.count()

    if existing_count > 0:
        print(f"✅ Found existing collection with {existing_count} items")
        print("   Loading existing index...")

        # load existing index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("📝 Creating new collection...")

        # create new documents
        documents = [
            Document(
                text=cleandoc("""
                    ChromaDB is an open-source vector database designed for AI applications.
                    It stores embeddings and enables fast similarity search.
                    ChromaDB can run in-memory or persist data to disk.
                    It integrates seamlessly with LlamaIndex and LangChain.
                """)
            ),
            Document(
                text=cleandoc("""
                    Vector databases are optimized for similarity search using embeddings.
                    Unlike traditional databases, they find semantically similar content.
                    Popular options include ChromaDB, Pinecone, Qdrant, and Weaviate.
                    Local options (ChromaDB, FAISS) are free but less scalable.
                """)
            ),
        ]

        # create index with persistence
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"✅ Created and persisted {len(documents)} documents")

    # query the persistent index
    query_engine = index.as_query_engine()
    query = "What is ChromaDB and how does it work?"

    print(f"\n💬 Query: {query}")
    print("⏳ Generating response...\n")

    response = query_engine.query(query)
    print(f"📝 Response:\n{response.response}\n")

    print(
        cleandoc(f"""
        💡 Key Points:
        • Data persisted to: {db_path}
        • Run again to see it load existing data
        • Survives restarts - no re-indexing needed
        • 100% local - no cloud dependencies
    """)
    )

    # cleanup
    import shutil

    shutil.rmtree(db_path, ignore_errors=True)
    print("\n🧹 Cleaned up temp database")


# endregion


# region Demo 3: Local Chat Engine


def demo_local_chat():
    """demonstrate conversational RAG with memory"""
    print_section("Demo 3: Local Chat Engine")

    if not check_ollama_available():
        print("❌ Ollama not available. Skipping demo...")
        return

    # configure
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # create knowledge base
    documents = [
        Document(
            text=cleandoc("""
                LlamaIndex chat engines maintain conversation history automatically.
                They combine retrieval with conversational context for coherent dialogues.
                The chat engine remembers previous questions and answers.
                This enables follow-up questions and multi-turn conversations.
            """)
        ),
        Document(
            text=cleandoc("""
                Chat engines in LlamaIndex come in different modes:
                - CondenseQuestion mode reformulates queries based on history
                - React mode uses an agent loop for complex queries
                - Simple mode for basic conversational retrieval
                Each mode balances complexity and performance differently.
            """)
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # create chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question", similarity_top_k=2
    )

    print("💬 Starting conversation (with memory):\n")

    # simulate conversation
    conversations = [
        "What is a chat engine?",
        "What modes are available?",  # follow-up question
        "Which mode is best for complex queries?",  # another follow-up
    ]

    for i, message in enumerate(conversations, 1):
        print(f"User: {message}")
        print("⏳ Thinking...\n")

        response = chat_engine.chat(message)

        print(f"Assistant: {response.response}\n")
        print("-" * 70)

    print(
        cleandoc("""

        💡 Notice:
        • Follow-up questions work without re-stating context
        • Chat engine maintains conversation history
        • Completely local - privacy preserved
        • No conversation data sent to cloud
    """)
    )


# endregion


# region Demo 4: Local vs Cloud Comparison


def demo_local_vs_cloud():
    """compare local and cloud setups side-by-side"""
    print_section("Demo 4: Local vs Cloud Comparison")

    print(
        cleandoc("""
        📊 Setup Comparison:

        ┌─────────────────┬──────────────────────┬──────────────────────┐
        │ Aspect          │ Local (Ollama)       │ Cloud (OpenAI)       │
        ├─────────────────┼──────────────────────┼──────────────────────┤
        │ Cost            │ Free (after HW)      │ $0.01-0.06 per 1K    │
        │ Privacy         │ 100% private         │ Data sent to cloud   │
        │ Speed           │ Slower (your HW)     │ Fast (cloud GPUs)    │
        │ Quality         │ Good                 │ Excellent (GPT-4)    │
        │ Offline         │ Yes                  │ No                   │
        │ Setup           │ Medium complexity    │ Easy (API key)       │
        │ Scalability     │ Limited by HW        │ Unlimited            │
        │ Dependencies    │ Ollama required      │ Internet required    │
        └─────────────────┴──────────────────────┴──────────────────────┘

        💰 Cost Analysis (1M tokens/month):

        Local Setup:
        • LLM: $0 (runs on your hardware)
        • Embeddings: $0 (HuggingFace local)
        • Vector DB: $0 (ChromaDB local)
        • Total: $0/month (electricity only)

        Cloud Setup:
        • LLM: ~$10-60 (GPT-4 Turbo)
        • Embeddings: ~$0.10 (OpenAI)
        • Vector DB: ~$70 (Pinecone)
        • Total: ~$80-130/month

        🎯 Recommendation:
        • Development/Learning: Local (free, private)
        • Sensitive data: Local (HIPAA, GDPR compliance)
        • Production scale: Cloud (better quality/speed)
        • Hybrid: Local embeddings + Cloud LLM (balance cost/quality)

        ⚡ Performance (typical):
        • Local (CPU): 5-15 tokens/sec
        • Local (GPU): 30-100 tokens/sec
        • Cloud (OpenAI): 50-200 tokens/sec

        📝 Quality Comparison:
        • Llama 3.1 8B: 7-8/10 (good for most tasks)
        • GPT-4 Turbo: 9-10/10 (excellent)
        • GPT-3.5 Turbo: 7-8/10 (similar to Llama 3.1)
    """)
    )


# endregion


# region Main Menu


DEMOS = [
    Demo("1", "Basic Local Setup", "Ollama + HuggingFace", demo_local_basic),
    Demo("2", "Persistent Storage", "ChromaDB local vector store", demo_local_persistent),
    Demo("3", "Local Chat Engine", "conversational RAG with memory", demo_local_chat),
    Demo("4", "Local vs Cloud", "comparison and recommendations", demo_local_vs_cloud),
]


def main() -> None:
    """run interactive demo menu"""
    if not check_ollama_available():
        print(
            cleandoc("""
            ⚠️  Ollama not detected!

            To run these demos:
            1. Install Ollama: https://ollama.com
            2. Pull a model: ollama pull llama3.1
            3. Verify: ollama list

            Most demos will be skipped without Ollama.
            Demo 4 (comparison) will still work.
        """)
        )
        print()

    runner = MenuRunner(
        DEMOS,
        title="LlamaIndex Local Setup",
        subtitle="100% Privacy-Friendly RAG",
    )
    runner.run()


# endregion

if __name__ == "__main__":
    main()
