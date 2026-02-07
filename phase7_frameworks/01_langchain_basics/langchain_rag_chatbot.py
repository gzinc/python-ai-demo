"""
Module: LangChain RAG Chatbot - Hands-On Implementation

Build a complete RAG chatbot using LangChain to compare with Phase 3 implementation.
See where framework saves time vs where it adds overhead.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.langchain_rag_chatbot

Requirements:
- OPENAI_API_KEY in .env (or use mock mode)

Note: This is a simplified demo showing LangChain concepts.
      For production, use full LangChain installation with `uv add langchain`.
"""

import os
from inspect import cleandoc

from dotenv import load_dotenv
from common.demo_menu import Demo, MenuRunner

# load environment variables
load_dotenv()

# suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# region Helper Functions
def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def check_langchain_installed() -> tuple[bool, str]:
    """check if full LangChain is installed"""
    try:
        import langchain  # noqa: F401
        return True, "Full LangChain installed"
    except ImportError:
        return False, "LangChain core packages only (need: uv add langchain)"


def check_api_key() -> bool:
    """check if OpenAI API key is available"""
    return bool(os.getenv("OPENAI_API_KEY"))
# endregion


# region Sample Documents
def get_sample_documents() -> list[dict]:
    """create sample documents about AI concepts"""
    docs = [
        {
            "content": """
            Embeddings are numerical representations of text that capture semantic meaning.
            They convert words, sentences, or documents into vectors (arrays of numbers).
            Similar concepts have similar vector representations, enabling semantic search.
            Common embedding models include OpenAI's text-embedding-3-small and
            sentence-transformers like all-MiniLM-L6-v2.
            """,
            "metadata": {"source": "embeddings.txt", "topic": "embeddings"}
        },
        {
            "content": """
            RAG (Retrieval-Augmented Generation) combines retrieval and generation.
            First, relevant documents are retrieved from a knowledge base using semantic search.
            Then, these documents provide context to an LLM for generating accurate answers.
            RAG reduces hallucinations and allows LLMs to access current information.
            The key components are: document chunking, embeddings, vector database, and LLM.
            """,
            "metadata": {"source": "rag.txt", "topic": "rag"}
        },
        {
            "content": """
            Vector databases store high-dimensional embeddings and enable fast similarity search.
            They use specialized algorithms like HNSW (Hierarchical Navigable Small World)
            or IVF (Inverted File Index) for efficient nearest neighbor search.
            Popular vector databases include Chroma, Pinecone, Weaviate, and pgvector.
            ChromaDB is great for development with its simple API and local storage.
            """,
            "metadata": {"source": "vectordb.txt", "topic": "databases"}
        },
        {
            "content": """
            LangChain is a framework for building LLM applications with modular components.
            It provides abstractions for prompts, chains, memory, agents, and retrievers.
            The LCEL (LangChain Expression Language) uses | operator for chaining.
            LangChain supports multiple LLM providers (OpenAI, Anthropic, etc.)
            through a unified interface. It includes 100+ pre-built tools and integrations.
            """,
            "metadata": {"source": "langchain.txt", "topic": "frameworks"}
        },
        {
            "content": """
            Agents are LLM-powered systems that can use tools and make decisions.
            The ReAct pattern combines reasoning (Thought) and acting (Action).
            Agents loop: Think → Choose Tool → Execute → Observe → Repeat until done.
            Common tools include web search, file operations, calculators, and APIs.
            LangChain provides create_react_agent and AgentExecutor for building agents.
            """,
            "metadata": {"source": "agents.txt", "topic": "agents"}
        },
    ]

    return [
        {"page_content": doc["content"].strip(), "metadata": doc["metadata"]}
        for doc in docs
    ]
# endregion


# region LangChain RAG Setup
def setup_rag_system(use_mock: bool = False) -> None:
    """
    set up RAG system with LangChain components (conceptual demo)

    compare this with Phase 3 where you:
    1. manually chunked documents (chunking.py)
    2. generated embeddings (embedder.py)
    3. stored in ChromaDB (rag_pipeline.py)
    4. retrieved and assembled context (retrieval.py)
    5. generated answer with LLM (rag_pipeline.py)

    LangChain abstracts steps 1-5 into a few lines!
    """
    print_section("Setting Up RAG System (Conceptual)")

    # 1. load documents
    documents = get_sample_documents()
    print(f"✅ Loaded {len(documents)} documents")

    # 2. split documents (chunking)
    # Phase 3 equivalent: chunking.py with paragraph/sentence/fixed strategies
    print("\n📝 Conceptual: LangChain would use RecursiveCharacterTextSplitter:")
    print(cleandoc('''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
    '''))
    print(f"✅ Would split into ~{len(documents) * 3} chunks")

    print("\n📝 Conceptual: LangChain would create embeddings and vector store:")
    print(cleandoc('''
        # Phase 3 equivalent: embedder.py + rag_pipeline.py db.add()
        # LangChain does both in one call!
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./.chroma_langchain_demo"
        )
    '''))
    print("✅ Would create embeddings and store in ChromaDB")

    print("\n📝 Conceptual: LangChain would create retriever:")
    print(cleandoc('''
        # Phase 3 equivalent: retrieval.py Retriever class
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    '''))
    print("✅ Would create retriever (k=3)")

    print("\n📝 Conceptual: LangChain would create LLM:")
    print(cleandoc('''
        # Phase 3 equivalent: raw OpenAI API calls
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=500
        )
    '''))
    print("✅ Would create LLM (gpt-4o-mini)")

    print("\n📝 Conceptual: LangChain would create RAG chain:")
    print(cleandoc('''
        # Phase 3 equivalent: entire rag_pipeline.py RAGPipeline.query() method!
        # LangChain combines retrieval + generation in one chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # stuff all retrieved docs into context
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
    '''))
    print("✅ Would create RetrievalQA chain")

    print("\n💡 Compare with Phase 3:")
    print("   - Phase 3: ~300 lines across 5 files (chunking, embedder, retrieval, pipeline)")
    print("   - LangChain: ~20 lines above (framework handles details)")
    print("   - Trade-off: Less code but less control over each step")

    print("\n⚠️  This is a conceptual demo. To run with real LangChain:")
    print("   1. Run: uv add langchain langchain-openai langchain-chroma")
    print("   2. Set OPENAI_API_KEY in .env")
    print("   3. See langchain_concepts_demo.py for full comparisons")

    return None
# endregion


# region RAG Query Examples
def demo_rag_queries(qa_chain: None = None) -> None:
    """demonstrate RAG queries with LangChain (conceptual)"""
    print_section("RAG Query Examples (Conceptual)")

    print("\n⚠️  Conceptual demo - shows what queries would look like")
    print("   To run with real LangChain, see setup instructions above")

    queries = [
        "What are embeddings?",
        "How does RAG work?",
        "What vector databases are mentioned?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print('─' * 80)

        print("\n📝 With LangChain, you would invoke the RAG chain:")
        print(cleandoc('''
            # Phase 3 equivalent: rag_pipeline.query(query)
            result = qa_chain.invoke({"query": query})

            print(f"Answer: {result['result']}")
            print(f"Sources: {len(result['source_documents'])} docs")
        '''))

        print("\n💡 Behind the scenes:")
        print("   1. Query embedded → vector")
        print("   2. Similarity search in ChromaDB")
        print("   3. Top 3 docs retrieved")
        print("   4. Context assembled and sent to LLM")
        print("   5. LLM generates answer based on context")
# endregion


# region Chat Memory Integration
def setup_rag_with_memory() -> tuple[None, None]:
    """
    set up RAG with conversation memory (conceptual demo)

    Phase 3 equivalent: combining rag_pipeline.py + chat_memory.py
    """
    print_section("RAG + Chat Memory Setup (Conceptual)")

    print("\n📝 Conceptual: LangChain memory integration")
    print("   This shows how LangChain combines RAG with conversation history")

    print("\n1️⃣ Setup vector store (same as before)")
    print(cleandoc('''
        documents = get_sample_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./.chroma_langchain_demo"
        )
    '''))

    print("\n2️⃣ Create conversation memory")
    print(cleandoc('''
        # Phase 3 equivalent: chat_memory.py ChatMemory class
        memory = ConversationBufferWindowMemory(
            k=5,  # keep last 5 message pairs
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    '''))
    print("✅ Would create conversation memory (sliding window, k=5)")

    print("\n3️⃣ Create conversational RAG chain")
    print(cleandoc('''
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # custom prompt that includes chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the context to answer questions."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context: {context}\\n\\nQuestion: {question}")
        ])

        # full conversational RAG with memory
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    '''))

    print("\n💡 Note: Full conversational RAG with memory is more complex")
    print("   - Requires ConversationalRetrievalChain")
    print("   - Handles follow-up questions and references")
    print("   - Your Phase 3 ChatEngine does this already!")

    return None, None
# endregion


# region Interactive Demo
def demo_interactive_chat() -> None:
    """demonstrate interactive RAG chat (conceptual demo)"""
    print_section("Interactive RAG Chat Demo (Conceptual)")

    print("\n⚠️  Conceptual demo - shows what interactive chat would look like")
    print("   Example conversation:")
    print("\n   User: What are embeddings?")
    print("   AI: [explains embeddings using retrieved context]")
    print("   User: What are they used for?")
    print("   AI: [understands 'they' refers to embeddings from memory]")

    print("\n📝 With real LangChain, the chat loop would be:")
    print(cleandoc('''
        # setup RAG system with memory
        qa_chain, memory = setup_rag_with_memory()

        # simple chat loop
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            # query RAG system
            result = qa_chain.invoke({"query": user_input})

            print(f"AI: {result['result']}")
            print(f"(Based on {len(result['source_documents'])} sources)")
    '''))

    print("\n💡 Key features of conversational RAG:")
    print("   - Memory: Remembers previous exchanges")
    print("   - Context: References 'they', 'it', 'that' work correctly")
    print("   - Retrieval: Finds relevant documents for each question")
    print("   - Generation: Uses both retrieved docs + conversation history")

    print("\n⚠️  To run interactive chat:")
    print("   1. Install: uv add langchain langchain-openai langchain-chroma")
    print("   2. Set OPENAI_API_KEY in .env")
    print("   3. Run the real implementation (after installation)")
# endregion


# region Comparison Summary
def demo_comparison_summary() -> None:
    """summarize LangChain vs Phase 3 implementation"""
    print_section("LangChain vs Phase 3: What We Learned")

    print("\n✅ WHERE LANGCHAIN HELPED:")
    print("   1. Quick setup: ~20 lines vs ~300 lines (Phase 3)")
    print("   2. Automatic chunking: RecursiveCharacterTextSplitter handles it")
    print("   3. Integrated storage: from_documents() does embed + store")
    print("   4. Unified retrieval: as_retriever() abstracts similarity search")
    print("   5. Chain composition: RetrievalQA combines all steps")

    print("\n⚠️  WHERE PHASE 3 WAS BETTER:")
    print("   1. Explicit control: You chose exact chunking strategy")
    print("   2. Easy debugging: Clear what happens at each step")
    print("   3. Custom logic: Token budgets, smart retrieval, metadata filters")
    print("   4. Understanding: You know exactly what the code does")
    print("   5. Optimization: Your Phase 5 patterns (batching, caching) still needed")

    print("\n🔀 HYBRID APPROACH (Recommended):")
    print("   - Use LangChain for standard RAG setup (saves time)")
    print("   - Drop to Phase 3 patterns for custom retrieval logic")
    print("   - Apply Phase 5 optimization patterns (batching, semantic cache)")
    print("   - Example: LangChain RAG + your custom chunking + semantic cache")

    print("\n💡 KEY INSIGHT:")
    print("   You built Phase 3 from scratch → understand what LangChain abstracts")
    print("   Now you can choose framework vs custom based on needs")
    print("   Framework != requirement, it's a tool in your toolbox")
# endregion


# region Main


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Basic RAG Setup", "document loading and vectorization", demo_basic_rag_setup),
    Demo("2", "RAG Query", "query with retrieval", demo_rag_query),
    Demo("3", "RAG Chat Session", "multi-turn conversation with RAG", demo_rag_chat),
    Demo("4", "Comparison: RAG vs Phase 3", "framework vs custom implementation", demo_comparison),
    Demo("5", "Interactive RAG Chat", "full interactive session", demo_interactive_rag_chat),
]

# endregion

def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="LangChain Concepts Demo")
    runner.run()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


# endregion
