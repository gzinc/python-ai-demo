"""
Module: LangChain Concepts Demo - Code Patterns Without Dependencies

Shows LangChain code patterns and compares with Phase 3.
This is a conceptual demo - doesn't require full LangChain installation.

For hands-on LangChain: `uv add langchain` then use langchain_rag_chatbot.py

Run with: uv run python -m phase7_frameworks.01_langchain_basics.langchain_concepts_demo
"""

import os
from inspect import cleandoc
from common.demo_menu import Demo, MenuRunner

# suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# region Helper Functions
def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def print_code(label: str, code: str) -> None:
    """print code block with label"""
    print(f"\n┌─ {label} {'─' * (70 - len(label))}┐")
    print(code)
    print(f"└{'─' * 78}┘")
# endregion


# region 1. RAG Setup Comparison
def demo_rag_setup() -> None:
    """compare RAG setup: Phase 3 vs LangChain"""
    print_section("1. RAG System Setup")

    phase3_code = cleandoc('''
        # Phase 3: Manual RAG setup (5 files, ~300 lines)

        # chunking.py
        chunker = DocumentChunker(strategy="paragraph")
        chunks = chunker.chunk(document.content)

        # embedder.py (with ONNX Runtime)
        embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
        embeddings = embedder.embed_batch(chunks)

        # rag_pipeline.py
        db = ChromaDB(persist_directory="./chroma_db")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            db.add(chunk, embedding, metadata)

        # retrieval.py
        retriever = Retriever(db=db, k=3)
        results = retriever.search(query_embedding)

        # rag_pipeline.py - generate answer
        context = "\\n\\n".join([r.content for r in results])
        prompt = f"Context: {context}\\n\\nQuestion: {query}"
        answer = openai_client.chat.completions.create(...)
    ''')

    langchain_code = cleandoc('''
        # LangChain: Abstracted RAG setup (~20 lines)

        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA

        # 1. load and chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        # 2. create vector store (embeds + stores automatically!)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )

        # 3. create RAG chain (retrieval + generation combined!)
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

        # 4. query
        answer = qa_chain.invoke({"query": "What are embeddings?"})
    ''')

    print_code("PHASE 3: Explicit Control", phase3_code)
    print_code("LANGCHAIN: Abstracted", langchain_code)

    print("\n💡 ANALYSIS:")
    print("   ✅ LangChain wins: 20 lines vs 300 lines, faster to build")
    print("   ✅ Phase 3 wins: Full control over chunking, retrieval, context assembly")
    print("   🔀 Hybrid: Use LangChain setup + Phase 3 custom retrieval logic")


def demo_rag_query() -> None:
    """show RAG query flow comparison"""
    print_section("2. RAG Query Flow")

    print("\n📋 PHASE 3 FLOW (Explicit Steps):")
    print(cleandoc('''
        query = "What are embeddings?"

        # 1. embed query
        query_embedding = embedder.embed(query)

        # 2. retrieve relevant chunks
        results = retriever.search(query_embedding, k=3)

        # 3. assemble context
        context = retriever.assemble_context(results)

        # 4. build prompt
        prompt = f"Context: {context}\\n\\nQuestion: {query}"

        # 5. generate answer
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

        # you control each step, easy to debug, can customize everything
    '''))

    print("\n📋 LANGCHAIN FLOW (Abstracted):")
    print(cleandoc('''
        query = "What are embeddings?"

        # all steps combined into one call!
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

        # under the hood:
        # - query gets embedded (automatic)
        # - similarity search (automatic)
        # - context assembly (automatic)
        # - prompt building (automatic)
        # - LLM call (automatic)

        # faster to write, but less visibility into each step
    '''))

    print("\n💡 TRADE-OFF:")
    print("   - Debugging: Phase 3 easier (see each step)")
    print("   - Speed: LangChain faster (less code)")
    print("   - Customization: Phase 3 more flexible")
# endregion


# region 2. Memory Integration
def demo_memory() -> None:
    """compare conversation memory approaches"""
    print_section("3. Conversation Memory")

    phase3_code = cleandoc('''
        # Phase 3: Custom ChatMemory class

        class ChatMemory:
            def __init__(self, strategy: str = "sliding_window", max_messages: int = 10):
                self.strategy = strategy
                self.max_messages = max_messages
                self.messages: list[dict] = []

            def add_message(self, role: str, content: str) -> None:
                self.messages.append({"role": role, "content": content})

                if self.strategy == "sliding_window":
                    self.messages = self.messages[-self.max_messages:]
                elif self.strategy == "token_budget":
                    self._trim_to_token_budget()

            def get_messages(self) -> list[dict]:
                return self.messages

        # flexible: can add token budgets, summarization, custom logic
        memory = ChatMemory(strategy="sliding_window", max_messages=10)
        memory.add_message("user", "What are embeddings?")
        memory.add_message("assistant", "Embeddings are...")
    ''')

    langchain_code = cleandoc('''
        # LangChain: Pre-built memory strategies

        from langchain.memory import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(k=10)
        memory.save_context(
            {"input": "What are embeddings?"},
            {"output": "Embeddings are..."}
        )

        messages = memory.load_memory_variables({})

        # also available:
        # - ConversationSummaryMemory (LLM summarizes old messages)
        # - ConversationBufferMemory (unlimited history)
        # - VectorStoreBackedMemory (semantic retrieval of past conversations)

        # pre-built and tested, but less control over custom logic
    ''')

    print_code("PHASE 3: Custom Logic", phase3_code)
    print_code("LANGCHAIN: Pre-built Strategies", langchain_code)

    print("\n💡 INSIGHT:")
    print("   - Phase 3: Your token budget feature doesn't exist in LangChain!")
    print("   - LangChain: Summary and vector-based memory are pre-built")
    print("   - Hybrid: Use LangChain memory + your token budget logic")
# endregion


# region 3. Chains and LCEL
def demo_chains() -> None:
    """demonstrate LangChain Expression Language (LCEL)"""
    print_section("4. Chains & LCEL Syntax")

    phase3_code = cleandoc('''
        # Phase 3: Manual function composition

        def summarize_then_analyze(text: str) -> dict[str, str]:
            # step 1: summarize
            summary_prompt = f"Summarize this: {text}"
            summary = llm_call(summary_prompt)

            # step 2: analyze sentiment
            sentiment_prompt = f"Analyze sentiment: {summary}"
            sentiment = llm_call(sentiment_prompt)

            return {"summary": summary, "sentiment": sentiment}

        # explicit, easy to debug, clear control flow
        result = summarize_then_analyze("Long text...")
    ''')

    langchain_code = cleandoc('''
        # LangChain: LCEL (pipe operator composition)

        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOpenAI(model="gpt-4o-mini")

        # define steps
        summarize = (
            ChatPromptTemplate.from_template("Summarize: {text}")
            | llm
            | StrOutputParser()
        )

        analyze = (
            ChatPromptTemplate.from_template("Analyze sentiment: {summary}")
            | llm
            | StrOutputParser()
        )

        # compose with | operator (pipe)
        chain = summarize | analyze
        result = chain.invoke({"text": "Long text..."})

        # benefits: streaming, async, error handling built-in
        # cost: less explicit, harder to debug intermediate steps
    ''')

    print_code("PHASE 3: Explicit Functions", phase3_code)
    print_code("LANGCHAIN: LCEL Composition", langchain_code)

    print("\n💡 LCEL BENEFITS:")
    print("   ✅ Streaming: chain.stream() gives token-by-token output")
    print("   ✅ Async: chain.ainvoke() for concurrent execution")
    print("   ✅ Error handling: automatic retries and fallbacks")
    print("   ✅ Monitoring: integrates with LangSmith tracing")

    print("\n⚠️  LCEL COSTS:")
    print("   ❌ Debugging: Harder to inspect intermediate values")
    print("   ❌ Learning curve: | operator syntax non-obvious")
    print("   ❌ Abstraction: Don't see what's happening under hood")
# endregion


# region 4. Practical Examples
def demo_practical_examples() -> None:
    """show practical code snippets you'd actually use"""
    print_section("5. Practical Examples")

    print("\n📝 EXAMPLE 1: Simple RAG Query")
    print(cleandoc('''
        # what you'd write with LangChain:
        result = qa_chain.invoke({"query": "What are embeddings?"})
        print(result["result"])

        # what it replaces from Phase 3:
        # 1. embed query
        # 2. search vector db
        # 3. assemble context
        # 4. build prompt
        # 5. call LLM
        # 6. extract answer

        Trade-off: 1 line vs 30 lines, but less control
    '''))

    print("\n📝 EXAMPLE 2: Streaming Responses")
    print(cleandoc('''
        # LangChain makes streaming easy:
        for chunk in qa_chain.stream({"query": "Explain RAG"}):
            print(chunk, end="", flush=True)

        # Phase 3 equivalent requires:
        # - manual OpenAI streaming setup
        # - handling deltas
        # - accumulating full response

        LangChain wins here: streaming abstracted well
    '''))

    print("\n📝 EXAMPLE 3: Hybrid Approach")
    print(cleandoc('''
        # use LangChain for setup:
        vectorstore = Chroma.from_documents(...)
        retriever = vectorstore.as_retriever()

        # but use Phase 3 for custom retrieval:
        def custom_retrieve(query: str) -> list:
            # your Phase 3 smart retrieval logic
            results = retriever.invoke(query)

            # apply custom filters
            filtered = [r for r in results if meets_criteria(r)]

            # re-rank with cross-encoder
            reranked = rerank_with_model(query, filtered)

            return reranked[:3]

        # best of both worlds!
    '''))
# endregion


# region 5. Decision Framework
def demo_decision_framework() -> None:
    """help decide when to use LangChain"""
    print_section("6. Decision Framework: When to Use What")

    print("\n✅ USE PHASE 3 (Raw API) when:")
    print("   - Simple RAG (1 collection, basic retrieval)")
    print("   - Need exact control (custom chunking, retrieval, context)")
    print("   - Performance critical (minimize framework overhead)")
    print("   - Token budgets and cost tracking (your Phase 5 patterns)")
    print("   - Learning how RAG works (you've done this!)")

    print("\n✅ USE LANGCHAIN when:")
    print("   - Complex RAG (multiple collections, hybrid search)")
    print("   - Need pre-built components (document loaders, splitters)")
    print("   - Want LangSmith monitoring and debugging")
    print("   - Multi-provider support (OpenAI ↔ Anthropic switching)")
    print("   - Team standardization (consistent patterns)")

    print("\n🔀 USE HYBRID when:")
    print("   - LangChain setup + Phase 3 retrieval logic")
    print("   - LangChain RAG + Phase 5 optimization (semantic cache, batching)")
    print("   - Framework for 80%, custom code for 20%")
    print("   - This is most common in production!")

    print("\n📊 COMPLEXITY DECISION:")
    print(cleandoc('''
        Project Complexity → Best Approach

        • Single doc collection, basic search      → Phase 3 (simpler)
        • Multiple collections, metadata filters   → LangChain (abstracts complexity)
        • Custom chunking + retrieval logic        → Phase 3 (full control)
        • Standard RAG + monitoring               → LangChain (built-in tools)
        • Performance critical, cost sensitive     → Phase 3 + Phase 5 (optimized)
        • Multi-agent with tools                   → LangGraph (Module 2)
    '''))
# endregion


# region 6. Real-World Recommendations
def demo_recommendations() -> None:
    """provide real-world advice"""
    print_section("7. Real-World Recommendations")

    print("\n🎯 FOR YOUR PROJECTS:")

    print("\n1️⃣  PORTFOLIO PROJECT:")
    print("   - Stick with Phase 3 code (shows you understand fundamentals)")
    print("   - Demonstrates: chunking, embeddings, retrieval, generation")
    print("   - Add: Phase 5 optimizations (semantic cache, cost tracking)")
    print("   - Impress: \"I built this from scratch to understand RAG\"")

    print("\n2️⃣  HACKATHON / QUICK PROTOTYPE:")
    print("   - Use LangChain (speed matters)")
    print("   - Setup in minutes vs hours")
    print("   - Pre-built document loaders save time")
    print("   - Focus on unique features, not infrastructure")

    print("\n3️⃣  PRODUCTION APP:")
    print("   - Start with LangChain (80% standard patterns)")
    print("   - Custom code for critical paths (20%)")
    print("   - Example: LangChain RAG + your semantic cache")
    print("   - Monitor with LangSmith, optimize hot paths")

    print("\n4️⃣  LEARNING:")
    print("   - Phase 3 first (understand fundamentals) ✅ You did this!")
    print("   - Then LangChain (see what it abstracts) ← You are here")
    print("   - Result: Make informed decisions, not blind framework use")

    print("\n💡 KEY INSIGHT:")
    print("   You built Phase 3 → understand what LangChain does")
    print("   Most developers only know framework → can't debug when it breaks")
    print("   You have competitive advantage: fundamentals + framework")
# endregion


# region Main


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Prompts", "templates and few-shot examples", demo_prompts),
    Demo("2", "LLM Wrappers", "unified LLM interface", demo_llm_wrappers),
    Demo("3", "Chains", "sequential operations and LCEL", demo_chains),
    Demo("4", "Memory", "conversation context management", demo_memory),
    Demo("5", "RAG Basics", "retrieval-augmented generation", demo_rag_basics),
    Demo("6", "Agents", "tool-using autonomous agents", demo_agents),
    Demo("7", "All Concepts", "run all demos in sequence", demo_all_concepts),
]

# endregion

def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="LangChain Concepts Demo")
    runner.run()
if __name__ == "__main__":
    main()


# endregion
