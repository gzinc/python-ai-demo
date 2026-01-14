"""
Module: LangChain Migration Examples - Side-by-Side Comparisons

Compares your Phase 3/4 implementations with LangChain equivalents.
Shows what frameworks abstract and when they add value.

Run with: uv run python -m phase7_frameworks.01_langchain_basics.migration_examples
"""

import os
from inspect import cleandoc

# suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# region Helper Functions
def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def print_comparison(your_way: str, langchain_way: str) -> None:
    """print side-by-side comparison"""
    print("\n┌─ YOUR WAY (Phases 3/4) ─────────────────────────────────────────────┐")
    print(your_way)
    print("└──────────────────────────────────────────────────────────────────────┘")
    print("\n┌─ LANGCHAIN WAY ──────────────────────────────────────────────────────┐")
    print(langchain_way)
    print("└──────────────────────────────────────────────────────────────────────┘")
# endregion


# region 1. Prompts & Templates


def example_prompts() -> None:
    """compare prompt template approaches"""
    print_section("1. Prompts & Templates")

    your_way = cleandoc('''
        # phase2: string formatting with f-strings
        role = "helpful assistant"
        task = "explain embeddings"
        prompt = f"You are a {role}. {task}"

        # simple, direct, no dependencies
        messages = [
            {"role": "system", "content": f"You are a {role}"},
            {"role": "user", "content": task}
        ]
    ''')

    langchain_way = cleandoc('''
        # LangChain: ChatPromptTemplate
        from langchain.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}"),
            ("user", "{task}")
        ])

        # format with variables
        messages = template.format_messages(
            role="helpful assistant",
            task="explain embeddings"
        )

        # benefits: validation, reusability, few-shot examples
        # cost: extra abstraction layer
    ''')

    print_comparison(your_way, langchain_way)

    print("\n💡 INSIGHT:")
    print("   - Simple prompts: Your way is cleaner (less overhead)")
    print("   - Complex prompts: LangChain helps (few-shot, validation)")
    print("   - Team standardization: LangChain provides consistency")
# endregion


# region 2. LLM Integration


def example_llm_integration() -> None:
    """compare LLM API integration approaches"""
    print_section("2. LLM Integration")

    your_way = cleandoc('''
        # phase2: raw API calls with full control
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100
        )
        result = response.choices[0].message.content

        # explicit, predictable, full control
        # you know exactly what API call is made
    ''')

    langchain_way = cleandoc('''
        # LangChain: unified interface across providers
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100
        )
        result = llm.invoke("Hello")

        # benefits:
        # - switch providers easily (OpenAI ↔ Anthropic)
        # - streaming, async, batch built-in
        # - callbacks for logging/monitoring

        # cost:
        # - abstraction hides details
        # - harder to debug API issues
    ''')

    print_comparison(your_way, langchain_way)

    print("\n💡 INSIGHT:")
    print("   - Single provider: Your way is simpler")
    print("   - Multi-provider: LangChain abstracts differences")
    print("   - Debugging: Raw API easier to troubleshoot")
# endregion


# region 3. Chains & Pipelines


def example_chains() -> None:
    """compare chain/pipeline approaches"""
    print_section("3. Chains & Pipelines")

    your_way = cleandoc('''
        # phase3/phase4: manual function composition
        def summarize_then_analyze(text: str) -> dict[str, Any]:
            # step 1: summarize
            summary_prompt = f"Summarize: {text}"
            summary = llm_call(summary_prompt)

            # step 2: analyze sentiment
            sentiment_prompt = f"Analyze sentiment: {summary}"
            sentiment = llm_call(sentiment_prompt)

            return {"summary": summary, "sentiment": sentiment}

        # explicit control flow, easy to debug
        # clear what happens at each step
    ''')

    langchain_way = cleandoc('''
        # LangChain: LCEL (LangChain Expression Language)
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

        # compose with | operator
        chain = summarize | analyze
        result = chain.invoke({"text": "..."})

        # benefits: streaming, async, error handling built-in
        # cost: harder to debug, less explicit control
    ''')

    print_comparison(your_way, langchain_way)

    print("\n💡 INSIGHT:")
    print("   - Simple workflows: Your way is more readable")
    print("   - Complex pipelines: LangChain handles edge cases")
    print("   - Streaming: LangChain has built-in support")
# endregion


# region 4. Conversation Memory


def example_memory() -> None:
    """compare conversation memory approaches"""
    print_section("4. Conversation Memory")

    your_way = cleandoc('''
        # phase3: custom ChatMemory class
        class ChatMemory:
            def __init__(self, strategy: str, max_messages: int = 10):
                self.strategy = strategy
                self.max_messages = max_messages
                self.messages: list[dict] = []

            def add_message(self, role: str, content: str) -> None:
                self.messages.append({"role": role, "content": content})
                if self.strategy == "sliding_window":
                    self.messages = self.messages[-self.max_messages:]

            def get_messages(self) -> list[dict]:
                return self.messages

        # explicit, customizable, easy to extend
        memory = ChatMemory(strategy="sliding_window", max_messages=10)
    ''')

    langchain_way = cleandoc('''
        # LangChain: ConversationBufferWindowMemory
        from langchain.memory import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(k=10)
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )

        # also available:
        # - ConversationSummaryMemory (LLM summarizes old messages)
        # - ConversationBufferMemory (unlimited)
        # - VectorStoreBackedMemory (semantic retrieval)

        messages = memory.load_memory_variables({})

        # benefits: pre-built strategies, tested patterns
        # cost: less control over custom logic
    ''')

    print_comparison(your_way, langchain_way)

    print("\n💡 INSIGHT:")
    print("   - Custom logic: Your way gives full control")
    print("   - Standard patterns: LangChain saves time")
    print("   - Token budgets: Your implementation handles this")
# endregion


# region 5. RAG System


def example_rag() -> None:
    """compare RAG pipeline approaches"""
    print_section("5. RAG System")

    your_way = cleandoc('''
        # phase3: full RAG pipeline with explicit steps
        from dataclasses import dataclass

        @dataclass
        class Document:
            content: str
            metadata: dict[str, Any]

        # 1. chunk documents
        chunks = chunker.chunk_paragraph(document.content)

        # 2. generate embeddings
        embeddings = embedder.embed_batch(chunks)

        # 3. store in vector db
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            db.add(chunk, embedding, document.metadata)

        # 4. query
        query_embedding = embedder.embed(query)
        results = db.search(query_embedding, k=3)

        # 5. build context
        context = "\\n\\n".join([r.content for r in results])

        # 6. generate answer
        prompt = f"Context: {context}\\n\\nQuestion: {query}"
        answer = llm.generate(prompt)

        # explicit control at every step
        # easy to customize chunking, retrieval, generation
    ''')

    langchain_way = cleandoc('''
        # LangChain: RetrievalQA chain
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # 1. load documents
        documents = [...]  # Document objects

        # 2. split (chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # 3. create vector store (embeds + stores automatically)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )

        # 4. create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

        # 5. query
        answer = qa.invoke({"query": query})

        # benefits: batteries included, standard patterns
        # cost: less control over intermediate steps
    ''')

    print_comparison(your_way, langchain_way)

    print("\n💡 INSIGHT:")
    print("   - Custom retrieval: Your way gives flexibility")
    print("   - Standard RAG: LangChain is faster to build")
    print("   - Debugging: Your way easier to trace issues")
    print("   - Production: Both need similar optimization (Phase 5)")
# endregion


# region 6. Agents & Tools


def example_agents() -> None:
    """compare agent implementation approaches"""
    print_section("6. Agents & Tools")

    your_way = cleandoc('''
        # phase4: custom ReActAgent with tool registry
        from enum import Enum

        class AgentState(Enum):
            THINKING = "thinking"
            ACTING = "acting"
            FINISHED = "finished"

        class ReActAgent:
            def __init__(self, tools: list[BaseTool], llm):
                self.tools = tools
                self.llm = llm
                self.state = AgentState.THINKING

            def run(self, task: str) -> str:
                iterations = 0
                while self.state != AgentState.FINISHED:
                    # THINK
                    thought = self.think(task)

                    # ACT
                    if "ACTION:" in thought:
                        result = self.execute_tool(thought)
                        self.state = AgentState.THINKING
                    else:
                        self.state = AgentState.FINISHED

                    iterations += 1
                    if iterations > 10:
                        break

                return self.final_answer

        # explicit loop, easy to debug, full control
    ''')

    langchain_way = cleandoc('''
        # LangChain: create_react_agent
        from langchain.agents import create_react_agent, AgentExecutor, Tool
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_openai import ChatOpenAI

        # define tools
        search = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="search",
                func=search.run,
                description="Search the web for current information"
            )
        ]

        # create agent
        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        # execute
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10
        )
        result = agent_executor.invoke({"input": "What's the weather?"})

        # benefits: pre-built patterns, tool ecosystem
        # cost: less visibility into agent loop
    ''')

    print_comparison(your_way, langchain_way)

    print("\n💡 INSIGHT:")
    print("   - Learning: Your way teaches fundamentals")
    print("   - Production: LangChain has more tools/integrations")
    print("   - Debugging: Your way easier to trace agent loop")
    print("   - Complex agents: LangGraph better (Module 2)")
# endregion


# region Key Takeaways


def example_takeaways() -> None:
    """summarize when to use what"""
    print_section("Key Takeaways: When to Use What")

    print("\n✅ USE YOUR WAY (Raw API) when:")
    print("   - Simple, single LLM calls")
    print("   - Maximum control and transparency needed")
    print("   - Performance critical (minimize overhead)")
    print("   - Custom logic doesn't fit framework patterns")
    print("   - Learning fundamentals (you've done this!)")

    print("\n✅ USE LANGCHAIN when:")
    print("   - Standard patterns (RAG, agents, chains)")
    print("   - Multi-provider support needed (OpenAI ↔ Anthropic)")
    print("   - Team standardization important")
    print("   - Want LangSmith monitoring/debugging")
    print("   - Need tool ecosystem (100+ pre-built tools)")

    print("\n🔀 HYBRID APPROACH (Common in Production):")
    print("   - Use LangChain for boilerplate (80%)")
    print("   - Drop to raw API for custom logic (20%)")
    print("   - Example: LangChain RAG + your Phase 5 optimization patterns")

    print("\n📊 COMPLEXITY THRESHOLD:")
    print(cleandoc('''
        Complexity
           ↑
           │                  ┌─ LangChain wins
           │                ┌─┘
           │              ┌─┘
           │            ┌─┘
           │          ┌─┘
           │        ┌─┘
           │      ┌─┘  ← Breakeven point (3-4 components)
           │    ┌─┘
           │  ┌─┘
           │┌─┘ ← Raw API wins
           └─────────────────────→ Time to Build
    '''))

    print("\n\n🎯 RECOMMENDATION:")
    print("   1. Start with raw API (you've done this in Phases 2-4)")
    print("   2. Learn LangChain patterns (this module)")
    print("   3. Use LangChain when it CLEARLY saves time")
    print("   4. Drop to raw API when you need control")
    print("   5. Never feel locked into framework!")
# endregion


# region Main
def main() -> None:
    """run all migration examples"""
    print("\n" + "=" * 80)
    print("  LANGCHAIN MIGRATION EXAMPLES")
    print("  Comparing Your Phase 3/4 Code vs LangChain")
    print("=" * 80)

    example_prompts()
    example_llm_integration()
    example_chains()
    example_memory()
    example_rag()
    example_agents()
    example_takeaways()

    print("\n" + "=" * 80)
    print("  NEXT STEPS")
    print("=" * 80)
    print(cleandoc('''
        1. Review these comparisons - understand abstraction costs/benefits
        2. Build small LangChain RAG chatbot to get hands-on experience
        3. Explore individual modules:
           - prompts_templates.py
           - llm_integration.py
           - chains.py
           - memory.py
           - rag.py
           - agents_tools.py
        4. Move to Module 2 (LangGraph) for multi-agent workflows
        5. Decide which patterns to adopt in your projects
    '''))

    print("\n💡 Remember: Frameworks are TOOLS, not REQUIREMENTS")
    print("   You have fundamentals to build without them!")


if __name__ == "__main__":
    main()
# endregion
