# Module 1: LangChain Basics

**Purpose**: Learn LangChain fundamentals by seeing how it simplifies your Phase 3/4 implementations

---

## Learning Objectives

By the end of this module, you will:
- Understand LangChain's core abstractions (chains, prompts, memory)
- Convert your Phase 3 code to LangChain equivalents
- Know when LangChain helps vs adds overhead
- Use LangChain for real-world agent tasks

---

## Topics Covered

### 1. Prompts & Templates
**What you built**: String formatting with f-strings
**LangChain**: `PromptTemplate`, `ChatPromptTemplate`

```python
# Your way (Phase 2)
prompt = f"You are a {role}. {task}"

# LangChain way
from langchain.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("user", "{task}")
])
prompt = template.format_messages(role="assistant", task="help me")
```

### 2. LLM Integration
**What you built**: Raw OpenAI/Anthropic API calls
**LangChain**: `ChatOpenAI`, `ChatAnthropic` with unified interface

```python
# Your way (Phase 2)
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[...])

# LangChain way
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("Your prompt here")
```

### 3. Chains
**What you built**: Manual function composition
**LangChain**: `LLMChain`, `SequentialChain`, `LCEL`

```python
# Your way
def pipeline(text):
    summary = summarize(text)
    sentiment = analyze_sentiment(summary)
    return sentiment

# LangChain way (LCEL)
from langchain_core.runnables import RunnablePassthrough
chain = summarize_chain | sentiment_chain
result = chain.invoke({"text": text})
```

### 4. Memory
**What you built**: `ChatMemory` class (Phase 3)
**LangChain**: `RunnableWithMessageHistory` (modern LCEL approach)

```python
# Your way (Phase 3)
memory = ChatMemory(strategy="sliding_window", max_messages=10)
memory.add_message("user", "Hello")

# LangChain way (modern LangChain 1.0+ with LCEL)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# simple chain
chain = prompt | llm

# add message history
store = {}  # session_id -> chat history

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# use with session ID
chain_with_history.invoke({"input": "Hello"}, config={"configurable": {"session_id": "user123"}})
```

> **Note**: For legacy `ConversationBufferMemory` pattern, see [04_memory](04_memory/README.md#legacy-patterns--deprecated)

### 5. RAG
**What you built**: Full RAG pipeline (Phase 3)
**LangChain**: LCEL-based RAG chains (modern approach)

```python
# Your way (Phase 3)
chunks = chunker.chunk(document)
embeddings = embedder.embed(chunks)
db.add(chunks, embeddings)
results = db.search(query)
context = assemble_context(results)
response = llm.generate(prompt + context)

# LangChain way (modern LangChain 1.0+ with LCEL)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# setup vectorstore
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# create RAG chain with LCEL
template = ChatPromptTemplate.from_template("""
Answer based on context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | llm
)

response = rag_chain.invoke("your question")
```

> **Note**: For legacy `RetrievalQA` pattern, see [05_rag](05_rag/README.md#legacy-patterns--deprecated)

### 6. Agents & Tools
**What you built**: `ReActAgent`, `ToolRegistry` (Phase 4)
**LangChain**: `create_agent` (LangGraph), `@tool` decorator

```python
# Your way (Phase 4)
class WebSearchTool(BaseTool):
    def execute(self, query: str) -> ToolResult:
        results = search_api(query)
        return ToolResult.ok(results)

registry = ToolRegistry()
registry.register(WebSearchTool())
agent = ReActAgent(registry=registry)

# LangChain way (modern LangChain 1.0+)
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

@tool
def web_search(query: str) -> str:
    """search the web for information"""
    return search_api(query)

# create_agent returns executable CompiledStateGraph
agent = create_agent(model=llm, tools=[web_search])

# execute with message-based API
result = agent.invoke({"messages": [HumanMessage(content="search for LangChain")]})
```

> **Note**: For legacy `create_react_agent` + `AgentExecutor` pattern, see [06_agents_tools](06_agents_tools/README.md#legacy-patterns--deprecated)

---

## Module Structure

```
01_langchain_basics/
├── README.md                    # This file
├── migration_examples.py        # ✅ Side-by-side: Your code → LangChain (6 comparisons)
├── langchain_concepts_demo.py   # ✅ Conceptual overview with comparisons
├── langchain_rag_chatbot.py     # ✅ Step-by-step RAG chatbot walkthrough
├── 01_prompts/                  # ✅ PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
│   ├── README.md
│   ├── concepts.py              # Conceptual (no API key)
│   └── practical.py             # Hands-on (requires API key)
├── 02_llm_integration/          # ✅ ChatOpenAI, ChatAnthropic, unified interface
│   ├── README.md
│   ├── concepts.py              # Conceptual (no API key)
│   └── practical.py             # Hands-on (requires API keys)
├── 03_chains/                   # ✅ LLMChain, SequentialChain, LCEL syntax
│   ├── README.md
│   ├── concepts.py              # Conceptual (no API key)
│   └── practical.py             # Hands-on (requires API keys)
├── 04_memory/                   # ✅ ConversationBufferMemory, ConversationSummaryMemory
│   ├── README.md
│   ├── concepts.py              # Conceptual (no API key)
│   └── practical.py             # Hands-on (requires API key)
├── 05_rag/                      # ✅ RetrievalQA, vector stores, document loaders
│   ├── README.md
│   ├── concepts.py              # Conceptual (no API key)
│   └── practical.py             # Hands-on (requires API key)
└── 06_agents_tools/             # ✅ create_react_agent, @tool, AgentExecutor
    ├── README.md
    ├── concepts.py              # Conceptual (no API key)
    └── practical.py             # Hands-on (requires API key)
```

**Current Status**: All 6 modules complete! (Prompts, LLM integration, chains, memory, RAG, agents & tools)

---

## Key Concepts

### LangChain Abstraction Layers

```
┌─────────────────────────────────────────────────────┐
│         Your Application Logic                      │
├─────────────────────────────────────────────────────┤
│  Chains/Agents (High-level orchestration)           │
│    - RetrievalQA, ConversationalRetrievalChain      │
│    - create_react_agent, AgentExecutor              │
├─────────────────────────────────────────────────────┤
│  Components (Mid-level building blocks)             │
│    - Prompts, Memory, Callbacks                     │
│    - Vector Stores, Document Loaders                │
├─────────────────────────────────────────────────────┤
│  LLMs (Low-level provider integration)              │
│    - ChatOpenAI, ChatAnthropic, HuggingFace         │
├─────────────────────────────────────────────────────┤
│         Provider APIs (OpenAI, Anthropic, etc.)     │
└─────────────────────────────────────────────────────┘
```

### LCEL (LangChain Expression Language)

**Philosophy**: Chain components with `|` operator

```python
# Traditional
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input)

# LCEL (modern LangChain)
chain = prompt | llm | output_parser
result = chain.invoke(input)
```

**Benefits**:
- More readable (Unix pipe style)
- Streaming support built-in
- Async by default
- Better error handling

---

## Prerequisites

Before starting this module:
- ✅ Completed Phase 2 (LLM fundamentals)
- ✅ Completed Phase 3 (RAG, chat, function calling)
- ✅ Completed Phase 4 (agents)

---

## Installation

✅ **Already Installed!** All LangChain packages are now available:

```bash
# Installed packages:
# - langchain==1.0.5
# - langchain-openai==1.0.2
# - langchain-anthropic==1.0.3
# - langchain-chroma==1.0.0
# - langchain-community==0.4.1
# - langchain-core==1.0.4
```

To install in a new environment:
```bash
uv add langchain langchain-openai langchain-anthropic langchain-chroma langchain-community
```

---

## Running Examples

```bash
# ✅ Conceptual Foundation (no API key needed):
uv run python -m phase7_frameworks.01_langchain_basics.migration_examples
uv run python -m phase7_frameworks.01_langchain_basics.langchain_concepts_demo
uv run python -m phase7_frameworks.01_langchain_basics.langchain_rag_chatbot

# ✅ Prompts Module:
uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.concepts
uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.practical

# ✅ LLM Integration Module:
uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.concepts
uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.practical

# ✅ Chains Module:
uv run python -m phase7_frameworks.01_langchain_basics.03_chains.concepts
uv run python -m phase7_frameworks.01_langchain_basics.03_chains.practical

# ✅ Memory Module:
uv run python -m phase7_frameworks.01_langchain_basics.04_memory.concepts
uv run python -m phase7_frameworks.01_langchain_basics.04_memory.practical

# ✅ RAG Module:
uv run python -m phase7_frameworks.01_langchain_basics.05_rag.concepts
uv run python -m phase7_frameworks.01_langchain_basics.05_rag.practical

# ✅ Agents & Tools Module:
uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.concepts
uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.practical
```

**Organization**: Each module has:
- `concepts.py` - Learn patterns without API key
- `practical.py` - Practice with real LLM calls (requires `OPENAI_API_KEY` in `.env`)

**✨ New Feature - Visual Documentation**: All 49 demos across 6 modules now include comprehensive ASCII diagrams showing:
- 📊 Architecture and workflow visualization
- → Step-by-step data flow with arrows
- ✅ Benefits highlighted for each pattern
- ⚠️  Important limitations and cautions
- 💡 Implementation details and code patterns
- 🎯 Real-world use cases

Example from Memory module:
```
Buffer Memory Pattern:
┌─────────────────────────────────────────────────────────────┐
│       Buffer Memory: Full Conversation History Storage      │
│                                                             │
│  Turn 1:                                                    │
│     User: "Hi, I'm learning about LangChain memory"         │
│           ▼                                                 │
│     Memory: [] (empty) → Store message                      │
│           ▼                                                 │
│     LLM: "Great! LangChain memory helps..."                 │
│                                                             │
│  ✅ Benefit: Perfect recall (all context retained)          │
│  ⚠️  Caution: Unbounded token growth over time              │
└─────────────────────────────────────────────────────────────┘
```

---

## Exercises

### Exercise 1: Prompt Templates
Convert your Phase 2 prompt engineering patterns to LangChain templates.

### Exercise 2: RAG Migration
Refactor your Phase 3 RAG pipeline using `RetrievalQA`.

### Exercise 3: Agent with Tools
Rebuild your Phase 4 `ReActAgent` using `create_react_agent`.

### Exercise 4: Memory Strategies
Compare `ConversationBufferMemory` vs your custom `ChatMemory`.

Solutions in `solutions/` directory.

---

## When to Use LangChain

### ✅ Good Use Cases
- Building agents with multiple tools
- RAG with standard patterns
- Need provider flexibility (OpenAI ↔ Anthropic)
- Team standardization
- Want LangSmith monitoring

### ❌ Skip LangChain
- Simple single LLM call
- Maximum performance critical
- Custom logic doesn't fit patterns
- Framework overhead too high

---

## Common Pitfalls

### 1. Over-Abstraction
```python
# Bad: Using framework for simple task
chain = prompt | llm | output_parser
result = chain.invoke({"input": "Hello"})

# Good: Direct API call
response = llm.invoke("Hello")
```

### 2. Version Lock-in
```python
# Bad: Unversioned imports
from langchain.chains import RetrievalQA  # might break

# Good: Explicit versions in requirements.txt
langchain==0.1.0
langchain-openai==0.0.5
```

### 3. Hidden Costs
```python
# Watch out: LangChain can make many API calls
chain = RetrievalQA.from_chain_type(llm, retriever)
result = chain.invoke(query)  # How many LLM calls? Check docs!
```

---

## Next Steps

After completing this module:
1. Build a small RAG chatbot with LangChain
2. Compare implementation complexity vs your Phase 3 code
3. Move to Module 2 (LangGraph) for multi-agent workflows
4. Decide which patterns to adopt in production

---

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)
- [Agent Types](https://python.langchain.com/docs/modules/agents/agent_types/)
- [LangSmith Monitoring](https://docs.smith.langchain.com/)
