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
**LangChain**: `ConversationBufferMemory`, `ConversationSummaryMemory`

```python
# Your way (Phase 3)
memory = ChatMemory(strategy="sliding_window", max_messages=10)
memory.add_message("user", "Hello")

# LangChain way
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=10)
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
```

### 5. RAG
**What you built**: Full RAG pipeline (Phase 3)
**LangChain**: `RetrievalQA`, `ConversationalRetrievalChain`

```python
# Your way (Phase 3)
chunks = chunker.chunk(document)
embeddings = embedder.embed(chunks)
db.add(chunks, embeddings)
results = db.search(query)
context = assemble_context(results)
response = llm.generate(prompt + context)

# LangChain way
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
response = qa.invoke({"query": query})
```

### 6. Agents & Tools
**What you built**: `ReActAgent`, `ToolRegistry` (Phase 4)
**LangChain**: `create_react_agent`, `@tool` decorator

```python
# Your way (Phase 4)
class WebSearchTool(BaseTool):
    def execute(self, query: str) -> ToolResult:
        results = search_api(query)
        return ToolResult.ok(results)

registry = ToolRegistry()
registry.register(WebSearchTool())
agent = ReActAgent(registry=registry)

# LangChain way
from langchain.agents import create_react_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
tools = [Tool(name="search", func=search.run, description="Search the web")]
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
```

---

## Module Structure

```
01_langchain_basics/
├── README.md                    # This file
├── prompts_templates.py         # PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
├── llm_integration.py           # ChatOpenAI, ChatAnthropic, unified interface
├── chains.py                    # LLMChain, SequentialChain, LCEL syntax
├── memory.py                    # ConversationBufferMemory, ConversationSummaryMemory
├── rag.py                       # RetrievalQA, vector stores, document loaders
├── agents_tools.py              # create_react_agent, @tool, AgentExecutor
└── migration_examples.py        # Side-by-side: Your code → LangChain
```

---

## Key Concepts

### LangChain Abstraction Layers

```
┌─────────────────────────────────────────────────────┐
│         Your Application Logic                      │
├─────────────────────────────────────────────────────┤
│  Chains/Agents (High-level orchestration)          │
│    - RetrievalQA, ConversationalRetrievalChain     │
│    - create_react_agent, AgentExecutor             │
├─────────────────────────────────────────────────────┤
│  Components (Mid-level building blocks)            │
│    - Prompts, Memory, Callbacks                    │
│    - Vector Stores, Document Loaders               │
├─────────────────────────────────────────────────────┤
│  LLMs (Low-level provider integration)             │
│    - ChatOpenAI, ChatAnthropic, HuggingFace        │
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

```bash
# Core LangChain
pip install langchain langchain-core langchain-community

# Specific integrations
pip install langchain-openai      # OpenAI models
pip install langchain-anthropic   # Claude models
pip install chromadb              # Vector store
```

Or with uv:
```bash
uv pip install langchain langchain-openai langchain-anthropic chromadb
```

---

## Running Examples

```bash
# Individual examples
uv run python -m phase7_frameworks.01_langchain_basics.prompts_templates
uv run python -m phase7_frameworks.01_langchain_basics.llm_integration
uv run python -m phase7_frameworks.01_langchain_basics.chains
uv run python -m phase7_frameworks.01_langchain_basics.memory
uv run python -m phase7_frameworks.01_langchain_basics.rag
uv run python -m phase7_frameworks.01_langchain_basics.agents_tools

# Side-by-side comparison
uv run python -m phase7_frameworks.01_langchain_basics.migration_examples
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
