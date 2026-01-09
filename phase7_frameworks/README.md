# Phase 7: Production Frameworks

**Purpose**: Learn industry-standard frameworks after understanding fundamentals

**Why Phase 7 comes last**: You've built everything from scratch (Phases 2-5), so you now understand:
- What these frameworks do under the hood
- When to customize vs use defaults
- How to debug when things break
- Whether you actually need a framework

---

## Module Overview

| Module | Focus | Key Concepts |
|--------|-------|--------------|
| **01_langchain_basics** | Chains, prompts, memory, agents | LangChain foundation |
| **02_langgraph** | State machines, multi-agent workflows | Graph-based orchestration |
| **03_llamaindex** | RAG-focused alternative | Document-centric apps |
| **04_framework_comparison** | Decision framework | When to use what |

---

## What You Built vs Framework Equivalents

### Phase 3: LLM Applications

| Your Implementation | LangChain | LlamaIndex |
|---------------------|-----------|------------|
| RAG pipeline | `RetrievalQA` | `VectorStoreIndex.as_query_engine()` |
| Chat memory | `ConversationBufferMemory` | `ChatMemoryBuffer` |
| Function calling | `Tool` + `AgentExecutor` | `FunctionTool` |
| Streaming | `StreamingStdOutCallbackHandler` | `StreamingResponse` |

### Phase 4: AI Agents

| Your Implementation | LangChain | LangGraph |
|---------------------|-----------|-----------|
| ReAct agent | `create_react_agent()` | `create_react_agent()` |
| Tool registry | `@tool` decorator | `ToolNode` |
| Multi-agent | Manual orchestration | `StateGraph` + conditional edges |
| Agent loop | While loop | Graph execution |

### Phase 5: Production

| Your Implementation | LangChain | Built-in? |
|---------------------|-----------|-----------|
| Semantic cache | `GPTCache` integration | ❌ External |
| Cost tracking | `get_openai_callback()` | ✅ Yes |
| Rate limiting | - | ❌ External |
| Batching | `batch()` on chains | ✅ Yes |

---

## Framework Philosophy Comparison

### LangChain
**Philosophy**: General-purpose LLM orchestration with composable primitives

**Strengths**:
- Broad ecosystem (100+ integrations)
- Agent frameworks (ReAct, Plan-and-Execute, etc.)
- Memory systems (conversation, summary, vector, entity)
- Production monitoring (LangSmith)

**When to use**:
- Building agents with multiple tools
- Need flexibility across providers (OpenAI, Anthropic, local models)
- Want built-in memory/prompt management
- LangSmith observability

**Trade-offs**:
- Abstraction overhead (harder to debug)
- Breaking changes in updates
- Magic behavior can be surprising

---

### LangGraph
**Philosophy**: Build stateful multi-agent systems as graphs

**Strengths**:
- Explicit state management (no hidden magic)
- Visual debugging (graph visualization)
- Conditional branching (decision trees)
- Cyclical workflows (agent loops, human-in-the-loop)

**When to use**:
- Multi-agent collaboration (specialist agents)
- Complex workflows with branching logic
- Need human-in-the-loop approvals
- Want to visualize agent flow

**Trade-offs**:
- More verbose than simple chains
- Graph mental model required
- Overkill for simple tasks

---

### LlamaIndex
**Philosophy**: Data framework optimized for RAG

**Strengths**:
- Document-centric design (ingestion → indexing → querying)
- Advanced retrieval (hybrid search, auto-merging)
- Query engines (with filters, metadata)
- Storage abstractions (vector stores, document stores)

**When to use**:
- Primary use case is RAG
- Complex document structures (PDFs, tables, images)
- Need advanced retrieval strategies
- Document-heavy applications

**Trade-offs**:
- Less flexible for non-RAG tasks
- Smaller ecosystem than LangChain
- Agent support less mature

---

## Learning Path

### Prerequisites (You've Built These!)
✅ Phase 2: LLM fundamentals
✅ Phase 3: RAG, chat, function calling
✅ Phase 4: Agents, tools, multi-agent
✅ Phase 5: Production patterns

### Recommended Order
1. **01_langchain_basics** - Learn the framework that abstracts your Phase 3/4 work
2. **02_langgraph** - See how graphs improve your Phase 4 multi-agent system
3. **03_llamaindex** - Compare RAG-focused approach to your Phase 3 pipeline
4. **04_framework_comparison** - Decide when to use which (or none!)

---

## Key Insights

### When Frameworks Help
- **Standardization**: Team uses same patterns
- **Integrations**: 100+ data sources, vector stores, LLMs pre-integrated
- **Monitoring**: LangSmith, Phoenix built-in observability
- **Best practices**: Memory management, retry logic, error handling

### When to Skip Frameworks
- **Simple use cases**: Single LLM call with prompt template
- **Maximum control**: Custom logic doesn't fit framework patterns
- **Performance critical**: Framework overhead matters (e.g., high-throughput)
- **Learning**: Understanding fundamentals (you've done this!)

### Hybrid Approach (Common in Production)
```python
# Use framework for 80% of boilerplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Drop to raw API for custom 20%
from openai import OpenAI
client = OpenAI()  # your custom logic here
```

---

## Anti-Patterns to Avoid

### 🚫 Framework Lock-in
```python
# Bad: Everything through framework
langchain.do_everything()

# Good: Use framework where it helps, raw API where you need control
if simple_task:
    use_framework()
else:
    use_raw_api()
```

### 🚫 Over-Abstraction
```python
# Bad: 5 layers of framework abstractions for simple task
Chain(Router(Agent(Tool(LLM()))))

# Good: Direct API call for simple needs
client.chat.completions.create(...)
```

### 🚫 Version Hell
```python
# Bad: Update framework without testing
pip install langchain --upgrade  # breaks production

# Good: Pin versions, test upgrades
langchain==0.1.0  # explicit version in requirements
```

---

## Project Ideas

After completing Phase 7, you can build:

1. **Documentation Chatbot** (LlamaIndex)
   - Ingest company docs
   - Hybrid search with metadata filtering
   - Streaming chat interface

2. **Research Assistant** (LangGraph)
   - Multi-agent collaboration (researcher, analyst, writer)
   - Human-in-the-loop approval
   - Graph visualization of workflow

3. **Customer Support Agent** (LangChain)
   - Multi-tool integration (CRM, knowledge base, email)
   - Conversation memory with summarization
   - Cost tracking and monitoring

4. **Code Analysis Tool** (Custom + Framework Mix)
   - Custom AST parsing
   - LangChain for LLM orchestration
   - Your Phase 5 patterns for production

---

## Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)

### Tutorials
- [LangChain Academy](https://academy.langchain.com/)
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LlamaIndex Starter Tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)

### Monitoring & Observability
- [LangSmith](https://docs.smith.langchain.com/)
- [Phoenix (Arize)](https://docs.arize.com/phoenix)
- [LangFuse](https://langfuse.com/)

---

## Next Steps

1. Complete Module 1 (LangChain Basics)
2. Build a small project using LangChain
3. Learn Module 2 (LangGraph) and refactor project
4. Compare with Module 3 (LlamaIndex) approach
5. Make informed framework decision in Module 4

**Remember**: Frameworks are tools, not requirements. You have the fundamentals to build without them, which means you'll use them effectively when you choose to.
