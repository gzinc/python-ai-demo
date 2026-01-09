# Module 4: Framework Comparison & Decision Framework

**Purpose**: Learn when to use which framework (or none) based on real-world scenarios

---

## Learning Objectives

- Build framework decision trees
- Understand trade-offs between frameworks
- Know when to skip frameworks entirely
- Apply decision framework to real projects
- Plan migration strategies

---

## The Decision Framework

```
START: New LLM Project
│
├─ Is it primarily RAG/document-heavy?
│  ├─ Yes → Consider LlamaIndex
│  └─ No → Continue
│
├─ Need multi-agent collaboration?
│  ├─ Yes → Consider LangGraph
│  └─ No → Continue
│
├─ Need agents with tools?
│  ├─ Yes → Consider LangChain
│  └─ No → Continue
│
├─ Is it a simple LLM call/chain?
│  ├─ Yes → Skip frameworks, use raw API
│  └─ No → Consider LangChain for orchestration
│
└─ Hybrid needs?
   └─ Mix frameworks or custom solution
```

---

## Framework Comparison Matrix

### Core Strengths

| Framework | Best For | Avoid For |
|-----------|----------|-----------|
| **LangChain** | Agents, tools, orchestration | Simple tasks, pure RAG |
| **LangGraph** | Multi-agent, workflows, human-in-loop | Linear chains, simple agents |
| **LlamaIndex** | RAG, documents, knowledge retrieval | Agent-heavy apps, tool use |
| **None (Raw)** | Simple calls, max control, performance | Complex orchestration |

### Technical Comparison

| Feature | LangChain | LangGraph | LlamaIndex | Raw API |
|---------|-----------|-----------|------------|---------|
| **Learning Curve** | Medium | High | Medium | Low |
| **Performance** | Medium | Medium | Medium | High |
| **Flexibility** | High | High | Medium | Maximum |
| **Debugging** | Medium | Hard | Medium | Easy |
| **Abstraction** | High | Medium | High | None |
| **Ecosystem** | Largest | Growing | RAG-focused | N/A |

---

## Real-World Scenarios

### Scenario 1: Documentation Chatbot
**Requirements**:
- Answer questions from company docs
- Support follow-up questions
- Filter by document type

**Recommendation**: **LlamaIndex**

**Why**:
- Primary use case is RAG
- Advanced retrieval (hybrid search, metadata filters)
- Chat engine with memory built-in
- Document loaders for various formats

**Implementation**:
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine()
response = chat_engine.chat("What is X?")
```

---

### Scenario 2: Customer Support Agent
**Requirements**:
- Access CRM, knowledge base, email
- Handle different request types
- Escalate to human when needed

**Recommendation**: **LangChain + LangGraph**

**Why**:
- Multi-tool integration (LangChain)
- Conditional routing (LangGraph)
- Human-in-loop for escalation (LangGraph)

**Implementation**:
```python
from langchain.agents import create_react_agent
from langgraph.graph import StateGraph

# LangChain for tools
tools = [CRMTool(), KnowledgeBaseTool(), EmailTool()]
agent = create_react_agent(llm, tools)

# LangGraph for routing + escalation
graph = StateGraph(SupportState)
graph.add_node("agent", agent)
graph.add_node("human", human_escalation)
graph.add_conditional_edges("agent", should_escalate)
```

---

### Scenario 3: Code Analysis Tool
**Requirements**:
- Parse code files
- Analyze patterns
- Generate reports
- Custom business logic

**Recommendation**: **Custom (Your Phase 3-5 code)**

**Why**:
- Need custom AST parsing
- Frameworks don't add value here
- Performance matters (batch processing)
- Maximum control required

**Implementation**:
```python
# Use your fundamentals
from phase4_ai_agents import ReActAgent
from phase3_llm_applications import RAGPipeline

# Custom logic
ast_tree = parse_code(file)
patterns = analyze_patterns(ast_tree)

# LLM for insights only
insights = llm.generate(f"Analyze: {patterns}")
```

---

### Scenario 4: Research Assistant
**Requirements**:
- Web search → summarize → analyze → write report
- Multiple specialist agents
- Complex workflow

**Recommendation**: **LangGraph**

**Why**:
- Multi-agent collaboration
- Sequential + parallel steps
- Visualization helps debugging
- Stateful workflow

**Implementation**:
```python
from langgraph.graph import StateGraph

graph = StateGraph(ResearchState)
graph.add_node("researcher", research_agent)
graph.add_node("analyzer", analysis_agent)
graph.add_node("writer", writer_agent)
graph.add_edge("researcher", "analyzer")
graph.add_edge("analyzer", "writer")
```

---

### Scenario 5: Simple Prompt Chain
**Requirements**:
- Summarize → translate → format
- Fixed pipeline

**Recommendation**: **LangChain LCEL or Raw**

**Why**:
- Simple linear chain
- LCEL readable and concise
- Or just 3 API calls if performance matters

**Implementation**:
```python
# LangChain LCEL
from langchain_core.runnables import RunnablePassthrough
chain = summarize | translate | format
result = chain.invoke(text)

# Or raw (faster)
summary = llm("Summarize: {text}")
translation = llm(f"Translate to Spanish: {summary}")
formatted = format_output(translation)
```

---

## Migration Strategies

### From Raw → Framework

**When to migrate**:
- Codebase growing complex
- Team needs standardization
- Want monitoring (LangSmith)
- Repetitive boilerplate

**How**:
1. Start with highest-value module (RAG → LlamaIndex)
2. Migrate incrementally (not all at once)
3. Keep raw API for custom logic
4. Compare performance before/after

---

### From Framework → Raw

**When to migrate**:
- Framework overhead too high
- Need maximum performance
- Custom logic doesn't fit patterns
- Debugging too difficult

**How**:
1. Identify framework bottlenecks
2. Rewrite critical path with raw API
3. Keep framework for boilerplate
4. Measure performance gains

---

### Hybrid Approach (Common in Production)

```python
# Use framework for standard stuff
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI()
memory = ConversationBufferMemory()

# Drop to raw API for custom logic
from openai import OpenAI
client = OpenAI()

def custom_workflow():
    # Your Phase 3-5 code here
    pass
```

---

## Framework Decision Tree

```
┌─────────────────────────────────────────┐
│  Is performance critical (< 100ms)?     │
│  ├─ Yes → Skip frameworks (raw API)     │
│  └─ No → Continue                       │
└─────────────────────────────────────────┘
               │
┌─────────────────────────────────────────┐
│  Primary use case?                      │
│  ├─ RAG/Documents → LlamaIndex          │
│  ├─ Multi-agent → LangGraph             │
│  ├─ Agents + Tools → LangChain          │
│  └─ Simple chains → LangChain or raw    │
└─────────────────────────────────────────┘
               │
┌─────────────────────────────────────────┐
│  Need monitoring/observability?         │
│  ├─ Yes → Framework (LangSmith, etc.)   │
│  └─ No → Raw API is fine               │
└─────────────────────────────────────────┘
```

---

## Cost-Benefit Analysis

### Using Frameworks

**Benefits**:
- ✅ Faster development (boilerplate handled)
- ✅ Team standardization
- ✅ Monitoring built-in (LangSmith)
- ✅ Best practices enforced

**Costs**:
- ❌ Performance overhead
- ❌ Learning curve
- ❌ Debugging complexity
- ❌ Version lock-in
- ❌ Abstraction leaks

### Skipping Frameworks

**Benefits**:
- ✅ Maximum performance
- ✅ Full control
- ✅ Easy debugging
- ✅ No version dependencies

**Costs**:
- ❌ More code to write
- ❌ Reinvent patterns
- ❌ Manual monitoring
- ❌ Team inconsistency

---

## Module Structure

```
04_framework_comparison/
├── README.md                    # This file
├── decision_tree.py             # Interactive decision helper
├── scenario_examples.py         # 10 real-world scenarios analyzed
├── migration_guide.py           # Step-by-step migration examples
├── performance_comparison.py    # Benchmark framework vs raw
└── hybrid_patterns.py           # Mix frameworks with raw API
```

---

## Exercises

### Exercise 1: Analyze Your Project
Run through decision tree for your own project.

### Exercise 2: Performance Benchmark
Compare LangChain vs raw API for same task.

### Exercise 3: Migration Plan
Create migration strategy for a complex project.

### Exercise 4: Hybrid Implementation
Build app using multiple frameworks + raw API.

---

## Key Takeaways

1. **Start simple**: Raw API until you need framework
2. **Framework for boilerplate**: Use where it adds clear value
3. **Raw for custom**: Don't force custom logic into framework patterns
4. **Measure first**: Benchmark before committing to framework
5. **Hybrid is OK**: Mix frameworks with raw API is common in production

---

## Next Steps

After Phase 7:
1. Choose framework(s) for your capstone project
2. Build production app combining Phases 2-7 learnings
3. Contribute to open source frameworks
4. Explore Phase 6 (fine-tuning) if interested

---

## Resources

- [LangChain vs LlamaIndex Comparison](https://python.langchain.com/docs/langchain_vs_llamaindex)
- [When NOT to use LangChain](https://www.reddit.com/r/LangChain/comments/13fcw36/when_not_to_use_langchain/)
- [Framework Benchmarks](https://github.com/run-llama/llama_index/tree/main/benchmarks)
