# AI Development Taxonomy

A hierarchical overview of AI/LLM development concepts, from foundations to autonomous agents.

---

## The Four Levels

```
┌─────────────────────────────────────────────────────────────┐
│  AI DEVELOPMENT TAXONOMY                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: FOUNDATIONS                                       │
│  ├── Embeddings (text → vectors)                            │
│  ├── Vector databases (similarity search)                   │
│  └── LLM APIs (completion, chat)                            │
│                                                             │
│  Level 2: PATTERNS                                          │
│  ├── Prompting (few-shot, chain-of-thought)                 │
│  ├── RAG (retrieval + generation)                           │
│  └── Function/Tool calling (structured outputs)             │
│                                                             │
│  Level 3: APPLICATIONS                                      │
│  ├── Chatbots (conversation + memory)                       │
│  ├── Search systems (RAG + ranking)                         │
│  └── Pipelines (chained operations)                         │
│                                                             │
│  Level 4: AGENTS                                            │
│  ├── Single agent (loop + tools + reasoning)                │
│  ├── Multi-agent (orchestration + delegation)               │
│  └── Autonomous systems (planning + execution)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Level 1: Foundations

The building blocks everything else is built on.

### Embeddings
- **What**: Convert text to numerical vectors that capture meaning
- **Why**: Enable mathematical comparison of text similarity
- **Tools**: sentence-transformers, OpenAI embeddings, Cohere
- **Example**: "happy" → [0.23, -0.45, 0.67, ...]

### Vector Databases
- **What**: Specialized storage for high-dimensional vectors
- **Why**: Fast similarity search across millions of items
- **Tools**: ChromaDB, Pinecone, Weaviate, Qdrant
- **Example**: Find 10 most similar documents in 50ms

### LLM APIs
- **What**: Interface to large language models
- **Why**: Generate text, answer questions, follow instructions
- **Tools**: OpenAI, Anthropic, Google, local models (Ollama)
- **Example**: Send prompt → receive completion

---

## Level 2: Patterns

Established techniques for solving common problems.

### Prompting Techniques
- **What**: Strategies for getting better LLM outputs
- **Types**:
  - Zero-shot: Direct question
  - Few-shot: Examples in prompt
  - Chain-of-thought: Step-by-step reasoning
  - System prompts: Role and behavior definition

### RAG (Retrieval-Augmented Generation)
- **What**: Combine retrieval with generation
- **Why**: Ground LLM responses in actual data
- **Flow**: Query → Embed → Search → Retrieve → Generate
- **Use case**: Q&A over documents, knowledge bases

### Function/Tool Calling
- **What**: LLM outputs structured data to call functions
- **Why**: Bridge LLM reasoning to real actions
- **Example**: "What's the weather?" → `{"function": "get_weather", "args": {"city": "Tokyo"}}`

---

## Level 3: Applications

Complete solutions built from patterns.

### Chatbots
- **What**: Conversational interface with memory
- **Components**: LLM + conversation history + (optional) RAG
- **Challenge**: Context window management, personality consistency

### Search Systems
- **What**: Semantic search over content
- **Components**: Embeddings + vector DB + ranking
- **Advantage**: Finds meaning, not just keywords

### Pipelines
- **What**: Chained operations for complex tasks
- **Example**: Document → Chunk → Embed → Store → Index
- **Tools**: LangChain, LlamaIndex

---

## Level 4: Agents

Autonomous systems that reason and act.

### Single Agent
- **What**: LLM in a loop with tools and reasoning
- **Components**:
  - Loop: Observe → Think → Act → Repeat
  - Tools: Functions the agent can call
  - Memory: State across iterations
- **Example**: ReAct agent that searches web and synthesizes answer

### Multi-Agent Systems
- **What**: Multiple specialized agents working together
- **Patterns**:
  - Hierarchical: Orchestrator delegates to specialists
  - Peer-to-peer: Agents collaborate as equals
  - Pipeline: Agents in sequence
- **Example**: Research agent → Analysis agent → Writer agent

### Autonomous Systems
- **What**: Agents that plan and execute complex goals
- **Components**:
  - Planning: Break goal into subtasks
  - Execution: Complete subtasks with tools
  - Reflection: Evaluate and adjust
- **Challenge**: Reliability, cost control, safety

---

## How Levels Build on Each Other

```
Level 4: AGENTS
    │
    ├── Uses Level 3 applications as components
    │   (agent might have a RAG tool, chat memory)
    │
    ├── Uses Level 2 patterns for reasoning
    │   (chain-of-thought, function calling)
    │
    └── Uses Level 1 foundations for everything
        (embeddings for memory, LLM for reasoning)

Example: Multi-Agent Research System
─────────────────────────────────────
Orchestrator Agent
    │
    ├── Research Agent
    │   └── Uses: RAG (L2) + Web Search Tool (L2)
    │
    ├── Analysis Agent
    │   └── Uses: Chain-of-thought (L2) + LLM reasoning (L1)
    │
    └── Writer Agent
        └── Uses: LLM generation (L1) + Prompting (L2)
```

---

## Key Insight: Composition

Each level composes elements from lower levels:

| Level | Composes | Into |
|-------|----------|------|
| 2 (Patterns) | Embeddings + Vector DB | RAG |
| 2 (Patterns) | LLM + Schema | Function Calling |
| 3 (Applications) | RAG + Memory | Chatbot |
| 4 (Agents) | Tools + Loop + Reasoning | Single Agent |
| 4 (Agents) | Multiple Agents + Orchestration | Multi-Agent |

---

## Comparison: What Each Level Answers

| Level | Question Type | Example |
|-------|---------------|---------|
| 1 - Foundations | "What is similar to X?" | Find similar documents |
| 2 - Patterns | "What does the data say about X?" | RAG answer from docs |
| 3 - Applications | "Help me with X" | Chat about topic |
| 4 - Agents | "Figure out how to solve X" | Plan and execute solution |

**The key difference:**
- Levels 1-3: You define the steps
- Level 4: Agent defines its own steps

---

## Learning Path Mapping

This taxonomy maps directly to the learning roadmap:

| Phase | Taxonomy Levels | Focus |
|-------|-----------------|-------|
| Phase 1 | Pre-requisites | NumPy, Pandas (math foundations) |
| Phase 2 | Level 1 + 2 | Embeddings, APIs, Prompting |
| Phase 3 | Level 2 + 3 | RAG, Chat, Function Calling |
| Phase 4 | Level 4 | Agents, Multi-Agent, Orchestration |
| Phase 5 | All levels | Production deployment |

---

## When to Use Each Level

### Use Level 1-2 (Foundations/Patterns) when:
- Task is well-defined
- Single operation needed
- Low latency required
- Cost sensitivity high

### Use Level 3 (Applications) when:
- User interaction needed
- Memory/context required
- Multiple patterns combined
- Standard use case

### Use Level 4 (Agents) when:
- Task requires planning
- Multiple steps unknown upfront
- Tool selection is dynamic
- Complex reasoning needed

---

## Industry Perspective

The industry has largely converged on this hierarchy:

**Level 1-2**: Commoditized
- Standard APIs, well-documented
- Focus on cost/performance optimization

**Level 3**: Mature
- Established patterns (RAG, chat)
- Many frameworks (LangChain, LlamaIndex)

**Level 4**: Active Development
- Rapid evolution (2023-2025)
- Frameworks: OpenAI Agents SDK, LangGraph, AutoGen, CrewAI
- Still establishing best practices

---

## Related Documents

- [ai_application_flow.md](ai_application_flow.md) - Detailed RAG flow (Level 2-3)
- [rag_explained.md](rag_explained.md) - Deep dive into RAG
- [Phase 4 README](../../phase4_ai_agents/README.md) - Agent implementation details

---

*Last updated: 2025-01-14*
*Part of: AI Development Learning Roadmap*
