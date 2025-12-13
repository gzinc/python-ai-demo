# Phase 3: LLM Applications

Build real-world AI applications: RAG systems, chat interfaces, and function calling.

## What You'll Learn

- Building RAG pipelines with document ingestion and retrieval
- Creating multi-turn chat interfaces with memory management
- Implementing function calling for LLM-powered tools
- Connecting LLMs to external data and APIs

## Prerequisites

- Phase 2: LLM Fundamentals (API Integration, Embeddings)
- Understanding of LLM APIs (OpenAI, Anthropic)

## Modules

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [01_rag_system](01_rag_system/) | RAG Pipeline | Chunking, retrieval, context integration |
| [02_chat_interface](02_chat_interface/) | Chat & Memory | Conversation memory, streaming, reference resolution |
| [03_function_calling](03_function_calling/) | Tool Integration | Function schemas, tool execution, orchestration |

## Quick Start

```bash
# Module 1: RAG System
cd phase3_llm_applications/01_rag_system
uv run python examples.py

# Module 2: Chat Interface
cd phase3_llm_applications/02_chat_interface
uv run python examples.py

# Module 3: Function Calling (needs API key)
cd phase3_llm_applications/03_function_calling
OPENAI_API_KEY=your-key uv run python examples.py
```

## The Big Picture

```
Phase 2: Fundamentals          Phase 3: Applications
┌──────────────────────┐      ┌──────────────────────────────┐
│  Embeddings          │  →   │  RAG System                  │
│  (text → vectors)    │      │  (retrieve relevant context) │
└──────────────────────┘      └──────────────────────────────┘
┌──────────────────────┐      ┌──────────────────────────────┐
│  API Integration     │  →   │  Chat Interface              │
│  (messages=[])       │      │  (multi-turn + memory)       │
└──────────────────────┘      └──────────────────────────────┘
┌──────────────────────┐      ┌──────────────────────────────┐
│  Prompt Engineering  │  →   │  Function Calling            │
│  (structured output) │      │  (LLM → tool execution)      │
└──────────────────────┘      └──────────────────────────────┘
```

## Module Progression

```
01_rag_system/       → How to give LLMs access to YOUR data
        ↓
02_chat_interface/   → How to maintain conversation context
        ↓
03_function_calling/ → How to let LLMs take actions
        ↓
Phase 4: Agents      → Autonomous systems that reason and act
```

## Key Patterns

### RAG (Retrieval-Augmented Generation)

The pattern: **Don't fine-tune, just retrieve**

```
User Question → Embed → Search Vector DB → Get relevant docs → Add to prompt → LLM generates
```

Why it matters:
- Use your own data without training
- Always up-to-date (just update the docs)
- Much cheaper than fine-tuning

### Chat Memory

The pattern: **LLMs are stateless, you manage history**

```
User says "What about Tokyo?"
    ↓
Your code maintains history: [previous messages about cities]
    ↓
LLM sees full context: "User asked about Paris, now asking about Tokyo"
    ↓
LLM responds with context-aware answer
```

Why it matters:
- Natural multi-turn conversations
- Reference resolution ("it", "them", "the other one")
- Coherent long interactions

### Function Calling

The pattern: **LLM decides, your code executes**

```
User: "What's the weather in Tel Aviv?"
    ↓
LLM returns: {"tool": "get_weather", "args": {"city": "Tel Aviv"}}
    ↓
Your code: Actually calls the weather API
    ↓
LLM: Formats the response for the user
```

Why it matters:
- LLMs can take real actions
- Access to live data and APIs
- Foundation for AI agents

## Connection to Phase 4

Phase 3's function calling is the **foundation** for Phase 4's agents:

| Phase 3: Function Calling | Phase 4: Agents |
|---------------------------|-----------------|
| LLM picks ONE function    | LLM picks functions in a LOOP |
| You execute it            | Agent executes and continues |
| Single action             | Multi-step workflows |
| Response ends             | Continues until goal met |

## Further Reading

- [RAG Explained](../docs/concepts/rag_explained.md) - Deep dive into RAG architecture
- [Phase 4: AI Agents](../phase4_ai_agents/) - Next step: autonomous systems
- [Phase 4 Concepts](../phase4_ai_agents/CONCEPTS.md) - Agent fundamentals
