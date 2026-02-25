# Phase 2: LLM Fundamentals

Understand how to work with large language models: prompt them effectively, call their APIs, and turn text into searchable vectors.

## What You'll Learn

- Write prompts that reliably produce the output you need
- Connect to OpenAI and Anthropic APIs with proper error handling
- Understand tokens, costs, and streaming responses
- Convert text into embedding vectors that capture meaning
- Measure similarity between texts using cosine similarity
- See how all of this flows into the RAG systems of Phase 3

## Prerequisites

- Python basics (functions, classes, loops)
- `uv sync` to install dependencies
- API keys optional — all three modules have no-key demos

## Modules

| Module | Topic | Key Concepts | API Key? |
|--------|-------|--------------|----------|
| [01_prompt_engineering](01_prompt_engineering/) | Prompting | System prompts, few-shot, CoT, output formatting | No |
| [02_api_integration](02_api_integration/) | LLM APIs | Request/response, streaming, token tracking, error handling | Optional |
| [03_embeddings](03_embeddings/) | Embeddings | Vector similarity, semantic search, sentence-transformers | No |

## Quick Start

```bash
# Module 1: Prompt Engineering (no API key needed)
uv run python phase2_llm_fundamentals/01_prompt_engineering/examples.py

# Module 2: API Integration (mock demos work without keys)
uv run python phase2_llm_fundamentals/02_api_integration/examples.py

# Module 3: Embeddings (no API key needed — local model)
uv run python phase2_llm_fundamentals/03_embeddings/examples.py
```

## The Big Picture

```
Phase 2: LLM Fundamentals
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  01_prompt_engineering        What you say to the LLM       │
│  ┌────────────────────┐                                      │
│  │ "You are a ..."    │  system prompts                      │
│  │ examples → answer  │  few-shot learning                   │
│  │ "Think step by..." │  chain-of-thought                    │
│  │ {"field": ...}     │  structured output                   │
│  └────────────────────┘                                      │
│                                                              │
│  02_api_integration           How you talk to the LLM       │
│  ┌────────────────────┐                                      │
│  │ messages=[]        │  conversation structure              │
│  │ stream=True        │  real-time token delivery            │
│  │ usage.tokens       │  cost tracking                       │
│  │ retry/backoff      │  production error handling           │
│  └────────────────────┘                                      │
│                                                              │
│  03_embeddings                How text becomes searchable    │
│  ┌────────────────────┐                                      │
│  │ "hello" → [0.021,  │  text → vector                      │
│  │  -0.14, 0.89, ...]  │  384 dimensions                    │
│  │ cos_sim(A, B)      │  similarity = angle                  │
│  │ rank by similarity │  semantic search                     │
│  └────────────────────┘                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
               Phase 3: LLM Applications
               (RAG, Chat, Function Calling)
```

## Module Progression

```
01_prompt_engineering/   → how to instruct the LLM reliably
        ↓
02_api_integration/      → how to call the LLM programmatically
        ↓
03_embeddings/           → how to make text searchable by meaning
        ↓
Phase 3: Applications    → combine all three to build real products
```

## Key Patterns

### Prompting

The pattern: **be explicit about role, task, format, and constraints**

```
System: "You are a code reviewer. Return JSON: {severity, issue, fix}"
User:   "Review: for i in range(len(items)):"
LLM:    {"severity": "low", "issue": "use enumerate()", "fix": "..."}
```

Why it matters:
- Consistent, parseable output for downstream code
- System prompt sets permanent context without repeating in every message
- Few-shot examples teach output format better than instructions

### API Integration

The pattern: **messages=[] is the conversation; you manage it**

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for..."},  # ← LLM response added here
    {"role": "user",      "content": "Give me an example."},  # ← next turn
]
```

Why it matters:
- LLMs are stateless — every call starts fresh
- The messages list IS the memory
- Phase 3's chat interface builds on this directly

### Embeddings

The pattern: **embed once, search many times by meaning**

```
Documents (index time):    Query (search time):
  "Python tutorial"           "how to code in Python"
  → [0.021, -0.14, ...]         → [0.019, -0.13, ...]
  stored in vector DB           compare → cosine 0.94 ✅
```

Why it matters:
- Finds "automobile motor issues" when you search "car engine problems"
- Keyword search fails on synonyms; embedding search doesn't
- Foundation for every RAG system

## Connection to Phase 3

| Phase 2 Concept | Phase 3 Usage |
|-----------------|---------------|
| Prompt templates | RAG context injection, function calling schemas |
| `messages=[]` | Multi-turn chat interface with memory |
| Streaming API | Chat UI token-by-token delivery |
| Embeddings | Document indexing in ChromaDB |
| Cosine similarity | Retrieval: find chunks relevant to user query |

## Further Reading

- [Phase 3: LLM Applications](../phase3_llm_applications/) — build RAG, chat, and tool-calling apps
- [docs/concepts/rag_explained.md](../docs/concepts/rag_explained.md) — deep dive into RAG architecture
