# Module 1: RAG System - Build a Complete RAG Pipeline

## Learning Objectives

By the end of this module, you will:
- Build a complete RAG system from scratch
- Implement document ingestion and chunking
- Create a retrieval pipeline with ChromaDB
- Integrate retrieved context with LLM generation
- Handle edge cases and improve response quality

## Prerequisites

- Phase 2 complete (Prompt Engineering, API Integration, Embeddings)
- ChromaDB experience from playground
- Understanding of RAG architecture (docs/concepts/rag_explained.md)

## RAG Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PHASE (One-time)                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────────────┐
  │Documents │ ───► │  Chunk   │ ───► │  Embed   │ ───► │   Vector Store   │
  │ .txt .md │      │ Strategy │      │  Model   │      │    (ChromaDB)    │
  │  .pdf    │      │          │      │          │      │                  │
  └──────────┘      └──────────┘      └──────────┘      └──────────────────┘
       │                 │                 │                     │
       ▼                 ▼                 ▼                     ▼
  "Full text..."   ["chunk1",        [[0.1, 0.3,         {id: vectors}
                    "chunk2",         -0.2, ...],        searchable!
                    "chunk3"]         [0.4, ...]]

┌─────────────────────────────────────────────────────────────────────────────┐
│                          QUERY PHASE (Per request)                          │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────────────┐
  │  User    │ ───► │  Embed   │ ───► │  Search  │ ───► │  Top-K Results   │
  │  Query   │      │  Query   │      │ ChromaDB │      │    (Context)     │
  └──────────┘      └──────────┘      └──────────┘      └──────────────────┘
       │                 │                 │                     │
       ▼                 ▼                 ▼                     ▼
  "How do I..."    [0.2, 0.1,        similarity         ["relevant chunk1",
                    0.4, ...]        matching            "relevant chunk2"]

                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                  GENERATION PHASE                   │
                    └─────────────────────────────────────────────────────┘

                    ┌──────────┐      ┌──────────┐      ┌──────────┐
                    │ Context  │ ───► │   LLM    │ ───► │  Answer  │
                    │ + Query  │      │ Generate │      │          │
                    │ (Prompt) │      │          │      │          │
                    └──────────┘      └──────────┘      └──────────┘
                         │                 │                 │
                         ▼                 ▼                 ▼
                    "Context:         GPT-4/Claude      "Based on the
                     [chunks]         processes          documentation,
                     Question:        grounded           you should..."
                     [query]"         response
```

## Chunking Strategies Visualized

```
Original Document:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Introduction to Python. Python is a high-level programming language.       │
│ It was created by Guido van Rossum. Python emphasizes code readability.    │
│                                                                             │
│ Variables in Python. Variables store data values. You don't need to        │
│ declare variable types. Python infers the type automatically.              │
│                                                                             │
│ Functions in Python. Functions are defined using the def keyword.          │
│ They help organize code into reusable blocks.                              │
└─────────────────────────────────────────────────────────────────────────────┘

Strategy 1: FIXED SIZE (500 chars)
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────┐
│ Chunk 1 (500 chars)     │  │ Chunk 2 (500 chars)     │  │ Chunk 3 (rest)  │
│ Introduction to Python. │  │ don't need to declare   │  │ They help       │
│ Python is a high-level  │  │ variable types. Python  │  │ organize code   │
│ programming language... │  │ infers the type...      │  │ into reusable...│
└─────────────────────────┘  └─────────────────────────┘  └─────────────────┘
     ⚠️ May cut mid-sentence!

Strategy 2: PARAGRAPH-BASED (natural breaks)
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────┐
│ Chunk 1 (Intro section) │  │ Chunk 2 (Variables)     │  │ Chunk 3 (Funcs) │
│ Introduction to Python. │  │ Variables in Python.    │  │ Functions in    │
│ Python is a high-level  │  │ Variables store data... │  │ Python. Funcs   │
│ programming language... │  │ Python infers type...   │  │ are defined...  │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────┘
     ✅ Preserves context!

Strategy 3: OVERLAPPING (with sliding window)
┌─────────────────────────────────┐
│ Chunk 1                         │
│ Introduction to Python...       │
│ ...code readability.            │
│ ┌───────────────────────────────┼───────────────────────┐
│ │         OVERLAP               │ Chunk 2               │
│ │ Python emphasizes code...     │ Variables in Python...│
└─┼───────────────────────────────┘                       │
  │                               ┌───────────────────────┼─────────────────┐
  │                               │         OVERLAP       │ Chunk 3         │
  └───────────────────────────────┼ Python infers type... │ Functions in... │
                                  └───────────────────────┴─────────────────┘
     ✅ Captures boundary context!
```

## Retrieval Quality Diagram

```
User Query: "How do I define a function in Python?"

                    Query Embedding
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VECTOR SIMILARITY SEARCH                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query Vector:  [0.2, 0.8, 0.1, -0.3, ...]                                 │
│                          │                                                  │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Chunk Vectors in ChromaDB                                           │   │
│  │                                                                     │   │
│  │  "Functions in Python..."  [0.3, 0.7, 0.2, -0.2, ...]  ──► 0.95 ✓  │   │
│  │  "Variables store data..." [0.1, 0.2, 0.8, -0.1, ...]  ──► 0.42    │   │
│  │  "def keyword creates..."  [0.2, 0.8, 0.1, -0.4, ...]  ──► 0.91 ✓  │   │
│  │  "Python is high-level..." [0.4, 0.3, 0.5, 0.1, ...]   ──► 0.38    │   │
│  │  "Return statements..."    [0.3, 0.6, 0.2, -0.3, ...]  ──► 0.87 ✓  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Top-3 Results (highest similarity):                                        │
│    1. "Functions in Python..."  (0.95)                                     │
│    2. "def keyword creates..."  (0.91)                                     │
│    3. "Return statements..."    (0.87)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Context Window Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM CONTEXT WINDOW (e.g., 8K tokens)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ System Prompt   │  ~200 tokens                                          │
│  │ "You are a..."  │                                                        │
│  └─────────────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ Retrieved Context                                       │               │
│  │ "Chunk 1: Functions in Python are defined using..."    │  ~2000 tokens │
│  │ "Chunk 2: The def keyword creates a function..."       │               │
│  │ "Chunk 3: Return statements send values back..."       │               │
│  └─────────────────────────────────────────────────────────┘               │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ User Question   │  ~50 tokens                                           │
│  │ "How do I..."   │                                                        │
│  └─────────────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ ═══════════════ RESERVED FOR RESPONSE ═══════════════  │  ~4000 tokens │
│  │                                                         │               │
│  │ (LLM generates answer here)                            │               │
│  │                                                         │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                             │
│  TOTAL: 200 + 2000 + 50 + 4000 = 6250 tokens (within 8K limit) ✓          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️ WARNING: If context too large:
┌─────────────────────────────────────────────────────────────────────────────┐
│  System: 200 │ Context: 7000 │ Question: 50 │ Response: ??? │              │
│              │               │              │               │              │
│  ════════════════════════════════════════════════════════════              │
│                         8K LIMIT EXCEEDED!                                  │
│              │               │              │               │              │
│              └───────────────┴──────────────┴───────────────┘              │
│                        TRUNCATED OR ERROR!                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Solution: Limit chunks or use summarization
```

## Files in This Module

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | Main orchestrator - RAGPipeline class coordinating all components |
| `chunking.py` | Document/Chunk dataclasses + 3 strategies (paragraph, sentence, fixed) |
| `retrieval.py` | RetrievalResult dataclass + Retriever class + context assembly |
| `examples.py` | Comprehensive demos showing different usage patterns |

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG Module Structure                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      rag_pipeline.py                                 │   │
│  │                     (Main Orchestrator)                              │   │
│  │                                                                      │   │
│  │  RAGPipeline class:                                                  │   │
│  │  - add_documents() → chunk → embed → store                          │   │
│  │  - query() → retrieve → build_prompt → generate                     │   │
│  │  - clear(), get_stats()                                             │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│              ┌──────────────┴──────────────┐                               │
│              │                             │                                │
│              ▼                             ▼                                │
│  ┌─────────────────────────┐   ┌─────────────────────────┐                 │
│  │     chunking.py         │   │     retrieval.py        │                 │
│  │                         │   │                         │                 │
│  │  • Document dataclass   │   │  • RetrievalResult      │                 │
│  │  • Chunk dataclass      │   │  • Retriever class      │                 │
│  │  • chunk_by_paragraph() │   │  • assemble_context()   │                 │
│  │  • chunk_by_sentence()  │   │  • format_results()     │                 │
│  │  • chunk_fixed_size()   │   │                         │                 │
│  └─────────────────────────┘   └─────────────────────────┘                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        examples.py                                   │   │
│  │                                                                      │   │
│  │  • example_basic_rag() - Add docs, query, full pipeline             │   │
│  │  • example_chunking_strategies() - Compare strategies               │   │
│  │  • example_retrieval_tuning() - Different top_k values              │   │
│  │  • example_interactive() - Sample Q&A session                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Running Examples

```bash
# run comprehensive examples (recommended)
uv run python phase3_llm_applications/01_rag_system/examples.py

# run individual modules with their demos
uv run python phase3_llm_applications/01_rag_system/chunking.py    # chunking strategies
uv run python phase3_llm_applications/01_rag_system/retrieval.py   # retrieval demo
uv run python phase3_llm_applications/01_rag_system/rag_pipeline.py # quick pipeline demo
```

## Exercises

1. **Basic RAG**: Build RAG over your own documents
2. **Chunking Experiment**: Compare different chunk sizes
3. **Multi-doc RAG**: Handle multiple document sources
4. **Citation RAG**: Add source citations to answers
5. **Evaluation**: Measure retrieval and answer quality

## Key Takeaways

1. **Chunking matters** - experiment with sizes for your use case
2. **Retrieval quality** - garbage in, garbage out
3. **Context window** - don't exceed LLM's limit
4. **Prompt engineering** - ground answers in context
5. **Evaluation** - measure both retrieval and generation

## Next Steps

After completing this module:
- → Module 2: Chat Interface (conversation memory)
- → Module 3: Function Calling (tool integration)