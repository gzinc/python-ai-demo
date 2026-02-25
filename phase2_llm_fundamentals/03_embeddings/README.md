# Module 3: Embeddings — Vector Representations of Text

## Learning Objectives

By the end of this module, you will:
- Understand what embeddings are and why they matter for AI
- Compute cosine similarity and explain why it beats euclidean distance for text
- Generate real embeddings using `sentence-transformers` (no API key needed)
- Build a semantic search system from scratch
- Understand why semantic search outperforms keyword search
- See how embeddings directly power Phase 3 RAG systems

## Prerequisites

- [Phase 2 Module 2: API Integration](../02_api_integration/)
- No API keys needed — all demos use local sentence-transformers

## Key Concepts

### 1. What Is an Embedding?

An embedding converts text (or images, audio, etc.) into a fixed-size vector of numbers.
The critical property: similar meaning → similar vectors.

```
Text                           Embedding Vector (384 dims)
──────────────────────         ────────────────────────────────────
"Python is a language"    →    [0.021, -0.143, 0.891, ..., 0.034]
"Python is for coding"    →    [0.019, -0.138, 0.887, ..., 0.031]  ← close!
"Snakes eat mice"         →    [-0.412, 0.721, -0.33, ..., 0.195]  ← far away
```

### 2. Embedding Space

Think of embedding space as a high-dimensional map where:
- Nearby points = similar meaning
- Distant points = different meaning
- Directions encode relationships (gender, tense, topic)

```
            Technical axis →
            ┌────────────────────────────────────────┐
    high ↑  │  "neural net"         "deep learning"  │
            │                                        │
   Animal   │             "machine learning"         │
    axis    │                                        │
            │  "Python snake"                        │
            │                "Python code"           │
    low  ↓  │  "dog"  "cat"                          │
            └────────────────────────────────────────┘
```

### 3. Cosine Similarity

Measures the angle between two vectors (not their length).
Range: -1 (opposite) → 0 (unrelated) → 1 (identical)

```
Formula:
                    A · B
cos_sim(A, B) = ───────────
                  ||A|| × ||B||

where:
  A · B   = dot product (sum of element-wise products)
  ||A||   = L2 norm = sqrt(sum of squared elements)
```

**Why cosine over euclidean distance?**

```
vec_short = [1, 0]      same direction
vec_long  = [100, 0]    but different magnitude

euclidean distance:  99.0   ← wrongly suggests they differ
cosine similarity:    1.0   ← correctly shows identical direction

Embedding vectors often vary in magnitude, not meaning.
Cosine similarity is scale-invariant — the right tool for text.
```

### 4. Semantic vs Keyword Search

```
Query: "car engine problems"

┌──────────────────────────────────────────┬─────────────┬──────────┐
│ Document                                 │ Keyword     │ Semantic │
├──────────────────────────────────────────┼─────────────┼──────────┤
│ "The automobile engine makes a rattle"   │ ❌ 0 words  │ ✅ 0.61  │
│ "How to fix motor issues in your ride"   │ ❌ 0 words  │ ✅ 0.58  │
│ "My car breaks down frequently"          │ ✅ 1 word   │ ✅ 0.47  │
│ "Python is great for data science"       │ ❌ 0 words  │ ❌ 0.04  │
└──────────────────────────────────────────┴─────────────┴──────────┘

Keyword search misses synonyms: automobile, motor, ride, rattle
Semantic search captures meaning regardless of exact words used
```

### 5. The Model: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Size | ~22 MB |
| Output dims | 384 |
| Runs locally | Yes (no API key) |
| Training data | 1 billion sentence pairs |
| Speed | ~1000 sentences/second on CPU |

Used in Phase 3 RAG as `LocalEmbedder(model="all-MiniLM-L6-v2")`.

### 6. How Embeddings Power RAG

```
INDEXING (one-time setup)          QUERYING (per user question)
──────────────────────────         ──────────────────────────────

Documents                          User question
    │                                  │
    ▼                                  ▼
Split into chunks              Embed question
    │                          [0.021, -0.14, ...]
    ▼                                  │
Embed each chunk               Cosine similarity
[0.89, -0.23, ...]             against all stored vectors
    │                                  │
    ▼                                  ▼
Store in ChromaDB              Top-K most similar chunks
(persisted to disk)                    │
                                       ▼
                               Inject into LLM prompt:
                               "Context: [chunk 1] [chunk 2]
                                Question: user question"
                                       │
                                       ▼
                               LLM generates grounded answer
```

## Files in This Module

| File | Purpose |
|------|---------|
| [examples.py](examples.py) | 7 runnable demos, no API key needed |

## Running Examples

```bash
# all demos via interactive menu
uv run python phase2_llm_fundamentals/03_embeddings/examples.py

# run a specific demo directly (e.g. semantic search)
uv run python -c "
from phase2_llm_fundamentals.03_embeddings.examples import demo_semantic_search
demo_semantic_search()
"
```

## Demo Overview

| # | Demo | What You Learn |
|---|------|----------------|
| 1 | What Are Embeddings? | text → vector intuition with fake 3D space |
| 2 | Vector Math | cosine similarity formula, why not euclidean |
| 3 | Real Embeddings | sentence-transformers, 384-dim output, similarity matrix |
| 4 | Semantic Search | embed corpus, rank by query similarity |
| 5 | Semantic vs Keyword | synonyms, paraphrases, why keywords fail |
| 6 | Word Analogies | king − man + woman ≈ queen (vector arithmetic) |
| 7 | RAG Connection | how this all flows into Phase 3 |

## Key Takeaways

1. **Embeddings encode meaning as geometry** — similar text, similar vectors
2. **Cosine similarity is scale-invariant** — right tool for comparing text vectors
3. **Local models work great** — all-MiniLM-L6-v2 needs no API key, runs offline
4. **Semantic search beats keywords** — handles synonyms, paraphrases, context
5. **Everything here powers Phase 3** — RAG is just "embed → search → inject"

## Connection to Other Phases

```
Phase 2 Module 3 (this)
    embeddings fundamentals
    ↓
Phase 3: RAG System
    phase3_llm_applications/01_rag_system/
    uses LocalEmbedder + ChromaDB + cosine retrieval
    ↓
Phase 5: Production
    semantic caching (embed user query, match against cached responses)
    ↓
Phase 7: Frameworks
    LangChain: OpenAIEmbeddings / HuggingFaceEmbeddings
    LlamaIndex: embed model configuration
```

## Next Steps

- → [Phase 3: RAG System](../../phase3_llm_applications/01_rag_system/) — build a full RAG pipeline
- → [Phase 7: LlamaIndex](../../phase7_frameworks/03_llamaindex/) — framework for RAG at scale
