# Phase 1 Playground

This directory contains **exploratory demos** and **practical examples** that answer real questions that emerge naturally during your Phase 1 learning journey.

## Purpose

Unlike the numbered modules (01, 02, 03) which are part of the formal curriculum, the playground contains:
- Practical examples that bridge theory to real AI applications
- Demos that answer "how does this relate to LLM work?"
- Exploratory code you can experiment with
- Real-world applications of Phase 1 concepts

## When to Use

The playground demos are meant to be discovered **organically** during your learning:
- You're working through NumPy and wonder "how is this used in AI?"
- You learn about vectors and ask "what are embeddings?"
- You complete exercises and want to see "real AI applications"

## Available Demos

### embeddings_demo/
**When to explore**: After learning NumPy arrays and cosine similarity

**Real question it answers**: "How do NumPy arrays relate to actual LLM applications?"

**What it does**: Processes your Serena learning memories and shows how text becomes vectors

**Run it**:
```bash
uv run python phase1_foundations/playground/embeddings_demo/memory_embeddings.py
```

**Key insights**:
- Embeddings are just NumPy arrays
- Cosine similarity measures semantic meaning
- This is the foundation for RAG systems (Phase 3)

## Philosophy

**Formal Curriculum** (01, 02, 03):
- Structured learning path
- Required modules
- Exercises with solutions
- Progress tracking

**Playground** (this directory):
- Curiosity-driven exploration
- Optional but valuable
- Real-world connections
- "Aha!" moment generators

## Adding More Demos

As you progress through Phase 1, more demos can be added here based on questions that arise:
- `vector_search_demo/` - when you ask about semantic search
- `text_similarity_demo/` - when you wonder about comparing documents
- `data_visualization_demo/` - when you want to see embeddings visually

---

**Remember**: Don't feel obligated to explore everything here. Let your curiosity guide you!
