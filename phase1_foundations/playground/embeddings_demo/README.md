# 04 - Introduction to Embeddings

## Overview

Embeddings are the foundation of modern AI and LLM applications. This module introduces you to text embeddings through a practical example: processing your learning journey memories.

**Learning Time**: 1-2 hours
**Prerequisites**: [NumPy basics](../../01_numpy_basics/)

## What You'll Learn

- What embeddings are and why they're crucial for AI
- How to generate embeddings from text using sentence transformers
- Understanding embedding dimensions and vector space
- Calculating semantic similarity between documents
- Real-world application: analyzing your learning memories

## Why Embeddings Matter for AI

**Embeddings convert text into numbers** that capture semantic meaning. This enables:

- **Semantic search**: Find similar documents by meaning, not just keywords
- **RAG systems**: Retrieve relevant context for LLM responses
- **Clustering**: Group similar texts automatically
- **Classification**: Categorize text based on meaning
- **Recommendation**: Suggest related content

**Key insight**: Two sentences with completely different words can have similar embeddings if they mean the same thing.

## Embeddings in the AI Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS: FROM TEXT TO MEANING                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   "NumPy is essential for AI"        "I love machine learning"              │
│            │                                    │                           │
│            ▼                                    ▼                           │
│   ┌────────────────────────────────────────────────────────┐               │
│   │              EMBEDDING MODEL                           │               │
│   │        (sentence-transformers / OpenAI)                │               │
│   └────────────────────────────────────────────────────────┘               │
│            │                                    │                           │
│            ▼                                    ▼                           │
│   [0.23, -0.45, 0.12, ...]           [0.31, -0.52, 0.19, ...]              │
│        384 dimensions                     384 dimensions                    │
│                                                                             │
│                      SEMANTIC MEANING ENCODED!                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      VECTOR SPACE                                    │  │
│   │                                                                      │  │
│   │              "AI basics"  •                                          │  │
│   │                            \        • "machine learning"             │  │
│   │   "NumPy essential" •──────•──────•                                  │  │
│   │                                   /                                  │  │
│   │             "Python data" •──────•                                   │  │
│   │                                                                      │  │
│   │   (Similar topics cluster together!)                                 │  │
│   │                                                                      │  │
│   │   "Weather today" •                          • "Pizza recipe"       │  │
│   │                    (Far from AI topics!)                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   💡 KEY INSIGHT: Similar meanings = Close vectors = Low cosine distance   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Phase 1**: Learn what embeddings are (this module)
**[Phase 2](../../../phase2_llm_fundamentals/)**: Use embeddings with LLM APIs
**[Phase 3](../../../phase3_llm_applications/)**: Build RAG systems with embeddings + retrieval

## Hands-On: Process Your Learning Memories

This module includes a practical example that:

1. Reads all your Serena memory files (`.serena/memories/*.md`)
2. Generates embeddings for each memory using a multilingual model
3. Displays the embeddings and their properties
4. Shows how embeddings capture semantic meaning

**Why this example?**
- Uses real data (your actual learning journey)
- Demonstrates practical embedding workflow
- Shows embeddings are just NumPy arrays
- Prepares you for RAG systems in Phase 3

## Running the Example

```bash
# From project root
uv run python phase1_foundations/playground/embeddings_demo/memory_embeddings.py
```

**What you'll see**:
- Each memory file processed
- Embedding dimensions (384 for this model)
- Embedding vector preview (first 10 values)
- Embedding statistics (min, max, mean)
- Semantic similarity between memories

## The Model: paraphrase-multilingual-MiniLM-L12-v2

**What is it?**
- Sentence transformer model for converting text to embeddings
- Multilingual: works with multiple languages
- MiniLM: smaller, faster model (good for learning)
- L12: 12 transformer layers
- v2: version 2 of the model

**Embedding dimension**: 384 (each text becomes a 384-dimensional vector)

**Why this model?**
- Fast to download and run locally
- Good quality embeddings for learning
- Multilingual support (useful for diverse text)
- Widely used in tutorials and examples

**Production alternatives**:
- OpenAI `text-embedding-3-small` (1536 dimensions)
- Anthropic embeddings (coming soon)
- BGE models (better performance, larger size)

## Understanding Embeddings

### What is an embedding?

An embedding is a **dense vector representation** of text that captures semantic meaning.

**Example**:
```python
text = "AI is transforming software development"
embedding = model.encode(text)
# embedding.shape = (384,)
# embedding = [0.123, -0.456, 0.789, ..., 0.234]
```

### Key Properties

1. **Fixed dimension**: All embeddings from a model have the same length
   - This model: 384 dimensions
   - OpenAI: 1536 dimensions

2. **Dense vectors**: Every position has a meaningful value (not sparse)

3. **Semantic similarity**: Similar meanings → similar vectors
   ```python
   embed("I love AI") ≈ embed("I enjoy artificial intelligence")
   embed("I love AI") ≉ embed("The weather is nice")
   ```

4. **Vector operations**: Can use NumPy operations
   - Cosine similarity
   - Euclidean distance
   - Vector arithmetic

### From Text to Vector

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FROM TEXT TO EMBEDDING VECTOR                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: "NumPy is essential for AI"                                       │
│                      │                                                      │
│                      ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  1. TOKENIZATION                                                     │  │
│   │     "NumPy" │ "is" │ "essential" │ "for" │ "AI"                     │  │
│   │      [102]  [15]    [4523]        [27]    [873]                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                      │                                                      │
│                      ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  2. TRANSFORMER LAYERS (12 layers of attention + feed-forward)      │  │
│   │                                                                      │  │
│   │     Each token gets contextualized:                                  │  │
│   │     "AI" in "AI for art" ≠ "AI" in "AI for computing"               │  │
│   │                                                                      │  │
│   │     Layer 1 → Layer 2 → ... → Layer 12                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                      │                                                      │
│                      ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  3. POOLING (combine token vectors → single sentence vector)        │  │
│   │                                                                      │  │
│   │     Token 1 [0.1, 0.2, ...]  ─┐                                      │  │
│   │     Token 2 [0.3, 0.1, ...]  ─┼──► Mean pooling ──► [0.12, -0.45...]│  │
│   │     Token 3 [0.2, 0.4, ...]  ─┤                     (384 values)     │  │
│   │     ...                       ─┘                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                      │                                                      │
│                      ▼                                                      │
│   Output: [0.12, -0.45, 0.78, 0.02, -0.33, ..., 0.23]                      │
│           └──────────────────────────────────────────┘                     │
│                     384 floating point numbers                              │
│                                                                             │
│   💡 This vector "encodes" the meaning of "NumPy is essential for AI"     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Semantic Similarity

**Cosine similarity** measures how similar two embeddings are:

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([embedding1], [embedding2])[0][0]
# similarity ranges from -1 to 1
# 1.0 = identical meaning
# 0.0 = unrelated
# -1.0 = opposite meaning (rare)
```

**Typical ranges**:
- 0.9-1.0: Nearly identical
- 0.7-0.9: Very similar
- 0.5-0.7: Somewhat similar
- 0.3-0.5: Loosely related
- < 0.3: Unrelated

## Exercises

After running the example, try these:

### Exercise 1: Add Custom Text
Modify the script to embed your own text samples and compare their similarity.

### Exercise 2: Find Most Similar Memories
Extend the script to find the two most similar memory files.

### Exercise 3: Cluster Memories
Group your memories by topic using embedding similarity.

### Exercise 4: Visualize Embeddings
Use dimensionality reduction (PCA or t-SNE) to visualize embeddings in 2D.

## Key Takeaways

✅ Embeddings convert text to dense numeric vectors
✅ Similar meanings produce similar embeddings
✅ Embeddings enable semantic search and similarity
✅ All embeddings from a model have the same dimension
✅ Embeddings are just NumPy arrays (you can manipulate them!)

## Real-World Applications

**Semantic Search** (Phase 3):
```python
query_embedding = model.encode("How do I learn NumPy?")
# Find most similar documents from your knowledge base
```

**RAG System** (Phase 3):
```python
# 1. Embed your documents
# 2. Store in vector database
# 3. When user asks question, embed the question
# 4. Find similar documents
# 5. Send to LLM as context
```

**Chatbot Memory** (Phase 3):
```python
# Store conversation history as embeddings
# Retrieve relevant past conversations
# Give chatbot "memory" of previous interactions
```

## Next Steps

1. ✅ Run the memory embeddings example
2. ✅ Understand what embeddings are
3. ✅ See how embeddings capture meaning
4. ✅ Complete exercises

**Phase 2 Preview**: You'll use embeddings with:
- OpenAI embeddings API
- ChromaDB vector database
- Semantic search systems
- RAG applications

## Resources

**Sentence Transformers**:
- Docs: https://www.sbert.net/
- Model hub: https://huggingface.co/sentence-transformers

**Embeddings Guides**:
- OpenAI: https://platform.openai.com/docs/guides/embeddings
- Anthropic: https://docs.anthropic.com/claude/docs/embeddings (coming)
- Pinecone: https://www.pinecone.io/learn/embeddings/

**Vector Similarity**:
- Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- Distance metrics: https://scikit-learn.org/stable/modules/metrics.html

---

**Ready to see embeddings in action? Run the example!**

```bash
uv run python phase1_foundations/04_embeddings_intro/memory_embeddings.py
```
