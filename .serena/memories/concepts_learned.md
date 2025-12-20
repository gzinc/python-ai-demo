# Concepts Learned - AI Development Journey

Track key concepts mastered during the learning journey.

## Session: 2025-01-13 - Embeddings Breakthrough

### Embeddings (★★★★★ Critical Concept)

**What**: Text converted to numbers (vectors) that preserve semantic meaning

**How**:
- AI models (OpenAI, Hugging Face) convert text → fixed-length array of numbers
- Typical sizes: 384, 768, 1536, or 3072 dimensions
- Each dimension captures an aspect of meaning
- Similar meanings → similar number patterns

**Example**:
```python
"cat"    → [0.2, 0.8, 0.1, ...]  # 768 numbers
"kitten" → [0.25, 0.75, 0.15, ...] # Similar!
"car"    → [-0.5, 0.1, 0.9, ...]   # Very different
```

**Why It Matters**:
- Foundation of ALL modern AI applications
- Enables semantic search (meaning-based, not keyword-based)
- Powers RAG systems, chatbots, recommendations
- Without embeddings, LLM apps can't work

**Real Usage**:
```python
# Document Q&A (RAG)
doc_embedding = get_embedding("Vacation policy: 15 days")
query_embedding = get_embedding("How many vacation days?")
similarity = np.dot(doc_embedding, query_embedding)  # High similarity!
```

---

### Vector Databases (★★★★ Important)

**What**: Specialized databases for storing and searching high-dimensional vectors (embeddings)

**Why Needed**:
- Regular databases can't efficiently search vectors
- Need specialized indexes (HNSW, IVF) for fast similarity search
- 100-1000x faster than scanning all rows

**Options**:
- **ChromaDB**: Easy, perfect for learning
- **Pinecone**: Managed cloud, scales to billions
- **pgvector**: PostgreSQL extension
- **FAISS**: Fastest, but low-level

**How It Works**:
```python
# Store embeddings once
db.add(
    id="doc1",
    embedding=[0.2, 0.8, ...],
    text="Document content"
)

# Search is fast (uses specialized indexes)
results = db.query(
    query_embedding=[0.25, 0.75, ...],
    top_k=5
)
```

---

### NumPy for AI (★★★★★ Critical)

**What**: Python library for numerical operations on arrays

**Why Critical for AI**:
- Embeddings ARE NumPy arrays
- All similarity calculations use NumPy operations
- Every AI library builds on NumPy
- 10-100x faster than Python loops

**Key Operations**:
```python
# Cosine similarity (used everywhere in AI)
similarities = np.dot(embeddings, query)

# Find best match
best_idx = np.argmax(similarities)

# Batch operations (no loops!)
normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

**Connection**:
```
Text → Embedding (NumPy array) → Store in Vector DB → 
Search (NumPy operations) → Find similar → Use in LLM
```

---

### RAG Architecture (★★★★ Key Pattern)

**What**: Retrieval Augmented Generation - Pattern for giving LLMs access to your documents

**How It Works**:
1. **Index**: Convert documents to embeddings, store in vector DB
2. **Retrieve**: User query → embedding → find similar documents
3. **Augment**: Add retrieved docs as context
4. **Generate**: LLM answers based on your documents

**Code Pattern**:
```python
# 1. Index (one-time)
for doc in documents:
    embedding = get_embedding(doc)
    vector_db.store(doc, embedding)

# 2-4. Query (every search)
query_emb = get_embedding(user_question)
relevant_docs = vector_db.search(query_emb, top_k=3)
response = llm.complete(f"Context: {relevant_docs}\nQuestion: {user_question}")
```

**Why It Matters**:
- Solves LLM knowledge limitation (can't know your private docs)
- More reliable than fine-tuning for most use cases
- Core pattern for AI applications

---

### Semantic Search (★★★★ Core Feature)

**What**: Search by meaning, not just keywords

**How**:
- Convert search query to embedding
- Compare to stored document embeddings
- Return most similar by cosine similarity

**Example**:
```python
Query: "scheduled appointments"
Matches: "upcoming meetings" (similar meaning!)
Doesn't match: "meeting room booking" (different meaning)

# Even though "meeting" appears in second result
# Semantic search finds the right one
```

**Use Cases**:
- Document search systems
- Customer support (find similar tickets)
- Recommendation engines
- Duplicate detection

---

### Embedding Cost Models (★★★ Practical)

**API-Based**:
- OpenAI: $0.02 per 1M tokens (~$10 for 10K documents)
- Generate once, store forever
- Best quality

**Open-Source**:
- sentence-transformers: Free, runs locally
- Good quality, no API costs
- Requires compute resources

**Strategy**:
- Development: Free models (sentence-transformers)
- Production: Evaluate if OpenAI quality worth cost
- Pre-compute and cache (don't regenerate on every search)

---

### Storage Pattern (★★★★ Critical Pattern)

**The Pattern**:
```python
# ONE TIME (expensive)
for doc in all_documents:
    embedding = get_embedding(doc)  # API call or compute
    db.store(doc, embedding)         # Save both text and embedding

# EVERY SEARCH (cheap and fast)
query_emb = get_embedding(query)    # 1 API call
stored_embs = db.get_all()          # Just read from storage (free)
similarities = np.dot(stored_embs, query_emb)  # Math (free)
```

**Why This Matters**:
- Embedding generation is slow/expensive
- Similarity calculation is fast/free
- Generate once, search millions of times
- Critical for cost and performance

---

## Concepts Still To Learn

### Phase 1 Remaining
- [ ] Pandas DataFrames (data manipulation)
- [ ] Data preprocessing for ML
- [ ] Train/test splits
- [ ] Feature engineering

### Phase 2
- [ ] Prompt engineering techniques
- [ ] LLM API integration (OpenAI, Anthropic)
- [ ] Token management
- [ ] Generating embeddings with code

### Phase 3
- [ ] Building complete RAG system
- [ ] Document chunking strategies
- [ ] Chat interfaces with memory
- [ ] Function calling

### Phase 4
- [ ] Agent architectures
- [ ] Tool use and orchestration
- [ ] Multi-agent systems

### Phase 5
- [ ] Production deployment
- [ ] Monitoring and observability
- [ ] Cost optimization
- [ ] Scaling strategies

---

## Key Realizations

1. **Embeddings are the bridge**: Between human text and computer math
2. **NumPy is foundational**: Not optional for AI development
3. **Pattern is consistent**: Text → Embedding → Store → Search → LLM
4. **Can start free**: sentence-transformers + ChromaDB for learning
5. **Learning path makes sense**: Each phase builds on previous

---

## Confidence Tracker

| Concept | Confidence | Notes |
|---------|-----------|-------|
| Embeddings | ★★★★★ | Clear mental model |
| Vector DBs | ★★★★☆ | Understand purpose, will learn details in Phase 3 |
| NumPy for AI | ★★★★☆ | See the connection, need more practice |
| RAG Architecture | ★★★★☆ | Understand flow, ready to build |
| Semantic Search | ★★★★★ | Clear understanding |
| Cost Models | ★★★★★ | Know trade-offs |

---

---

## Session: 2025-12-20 - Deep Learning Fundamentals

### Tensors (★★★★★ Critical Concept)

**What**: Multi-dimensional arrays of numbers - the data structure of deep learning

```python
# 0D: scalar     42
# 1D: vector     [1, 2, 3]           ← embeddings are 1D tensors
# 2D: matrix     [[1, 2], [3, 4]]    ← weight matrices
# 3D+: stacks    [[[...]]]           ← batches, sequences
```

**Why It Matters**:
- Neural networks = tensor operations
- Inputs, outputs, weights - all tensors
- Understanding shapes is key to debugging AI code

---

### PyTorch (★★★★★ Critical Library)

**What**: Python library for tensor operations with GPU support and auto-gradients

**Why It Exists**:
- NumPy can't run on GPU
- NumPy can't auto-compute gradients (needed for training)
- PyTorch = "NumPy that runs on GPU + auto-gradients"

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
x = x.to('cuda')  # move to GPU
x.requires_grad = True  # enable gradient tracking
```

**Ecosystem**:
- torch → core tensor operations
- torchvision → image models
- transformers (HuggingFace) → LLMs, built on PyTorch

---

### CUDA (★★★★ Important for Performance)

**What**: NVIDIA's parallel computing platform for running code on GPUs

```
CPU: 4-16 powerful cores     → sequential, complex logic
GPU: 1000s of tiny cores     → parallel, same math on many numbers
```

**Why GPUs for AI**:
- Neural nets = massive matrix multiplications
- Each multiplication is independent (parallelizable)
- GPUs are 2-100x faster for AI workloads

**In Code**:
```python
model.to('cpu')   # uses CPU (Intel/AMD)
model.to('cuda')  # uses GPU via CUDA (NVIDIA)
```

---

### Model Weights (★★★★★ Core Understanding)

**What**: The learned parameters stored in model files

**Example - LLMLingua-2 BERT model**:
- 677MB file = 177 million floating point numbers
- Format: SafeTensors (modern) or pickle (legacy)
- Structure:
  - Embeddings: 92M params (word → vector lookup)
  - 12 Transformer layers: 85M params (attention + feed-forward)
  - Classifier head: keep/drop decision

**Why Loading is Slow (even when cached)**:
1. Download: once (first run only)
2. Load from disk: every time (~677MB read)
3. Allocate RAM: create tensor objects
4. Build compute graph: prepare for inference

**Storage Location**:
```
~/.cache/huggingface/hub/models--<org>--<model>/
├── blobs/      # actual weight files (SHA-addressed)
├── snapshots/  # symlinks by version
└── refs/       # branch pointers
```

---

### Lazy Import Pattern (★★★ Practical Pattern)

**What**: Import heavy dependencies inside functions, not at module level

```python
# ❌ Eager: loads torch on import
import torch
class MyClass:
    pass

# ✅ Lazy: loads torch only when instantiated
class MyClass:
    def __init__(self):
        import torch  # deferred
```

**When to Use**:
- Heavy dependencies (torch, tensorflow)
- Optional features
- CLI tools where not all code paths are used

---

### Prompt Compression (★★★★ Production Pattern)

**What**: Reduce token count while preserving meaning → reduce cost

**LLMLingua-2 (state-of-the-art 2025)**:
- BERT model trained on GPT-4 compression examples
- Per-token binary classification: keep or drop
- 50-70% reduction with minimal quality loss

**Results**:
| Rate | Meaning | Actual Reduction |
|------|---------|------------------|
| 0.7 | Keep 70% | 32.9% savings |
| 0.5 | Keep 50% | 52.9% savings |
| 0.3 | Keep 30% | 70.6% savings |

**Alternatives**:
- Naive regex: ~25% (poor quality)
- LLM summarization: good quality but slow + costs API
- Soft prompts: different approach (learned embeddings)

---

## Updated Confidence Tracker

| Concept | Confidence | Notes |
|---------|-----------|-------|
| Embeddings | ★★★★★ | Clear mental model |
| Tensors | ★★★★★ | Understand shapes and operations |
| PyTorch | ★★★★☆ | Know purpose, need more practice |
| CUDA/GPU | ★★★★☆ | Understand why, tested on RTX 5070 Ti |
| Model Weights | ★★★★★ | Inspected actual file structure |
| Prompt Compression | ★★★★★ | Built production implementation |

---

Last updated: 2025-12-20
