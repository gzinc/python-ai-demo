# AI/LLM Application Flow - Complete Guide

Understanding the end-to-end flow of AI applications using embeddings, vector databases, and LLMs.

## Table of Contents
- [Overview](#overview)
- [The 5 Stages](#the-5-stages)
- [Stage Details](#stage-details)
- [Complete Example](#complete-end-to-end-example)
- [Real-World Architectures](#real-world-application-architectures)
- [Common Patterns](#common-patterns)
- [Cost Analysis](#cost-breakdown)
- [Performance Metrics](#performance-metrics)
- [Learning Path Connection](#the-flow-in-your-learning-path)

---

## Overview

Every AI/LLM application follows a standard pattern of 5 stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI APPLICATION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║                    ONE-TIME SETUP PHASE                           ║    │
│   ╠═══════════════════════════════════════════════════════════════════╣    │
│   ║  ┌──────────┐         ┌──────────┐         ┌──────────┐          ║    │
│   ║  │ 1.SETUP  │────────►│2.INDEXING│────────►│  STORED  │          ║    │
│   ║  │ Configure│         │ Chunk &  │         │ Vectors  │          ║    │
│   ║  │ Tools    │         │ Embed    │         │ Ready!   │          ║    │
│   ║  └──────────┘         └──────────┘         └──────────┘          ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│                                                       │                     │
│                                                       ▼                     │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║                    PER-REQUEST PHASE                              ║    │
│   ╠═══════════════════════════════════════════════════════════════════╣    │
│   ║  ┌──────────┐         ┌──────────┐         ┌──────────┐          ║    │
│   ║  │ 3.QUERY  │────────►│4.RETRIEVE│────────►│5.GENERATE│          ║    │
│   ║  │ User     │         │ Find     │         │ LLM      │          ║    │
│   ║  │ Question │         │ Relevant │         │ Answer   │          ║    │
│   ║  └──────────┘         └──────────┘         └──────────┘          ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

TIMING & COST BREAKDOWN:
┌─────────────┬────────────────┬──────────────┬────────────────────────────┐
│   Stage     │     When       │    Cost      │         Time               │
├─────────────┼────────────────┼──────────────┼────────────────────────────┤
│ 1. Setup    │ Once/project   │ $0           │ 5 minutes                  │
│ 2. Indexing │ Per document   │ ~$0.001/doc  │ ~1 second/doc              │
│ 3. Query    │ Per request    │ $0           │ instant                    │
│ 4. Retrieve │ Per request    │ $0           │ 10-50ms                    │
│ 5. Generate │ Per request    │ $0.001-0.01  │ 1-5 seconds                │
└─────────────┴────────────────┴──────────────┴────────────────────────────┘
```

**Key principle**: Prepare once (expensive), execute many times (cheap and fast).

---

## The 5 Stages

### Stage 1: SETUP (One-Time)
**What**: Configure your AI stack
**When**: Once per project
**Cost**: $0 for open-source, API keys for cloud
**Time**: 5 minutes

### Stage 2: INDEXING (When Adding Content)
**What**: Convert documents to searchable format
**When**: Every time you add/update content
**Cost**: API cost for embeddings OR compute time
**Time**: ~1s per document (API) or ~0.1s (local)

### Stage 3: USER QUERY (Every Request)
**What**: User asks a question
**When**: Every request
**Cost**: Free (just receiving input)
**Time**: Instant

### Stage 4: RETRIEVAL (Find Relevant Content)
**What**: Find documents that can answer the question
**When**: Every request
**Cost**: Free (just math!)
**Time**: ~10-50ms for 10K documents

### Stage 5: GENERATION (LLM Creates Answer)
**What**: Send relevant context + question to LLM
**When**: Every request
**Cost**: ~$0.001-0.01 per query
**Time**: 1-5 seconds

---

## Stage Details

### Stage 1: SETUP (One-Time)

Configure your AI infrastructure.

```python
# 1a. Choose your tools
from sentence_transformers import SentenceTransformer
import chromadb
import openai

# 1b. Initialize embedding model (local or API)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, local
# OR
# openai.api_key = "your-key"  # Paid, API

# 1c. Initialize vector database
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_documents")

# 1d. Initialize LLM
openai.api_key = "your-llm-key"

# DONE - ready to add documents!
```

**Tools Decisions**:
- **Embeddings**: sentence-transformers (free) vs OpenAI (paid, better quality)
- **Vector DB**: ChromaDB (local) vs Pinecone (cloud, scalable)
- **LLM**: OpenAI, Anthropic, or local models

---

### Stage 2: INDEXING (When Adding Content)

Convert documents into searchable embeddings.

```python
# User uploads documents or you have existing content
documents = [
    "Python is a programming language used for web development and AI.",
    "Machine learning is a subset of artificial intelligence.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Docker containers help deploy applications consistently."
]

# INDEXING FLOW:
for i, doc in enumerate(documents):
    # 2a. Create embedding (text → numbers)
    embedding = embedding_model.encode(doc)
    # Result: [0.234, -0.891, 0.456, ..., 0.123]  # 384 numbers

    # 2b. Store in vector database
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding.tolist()],
        documents=[doc],
        metadatas=[{"source": f"doc_{i}.txt"}]
    )

    print(f"Indexed document {i}")

# NOW: Documents are searchable by meaning!
```

**What Gets Stored**:
```
Document 1:
  Text: "Python is a programming language..."
  Embedding: [0.234, -0.891, 0.456, ..., 0.123]  # 384 numbers
  Metadata: {"source": "doc_0.txt"}

Document 2:
  Text: "Machine learning is a subset..."
  Embedding: [0.145, -0.672, 0.889, ..., 0.445]
  Metadata: {"source": "doc_1.txt"}
```

**Key Points**:
- Embedding captures the **meaning** of text as numbers
- Store both text and embedding together
- Metadata helps with filtering and source attribution
- This is expensive/slow - do it once per document

---

### Stage 3: USER QUERY (Every Request)

User interaction begins.

```python
# User types into your app
user_question = "How do I build web APIs with Python?"

print(f"User asked: {user_question}")
```

**This triggers the next 2 stages** (retrieval → generation).

---

### Stage 4: RETRIEVAL (Find Relevant Content)

Find documents that can answer the question using similarity search.

```python
# 4a. Convert question to embedding
question_embedding = embedding_model.encode(user_question)
# Result: [0.156, -0.723, 0.892, ..., 0.234]

# 4b. Search vector database for similar embeddings
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3  # Get top 3 most relevant
)

# 4c. Extract relevant documents
relevant_docs = results['documents'][0]
print("Found relevant documents:")
for i, doc in enumerate(relevant_docs):
    print(f"  {i+1}. {doc[:50]}...")

# Output:
#   1. FastAPI is a modern Python web framework...
#   2. Python is a programming language used for...
#   3. Docker containers help deploy applications...
```

**What Happens Behind the Scenes** (ChromaDB):
```python
# ChromaDB calculates similarity between query and all stored docs
for doc_embedding in stored_embeddings:
    similarity = cosine_similarity(question_embedding, doc_embedding)

# Returns top N most similar
# "FastAPI..." has similarity 0.89 (very relevant!)
# "Python..." has similarity 0.67 (somewhat relevant)
# "Machine learning..." has similarity 0.12 (not relevant)
```

**Why This is Fast**:
- No text comparison needed
- Just math: `np.dot(embeddings, query)`
- Vector DBs optimize this with special indexes (HNSW, IVF)
- 10-50ms for thousands of documents

---

### Stage 5: GENERATION (LLM Creates Answer)

Send relevant context + question to LLM for answer generation.

```python
# 5a. Build prompt with retrieved context
context = "\n\n".join(relevant_docs)

prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question: {user_question}

Answer:"""

# 5b. Send to LLM
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

# 5c. Extract answer
answer = response.choices[0].message.content

print(f"\nAnswer: {answer}")
# Output: "You can build web APIs with Python using FastAPI,
#          a modern framework designed for building APIs quickly..."
```

**What the LLM Receives**:
```
Context:
FastAPI is a modern Python web framework for building APIs.
Python is a programming language used for web development and AI.
Docker containers help deploy applications consistently.

Question: How do I build web APIs with Python?
```

**LLM Reasoning**: "I have context about FastAPI and Python. I can answer this accurately!"

**Key Benefits**:
- LLM answers based on YOUR documents (not just training data)
- Reduces hallucination (answer grounded in provided context)
- Can cite sources
- Works with private/proprietary information

---

## Complete End-to-End Example

Everything together in one script:

```python
from sentence_transformers import SentenceTransformer
import chromadb
import openai

# ============= STAGE 1: SETUP =============
print("Stage 1: Setup")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("docs")
openai.api_key = "your-key"

# ============= STAGE 2: INDEXING =============
print("\nStage 2: Indexing documents")
documents = [
    "Python is a programming language for web and AI development.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Docker containers help deploy applications consistently."
]

for i, doc in enumerate(documents):
    embedding = model.encode(doc)
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding.tolist()],
        documents=[doc]
    )
    print(f"  ✓ Indexed: {doc[:40]}...")

# ============= STAGE 3: USER QUERY =============
print("\nStage 3: User query")
question = "How do I build web APIs with Python?"
print(f"  User asked: {question}")

# ============= STAGE 4: RETRIEVAL =============
print("\nStage 4: Retrieval")
question_emb = model.encode(question)
results = collection.query(
    query_embeddings=[question_emb.tolist()],
    n_results=2
)
relevant_docs = results['documents'][0]
print(f"  Found {len(relevant_docs)} relevant documents")

# ============= STAGE 5: GENERATION =============
print("\nStage 5: Generation")
context = "\n".join(relevant_docs)
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    }]
)
answer = response.choices[0].message.content

print(f"\n✅ Final Answer:\n{answer}")
```

**Expected Output**:
```
Stage 1: Setup

Stage 2: Indexing documents
  ✓ Indexed: Python is a programming language for...
  ✓ Indexed: FastAPI is a modern Python web framew...
  ✓ Indexed: Docker containers help deploy applica...

Stage 3: User query
  User asked: How do I build web APIs with Python?

Stage 4: Retrieval
  Found 2 relevant documents

Stage 5: Generation

✅ Final Answer:
You can build web APIs with Python using FastAPI, a modern framework
specifically designed for creating APIs quickly and efficiently. FastAPI
provides automatic API documentation and excellent performance.
```

---

## Real-World Application Architectures

### Simple App (Learning/MVP)

```
User Browser
    ↓
FastAPI Backend
    ↓
┌──────────────────┐
│ Embedding Model  │ (sentence-transformers, local)
└──────────────────┘
    ↓
┌──────────────────┐
│ ChromaDB         │ (vector database, local)
└──────────────────┘
    ↓
┌──────────────────┐
│ OpenAI API       │ (LLM generation)
└──────────────────┘
    ↓
Response to User
```

**Characteristics**:
- Runs on single server
- Good for prototypes and small apps
- Low cost (~$50-100/month)
- Handles 10-100 requests/minute

---

### Production App (Scalable)

```
User Browser
    ↓
Load Balancer (Nginx/AWS ALB)
    ↓
FastAPI Servers (3+ instances, auto-scaling)
    ↓
┌─────────────┬──────────────┬─────────────┐
│ Embedding   │ Vector DB    │ LLM API     │
│ Service     │ (Pinecone)   │ (OpenAI)    │
│ (cached)    │ (scaled)     │ (with retry)│
└─────────────┴──────────────┴─────────────┘
    ↓
Redis Cache (for frequent queries)
    ↓
Monitoring/Logging (Prometheus, Grafana, Sentry)
    ↓
Database (PostgreSQL - for user data, not embeddings)
```

**Characteristics**:
- Handles 1000+ requests/minute
- Auto-scales based on load
- High availability (99.9% uptime)
- Cost: $500-2000/month depending on usage

**Key Components**:
- **Caching**: Redis stores frequent query results
- **Retry Logic**: LLM calls can fail, need retries
- **Monitoring**: Track latency, costs, errors
- **Rate Limiting**: Prevent abuse

---

## Common Patterns

### Pattern 1: Chat Interface with Memory

Maintain conversation history and retrieve relevant past context.

```python
# Stage 2: Index conversation history as it grows
conversation = []

def chat(user_message):
    # Stage 3: User query
    conversation.append({"role": "user", "content": user_message})

    # Stage 4: Retrieve relevant past messages
    # (not just last N messages - find RELEVANT ones!)
    message_embedding = get_embedding(user_message)

    # Search past conversation for relevant context
    relevant_history = []
    for past_msg in conversation[:-5]:  # Skip last 5 (already in context)
        past_emb = get_embedding(past_msg["content"])
        similarity = np.dot(message_embedding, past_emb)

        if similarity > 0.7:  # High similarity
            relevant_history.append(past_msg)

    # Stage 5: Generate with relevant context
    context_messages = relevant_history[-3:] + conversation[-5:]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=context_messages
    )

    conversation.append({"role": "assistant", "content": response})
    return response
```

**Benefits**:
- Doesn't send entire conversation (expensive, hits token limits)
- Finds relevant past context even from 100 messages ago
- User: "What was that policy we discussed?" → Finds it!

---

### Pattern 2: Multi-Document Q&A

Index multiple PDFs and answer questions across all of them.

```python
# Stage 2: Index multiple PDFs
for pdf_file in pdf_files:
    text = extract_text(pdf_file)
    chunks = split_into_chunks(text, size=500)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        db.store(
            id=f"{pdf_file}_{i}",
            embedding=embedding,
            text=chunk,
            metadata={"source": pdf_file, "page": i}
        )

# Stage 3-5: Query across all documents
def answer_question(question):
    # Retrieve
    question_emb = get_embedding(question)
    results = db.search(question_emb, top_k=5)

    relevant_chunks = [r["text"] for r in results]
    sources = [r["metadata"]["source"] for r in results]

    # Generate
    context = "\n\n".join(relevant_chunks)
    answer = llm.generate(
        prompt=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    # Include sources
    return {
        "answer": answer,
        "sources": list(set(sources))  # Unique sources
    }
```

**Use Cases**:
- Corporate knowledge base (search all company docs)
- Legal document analysis
- Research paper Q&A
- Technical documentation search

---

### Pattern 3: Recommendation System

Find similar items without LLM (pure vector similarity).

```python
# Stage 2: Index products/items
for product in products:
    description = f"{product.name} {product.description} {product.category}"
    embedding = get_embedding(description)
    db.store(
        id=product.id,
        embedding=embedding,
        data=product
    )

# Stage 3-4: Find similar (NO LLM NEEDED!)
def recommend_similar(product_id, num_recommendations=5):
    # Get product embedding
    product_embedding = db.get_embedding(product_id)

    # Find similar
    similar = db.search(
        query_embedding=product_embedding,
        top_k=num_recommendations + 1  # +1 because includes itself
    )

    # Remove the product itself
    recommendations = [p for p in similar if p.id != product_id]

    return recommendations[:num_recommendations]

# Usage:
# User views "Wireless Mouse"
# System recommends: ["Mechanical Keyboard", "USB Hub", "Laptop Stand"]
# All similar based on descriptions!
```

**Benefits**:
- Fast (no LLM call needed)
- Cheap (just vector search)
- Works for any item type
- Captures semantic similarity

---

### Pattern 4: Semantic Search Engine

Search by meaning, not keywords.

```python
# Stage 2: Index website content
for page in website_pages:
    # Index each section separately for better granularity
    sections = split_page_into_sections(page)

    for section in sections:
        embedding = get_embedding(section.text)
        db.store(
            id=f"{page.url}#{section.id}",
            embedding=embedding,
            text=section.text,
            metadata={
                "url": page.url,
                "title": page.title,
                "section": section.heading
            }
        )

# Stage 3-4: Semantic search (no LLM generation)
def search(query, num_results=10):
    query_emb = get_embedding(query)

    results = db.search(
        query_embedding=query_emb,
        top_k=num_results
    )

    return [{
        "title": r["metadata"]["title"],
        "url": r["metadata"]["url"],
        "snippet": r["text"][:200],
        "relevance": r["score"]
    } for r in results]

# Example:
# Query: "how to reset password"
# Matches: "Account Recovery" page (even though no "password" in title!)
```

---

## Cost Breakdown

### For 10,000 Documents, 1,000 Queries/Day

#### Setup & Indexing (One-Time)

**Option A: OpenAI Embeddings**
- 10,000 docs × ~200 words = 2M words
- 2M words ≈ 2.7M tokens
- Cost: $0.02 per 1M tokens
- **Total: ~$0.05 one-time**

**Option B: Local Embeddings (sentence-transformers)**
- **Cost: $0** (free, runs on your hardware)
- Compute time: ~20 minutes on CPU, ~2 minutes on GPU

#### Vector Database

**ChromaDB (Local)**:
- **Cost: $0**
- Runs on your server
- Good for < 1M vectors

**Pinecone (Cloud)**:
- Free tier: 1 index, 5M queries/month
- Paid: $70/month for 5M vectors
- **Cost: $0-70/month**

#### Per Query Costs

**Embedding the Query**:
- OpenAI: $0.00002 per query
- Local: $0

**Vector Search**:
- **Cost: $0** (just math!)
- 10-50ms regardless of database size

**LLM Generation**:
- GPT-4: ~$0.01 per query
- GPT-3.5-turbo: ~$0.001 per query
- Claude: ~$0.008 per query

#### Monthly Total (1,000 queries/day = 30,000/month)

**Budget Option** (all local except LLM):
- Embeddings: $0 (local)
- Vector DB: $0 (ChromaDB)
- LLM: 30K × $0.001 = $30
- **Total: ~$30/month**

**Premium Option** (all cloud):
- Embeddings: 30K × $0.00002 = $0.60
- Vector DB: $70 (Pinecone)
- LLM: 30K × $0.01 = $300
- **Total: ~$370/month**

**Hybrid Option** (recommended):
- Embeddings: $0 (local)
- Vector DB: $0 (ChromaDB) or $70 (Pinecone if scaling)
- LLM: 30K × $0.001 = $30 (GPT-3.5-turbo)
- **Total: ~$30-100/month**

---

## Performance Metrics

### Typical Response Times

```
Stage 1 (Setup):         One-time, ~5 minutes
Stage 2 (Indexing):      1-2 seconds per document
Stage 3 (User Query):    Instant
Stage 4 (Retrieval):     10-50ms
Stage 5 (LLM):           1-5 seconds
---
Total User Wait:         1-5 seconds
```

### With Streaming

```
Stage 4 (Retrieval):     10-50ms
Stage 5 (First Token):   ~500ms
Stage 5 (Full Response): 3-10 seconds (but user sees progress!)
---
Time to First Word:      ~500ms (much better UX!)
```

### Accuracy Factors

**Embedding Quality**:
- Better embeddings → better retrieval
- OpenAI > sentence-transformers > simple word vectors

**Chunk Size**:
- Too small (< 100 words): Lose context
- Too large (> 1000 words): Too much noise
- **Optimal: 200-500 words**

**Number of Retrieved Documents**:
- Too few (1-2): Might miss relevant info
- Too many (10+): Too much noise for LLM
- **Optimal: 3-5 chunks**

**LLM Prompt Quality**:
- Clear instructions improve accuracy
- Few-shot examples help
- Temperature: 0.0-0.3 for factual, 0.7-1.0 for creative

### Optimization Strategies

**Caching**:
```python
# Cache frequent queries
cache = {}

def search_with_cache(query):
    if query in cache:
        return cache[query]  # Instant!

    result = full_search(query)
    cache[query] = result
    return result

# 50% cache hit rate → 50% cost reduction!
```

**Batch Processing**:
```python
# Instead of:
for doc in docs:
    embedding = get_embedding(doc)  # N API calls

# Do this:
embeddings = get_embeddings_batch(docs)  # 1 API call
```

**Progressive Loading**:
```python
# Show results as they arrive
async def search_and_stream(query):
    # Start retrieval immediately
    relevant_docs = await retrieve(query)

    # Stream LLM response
    async for chunk in llm.stream(query, relevant_docs):
        yield chunk  # User sees words appearing!
```

---

## The Flow in Your Learning Path

### How This Maps to Your Roadmap

**Phase 1: Foundations (Current)**
- Learn NumPy → Powers Stage 4 (similarity calculations)
- Learn Pandas → For processing documents in Stage 2
- Learn ML concepts → Understanding embeddings

**Phase 2: LLM Fundamentals (Weeks 3-4)**
- **Stage 2**: Learn to generate embeddings
- **Stage 5**: Learn LLM API integration
- Practice with OpenAI/Anthropic APIs

**Phase 3: LLM Applications (Weeks 5-7)**
- **Stages 1-5**: Build complete RAG system!
  - Document ingestion (Stage 2)
  - ChromaDB setup (Stage 1)
  - Query handling (Stages 3-5)
- Build chat interface with memory (Pattern 1)
- Implement semantic search (Pattern 4)

**Phase 4: AI Agents (Weeks 8-11)**
- Add decision-making layer on top of RAG
- Tool use (agents can search, then decide next action)
- Multi-step reasoning

**Phase 5: Production (Weeks 12-15)**
- Scale the architecture
- Add caching, monitoring, error handling
- Optimize costs
- Deploy to production

### You'll Build This Flow Three Times

1. **Week 5-6**: Basic RAG (learning)
   - Simple documents
   - Basic search
   - Console interface

2. **Week 7**: Advanced RAG (features)
   - Multiple documents
   - Source attribution
   - Web interface

3. **Week 12-15**: Production RAG (scalable)
   - Caching
   - Monitoring
   - Error handling
   - FastAPI deployment

---

## Key Takeaways

1. **The pattern is consistent**: Setup → Index → Query → Retrieve → Generate
2. **Prepare once, execute many times**: Indexing is expensive, searching is cheap
3. **Embeddings are the bridge**: Convert text to math for similarity
4. **Vector databases are specialized**: Built specifically for high-dimensional vectors
5. **LLMs need context**: Retrieval provides relevant information to reduce hallucination
6. **This powers everything**: RAG, search, chat, recommendations all use this flow

---

## Next Steps

**Immediate**:
- Complete Phase 1 (NumPy, Pandas)
- Understand the math behind similarity search

**Soon** (Phase 2):
- Generate your first embeddings
- Practice similarity calculations
- Set up ChromaDB

**Build** (Phase 3):
- Implement Stages 1-5 yourself
- Create a document Q&A system
- Deploy a semantic search engine

**This flow is the foundation of modern AI applications. Master it, and you can build anything!** 🚀

---

## Additional Resources

### Documentation
- [ChromaDB Guide](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)

### Example Projects
- [phase3_llm_applications/01_rag_system/](../../phase3_llm_applications/01_rag_system/) - You'll build this!
- [phase3_llm_applications/02_chat_interface/](../../phase3_llm_applications/02_chat_interface/) - Coming soon
- [phase3_llm_applications/03_function_calling/](../../phase3_llm_applications/03_function_calling/) - Advanced

### Related Concepts
- [rag_explained.md](rag_explained.md) - Deep dive into RAG architecture

---

*Last updated: 2025-01-13*
*Part of: AI Development Learning Roadmap*
