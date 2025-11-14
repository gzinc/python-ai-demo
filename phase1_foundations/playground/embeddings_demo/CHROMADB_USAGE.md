# ChromaDB Integration

This demonstrates how to save embeddings to a **vector database** for semantic search.

## Files

### Core Files
- `vector_db.py` - **VectorDB class** for ChromaDB operations (standard pattern)
- `save_to_chromadb.py` - **Script** that uses VectorDB to save memories
- `memory_embeddings.py` - **Original** demo (display only, no database)

### Database Storage
- `.chromadb/` - **Persistent storage** for vector database (gitignored)
- `.chromadb/chroma.sqlite3` - SQLite database (~412 KB)

## Usage

### Basic: Save Embeddings

```bash
uv run python phase1_foundations/playground/embeddings_demo/save_to_chromadb.py
```

**What it does**:
1. Loads all `.serena/memories/*.md` files
2. Generates embeddings using sentence-transformers
3. Saves to ChromaDB vector database
4. Demonstrates semantic search

### Advanced: Use VectorDB Class

```python
from vector_db import VectorDB
from sentence_transformers import SentenceTransformer

# initialize
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
db = VectorDB(collection_name="my_docs")

# save embeddings
db.save_embeddings(embeddings, documents)

# search by text
results = db.search_by_text("What is embeddings?", model, top_k=3)

# search by embedding
query_embedding = model.encode("some text")
results = db.search(query_embedding, top_k=3)

# get count
count = db.count()

# delete collection
db.delete_collection()
```

## VectorDB API

### Class: `VectorDB`

**Constructor**:
```python
VectorDB(
    collection_name: str = "learning_memories",
    db_path: Path = None  # defaults to .chromadb
)
```

**Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `create_collection(reset=True)` | Create new collection | Collection |
| `get_collection()` | Get existing collection | Collection |
| `save_embeddings(embeddings, documents, metadata)` | Save to DB | None |
| `search(query_embedding, top_k=3)` | Search by embedding | Results dict |
| `search_by_text(query, model, top_k=3)` | Search by text | Results dict |
| `get_all()` | Get all documents | Dict |
| `count()` | Count documents | int |
| `delete_collection()` | Delete collection | None |
| `list_collections()` | List all collections | List[str] |

**Results Format**:
```python
{
    'ids': [['doc1', 'doc2', ...]],
    'documents': [['text1', 'text2', ...]],
    'distances': [[0.1, 0.2, ...]],  # lower = more similar
    'metadatas': [[{...}, {...}, ...]]
}
```

## How It Works

### 1. Embedding Generation
```python
# text → vector
embedding = model.encode("AI is transforming development")
# embedding.shape = (384,)  # 384-dimensional vector
```

### 2. Save to ChromaDB
```python
db.save_embeddings(
    embeddings={'doc1': embedding_array},
    documents={'doc1': 'full text content'}
)
```

### 3. Semantic Search
```python
# user query
results = db.search_by_text("What is AI?", model, top_k=3)

# returns most similar documents based on meaning, not keywords!
```

### 4. Distance → Similarity
```python
# ChromaDB returns distance (lower = more similar)
distance = results['distances'][0][0]  # e.g., 0.15

# convert to similarity score
similarity = 1 - distance  # e.g., 0.85 (85% similar)
```

## Storage Details

**Location**: `.chromadb/` (project root, gitignored)

**Contents**:
- `chroma.sqlite3` - Main database file
- `<uuid>/` - Collection data directory

**Size**: ~600 KB for 5 documents with embeddings

**Persistence**: Data survives script restarts (unlike in-memory)

**Git**: Excluded via `.gitignore` (don't commit vector databases!)

## Semantic Search Examples

### Query: "What is my learning progress?"
**Top Result**: `learning_progress.md`
**Why**: Contains progress tracking, phase completion, module checklists

### Query: "How do I get started?"
**Top Result**: `getting_started.md`
**Why**: Contains setup instructions, first session guide, quickstart

### Query: "What are embeddings?"
**Top Result**: `concepts_learned.md` (your session notes!)
**Why**: Contains embeddings concepts, explanations, examples

## Key Concepts

### Vector Database
A database optimized for storing and searching **high-dimensional vectors** (embeddings).

**Traditional DB**:
```sql
SELECT * FROM docs WHERE title LIKE '%embeddings%'
```
→ Keyword matching only

**Vector DB**:
```python
db.search("what are embeddings", top_k=3)
```
→ **Semantic matching** (finds documents by meaning)

### Why ChromaDB?

**Pros**:
- ✅ Easy to use (Python-native)
- ✅ Persistent storage (SQLite backend)
- ✅ Free and open source
- ✅ Perfect for learning and prototypes
- ✅ Can scale to production

**Alternatives**:
- **Pinecone** - Cloud-hosted, production-ready
- **Weaviate** - Self-hosted, GraphQL API
- **Qdrant** - High performance, Rust-based
- **FAISS** - Facebook, in-memory only

**For learning**: ChromaDB is perfect!

## RAG System Preview

This is **exactly** how RAG (Retrieval Augmented Generation) works:

```python
# 1. User asks question
user_question = "How do I use NumPy for AI?"

# 2. Search knowledge base (your docs)
results = db.search_by_text(user_question, model, top_k=3)
context = results['documents'][0]  # top 3 relevant docs

# 3. Send to LLM with context (Phase 3!)
prompt = f"""
Context: {context}

Question: {user_question}

Answer based on the context above.
"""
# response = openai.chat.completions.create(...)
```

**You just built the retrieval part of RAG!** 🎉

## Next Steps

### Experiment

1. **Add more memories**:
   ```bash
   # Create new memory file
   echo "# New Learning" > .serena/memories/test.md

   # Re-run to update database
   uv run python phase1_foundations/playground/embeddings_demo/save_to_chromadb.py
   ```

2. **Try custom queries**:
   Edit `save_to_chromadb.py` and add your own queries to test

3. **Interactive search**:
   Create a simple script that accepts user input for queries

### Phase 3 Preview

In **Phase 3**, you'll build a complete RAG system:
- ✅ Document ingestion (chunking)
- ✅ Embedding generation (you know this!)
- ✅ Vector database storage (you know this!)
- ✅ Semantic retrieval (you know this!)
- 🆕 LLM integration (OpenAI/Anthropic)
- 🆕 Context management
- 🆕 Production deployment

**You're already 60% there!** 🚀

## Troubleshooting

### Database locked
```bash
# Delete and recreate
rm -rf .chromadb/
uv run python phase1_foundations/playground/embeddings_demo/save_to_chromadb.py
```

### Different model
```python
# In save_to_chromadb.py, change model:
model = SentenceTransformer('all-MiniLM-L6-v2')  # faster, smaller
model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # better, larger
```

### Memory usage
ChromaDB loads everything into memory. For large datasets (>10k docs), consider:
- Batch processing
- Cloud vector databases (Pinecone)
- Dedicated vector DB servers (Weaviate, Qdrant)

## Resources

**ChromaDB**:
- Docs: https://docs.trychroma.com/
- GitHub: https://github.com/chroma-core/chroma

**Vector Databases**:
- Pinecone: https://www.pinecone.io/
- Weaviate: https://weaviate.io/
- Qdrant: https://qdrant.tech/

**Patterns**:
- This implements the **Repository Pattern** for vector storage
- `VectorDB` class abstracts ChromaDB implementation
- Easy to swap ChromaDB for Pinecone/Weaviate later
