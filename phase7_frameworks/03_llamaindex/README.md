# Module 3: LlamaIndex

**Purpose**: Learn document-centric RAG framework optimized for knowledge retrieval

---

## Learning Objectives

- Understand LlamaIndex's data-first philosophy
- Compare to your Phase 3 RAG implementation
- Learn advanced retrieval strategies
- Build production RAG applications
- Know when LlamaIndex fits better than LangChain

---

## What is LlamaIndex?

**LlamaIndex** = RAG framework optimized for document ingestion, indexing, and querying

```
Your RAG (Phase 3):
  Documents → Chunking → Embeddings → ChromaDB → Retrieval → Generation

LlamaIndex:
  Documents → VectorStoreIndex → QueryEngine → Response
              ↑                      ↑
         (handles all)       (optimized retrieval)
```

---

## Core Philosophy

### LangChain: Tool Orchestration
"I'm building agents that use various tools, one of which is RAG"

### LlamaIndex: Data Connectors
"I'm building apps around my documents/data"

---

## Key Concepts

### 1. Data Connectors
**What it is**: 100+ integrations for loading documents

```python
from llama_index import SimpleDirectoryReader, PDFReader

# Load from directory
documents = SimpleDirectoryReader("./docs").load_data()

# Load specific formats
from llama_index.readers.notion import NotionPageReader
documents = NotionPageReader(notion_api_key).load_data()
```

### 2. Indexes
**What it is**: Data structures optimized for retrieval

```python
from llama_index import VectorStoreIndex, DocumentSummaryIndex

# Vector index (like your ChromaDB approach)
vector_index = VectorStoreIndex.from_documents(documents)

# Summary index (abstractive summaries for each doc)
summary_index = DocumentSummaryIndex.from_documents(documents)

# Tree index (hierarchical summaries)
tree_index = TreeIndex.from_documents(documents)
```

### 3. Query Engines
**What it is**: Retrieval + generation pipeline

```python
# Basic query
query_engine = index.as_query_engine()
response = query_engine.query("What is X?")

# With filters
from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(filters=[
    ExactMatchFilter(key="category", value="technical")
])
query_engine = index.as_query_engine(filters=filters)

# Streaming
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("Explain this")
for text in response.response_gen:
    print(text, end="")
```

### 4. Chat Engines
**What it is**: Query engine + conversation memory

```python
from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory
)

response = chat_engine.chat("What is X?")
response = chat_engine.chat("Tell me more")  # maintains context
```

---

## Module Structure

```
03_llamaindex/
├── README.md                   # This file
├── data_loading.py             # SimpleDirectoryReader, various loaders
├── indexing.py                 # VectorStoreIndex, TreeIndex, SummaryIndex
├── query_engines.py            # Basic, filtered, streaming queries
├── chat_engines.py             # Conversational RAG
├── advanced_retrieval.py       # Hybrid search, auto-merging retrieval
├── metadata_filtering.py       # Filter by date, category, etc.
└── migration_from_phase3.py    # Your RAG → LlamaIndex
```

---

## Comparison to Your Phase 3 RAG

### Your Implementation (Phase 3)
```python
# 1. Load documents
documents = load_docs("./docs")

# 2. Chunk
chunks = chunker.chunk_documents(documents)

# 3. Embed
embeddings = embedder.embed_all(chunks)

# 4. Store
db = ChromaDB()
db.add(chunks, embeddings)

# 5. Query
query_embedding = embedder.embed(query)
results = db.search(query_embedding, k=5)
context = assemble_context(results)

# 6. Generate
response = llm.generate(prompt_template.format(context=context, query=query))
```

### LlamaIndex Equivalent
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# 1-4: Load, chunk, embed, store (one line!)
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# 5-6: Query and generate (one line!)
query_engine = index.as_query_engine()
response = query_engine.query(query)
```

**Key Difference**: LlamaIndex abstracts chunking, embedding, storage behind `VectorStoreIndex`.

---

## Advanced Retrieval Strategies

### 1. Hybrid Search (Vector + Keyword)
```python
from llama_index.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.retrievers import QueryFusionRetriever

vector_retriever = VectorIndexRetriever(index)
bm25_retriever = BM25Retriever.from_defaults(index)

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5
)
```

### 2. Auto-Merging Retrieval
**What it is**: Retrieve small chunks, expand to parent context

```python
from llama_index.node_parser import HierarchicalNodeParser

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # parent → child hierarchy
)
index = VectorStoreIndex(nodes, node_parser=node_parser)

# Retrieves small chunks, expands to parent for context
query_engine = index.as_query_engine(
    retrieval_mode="auto_merging"
)
```

### 3. Sentence Window Retrieval
**What it is**: Retrieve sentences, expand to surrounding window

```python
from llama_index.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3  # 3 sentences before/after
)
```

---

## When to Use LlamaIndex

### ✅ Use LlamaIndex
- Primary use case is RAG
- Working with complex documents (PDFs, tables, images)
- Need advanced retrieval (hybrid, auto-merging, etc.)
- Document-heavy applications
- Want RAG-specific optimizations

### ❌ Skip LlamaIndex
- Building agents with tools (use LangChain/LangGraph)
- Simple Q&A without docs (direct LLM call)
- Need maximum control over chunking/retrieval (your Phase 3 code)
- RAG is minor part of app

---

## LlamaIndex vs LangChain for RAG

| Feature | LlamaIndex | LangChain |
|---------|------------|-----------|
| **Focus** | Document-centric RAG | General LLM orchestration |
| **Retrieval** | Advanced strategies | Basic vector search |
| **Data Connectors** | 100+ built-in | Fewer, community-driven |
| **Query Engines** | Highly optimized | Generic chains |
| **Agents** | Limited support | Strong support |
| **Multi-Agent** | Not primary use case | LangGraph specializes |

**Rule of thumb**:
- **Pure RAG app** → LlamaIndex
- **RAG + agents + tools** → LangChain
- **Complex multi-agent** → LangGraph

---

## Exercises

### Exercise 1: Migrate Phase 3 RAG
Convert your RAG pipeline to LlamaIndex.

### Exercise 2: Advanced Retrieval
Implement hybrid search (vector + BM25).

### Exercise 3: Metadata Filtering
Add date/category filters to retrieval.

### Exercise 4: Chat Engine
Build conversational RAG with memory.

---

## Resources

- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Data Connectors](https://llamahub.ai/)
- [Advanced Retrieval Guide](https://docs.llamaindex.ai/en/stable/examples/retrievers/advanced_retrieval.html)
- [Query Engines](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/index.html)

---

## Next Steps

After this module:
1. Build RAG app with LlamaIndex
2. Compare performance to your Phase 3 implementation
3. Experiment with advanced retrieval strategies
4. Move to Module 4 for framework decision framework
