# RAG (Retrieval-Augmented Generation) Module

**Purpose**: Learn to build RAG systems that combine document retrieval with LLM generation for question answering

---

## Learning Objectives

By the end of this module, you will:
1. **Document Loading**: Load PDFs, text files, web pages, and structured data
2. **Text Splitting**: Chunk documents for optimal retrieval
3. **Embeddings**: Convert text to semantic vectors
4. **Vector Stores**: Store and search embeddings efficiently
5. **Retrievers**: Implement similarity search and hybrid retrieval
6. **RAG Chains**: Build end-to-end question-answering systems

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** enhances LLMs by retrieving relevant context from external documents:

```
Without RAG:
User: "What did the CEO say in Q3 earnings?"
LLM:  "I don't have access to your company's earnings reports"

With RAG:
1. Retrieve relevant sections from earnings report
2. Include in LLM context
3. LLM:  "The CEO mentioned 15% revenue growth in Q3..."
```

### Why RAG?

**Problems RAG Solves:**
- ❌ LLMs have knowledge cutoff dates
- ❌ LLMs can't access private/proprietary data
- ❌ LLMs hallucinate facts
- ❌ Fine-tuning is expensive and slow

**RAG Benefits:**
- ✅ Real-time access to current information
- ✅ Works with private documents
- ✅ Grounded in actual sources (reduces hallucinations)
- ✅ Cost-effective vs fine-tuning
- ✅ Easy to update knowledge (just add documents)

---

## RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INGESTION PHASE (One-time setup)                        │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│     │Documents │ -> │ Chunking │ -> │Embeddings│            │
│     └──────────┘    └──────────┘    └──────────┘            │
│            │              │                │                │
│            v              v                v                │
│     ┌─────────────────────────────────────────┐             │
│     │        Vector Store (ChromaDB)          │             │
│     └─────────────────────────────────────────┘             │
│                                                             │
│  2. QUERY PHASE (Runtime)                                   │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│     │  Query   │ -> │ Retrieve │ -> │   LLM    │            │
│     └──────────┘    └──────────┘    └──────────┘            │
│                          │                │                 │
│                          v                v                 │
│                    [Top K docs]     [Answer + Sources]      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Document Loaders

Load data from various sources:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,          # PDF files
    TextLoader,           # Plain text
    CSVLoader,            # CSV data
    WebBaseLoader,        # Web pages
    DirectoryLoader,      # Multiple files
)

# Load PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Each document has:
# - page_content: The text
# - metadata: {'source': 'document.pdf', 'page': 0}
```

**Common Loaders:**

| Loader | Use Case | Metadata |
|--------|----------|----------|
| `PyPDFLoader` | PDF files | source, page |
| `TextLoader` | .txt files | source |
| `CSVLoader` | Tabular data | source, row |
| `WebBaseLoader` | Web scraping | source, title |
| `UnstructuredFileLoader` | Word, HTML, more | source, filetype |

---

### 2. Text Splitters

Break documents into chunks for embedding:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Max characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]  # Split hierarchy
)

chunks = splitter.split_documents(documents)
```

**Why Chunk?**
- Embeddings have token limits (e.g., 8191 tokens)
- Smaller chunks = more precise retrieval
- Overlap preserves context across boundaries

**Chunking Strategies:**

| Strategy | Approach | Best For |
|----------|----------|----------|
| **Character Split** | Fixed character count | Simple, consistent chunks |
| **Recursive Split** | Try separators in order | General purpose (default) |
| **Token Split** | Split by token count | Embedding model limits |
| **Semantic Split** | Split by meaning | Preserving context |

**Chunk Size Trade-offs:**

| Size | Precision | Context | Speed |
|------|-----------|---------|-------|
| Small (200-500) | ✅ High | ❌ Limited | ✅ Fast |
| Medium (500-1000) | ✓ Good | ✓ Good | ✓ Moderate |
| Large (1000-2000) | ❌ Lower | ✅ Rich | ❌ Slower |

---

### 3. Embeddings

Convert text to semantic vectors:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 1536 dimensions
)

# Embed a single text
vector = embeddings.embed_query("What is machine learning?")
# Returns: [0.123, -0.456, 0.789, ...] (1536 numbers)

# Embed multiple documents
vectors = embeddings.embed_documents([
    "Machine learning is...",
    "Deep learning is..."
])
```

**How Embeddings Work:**
1. Text → Neural network → High-dimensional vector
2. Similar meanings → Similar vectors
3. Measure similarity with cosine distance

**Popular Embedding Models:**

| Model | Dimensions | Cost | Quality |
|-------|------------|------|---------|
| OpenAI text-embedding-3-small | 1536 | $0.02/1M tokens | Good |
| OpenAI text-embedding-3-large | 3072 | $0.13/1M tokens | Best |
| HuggingFace all-MiniLM-L6-v2 | 384 | Free | Decent |

---

### 4. Vector Stores

Store and search embeddings efficiently:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"  # Save to disk
)

# Similarity search
results = vectorstore.similarity_search(
    "What is the revenue?",
    k=4  # Return top 4 most similar chunks
)
```

**Vector Store Options:**

| Vector Store | Type | Best For |
|--------------|------|----------|
| **ChromaDB** | Local | Development, small datasets |
| **FAISS** | In-memory | Fast local search |
| **Pinecone** | Cloud | Production, large scale |
| **Weaviate** | Cloud/Local | Advanced features |
| **Qdrant** | Cloud/Local | Production, filtering |

**Search Methods:**

```python
# 1. Similarity search (most common)
docs = vectorstore.similarity_search("query", k=4)

# 2. Similarity search with scores
docs_with_scores = vectorstore.similarity_search_with_score("query", k=4)
# Returns: [(doc, 0.85), (doc, 0.72), ...]

# 3. Max Marginal Relevance (diversity)
docs = vectorstore.max_marginal_relevance_search("query", k=4)
# Balances relevance with diversity
```

---

### 5. Retrievers

Wrap vector stores with retrieval strategies:

```python
# Basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Multi-query retriever (generates multiple queries)
from langchain.retrievers import MultiQueryRetriever
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# Contextual compression (rerank and compress)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

**Retrieval Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Similarity** | Cosine similarity | Default, works well |
| **MMR** | Max Marginal Relevance | Need diverse results |
| **Similarity Score** | Filter by threshold | Quality control |
| **Multi-Query** | Generate query variations | Complex questions |
| **Compression** | Rerank + extract relevant parts | Long documents |

---

### 6. RAG Chains (Modern LCEL Pattern) ✅

**Modern Approach** (LangChain 1.0+): Use LCEL (LangChain Expression Language) for composable RAG chains:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# define prompt
template = """Answer based on the following context:

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# build chain with pipe operators
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# use chain
answer = rag_chain.invoke("What is the revenue?")
```

**LCEL Benefits:**
- ✅ Composable with `|` operator (like Unix pipes)
- ✅ Streaming support (see responses as they generate)
- ✅ Async execution (better performance)
- ✅ Better error handling and debugging
- ✅ Type safety and validation

### Advanced RAG Patterns

**With Source Citation:**
```python
from langchain_core.runnables import RunnableParallel

rag_chain_with_source = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
).assign(
    answer=lambda x: (
        {"context": format_docs(x["context"]), "question": x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
)

# returns both answer and source documents
result = rag_chain_with_source.invoke("What is the revenue?")
# {'context': [doc1, doc2], 'question': '...', 'answer': '...'}
```

**With Streaming:**
```python
# stream answer token by token
for chunk in rag_chain.stream("What is the revenue?"):
    print(chunk, end="", flush=True)
```

---

## Legacy RAG Patterns ⚠️ DEPRECATED

> **⚠️ Deprecation Notice**: The following patterns are deprecated as of LangChain 1.0.
> They are shown for reference when working with legacy codebases.
> Use the modern LCEL patterns above for new projects.

### RetrievalQA (Legacy)

**Old Pattern** (LangChain < 1.0):
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # how to combine docs
    retriever=retriever,
    return_source_documents=True  # include sources
)

result = qa_chain.invoke("What is the main topic?")
# {
#   'query': 'What is the main topic?',
#   'result': 'The main topic is...',
#   'source_documents': [doc1, doc2, doc3]
# }
```

**Legacy Chain Types:**

| Type | Description | Token Usage | Best For |
|------|-------------|-------------|----------|
| **stuff** | Put all docs in prompt | High | Small doc sets |
| **map_reduce** | Summarize each, combine | Medium | Large doc sets |
| **refine** | Iteratively refine answer | High | Detailed answers |
| **map_rerank** | Score each, pick best | Medium | Multiple sources |

**Modern Alternative** (LangChain 1.0+):
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# modern LCEL pattern with source documents
rag_chain = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
).assign(
    answer=lambda x: (
        {"context": format_docs(x["context"]), "question": x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
)

result = rag_chain.invoke("What is the main topic?")
# {'context': [doc1, doc2], 'question': '...', 'answer': '...'}
```

**Why deprecated?**
- `RetrievalQA` is less flexible than LCEL
- LCEL supports streaming (RetrievalQA doesn't)
- LCEL has better type safety and error handling
- LCEL is the future direction of LangChain

### ConversationalRetrievalChain (Legacy)

**Old Pattern**:
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

result = qa({"question": "What is the revenue?"})
```

**Modern Alternative**:
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# create RAG chain (see modern pattern above)
rag_chain = (...)  # LCEL RAG chain

# add message history
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# use with session ID
conversational_rag.invoke(
    {"question": "What is the revenue?"},
    config={"configurable": {"session_id": "user123"}}
)
```

---

## RAG Evaluation

### Quality Metrics

**Retrieval Quality:**
- **Precision**: Relevant docs / Retrieved docs
- **Recall**: Retrieved relevant docs / All relevant docs
- **MRR (Mean Reciprocal Rank)**: Position of first relevant doc

**Generation Quality:**
- **Faithfulness**: Answer grounded in retrieved docs?
- **Answer Relevance**: Addresses the question?
- **Context Relevance**: Retrieved docs relevant to question?

### Testing Strategies

```python
# Test retrieval
test_queries = [
    "What is the revenue?",
    "Who is the CEO?",
    "When was the company founded?"
]

for query in test_queries:
    docs = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} docs")
    print(f"Top result: {docs[0].page_content[:100]}...")
```

---

## Advanced RAG Patterns

### 1. Hybrid Search (Keyword + Semantic)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (keyword search)
bm25_retriever = BM25Retriever.from_documents(documents)

# Semantic search
semantic_retriever = vectorstore.as_retriever()

# Combine both
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7]  # 30% keyword, 70% semantic
)
```

### 2. Parent Document Retriever

Store small chunks for retrieval, but return larger parent documents:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=small_splitter,   # For embedding
    parent_splitter=large_splitter,  # For context
)
```

### 3. Self-Query Retriever

LLM extracts filters from natural language:

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_fields = [
    AttributeInfo(name="source", description="Document source", type="string"),
    AttributeInfo(name="page", description="Page number", type="integer"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Research papers",
    metadata_field_info=metadata_fields
)

# Query: "Papers from 2023 about AI"
# Automatically filters: source_date == 2023 AND topic == "AI"
```

---

## Production Best Practices

### Document Processing

```python
# 1. Clean documents
def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    return text

# 2. Add metadata
for doc in documents:
    doc.metadata["processed_date"] = datetime.now().isoformat()
    doc.metadata["source_type"] = "pdf"

# 3. Deduplicate chunks
unique_chunks = list({chunk.page_content: chunk for chunk in chunks}.values())
```

### Chunking Strategy

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Production-ready splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # ~250 tokens
    chunk_overlap=200,        # 20% overlap
    length_function=len,
    separators=[
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        " ",     # Words
        ""       # Characters
    ],
    keep_separator=True  # Preserve context
)
```

### Embedding Optimization

```python
# Batch embedding for efficiency
from typing import List

def batch_embed(texts: List[str], batch_size: int = 100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def safe_embed(text: str):
    """retry embedding on failures"""
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        print(f"Embedding failed: {e}")
        raise
```

---

## Common Pitfalls

### ❌ Problem: Retrieved docs not relevant

**Solutions:**
- Improve chunking (smaller chunks, better boundaries)
- Use better embeddings (upgrade model)
- Add metadata filtering
- Try hybrid search (keyword + semantic)

### ❌ Problem: Answer not faithful to sources

**Solutions:**
- Use explicit prompt: "Answer ONLY from context"
- Add citations to prompt
- Use smaller chunk size (less noise)
- Implement answer verification step

### ❌ Problem: Slow retrieval

**Solutions:**
- Use faster vector store (FAISS vs ChromaDB)
- Reduce `k` (fewer docs retrieved)
- Cache frequently queried results
- Use approximate nearest neighbor (ANN)

### ❌ Problem: High costs

**Solutions:**
- Use cheaper embedding model (3-small vs 3-large)
- Cache embeddings (don't re-embed same text)
- Compress retrieved docs before LLM
- Use smaller LLM for generation

---

## Phase 3 vs LangChain RAG

### Your Implementation (Phase 3)

```python
# phase3_llm_applications/04_rag/basic_rag.py
class SimpleRAG:
    def __init__(self, documents: list[str]):
        self.chunks = self._chunk_documents(documents)
        self.embeddings = self._embed_chunks()

    def query(self, question: str) -> str:
        # Find similar chunks
        query_embedding = embed(question)
        similar = find_top_k(query_embedding, self.embeddings)

        # Generate answer
        context = "\n".join([self.chunks[i] for i in similar])
        return llm.generate(f"Context: {context}\n\nQuestion: {question}")
```

### LangChain Equivalent

```python
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
answer = qa_chain.invoke(question)
```

**Key Differences:**
- LangChain: Production-ready vector stores, persistence, metadata
- Your Implementation: Educational, full control, understanding

---

## Exercises

1. **Basic RAG**: Build QA system over a PDF document
2. **Chunk Comparison**: Test different chunk sizes and compare retrieval quality
3. **Multi-Document**: Create RAG system across multiple PDFs
4. **Hybrid Search**: Combine keyword and semantic search
5. **Source Attribution**: Add citations to answers

---

## Key Takeaways

✅ **RAG = Retrieval + Generation**: Combine document search with LLM reasoning
✅ **Chunking Matters**: Size and overlap significantly impact quality
✅ **Embeddings are Key**: Better embeddings = better retrieval
✅ **Trade-offs Exist**: Precision vs context, speed vs quality, cost vs performance
✅ **Evaluation Critical**: Test retrieval and generation separately

---

## Next Steps

After mastering RAG:
- **Agents & Tools**: RAG as a tool for AI agents
- **LangGraph**: Complex RAG workflows with state management
- **Production**: Optimize for scale, cost, and latency

---

## Run Examples

**📊 Visual Learning**: All practical demos include comprehensive ASCII diagrams showing RAG workflows, retrieval processes, and chain execution.

```bash
# Conceptual demos (no API key required)
uv run python -m phase7_frameworks.01_langchain_basics.05_rag.concepts

# Practical demos (requires OPENAI_API_KEY)
uv run python -m phase7_frameworks.01_langchain_basics.05_rag.practical
```

---

## 🎨 Visual Learning Features

**All 8 practical demos now include comprehensive ASCII diagrams** to visualize RAG concepts:

1. **Basic RAG Pipeline**: 7-step flow from query → embedding → retrieval → generation
2. **Text Chunking**: Visual representation of chunking with overlap
3. **Similarity Search Methods**: Three search approaches (basic, with scores, MMR) compared
4. **Metadata Filtering**: Filtered vs unfiltered search comparison
5. **Document Loading**: Complete file-to-RAG pipeline (8 steps)
6. **Custom Retriever**: Three configuration patterns with code examples
7. **Multi-Query RAG**: Complete 9-step pipeline including answer generation
8. **Chain Comparison**: Four RAG chain architectures with trade-offs

**Educational Benefit**: Visual diagrams make complex RAG workflows easier to understand, showing data flow and component interactions at each step.
