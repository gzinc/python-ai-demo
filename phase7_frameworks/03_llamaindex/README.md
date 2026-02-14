# LlamaIndex - Data Framework for LLM Applications

**Status**: ✅ Reference Implementation

## Overview

LlamaIndex (formerly GPT Index) is a data framework designed specifically for LLM applications. While LangChain focuses on chains and agents, LlamaIndex specializes in **data ingestion, indexing, and retrieval** - making it the go-to choice for RAG systems.

## Key Differentiators

**LlamaIndex vs LangChain:**
- 🎯 **Focus**: Data-centric RAG vs general LLM orchestration
- 📚 **Strength**: Best-in-class indexing vs broad ecosystem
- 🔍 **Retrieval**: Advanced strategies vs basic retrieval
- 📊 **Use Case**: Document Q&A, knowledge bases vs chatbots, agents

**Why LlamaIndex:**
- ✅ Sophisticated indexing strategies (tree, graph, list)
- ✅ Advanced retrieval (reranking, hybrid search, query transformation)
- ✅ 100+ data connectors
- ✅ Production RAG optimizations
- ✅ Works with **both cloud and local models**

## Module Structure

### 01_basic_indexing.py
Core LlamaIndex concepts and patterns:
- ✅ In-memory document indexing
- ✅ Loading from directories
- ✅ Custom node parsing (chunk sizes)
- ✅ Metadata filtering
- ✅ Response synthesis modes
- ✅ Streaming responses

**Run**: `uv run python phase7_frameworks/03_llamaindex/01_basic_indexing.py`

### 02_local_setup.py
**100% Privacy-Friendly Local RAG:**
- ✅ Local LLM via Ollama (llama3.1)
- ✅ Local embeddings (HuggingFace BGE)
- ✅ Local vector store (ChromaDB)
- ✅ Persistent storage
- ✅ Local chat engine
- ✅ Local vs Cloud comparison

**No API keys required!**

**Prerequisites:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.1

# Install dependencies
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install chromadb
```

**Run**: `uv run python phase7_frameworks/03_llamaindex/02_local_setup.py`

### 03_advanced_rag.py
Production RAG patterns and optimization:
- ✅ Custom retrieval parameters (top-k)
- ✅ Query transformation strategies
- ✅ Response synthesis modes
- ✅ Retrieval quality evaluation
- ✅ Context window management
- ✅ Production best practices checklist

**Run**: `uv run python phase7_frameworks/03_llamaindex/03_advanced_rag.py`

## Quick Start

### Cloud Setup (OpenAI)
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Create index (uses OpenAI by default)
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

### Local Setup (Ollama - No API Key!)
```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure local models
Settings.llm = Ollama(model="llama3.1")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create index
documents = [Document(text="Your content here")]
index = VectorStoreIndex.from_documents(documents)

# Query locally!
response = index.as_query_engine().query("Your question")
```

## Key Concepts

### Index Types
- **VectorStoreIndex**: Semantic search (most common)
- **ListIndex**: Linear scan (small datasets)
- **TreeIndex**: Hierarchical summarization
- **KeywordTableIndex**: Keyword-based retrieval

### Retrieval Strategies
- **Vector Search**: Embedding similarity
- **Keyword Search**: Traditional text matching
- **Hybrid Search**: Combine vector + keyword
- **Metadata Filtering**: Filter by doc properties

### Query Engines
- **Standard**: Basic retrieval + synthesis
- **Chat**: Conversational with memory
- **Streaming**: Real-time output
- **Custom**: Full control over pipeline

## Dependencies

**Core:**
```bash
pip install llama-index
```

**Cloud (OpenAI):**
```bash
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
```

**Local (Ollama):**
```bash
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install sentence-transformers
pip install chromadb
```

## Cost Comparison

### Cloud (OpenAI + Pinecone)
- 💰 LLM: ~$0.01-0.06 per 1K tokens
- 💰 Embeddings: ~$0.0001 per 1K tokens
- 💰 Vector DB: ~$70/month
- **Total: $100-500+/month**

### Local (Ollama + HuggingFace + ChromaDB)
- ✅ LLM: **FREE**
- ✅ Embeddings: **FREE**
- ✅ Vector DB: **FREE**
- **Total: $0/month** (only electricity)

**Trade-off**: Local requires good hardware (16GB+ RAM, GPU recommended)

## Production Checklist

- [ ] Chunk size: 200-512 tokens
- [ ] Chunk overlap: 10-20%
- [ ] Top-K: 3-5 chunks
- [ ] Reranking: Enable for better quality
- [ ] Caching: Cache embeddings and queries
- [ ] Monitoring: Track latency and accuracy
- [ ] Fallback: Handle no-results gracefully
- [ ] Citations: Include source references

## Integration Examples

### With LangChain
```python
from llama_index.core import VectorStoreIndex
from langchain.chains import RetrievalQA

# Use LlamaIndex for retrieval
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

# Use LangChain for orchestration
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

### With LangGraph
```python
# Use LlamaIndex as a tool in LangGraph agent
def search_knowledge_base(query: str) -> str:
    """search using LlamaIndex"""
    return index.as_query_engine().query(query).response

# Add as tool to LangGraph
tools = [search_knowledge_base]
```

## When to Use LlamaIndex

**✅ Use LlamaIndex when:**
- Building RAG systems
- Querying large document sets
- Need advanced retrieval strategies
- Want production RAG optimizations
- Privacy is critical (local setup)

**❌ Consider alternatives when:**
- Need agent orchestration (use LangGraph)
- Simple chatbot (use LangChain)
- No retrieval needed (use LLM directly)

## Learning Path

1. **Start**: 01_basic_indexing.py (core concepts)
2. **Privacy**: 02_local_setup.py (local models)
3. **Advanced**: 03_advanced_rag.py (production patterns)
4. **Combine**: Use with LangChain/LangGraph

## Next Steps

- Combine LlamaIndex (RAG) + LangGraph (agents)
- Try local setup for privacy-sensitive projects
- Benchmark local vs cloud for your use case
- Implement production RAG checklist

## Resources

- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Ollama Models](https://ollama.com/library)
- [HuggingFace Embeddings](https://huggingface.co/spaces/mteb/leaderboard)
