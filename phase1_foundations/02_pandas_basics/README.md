# Pandas Basics for AI Development

Master data manipulation with Pandas - essential for preparing documents and datasets for LLM applications.

## Why Pandas for AI?

Before you can embed documents or build RAG systems, you need to:
- **Load data** from various sources (CSV, JSON, databases)
- **Clean text** (remove noise, handle missing values)
- **Transform data** (chunk documents, extract metadata)
- **Analyze patterns** (token counts, content distribution)

**Every AI project starts with data preparation. Pandas is your tool.**

## Learning Objectives

By the end of this module, you will:
1. Load and save data from multiple formats
2. Clean and preprocess text data
3. Filter, group, and transform datasets
4. Prepare documents for embedding generation
5. Analyze text data for AI applications

## Key Concepts

### DataFrames and Series
```python
import pandas as pd

# Series: 1D array with labels
embeddings = pd.Series([0.2, 0.5, 0.8], name="embedding_values")

# DataFrame: 2D table (like Excel/SQL)
docs = pd.DataFrame({
    "id": ["doc1", "doc2", "doc3"],
    "text": ["Hello world", "AI is amazing", "Pandas rocks"],
    "tokens": [2, 3, 2]
})
```

### AI Application: Document Management
```python
# typical document dataset for RAG
documents = pd.DataFrame({
    "doc_id": ["d001", "d002", "d003"],
    "content": ["Policy text...", "Guide text...", "FAQ text..."],
    "source": ["hr", "engineering", "support"],
    "char_count": [1500, 2300, 800],
    "chunk_ready": [False, False, False]
})

# filter by source
hr_docs = documents[documents["source"] == "hr"]

# calculate token estimates (rough: chars / 4)
documents["est_tokens"] = documents["char_count"] // 4
```

## Connection to Your AI Journey

| Pandas Skill | AI Application |
|--------------|----------------|
| Load CSV/JSON | Import document datasets |
| Text cleaning | Prepare docs for embedding |
| Filtering | Select relevant documents |
| Grouping | Analyze by source/category |
| Chunking | Split docs for RAG |
| Token counting | Manage context windows |

## Module Structure

```
02_pandas_basics/
├── README.md          # This file
├── examples.py        # Runnable examples (start here!)
├── exercises.py       # Practice problems (coming soon)
└── solutions/         # Exercise solutions (coming soon)
```

## Quick Start

```bash
# run the examples
uv run python phase1_foundations/02_pandas_basics/examples.py
```

## Topics Covered

### 1. Data Loading
- CSV, JSON, Excel files
- Creating DataFrames from dictionaries
- Reading from databases (preview)

### 2. Data Exploration
- `.head()`, `.info()`, `.describe()`
- Column selection and filtering
- Understanding data types

### 3. Text Data Processing
- String operations
- Cleaning and normalization
- Handling missing values

### 4. Data Transformation
- Adding/removing columns
- Applying functions
- Grouping and aggregation

### 5. AI-Specific Operations
- Document chunking
- Token estimation
- Metadata extraction
- Batch processing preparation

## Prerequisites

- Completed [NumPy basics](../01_numpy_basics/)
- Understanding of embeddings concept
- Python fundamentals

## Next Steps

After completing this module:
1. **Phase 1**: [ML Concepts](../03_ml_concepts/) (train/test splits, features)
2. **Phase 2**: Use Pandas to load docs for [embedding](../../phase2_llm_fundamentals/)
3. **Phase 3**: Build document ingestion pipeline for [RAG](../../phase3_llm_applications/01_rag_system/)

---

**Run the examples now:**
```bash
uv run python phase1_foundations/02_pandas_basics/examples.py
```
