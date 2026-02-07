"""
Pandas Basics for AI Development

This module covers essential Pandas operations for AI/LLM applications:
- Loading and exploring data
- Text processing and cleaning
- Document preparation for embeddings
- Batch processing patterns

Run with: uv run python phase1_foundations/02_pandas_basics/examples.py
"""

from inspect import cleandoc

import pandas as pd
import numpy as np
from typing import List, Dict

from common.demo_menu import Demo, MenuRunner


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


# =============================================================================
# PART 1: DataFrame Basics
# =============================================================================

def dataframe_creation() -> None:
    """creating DataFrames - the foundation of Pandas"""
    print_section("1. DataFrame Creation")

    # method 1: from dictionary (most common)
    docs = pd.DataFrame({
        "doc_id": ["d001", "d002", "d003", "d004"],
        "title": ["Vacation Policy", "Remote Work Guide", "Benefits FAQ", "Onboarding"],
        "content": [
            "Employees receive 15 days of paid vacation per year.",
            "Remote work is allowed 3 days per week with manager approval.",
            "Health insurance enrollment opens in November each year.",
            "New employees complete orientation in their first week."
        ],
        "category": ["hr", "hr", "hr", "hr"],
        "last_updated": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-01-05"]
    })

    print("document DataFrame:")
    print(docs)
    print(f"\nshape: {docs.shape} (rows, columns)")
    print(f"columns: {list(docs.columns)}")

    # method 2: from list of dictionaries
    embeddings_data = [
        {"doc_id": "d001", "embedding": [0.1, 0.2, 0.3]},
        {"doc_id": "d002", "embedding": [0.4, 0.5, 0.6]},
    ]
    embeddings_df = pd.DataFrame(embeddings_data)
    print(f"\nembeddings DataFrame from list of dicts:")
    print(embeddings_df)


def dataframe_exploration() -> None:
    """exploring DataFrames - understand your data"""
    print_section("2. DataFrame Exploration")

    # create sample document dataset
    np.random.seed(42)
    n_docs = 100

    docs = pd.DataFrame({
        "doc_id": [f"doc_{i:03d}" for i in range(n_docs)],
        "content": [f"This is document {i} with some sample content." for i in range(n_docs)],
        "category": np.random.choice(["hr", "engineering", "sales", "support"], n_docs),
        "char_count": np.random.randint(500, 5000, n_docs),
        "created_date": pd.date_range("2024-01-01", periods=n_docs, freq="D")
    })

    # basic exploration
    print("first 5 rows (.head()):")
    print(docs.head())

    print("\ndata types (.dtypes):")
    print(docs.dtypes)

    print("\nbasic statistics (.describe()):")
    print(docs.describe())

    print("\ninfo summary (.info()):")
    docs.info()

    # value counts - useful for understanding categories
    print("\ncategory distribution:")
    print(docs["category"].value_counts())


# =============================================================================
# PART 2: Selection and Filtering
# =============================================================================

def selection_and_filtering() -> None:
    """selecting and filtering data - essential for document processing"""
    print_section("3. Selection and Filtering")

    docs = pd.DataFrame({
        "doc_id": ["d001", "d002", "d003", "d004", "d005"],
        "title": ["Vacation Policy", "API Guide", "Benefits FAQ", "Code Standards", "Remote Work"],
        "content": [
            "Employees receive 15 days of paid vacation.",
            "Use REST endpoints for all API calls.",
            "Health insurance enrollment in November.",
            "Follow PEP8 style guidelines for Python.",
            "Remote work allowed 3 days per week."
        ],
        "category": ["hr", "engineering", "hr", "engineering", "hr"],
        "char_count": [150, 200, 180, 220, 160],
        "priority": [1, 2, 1, 3, 2]
    })

    print("original DataFrame:")
    print(docs)

    # select single column (returns Series)
    print("\nselect 'title' column:")
    print(docs["title"])

    # select multiple columns (returns DataFrame)
    print("\nselect multiple columns:")
    print(docs[["doc_id", "title", "category"]])

    # filter rows by condition
    print("\nfilter: category == 'hr':")
    hr_docs = docs[docs["category"] == "hr"]
    print(hr_docs)

    # multiple conditions (use & for AND, | for OR)
    print("\nfilter: hr docs with char_count > 155:")
    filtered = docs[(docs["category"] == "hr") & (docs["char_count"] > 155)]
    print(filtered)

    # filter with isin() - useful for multiple categories
    print("\nfilter: category in ['hr', 'engineering']:")
    print(docs[docs["category"].isin(["hr", "engineering"])])

    # string contains - useful for text search
    print("\nfilter: content contains 'work':")
    print(docs[docs["content"].str.contains("work", case=False)])


# =============================================================================
# PART 3: Text Processing for AI
# =============================================================================

def text_processing() -> None:
    """text processing - preparing documents for embeddings"""
    print_section("4. Text Processing for AI")

    docs = pd.DataFrame({
        "doc_id": ["d001", "d002", "d003"],
        "raw_content": [
            "  VACATION POLICY\n\nEmployees receive 15 days...  ",
            "API Guide - v2.0\n\n\nUse REST endpoints...",
            "Benefits FAQ\n\nQ: When is enrollment?\nA: November"
        ]
    })

    print("raw content (notice whitespace, newlines, case):")
    for i, row in docs.iterrows():
        print(f"  {row['doc_id']}: {repr(row['raw_content'][:50])}")

    # text cleaning pipeline
    docs["clean_content"] = (
        docs["raw_content"]
        .str.strip()                    # remove leading/trailing whitespace
        .str.lower()                    # lowercase
        .str.replace(r"\n+", " ", regex=True)  # newlines to spaces
        .str.replace(r"\s+", " ", regex=True)  # multiple spaces to single
    )

    print("\ncleaned content:")
    for i, row in docs.iterrows():
        print(f"  {row['doc_id']}: {row['clean_content'][:50]}...")

    # add useful metadata columns
    docs["char_count"] = docs["clean_content"].str.len()
    docs["word_count"] = docs["clean_content"].str.split().str.len()
    docs["est_tokens"] = docs["char_count"] // 4  # rough token estimate

    print("\nwith metadata:")
    print(docs[["doc_id", "char_count", "word_count", "est_tokens"]])

    # AI insight: token estimation matters for context windows
    print("\n💡 AI Insight:")
    print("   GPT-4 context: ~128K tokens")
    print("   Claude context: ~200K tokens")
    print("   Typical chunk size: 500-1000 tokens")
    print(f"   Your docs: {docs['est_tokens'].sum()} estimated tokens total")


def document_chunking() -> None:
    """document chunking - splitting docs for RAG"""
    print_section("5. Document Chunking for RAG")

    # simulate a long document
    long_doc = cleandoc("""
        Chapter 1: Introduction to AI

        Artificial Intelligence (AI) is transforming how we build software.
        Machine learning models can now understand natural language, generate
        code, and assist with complex tasks.

        Chapter 2: Embeddings

        Embeddings convert text into numerical vectors that capture semantic
        meaning. Similar concepts have similar embeddings. This enables
        semantic search and RAG systems.

        Chapter 3: RAG Systems

        Retrieval Augmented Generation combines search with LLM generation.
        First, relevant documents are retrieved. Then, they're used as
        context for the LLM to generate accurate answers.

        Chapter 4: Production

        Deploying AI systems requires careful attention to cost, latency,
        and reliability. Caching, monitoring, and optimization are essential.
    """)

    print(f"original document: {len(long_doc)} characters")

    # simple chunking by character count
    def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """chunk text with overlap for context preservation"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    chunks = chunk_text(long_doc, chunk_size=300, overlap=50)

    print(f"\nchunked into {len(chunks)} pieces:")
    for i, chunk in enumerate(chunks):
        print(f"\n  chunk {i+1} ({len(chunk)} chars):")
        print(f"    '{chunk[:60]}...'")

    # create DataFrame for chunked documents
    chunks_df = pd.DataFrame({
        "chunk_id": [f"chunk_{i}" for i in range(len(chunks))],
        "content": chunks,
        "char_count": [len(c) for c in chunks],
        "source_doc": "ai_guide.md"
    })

    print("\nchunks DataFrame (ready for embedding):")
    print(chunks_df[["chunk_id", "char_count", "source_doc"]])

    print("\n💡 RAG Insight:")
    print("   - Smaller chunks = more precise retrieval")
    print("   - Overlap preserves context at boundaries")
    print("   - Typical chunk: 500-1000 tokens")


# =============================================================================
# PART 4: Data Transformation
# =============================================================================

def data_transformation() -> None:
    """data transformation - preparing for batch processing"""
    print_section("6. Data Transformation")

    docs = pd.DataFrame({
        "doc_id": ["d001", "d002", "d003", "d004"],
        "title": ["Vacation", "Remote Work", "Benefits", "Onboarding"],
        "content": [
            "15 days vacation per year",
            "3 days remote per week",
            "Health insurance in November",
            "Orientation in first week"
        ],
        "category": ["hr", "hr", "hr", "hr"],
        "views": [150, 230, 180, 95]
    })

    print("original DataFrame:")
    print(docs)

    # add new columns
    docs["content_length"] = docs["content"].str.len()
    docs["popularity"] = docs["views"].apply(lambda x: "high" if x > 150 else "low")

    print("\nwith new columns:")
    print(docs)

    # apply custom function
    def create_search_text(row) -> str:
        """combine title and content for search indexing"""
        return f"{row['title']}: {row['content']}"

    docs["search_text"] = docs.apply(create_search_text, axis=1)

    print("\nwith search_text column:")
    print(docs[["doc_id", "search_text"]])

    # groupby - analyze by category
    print("\ngroupby category - aggregations:")
    category_stats = docs.groupby("category").agg({
        "views": ["sum", "mean"],
        "content_length": "mean",
        "doc_id": "count"
    })
    print(category_stats)


def batch_processing_pattern() -> None:
    """batch processing - pattern for embedding generation"""
    print_section("7. Batch Processing Pattern")

    # simulate document dataset
    docs = pd.DataFrame({
        "doc_id": [f"doc_{i:03d}" for i in range(10)],
        "content": [f"Document {i} content for processing" for i in range(10)],
        "processed": [False] * 10
    })

    print("documents to process:")
    print(docs)

    # batch processing pattern (like embedding generation)
    batch_size = 3

    def mock_generate_embeddings(texts: List[str]) -> List[List[float]]:
        """simulate embedding generation"""
        return [[0.1 * i, 0.2 * i, 0.3 * i] for i, _ in enumerate(texts)]

    all_embeddings = []

    print(f"\nprocessing in batches of {batch_size}:")
    for i in range(0, len(docs), batch_size):
        batch = docs.iloc[i:i+batch_size]
        texts = batch["content"].tolist()

        # generate embeddings for batch
        embeddings = mock_generate_embeddings(texts)
        all_embeddings.extend(embeddings)

        print(f"  batch {i//batch_size + 1}: processed {len(texts)} documents")

    # add embeddings to DataFrame
    docs["embedding"] = all_embeddings
    docs["processed"] = True

    print("\nprocessed DataFrame:")
    print(docs[["doc_id", "processed", "embedding"]])

    print("\n💡 Batch Processing Insight:")
    print("   - Process in batches to manage memory")
    print("   - Typical batch size: 32-128 documents")
    print("   - Can resume from last processed if interrupted")


# =============================================================================
# PART 5: Loading and Saving Data
# =============================================================================

def loading_and_saving() -> None:
    """loading and saving - real-world data sources"""
    print_section("8. Loading and Saving Data")

    # create sample data
    docs = pd.DataFrame({
        "doc_id": ["d001", "d002", "d003"],
        "title": ["Policy A", "Guide B", "FAQ C"],
        "content": ["Content A", "Content B", "Content C"],
        "tokens": [100, 150, 80]
    })

    # save to CSV
    csv_path = "/tmp/sample_docs.csv"
    docs.to_csv(csv_path, index=False)
    print(f"saved to CSV: {csv_path}")

    # load from CSV
    loaded_csv = pd.read_csv(csv_path)
    print("\nloaded from CSV:")
    print(loaded_csv)

    # save to JSON (useful for nested data)
    json_path = "/tmp/sample_docs.json"
    docs.to_json(json_path, orient="records", indent=2)
    print(f"\nsaved to JSON: {json_path}")

    # load from JSON
    loaded_json = pd.read_json(json_path)
    print("\nloaded from JSON:")
    print(loaded_json)

    # JSON Lines format (better for streaming/large files)
    jsonl_path = "/tmp/sample_docs.jsonl"
    docs.to_json(jsonl_path, orient="records", lines=True)
    print(f"\nsaved to JSON Lines: {jsonl_path}")

    print("\n💡 Format Recommendations:")
    print("   - CSV: simple tabular data, human-readable")
    print("   - JSON: nested data, API responses")
    print("   - JSONL: streaming, large datasets")
    print("   - Parquet: large datasets, preserves types (Phase 5)")


# =============================================================================
# PART 6: AI Application Example
# =============================================================================

def ai_application_example() -> None:
    """complete example: prepare documents for RAG"""
    print_section("9. AI Application: Document Preparation Pipeline")

    # simulate raw documents (e.g., from database or files)
    raw_docs = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "title": [
            "Employee Handbook - Vacation",
            "Engineering Standards",
            "Customer Support Guide",
            "Remote Work Policy",
            "Benefits Overview"
        ],
        "body": [
            "  VACATION POLICY\n\nFull-time employees receive 15 days...\n\n",
            "CODE STANDARDS\n\nFollow PEP8 for Python. Use type hints...",
            "SUPPORT GUIDE\n\nAlways greet customers warmly. Listen first...",
            "REMOTE WORK\n\n3 days per week allowed. Must be available...",
            "BENEFITS\n\nHealth, dental, vision insurance. 401k matching..."
        ],
        "department": ["hr", "engineering", "support", "hr", "hr"],
        "last_modified": ["2024-01-15", "2024-02-20", "2024-01-10", "2024-03-01", "2024-02-15"]
    })

    print("raw documents:")
    print(raw_docs[["id", "title", "department"]])

    # step 1: clean content
    print("\nstep 1: cleaning content...")
    raw_docs["clean_content"] = (
        raw_docs["body"]
        .str.strip()
        .str.lower()
        .str.replace(r"\n+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    # step 2: add metadata
    print("step 2: adding metadata...")
    raw_docs["char_count"] = raw_docs["clean_content"].str.len()
    raw_docs["word_count"] = raw_docs["clean_content"].str.split().str.len()
    raw_docs["est_tokens"] = raw_docs["char_count"] // 4

    # step 3: create search-friendly format
    print("step 3: creating search format...")
    raw_docs["search_text"] = raw_docs["title"] + " | " + raw_docs["clean_content"]

    # step 4: prepare final DataFrame for embedding
    print("step 4: preparing for embedding...")
    embedding_ready = raw_docs[[
        "id", "title", "search_text", "department",
        "char_count", "est_tokens", "last_modified"
    ]].copy()
    embedding_ready.columns = [
        "doc_id", "title", "content", "source",
        "char_count", "est_tokens", "updated"
    ]

    print("\n✅ documents ready for embedding:")
    print(embedding_ready)

    print("\nsummary statistics:")
    print(f"  total documents: {len(embedding_ready)}")
    print(f"  total characters: {embedding_ready['char_count'].sum()}")
    print(f"  estimated tokens: {embedding_ready['est_tokens'].sum()}")
    print(f"  departments: {embedding_ready['source'].unique().tolist()}")

    print("\n💡 Next Steps:")
    print("   1. Generate embeddings with sentence-transformers")
    print("   2. Store in ChromaDB with metadata")
    print("   3. Build semantic search (you've done this!)")
    print("   4. Add LLM generation = complete RAG system")


# =============================================================================
# Main
# =============================================================================


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "DataFrame creation", "dataframe creation", dataframe_creation),
    Demo("2", "DataFrame exploration", "dataframe exploration", dataframe_exploration),
    Demo("3", "selection and filtering", "selection and filtering", selection_and_filtering),
    Demo("4", "text processing", "text processing", text_processing),
    Demo("5", "document chunking", "document chunking", document_chunking),
    Demo("6", "data transformation", "data transformation", data_transformation),
    Demo("7", "batch processing", "batch processing", batch_processing_pattern),
    Demo("8", "loading and saving", "loading and saving", loading_and_saving),
    Demo("9", "AI application example", "ai application example", ai_application_example),
]

# endregion

def main() -> None:
    """interactive demo runner"""
    
    runner = MenuRunner(DEMOS, title="Pandas Basics - Examples")
    runner.run()


if __name__ == "__main__":
    main()
