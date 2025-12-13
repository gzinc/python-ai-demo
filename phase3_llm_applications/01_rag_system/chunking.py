"""
Chunking Module - Document splitting strategies for RAG

This module provides different strategies for splitting documents into chunks
optimized for embedding and retrieval.

Module Structure:
- schemas/         → Document, Chunk, RetrievalResult (data classes)
- chunking.py     → Chunking strategies (this file)
- retrieval.py    → Retriever class
- rag_pipeline.py → RAGPipeline orchestrator
- examples.py     → Demo functions

Chunking Strategies:
┌─────────────────────────────────────────────────────────────────┐
│                    Chunking Approaches                          │
├─────────────────────────────────────────────────────────────────┤
│  Fixed Size      │  Split by character/token count             │
│  Paragraph       │  Split by paragraph boundaries              │
│  Sentence        │  Split by sentence boundaries               │
│  Semantic        │  Split by meaning/topic changes             │
└─────────────────────────────────────────────────────────────────┘

Run with: uv run python phase3_llm_applications/01_rag_system/chunking.py
"""

import re
from textwrap import dedent
from typing import Callable

from schemas import Document, Chunk


# ─────────────────────────────────────────────────────────────
# CHUNKING STRATEGIES
# ─────────────────────────────────────────────────────────────


def chunk_by_paragraph(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    split document by paragraphs with size limits

    Strategy visualization:
    ┌────────────────────────────────────────────────────────┐
    │                   Original Document                    │
    │  Paragraph 1                                           │
    │  Paragraph 2                                           │
    │  Paragraph 3                                           │
    └────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────┐
    │ Chunk 1          │  ← Paragraphs combined until size limit
    │ Para 1 + Para 2  │
    └────────┬─────────┘
             │  overlap (last N chars)
             ▼
          ┌──────────────────┐
          │ Chunk 2          │
          │ ...+ Para 3      │
          └──────────────────┘

    Args:
        document: document to chunk
        chunk_size: target maximum characters per chunk
        chunk_overlap: characters to overlap between chunks

    Returns:
        list of Chunk objects
    """
    text = document.content
    chunks = []

    paragraphs = text.split("\n\n")

    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    source=document.source,
                )
            )
            chunk_index += 1

            if chunk_overlap > 0:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(
            Chunk(
                content=current_chunk.strip(),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                source=document.source,
            )
        )

    return chunks


def chunk_by_sentence(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    split document by sentences with size limits

    Better for documents without clear paragraph structure.
    Uses regex to identify sentence boundaries.

    Strategy:
    ┌─────────────────────────────────────────────────────────┐
    │  "First sentence. Second sentence. Third sentence..."  │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────┐
    │ "First sentence. Second..." │  ← Complete sentences only
    └──────────────────────────────┘
    """
    text = document.content
    chunks = []

    # split by sentence-ending punctuation followed by space
    sentences = re.split(r"(?<=[.!?])\s+", text)

    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    source=document.source,
                )
            )
            chunk_index += 1

            if chunk_overlap > 0:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(
            Chunk(
                content=current_chunk.strip(),
                doc_id=document.doc_id,
                chunk_index=chunk_index,
                source=document.source,
            )
        )

    return chunks


def chunk_fixed_size(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    split document by fixed character count

    Simplest strategy - just splits at character boundaries.
    May cut words in half, so use with caution.

    Strategy:
    ┌─────────────────────────────────────────────────────────┐
    │  "The quick brown fox jumps over the lazy dog and..."  │
    │   |<----- 500 chars ----->|                            │
    │                    |<----- 500 chars ----->|           │
    │                       ^overlap^                        │
    └─────────────────────────────────────────────────────────┘
    """
    text = document.content
    chunks = []
    chunk_index = 0

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end].strip()

        if chunk_content:
            chunks.append(
                Chunk(
                    content=chunk_content,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    source=document.source,
                )
            )
            chunk_index += 1

        start = end - chunk_overlap

    return chunks


# ─────────────────────────────────────────────────────────────
# CHUNKING FACTORY
# ─────────────────────────────────────────────────────────────

ChunkingStrategy = Callable[[Document, int, int], list[Chunk]]

STRATEGIES: dict[str, ChunkingStrategy] = {
    "paragraph": chunk_by_paragraph,
    "sentence": chunk_by_sentence,
    "fixed": chunk_fixed_size,
}


def get_chunking_strategy(name: str) -> ChunkingStrategy:
    """get chunking function by name"""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def chunk_document(
    document: Document,
    strategy: str = "paragraph",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    chunk document using specified strategy

    Args:
        document: document to chunk
        strategy: "paragraph", "sentence", or "fixed"
        chunk_size: target characters per chunk
        chunk_overlap: overlap between chunks

    Returns:
        list of Chunk objects
    """
    chunking_fn = get_chunking_strategy(strategy)
    return chunking_fn(document, chunk_size, chunk_overlap)


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────


def main():
    """demonstrate different chunking strategies"""
    print("=" * 60)
    print("  Chunking Strategies Demo")
    print("=" * 60)

    sample_text = dedent("""
        Introduction to Machine Learning

        Machine learning is a subset of artificial intelligence that enables systems to learn from data. Instead of being explicitly programmed, these systems identify patterns and make decisions with minimal human intervention.

        Types of Machine Learning

        There are three main types: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns through trial and error.

        Applications

        Machine learning powers many modern applications. These include recommendation systems, fraud detection, image recognition, and natural language processing. The field continues to evolve rapidly with new breakthroughs.

        Getting Started

        To begin with machine learning, you should first learn Python and basic statistics. Then explore libraries like scikit-learn, TensorFlow, or PyTorch. Practice with real datasets to build your skills.
    """).strip()

    doc = Document(content=sample_text, source="ml_intro.md")

    for strategy_name in ["paragraph", "sentence", "fixed"]:
        print(f"\n{'─' * 60}")
        print(f"  Strategy: {strategy_name.upper()}")
        print("─" * 60)

        chunks = chunk_document(
            doc,
            strategy=strategy_name,
            chunk_size=300,
            chunk_overlap=30,
        )

        print(f"  Total chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"\n  Chunk {chunk.chunk_index}:")
            print(f"  Length: {len(chunk.content)} chars")
            preview = chunk.content[:80].replace("\n", " ")
            print(f"  Preview: {preview}...")

    print("\n" + "=" * 60)
    print("  Chunking Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
