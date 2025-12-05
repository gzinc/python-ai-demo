"""
Retrieval Module - Search and ranking for RAG

This module handles the retrieval phase of RAG:
- Vector similarity search
- Result ranking and filtering
- Context assembly for generation

Retrieval Flow:
┌─────────────────────────────────────────────────────────────────┐
│                     Retrieval Pipeline                          │
│                                                                 │
│  Query ──► Embed ──► Vector Search ──► Rank ──► Filter ──► Top-K│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Run with: uv run python phase3_llm_applications/01_rag_system/retrieval.py
"""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class RetrievalResult:
    """result from retrieval with metadata"""

    content: str
    source: str
    similarity: float
    chunk_id: str

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f"RetrievalResult(source={self.source}, sim={self.similarity:.3f}, preview='{preview}...')"


class Retriever:
    """
    Handles retrieval from vector store

    Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                        Retriever                             │
    │                                                              │
    │  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐  │
    │  │  Query  │───►│ ChromaDB    │───►│ Post-processing     │  │
    │  │         │    │ collection  │    │ (rank, filter)      │  │
    │  └─────────┘    └─────────────┘    └─────────────────────┘  │
    │                                              │               │
    │                                              ▼               │
    │                                    ┌─────────────────┐      │
    │                                    │ RetrievalResult │      │
    │                                    │     list        │      │
    │                                    └─────────────────┘      │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        collection: Any,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ):
        """
        initialize retriever

        Args:
            collection: ChromaDB collection
            top_k: default number of results to return
            min_similarity: minimum similarity threshold (0.0 to 1.0)
        """
        self.collection = collection
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """
        retrieve relevant chunks for a query

        Query flow:
        ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
        │  Query  │───►│  Embed  │───►│ Search  │───►│ Top-K   │
        │  text   │    │  query  │    │ vectors │    │ results │
        └─────────┘    └─────────┘    └─────────┘    └─────────┘

        Args:
            query: search query text
            top_k: number of results (default: self.top_k)
            min_similarity: minimum similarity threshold
            filter_metadata: optional metadata filter dict

        Returns:
            list of RetrievalResult objects
        """
        k = top_k or self.top_k
        threshold = min_similarity if min_similarity is not None else self.min_similarity

        # build query kwargs
        query_kwargs = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self.collection.query(**query_kwargs)

        return self._process_results(results, threshold)

    def _process_results(
        self,
        raw_results: dict,
        min_similarity: float,
    ) -> list[RetrievalResult]:
        """
        convert ChromaDB results to RetrievalResult objects

        Includes:
        - Distance to similarity conversion
        - Filtering by threshold
        - Sorting by similarity
        """
        retrieval_results = []

        if not raw_results["documents"] or not raw_results["documents"][0]:
            return retrieval_results

        for i, doc in enumerate(raw_results["documents"][0]):
            # convert distance to similarity (ChromaDB returns distance)
            distance = raw_results["distances"][0][i] if raw_results["distances"] else 0
            similarity = 1 - distance  # cosine distance to similarity

            # filter by threshold
            if similarity < min_similarity:
                continue

            retrieval_results.append(
                RetrievalResult(
                    content=doc,
                    source=raw_results["metadatas"][0][i].get("source", "unknown"),
                    similarity=similarity,
                    chunk_id=raw_results["ids"][0][i] if raw_results["ids"] else "",
                )
            )

        # sort by similarity (highest first)
        retrieval_results.sort(key=lambda x: x.similarity, reverse=True)

        return retrieval_results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[RetrievalResult, float]]:
        """
        retrieve with explicit scores for debugging

        Returns:
            list of (result, raw_distance) tuples
        """
        results = self.retrieve(query, top_k)
        return [(r, 1 - r.similarity) for r in results]


# ─────────────────────────────────────────────────────────────
# CONTEXT ASSEMBLY
# ─────────────────────────────────────────────────────────────


def assemble_context(
    results: list[RetrievalResult],
    max_tokens: int = 3000,
    include_sources: bool = True,
) -> str:
    """
    assemble retrieved chunks into context string

    Context assembly:
    ┌─────────────────────────────────────────────────────────┐
    │  Retrieved Chunks                                       │
    │  ┌─────┐ ┌─────┐ ┌─────┐                               │
    │  │ C1  │ │ C2  │ │ C3  │                               │
    │  └──┬──┘ └──┬──┘ └──┬──┘                               │
    │     │      │      │                                    │
    │     └──────┼──────┘                                    │
    │            │                                           │
    │            ▼                                           │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ [Source 1: doc1.txt]                            │  │
    │  │ Content from chunk 1...                         │  │
    │  │                                                 │  │
    │  │ [Source 2: doc2.txt]                            │  │
    │  │ Content from chunk 2...                         │  │
    │  └─────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘

    Args:
        results: list of RetrievalResult objects
        max_tokens: approximate max tokens (uses 4 chars/token estimate)
        include_sources: whether to include source labels

    Returns:
        assembled context string
    """
    max_chars = max_tokens * 4  # rough estimate
    context_parts = []
    current_length = 0

    for i, result in enumerate(results, 1):
        if include_sources:
            part = f"[Source {i}: {result.source}]\n{result.content}"
        else:
            part = result.content

        # check if adding this would exceed limit
        if current_length + len(part) > max_chars:
            # try to fit partial content
            remaining = max_chars - current_length
            if remaining > 100:  # only add if meaningful space left
                truncated = part[:remaining] + "..."
                context_parts.append(truncated)
            break

        context_parts.append(part)
        current_length += len(part) + 2  # account for separator

    return "\n\n".join(context_parts)


def format_results_for_display(
    results: list[RetrievalResult],
    show_content: bool = True,
    max_preview_length: int = 100,
) -> str:
    """
    format results for human-readable display

    Args:
        results: list of RetrievalResult objects
        show_content: whether to show content preview
        max_preview_length: max characters for preview

    Returns:
        formatted string
    """
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results:\n"]

    for i, result in enumerate(results, 1):
        lines.append(f"{i}. [{result.source}] (similarity: {result.similarity:.3f})")
        if show_content:
            preview = result.content[:max_preview_length].replace("\n", " ")
            if len(result.content) > max_preview_length:
                preview += "..."
            lines.append(f"   {preview}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────


def main():
    """demonstrate retrieval functionality"""
    print("=" * 60)
    print("  Retrieval Module Demo")
    print("=" * 60)

    # create sample results for demo
    sample_results = [
        RetrievalResult(
            content="Python functions are defined using the def keyword. They allow code reuse and organization.",
            source="functions.md",
            similarity=0.92,
            chunk_id="doc1_0",
        ),
        RetrievalResult(
            content="Variables in Python store data values. You don't need to declare types explicitly.",
            source="variables.md",
            similarity=0.78,
            chunk_id="doc2_0",
        ),
        RetrievalResult(
            content="Error handling uses try-except blocks to catch and handle exceptions gracefully.",
            source="errors.md",
            similarity=0.65,
            chunk_id="doc3_0",
        ),
    ]

    print("\n1. Sample retrieval results:")
    print(format_results_for_display(sample_results))

    print("\n2. Assembled context for LLM:")
    print("-" * 40)
    context = assemble_context(sample_results, max_tokens=500)
    print(context)
    print("-" * 40)

    print("\n3. Context with token limit (200 tokens):")
    print("-" * 40)
    limited_context = assemble_context(sample_results, max_tokens=200)
    print(limited_context)
    print("-" * 40)

    print("\n" + "=" * 60)
    print("  Retrieval Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
