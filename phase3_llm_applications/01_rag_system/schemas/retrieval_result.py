"""
RetrievalResult model for RAG system

Represents a search result with similarity score and metadata.
"""

from dataclasses import dataclass


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
