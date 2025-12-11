"""
Chunk model for RAG system

Represents a piece of a document after chunking.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """represents a chunk of a document"""

    content: str
    doc_id: str
    chunk_index: int
    source: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_{self.chunk_index}"
