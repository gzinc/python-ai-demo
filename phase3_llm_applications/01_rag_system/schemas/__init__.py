"""
RAG System Data Models

Module Structure:
- document.py        → Document dataclass
- chunk.py           → Chunk dataclass
- retrieval_result.py → RetrievalResult dataclass

Usage:
    from models import Document, Chunk, RetrievalResult
"""

from .chunk import Chunk
from .document import Document
from .retrieval_result import RetrievalResult

__all__ = ["Document", "Chunk", "RetrievalResult"]
