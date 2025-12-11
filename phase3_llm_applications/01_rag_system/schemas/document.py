"""
Document model for RAG system

Represents a source document with content and metadata.
"""

import hashlib
from dataclasses import dataclass, field


@dataclass
class Document:
    """represents a document with metadata"""

    content: str
    source: str
    doc_id: str = field(default="")

    def __post_init__(self):
        if not self.doc_id:
            # generate id from content hash
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
