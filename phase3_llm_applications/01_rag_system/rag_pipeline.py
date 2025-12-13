"""
RAG Pipeline - Complete Retrieval Augmented Generation System

This module implements a full RAG pipeline orchestrating:
1. Document ingestion
2. Chunking (via chunking.py)
3. Embedding & storage (ChromaDB)
4. Retrieval (via retrieval.py)
5. Generation (LLM)

Architecture:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Ingest  │───►│  Chunk   │───►│  Embed   │───►│  Store   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                     │
┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  Answer  │◄───│ Generate │◄───│ Retrieve │◄────────┘
└──────────┘    └──────────┘    └──────────┘

Run with: uv run python phase3_llm_applications/01_rag_system/rag_pipeline.py
"""

import os
from pathlib import Path
from inspect import cleandoc
from typing import Optional

from dotenv import load_dotenv

from schemas import Document, Chunk, RetrievalResult
from chunking import chunk_by_paragraph
from retrieval import Retriever, assemble_context

load_dotenv()


class RAGPipeline:
    """
    Complete RAG Pipeline Orchestrator

    Flow diagram:
    ┌─────────────────────────────────────────────────────────────┐
    │                      RAGPipeline                            │
    │                                                             │
    │  add_documents()                                            │
    │       │                                                     │
    │       ▼                                                     │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
    │  │ Ingest  │──►│ Chunk   │──►│ Embed   │──►│ Store   │      │
    │  └─────────┘   └─────────┘   └─────────┘   └─────────┘      │
    │                                                  │          │
    │  query()                                         │          │
    │       │                                          │          │
    │       ▼                                          ▼          │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
    │  │ Answer  │◄──│Generate │◄──│Retrieve │◄──│ Search  │      │
    │  └─────────┘   └─────────┘   └─────────┘   └─────────┘      │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        persist_directory: str = "./chroma_rag_db",
    ):
        """
        initialize RAG pipeline

        Args:
            collection_name: name for ChromaDB collection
            chunk_size: target size for each chunk (characters)
            chunk_overlap: overlap between chunks (characters)
            top_k: number of chunks to retrieve
            persist_directory: where to store ChromaDB data
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.persist_directory = persist_directory

        # lazy initialization
        self._collection = None
        self._chroma_client = None
        self._retriever = None

    @property
    def collection(self):
        """lazy load ChromaDB collection"""
        if self._collection is None:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def retriever(self) -> Retriever:
        """lazy load retriever"""
        if self._retriever is None:
            self._retriever = Retriever(
                collection=self.collection,
                top_k=self.top_k,
            )
        return self._retriever

    # ─────────────────────────────────────────────────────────────
    # CHUNKING (delegates to chunking.py)
    # ─────────────────────────────────────────────────────────────

    def chunk_document(self, document: Document) -> list[Chunk]:
        """split document into chunks with overlap"""
        return chunk_by_paragraph(
            document,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    # ─────────────────────────────────────────────────────────────
    # INGESTION
    # ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> int:
        """
        ingest documents into the RAG system

        Pipeline:
        Documents ──► Chunk ──► Embed ──► Store

        Returns:
            number of chunks added
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        ids = [chunk.chunk_id for chunk in all_chunks]
        contents = [chunk.content for chunk in all_chunks]
        metadatas = [
            {
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in all_chunks
        ]

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )

        return len(all_chunks)

    def add_text(self, text: str, source: str = "unknown") -> int:
        """convenience method to add raw text"""
        doc = Document(content=text, source=source)
        return self.add_documents([doc])

    def add_file(self, file_path: str) -> int:
        """add a text file to the RAG system"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        doc = Document(content=content, source=str(path.name))
        return self.add_documents([doc])

    # ─────────────────────────────────────────────────────────────
    # RETRIEVAL (delegates to retrieval.py)
    # ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """retrieve relevant chunks for a query"""
        return self.retriever.retrieve(query, top_k)

    # ─────────────────────────────────────────────────────────────
    # GENERATION
    # ─────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, context_chunks: list[RetrievalResult]) -> str:
        """
        build RAG prompt with retrieved context

        Prompt structure:
        ┌─────────────────────────────────────────────────────────┐
        │ System instruction                                      │
        ├─────────────────────────────────────────────────────────┤
        │ Context:                                                │
        │   [Source: doc1.txt]                                    │
        │   chunk content 1...                                    │
        ├─────────────────────────────────────────────────────────┤
        │ Question: user query                                    │
        ├─────────────────────────────────────────────────────────┤
        │ Instructions for answering                              │
        └─────────────────────────────────────────────────────────┘
        """
        context = assemble_context(context_chunks)

        prompt = cleandoc(f"""
            You are a helpful assistant that answers questions based on the provided context.
            Use ONLY the information from the context below. If the answer is not in the context, say "I don't have enough information to answer that."

            Context:
            {context}

            Question: {query}

            Instructions:
            - Answer based only on the context above
            - If citing information, mention the source number
            - Be concise but complete
            - If unsure, say so

            Answer:
        """)

        return prompt

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
    ) -> str | tuple[str, list[RetrievalResult]]:
        """
        query the RAG system

        Full pipeline:
        ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Question │──►│ Retrieve │──►│  Build   │──►│ Generate │──► Answer
        │          │   │ Context  │   │  Prompt  │   │   LLM    │
        └──────────┘   └──────────┘   └──────────┘   └──────────┘

        Args:
            question: user's question
            top_k: number of chunks to retrieve
            return_context: if True, also return retrieved chunks

        Returns:
            answer string, or (answer, context) if return_context=True
        """
        context_chunks = self.retrieve(question, top_k)

        if not context_chunks:
            no_context_answer = "I don't have any relevant information to answer this question."
            if return_context:
                return no_context_answer, []
            return no_context_answer

        prompt = self._build_prompt(question, context_chunks)
        answer = self._generate(prompt)

        if return_context:
            return answer, context_chunks
        return answer

    def _generate(self, prompt: str) -> str:
        """
        generate answer using LLM

        Supports: OpenAI, Anthropic, or fallback for testing
        """
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error: {e}")

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from anthropic import Anthropic

                client = Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                print(f"Anthropic error: {e}")

        # fallback: return context summary
        return "[No LLM configured - showing retrieved context]\n" + prompt.split("Context:")[1].split("Question:")[0].strip()

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """get collection statistics"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
        }

    def clear(self) -> None:
        """clear all documents from the collection"""
        if self._chroma_client:
            self._chroma_client.delete_collection(self.collection_name)
            self._collection = None
            self._retriever = None


def main():
    """quick demo of the RAG pipeline"""
    print("=" * 60)
    print("  RAG Pipeline Quick Demo")
    print("=" * 60)
    print("\nFor comprehensive examples, run:")
    print("  uv run python phase3_llm_applications/01_rag_system/examples.py")

    # minimal demo - use temp directory for demo
    import tempfile
    temp_dir = tempfile.mkdtemp()
    rag = RAGPipeline(
        collection_name="quick_demo",
        chunk_size=300,
        persist_directory=temp_dir,
    )

    rag.add_text(
        "Python is a high-level programming language. It emphasizes code readability.",
        source="python_intro.md",
    )

    print("\n1. Added document")
    print(f"   Stats: {rag.get_stats()}")

    print("\n2. Query: 'What is Python?'")
    results = rag.retrieve("What is Python?")
    for r in results:
        print(f"   - {r.source}: {r.similarity:.3f}")

    rag.clear()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
