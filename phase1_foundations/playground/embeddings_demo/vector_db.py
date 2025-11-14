"""
Vector Database Operations

Handles ChromaDB storage and retrieval for embeddings.
Demonstrates vector database patterns for semantic search.

Usage:
    from vector_db import VectorDB

    db = VectorDB()
    db.save_embeddings(embeddings, documents)
    results = db.search("query text", top_k=3)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorDB:
    """manages ChromaDB operations for learning memories"""

    def __init__(
        self,
        collection_name: str = "learning_memories",
        db_path: Optional[Path] = None
    ):
        """
        initialize vector database

        args:
            collection_name: name for the ChromaDB collection
            db_path: path to store database (default: .chromadb)
        """
        self.collection_name = collection_name

        # default to .chromadb at project root
        if db_path is None:
            db_path = Path(".chromadb")

        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)

        # initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = None

    def create_collection(self, reset: bool = True) -> chromadb.Collection:
        """
        create or get ChromaDB collection

        args:
            reset: if True, delete existing collection and create new one

        returns:
            ChromaDB collection
        """
        # delete existing if reset requested
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"♻️  deleted existing '{self.collection_name}' collection")
            except Exception:
                pass

        # create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Serena learning memories with embeddings"}
        )

        print(f"✅ created collection: '{self.collection_name}'")
        return self.collection

    def get_collection(self) -> chromadb.Collection:
        """
        get existing collection

        returns:
            ChromaDB collection

        raises:
            ValueError: if collection doesn't exist
        """
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            return self.collection
        except Exception as e:
            raise ValueError(
                f"collection '{self.collection_name}' not found. "
                f"call create_collection() first."
            ) from e

    def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        documents: Dict[str, str],
        metadata: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        save embeddings and documents to ChromaDB

        args:
            embeddings: dict of id → embedding vector
            documents: dict of id → document content
            metadata: optional dict of id → metadata dict
        """
        if self.collection is None:
            self.create_collection()

        # prepare data for ChromaDB
        ids = []
        docs = []
        embeddings_list = []
        metadatas = []

        for doc_id, content in documents.items():
            ids.append(doc_id)
            docs.append(content)
            embeddings_list.append(embeddings[doc_id].tolist())

            # add metadata
            meta = {
                "filename": doc_id,
                "length": len(content),
                "type": "learning_memory"
            }
            if metadata and doc_id in metadata:
                meta.update(metadata[doc_id])
            metadatas.append(meta)

        # add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=docs,
            metadatas=metadatas
        )

        print(f"✅ saved {len(ids)} documents to ChromaDB")
        print(f"📁 database location: {self.db_path.absolute()}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> Dict:
        """
        search for similar documents using embedding

        args:
            query_embedding: embedding vector for query
            top_k: number of results to return

        returns:
            dict with ids, documents, distances, metadatas
        """
        if self.collection is None:
            self.get_collection()

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        return results

    def search_by_text(
        self,
        query: str,
        model: SentenceTransformer,
        top_k: int = 3
    ) -> Dict:
        """
        search using text query (generates embedding automatically)

        args:
            query: text query
            model: sentence transformer model to encode query
            top_k: number of results to return

        returns:
            dict with ids, documents, distances, metadatas
        """
        query_embedding = model.encode(query)
        return self.search(query_embedding, top_k=top_k)

    def get_all(self) -> Dict:
        """
        retrieve all documents from collection

        returns:
            dict with all documents and metadata
        """
        if self.collection is None:
            self.get_collection()

        return self.collection.get()

    def count(self) -> int:
        """
        get number of documents in collection

        returns:
            document count
        """
        if self.collection is None:
            self.get_collection()

        return self.collection.count()

    def delete_collection(self) -> None:
        """delete the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            print(f"🗑️  deleted collection: '{self.collection_name}'")
        except Exception as e:
            print(f"⚠️  error deleting collection: {e}")

    def list_collections(self) -> List[str]:
        """
        list all collections in database

        returns:
            list of collection names
        """
        collections = self.client.list_collections()
        return [c.name for c in collections]


def demonstrate_search(
    db: VectorDB,
    model: SentenceTransformer,
    queries: List[str]
) -> None:
    """
    demonstrate semantic search with example queries

    args:
        db: VectorDB instance
        model: sentence transformer model
        queries: list of example queries
    """
    print(f"\n{'=' * 80}")
    print("  Semantic Search Demo")
    print('=' * 80)

    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")

        results = db.search_by_text(query, model, top_k=2)

        print("   Top matches:")
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            similarity = 1 - distance  # convert distance to similarity

            print(f"   {i+1}. {doc_id} (similarity: {similarity:.4f})")

            # show snippet
            snippet = results['documents'][0][i][:100].replace('\n', ' ')
            print(f"      \"{snippet}...\"")
        print()
