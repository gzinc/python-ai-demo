"""
Save Memory Embeddings to ChromaDB

Extends the basic embedding demo by saving to a vector database.
Demonstrates semantic search capabilities.

Run with: uv run python phase1_foundations/playground/embeddings_demo/save_to_chromadb.py

Requirements:
- sentence-transformers
- chromadb
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict

from vector_db import VectorDB, demonstrate_search
from common.util.utils import print_section



def load_memory_files() -> Dict[str, str]:
    """
    load all markdown files from .serena/memories/

    returns:
        dict mapping filename to file content
    """
    memories = {}
    memories_dir = Path('.serena/memories')

    if not memories_dir.exists():
        print(f"❌ Memories directory not found: {memories_dir}")
        return memories

    for md_file in memories_dir.glob('*.md'):
        try:
            content = md_file.read_text(encoding='utf-8')
            memories[md_file.name] = content
        except Exception as e:
            print(f"⚠️  Error reading {md_file.name}: {e}")

    return memories


def generate_embeddings(
    model: SentenceTransformer,
    memories: Dict[str, str]
) -> Dict[str, np.ndarray]:
    """
    generate embeddings for all memory files

    args:
        model: sentence transformer model
        memories: dict of filename → content

    returns:
        dict of filename → embedding vector
    """
    embeddings = {}

    print_section("Generating Embeddings")
    print(f"model: {model.get_sentence_embedding_dimension()}-dimensional embeddings\n")

    for filename, content in memories.items():
        # generate embedding for full document
        embedding = model.encode(content)
        embeddings[filename] = embedding

        print(f"✅ {filename}")
        print(f"   content length: {len(content)} chars")
        print(f"   embedding shape: {embedding.shape}")
        print()

    return embeddings


def main() -> None:
    """main execution"""
    print_section("Memory Embeddings → ChromaDB")
    print("\nThis script saves your learning memories to ChromaDB vector database.")
    print("You can then perform semantic search over your memories!")

    # load sentence transformer model
    print_section("Loading Model")
    print("loading paraphrase-multilingual-MiniLM-L12-v2...")

    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print(f"✅ model loaded: {model.get_sentence_embedding_dimension()}-dimensional embeddings")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease install sentence-transformers:")
        print("  uv add sentence-transformers")
        return

    # load memory files
    print_section("Loading Memory Files")
    memories = load_memory_files()

    if not memories:
        print("❌ No memory files found!")
        print("\nPlease ensure you're running from the project root:")
        print("  cd /home/gzuss/dev-wsl/python-ai-demo")
        print("  uv run python phase1_foundations/playground/embeddings_demo/save_to_chromadb.py")
        return

    print(f"✅ loaded {len(memories)} memory files:")
    for filename in memories.keys():
        print(f"   - {filename}")

    # generate embeddings
    embeddings = generate_embeddings(model, memories)

    # initialize ChromaDB
    print_section("Initializing ChromaDB")
    db = VectorDB(collection_name="learning_memories")

    # save to database
    print_section("Saving to Vector Database")
    db.save_embeddings(embeddings, memories)

    # verify save
    count = db.count()
    print(f"\n✅ verified: {count} documents in database")

    # demonstrate semantic search
    example_queries = [
        "What is my learning progress?",
        "How do I get started with this project?",
        "What are the project goals and objectives?"
    ]

    demonstrate_search(db, model, example_queries)

    # summary
    print_section("Summary")
    print(f"\n✅ saved {len(embeddings)} memory files to ChromaDB")
    print(f"✅ database location: .serena/chromadb/")
    print(f"✅ can now perform semantic search over your memories")

    print("\n💡 What You Can Do Now:")
    print("   - Search memories by meaning, not just keywords")
    print("   - Find similar documents automatically")
    print("   - Build RAG system with your learning notes (Phase 3!)")

    print("\n📚 Next Steps:")
    print("   - Try custom search queries")
    print("   - Add more memories and see them automatically searchable")
    print("   - Use this pattern for RAG systems in Phase 3")

    print("\n🔧 ChromaDB Commands:")
    print("   - View all collections: db.list_collections()")
    print("   - Count documents: db.count()")
    print("   - Search: db.search_by_text('your query', model)")
    print("   - Delete collection: db.delete_collection()")

    print()


if __name__ == "__main__":
    main()
