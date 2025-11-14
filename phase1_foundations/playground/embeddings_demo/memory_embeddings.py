"""
Memory Embeddings Processor

Generate and display embeddings for all Serena memory files.
Demonstrates practical embedding usage with real learning data.

Run with: uv run python phase1_foundations/04_embeddings_intro/memory_embeddings.py

Requirements:
- sentence-transformers
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


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
        print(f"   embedding preview: {embedding[:5]}...")
        print()

    return embeddings


def display_embedding_stats(embeddings: Dict[str, np.ndarray]) -> None:
    """display statistics about the embeddings"""
    print_section("Embedding Statistics")

    for filename, embedding in embeddings.items():
        print(f"\n{filename}:")
        print(f"  dimension: {len(embedding)}")
        print(f"  min value: {embedding.min():.4f}")
        print(f"  max value: {embedding.max():.4f}")
        print(f"  mean value: {embedding.mean():.4f}")
        print(f"  std dev: {embedding.std():.4f}")
        print(f"  norm (L2): {np.linalg.norm(embedding):.4f}")


def calculate_similarity_matrix(
    embeddings: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, List[str]]:
    """
    calculate pairwise cosine similarity between all embeddings

    args:
        embeddings: dict of filename → embedding

    returns:
        similarity matrix and list of filenames
    """
    filenames = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[f] for f in filenames])

    # calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embedding_matrix)

    return similarity_matrix, filenames


def display_similarity_results(
    similarity_matrix: np.ndarray,
    filenames: List[str]
) -> None:
    """display similarity matrix and most similar pairs"""
    print_section("Semantic Similarity Between Memories")

    # display similarity matrix
    print("\nSimilarity Matrix:")
    print(f"{'':20}", end='')
    for name in filenames:
        print(f"{name[:15]:>15}", end=' ')
    print()

    for i, name1 in enumerate(filenames):
        print(f"{name1[:20]:20}", end=' ')
        for j, _ in enumerate(filenames):
            sim = similarity_matrix[i][j]
            if i == j:
                print(f"{'1.0000':>15}", end=' ')
            else:
                print(f"{sim:>15.4f}", end=' ')
        print()

    # find most similar pairs (excluding self-similarity)
    print("\nMost Similar Pairs:")
    pairs = []
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            sim = similarity_matrix[i][j]
            pairs.append((sim, filenames[i], filenames[j]))

    pairs.sort(reverse=True)

    for sim, file1, file2 in pairs[:5]:
        print(f"  {sim:.4f} - {file1} ↔ {file2}")


def display_embedding_vectors(embeddings: Dict[str, np.ndarray]) -> None:
    """display first 10 values of each embedding vector"""
    print_section("Embedding Vectors (First 10 Dimensions)")

    for filename, embedding in embeddings.items():
        print(f"\n{filename}:")
        print(f"  {embedding[:10]}")


def main() -> None:
    """main execution"""
    print_section("Memory Embeddings Processor")
    print("\nThis script demonstrates embeddings by processing your learning journey.")
    print("It reads all Serena memory files and generates semantic embeddings.")

    # load sentence transformer model
    print_section("Loading Model")
    print("downloading/loading paraphrase-multilingual-MiniLM-L12-v2...")
    print("(this may take a moment on first run)")

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
        print("  uv run python phase1_foundations/04_embeddings_intro/memory_embeddings.py")
        return

    print(f"✅ loaded {len(memories)} memory files:")
    for filename in memories.keys():
        print(f"   - {filename}")

    # generate embeddings
    embeddings = generate_embeddings(model, memories)

    # display embedding vectors
    display_embedding_vectors(embeddings)

    # display statistics
    display_embedding_stats(embeddings)

    # calculate and display similarities
    if len(embeddings) > 1:
        similarity_matrix, filenames = calculate_similarity_matrix(embeddings)
        display_similarity_results(similarity_matrix, filenames)
    else:
        print_section("Similarity Analysis")
        print("⚠️  Only one memory file found - need at least 2 for similarity comparison")

    # summary
    print_section("Summary")
    print(f"\n✅ Processed {len(embeddings)} memory files")
    print(f"✅ Generated {model.get_sentence_embedding_dimension()}-dimensional embeddings")
    print(f"✅ Each embedding is a NumPy array capturing semantic meaning")

    print("\n💡 Key Insights:")
    print("   - Embeddings convert text to dense numeric vectors")
    print("   - Similar documents have similar embeddings")
    print("   - You can use these embeddings for semantic search")
    print("   - This is the foundation for RAG systems (Phase 3!)")

    print("\n📚 What's Next:")
    print("   - Phase 2: Use embeddings with OpenAI/Anthropic APIs")
    print("   - Phase 3: Build RAG system with vector database")
    print("   - Phase 3: Semantic search over your documents")

    print()


if __name__ == "__main__":
    main()
