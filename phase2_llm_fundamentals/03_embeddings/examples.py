"""
Module: Embeddings - Vector Representations of Text

how text becomes numbers that capture meaning.

Works without API keys — uses numpy for concepts, sentence-transformers for real embeddings.
sentence-transformers runs locally (no API key, no internet required after first download).

Run with: uv run python phase2_llm_fundamentals/03_embeddings/examples.py
"""

from inspect import cleandoc

import numpy as np

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section

# region Helper Functions

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """compute cosine similarity between two vectors (range: -1 to 1)"""
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (magnitude_a * magnitude_b))


def rank_by_similarity(
        query_vec: np.ndarray,
        candidates: list[tuple[str, np.ndarray]],
        top_k: int = 3,
) -> list[tuple[str, float]]:
    """rank candidate (label, vector) pairs by cosine similarity to query_vec"""
    scored = [(label, cosine_similarity(query_vec, vec)) for label, vec in candidates]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_k]


# endregion


# region Demo 1: What Are Embeddings?

def demo_what_are_embeddings() -> None:
    """intuition: text → fixed-size vector that encodes meaning"""
    print_section("1. What Are Embeddings?")

    print("\n" + cleandoc("""
        An embedding converts text into a fixed-size list of numbers (a vector).
        The key property: similar meanings → similar vectors.

        Imagine a simplified 3D space where each axis represents a concept:
          dim 0 → how animal-related is this?
          dim 1 → how technical / code-related is this?
          dim 2 → how emotional is this?

        (Real embeddings have 384–1536 dimensions, not 3.)
    """))

    # fake 3D embeddings to build intuition — not real model output
    fake_vocab: dict[str, np.ndarray] = {
        "dog": np.array([0.90, 0.05, 0.40]),
        "cat": np.array([0.85, 0.05, 0.35]),
        "Python code": np.array([0.02, 0.92, 0.08]),
        "Java class": np.array([0.02, 0.88, 0.08]),
        "happy": np.array([0.05, 0.02, 0.95]),
        "sad": np.array([0.03, 0.02, 0.88]),
    }

    print("\n📊 Fake 3D Embeddings (animal | technical | emotional):")
    print(f"  {'Text':<14}  {'Vector':<32}  Closest word")
    print(f"  {'─' * 14}  {'─' * 32}  {'─' * 20}")
    for word, vec in fake_vocab.items():
        other_pairs = [
            (other_word, cosine_similarity(vec, other_vec))
            for other_word, other_vec in fake_vocab.items()
            if other_word != word
        ]
        closest_word, closest_score = max(other_pairs, key=lambda pair: pair[1])
        vec_str = f"[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]"
        print(f"  {word:<14}  {vec_str:<32}  {closest_word} ({closest_score:.3f})")

    print("\n💡 KEY INSIGHT:")
    print("   'dog' ↔ 'cat' are close    (both high on dim 0: animal)")
    print("   'Python code' ↔ 'Java class' are close  (both high on dim 1: technical)")
    print("   'dog' ↔ 'Java class' are far (completely different axes)")
    print("\n   Real models learn these dimensions automatically from billions of sentences.")


# endregion


# region Demo 2: Vector Math

def demo_vector_math() -> None:
    """cosine similarity, dot product, why cosine beats euclidean for text"""
    print_section("2. Vector Math: How Similarity Is Measured")

    # simple 2D vectors — easy to reason about geometrically
    vec_right = np.array([1.0, 0.0])  # points right (0°)
    vec_diagonal = np.array([0.707, 0.707])  # points 45°
    vec_left = np.array([-1.0, 0.0])  # points left (180°)

    print("\n📐 Example Vectors (2D):")
    print(f"   right    = {vec_right}   (points →)")
    print(f"   diagonal = {vec_diagonal}  (points ↗, 45°)")
    print(f"   left     = {vec_left}  (points ←)")

    print("\n1️⃣  Cosine Similarity — measures the angle between vectors:")
    sim_right_diagonal = cosine_similarity(vec_right, vec_diagonal)
    sim_right_left = cosine_similarity(vec_right, vec_left)
    sim_right_right = cosine_similarity(vec_right, vec_right)
    print(f"   cos_sim(right, diagonal) = {sim_right_diagonal:.3f}  (somewhat similar)")
    print(f"   cos_sim(right, left)     = {sim_right_left:.3f}  (opposite directions)")
    print(f"   cos_sim(right, right)    = {sim_right_right:.3f}  (identical)")

    print("\n2️⃣  Why cosine similarity — not euclidean distance — for text?")
    short_vec = np.array([1.0, 0.0])
    long_vec = np.array([100.0, 0.0])  # same direction, 100× longer
    euclidean_dist = float(np.linalg.norm(short_vec - long_vec))
    cos_result = cosine_similarity(short_vec, long_vec)
    print("   short=[1, 0]  vs  long=[100, 0]  (same direction, different length)")
    print(f"   euclidean distance : {euclidean_dist:.1f}  ← says they differ greatly!")
    print(f"   cosine similarity  : {cos_result:.3f}  ← correctly shows identical direction")
    print("   → embedding vectors often differ in magnitude, not meaning")
    print("   → cosine similarity is scale-invariant: the right tool for text")

    print("\n3️⃣  The formula:")
    print("\n" + cleandoc("""
        cos_sim(A, B) = (A · B) / (||A|| × ||B||)

          A · B   = dot product = sum of element-wise products
          ||A||   = L2 norm = sqrt(sum of squared elements)

          range: -1 (opposite) → 0 (unrelated) → 1 (identical)
    """))

    print("4️⃣  numpy implementation (what sentence-transformers uses internally):")
    print(cleandoc("""
        import numpy as np

        def cosine_similarity(vec_a, vec_b):
            return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    """))


# endregion


# region Demo 3: Real Embeddings

def demo_real_embeddings() -> None:
    """generate actual embeddings with sentence-transformers (no API key)"""
    print_section("3. Real Embeddings with sentence-transformers")

    print("\n" + cleandoc("""
        Model: all-MiniLM-L6-v2
          - 22 MB download, runs entirely locally
          - 384-dimensional output vectors
          - trained on 1 billion sentence pairs
          - no API key required
    """))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌  sentence-transformers not installed. Run: uv add sentence-transformers")
        return

    print("\nloading model (first run downloads ~22MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "Python is a programming language",
        "Python is used for machine learning",
        "Snakes are reptiles that eat mice",
        "Dogs are loyal companions",
        "Neural networks learn from data",
    ]

    short_labels = ["Py-lang", "Py-ML  ", "Snakes ", "Dogs   ", "Neural "]

    print(f"\nembedding {len(sentences)} sentences...")
    embeddings = model.encode(sentences)

    print("\n📊 Embedding properties:")
    print(f"   shape : {embeddings.shape}  ({len(sentences)} sentences × {embeddings.shape[1]} dims)")
    print(f"   dtype : {embeddings.dtype}")
    print(f"   range : [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    print("\n📊 Pairwise Cosine Similarity Matrix:")
    print(f"  {'':>20}", end="")
    for label in short_labels:
        print(f"  {label}", end="")
    print()
    print(f"  {'':>20}", end="")
    for _ in short_labels:
        print("  ───────", end="")
    print()

    for row_idx, label in enumerate(short_labels):
        print(f"  {sentences[row_idx][:20]:>20}", end="")
        for col_idx in range(len(sentences)):
            sim_score = cosine_similarity(embeddings[row_idx], embeddings[col_idx])
            print(f"  {sim_score:>7.3f}", end="")
        print()

    print("\n💡 NOTICE:")
    print("   'Python is a programming language' ↔ 'Python is used for ML': HIGH")
    print("   'Snakes are reptiles'              ↔ 'Python programming'    : LOW")
    print("   → the model understands 'Python' means different things in each context!")


# endregion


# region Demo 4: Semantic Search

def demo_semantic_search() -> None:
    """embed a corpus, then find most similar documents to a query"""
    print_section("4. Semantic Search")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌  sentence-transformers not installed")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")

    corpus = [
        "Python is a high-level programming language",
        "Machine learning uses algorithms to learn from data",
        "Neural networks are inspired by the human brain",
        "Dogs make great companions for humans",
        "The Eiffel Tower is located in Paris, France",
        "Gradient descent optimizes model parameters",
        "Cats are independent and curious animals",
        "Deep learning achieves superhuman performance on images",
        "Rome was not built in a day",
        "Backpropagation computes gradients for neural networks",
    ]

    print(f"\ncorpus: {len(corpus)} documents")
    print("embedding corpus...")
    corpus_embeddings = model.encode(corpus)

    queries = [
        "how do neural networks learn?",
        "cute pets and animals",
        "optimizing an AI model's weights",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        query_embedding = model.encode([query])[0]

        candidate_pairs = list(zip(corpus, corpus_embeddings))
        top_results = rank_by_similarity(query_embedding, candidate_pairs, top_k=3)

        print("   Top 3 matches:")
        for rank, (doc_text, score) in enumerate(top_results, 1):
            print(f"   {rank}. [{score:.3f}] {doc_text}")

    print("\n💡 NOTE:")
    print("   this is exactly what phase 3 RAG does — but at scale with ChromaDB.")
    print("   ChromaDB stores millions of vectors and does this search in milliseconds.")


# endregion


# region Demo 5: Semantic vs Keyword Search

def demo_semantic_vs_keyword() -> None:
    """contrast keyword matching with embedding similarity on synonym-heavy text"""
    print_section("5. Semantic vs Keyword Search")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌  sentence-transformers not installed")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # docs intentionally use synonyms/paraphrases — keyword search will miss them
    documents = [
        "The automobile motor makes a rattling noise",   # no 'car', no 'engine', no 'problem'
        "My car breaks down frequently on highways",     # 'car' matches, no 'engine'
        "Vehicle maintenance is important for safety",   # related, different framing
        "How to fix motor issues in your ride",          # motor=engine, ride=car — all synonyms
        "Python is great for data science projects",  # completely unrelated
        "I love cooking homemade pasta on weekends",  # completely unrelated
    ]

    query = "car engine problems"
    query_words = set(query.lower().split())

    print(f"\nQuery: '{query}'")
    print(f"\n{'Document':<46}  {'Keyword':<12}  Semantic")
    print(f"{'─' * 46}  {'─' * 12}  {'─' * 8}")

    query_embedding = model.encode([query])[0]
    doc_embeddings = model.encode(documents)

    for doc_text, doc_embedding in zip(documents, doc_embeddings):
        doc_words = set(doc_text.lower().split())
        keyword_matches = len(query_words & doc_words)
        keyword_cell = f"{'✅' if keyword_matches > 0 else '❌'} {keyword_matches} word{'s' if keyword_matches != 1 else ''}"

        sem_score = cosine_similarity(query_embedding, doc_embedding)
        sem_cell = f"{'✅' if sem_score > 0.25 else '❌'} {sem_score:.3f}"

        print(f"{doc_text[:46]:<46}  {keyword_cell:<12}  {sem_cell}")

    print("\n💡 KEY EXAMPLES:")
    print("   'The automobile engine makes a rattling noise'")
    print("   → keyword: 0 matches (no 'car', no 'engine', no 'problems')")
    print("   → semantic: HIGH score (automobile=car, engine=engine, rattle=problem)")
    print()
    print("   'How to fix motor issues in your ride'")
    print("   → keyword: 0 matches (motor≠engine, ride≠car)")
    print("   → semantic: HIGH score (understands synonym relationships)")


# endregion


# region Demo 6: Word Analogies

def demo_word_analogies() -> None:
    """king - man + woman ≈ queen: vector arithmetic encodes relationships"""
    print_section("6. Word Analogies: Embedding Arithmetic")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌  sentence-transformers not installed")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n" + cleandoc("""
        Classic NLP discovery: you can do arithmetic on word embeddings!

            king  - man + woman  ≈ queen
            Paris - France + Italy  ≈ Rome

        This works because embeddings encode semantic relationships —
        the vector from 'man' → 'king' is similar to 'woman' → 'queen'.
        The model learns these gender/royalty/geography axes from training data.
    """))

    vocabulary = [
        "king", "queen", "man", "woman", "prince", "princess",
        "paris", "france", "rome", "italy", "london", "england",
    ]

    print(f"\nembedding {len(vocabulary)} words...")
    word_embeddings = model.encode(vocabulary)
    word_to_vec = dict(zip(vocabulary, word_embeddings))

    analogies: list[tuple[str, str, str, str]] = [
        ("king", "man", "woman", "queen"),  # royalty + gender
        ("paris", "france", "italy", "rome"),  # capital + country
    ]

    for base_word, minus_word, plus_word, expected in analogies:
        # vector arithmetic: base - minus + plus
        result_vec = (
                word_to_vec[base_word]
                - word_to_vec[minus_word]
                + word_to_vec[plus_word]
        )

        print(f"\n🧮 {base_word} - {minus_word} + {plus_word} = ?")
        print(f"   expected answer: '{expected}'")

        # rank all other words by similarity to result vector
        candidate_pairs = [
            (word, word_to_vec[word])
            for word in vocabulary
            if word not in {base_word, minus_word, plus_word}
        ]
        ranked_results = rank_by_similarity(result_vec, candidate_pairs, top_k=4)

        print("   top matches:")
        for rank, (word, score) in enumerate(ranked_results, 1):
            marker = "  ← ✅ expected!" if word == expected else ""
            print(f"   {rank}. {word:<12}  ({score:.3f}){marker}")

    print("\n💡 INSIGHT:")
    print("   embeddings don't just store words — they encode geometric relationships.")
    print("   the offset vector 'man'→'king' ≈ 'woman'→'queen' (royalty dimension).")
    print("   this is why semantic search, RAG, and recommendation systems all work.")


# endregion


# region Demo 7: RAG Connection

def demo_rag_connection() -> None:
    """explain how embeddings power the Phase 3 RAG pipeline"""
    print_section("7. Connection to Phase 3: RAG Systems")

    print("\n" + cleandoc("""
        Retrieval Augmented Generation (RAG) is built entirely on embeddings.
        Everything we learned here is the foundation of Phase 3.
    """))

    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                       RAG PIPELINE                                 │
    ├─────────────────────────────┬──────────────────────────────────────┤
    │  INDEXING (one-time setup)  │  QUERYING (per user question)        │
    │  ──────────────────────     │  ──────────────────────────          │
    │                             │                                      │
    │  Documents                  │  User question                       │
    │       │                     │       │                              │
    │       ▼                     │       ▼                              │
    │  Split into chunks          │  Embed question        ← Demo 3      │
    │       │                     │       │                              │
    │       ▼                     │       ▼                              │
    │  Embed each chunk ← Demo 3  │  Cosine similarity     ← Demo 2      │
    │       │                     │  against all chunks                  │
    │       ▼                     │       │                              │
    │  Store in ChromaDB          │  Top-K relevant chunks               │
    │  (vector database) ← Demo 4 │       │                              │
    │                             │       ▼                              │
    │                             │  Inject chunks into prompt           │
    │                             │       │                              │
    │                             │       ▼                              │
    │                             │  LLM generates answer                │
    │                             │  grounded in your docs               │
    └─────────────────────────────┴──────────────────────────────────────┘
    """)

    print("📝 Phase 3 code you already have (phase3_llm_applications/01_rag_system/):")
    print()
    print(cleandoc("""
        # embedder.py — this is demo 3 at production scale
        embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
        chunk_vecs = embedder.embed_batch(text_chunks)

        # rag_pipeline.py — stores embeddings in ChromaDB (demo 4 at scale)
        db = ChromaDB(persist_directory="./chroma_db")
        for chunk, vec in zip(text_chunks, chunk_vecs, strict=True):
            db.add(chunk, vec, metadata)

        # retrieval.py — demo 4's rank_by_similarity, but inside ChromaDB
        query_vec = embedder.embed(user_question)
        top_chunks = db.search(query_vec, k=3)   # cosine similarity inside

        # rag_pipeline.py — uses retrieved context to ground LLM answer
        context = "\\n\\n".join([chunk.content for chunk in top_chunks])
        answer = llm.answer(question=user_question, context=context)
    """))

    print("\n🎯 What you now understand about Phase 3:")
    print("   ✅ Why we embed documents      (enables fast similarity search)")
    print("   ✅ What ChromaDB does          (stores vectors, runs cosine search at scale)")
    print("   ✅ Why local models work       (all-MiniLM runs without any API key)")
    print("   ✅ The full retrieval pipeline (embed → search → inject → generate)")


# endregion


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "What Are Embeddings?", "text → vector intuition", demo_what_are_embeddings),
    Demo("2", "Vector Math", "cosine similarity and why it works", demo_vector_math),
    Demo("3", "Real Embeddings", "sentence-transformers in action", demo_real_embeddings),
    Demo("4", "Semantic Search", "rank documents by meaning", demo_semantic_search),
    Demo("5", "Semantic vs Keyword", "where keyword search fails", demo_semantic_vs_keyword),
    Demo("6", "Word Analogies", "king - man + woman = queen", demo_word_analogies),
    Demo("7", "RAG Connection", "how this powers Phase 3", demo_rag_connection),
]


# endregion


def main() -> None:
    """interactive demo runner"""
    runner = MenuRunner(DEMOS, title="Embeddings: Vector Representations of Text")
    runner.run()


if __name__ == "__main__":
    main()
