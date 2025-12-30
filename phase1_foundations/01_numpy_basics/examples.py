"""
NumPy Basics - Practical Examples for AI Development

Run with: uv run python phase1_foundations/01_numpy_basics/examples.py
"""

import numpy as np
from typing import Tuple


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def array_creation_examples() -> None:
    """demonstrate array creation methods"""
    print_section("Array Creation")

    # from Python lists
    arr = np.array([1, 2, 3, 4, 5])
    print(f"From list: {arr}")
    print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")

    # 2D array (matrix) - all rows must have same length!
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n2D array:\n{matrix}")
    print(f"Shape: {matrix.shape} (rows, cols)")

    # common patterns
    zeros = np.zeros((3, 4))
    print(f"\nZeros (3x4):\n{zeros}")

    ones = np.ones((2, 3))
    print(f"\nOnes (2x3):\n{ones}")

    identity = np.eye(3)
    print(f"\nIdentity (3x3):\n{identity}")

    # ranges
    sequence = np.arange(0, 10, 2)
    print(f"\nArange [0, 10) step 2: {sequence}")

    linspace = np.linspace(0, 1, 5)
    print(f"Linspace [0, 1] 5 points: {linspace}")

    # random arrays (useful for simulating embeddings)
    np.random.seed(42)  # reproducible random
    random_arr = np.random.rand(3, 3)
    print(f"\nRandom 3x3:\n{random_arr}")


def vectorized_operations() -> None:
    """demonstrate vectorized operations (no loops!)"""
    print_section("Vectorized Operations")

    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([10, 20, 30, 40, 50])

    print(f"arr1: {arr1}")
    print(f"arr2: {arr2}")

    # element-wise operations
    print(f"\narr1 + arr2 = {arr1 + arr2}")
    print(f"arr1 * 2 = {arr1 * 2}")
    print(f"arr1 ** 2 = {arr1 ** 2}")
    print(f"arr1 * arr2 = {arr1 * arr2}")

    # math functions
    print(f"\nsqrt(arr1) = {np.sqrt(arr1)}")
    print(f"exp(arr1) = {np.exp(arr1)}")

    # aggregations
    print(f"\nsum: {arr1.sum()}")
    print(f"mean: {arr1.mean()}")
    print(f"std: {arr1.std()}")
    print(f"min: {arr1.min()}, max: {arr1.max()}")


def indexing_and_slicing() -> None:
    """demonstrate array indexing"""
    print_section("Indexing and Slicing")

    arr = np.array([10, 20, 30, 40, 50])
    print(f"array: {arr}")

    print(f"\narr[0] = {arr[0]}")
    print(f"arr[-1] = {arr[-1]} (last)")
    print(f"arr[1:4] = {arr[1:4]}")
    print(f"arr[::2] = {arr[::2]} (every 2nd)")

    # 2D indexing
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\nmatrix:\n{matrix}")

    print(f"\nmatrix[0, 1] = {matrix[0, 1]}")
    print(f"matrix[:, 1] = {matrix[:, 1]} (all rows, col 1)")
    print(f"matrix[1, :] = {matrix[1, :]} (row 1, all cols)")


def broadcasting_examples() -> None:
    """demonstrate broadcasting (critical for AI!)"""
    print_section("Broadcasting")

    # scalar + array
    arr = np.array([1, 2, 3, 4])
    print(f"arr: {arr}")
    print(f"arr + 10 = {arr + 10}")

    # 1D + 2D
    row = np.array([1, 2, 3])
    col = np.array([[10], [20], [30]])

    print(f"\nrow: {row}")
    print(f"col:\n{col}")
    print(f"\nrow + col (broadcasting):\n{row + col}")

    # real AI example: normalize embeddings
    print("\n--- Real AI Example: Batch Normalization ---")
    embeddings = np.random.rand(5, 4)  # 5 texts, 4 dimensions
    print(f"embeddings shape: {embeddings.shape}")
    print(f"sample embedding:\n{embeddings}")

    # normalize each embedding (subtract mean, divide by std)
    mean = embeddings.mean(axis=1, keepdims=True)
    std = embeddings.std(axis=1, keepdims=True)
    normalized = (embeddings - mean) / std

    print(f"\nmean shape: {mean.shape}")
    print(f"normalized embeddings:\n{normalized}")
    print(f"normalized mean: {normalized.mean(axis=1)}")  # should be ~0
    print(f"normalized std: {normalized.std(axis=1)}")    # should be ~1


def reshaping_examples() -> None:
    """demonstrate array reshaping"""
    print_section("Reshaping")

    arr = np.arange(12)
    print(f"original: {arr}")

    # reshape to 2D
    matrix = arr.reshape(3, 4)
    print(f"\nreshaped (3x4):\n{matrix}")

    # reshape to 3D
    cube = arr.reshape(2, 2, 3)
    print(f"\nreshaped (2x2x3):\n{cube}")

    # flatten
    flat = matrix.flatten()
    print(f"\nflattened: {flat}")

    # transpose
    transposed = matrix.T
    print(f"\ntransposed (4x3):\n{transposed}")

    # -1 for auto dimension
    auto = arr.reshape(2, -1)  # 2 rows, figure out cols
    print(f"\nauto reshape (2, -1):\n{auto}")


def ai_application_examples() -> None:
    """practical AI examples"""
    print_section("AI Application Examples")

    # 1. cosine similarity (used in RAG!)
    print("--- 1. Cosine Similarity (RAG systems) ---")
    text1_emb = np.array([0.2, 0.5, 0.8])
    text2_emb = np.array([0.3, 0.6, 0.7])
    text3_emb = np.array([-0.1, -0.2, -0.3])

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_12 = cosine_similarity(text1_emb, text2_emb)
    sim_13 = cosine_similarity(text1_emb, text3_emb)

    print(f"text1 vs text2 similarity: {sim_12:.4f}")
    print(f"text1 vs text3 similarity: {sim_13:.4f}")
    print("(1.0 = identical, 0.0 = unrelated, -1.0 = opposite)")

    # 2. batch processing (like LLM batching)
    print("\n--- 2. Batch Processing ---")
    batch_size = 8
    embedding_dim = 768  # common embedding size

    batch_embeddings = np.random.rand(batch_size, embedding_dim)
    print(f"batch shape: {batch_embeddings.shape}")

    # normalize all at once (no loops!)
    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    normalized_batch = batch_embeddings / norms
    print(f"normalized batch shape: {normalized_batch.shape}")

    # verify normalization
    new_norms = np.linalg.norm(normalized_batch, axis=1)
    print(f"norms after normalization: {new_norms[:3]}... (should be ~1.0)")

    # 3. attention scores (simplified)
    print("\n--- 3. Attention Mechanism (simplified) ---")
    query = np.array([1.0, 0.5, 0.2])
    keys = np.array([
        [1.0, 0.3, 0.1],  # key 1
        [0.5, 0.9, 0.4],  # key 2
        [0.2, 0.1, 0.8],  # key 3
    ])

    # compute attention scores
    scores = np.dot(keys, query)
    print(f"attention scores: {scores}")

    # softmax to get weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    print(f"attention weights: {attention_weights}")
    print(f"sum of weights: {attention_weights.sum():.4f} (should be 1.0)")


def performance_comparison() -> None:
    """compare loop vs vectorization performance"""
    print_section("Performance: Loops vs Vectorization")

    import time

    size = 1_000_000
    arr1 = np.random.rand(size)
    arr2 = np.random.rand(size)

    # loop version
    start = time.time()
    result_loop = []
    for i in range(len(arr1)):
        result_loop.append(arr1[i] + arr2[i])
    loop_time = time.time() - start

    # vectorized version
    start = time.time()
    result_vec = arr1 + arr2
    vec_time = time.time() - start

    print(f"array size: {size:,}")
    print(f"\nloop time: {loop_time:.4f}s")
    print(f"vectorized time: {vec_time:.4f}s")
    print(f"speedup: {loop_time / vec_time:.1f}x faster")
    print("\n✨ Vectorization is critical for AI performance!")


def main() -> None:
    """run all examples"""
    print("\n" + "=" * 60)
    print("  NumPy Basics - AI Development Examples")
    print("=" * 60)

    array_creation_examples()
    vectorized_operations()
    indexing_and_slicing()
    broadcasting_examples()
    reshaping_examples()
    ai_application_examples()
    performance_comparison()

    print("\n" + "=" * 60)
    print("  ✅ All examples completed!")
    print("  Next: Try exercises.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
