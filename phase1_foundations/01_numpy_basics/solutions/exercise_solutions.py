"""
NumPy Exercise Solutions - YOUR IMPLEMENTATIONS

Fill in each function below. Run the exercises.py to test your solutions.

Run with: uv run python phase1_foundations/01_numpy_basics/exercises.py
"""

import numpy as np


# =============================================================================
# EXERCISE 1: Array Manipulation
# =============================================================================

def exercise_1_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row to have mean=0 and std=1.

    Formula: normalized = (value - mean) / std

    Hints:
    - use matrix.mean(axis=1, keepdims=True) to get mean per row
    - use matrix.std(axis=1, keepdims=True) to get std per row
    - keepdims=True is important for broadcasting!
    """
    return (matrix - matrix.mean(axis=1, keepdims=True)) / matrix.std(axis=1, keepdims=True)


# =============================================================================
# EXERCISE 2: Vector Operations
# =============================================================================

def exercise_2_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance: sqrt(sum((a - b)^2))

    Hints:
    - np.sqrt() for square root
    - np.sum() to sum elements
    - or use np.linalg.norm(a - b) as shortcut
    """
    return np.sqrt(np.sum((a - b) ** 2))


def exercise_2_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity: dot(a, b) / (norm(a) * norm(b))

    Hints:
    - np.dot(a, b) for dot product
    - np.linalg.norm(a) for vector length
    """
    # YOUR CODE HERE
    pass


def exercise_2_find_nearest(query: np.ndarray, vectors: np.ndarray) -> int:
    """
    Find index of vector most similar to query.

    Hints:
    - compute cosine_similarity for each vector
    - np.argmax() returns index of maximum value
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Batch Operations
# =============================================================================

def exercise_3_batch_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize each embedding to unit length (norm=1).

    Hints:
    - np.linalg.norm(embeddings, axis=1, keepdims=True)
    - divide embeddings by their norms
    """
    # YOUR CODE HERE
    pass


def exercise_3_pairwise_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise cosine similarities.

    Result shape: (n, n) where result[i,j] = similarity(emb[i], emb[j])

    Hints:
    - first normalize all embeddings to unit length
    - for unit vectors: cosine_sim = dot product
    - np.dot(normalized, normalized.T) gives all pairs at once!
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Reshaping Challenge
# =============================================================================

def exercise_4_reshape_for_attention(
    queries: np.ndarray,
    num_heads: int
) -> np.ndarray:
    """
    Reshape for multi-head attention.

    Input shape: (batch, seq_len, hidden_dim)
    Output shape: (batch, num_heads, seq_len, head_dim)

    Where: head_dim = hidden_dim // num_heads

    Hints:
    - get dimensions: batch, seq_len, hidden_dim = queries.shape
    - calculate head_dim = hidden_dim // num_heads
    - reshape: queries.reshape(batch, seq_len, num_heads, head_dim)
    - transpose to move num_heads before seq_len: .transpose(0, 2, 1, 3)
    """
    # YOUR CODE HERE
    pass
