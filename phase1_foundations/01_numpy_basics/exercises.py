"""
NumPy Exercises - Practice Problems

Instructions:
1. Implement functions in solutions/exercise_solutions.py
2. Run this file to test your solutions
3. Hints are in the solution file docstrings

Run with: uv run python phase1_foundations/01_numpy_basics/exercises.py
"""

import numpy as np

# import your solutions
from solutions.exercise_solutions import (
    exercise_1_normalize_rows,
    exercise_2_euclidean_distance,
    exercise_2_cosine_similarity,
    exercise_2_find_nearest,
    exercise_3_batch_normalize,
    exercise_3_pairwise_similarity,
    exercise_4_reshape_for_attention,
)
from common.util.utils import print_section



def run_tests() -> None:
    """run all exercise tests"""

    print_section("Exercise 1: Row Normalization")
    try:
        arr = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        result = exercise_1_normalize_rows(arr)
        if result is None:
            print("❌ Not implemented yet")
        else:
            row_means = result.mean(axis=1)
            row_stds = result.std(axis=1)
            if np.allclose(row_means, 0, atol=1e-10) and np.allclose(row_stds, 1, atol=1e-10):
                print("✅ Passed!")
                print(f"   row means: {row_means}")
                print(f"   row stds: {row_stds}")
            else:
                print(f"❌ Failed: means={row_means}, stds={row_stds}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 2a: Euclidean Distance")
    try:
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        result = exercise_2_euclidean_distance(a, b)
        if result is None:
            print("❌ Not implemented yet")
        elif np.isclose(result, 5.0):
            print(f"✅ Passed! distance = {result}")
        else:
            print(f"❌ Failed: expected 5.0, got {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 2b: Cosine Similarity")
    try:
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        result = exercise_2_cosine_similarity(a, b)
        if result is None:
            print("❌ Not implemented yet")
        elif np.isclose(result, 1.0):
            print(f"✅ Passed! similarity = {result}")
        else:
            print(f"❌ Failed: expected 1.0, got {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 2c: Find Nearest")
    try:
        query = np.array([1.0, 0.0, 0.0])
        vectors = np.array([
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0]
        ])
        result = exercise_2_find_nearest(query, vectors)
        if result is None:
            print("❌ Not implemented yet")
        elif result == 1:
            print(f"✅ Passed! nearest index = {result}")
        else:
            print(f"❌ Failed: expected 1, got {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 3a: Batch Normalize")
    try:
        emb = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = exercise_3_batch_normalize(emb)
        if result is None:
            print("❌ Not implemented yet")
        else:
            norms = np.linalg.norm(result, axis=1)
            if np.allclose(norms, 1.0):
                print(f"✅ Passed! norms = {norms}")
            else:
                print(f"❌ Failed: norms should be 1.0, got {norms}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 3b: Pairwise Similarity")
    try:
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        result = exercise_3_pairwise_similarity(emb)
        if result is None:
            print("❌ Not implemented yet")
        elif result.shape == (3, 3) and np.isclose(result[0, 2], 1.0):
            print(f"✅ Passed! similarity[0,2] = {result[0, 2]}")
            print(f"   full matrix:\n{result}")
        else:
            print(f"❌ Failed: expected [0,2]=1.0, got matrix:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 4: Reshape for Attention")
    try:
        q = np.arange(64).reshape(2, 4, 8)  # batch=2, seq=4, hidden=8
        result = exercise_4_reshape_for_attention(q, num_heads=2)
        if result is None:
            print("❌ Not implemented yet")
        elif result.shape == (2, 2, 4, 4):
            print(f"✅ Passed! shape = {result.shape}")
        else:
            print(f"❌ Failed: expected (2, 2, 4, 4), got {result.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("  Exercise Summary")
    print("=" * 60)
    print("Implement functions in: solutions/exercise_solutions.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()