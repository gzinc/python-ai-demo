"""
Pandas Exercises - Practice Problems

Instructions:
1. Implement functions in solutions/exercise_solutions.py
2. Run this file to test your solutions
3. Hints are in the solution file docstrings

Run with: uv run python phase1_foundations/02_pandas_basics/exercises.py
"""

import pandas as pd
import numpy as np

# import your solutions
from solutions.exercise_solutions import (
    exercise_1_filter_by_category,
    exercise_1_filter_by_length,
    exercise_2_clean_text,
    exercise_2_add_token_estimate,
    exercise_3_category_stats,
    exercise_3_find_top_n,
    exercise_4_prepare_for_embedding,
    exercise_4_chunk_dataframe,
)
from common.util.utils import print_section



# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests() -> None:
    """run all exercise tests"""

    print_section("Exercise 1a: Filter by Category")
    try:
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': ['hr', 'eng', 'hr', 'sales']
        })
        result = exercise_1_filter_by_category(df, 'hr')
        if result is None:
            print("❌ Not implemented yet")
        elif len(result) == 2 and list(result['id']) == [1, 3]:
            print("✅ Passed!")
        else:
            print(f"❌ Failed: expected 2 hr rows, got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 1b: Filter by Length")
    try:
        df = pd.DataFrame({
            'content': ['hi', 'hello there', 'a very long string indeed']
        })
        result = exercise_1_filter_by_length(df, 5, 15)
        if result is None:
            print("❌ Not implemented yet")
        elif len(result) == 1 and result['content'].iloc[0] == 'hello there':
            print("✅ Passed!")
        else:
            print(f"❌ Failed: expected 'hello there', got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 2a: Clean Text")
    try:
        df = pd.DataFrame({'raw_text': ['  HELLO\n\nWorld  ', '  Multiple   Spaces  ']})
        result = exercise_2_clean_text(df)
        if result is None:
            print("❌ Not implemented yet")
        elif 'clean_text' in result.columns and result['clean_text'].iloc[0] == 'hello world':
            print("✅ Passed!")
            print(f"   clean_text: {result['clean_text'].tolist()}")
        else:
            print(f"❌ Failed: expected 'hello world', got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 2b: Token Estimate")
    try:
        df = pd.DataFrame({'content': ['Hello world!', 'Hi']})  # 12 and 2 chars
        result = exercise_2_add_token_estimate(df)
        if result is None:
            print("❌ Not implemented yet")
        elif 'est_tokens' in result.columns and list(result['est_tokens']) == [3, 0]:
            print("✅ Passed!")
        else:
            print(f"❌ Failed: expected [3, 0], got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 3a: Category Stats")
    try:
        df = pd.DataFrame({
            'category': ['hr', 'hr', 'eng'],
            'char_count': [100, 200, 150]
        })
        result = exercise_3_category_stats(df)
        if result is None:
            print("❌ Not implemented yet")
        elif len(result) == 2 and 'total_chars' in result.columns:
            print("✅ Passed!")
            print(result)
        else:
            print(f"❌ Failed: expected stats DataFrame, got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 3b: Find Top N")
    try:
        df = pd.DataFrame({
            'name': ['a', 'b', 'c', 'd'],
            'score': [10, 40, 30, 20]
        })
        result = exercise_3_find_top_n(df, 'score', 2)
        if result is None:
            print("❌ Not implemented yet")
        elif len(result) == 2 and list(result['score']) == [40, 30]:
            print("✅ Passed!")
        else:
            print(f"❌ Failed: expected scores [40, 30], got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 4a: Prepare for Embedding")
    try:
        df = pd.DataFrame({
            'title': ['Policy', 'Guide'],
            'content': ['Vacation rules', 'Work from home'],
            'source': ['hr', 'hr']
        })
        result = exercise_4_prepare_for_embedding(df)
        if result is None:
            print("❌ Not implemented yet")
        elif 'doc_id' in result.columns and 'text' in result.columns:
            if result['text'].iloc[0] == 'Policy: Vacation rules':
                print("✅ Passed!")
                print(result)
            else:
                print(f"❌ Failed: text format wrong:\n{result}")
        else:
            print(f"❌ Failed: missing columns:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Exercise 4b: Chunk DataFrame")
    try:
        df = pd.DataFrame({
            'doc_id': ['d1', 'd2'],
            'content': ['a' * 1000, 'short']  # 1000 chars and 5 chars
        })
        result = exercise_4_chunk_dataframe(df, chunk_size=500)
        if result is None:
            print("❌ Not implemented yet")
        elif len(result) == 3 and 'chunk_id' in result.columns:  # 2 chunks + 1 short
            print("✅ Passed!")
            print(f"   chunk_ids: {result['chunk_id'].tolist()}")
        else:
            print(f"❌ Failed: expected 3 rows with chunk_id, got:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("  Exercise Summary")
    print("=" * 60)
    print("Implement functions in: solutions/exercise_solutions.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()