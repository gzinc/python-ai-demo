"""
Pandas Exercise Solutions - YOUR IMPLEMENTATIONS

Fill in each function below. Run the exercises.py to test your solutions.

Run with: uv run python phase1_foundations/02_pandas_basics/exercises.py
"""

import pandas as pd
import numpy as np
from typing import List


# =============================================================================
# EXERCISE 1: DataFrame Filtering
# =============================================================================

def exercise_1_filter_by_category(
    df: pd.DataFrame,
    category: str
) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows matching the category.

    Args:
        df: DataFrame with 'category' column
        category: category value to filter by

    Returns:
        Filtered DataFrame

    Hints:
    - use df[df['column'] == value] to filter rows
    - this returns a new DataFrame with matching rows only
    """
    # YOUR CODE HERE
    pass


def exercise_1_filter_by_length(
    df: pd.DataFrame,
    min_length: int,
    max_length: int
) -> pd.DataFrame:
    """
    Filter DataFrame to rows where 'content' length is within range.

    Args:
        df: DataFrame with 'content' column
        min_length: minimum content length (inclusive)
        max_length: maximum content length (inclusive)

    Returns:
        Filtered DataFrame

    Hints:
    - df['content'].str.len() gives the length of each string
    - combine conditions with & (and) operator
    - wrap each condition in parentheses: (cond1) & (cond2)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Text Processing
# =============================================================================

def exercise_2_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'raw_text' column and create 'clean_text' column.

    Cleaning steps:
    1. Strip leading/trailing whitespace
    2. Convert to lowercase
    3. Replace newlines with spaces
    4. Replace multiple spaces with single space

    Args:
        df: DataFrame with 'raw_text' column

    Returns:
        DataFrame with new 'clean_text' column

    Hints:
    - chain .str methods: df['col'].str.strip().str.lower()...
    - .str.replace('\\n', ' ', regex=True) for newlines
    - .str.replace('\\s+', ' ', regex=True) for multiple spaces
    - assign result to df['clean_text']
    - return df (or df.copy() to avoid modifying original)
    """
    # YOUR CODE HERE
    pass


def exercise_2_add_token_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'est_tokens' column estimating token count from 'content'.

    Rule: ~4 characters per token (rough estimate)

    Args:
        df: DataFrame with 'content' column

    Returns:
        DataFrame with new 'est_tokens' column

    Hints:
    - df['content'].str.len() gives character count
    - use integer division: // 4
    - assign to df['est_tokens']
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Aggregation
# =============================================================================

def exercise_3_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics per category.

    For each category, compute:
    - count: number of documents
    - total_chars: sum of all char_count
    - avg_chars: average char_count

    Args:
        df: DataFrame with 'category' and 'char_count' columns

    Returns:
        DataFrame with category stats

    Hints:
    - use df.groupby('category')
    - use .agg() with named aggregations:
      .agg(
          count=('char_count', 'count'),
          total_chars=('char_count', 'sum'),
          avg_chars=('char_count', 'mean')
      )
    - use .reset_index() to make category a regular column
    """
    # YOUR CODE HERE
    pass


def exercise_3_find_top_n(
    df: pd.DataFrame,
    column: str,
    n: int = 3
) -> pd.DataFrame:
    """
    Return top N rows by the given column value.

    Args:
        df: DataFrame
        column: column to sort by (descending)
        n: number of top rows to return

    Returns:
        Top N rows sorted by column descending

    Hints:
    - df.sort_values(column, ascending=False) sorts descending
    - .head(n) takes first n rows
    - chain them together
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Document Preparation
# =============================================================================

def exercise_4_prepare_for_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare documents for embedding generation.

    Steps:
    1. Create 'doc_id' column from index (format: 'doc_000', 'doc_001', etc.)
    2. Create 'text' column combining title and content: "title: content"
    3. Add 'char_count' column
    4. Return only: doc_id, text, char_count, source

    Args:
        df: DataFrame with 'title', 'content', 'source' columns

    Returns:
        Cleaned DataFrame ready for embedding

    Hints:
    - use f-strings or .apply() for doc_id: f'doc_{i:03d}'
    - concatenate strings: df['title'] + ': ' + df['content']
    - df['text'].str.len() for char_count
    - df[['col1', 'col2', ...]] to select specific columns
    """
    # YOUR CODE HERE
    pass


def exercise_4_chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 500
) -> pd.DataFrame:
    """
    Split long documents into chunks for embedding.

    For each row, if 'content' > chunk_size chars, split into multiple rows.
    Add 'chunk_id' column (format: 'doc_id_chunk_0', 'doc_id_chunk_1', etc.)

    Args:
        df: DataFrame with 'doc_id' and 'content' columns
        chunk_size: max characters per chunk

    Returns:
        DataFrame with chunks (may have more rows than input)

    Hints:
    - iterate over df.iterrows() or use df.itertuples()
    - for each row, split content into chunks of chunk_size
    - use: content[i:i+chunk_size] for slicing
    - create new rows for each chunk
    - build a list of dicts, then pd.DataFrame(list_of_dicts)
    """
    # YOUR CODE HERE
    pass