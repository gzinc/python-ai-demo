# Module 1: NumPy Basics

**Duration**: 3-4 days
**Goal**: Master NumPy arrays and vectorized operations

## 🎯 Why NumPy Matters for AI

- **Embeddings**: Text → vectors (NumPy arrays)
- **Model inputs**: All ML models take NumPy arrays
- **Performance**: 10-100x faster than Python loops
- **Foundation**: TensorFlow, PyTorch build on NumPy concepts

## 📚 Topics Covered

### 1. Array Creation
```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4, 5])

# Common patterns
zeros = np.zeros((3, 4))        # 3x4 array of zeros
ones = np.ones((2, 3))          # 2x3 array of ones
identity = np.eye(4)            # 4x4 identity matrix
random = np.random.rand(3, 3)   # 3x3 random values [0, 1)

# Ranges
sequence = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # 5 values from 0 to 1
```

### 2. Array Operations
```python
# Element-wise operations (vectorized!)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result = arr1 + arr2  # [5, 7, 9]
result = arr1 * 2     # [2, 4, 6]
result = arr1 ** 2    # [1, 4, 9]

# Math functions
np.sqrt(arr1)
np.exp(arr1)
np.log(arr1)
```

### 3. Indexing and Slicing
```python
arr = np.array([10, 20, 30, 40, 50])

# Basic indexing
arr[0]      # 10
arr[-1]     # 50 (last element)

# Slicing
arr[1:4]    # [20, 30, 40]
arr[:3]     # [10, 20, 30]
arr[::2]    # [10, 30, 50] (every 2nd element)

# 2D arrays
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

matrix[0, 1]    # 2 (row 0, col 1)
matrix[:, 1]    # [2, 5, 8] (all rows, col 1)
matrix[1, :]    # [4, 5, 6] (row 1, all cols)
```

### 4. Broadcasting
**The secret to efficient AI code!**

```python
# Different shapes work together
arr = np.array([1, 2, 3])
matrix = np.array([[1],
                   [2],
                   [3]])

# Broadcasting automatically aligns dimensions
result = arr + matrix
# [[2, 3, 4],
#  [3, 4, 5],
#  [4, 5, 6]]

# Real AI example: normalize embeddings
embeddings = np.random.rand(100, 768)  # 100 texts, 768 dimensions
mean = embeddings.mean(axis=1, keepdims=True)
std = embeddings.std(axis=1, keepdims=True)
normalized = (embeddings - mean) / std  # Broadcasting!
```

### 5. Reshaping
```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
matrix = arr.reshape(3, 4)  # 3 rows, 4 cols
cube = arr.reshape(2, 2, 3)  # 3D array

# Flatten
flat = matrix.flatten()  # Back to 1D

# Transpose
transposed = matrix.T  # Swap rows and columns
```

## 🚀 Hands-On Examples

Run these examples:

```bash
# Interactive exploration
uv run python examples.py

# Exercises (try first!)
uv run python exercises.py

# Check solutions
uv run python solutions/exercise_solutions.py
```

## 🎓 AI Application Examples

### Example 1: Similarity Between Vectors (Embeddings)
```python
# Simulating text embeddings
text1_embedding = np.array([0.2, 0.5, 0.8])
text2_embedding = np.array([0.3, 0.6, 0.7])

# Cosine similarity (used in RAG systems!)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(text1_embedding, text2_embedding)
print(f"Similarity: {similarity:.4f}")  # 0.0 = different, 1.0 = identical
```

### Example 2: Batch Processing (like LLM token processing)
```python
# Process multiple texts at once
batch_embeddings = np.random.rand(32, 768)  # 32 texts, 768 dims

# Normalize all at once (no loops!)
norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
normalized_batch = batch_embeddings / norms
```

### Example 3: Attention Scores (simplified transformer concept)
```python
# Simplified attention mechanism
query = np.array([1.0, 0.5, 0.2])
keys = np.array([
    [1.0, 0.3, 0.1],
    [0.5, 0.9, 0.4],
    [0.2, 0.1, 0.8]
])

# Compute attention scores
scores = np.dot(keys, query)  # How relevant each key is to query
attention_weights = np.exp(scores) / np.sum(np.exp(scores))  # Softmax

print(f"Attention weights: {attention_weights}")
# Shows which keys are most relevant
```

## 📝 Exercises

### Exercise 1: Array Manipulation
Create a function that:
1. Generates a random 5x5 matrix
2. Normalizes each row (subtract mean, divide by std)
3. Returns the normalized matrix

### Exercise 2: Vector Operations
Implement these functions:
- `euclidean_distance(a, b)` - Distance between vectors
- `cosine_similarity(a, b)` - Similarity score
- `find_nearest(query, vectors)` - Find most similar vector

### Exercise 3: Performance Comparison
Compare loop vs vectorization:
```python
# Loop version
def add_loop(arr1, arr2):
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result

# Vectorized version
def add_vectorized(arr1, arr2):
    return arr1 + arr2

# Time both with large arrays
```

## 🎯 Key Takeaways

1. **No Loops**: Use vectorized operations whenever possible
2. **Broadcasting**: Let NumPy handle dimension alignment
3. **Memory**: NumPy arrays are more efficient than Python lists
4. **Foundation**: These concepts apply to all AI/ML work

## ✅ Completion Checklist

- [ ] Understand array creation methods
- [ ] Comfortable with indexing and slicing
- [ ] Can explain broadcasting
- [ ] Know when to use reshape vs transpose
- [ ] Completed all exercises
- [ ] Built similarity search example
- [ ] Documented learnings in memory

## 🔄 Next: Pandas Data Manipulation

Once comfortable with NumPy:
- Move to `02_pandas_data/`
- DataFrames build on NumPy arrays
- Learn data manipulation for LLM applications

---

**Practice makes perfect! Spend time experimenting. 🚀**
