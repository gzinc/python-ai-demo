# Phase 1: Foundations - Python for AI Development

> ⚡ **OPTIONAL FOR AI APP DEVELOPERS**: If you're building LLM apps with LangChain/LangGraph,
> you can skip to Phase 2. These foundations help with debugging and optimization but aren't
> required for day-to-day AI application development. Come back when needed.

**Duration**: 2-3 weeks (optional)
**Goal**: Master Python libraries essential for AI/ML work

## 🎯 Learning Objectives

By the end of this phase, you will:
- ✅ Understand NumPy arrays and vectorized operations
- ✅ Manipulate data with Pandas DataFrames
- ✅ Prepare data for machine learning
- ✅ Understand basic ML concepts (features, labels, train/test splits)
- ✅ Visualize data effectively

## 📚 Module Overview

### Module 1: NumPy Basics (3-4 days)
**Location**: [01_numpy_basics/](01_numpy_basics/)

**Topics**:
- Array creation and manipulation
- Indexing and slicing
- Broadcasting and vectorization
- Mathematical operations
- Array reshaping and stacking

**Key Concepts**:
```python
# Arrays are the foundation of AI/ML
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Vectorized operations (no loops needed!)
result = arr * 2  # [2, 4, 6, 8, 10]

# Broadcasting (different shapes work together)
matrix + arr  # NumPy handles the dimensions
```

**Why This Matters**:
- All AI/ML libraries build on NumPy
- Vectors and matrices are how we represent data
- Efficient operations = faster training

**Exercises**:
1. Array manipulation practice
2. Matrix operations for embeddings simulation
3. Performance comparison: loops vs vectorization

---

### Module 2: Pandas Data Manipulation (4-5 days)
**Location**: [02_pandas_basics/](02_pandas_basics/)

**Topics**:
- DataFrames and Series
- Loading/saving data (CSV, JSON, Excel)
- Filtering, sorting, grouping
- Data cleaning and preprocessing
- Merging and joining datasets

**Key Concepts**:
```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Explore
df.head()
df.info()
df.describe()

# Filter
adults = df[df['age'] >= 18]

# Group and aggregate
df.groupby('category')['value'].mean()

# Handle missing data
df.fillna(0)
df.dropna()
```

**Why This Matters**:
- Real-world data is messy
- LLM applications need data preprocessing
- RAG systems work with structured data

**Exercises**:
1. Load and explore a real dataset
2. Clean and prepare data for ML
3. Build a data analysis report

---

### Module 3: ML Concepts & Preparation (3-4 days)
**Location**: [03_ml_concepts/](03_ml_concepts/)

**Topics**:
- Features and labels
- Train/test splits
- Feature engineering
- Normalization and scaling
- Basic ML workflow

**Key Concepts**:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why This Matters**:
- Understanding data preparation
- Foundation for embeddings and vectors
- Quality data = better AI results

**Exercises**:
1. Prepare a dataset for ML
2. Feature engineering practice
3. Build a simple ML pipeline

---

## 🚀 Getting Started

1. **Start with Module 1**:
   ```bash
   cd 01_numpy_basics
   cat README.md
   uv run python examples.py
   ```

2. **Complete exercises**:
   - Each module has `exercises.py`
   - Solutions in `solutions/`
   - Try before checking solutions!

3. **Build the project**:
   - Each module has a `project/` folder
   - Apply concepts to real problems
   - Document your learnings

4. **Track progress**:
   - Update [.serena/memories/phase1_progress.md](../.serena/memories/phase1_progress.md)
   - Note challenges and solutions
   - Record "aha!" moments

---

## 📊 Concepts You'll Learn

### From Java to Python: Key Differences

**Arrays vs Lists**:
```java
// Java
int[] arr = {1, 2, 3};
ArrayList<Integer> list = new ArrayList<>();
```
```python
# Python
arr = np.array([1, 2, 3])  # NumPy array (typed, efficient)
list = [1, 2, 3]            # Python list (dynamic, flexible)
```

**Data Structures**:
```java
// Java
Map<String, Integer> map = new HashMap<>();
```
```python
# Python
dict = {'key': value}      # Dictionary
df = pd.DataFrame(data)    # Pandas DataFrame (like SQL table)
```

**Iteration**:
```java
// Java
for (int i = 0; i < arr.length; i++) {
    arr[i] = arr[i] * 2;
}
```
```python
# Python - avoid loops!
arr = arr * 2  # Vectorized operation
# or
result = [x * 2 for x in arr]  # List comprehension
```

---

## 🎓 Resources

### Official Documentation
- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/
- Scikit-learn: https://scikit-learn.org/stable/

### Recommended Reading
- [docs/concepts/numpy_fundamentals.md](../docs/concepts/numpy_fundamentals.md)
- [docs/concepts/pandas_guide.md](../docs/concepts/pandas_guide.md)
- [docs/guides/data_preparation.md](../docs/guides/data_preparation.md)

### Practice Datasets
Located in `data/`:
- `raw/sample_data.csv` - Customer data
- `raw/text_samples.txt` - Text for NLP prep
- `raw/embeddings_demo.json` - Vector data

---

## ✅ Phase Completion Checklist

Track your progress:

- [ ] Completed NumPy basics exercises
- [ ] Understand array operations and broadcasting
- [ ] Completed Pandas data manipulation exercises
- [ ] Can load, clean, and explore datasets
- [ ] Understand train/test splits
- [ ] Can prepare data for ML
- [ ] Built at least one complete data analysis project
- [ ] Documented learnings in Serena memories

**Ready to move on when**:
- ✅ Comfortable with NumPy arrays
- ✅ Can manipulate data with Pandas
- ✅ Understand basic ML workflow
- ✅ Completed all exercises

---

## 🔄 Next Phase Preview

**Phase 2: LLM Fundamentals**
- Prompt engineering techniques
- Working with OpenAI/Anthropic APIs
- Embeddings and semantic search
- Vector databases

**Connection to Phase 1**:
- Embeddings are NumPy arrays (vectors)
- Data preparation skills apply to text
- Understanding vectors is crucial for RAG

---

## 💡 Tips

1. **Don't rush**: Foundations are critical
2. **Practice daily**: 1-2 hours consistently
3. **Build intuition**: Try variations, break things
4. **Use IPython/Jupyter**: Experiment interactively
5. **Document learnings**: Update memories regularly

**Common Pitfalls**:
- Skipping NumPy fundamentals (needed for embeddings!)
- Not practicing enough with real data
- Trying to memorize instead of understanding
- Not using vectorization (writing loops instead)

---

## 📝 Session Template

Copy this to your daily Serena memory:

```markdown
# Session: YYYY-MM-DD - Phase 1

## What I Learned Today
- Key concept:
- New technique:
- Interesting discovery:

## Exercises Completed
- [ ] Exercise 1:
- [ ] Exercise 2:

## Challenges Faced
- Problem:
- Solution:

## Questions for Next Session
1.
2.

## Next Steps
-
```

**Let's build a strong foundation! 🚀**
