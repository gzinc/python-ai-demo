# ML Concepts for AI Development

Core machine learning concepts that help you understand how LLMs and AI systems work under the hood.

## Why ML Concepts for AI App Developers?

You don't need to implement ML from scratch, but understanding these concepts helps you:
- **Debug AI systems** when they behave unexpectedly
- **Choose the right approach** (fine-tuning vs prompting vs RAG)
- **Communicate with ML engineers** using the right terminology
- **Understand model cards** and benchmarks

**You won't implement these daily, but knowing them makes you a better AI developer.**

## Learning Objectives

By the end of this module, you will understand:
1. Features, labels, and how ML models learn
2. Train/test splits and why they matter
3. Overfitting and how to prevent it
4. When to use different ML approaches
5. How these concepts apply to LLMs

## Key Concepts

### Features and Labels

```
Features (X) = Input data the model uses to make predictions
Labels (y)   = Correct answers the model learns from

Example - Spam Detection:
┌─────────────────────────────────────┬───────────┐
│ Features (X)                        │ Label (y) │
├─────────────────────────────────────┼───────────┤
│ "Buy now! Limited offer!"           │ spam      │
│ "Meeting at 3pm tomorrow"           │ not_spam  │
│ "You won $1,000,000!!!"             │ spam      │
│ "Here's the report you requested"   │ not_spam  │
└─────────────────────────────────────┴───────────┘
```

### Train/Test Split

```
All Data (100 examples)
        │
        ├──► Training Set (80 examples)
        │    └── Model learns from these
        │
        └──► Test Set (20 examples)
             └── Evaluate model on unseen data

WHY? If you test on training data, you don't know if the model
     actually learned or just memorized the answers.
```

### Overfitting vs Underfitting

```
Underfitting          Just Right           Overfitting
(too simple)          (good fit)           (too complex)
     │                    │                     │
     ▼                    ▼                     ▼
  -------              ╭─────╮              ╭╮ ╭╮ ╭╮
                      ╱       ╲            ╱  ╲╱  ╲╱  ╲
 ● ● ●              ●╱  ●   ●  ╲●         ●    ●    ●
  ● ●              ●    ●  ●    ●        ●    ●    ●

Training: 60%        Training: 85%        Training: 99%
Test: 55%            Test: 83%            Test: 60%  ← Problem!

Memorized training data, fails on new data
```

### For LLM Developers: When This Matters

| Concept | How It Applies to LLMs |
|---------|------------------------|
| **Features** | The prompt IS your features - what you give the model |
| **Labels** | Expected outputs in fine-tuning datasets |
| **Train/Test** | Evaluation sets for RAG quality, fine-tuning |
| **Overfitting** | Model memorizes training examples instead of learning patterns |
| **Epochs** | How many times fine-tuning sees the training data |

## The ML Workflow

```
1. DATA COLLECTION
   └── Gather documents, examples, user queries

2. DATA PREPARATION
   └── Clean, chunk, label (Pandas skills!)

3. FEATURE ENGINEERING
   └── For LLMs: This is prompt engineering!

4. MODEL TRAINING/SELECTION
   └── Fine-tune OR choose base model + prompting

5. EVALUATION
   └── Test on held-out data, measure metrics

6. DEPLOYMENT
   └── API, caching, monitoring (Phase 5!)

7. MONITORING & ITERATION
   └── Track performance, retrain as needed
```

## Module Structure

```
03_ml_concepts/
├── README.md          # This file
├── examples.py        # Runnable code examples
├── exercises.py       # Practice problems
└── solutions/         # Exercise solutions
```

## Quick Start

```bash
# run the examples
uv run python phase1_foundations/03_ml_concepts/examples.py
```

## Topics Covered

### 1. Supervised vs Unsupervised Learning
- **Supervised**: Has labels (classification, regression)
- **Unsupervised**: No labels (clustering, embeddings)
- LLMs: Pre-training is unsupervised, fine-tuning is supervised

### 2. Classification vs Regression
- **Classification**: Predict categories (spam/not spam, sentiment)
- **Regression**: Predict numbers (price, score)
- LLMs: Text generation is actually "next token classification"

### 3. Training Dynamics
- **Epochs**: Full passes through training data
- **Batch size**: Examples processed together
- **Learning rate**: How fast the model updates
- For fine-tuning: These parameters matter!

### 4. Evaluation Metrics
- **Accuracy**: % correct (good for balanced data)
- **Precision/Recall**: Better for imbalanced data
- **F1 Score**: Balance between precision and recall
- For RAG: Relevance, faithfulness, groundedness

### 5. The Bias-Variance Tradeoff
- **High bias** = Underfitting (model too simple)
- **High variance** = Overfitting (model too complex)
- Sweet spot: Good performance on both train AND test data

## Connection to Your AI Journey

| ML Concept | Where You'll Use It |
|------------|---------------------|
| Train/test split | RAG evaluation, fine-tuning |
| Features | Prompt engineering is feature engineering! |
| Overfitting | Fine-tuning with too few examples |
| Metrics | Evaluating RAG quality (Phase 5) |
| Epochs | Fine-tuning hyperparameters |

## When to Fine-Tune vs Prompt vs RAG

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION TREE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Need custom knowledge?                                     │
│      │                                                      │
│      ├── Yes ──► Is it factual/retrievable?                │
│      │              │                                       │
│      │              ├── Yes ──► RAG                        │
│      │              │                                       │
│      │              └── No ──► Fine-tuning                 │
│      │                        (style, format, reasoning)    │
│      │                                                      │
│      └── No ──► Is base model good enough?                 │
│                    │                                        │
│                    ├── Yes ──► Prompting only              │
│                    │                                        │
│                    └── No ──► Fine-tuning                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Completed [NumPy basics](../01_numpy_basics/) (helpful but not required)
- Completed [Pandas basics](../02_pandas_basics/) (helpful but not required)
- Python fundamentals

## Next Steps

After this module:
1. **Phase 2**: Apply these concepts to [LLM APIs](../../phase2_llm_fundamentals/)
2. **Phase 3**: Use evaluation concepts in [RAG systems](../../phase3_llm_applications/)
3. **Phase 5**: Deep dive into [evaluation metrics](../../phase5_production/02_evaluation/)
4. **Phase 6**: Apply training concepts to [fine-tuning](../../phase6_model_customization/)

---

**This module is OPTIONAL for AI app development.**

The concepts here help you understand WHY things work, but you can build functional AI apps without implementing ML algorithms yourself.

---

**Run the examples:**
```bash
uv run python phase1_foundations/03_ml_concepts/examples.py
```