"""
ML Concepts - Practical Examples for AI Developers

This module demonstrates core ML concepts that help you understand
how AI systems work. You don't need to implement these daily,
but understanding them makes you a better AI developer.

Run with: uv run python phase1_foundations/03_ml_concepts/examples.py
"""

import numpy as np
from typing import Tuple, List
import random


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# =============================================================================
# 1. FEATURES AND LABELS
# =============================================================================

def features_and_labels_demo() -> None:
    """demonstrate the concept of features (X) and labels (y)"""
    print_section("Features and Labels")

    # simple spam detection example
    emails = [
        {"text": "Buy now! Limited offer!", "has_exclamation": True, "word_count": 4},
        {"text": "Meeting at 3pm tomorrow", "has_exclamation": False, "word_count": 4},
        {"text": "You won $1,000,000!!!", "has_exclamation": True, "word_count": 3},
        {"text": "Here's the report", "has_exclamation": False, "word_count": 3},
    ]

    labels = ["spam", "not_spam", "spam", "not_spam"]

    print("Spam Detection Dataset:")
    print("-" * 50)
    for email, label in zip(emails, labels):
        print(f"  '{email['text'][:30]}...' → {label}")

    # convert to numeric features
    X = np.array([[e["has_exclamation"], e["word_count"]] for e in emails])
    y = np.array([1 if l == "spam" else 0 for l in labels])

    print(f"\nAs numeric arrays:")
    print(f"  Features (X) shape: {X.shape}")
    print(f"  Labels (y) shape: {y.shape}")
    print(f"\n  X = {X}")
    print(f"  y = {y}")

    print("\n💡 For LLMs: The PROMPT is your features!")
    print("   Your prompt engineering IS feature engineering.")


# =============================================================================
# 2. TRAIN/TEST SPLIT
# =============================================================================

def train_test_split_demo() -> None:
    """demonstrate why we split data into train and test sets"""
    print_section("Train/Test Split")

    # create sample data
    np.random.seed(42)
    data = list(range(100))

    # manual split (80/20)
    random.seed(42)
    random.shuffle(data)

    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    print(f"Total samples: {len(data)}")
    print(f"Training set: {len(train_data)} samples (80%)")
    print(f"Test set: {len(test_data)} samples (20%)")

    print("\n📊 Why split?")
    print("  - Train: Model learns patterns from this data")
    print("  - Test: Check if model works on NEW, unseen data")
    print("  - If you test on training data, you're just checking memorization!")

    # using sklearn-style function
    print("\n🔧 In practice, use sklearn:")
    print("  from sklearn.model_selection import train_test_split")
    print("  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)")

    print("\n💡 For LLMs/RAG:")
    print("  - Hold out test queries to evaluate retrieval quality")
    print("  - Don't tune prompts on your test set!")


def simple_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """simple train/test split implementation"""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))

    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =============================================================================
# 3. OVERFITTING DEMONSTRATION
# =============================================================================

def overfitting_demo() -> None:
    """demonstrate overfitting vs good generalization"""
    print_section("Overfitting vs Generalization")

    # simulate training and test accuracy for different model complexities
    print("Imagine training models of increasing complexity:\n")

    results = [
        ("Too Simple", 60, 58, "Underfitting - model can't capture patterns"),
        ("Simple", 75, 72, "Getting better..."),
        ("Medium", 85, 83, "Good fit! Similar train/test performance"),
        ("Complex", 92, 78, "Starting to overfit..."),
        ("Very Complex", 99, 65, "Overfitting! Memorized training data"),
    ]

    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Status'}")
    print("-" * 70)
    for model, train_acc, test_acc, status in results:
        gap = train_acc - test_acc
        flag = "⚠️" if gap > 10 else "✅" if gap < 5 else ""
        print(f"{model:<15} {train_acc}%{'':<10} {test_acc}%{'':<10} {flag} {status}")

    print("\n🎯 Goal: Small gap between training and test accuracy")
    print("   Large gap = model memorized training data, won't generalize")

    print("\n💡 For LLM Fine-tuning:")
    print("  - Too few epochs: Underfitting (model hasn't learned)")
    print("  - Too many epochs: Overfitting (memorizes exact examples)")
    print("  - Sweet spot: Usually 1-3 epochs for LLMs!")


# =============================================================================
# 4. EVALUATION METRICS
# =============================================================================

def evaluation_metrics_demo() -> None:
    """demonstrate different evaluation metrics"""
    print_section("Evaluation Metrics")

    # simulated predictions
    actual =    [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]  # true labels
    predicted = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # model predictions

    # calculate metrics manually
    tp = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)  # true positive
    tn = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)  # true negative
    fp = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)  # false positive
    fn = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)  # false negative

    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Pos   Neg")
    print(f"  Actual Pos      {tp}     {fn}    (TP, FN)")
    print(f"         Neg      {fp}     {tn}    (FP, TN)")

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.2%} - Overall correct predictions")
    print(f"  Precision: {precision:.2%} - Of predicted positives, how many correct?")
    print(f"  Recall:    {recall:.2%} - Of actual positives, how many found?")
    print(f"  F1 Score:  {f1:.2%} - Balance of precision and recall")

    print("\n💡 For RAG Systems (Phase 5):")
    print("  - Relevance: Are retrieved docs related to query?")
    print("  - Faithfulness: Does answer match retrieved context?")
    print("  - Groundedness: Is answer supported by evidence?")


# =============================================================================
# 5. LEARNING RATE INTUITION
# =============================================================================

def learning_rate_demo() -> None:
    """demonstrate learning rate concept"""
    print_section("Learning Rate Intuition")

    print("Learning rate = How big of a step the model takes when learning\n")

    # simulate gradient descent with different learning rates
    def simulate_learning(lr: float, target: float = 10.0, start: float = 0.0, steps: int = 10) -> List[float]:
        """simulate simple gradient descent"""
        value = start
        history = [value]
        for _ in range(steps):
            gradient = target - value  # simplified gradient
            value = value + lr * gradient
            history.append(value)
        return history

    target = 10.0
    print(f"Goal: Reach {target}\n")

    rates = [
        (0.01, "Too small - very slow convergence"),
        (0.1, "Good - steady progress"),
        (0.5, "Fast - quick but may overshoot"),
        (1.5, "Too large - oscillates wildly"),
    ]

    for lr, description in rates:
        history = simulate_learning(lr, target, steps=6)
        values = " → ".join(f"{v:.1f}" for v in history[:5])
        final = history[-1]
        status = "✅" if abs(final - target) < 1 else "⚠️"
        print(f"LR={lr:<4} : {values}... → {final:.1f} {status} ({description})")

    print("\n💡 For LLM Fine-tuning:")
    print("  - Default learning rates: 1e-5 to 5e-5 (very small!)")
    print("  - Too high: Model forgets pre-training, outputs garbage")
    print("  - Too low: Takes forever, may not learn new task")


# =============================================================================
# 6. WHEN TO USE WHAT
# =============================================================================

def decision_framework_demo() -> None:
    """show when to use prompting vs RAG vs fine-tuning"""
    print_section("Decision Framework: Prompting vs RAG vs Fine-tuning")

    scenarios = [
        {
            "need": "Answer questions about company docs",
            "solution": "RAG",
            "why": "Factual, retrievable information"
        },
        {
            "need": "Respond in brand voice/style",
            "solution": "Fine-tuning",
            "why": "Style/format isn't retrievable"
        },
        {
            "need": "Translate text to French",
            "solution": "Prompting",
            "why": "Base model already knows how"
        },
        {
            "need": "Follow specific output format",
            "solution": "Prompting + Examples",
            "why": "Few-shot learning works well"
        },
        {
            "need": "Domain-specific reasoning",
            "solution": "Fine-tuning + RAG",
            "why": "Need both knowledge and reasoning patterns"
        },
    ]

    print(f"{'Need':<40} {'Solution':<20} {'Why'}")
    print("-" * 90)
    for s in scenarios:
        print(f"{s['need']:<40} {s['solution']:<20} {s['why']}")

    print("\n📊 Cost Comparison:")
    print("  Prompting:    ~$0 (just API calls)")
    print("  RAG:          ~$100-1000 (embedding + vector DB)")
    print("  Fine-tuning:  ~$50-5000+ (compute + data prep)")

    print("\n🎯 Rule of thumb:")
    print("  1. Try prompting first (cheapest, fastest)")
    print("  2. Add RAG for custom knowledge")
    print("  3. Fine-tune only if above doesn't work")


# =============================================================================
# 7. EMBEDDINGS AS FEATURES
# =============================================================================

def embeddings_as_features_demo() -> None:
    """show how embeddings ARE feature engineering"""
    print_section("Embeddings = Automatic Feature Engineering")

    print("Traditional ML: You manually create features")
    print("-" * 50)
    print("  Text: 'I love this movie!'")
    print("  Manual features:")
    print("    - word_count: 4")
    print("    - has_exclamation: True")
    print("    - positive_words: ['love']")
    print("    - negative_words: []")
    print("    ... tedious, domain-specific, incomplete")

    print("\nWith Embeddings: Model learns features automatically")
    print("-" * 50)
    print("  Text: 'I love this movie!'")
    print("  Embedding: [0.23, -0.15, 0.87, ..., 0.42]  (768+ dimensions)")
    print("    - Captures meaning, sentiment, context")
    print("    - Works across domains")
    print("    - No manual feature engineering!")

    print("\n💡 This is why embeddings revolutionized NLP:")
    print("  - Before: Spend weeks on feature engineering")
    print("  - After: Just embed the text, model handles the rest")

    # simulated similarity
    print("\n📊 Embedding similarity example:")
    texts = [
        ("I love this movie!", "This film is amazing!"),
        ("I love this movie!", "The stock market crashed"),
    ]
    # simulated similarities
    sims = [0.92, 0.15]

    for (t1, t2), sim in zip(texts, sims):
        print(f"  '{t1}' vs '{t2}'")
        print(f"    Similarity: {sim:.2f} {'(similar)' if sim > 0.5 else '(different)'}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """run all examples"""
    print("\n" + "=" * 60)
    print("  ML Concepts - Essential Knowledge for AI Developers")
    print("=" * 60)
    print("\n⚡ Note: You don't implement these daily, but understanding")
    print("   them makes you a better AI developer.\n")

    features_and_labels_demo()
    train_test_split_demo()
    overfitting_demo()
    evaluation_metrics_demo()
    learning_rate_demo()
    decision_framework_demo()
    embeddings_as_features_demo()

    print("\n" + "=" * 60)
    print("  ✅ All concepts covered!")
    print("=" * 60)
    print("\n📚 Key Takeaways:")
    print("  1. Features = model inputs (prompts!), Labels = expected outputs")
    print("  2. Always split data: train on some, test on others")
    print("  3. Watch for overfitting: training acc >> test acc is bad")
    print("  4. Start with prompting, add RAG, fine-tune last")
    print("  5. Embeddings = automatic feature engineering")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()