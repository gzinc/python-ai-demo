# Module 02: Fine-Tuning Basics

Transfer learning - start from pre-trained models instead of random weights.

## Status: TODO

## Learning Objectives

- Understand transfer learning: leverage existing knowledge
- Fine-tune BERT for text classification
- Learn when to freeze vs train layers
- Handle overfitting on small datasets

## Key Concepts

### Why Fine-Tune?
- Pre-trained models already understand language/images
- You just teach them your specific task
- Requires much less data and compute than training from scratch

### Transfer Learning Flow
```
Pre-trained Model (general knowledge)
         ↓
    Add task head (classification layer)
         ↓
    Train on your data (small dataset OK)
         ↓
    Fine-tuned Model (your specific task)
```

## Planned Content

| File | Description |
|------|-------------|
| bert_classification.py | Fine-tune BERT for sentiment |
| freezing_layers.py | Compare frozen vs unfrozen |
| learning_rates.py | Why lower LR for fine-tuning |
| exercises.py | Practice exercises |

## Prerequisites

- Complete Module 01 (understand training loop)
- Understanding of transformers (Phase 2)

## Connection to Earlier Learning

- Phase 2 embeddings: now you'll see how they're trained
- LLMLingua-2 BERT: same architecture, different task head
