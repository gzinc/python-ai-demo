# Module 01: Training Fundamentals

Understand how neural networks learn through the training loop.

## Learning Objectives

- Understand the 4-step training loop: forward → loss → backward → update
- See gradients in action (automatic differentiation)
- Experience GPU vs CPU training speed difference
- Build intuition for hyperparameters (learning rate, batch size, epochs)

## The Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    THE TRAINING LOOP                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. FORWARD PASS                                           │
│      input → model → prediction                             │
│      "What does the model think this image is?"             │
│                                                             │
│   2. LOSS CALCULATION                                       │
│      loss = how_wrong(prediction, actual_label)             │
│      "How far off was the guess?"                           │
│                                                             │
│   3. BACKWARD PASS (backpropagation)                        │
│      loss.backward() → compute gradients                    │
│      "Which weights contributed to the error?"              │
│                                                             │
│   4. OPTIMIZER STEP                                         │
│      optimizer.step() → update weights                      │
│      "Nudge weights to reduce future error"                 │
│                                                             │
│   Repeat thousands of times...                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Gradient Descent
- **Gradient**: direction of steepest increase in loss
- **Descent**: we go the opposite direction (to reduce loss)
- Each weight gets nudged by: `weight -= learning_rate * gradient`

### Learning Rate
- Too high: overshoots, loss explodes
- Too low: very slow learning
- Typical values: 0.001, 0.0001

### Batch Size
- How many samples to process before updating weights
- Larger batch = more stable gradients, more memory
- Typical values: 32, 64, 128

### Epochs
- One epoch = one pass through all training data
- More epochs = more learning (but risk of overfitting)

## Hands-On Experiments

### Basic Run
```bash
uv run python phase6_model_customization/01_training_fundamentals/mnist_from_scratch.py
```

### GPU Mode (if available)
```bash
# install CUDA PyTorch first
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# run with GPU
uv run python phase6_model_customization/01_training_fundamentals/mnist_from_scratch.py --gpu
```

### Experiment with Hyperparameters
```bash
# more epochs (watch for overfitting)
uv run python ... --epochs 10

# higher learning rate (watch for instability)
uv run python ... --lr 0.01

# lower learning rate (slower but stable)
uv run python ... --lr 0.0001

# larger batch size
uv run python ... --batch-size 128
```

## What to Watch For

| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| Training loss | Decreasing | Stuck or increasing |
| Test accuracy | Increasing | Much lower than train (overfitting) |
| Loss values | 0.1 - 2.0 | NaN, inf, or very high |

## Connection to LLMs

Everything you see here applies to LLM training:

| This Experiment | LLM Training |
|-----------------|--------------|
| 235K parameters | Billions of parameters |
| Images → digits | Text → next token |
| CrossEntropyLoss | Same loss function |
| Adam optimizer | Same or AdamW |
| Batches of 64 | Batches of millions of tokens |

The **concepts are identical** - just scaled up massively.

## Files

| File | Description |
|------|-------------|
| [mnist_from_scratch.py](mnist_from_scratch.py) | Complete training script |
| [exercises.py](exercises.py) | TODO: Practice exercises |
| [solutions/](solutions/) | TODO: Exercise solutions |

## Next Steps

After understanding the training loop:
- → Module 02: Fine-tune pre-trained models (don't start from scratch)
- → Module 03: LoRA (efficient fine-tuning for large models)
