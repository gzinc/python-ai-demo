# Module 03: LoRA and QLoRA

Efficient fine-tuning for large language models.

## Status: TODO

## Learning Objectives

- Understand LoRA: Low-Rank Adaptation
- Apply QLoRA for memory-efficient training
- Use HuggingFace PEFT library
- Fine-tune a small LLM on your GPU

## Key Concepts

### The Problem
- Full fine-tuning a 7B model needs ~28GB VRAM
- Your GPU (5070 Ti ~16GB) can't handle that
- Solution: train only small adapter matrices

### LoRA Intuition
```
Original weights: W (frozen, huge)
LoRA adapters: A × B (trainable, tiny)

Output = W × input + (A × B) × input

Instead of updating millions of weights,
train two small matrices that adjust the output.
```

### Memory Savings
| Technique | 7B Model VRAM | Trainable Params |
|-----------|---------------|------------------|
| Full fine-tune | 28+ GB | 7B (100%) |
| LoRA | 12-16 GB | ~8M (0.1%) |
| QLoRA | 6-10 GB | ~8M (0.1%) |

## Planned Content

| File | Description |
|------|-------------|
| lora_basics.py | LoRA concept demonstration |
| qlora_phi2.py | Fine-tune Phi-2 with QLoRA |
| peft_tutorial.py | HuggingFace PEFT library |
| merge_adapters.py | Merge LoRA into base model |
| exercises.py | Practice exercises |

## Prerequisites

- Module 01: Training fundamentals
- Module 02: Fine-tuning basics
- GPU with 12GB+ VRAM

## Libraries

```python
peft          # Parameter-Efficient Fine-Tuning
bitsandbytes  # Quantization for QLoRA
accelerate    # Distributed training
transformers  # Model loading
```
