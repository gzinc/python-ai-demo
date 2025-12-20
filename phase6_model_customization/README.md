# Phase 6: Model Customization

Move beyond API usage to customize and fine-tune models for specific use cases.

## Learning Objectives

By the end of this phase, you will:
- Understand the training loop: forward pass → loss → backward pass → update
- Fine-tune pre-trained models (BERT, small LLMs) for specific tasks
- Apply efficient fine-tuning techniques (LoRA, QLoRA)
- Prepare and validate datasets for training
- Know when to fine-tune vs prompt engineer vs RAG

## Prerequisites

- Phase 1-5 completed (especially embeddings, RAG concepts)
- Understanding of tensors and PyTorch basics
- GPU available (RTX 5070 Ti or similar with 12GB+ VRAM)
- CUDA-enabled PyTorch installed for experiments

## Modules

### [01_training_fundamentals/](01_training_fundamentals/)
**The Training Loop** - Understand how models learn

| Topic | Description |
|-------|-------------|
| Forward pass | Input → model → prediction |
| Loss calculation | How wrong was the prediction? |
| Backward pass | Compute gradients (blame assignment) |
| Optimizer step | Update weights to reduce loss |
| GPU acceleration | Why training needs CUDA |

**Hands-on**: Train a simple neural network on MNIST from scratch

---

### [02_fine_tuning_basics/](02_fine_tuning_basics/)
**Transfer Learning** - Start from pre-trained models

| Topic | Description |
|-------|-------------|
| Why fine-tune | Leverage existing knowledge |
| BERT fine-tuning | Text classification task |
| Freezing layers | What to train vs keep fixed |
| Learning rates | Lower than training from scratch |
| Overfitting | When your model memorizes |

**Hands-on**: Fine-tune BERT for sentiment classification

---

### [03_lora_qlora/](03_lora_qlora/)
**Efficient LLM Fine-Tuning** - Modern techniques for large models

| Topic | Description |
|-------|-------------|
| LoRA concept | Low-Rank Adaptation - train small matrices |
| QLoRA | Quantized LoRA - even more memory efficient |
| PEFT library | HuggingFace's parameter-efficient fine-tuning |
| Adapter patterns | Plug-in modules for customization |
| Merging weights | Combine LoRA back into base model |

**Hands-on**: LoRA fine-tune a small LLM (Phi-2 or similar)

---

### [04_dataset_preparation/](04_dataset_preparation/)
**Data for Training** - Garbage in, garbage out

| Topic | Description |
|-------|-------------|
| Data formats | Instruction tuning, chat, completion |
| Quality over quantity | Why 1K good examples beats 100K bad |
| Data augmentation | Expand dataset intelligently |
| Train/val/test splits | Proper evaluation setup |
| Synthetic data | Using LLMs to generate training data |

**Hands-on**: Prepare a custom dataset for instruction tuning

---

### [05_evaluation_selection/](05_evaluation_selection/)
**When to Fine-Tune** - Decision framework

| Approach | When to Use |
|----------|-------------|
| Prompt engineering | Quick iteration, general tasks |
| RAG | Need current/private knowledge |
| Fine-tuning | Specific style, domain, or behavior |
| From scratch | Novel architecture (rarely needed) |

**Hands-on**: Compare approaches on a real task

---

## Hardware Requirements

| Technique | VRAM Needed | Your GPU (5070 Ti) |
|-----------|-------------|-------------------|
| Small model from scratch | 2-4 GB | ✅ Easy |
| BERT fine-tuning | 4-8 GB | ✅ Easy |
| LoRA on 7B model | 8-12 GB | ✅ Possible |
| QLoRA on 7B model | 6-10 GB | ✅ Comfortable |
| Full fine-tune 7B | 28+ GB | ❌ Too big |

## Quick Start

```bash
# ensure CUDA PyTorch for experiments
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# run first experiment
uv run python phase6_model_customization/01_training_fundamentals/mnist_from_scratch.py
```

## Key Libraries

```python
torch          # training framework
transformers   # pre-trained models
peft           # LoRA/QLoRA implementation
datasets       # data loading and processing
accelerate     # distributed training helpers
bitsandbytes   # quantization for QLoRA
```

## Connection to Earlier Phases

| Phase | Connection |
|-------|------------|
| Phase 1 | NumPy arrays → PyTorch tensors |
| Phase 2 | Embeddings → now you see how they're trained |
| Phase 3 | RAG retrieval → alternative to fine-tuning |
| Phase 4 | Agents → can use fine-tuned models |
| Phase 5 | Production → deploy your fine-tuned models |

## Status

- [ ] Module 01: Training Fundamentals - **TODO**
- [ ] Module 02: Fine-Tuning Basics - **TODO**
- [ ] Module 03: LoRA/QLoRA - **TODO**
- [ ] Module 04: Dataset Preparation - **TODO**
- [ ] Module 05: Evaluation & Selection - **TODO**

---

**Note**: This phase is exploratory. Jump between modules based on interest. The goal is hands-on experimentation, not linear completion.
