# Session: 2025-12-20 - LLMLingua-2 Deep Dive

## Module 3: Optimization - Prompt Compression

### What We Built
- Production prompt compression using Microsoft LLMLingua-2
- Modular structure: `03_optimization/01_compression/token_optimization.py`
- Abstract `PromptCompressor` base with two implementations:
  - `LLMLingua2Compressor` - production (BERT-based, 50-70% reduction)
  - `NaiveCompressor` - demo/fallback (regex-based, ~25% reduction)

### LLMLingua-2 Key Concepts

**Why LLMLingua-2 over v1:**
- BERT-based (trained specifically for compression task)
- 3x faster than LLMLingua-1
- Better quality (trained on GPT-4 distillation data)
- Task-agnostic (works for any prompt type)

**Compression rates:**
| Rate | Meaning | Use Case |
|------|---------|----------|
| 0.7 | Keep 70% | Light compression |
| 0.5 | Keep 50% | Balanced (default) |
| 0.3 | Keep 30% | Aggressive |

**Actual results from our demo:**
- rate=0.7: 32.9% reduction
- rate=0.5: 52.9% reduction  
- rate=0.3: 70.6% reduction

### Model Architecture Deep Dive

**Where model is stored:**
```
~/.cache/huggingface/hub/models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank/
├── blobs/      # actual files (content-addressed by SHA)
├── snapshots/  # symlinks organized by commit
└── refs/       # branch pointers
```

**What's in the 677MB weights file:**
- 177 million floating point numbers (parameters)
- Format: SafeTensors (not pickle)

**Model structure (199 tensors):**
```
EMBEDDINGS (92M params)
├── word_embeddings     [119647, 768]  → 92M (vocab → vectors)
├── position_embeddings [512, 768]     → 393K
└── token_type_embeddings [2, 768]     → 1.5K

TRANSFORMER LAYERS (12 layers × 7M each = 85M params)
├── attention.query/key/value  [768, 768] × 3
├── attention.output.dense     [768, 768]
├── intermediate.dense         [3072, 768]
└── output.dense               [768, 3072]

CLASSIFIER HEAD
└── Binary classification: keep/drop per token
```

**Why loading is slow even when cached:**
- Download: once (first run)
- Loading: every instantiation
  - Deserialize 677MB from disk
  - Allocate RAM for 177M parameters
  - Initialize PyTorch tensors
  - Build computation graph

### Lazy Import Pattern

```python
class LLMLingua2Compressor:
    def __init__(self, ...):
        # import inside __init__, not at top of file
        from llmlingua import PromptCompressor as LLMCompressor
        self._compressor = LLMCompressor(...)
```

**Benefits:**
- `import token_optimization` → fast, no torch loaded
- `LLMLingua2Compressor()` → slow, but only when needed
- If only using `NaiveCompressor`, torch never loads

### CUDA, PyTorch, Tensors

**Tensor** = multi-dimensional array of numbers
```python
# 0D: scalar     42
# 1D: vector     [1, 2, 3]
# 2D: matrix     [[1, 2], [3, 4]]
# 3D+: stacks    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

**PyTorch (torch)** = tensor library with GPU support + auto-gradients
- Like NumPy but runs on GPU
- Automatic differentiation for training
- Foundation for most AI/ML in Python

**CUDA** = NVIDIA's parallel computing platform
```
CPU: 4-16 powerful cores, sequential
GPU: thousands of tiny cores, parallel

Neural nets = massive matrix math = embarrassingly parallel
→ GPUs are 2-100x faster for AI workloads
```

### GPU vs CPU Benchmark (RTX 5070 Ti)

| Metric | GPU | CPU |
|--------|-----|-----|
| Load time | 1.54s | 0.70s |
| Inference | 109ms | 260ms |
| Speedup | 2.4x | baseline |

**When to use GPU:**
- High volume (thousands/min)
- Very long prompts (>10K tokens)
- Training models

**For learning/dev:** CPU is fine (~260ms per compression)

### Alternative Compression Methods (2025)

| Method | Type | Notes |
|--------|------|-------|
| LLMLingua-2 | Hard prompt | ✅ Best OSS, production-ready |
| SCOPE (2025) | Generative | No training, embedding+summarization |
| LongLLMLingua | Hard prompt | Better for very long contexts |
| Soft prompts | Embedding | Different approach (xRAG, PISCO) |
| TokenCrush | Commercial | LangChain integration |

### Files Created/Modified

- `phase5_production/03_optimization/01_compression/token_optimization.py`
- `phase5_production/03_optimization/01_compression/README.md`
- Split module into subdirectories matching Phase 4 structure

### Key Insight

The 677MB file is just 177 million decimal numbers learned through training. When you compress a prompt, input tokens flow through these weight matrices as matrix multiplications, producing keep/drop decisions per token.

```
Input → Embeddings → 12 Transformer Layers → Classifier → [0.92, 0.08] → keep
```
