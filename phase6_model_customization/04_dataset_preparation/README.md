# Module 04: Dataset Preparation

Quality data is more important than model architecture.

## Status: TODO

## Learning Objectives

- Format data for different training objectives
- Understand quality > quantity principle
- Generate synthetic training data with LLMs
- Create proper train/validation/test splits

## Key Concepts

### Data Formats

#### Instruction Tuning
```json
{
  "instruction": "Summarize this article",
  "input": "Article text here...",
  "output": "Summary here..."
}
```

#### Chat Format
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

#### Completion Format
```json
{
  "prompt": "Question: ...\nAnswer:",
  "completion": " The answer is..."
}
```

### Quality Principles
- 1,000 high-quality examples > 100,000 low-quality
- Diversity matters: cover edge cases
- Consistency: same format throughout
- No contamination: keep test set separate

## Planned Content

| File | Description |
|------|-------------|
| format_converters.py | Convert between formats |
| quality_filters.py | Filter low-quality samples |
| synthetic_data.py | Generate data with LLMs |
| data_splits.py | Proper train/val/test splits |
| exercises.py | Practice exercises |

## Prerequisites

- Understanding of LLM APIs (Phase 2)
- Basic Python data manipulation (Phase 1)
