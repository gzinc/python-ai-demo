# Module 05: Evaluation and Selection

Know when to fine-tune vs use other approaches.

## Status: TODO

## Learning Objectives

- Evaluate fine-tuned models properly
- Understand when fine-tuning is worth it
- Compare: prompt engineering vs RAG vs fine-tuning
- Make data-driven decisions about model customization

## Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│              WHEN TO USE EACH APPROACH                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROMPT ENGINEERING                                         │
│  ✓ Quick iteration needed                                   │
│  ✓ General task, standard behavior                          │
│  ✓ No training data available                               │
│  ✓ Flexibility important                                    │
│                                                             │
│  RAG (Retrieval-Augmented Generation)                       │
│  ✓ Need current/private knowledge                           │
│  ✓ Knowledge changes frequently                             │
│  ✓ Need citations/sources                                   │
│  ✓ Large knowledge base                                     │
│                                                             │
│  FINE-TUNING                                                │
│  ✓ Specific style/tone/format                               │
│  ✓ Domain-specific behavior                                 │
│  ✓ High-volume, low-latency needs                           │
│  ✓ Proprietary task patterns                                │
│                                                             │
│  TRAIN FROM SCRATCH                                         │
│  ✓ Novel architecture needed (rare)                         │
│  ✓ Massive compute budget                                   │
│  ✓ Unique modality or task                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Evaluation Metrics

| Metric | What It Measures |
|--------|------------------|
| Perplexity | How "surprised" the model is |
| Task accuracy | Correct outputs for your task |
| Human eval | Quality rated by humans |
| A/B testing | Real-world performance |

## Planned Content

| File | Description |
|------|-------------|
| evaluation_metrics.py | Common evaluation approaches |
| comparison_study.py | Compare approaches on same task |
| cost_analysis.py | Training vs inference costs |
| decision_tree.py | Interactive decision helper |
| exercises.py | Practice exercises |

## Prerequisites

- Complete Modules 01-04
- Understanding of RAG (Phase 3)
- Production considerations (Phase 5)

## Connection to Production

This module bridges customization (Phase 6) back to production (Phase 5):
- Evaluate before deploying
- Monitor after deploying
- Iterate based on real usage
