# Session: 2026-02-25 - Phase 2: Embeddings Module Built

## What Was Done
Built the previously empty phase2_llm_fundamentals/03_embeddings/ module from scratch,
and added the missing phase2_llm_fundamentals/README.md.

## Files Created
- phase2_llm_fundamentals/README.md — phase-level overview with ASCII diagrams, module table, key patterns, Phase 2→3 connection table
- phase2_llm_fundamentals/03_embeddings/examples.py — 7 demos, all no-API-key, ruff clean
- phase2_llm_fundamentals/03_embeddings/README.md — with ASCII diagrams, cosine formula, semantic vs keyword table, RAG pipeline diagram

## 7 Demos in examples.py
1. What Are Embeddings? — fake 3D space (animal/technical/emotional axes) to build intuition
2. Vector Math — cosine similarity formula, why it beats euclidean distance (scale invariance)
3. Real Embeddings — sentence-transformers all-MiniLM-L6-v2, 384-dim output, 5×5 similarity matrix
4. Semantic Search — embed 10-doc corpus, rank by query similarity
5. Semantic vs Keyword — synonym trap demo (automobile/motor/ride all score high, keywords miss them)
6. Word Analogies — king - man + woman = queen ✅, paris - france + italy = rome ✅
7. RAG Connection — ASCII pipeline diagram showing how Demos 2-4 map directly to Phase 3 code

## Key Helpers
- cosine_similarity(vec_a, vec_b) — pure numpy implementation
- rank_by_similarity(query_vec, candidates, top_k) — returns sorted (label, score) pairs

## Technical Notes
- sentence-transformers runs locally (22MB download, no API key ever needed)
- all try/except ImportError blocks for graceful degradation
- Pyright shows reportMissingImports for sentence_transformers (no stubs) — expected, harmless
- ruff auto-fixed: import sort (cleandoc before numpy), 2 bare f-strings

## Phase 2 Status
All 3 modules complete: prompt_engineering + api_integration + embeddings
Phase README was missing — now added.
