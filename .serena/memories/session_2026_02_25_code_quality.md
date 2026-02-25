# Session: 2026-02-25 - Code Quality Pass (Regions + Decorators)

## What Was Done

### 1. @requires_* Decorator Migration (02_llm_integration/practical.py)
- Replaced manual `check_api_keys()` + early return guards in 4 demos
- Demo 1 → `@requires_openai`, Demo 2 → `@requires_anthropic`
- Demos 5 & 6 → `@requires_both_keys`
- Pattern: decorator handles guard, function body starts clean

### 2. # region / # endregion Blocks (Project-Wide)
- Replaced `# === PART N: Title ===` style dividers across ~100 files
- Covered: phases 1-7, common/, tests/
- Files with `# ===` but NO regions → all converted
- Region naming: `# region Demo N: Title`, `# region Utilities`, etc.
- Subagent ran conversion; introduced 3 bugs that were caught and fixed

### 3. Bug Fixes from Subagent Region Pass
- `phase4_ai_agents/02_tool_use/examples.py` DEMOS — had 4 invented function names
  Fixed to: example_tool_schemas, example_file_tools, example_web_search,
  example_http_tool, example_tool_registry, example_custom_tool
- `phase4_ai_agents/03_multi_agent/examples.py` DEMOS — had 4 invented names
  Fixed to: demo_basic_delegation, demo_research_pipeline, demo_direct_specialists,
  demo_custom_specialist, demo_team_config, demo_real_api
- `langchain_concepts_demo.py` DEMOS — had 5 invented names
  Fixed to: demo_rag_setup, demo_rag_query, demo_memory, demo_chains,
  demo_practical_examples, demo_decision_framework, demo_recommendations
- `orchestrator.py` — removed unused `import json` added by agent

### 4. Ruff Auto-Fix
- 336 issues fixed automatically (import sorting, deprecated typing, trailing newlines)
- 48 remaining are pre-existing (unused vars in demo code, E402 sys.path patterns)
- Pre-existing F823/F402 false positive on `@tool` decorator in 06_agents_tools/practical.py

## Patterns Established
- Always verify DEMOS list function references after any bulk refactor
- Subagent region conversion is reliable for structure but may invent DEMOS names
- `@requires_openai/anthropic/both_keys` decorators are the canonical pattern
- `check_api_keys()` in `main()` for MenuRunner `has_api=` param is still correct

## Files NOT Changed (pre-existing issues left alone)
- LlamaIndex scaffolded files (01_basic_indexing, 03_advanced_rag) — still use old pattern in main()
- Pyright false positives on execute() override in tool demo code
