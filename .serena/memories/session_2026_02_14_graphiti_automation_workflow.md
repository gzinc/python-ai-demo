# Session: 2026-02-14 - Graphiti Automation Workflow

## What I Learned

**Graphiti Knowledge Graph Integration**:
- Graphiti MCP requires explicit `add_memory()` calls (not automatic)
- Episodes processed in background (entity extraction + relationship detection)
- Knowledge evolves through: extraction → relationships → temporal tracking → graph merge
- Current graph state: ~193 nodes, ~470 relationships (after today's session)

**Knowledge Evolution Process**:
- Stage 1: LLM extracts entities from episode text
- Stage 2: LLM identifies relationships between entities  
- Stage 3: System adds temporal metadata (created_at, valid_from, valid_until)
- Stage 4: Graph merges new knowledge with existing nodes/relationships

**Automatic Triggering Options Explored**:
- Option 1: Extend /sc:save command (requires skill creation)
- Option 2: Post-session hook (fully automatic, complex setup)
- Option 3: Periodic snapshots (30-min intervals, background process)
- Option 4: Watch Serena memories (file watcher script)
- **Decision**: Integrated workflow - Claude automatically does both Serena + Graphiti on /sc:save

## Implementation Completed

**Integrated /sc:save Workflow**:
1. User types `/sc:save` (single command)
2. Claude saves to Serena MCP (session summary)
3. Claude saves to Graphiti MCP (knowledge graph episode)
4. Both operations complete in one response

**Benefits**:
- No scripts, no watchers, no background processes
- Simple one-command workflow
- Guaranteed synchronization (both always updated together)
- No additional cognitive overhead (same command as before)

## Git Commit Completed

**Commit**: a34a1f4
**Message**: docs: phase 7 scaffolding + graphiti integration session
**Files**: 18 files changed, 6100 insertions(+), 245 deletions(-)

**What was committed**:
- Phase 7 scaffolding (LangGraph 6 files, LlamaIndex 4 files)
- Graphiti session memories (3 memory files)
- Documentation updates (CLAUDE.md: 5→7 phases, Phase 6-7 guidance)
- Docker setup (docker-compose.yml for Graphiti + FalkorDB)

**What was excluded**:
- ChromaDB database files (binary, frequently changing, in .gitignore)

## Key Insights

**Temporal Knowledge** (not temporary):
- Graphiti tracks when facts were created
- Supports fact invalidation when new information supersedes old
- Example: Cost estimate $100-300 → corrected to $0.30, old fact marked invalid

**Multi-Hop Reasoning**:
- Graph enables complex queries across multiple relationships
- Example: "Why FalkorDB over Neo4j?" → traces through migration failure → Graphiti MCP bug → decision to stick with FalkorDB

**Graph vs Vector DB**:
- Graph DB: Better for "why" questions, relationships, causal chains (3x cost)
- Vector DB: Better for "what" questions, semantic similarity, fast retrieval (1x cost)
- Not replacement, but complement to existing RAG system

## Phase 7 Status Documented

**Current State** (50% complete):
- LangChain basics: 75% (Demos 6-7 pending in agents_tools)
- LangGraph: 17% (Demo 4/6 in state_basics in progress)
- LlamaIndex: 0% (scaffolded only)

**Strategic Decision**:
- LangGraph/LlamaIndex optional for roadmap completion (87% done without them)
- Can build production AI apps with Phases 2-5 knowledge + LangChain basics
- Files committed as scaffolding (shows planning, allows future continuation)

## Next Steps

**Immediate**:
- /sc:save now auto-syncs to both Serena + Graphiti (no scripts needed)
- Continue LangGraph state basics (complete Demos 4-6)

**Optional**:
- Explore conditional routing (02_conditional_routing.py)
- Start real project using Phases 2-5 + LangChain basics
- Add Ollama for cost optimization (free local embeddings)

## Session Summary

Duration: ~90 minutes
Focus: Graphiti automation, commit workflow, documentation updates
Outcome: Integrated /sc:save workflow (Serena + Graphiti in one command)
