# Graphiti Knowledge Graph Integration

**Status**: Integrated and operational (2026-02-14)
**Purpose**: Enhanced project memory using knowledge graph + temporal tracking

## Architecture

**Graphiti MCP Server**:
- Running on localhost:8000 (MCP API endpoint)
- FalkorDB Backend (Redis-compatible graph database)
  - Database: port 6379
  - Web UI: port 3000 (browse graph visually)
- Storage: 181 nodes, 458 relationships (as of 2026-02-14)

**Multi-Project Setup**:
- Separation via `group_id` parameter
- python-ai-demo: Default group (learning project)
- python-demo: Separate group (FastAPI project)
- Can query single project or cross-project

## Integration Points

**Serena MCP** (Session Memory):
- Markdown files in `.serena/memories/`
- Session summaries, learning progress
- Human-readable, git-tracked

**Graphiti MCP** (Knowledge Graph):
- Graph database with entity extraction
- Temporal tracking (created_at, invalidation)
- Relationship mapping between concepts

**Separation**:
- `/sc:save` → Writes to Serena only
- Graphiti requires explicit `add_memory()` calls
- Independent but complementary systems

## Technical Details

**Entity Extraction**:
- LLM: gpt-4o-mini ($0.15/$0.60 per 1M tokens)
- Embeddings: text-embedding-3-small ($0.02 per 1M tokens)
- Cost: ~$0.30 per 100 episodes (~$0.002 per episode)

**Cost Optimization**:
- Ollama available for free local embeddings (nomic-embed-text)
- Saves 10-15% of total costs
- Configuration: `config-docker-falkordb.yaml`

## Known Issues

**Neo4j Migration Blocked**:
- Attempted migration from FalkorDB to Neo4j (2026-02-14)
- Failure: Graphiti MCP bug (version 0.23.1, pre-1.0)
- Error: `Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists`
- Decision: Stick with FalkorDB until bug fixed

**Pre-1.0 Software**:
- Graphiti is alpha/beta quality
- Expect bugs and breaking changes
- FalkorDB working correctly for current use case

## Usage Patterns

**Adding Episodes**:
```python
mcp__graphiti__add_memory(
    name="Episode Name",
    episode_body="content...",
    group_id="python-ai-demo",
    source="message"  # or "text" or "json"
)
```

**Searching**:
```python
# Search nodes (entities)
search_nodes("Neo4j migration", group_ids=["python-ai-demo"])

# Search facts (relationships)
search_memory_facts("cost analysis", group_ids=["python-ai-demo"])
```

## Value Proposition

**vs Vector DB (ChromaDB)**:
- Graph DB: Better for "why" questions, relationships, multi-hop reasoning
- Vector DB: Better for "what" questions, semantic similarity, fast retrieval
- Cost: Graph DB ~3x more ($0.30 vs $0.10 per 100 episodes)
- Not replacement, but complement to existing RAG system

**Use Cases**:
- Track learning progression (concept relationships)
- Map technical decisions and their rationale
- Historical context ("why did we choose X?")
- Cross-session knowledge accumulation

## Next Steps

- Continue using for session summaries (already added today's session)
- Monitor node/relationship growth over time
- Consider Ollama integration for cost reduction
- Report Neo4j migration bug to Graphiti team when appropriate
