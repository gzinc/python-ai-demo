# Learning Progress Tracker

Track your journey through AI development. Update after each session.

## Phase 1: Foundations

### NumPy Basics
- [x] Array creation and manipulation
- [x] Vectorized operations
- [x] Broadcasting
- [x] Indexing and slicing
- [ ] Reshaping and stacking
- [x] Performance optimization
- [ ] Exercises completed
- [ ] Project: Similarity search

### Pandas Data Manipulation
- [ ] DataFrames and Series
- [ ] Loading and saving data
- [ ] Data cleaning
- [ ] Filtering and grouping
- [ ] Merging datasets
- [ ] Exercises completed
- [ ] Project: Data analysis report

### ML Concepts
- [ ] Features and labels
- [ ] Train/test splits
- [ ] Feature engineering
- [ ] Normalization
- [ ] Basic ML workflow
- [ ] Exercises completed
- [ ] Project: Data pipeline

**Phase 1 Completion**: 33% (1/3 modules in progress)

---

## Phase 2: LLM Fundamentals

### Prompt Engineering
- [x] Basic prompting techniques
- [x] System prompts
- [x] Few-shot learning
- [x] Chain-of-thought
- [x] Output formatting
- [x] RAG prompt patterns
- [x] Error handling in prompts
- [x] Temperature and parameters

### API Integration
- [x] OpenAI API patterns
- [x] Anthropic API patterns
- [x] Azure OpenAI vs OpenAI vs AWS Bedrock differences
- [x] Token management and cost tracking
- [x] Error handling with retry/backoff
- [x] Streaming responses
- [x] Multi-turn conversation history (messages=[])
- [x] .env setup with python-dotenv

### Embeddings
- [x] Text embeddings basics
- [x] Semantic similarity
- [x] Vector databases
- [x] Similarity search
- [x] ChromaDB integration

**Phase 2 Completion**: 100% (All 3 modules completed! Prompt Engineering + API Integration + Embeddings)

---

## Phase 3: LLM Applications

### RAG System
- [x] RAG architecture understanding (docs created)
- [x] Document ingestion (Document dataclass, add_documents, add_file)
- [x] Chunking strategies (paragraph, sentence, fixed-size with overlap)
- [x] Retrieval pipeline (Retriever class, similarity search, context assembly)
- [x] Context integration (assemble_context, prompt building)
- [x] Full RAG implementation (RAGPipeline class with query method)

### Chat Interface
- [x] Conversation memory (ChatMemory class, sliding window, token budget)
- [x] Context management (messages=[] structure, memory strategies)
- [x] Streaming responses (OpenAI/Anthropic streaming, StreamPrinter)
- [x] Multi-turn conversations (reference resolution, history management)
- [x] Chat engine implementation (ChatEngine class with full pipeline)

### Function Calling
- [x] Function definitions (Tool, ToolParameter, ToolRegistry classes)
- [x] Tool integration (OpenAI and Anthropic schema formats)
- [x] Error handling (ToolExecutor with validation and safe execution)
- [x] Multi-tool orchestration (parallel tool calls, agent loop pattern)
- [x] Function calling demos (FunctionCallingEngine with live API)

**Phase 3 Completion**: 100% (All 3 modules complete! RAG + Chat + Function Calling) ✅

---

## Phase 4: AI Agents

### Simple Agent
- [x] Agent architecture (schemas: AgentState, AgentAction, AgentConfig)
- [x] Task planning (ReActAgent with thought → action → observation loop)
- [x] Execution loop (run() with max_iterations safety)
- [x] Basic ReAct pattern (THOUGHT + ACTION prompting)
- [x] Agent project (6 demos: simple, multi-step, knowledge+calc, compare, history, timeout)

### Tool Use
- [x] Tool definitions (BaseTool abstract class, ToolDefinition, ToolParameter)
- [x] Web search integration (WebSearchTool with mock + real Tavily)
- [x] File operations (ReadFileTool, WriteFileTool, ListDirectoryTool)
- [x] API calls (HttpGetTool with mock + real httpx)
- [x] Tool orchestration (ToolRegistry service container pattern)

### Multi-Agent
- [x] Agent collaboration (hierarchical orchestration pattern)
- [x] Task delegation (orchestrator → specialist agents)
- [x] Communication patterns (Agent-as-Tool pattern via ToolRegistry)
- [x] Orchestration (MultiAgentOrchestrator with ReAct loop)
- [x] Multi-agent system (Research, Analysis, Writer specialists)

**Phase 4 Completion**: 100% (All 3 modules completed!) ✅

---

## Phase 5: Production

### API Design (LLM-Specific Patterns)
- [x] Semantic caching (embedding-based similarity cache)
- [x] SSE streaming (token-by-token LLM output)
- [x] Cost tracking (per-request token/cost monitoring)
- Note: FastAPI basics already known from ac-agent

### Evaluation (Module 2)
- [x] RAG evaluation metrics (relevance, groundedness, faithfulness, answer relevance)
- [x] LLM-as-judge pattern for quality metrics
- [x] Hallucination detection (groundedness < 0.7)
- [x] LLM tracing with spans (OpenTelemetry-style)
- [x] Trace hierarchy (parent/child spans)

### Optimization (Module 3)
- [x] Prompt compression (LLMLingua-2 deep dive - tensors, PyTorch, CUDA, model weights)
- [ ] Batching strategies (code exists, not learned)
- [ ] Rate limiting (code exists, not learned)
- [ ] Cost budgets (code exists, not learned)

**Phase 5 Completion**: 40% (Module 1 done, Module 2 not learned, Module 3 compression only)

---

## Overall Progress

**Total Completion**: ~75% (Phase 2 + Phase 3 + Phase 4 complete!)

**Milestones Achieved**:
- [x] Understanding Embeddings (Critical conceptual breakthrough!)
- [x] ChromaDB hands-on experience
- [x] Semantic search working demo
- [x] RAG documentation comprehensive
- [x] Phase 2 completed ✅
- [x] RAG System implementation complete ✅
- [x] Chat Interface implementation complete ✅
- [x] Function Calling implementation complete ✅
- [x] Phase 3 completed ✅
- [x] Phase 4 completed ✅
- [ ] Phase 1 completed
- [ ] Phase 5 completed

**Projects Built**:
- embeddings_demo: Memory files → ChromaDB → Semantic search ✅
- rag_system: Full RAG pipeline with modular architecture ✅
- chat_interface: Multi-turn chat with memory and streaming ✅
- function_calling: Tool definitions, executor, agent loop pattern ✅
- simple_agent: ReAct pattern with mock tools ✅
- tool_use: Real tools (file, web, HTTP) with ToolRegistry ✅
- multi_agent: Hierarchical orchestration with specialist agents ✅

**Skills Acquired**:
- Understanding of embeddings and their role in AI
- Mental model of RAG architecture (detailed!)
- Knowledge of vector databases (ChromaDB, Pinecone, etc.)
- Connection between NumPy and AI applications
- Cost models for embeddings (API vs open-source)
- Hands-on ChromaDB experience
- Semantic search implementation
- LLM API integration (OpenAI, Anthropic, Azure, Bedrock patterns)
- messages=[] conversation structure
- Token tracking and cost calculation
- Streaming responses for UX
- RAG pipeline implementation (chunking, retrieval, generation)
- ONNX Runtime for local embeddings (all-MiniLM-L6-v2)
- Chat memory strategies (sliding window, token budget, summarization)
- Multi-turn conversation handling
- Reference resolution through conversation context
- When to use vector DB vs SQL vs live API queries
- Function calling / tool use (LLM decides, you execute)
- Tool schema definitions (OpenAI and Anthropic formats)
- Safe tool execution with validation and error handling
- Agent loop pattern (foundation for AI agents)

---

## Notes

**Strengths**:
- Strong conceptual understanding of embeddings
- Clear mental model of how AI apps work (RAG pattern)
- Good grasp of cost/tool trade-offs
- Hands-on experience with ChromaDB

**Areas for Improvement**:
- Complete Phase 1 NumPy exercises
- Build more hands-on projects
- Practice Pandas for data processing

**Interesting Discoveries**:
- Embeddings are just numbers that encode meaning!
- Can build real AI apps with free tools (sentence-transformers + ChromaDB)
- NumPy is critical because embeddings ARE NumPy arrays
- Vector databases exist specifically for high-dimensional vectors
- 95% of RAG cost is LLM generation, not retrieval!

**Questions to Explore**:
- How exactly are embeddings generated internally?
- What makes one embedding model better than another?
- How to choose chunk size for RAG systems?

---

## Session History

### 2025-01-13: Embeddings Milestone
- ✅ Major breakthrough understanding embeddings
- ✅ Learned about vector databases
- ✅ Understood NumPy's role in AI
- ✅ Clarified cost models (API vs open-source)
- ✅ Got clear mental model of RAG architecture
- Session time: ~90 minutes
- Confidence gain: Significant

### 2025-11-30: RAG Deep Dive & ChromaDB Hands-On
- ✅ Enhanced rag_explained.md with 600+ lines of detailed content
- ✅ Added flow diagrams with timing/cost breakdowns
- ✅ Added stage-by-stage deep dive (Indexing, Retrieval, Augmentation, Generation)
- ✅ Added storage pattern economics (critical cost optimization patterns)
- ✅ Ran ChromaDB demo successfully
- ✅ Stored learning memories in vector database
- ✅ Performed semantic search over memories
- ✅ Started Phase 2: Prompt Engineering module completed
- ✅ Created Pandas basics module (optional for LLM work)
- ✅ Learned all 7 prompt engineering patterns
- Session focus: RAG understanding + hands-on vector DB + Prompt Engineering
- Confidence gain: Strong practical foundation

### 2025-12-05: API Integration Module & Phase 2 Complete
- ✅ Created API Integration module (examples.py + live_examples.py)
- ✅ Learned OpenAI API patterns (client, messages, streaming)
- ✅ Learned Anthropic API patterns (different structure)
- ✅ Understood Azure OpenAI vs OpenAI vs AWS Bedrock differences
- ✅ Deep dive on messages=[] structure (system/user/assistant roles)
- ✅ Token tracking and cost calculation patterns
- ✅ Error handling with exponential backoff
- ✅ Set up .env with python-dotenv for API keys
- ✅ Phase 2 LLM Fundamentals complete!
- Session focus: API integration patterns across providers

### 2025-12-05: Phase 3 RAG System Implementation
- ✅ Built complete RAG pipeline with modular architecture
- ✅ Created chunking.py (3 strategies: paragraph, sentence, fixed-size)
- ✅ Created retrieval.py (Retriever class, context assembly)
- ✅ Created examples.py (comprehensive demos)
- ✅ Refactored rag_pipeline.py as main orchestrator
- ✅ Learned about chunk overlap and why it matters
- ✅ Understood ONNX Runtime for local embeddings
- ✅ Learned ChromaDB uses all-MiniLM-L6-v2 by default (80MB model)
- ✅ Added ASCII diagrams to documentation files
- ✅ Phase 3 Module 1 (RAG System) complete!
- Session focus: RAG implementation with production patterns
- Key insight: ChromaDB downloads MiniLM model for free local embeddings

### 2025-12-05: Phase 3 Chat Interface Implementation
- ✅ Built chat_memory.py (ChatMemory, SummarizingMemory classes)
- ✅ Implemented 3 memory strategies: full, sliding_window, token_budget
- ✅ Built streaming.py (OpenAI/Anthropic streaming, StreamPrinter)
- ✅ Created ChatEngine class combining memory + generation
- ✅ Demonstrated reference resolution ("its", "them", "which one")
- ✅ Phase 3 Module 2 (Chat Interface) complete!
- Key insight: LLM "memory" = re-reading entire conversation each turn
- Key insight: Vector DB not always needed - SQL or live API may be better
- Discussion: Portfolio tracker use case → SQL + live API, not vector DB

### 2025-12-11: Phase 3 Function Calling Implementation
- ✅ Built function_definitions.py (Tool, ToolParameter, ToolRegistry)
- ✅ Implemented OpenAI and Anthropic schema converters
- ✅ Built tool_executor.py (ToolExecutor with validation, error handling)
- ✅ Created FunctionCallingEngine (complete agent loop pattern)
- ✅ Demonstrated multi-tool parallel execution
- ✅ Phase 3 Module 3 (Function Calling) complete!
- ✅ PHASE 3 COMPLETE! 🎉
- Key insight: LLM DECIDES which tool to call, YOUR CODE executes
- Key insight: This is the foundation for AI agents (Phase 4)
- Discussion: Python vs Java for AI (Python wins for ecosystem, Java for enterprise)
- Discussion: 5 phases cover LLM applications, not all AI (ML, CV, etc.)
- Next: Phase 4 - AI Agents

### 2025-12-11: Phase 3 Refactoring + Phase 4 Simple Agent
- ✅ Refactored models/ → schemas/ across all Phase 3 modules (FastAPI/Pydantic convention)
- ✅ Discussed code quality: Phase 3 code is ~80% production-ready as blueprints
- ✅ Built Phase 4 Module 1: Simple Agent with ReAct pattern
- ✅ Created schemas: AgentState enum, AgentAction, AgentResult, AgentConfig
- ✅ Built ReActAgent class with THINK → ACT → OBSERVE loop
- ✅ Created 6 demos: simple query, multi-step, knowledge+calc, compare cities, history, timeout
- Key insight: Agent = loop around function calling + reasoning
- Key insight: LLM job changes from "pick a tool" to "think, then pick tool or finish"
- Key pattern: THOUGHT: + ACTION: prompting format for structured agent responses

### 2025-12-12: Phase 4 Module 2: Tool Use
- ✅ Built real tool system with BaseTool abstract class (like Java interface)
- ✅ Created ToolRegistry (service container / DI pattern)
- ✅ Implemented file tools: ReadFileTool, WriteFileTool, ListDirectoryTool
- ✅ Implemented WebSearchTool with mock + real Tavily support
- ✅ Implemented HttpGetTool with mock + real httpx support
- ✅ Added security features: allowed_directories, allowed_domains
- ✅ Created ToolResult with factory methods: ok(), fail()
- ✅ Built tool_agent.py integrating tools with ReActAgent
- Key insight: Tools = interface + definition + execute (3 parts)
- Key insight: Mock mode allows testing without API keys
- Key pattern: Service Registry pattern for tool management
- Java equivalents: Interface Tool, ToolRegistry @Component, Result<T,E>

### 2025-12-13: Agent Concepts Deep Dive + Module 2 Cleanup
- ✅ Made Module 2 fully self-contained (duplicated agent code, no imports from Module 1)
- ✅ Removed Java equivalent comments from code (kept in README for reference)
- ✅ Cleaned up sys.path hacks - now using pathlib for cleaner imports
- ✅ Created comprehensive CONCEPTS.md with core agent concepts
- ✅ Created phase4 README.md linking all modules
- ✅ Updated module READMEs with links to CONCEPTS.md
- Key insight: Agent = LLM + Tools + Decision Loop
- Key insight: Tool = Function + Description (metadata for LLM understanding)
- Key insight: LLM binds arguments semantically (by meaning) not syntactically (by position)
- Key insight: Static code = compile-time decisions; Agent = runtime semantic decisions
- Discussion: ReAct pattern (Think → Act → Observe → Repeat)
- Discussion: Alternative patterns (Plan-and-Execute, Reflexion, Tree of Thoughts)
- Reference: ReAct paper (Yao et al., 2022) - arxiv.org/abs/2210.03629

### 2025-12-14: Phase 5 Module 1 - LLM-Specific API Patterns
- ✅ Created Phase 5 README with production architecture overview
- ✅ Created AI Development Taxonomy document (4-level hierarchy)
- ✅ Refocused Module 1 on LLM-specific patterns (not generic FastAPI)
- ✅ Built semantic_cache.py (embedding-based similarity caching)
- ✅ Built llm_streaming.py (SSE for token-by-token output)
- ✅ Built cost_tracker.py (per-request token/cost tracking)
- Key insight: Semantic cache uses embedding similarity, not exact match
- Key insight: SSE streaming improves UX for slow LLM responses
- Key insight: Cost tracking essential for production budget control
- Key pattern: datetime.now(timezone.utc) not datetime.utcnow() (Python 3.12+)
- Note: Skipped generic FastAPI patterns - already known from ac-agent

### 2025-12-14: Phase 5 Module 2 - Evaluation & Observability
- ✅ Built rag_metrics.py (Ragas-style evaluation with 3 backends)
- ✅ Implemented 4 metrics: context relevance, groundedness, faithfulness, answer relevance
- ✅ Built 3 evaluation backends: EmbeddingScorer, LLMJudgeScorer, CrossEncoderScorer
- ✅ Built llm_tracing.py (OpenTelemetry-style spans)
- ✅ Implemented Span/Trace hierarchy with timing and attributes
- ✅ Built rag_eval_production.py (real API examples with all 3 options)
- Key insight: Different metrics need different backends!
  - Relevance → Embedding/CrossEncoder (semantic similarity, no reasoning)
  - Groundedness/Faithfulness → LLM-as-judge (needs reasoning about claims)
- Key insight: EmbeddingScorer = fast, cheap; LLMJudge = accurate, expensive
- Key insight: CrossEncoder = best accuracy for relevance, FREE (local model)
- Key insight: Groundedness < 0.7 indicates likely hallucination
- Cost comparison: Embedding ~$0.02/1K, LLM ~$0.15-3/1K, CrossEncoder FREE
- Tools: Ragas, TruLens, Phoenix for eval; LangSmith, W&B for tracing

### 2025-12-20: LLMLingua-2 Deep Dive + PyTorch/CUDA Fundamentals
- ✅ Replaced naive regex compression with production LLMLingua-2
- ✅ Split Module 3 into modular subdirectories (01_compression, 02_throughput, 03_cost_control)
- ✅ Deep dive: What's in a 677MB model file (177M floating point weights)
- ✅ Explored HuggingFace cache structure (~/.cache/huggingface/hub/)
- ✅ Learned: Tensor = multi-dimensional array of numbers
- ✅ Learned: PyTorch = NumPy + GPU support + auto-gradients
- ✅ Learned: CUDA = NVIDIA's parallel computing platform for GPUs
- ✅ Benchmarked GPU vs CPU (RTX 5070 Ti: 2.4x faster inference)
- ✅ Learned lazy import pattern (heavy deps inside __init__, not top-level)
- ✅ Researched current state of prompt compression (LLMLingua-2 still best OSS)
- Key insight: Model loading slow because 677MB deserialized every instantiation
- Key insight: BERT model = embeddings (92M) + 12 transformer layers (85M) + classifier
- Key insight: zip() stops at shortest list (use strict=True for safety)
- Next: Module 2 (evaluation) and rest of Module 3 (batching, rate limiting, cost)
