# Learning Progress Tracker

Track your journey through AI development. Update after each session.

## Phase 1: Foundations ⏸️ OPTIONAL

> **Decision (2025-12-30)**: Skipped to focus on practical AI app development.
> Exercises created but not completed. Can return if needed for debugging/optimization.

### NumPy Basics
- [x] Array creation and manipulation
- [x] Vectorized operations
- [x] Broadcasting
- [x] Indexing and slicing
- [ ] Reshaping and stacking
- [x] Performance optimization
- [ ] Exercises completed (7 exercises ready in solutions/exercise_solutions.py)
- [ ] Project: Similarity search

### Pandas Data Manipulation
- [ ] DataFrames and Series (examples.py exists)
- [ ] Exercises completed (8 exercises ready in solutions/exercise_solutions.py)
- [ ] Project: Data analysis report

### ML Concepts
- [ ] Not started (module not created)

**Phase 1 Status**: OPTIONAL - exercises scaffolded, skip to Phase 2

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
- [x] Batching strategies (EmbeddingBatcher, ParallelExecutor, BatchScheduler - 10x+ speedup)
- [x] Rate limiting (TokenBucket, SlidingWindow, AdaptiveRateLimiter - prevent 429 errors)
- [x] Cost budgets (RequestBudget, UserBudget, CostGuard - pre-flight budget checks)

**Phase 5 Completion**: 100% (All 3 modules complete! API Design, Evaluation, Optimization) ✅

---

## Phase 6: Model Customization (Exploratory)

### Training Fundamentals (Module 1)
- [ ] Training loop (forward → loss → backward → update)
- [ ] Gradients and backpropagation
- [ ] GPU vs CPU training
- [ ] Learning rate, batch size, epochs
- [ ] MNIST from scratch experiment

### Fine-Tuning Basics (Module 2)
- [ ] Transfer learning concept
- [ ] BERT fine-tuning
- [ ] Freezing layers
- [ ] Overfitting prevention

### LoRA/QLoRA (Module 3)
- [ ] LoRA concept (low-rank adaptation)
- [ ] QLoRA (quantized LoRA)
- [ ] PEFT library
- [ ] Small LLM fine-tuning

### Dataset Preparation (Module 4)
- [ ] Data formats (instruction, chat, completion)
- [ ] Quality vs quantity
- [ ] Synthetic data generation

### Evaluation & Selection (Module 5)
- [ ] When to fine-tune vs prompt engineer vs RAG
- [ ] Evaluation metrics
- [ ] Cost analysis

**Phase 6 Completion**: 0% (Structure scaffolded, ready to explore)

---

## Phase 7: Production Frameworks

### LangChain Basics (Module 1) - 🔄 In Progress (75% Complete)
- [x] Migration examples (side-by-side comparisons of Phase 3/4 vs LangChain)
- [x] Conceptual demos (patterns, RAG walkthrough without dependencies)
- [x] LangChain installation (langchain==1.0.5 + integrations)
- [x] Prompts & templates (01_prompts/ subfolder: concepts.py + practical.py)
- [x] LLM integration (02_llm_integration/ subfolder: concepts.py + practical.py)
- [x] Chains (03_chains/ subfolder: concepts.py + practical.py) ✅ **COMPLETE**
- [x] Shared utilities refactoring (phase7_frameworks/utils.py for all modules)
- [ ] Memory (04_memory/ subfolder: modern RunnableWithMessageHistory API)
- [ ] RAG (05_rag/ subfolder: modern LCEL patterns)
- [x] Agents & tools (06_agents_tools/ subfolder: **Demo 1-5 complete**, Demo 6-7 pending)
  - [x] Demo 1: Basic Tool Creation (@tool decorator)
  - [x] Demo 2: Simple Agent with LLM (time tool + agent)
  - [x] Demo 3: Multi-Tool Agent (calculator, string_reverse, string_upper)
  - [x] Demo 4: Custom Tool Class (WeatherTool with BaseTool)
  - [x] Demo 5: Async and Streaming (TODO: verify completion)
  - [ ] Demo 6: Error Handling Strategies
  - [ ] Demo 7: Agent Decision Tree / Schema Inspection
- [x] **Documentation Cleanup (2026-02-06)**: Removed all deprecated/legacy patterns
  - Cleaned 4 READMEs: chains, memory, RAG, agents (~200 lines removed)
  - Pure LangChain 1.0+ patterns only (no ⚠️ DEPRECATED sections)
  - Focus on LCEL, RunnableWithMessageHistory, create_agent
  - Cleaner learning experience without legacy cruft
- [x] **Code Quality Improvement (2026-02-07)**: Menu migration
  - Eliminated duplicate print_section functions (34 files)
  - Centralized in common/util/utils.py
- [x] **Code Quality Improvement (2026-02-08)**: Tool usage visibility
  - Added print statements to all tools for debugging/learning
  - Pattern: `🔧 Tool used: <name>(<params>)` in 6 tools

### LangGraph (Module 2) - 🔄 In Progress
- [x] State basics (01_state_basics.py) - 🔄 **In Progress: Demo 4/6**
  - [x] Demo 1: Simple State and Graph (StateGraph, nodes, edges)
  - [x] Demo 2: State Updates and Immutability (partial updates, shallow merge)
  - [x] Demo 3: Message State Pattern (add_messages, accumulation)
  - [ ] Demo 4: Sequential Multi-Node Graph (linear processing) **← CURRENT**
  - [ ] Demo 5: Stateful Counter (multi-invocation state persistence)
  - [ ] Demo 6: Simple Agent Loop (foundation for ReAct)
- [ ] Conditional routing (02_conditional_routing.py: conditional edges, loops, termination)
- [ ] Multi-agent collaboration (03_multi_agent.py: router pattern, supervisor pattern, sequential handoff)
- [ ] Human-in-the-loop (04_human_in_loop.py: checkpoints, interrupts, resume)
- [ ] Graph visualization (05_graph_visualization.py: mermaid rendering)
- [ ] Migration from Phase 4 (06_migration_from_phase4.py: while loop → graph comparison)

### LlamaIndex (Module 3)
- [ ] Data loading (SimpleDirectoryReader, various loaders)
- [ ] Indexing (VectorStoreIndex, TreeIndex, SummaryIndex)
- [ ] Query engines (basic, filtered, streaming queries)
- [ ] Chat engines (conversational RAG with memory)
- [ ] Advanced retrieval (hybrid search, auto-merging)
- [ ] Metadata filtering (date, category filters)

### Framework Comparison (Module 4)
- [ ] Decision framework (when to use which framework)
- [ ] Performance comparison (framework vs raw API benchmarks)
- [ ] Migration strategies (raw → framework, framework → raw, hybrid)
- [ ] Real-world scenarios (10+ use cases analyzed)

**Phase 7 Completion**: 50% (Module 1: prompts + LLM integration + chains + 5/7 agents demos complete)

**Note**: Phase 7 created 2026-01-09. Philosophy: Learn frameworks AFTER building from scratch (Phases 2-6) to understand what they do under the hood, when to use them, and when to skip them entirely.

---

## Overall Progress

**Total Completion**: ~87% (Phase 2 + Phase 3 + Phase 4 + Phase 5 complete!)

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
- [x] Phase 5 completed ✅
- [x] Phase 7 scaffolded (framework learning path ready) ⬜
- [ ] Phase 6 scaffolded ⬜ (exploratory)

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
- Production optimization patterns (batching, rate limiting, cost budgets)
- Batching strategies for reducing API latency (10x+ speedup)
- Rate limiting algorithms (TokenBucket, SlidingWindow, Adaptive)
- Pre-flight budget checks to prevent cost overruns
- Framework landscape (LangChain, LangGraph, LlamaIndex decision framework)
- **NEW (2026-02-08)**: LLM tool selection cost trade-offs (when worth it, when wasteful, hybrid approach)

---

## Notes

**Strengths**:
- Strong conceptual understanding of embeddings
- Clear mental model of how AI apps work (RAG pattern)
- Good grasp of cost/tool trade-offs
- Hands-on experience with ChromaDB

**Areas for Improvement**:
- Build more hands-on projects
- Return to Phase 1 if debugging embeddings/performance issues

**Interesting Discoveries**:
- Embeddings are just numbers that encode meaning!
- Can build real AI apps with free tools (sentence-transformers + ChromaDB)
- NumPy is critical because embeddings ARE NumPy arrays
- Vector databases exist specifically for high-dimensional vectors
- 95% of RAG cost is LLM generation, not retrieval!
- **NEW**: LLM tool selection costs ~$0.000065 but saves weeks of routing logic development

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

### 2025-12-20: Phase 6 Scaffolded + Career Discussion
- ✅ Created Phase 6: Model Customization (exploratory phase)
- ✅ Scaffolded 5 modules: training fundamentals, fine-tuning, LoRA/QLoRA, dataset prep, evaluation
- ✅ Created runnable MNIST experiment for hands-on training loop learning
- ✅ Updated project README and learning progress with Phase 6
- ✅ Career discussion: AI Engineer / AI Solutions Architect roles
- ✅ Researched salaries: US + Israel markets for AI engineer roles
- Key insight: Fine-tuning is rare skill that differentiates from API-only engineers
- Key insight: Phase 6 is exploratory - jump based on interest, not linear
- Discussion: User's skill combination (Java+AWS+Salesforce+Python+AI) = AI Solutions Architect level
- Hardware: RTX 5070 Ti (16GB) capable of LoRA on 7B models

### 2025-12-30: Phase 1 Made Optional + Learning Path Adjustment
- ✅ Created NumPy exercises (7 exercises with skeleton solutions)
- ✅ Created Pandas exercises (8 exercises with skeleton solutions)
- ✅ Fixed bug in examples.py (jagged array)
- ✅ Decided to skip Phase 1 math foundations - not needed for AI app development
- ✅ Phase 1 marked as OPTIONAL (can return if needed for debugging/optimization)
- Key insight: NumPy/Pandas not needed for LangChain/LangGraph agent development
- Key insight: Libraries abstract away the math - focus on building apps
- Decision: Skip to Phase 2 (LLM APIs) → practical AI development path
- Exercises preserved: Can return to Phase 1 if hit issues needing low-level understanding

### 2026-01-09: Phase 5 Complete + Phase 7 Scaffolding
- ✅ Completed Phase 5 Module 3: Optimization (batching, rate limiting, cost budgets)
- ✅ Learned batching strategies: EmbeddingBatcher (10x speedup), ParallelExecutor, BatchScheduler
- ✅ Learned rate limiting: TokenBucket (burstable), SlidingWindow (smooth), AdaptiveRateLimiter (smart)
- ✅ Learned cost budgets: RequestBudget, UserBudget, CostGuard (pre-flight checks)
- ✅ Created Phase 7: Production Frameworks (LangChain, LangGraph, LlamaIndex)
- ✅ Scaffolded 4 modules with comprehensive READMEs
- Key insight: Batching reduces latency (fewer round-trips), not cost (same tokens)
- Key insight: Rate limiting prevents 429 errors, makes app a good API citizen
- Key insight: Budget checks BEFORE API call prevent surprises on bill
- Key insight: Frameworks learned AFTER fundamentals = critical understanding
- Discussion: LiteLLM covers cost tracking + budgets in production
- Discussion: LangChain vs LlamaIndex vs LangGraph decision framework
- Phase 5 complete! Phase 7 scaffolded and ready to learn.

### 2026-01-13: Phase 7 Module 1 - LangChain Migration Examples
- ✅ Created migration_examples.py with 6 side-by-side comparisons
- ✅ Learned LangChain abstractions (prompts, LLM, chains, memory, RAG, agents)
- ✅ Understood when to use framework vs raw API (complexity threshold: 3-4 components)
- ✅ Learned hybrid approach (80% framework, 20% custom logic)
- ✅ Compared Phase 3/4 implementations with LangChain equivalents
- Key insight: Frameworks abstract standard patterns but add overhead
- Key insight: Your fundamentals mean you can choose when to use framework
- Key insight: Production commonly uses hybrid approach (framework + raw API)
- Key insight: Never locked into framework - can drop to raw API anytime
- Session focus: Understanding framework trade-offs and decision criteria

### 2026-01-14: Phase 7 Module 1 - Conceptual Demos + LangChain Installation
- ✅ Created langchain_concepts_demo.py (7 conceptual sections)
- ✅ Created langchain_rag_chatbot.py (RAG walkthrough)
- ✅ Installed full LangChain packages (langchain==1.0.5 + integrations)
- ✅ Updated all documentation to reflect current status
- ✅ Committed Phase 7 Module 1 foundation work
- Key insight: Conceptual demos (no API) teach patterns before hands-on
- Key insight: LangChain RAG setup: 20 lines vs Phase 3: 300 lines
- Key insight: Trade-off: Less code but less control over each step
- Key insight: Module 1 foundation complete - ready for hands-on examples
- Session focus: Conceptual understanding + environment setup
- Next: Build hands-on examples (prompts, chains, memory, RAG, agents)

### 2026-01-14: Phase 7 - Prompts Module Complete
- ✅ Created 01_prompts/ subfolder with concepts/practical split
- ✅ Built concepts.py (8 demos, no API key): templates, partials, chat, few-shot, composition, decision framework
- ✅ Built practical.py (7 demos with LLM): LCEL integration, MessagesPlaceholder, output parsers, runtime context
- ✅ Fixed LangChain 1.x imports (langchain_core instead of langchain)
- ✅ Tested both files (briefly reviewed and validated)
- ✅ Initialized MCP memory graph (developer profile, project, learning preferences)
- Key insight: Subfolder organization (concepts + practical) scales well for remaining modules
- Key insight: PromptTemplate validation prevents missing required variables (vs f-strings)
- Key insight: LCEL pipe operator (|) chains components elegantly
- Key insight: Output parsers (String, List, JSON) convert LLM text to structured data
- Session focus: Hands-on LangChain prompt templates with real API integration
- Next: 02_llm_integration/ (ChatOpenAI, ChatAnthropic unified interface)

### 2026-01-21: Phase 7 - LangChain API Modernization
- ✅ Updated all 6 module READMEs for LangChain 1.0+ API consistency
- ✅ Added comprehensive "Legacy Patterns ⚠️ DEPRECATED" sections
- ✅ Added migration guides (agents_tools, rag, memory, chains)
- ✅ Standardized "Run Examples" sections across all modules
- ✅ Added visual learning notes (ASCII diagrams) to all modules
- ✅ Deprecated APIs documented: create_react_agent, AgentExecutor, RetrievalQA, ConversationalRetrievalChain, ConversationChain, LLMChain, SequentialChain, .run()
- ✅ Modern APIs emphasized: create_agent (LangGraph), LCEL patterns, RunnableWithMessageHistory, .invoke()
- Key insight: RAG module most comprehensive (2 legacy patterns, advanced LCEL examples)
- Key insight: Message-based APIs are LangChain 1.0+ standard across all modules
- Key insight: LCEL pipe operators are the foundation (chains, RAG, memory, agents)
- Key pattern: "Modern pattern ✅" first, "Legacy pattern ⚠️" second with "Why deprecated?"
- Documentation quality: All modules now have consistent structure, clear deprecation warnings, and migration paths
- Session focus: Documentation modernization and consistency across all LangChain basics modules
- Next: Continue with LangGraph module (state graphs, multi-agent collaboration)

### 2026-01-14: Phase 7 - LLM Integration Module Complete
- ✅ Created 02_llm_integration/ subfolder with concepts/practical split
- ✅ Built concepts.py (8 demos, no API): unified interface, message types, configuration, streaming patterns, provider comparison, error handling, retry/fallback, token/cost estimation
- ✅ Built practical.py (8 demos with API): ChatOpenAI, ChatAnthropic, temperature control, streaming, provider switching, fallback chains, token tracking, LCEL integration
- ✅ Tested concepts.py successfully (all demos run without API keys)
- ✅ Lightly reviewed and committed (can revisit as needed)
- Key insight: Unified chat interface (invoke/stream/batch) works across all providers
- Key insight: Provider fallback pattern: primary.with_fallbacks([fallback1, fallback2])
- Key insight: Temperature controls creativity (0.0=deterministic, 1.0=creative)
- Key insight: Streaming improves UX with progressive token delivery
- Key insight: Token tracking with callbacks enables cost monitoring
- Session focus: Understanding LangChain's provider abstraction and configuration patterns
- Next: 03_chains/ (LLMChain, SequentialChain, LCEL composition)

### 2026-02-07: Demo Menu Migration - Code Quality & DRY Improvement
- ✅ Created unified menu system: `common/util/demo_menu.py` (Demo dataclass + MenuRunner class)
- ✅ Eliminated duplication: Demo metadata defined once (was 2 places: display + execution)
- ✅ Built comprehensive test suite: `tests/test_demo_menu.py` (8 tests, 98% coverage)
- ✅ Migrated 25 of 36 files (69% complete): Phases 1-2-3-5-7
- ✅ Code reduction: ~2,000-3,300 lines eliminated (85-90% per file)
- ✅ Pattern: `DEMOS = [Demo(...)]` + `MenuRunner(DEMOS).run()` (~10 lines vs ~90-155)
- ✅ Created migration documentation: `MIGRATION_SUMMARY.md`
- Key insight: Single source of truth prevents maintenance burden
- Key insight: Type-safe dataclass validation prevents errors
- Key insight: Centralized menu logic enables easy feature additions
- Key insight: Bulk migration script saved significant time (22 files automated)
- Remaining: 11 files with special patterns (conditional demo_map, 3-tuple with API provider)
- Session focus: Eliminating systematic code duplication across all interactive demos
- Impact: Consistent UX, easier maintenance, DRY principle applied

### 2026-02-08: Agents & Tools Progress + Tool Visibility
- ✅ Added tool usage print statements (6 tools across practical.py)
- ✅ Standardized format: `🔧 Tool used: <name>(<params>)`
- ✅ Explored LLM tool selection cost trade-offs
- ✅ Learned when LLM selection worth it vs wasteful
- ✅ Understood hybrid approach (pattern matching + LLM fallback)
- ✅ Completed Demos 1-5 in 06_agents_tools module
- Key insight: LLM tool selection ~$0.000065 but saves weeks of routing logic
- Key insight: Production uses hybrid: cheap pattern matching → LLM only if ambiguous
- Key insight: User-facing: LLM selection ✅; Internal APIs: direct calls ✅
- Session focus: Tool execution visibility + cost/value analysis
- Remaining: Demo 6 (error handling), Demo 7 (schema inspection)
- Next: Complete 06_agents_tools, then 04_memory, then 05_rag

### 2026-02-13: LangGraph State Basics + Code Style Improvements
- ✅ Created CLAUDE.md rule: Multi-line prints → `print("\n" + cleandoc(...))`
- ✅ Applied rule to all 6 LangGraph files (65 patterns consolidated)
- ✅ Started LangGraph Module 2: State basics (01_state_basics.py)
- ✅ Completed Demo 1: Simple State and Graph (StateGraph fundamentals)
- ✅ Completed Demo 2: State Updates (immutability, partial updates, shallow merge)
- ✅ Completed Demo 3: Message State Pattern (add_messages for accumulation)
- ✅ Working on Demo 4: Sequential Multi-Node Graph **← CURRENT**
- Key insight: StateGraph nodes return updates (dicts), graph merges automatically
- Key insight: `Annotated[list, add_messages]` accumulates instead of replacing
- Key insight: State flows through graph, updated at each node immutably
- Key pattern: `print("\n" + cleandoc("""...`))` preserves newlines, improves readability
- Session focus: LangGraph state management patterns and code quality
- Next: Complete remaining demos (5-6), then conditional routing
