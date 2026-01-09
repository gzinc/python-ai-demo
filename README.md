# AI Development Roadmap - From Basics to LLM Applications

A comprehensive, hands-on learning path for AI development focused on **LLM applications** and **AI agents**. Designed for experienced developers transitioning to AI/ML, with a focus on practical applications and modern tools.

## 🎯 Learning Path Overview

This roadmap follows a progressive structure:

**Phase 1: Foundations** (2-3 weeks)
→ Python for AI, NumPy/Pandas, data manipulation, basic ML concepts

**Phase 2: LLM Fundamentals** (2-3 weeks)
→ Prompt engineering, LLM APIs, embeddings, vector databases

**Phase 3: LLM Applications** (3-4 weeks)
→ RAG systems, chat interfaces, function calling, agents

**Phase 4: Advanced AI Agents** (4-5 weeks)
→ Multi-agent systems, tool use, memory, orchestration

**Phase 5: Production Systems** (3-4 weeks)
→ Deployment, monitoring, optimization, scaling

**Phase 6: Model Customization** (exploratory)
→ Training fundamentals, fine-tuning, LoRA/QLoRA, dataset prep

**Phase 7: Production Frameworks** (2-3 weeks)
→ LangChain, LangGraph, LlamaIndex, framework comparison

---

## 📁 Project Structure

| Directory | Purpose |
|-----------|---------|
| [phase1_foundations/](phase1_foundations/) | Python for AI basics |
| ↳ [01_numpy_basics/](phase1_foundations/01_numpy_basics/) | NumPy arrays and operations |
| ↳ [02_pandas_basics/](phase1_foundations/02_pandas_basics/) | Pandas DataFrames |
| [phase2_llm_fundamentals/](phase2_llm_fundamentals/) | LLM basics |
| ↳ [01_prompt_engineering/](phase2_llm_fundamentals/01_prompt_engineering/) | Prompt techniques |
| ↳ [02_api_integration/](phase2_llm_fundamentals/02_api_integration/) | LLM API usage |
| [phase3_llm_applications/](phase3_llm_applications/) | Building with LLMs |
| ↳ [01_rag_system/](phase3_llm_applications/01_rag_system/) | RAG pipeline |
| ↳ [02_chat_interface/](phase3_llm_applications/02_chat_interface/) | Chat with memory |
| [phase4_ai_agents/](phase4_ai_agents/) | Agent systems |
| [phase5_production/](phase5_production/) | Production deployment |
| [phase6_model_customization/](phase6_model_customization/) | Training & fine-tuning |
| [phase7_frameworks/](phase7_frameworks/) | LangChain, LangGraph, LlamaIndex |
| [docs/](docs/) | Learning resources |
| ↳ [docs/concepts/](docs/concepts/) | Theoretical background |
| ↳ [docs/guides/](docs/guides/) | Step-by-step tutorials |
| ↳ [docs/references/](docs/references/) | Quick references |
| [.serena/memories/](.serena/memories/) | Your learning progress |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (3.13 recommended)
- **uv** package manager (fast, modern Python tooling)
- OpenAI API key or Anthropic API key

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```

3. **Set up API keys**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Verify installation**:
   ```bash
   uv run python -c "import numpy, pandas; print('Ready!')"
   ```

---

## 📚 Phase 1: Foundations (2-3 weeks)

**Goal**: Master Python libraries essential for AI development

### Topics Covered
- ✅ NumPy: Arrays, operations, broadcasting
- ✅ Pandas: DataFrames, data manipulation, analysis
- ✅ Data preprocessing and cleaning
- ✅ Basic ML concepts: features, labels, training/test splits
- ✅ Visualization with matplotlib/seaborn

### Projects
1. **NumPy Basics**: Array operations, matrix math
2. **Data Analysis**: Real-world dataset exploration with Pandas
3. **ML Prep**: Feature engineering and data preprocessing

### Learning Approach
- **Theory**: Read [phase1_foundations/README.md](phase1_foundations/README.md)
- **Practice**: Complete exercises in each subfolder
- **Build**: Create your own data analysis project
- **Review**: Check [docs/concepts/](docs/concepts/) for detailed explanations

---

## 📚 Phase 2: LLM Fundamentals (2-3 weeks)

**Goal**: Understand LLMs and how to work with them

### Topics Covered
- ✅ Prompt engineering principles
- ✅ LLM API integration (OpenAI, Anthropic, local models)
- ✅ Embeddings and semantic search
- ✅ Vector databases (ChromaDB, Pinecone)
- ✅ Token management and costs

### Projects
1. **Prompt Lab**: Experiment with different prompting techniques
2. **API Integration**: Build a simple LLM-powered CLI
3. **Semantic Search**: Create a document search system with embeddings

### Key Concepts
- **Temperature**: Controls randomness (0 = deterministic, 1 = creative)
- **Tokens**: ~4 chars per token, models have context limits
- **Embeddings**: Vector representations of text for similarity
- **System prompts**: Set behavior and context for the LLM

---

## 📚 Phase 3: LLM Applications (3-4 weeks)

**Goal**: Build production-ready LLM applications

### Topics Covered
- ✅ RAG (Retrieval Augmented Generation) architecture
- ✅ Chat interfaces with memory/context
- ✅ Function calling and tool use
- ✅ Streaming responses
- ✅ Error handling and retry logic

### Projects
1. **RAG System**: Q&A over your documents
2. **Chat Interface**: Conversational AI with memory
3. **Function Calling**: LLM that uses tools/APIs
4. **Document Assistant**: Upload docs, ask questions

### Architecture Patterns
```
User Query
    ↓
Retrieve relevant context (vector DB)
    ↓
Build prompt (query + context + system)
    ↓
LLM generates response
    ↓
Return to user
```

---

## 📚 Phase 4: Advanced AI Agents (4-5 weeks)

**Goal**: Build autonomous AI agents with tool use

### Topics Covered
- ✅ Agent architecture and reasoning loops
- ✅ Tool use and function calling
- ✅ Memory systems (short-term, long-term)
- ✅ Multi-agent collaboration
- ✅ Agent orchestration patterns

### Projects
1. **Simple Agent**: Task planning and execution
2. **Tool-Using Agent**: Web search, calculations, file operations
3. **Research Agent**: Multi-step information gathering
4. **Multi-Agent System**: Specialized agents collaborating

### Agent Pattern
```
1. Receive task
2. Plan approach (ReAct/Chain-of-thought)
3. Execute actions (use tools)
4. Observe results
5. Decide next step
6. Repeat until complete
```

---

## 📚 Phase 5: Production Systems (3-4 weeks)

**Goal**: Deploy and maintain AI applications at scale

### Topics Covered
- ✅ API design for LLM applications
- ✅ Monitoring and observability
- ✅ Cost optimization
- ✅ Rate limiting and caching
- ✅ Security and safety
- ✅ Evaluation and testing

### Projects
1. **Production API**: FastAPI + LLM backend
2. **Monitoring Dashboard**: Track usage, costs, performance
3. **Optimization**: Caching, batching, prompt optimization
4. **Full Stack App**: End-to-end LLM application

### Best Practices
- **Streaming**: Better UX for long responses
- **Caching**: Save costs on repeated queries
- **Fallbacks**: Handle API failures gracefully
- **Evaluation**: Test LLM outputs systematically
- **Guardrails**: Input/output validation and safety

---

## 📚 Phase 6: Model Customization (Exploratory)

**Goal**: Understand and apply model training and fine-tuning

### Topics Covered
- ⬜ Training loop fundamentals (forward → loss → backward → update)
- ⬜ Fine-tuning pre-trained models (BERT, small LLMs)
- ⬜ LoRA/QLoRA for efficient fine-tuning
- ⬜ Dataset preparation and quality
- ⬜ When to fine-tune vs prompt engineer vs RAG

### Projects
1. **MNIST From Scratch**: Train a simple neural network, see the training loop
2. **BERT Classification**: Fine-tune BERT for sentiment analysis
3. **LoRA Experiment**: Fine-tune a small LLM on your GPU
4. **Custom Dataset**: Prepare data for instruction tuning

### Hardware
- Requires GPU for meaningful experimentation
- RTX 5070 Ti (16GB) can handle LoRA on 7B models

### Note
This phase is **exploratory** - jump between modules based on interest.
Not required for production AI engineering, but deepens understanding.

---

## 📚 Phase 7: Production Frameworks (2-3 weeks)

**Goal**: Learn industry-standard frameworks after understanding fundamentals

### Topics Covered
- ⬜ LangChain basics (chains, prompts, memory, agents)
- ⬜ LangGraph (state machines, multi-agent workflows)
- ⬜ LlamaIndex (RAG-focused framework)
- ⬜ Framework comparison and decision making

### Projects
1. **LangChain Migration**: Convert Phase 3/4 code to LangChain
2. **LangGraph Multi-Agent**: Rebuild multi-agent system with graphs
3. **LlamaIndex RAG**: Compare RAG approaches (yours vs LlamaIndex)
4. **Hybrid App**: Mix frameworks + raw API for optimal solution

### Why Phase 7 Comes Last
You've built everything from scratch (Phases 2-6), so you now understand:
- What these frameworks do under the hood
- When to customize vs use defaults
- How to debug when things break
- Whether you actually need a framework

### Decision Framework
- **Pure RAG app** → LlamaIndex
- **Agents + tools** → LangChain
- **Multi-agent workflows** → LangGraph
- **Simple tasks** → Skip frameworks, use raw API
- **Complex hybrid** → Mix frameworks + custom code

---

## 🎓 Learning Resources

### Official Documentation
- **OpenAI**: https://platform.openai.com/docs
- **Anthropic**: https://docs.anthropic.com
- **LangChain**: https://python.langchain.com
- **LlamaIndex**: https://docs.llamaindex.ai

### Key Concepts Reference
Located in [docs/concepts/](docs/concepts/):
- [llm_fundamentals.md](docs/concepts/llm_fundamentals.md): How LLMs work
- [prompting_guide.md](docs/concepts/prompting_guide.md): Effective prompt engineering
- [rag_architecture.md](docs/concepts/rag_architecture.md): Building RAG systems
- [agent_patterns.md](docs/concepts/agent_patterns.md): Agent design patterns
- [embeddings_guide.md](docs/concepts/embeddings_guide.md): Semantic search with vectors

### Guides
Located in [docs/guides/](docs/guides/):
- Step-by-step tutorials for each project
- Troubleshooting common issues
- Integration patterns
- Best practices

---

## 🔧 Development Workflow

### Running Examples
```bash
# Run a specific phase example
uv run python phase2_llm_fundamentals/01_prompt_engineering/basic_prompts.py

# Run tests for a phase
uv run pytest phase3_llm_applications/01_rag_system/

# Start a project server (if applicable)
uv run python phase3_llm_applications/02_chat_interface/app.py
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy .
```

---

## 📊 Progress Tracking

Your learning progress is tracked in [.serena/memories/](.serena/memories/):

- [project_overview.md](.serena/memories/project_overview.md): High-level goals and status
- `session_YYYY_MM_DD_*.md`: Daily learning sessions
- [concepts_learned.md](.serena/memories/concepts_learned.md): Key concepts you've mastered
- [projects_completed.md](.serena/memories/projects_completed.md): Projects you've built
- [questions_and_notes.md](.serena/memories/questions_and_notes.md): Your questions and insights

### Recommended Session Pattern
1. Start: Review previous session memory
2. Learn: Work through new content
3. Practice: Complete exercises
4. Build: Work on project
5. Document: Update session memory with learnings

---

## 🎯 Suggested Learning Path

### Week 1-2: Foundations
- NumPy arrays and operations
- Pandas DataFrames and data manipulation
- Basic ML concepts

### Week 3-4: LLM Basics
- Prompt engineering experimentation
- OpenAI/Anthropic API integration
- Embeddings and vector search

### Week 5-7: Applications
- Build RAG system
- Create chat interface
- Implement function calling

### Week 8-11: Agents
- Simple task agent
- Tool-using agent
- Multi-agent collaboration

### Week 12-15: Production
- FastAPI + LLM backend
- Monitoring and optimization
- Full-stack deployment

---

## 🔑 Key Dependencies

Installed via `uv sync`:

**Data & ML**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities

**LLM & AI**
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `langchain` - LLM application framework
- `llama-index` - RAG and data framework

**Vector & Search**
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings

**Web & API**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `httpx` - HTTP client

**Dev Tools**
- `pytest` - Testing
- `ruff` - Linting/formatting
- `mypy` - Type checking

---

## 💡 Tips for Success

1. **Build Projects**: Don't just read, build real applications
2. **Experiment**: Try different approaches and parameters
3. **Document**: Keep notes in Serena memories
4. **Iterate**: Start simple, add complexity gradually
5. **Ask Questions**: Use AI tools to clarify concepts
6. **Review Code**: Read examples thoroughly
7. **Test Everything**: Validate your understanding

---

## 🤝 Java Developer Notes

### Key Differences
| Java/Spring | Python/AI | Notes |
|-------------|-----------|-------|
| Strong typing | Dynamic + hints | Use type hints for clarity |
| Verbose | Concise | Learn Pythonic patterns |
| JVM ecosystem | pip/uv packages | Package management differs |
| Spring DI | Function injection | Different DI approach |
| Maven/Gradle | pyproject.toml | Dependency management |

### Python Patterns to Learn
- List comprehensions: `[x*2 for x in items if x > 0]`
- Context managers: `with open('file') as f:`
- Decorators: `@property`, `@lru_cache`
- Async/await: `async def func():` and `await call()`
- Type hints: `def func(x: int) -> str:`

---

## 📖 Next Steps

1. ✅ **Start with Phase 1**: [phase1_foundations/README.md](phase1_foundations/README.md)
2. ✅ **Set up environment**: Create `.env` with API keys
3. ✅ **Run first example**: Try [NumPy basics](phase1_foundations/01_numpy_basics/)
4. ✅ **Document progress**: Update [Serena memories](.serena/memories/)
5. ✅ **Build habit**: 1-2 hours daily, consistent progress

---

## 📝 License

Educational project for AI development learning.

**Happy Learning! 🚀**
