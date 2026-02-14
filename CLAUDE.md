# CLAUDE.md

This file provides guidance to Claude Code when working with this AI development learning project.

## Project Overview

**Purpose**: Comprehensive AI development roadmap focused on LLM applications and AI agents
**Learner**: Experienced Java developer transitioning to Python, with intermediate Python skills
**Learning Style**: Theory + hands-on projects, documented progress
**Structure**: 7 phases from foundations to production (Phases 2-5 complete, Phase 7 in progress)

## Project Structure

| Directory | Purpose |
|-----------|---------|
| [phase1_foundations/](phase1_foundations/) | NumPy, Pandas, ML basics (OPTIONAL) |
| [phase2_llm_fundamentals/](phase2_llm_fundamentals/) | Prompting, APIs, embeddings ✅ |
| [phase3_llm_applications/](phase3_llm_applications/) | RAG, chat, function calling ✅ |
| [phase4_ai_agents/](phase4_ai_agents/) | Agent systems and orchestration ✅ |
| [phase5_production/](phase5_production/) | Deployment and monitoring ✅ |
| [phase6_model_customization/](phase6_model_customization/) | Fine-tuning, LoRA/QLoRA (EXPLORATORY) |
| [phase7_frameworks/](phase7_frameworks/) | LangChain, LangGraph, LlamaIndex 🔄 |
| [docs/](docs/) | Learning resources |
| [.serena/memories/](.serena/memories/) | Progress tracking |
| [tests/](tests/) | Test suites |

## Essential Commands

```bash
# Setup
uv sync                          # Install dependencies

# Run examples
make run-numpy                   # Phase 1: NumPy examples
uv run python <path/to/file.py>  # Run specific example

# Development
make test                        # Run tests
make format                      # Format code
make lint                        # Lint code

# Progress
cat .serena/memories/learning_progress.md
```

## Learning Philosophy

### Progressive Complexity
- Phase 1: Foundations (data manipulation) - OPTIONAL ⏸️
- Phase 2: LLM basics (APIs, prompting) ✅
- Phase 3: Applications (RAG, chat) ✅
- Phase 4: Agents (autonomous systems) ✅
- Phase 5: Production (deployment, monitoring) ✅
- Phase 6: Model Customization (fine-tuning, LoRA) - EXPLORATORY ⬜
- Phase 7: Frameworks (LangChain, LangGraph, LlamaIndex) 🔄

### Hands-On Focus
- Every concept has practical examples
- Exercises before solutions
- Real-world AI applications
- Portfolio-worthy projects

### Documentation-Driven
- Session memories in [.serena/memories/](.serena/memories/)
- Progress tracking after each session
- Concepts, questions, and insights documented
- Pattern: `session_YYYY_MM_DD_topic.md`

## Working with Claude Code

### When Adding New Content

1. **Match Existing Style**:
   - Look at [phase1_foundations/01_numpy_basics/](phase1_foundations/01_numpy_basics/) for reference
   - Include: README, examples.py, exercises.py
   - Add AI-specific applications, not just generic tutorials

2. **Progressive Difficulty**:
   - Start simple, build complexity
   - Connect to previous modules
   - Preview next modules

3. **AI Focus**:
   - Always relate to LLM applications
   - Show real AI use cases
   - Explain "why this matters for AI"

### Memory Management

**Update memories after**:
- Completing a module
- Significant learning moments
- Building projects
- Overcoming challenges

**Memory files** (in [.serena/memories/](.serena/memories/)):
- [project_overview.md](.serena/memories/project_overview.md) - High-level status
- [learning_progress.md](.serena/memories/learning_progress.md) - Module completion tracking
- `session_YYYY_MM_DD_*.md` - Daily sessions
- [concepts_learned.md](.serena/memories/concepts_learned.md) - Key insights

### Code Style

**Python Conventions**:
- Use type hints throughout
- Docstrings for all functions
- lowercase comments (per global CLAUDE.md)
- Descriptive variable names

**Multi-line Strings**: Always use `inspect.cleandoc()` for multi-line strings inside functions:
```python
from inspect import cleandoc

def example():
    # ✅ correct - cleandoc handles dedent + strip in one call
    text = cleandoc('''
        content here
        indented with code
    ''')

    # ❌ wrong - flush-left inside function
    text = '''
content at column 0
'''
```

**Multi-line Print Statements**: Use single `print("\n" + cleandoc(...))` instead of multiple print calls:
```python
from inspect import cleandoc

def example():
    # ✅ correct - single print with cleandoc and newline prefix
    print("\n" + cleandoc("""
        1. State Schema Defined:
           class SimpleState(TypedDict):
               counter: int
               message: str
    """))

    # ❌ wrong - multiple print calls
    print("\n1. State Schema Defined:")
    print("   class SimpleState(TypedDict):")
    print("       counter: int")
    print("       message: str")
```
**Key points:**
- Use `"\n" + cleandoc(...)` to preserve leading newline
- No empty line after opening `"""`
- Benefits: More readable code, easier to maintain, follows DRY principle

**AI-Specific**:
- Comment why techniques matter for AI
- Show performance comparisons
- Relate to LLM/embedding concepts

## Phase-Specific Guidance

### Phase 1: Foundations
- **Focus**: NumPy arrays = embeddings, vectors
- **Connection**: All AI work builds on these fundamentals
- **Key**: Vectorization, broadcasting for performance

### Phase 2: LLM Fundamentals
- **Focus**: Working with LLM APIs, prompting
- **Connection**: Foundation for all LLM applications
- **Key**: Embeddings, token management, cost awareness

### Phase 3: LLM Applications
- **Focus**: RAG systems, chat, function calling
- **Connection**: Production-ready AI applications
- **Key**: Retrieval, context, tool integration

### Phase 4: AI Agents
- **Focus**: Autonomous systems, tool use
- **Connection**: Advanced AI architectures
- **Key**: Planning, execution, multi-agent orchestration

### Phase 5: Production
- **Focus**: Deployment, monitoring, optimization
- **Connection**: Real-world AI systems
- **Key**: FastAPI, caching, observability

### Phase 6: Model Customization (EXPLORATORY)
- **Focus**: Fine-tuning, LoRA/QLoRA, transfer learning
- **Connection**: Advanced model adaptation
- **Key**: When to fine-tune vs RAG, dataset preparation, PEFT
- **Status**: Scaffolded, ready to explore based on interest

### Phase 7: Production Frameworks (IN PROGRESS 🔄)
- **Focus**: LangChain, LangGraph, LlamaIndex
- **Connection**: Framework patterns for production AI
- **Key**: Learn AFTER building from scratch (Phases 2-6)
- **Status**: LangChain basics complete, LangGraph in progress (state basics Demo 4/6), LlamaIndex files created but not processed
- **Philosophy**: Understand what frameworks do under the hood, when to use them, and when to skip them

## Common Tasks

### Adding a New Module

1. Create directory: `phaseN/<module_name>/`
2. Add README.md with:
   - Learning objectives
   - Key concepts
   - AI applications
   - Exercises
3. Create `examples.py` with runnable code
4. Create `exercises.py` with practice problems
5. Create `solutions/` directory
6. Update phase README
7. Update [.serena/memories/learning_progress.md](.serena/memories/learning_progress.md)

### Adding Examples

**Template**:
```python
"""
Module: <Name> - <Purpose>

Run with: uv run python <path>
"""

import numpy as np  # or relevant imports
from typing import <types>


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def example_function() -> None:
    """demonstrate specific concept"""
    print_section("Example Title")

    # code with explanatory prints
    print(f"result: {result}")


def main() -> None:
    """run all examples"""
    example_function()
    # ... other examples


if __name__ == "__main__":
    main()
```

### Adding Documentation

**In [docs/concepts/](docs/concepts/)**:
- Theoretical background
- "How it works" explanations
- References to research/papers

**In [docs/guides/](docs/guides/)**:
- Step-by-step tutorials
- Best practices
- Troubleshooting

**In [docs/references/](docs/references/)**:
- Quick lookup tables
- Command references
- API cheat sheets

## Dependencies

**Core AI/ML**:
- `numpy`: Array operations
- `pandas`: Data manipulation
- `openai`: OpenAI API
- `anthropic`: Anthropic API
- `langchain`: LLM framework
- `llama-index`: RAG framework
- `chromadb`: Vector database

**Web/API**:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `httpx`: HTTP client

**Dev Tools**:
- `pytest`: Testing
- `ruff`: Linting/formatting
- `mypy`: Type checking

## Integration with Other Projects

**python-demo** (../python-demo):
- FastAPI patterns apply to Phase 5
- Testing patterns reusable
- Configuration management similar

**Shared Learnings**:
- Python fundamentals
- uv package manager
- Development workflows
- Code quality standards

## Tips for Claude

1. **Check Phase Context**: Understand which phase learner is in
2. **Reference Existing**: Point to similar examples in other modules
3. **Hands-On First**: Provide runnable code, not just explanations
4. **AI Connection**: Always relate to LLM/AI applications
5. **Update Memories**: Remind to document learnings
6. **Progressive**: Don't jump ahead to advanced concepts
7. **Practical**: Focus on building, not just theory

## Session Pattern

1. **Start**: Review previous session memory
2. **Learn**: Work through new content (README → examples.py)
3. **Practice**: Complete exercises
4. **Build**: Apply to project
5. **Document**: Update session memory
6. **Track**: Update [learning_progress.md](.serena/memories/learning_progress.md)

**Session Memory Template**:
```markdown
# Session: YYYY-MM-DD - Phase N: Module Name

## What I Learned
- Concept 1:
- Concept 2:

## Exercises Completed
- [ ] Exercise 1
- [ ] Exercise 2

## Challenges & Solutions
- Problem:
- Solution:

## AI Application Insights
- How this applies to LLM work:

## Next Steps
1.
2.
```

## Quality Standards

- **Code**: Type hints, docstrings, tested
- **Examples**: Runnable, practical, AI-relevant
- **Documentation**: Clear, progressive, hands-on
- **Exercises**: Challenging but achievable
- **Projects**: Portfolio-worthy, production-oriented

---

**Remember**: This is a learning journey. Make it practical, make it engaging, and always connect to real AI applications.
