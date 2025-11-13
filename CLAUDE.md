# CLAUDE.md

This file provides guidance to Claude Code when working with this AI development learning project.

## Project Overview

**Purpose**: Comprehensive AI development roadmap focused on LLM applications and AI agents
**Learner**: Experienced Java developer transitioning to Python, with intermediate Python skills
**Learning Style**: Theory + hands-on projects, documented progress
**Structure**: 5 progressive phases from foundations to production

## Project Structure

```
python-ai-demo/
├── phase1_foundations/          # NumPy, Pandas, ML basics
├── phase2_llm_fundamentals/     # Prompting, APIs, embeddings
├── phase3_llm_applications/     # RAG, chat, function calling
├── phase4_ai_agents/            # Agent systems and orchestration
├── phase5_production/           # Deployment and monitoring
├── docs/                        # Learning resources
├── .serena/memories/            # Progress tracking
└── tests/                       # Test suites
```

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
- Phase 1: Foundations (data manipulation)
- Phase 2: LLM basics (APIs, prompting)
- Phase 3: Applications (RAG, chat)
- Phase 4: Agents (autonomous systems)
- Phase 5: Production (deployment, monitoring)

### Hands-On Focus
- Every concept has practical examples
- Exercises before solutions
- Real-world AI applications
- Portfolio-worthy projects

### Documentation-Driven
- Session memories in `.serena/memories/`
- Progress tracking after each session
- Concepts, questions, and insights documented
- Pattern: `session_YYYY_MM_DD_topic.md`

## Working with Claude Code

### When Adding New Content

1. **Match Existing Style**:
   - Look at `phase1_foundations/01_numpy_basics/` for reference
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

**Memory files**:
- `project_overview.md` - High-level status
- `learning_progress.md` - Module completion tracking
- `session_YYYY_MM_DD_*.md` - Daily sessions
- `concepts_learned.md` - Key insights

### Code Style

**Python Conventions**:
- Use type hints throughout
- Docstrings for all functions
- lowercase comments (per global CLAUDE.md)
- Descriptive variable names

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
7. Update `.serena/memories/learning_progress.md`

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

**In `docs/concepts/`**:
- Theoretical background
- "How it works" explanations
- References to research/papers

**In `docs/guides/`**:
- Step-by-step tutorials
- Best practices
- Troubleshooting

**In `docs/references/`**:
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
6. **Track**: Update learning_progress.md

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
