# Quick Start Guide - AI Development Roadmap

Get up and running in 5 minutes!

## 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all project dependencies
uv sync
```

## 2. Set Up API Keys (Optional for Phase 1)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# (You'll need these for Phase 2+, not Phase 1)
```

## 3. Run Your First Example

```bash
# Run NumPy basics examples
make run-numpy

# Or directly:
uv run python phase1_foundations/01_numpy_basics/examples.py
```

**Expected Output**: You should see array examples, vectorization demos, and AI application examples.

## 4. Start Learning

### Recommended First Day

1. **Read the main README**:
   ```bash
   cat README.md
   ```

2. **Review Phase 1 overview**:
   ```bash
   cat phase1_foundations/README.md
   ```

3. **Study NumPy module**:
   ```bash
   cd phase1_foundations/01_numpy_basics
   cat README.md
   ```

4. **Run examples**:
   ```bash
   uv run python examples.py
   ```

5. **Try exercises** (don't peek at solutions!):
   ```bash
   uv run python exercises.py  # When you create them
   ```

6. **Document your session**:
   ```bash
   # Create your first session memory
   echo "# Session: $(date +%Y-%m-%d) - NumPy Basics" > .serena/memories/session_$(date +%Y_%m_%d)_numpy.md
   ```

## 5. Daily Workflow

**Each session**:

1. **Review** previous session memory
2. **Learn** new content (README → examples)
3. **Practice** exercises
4. **Build** mini-project
5. **Document** learnings

**Commands**:
```bash
# Check your progress
cat .serena/memories/learning_progress.md

# Run code quality checks
make format  # Format your code
make lint    # Check for issues

# Run tests (as you build projects)
make test
```

## 6. Learning Path

**Week 1-2**: Phase 1 - Foundations
- Day 1-3: NumPy basics
- Day 4-7: Pandas data manipulation
- Day 8-10: ML concepts
- Day 11-14: Practice projects

**Week 3-4**: Phase 2 - LLM Fundamentals
- Set up API keys
- Prompt engineering
- Embeddings and vectors

**Continue** through phases 3, 4, 5...

## 7. Getting Help

**Within this project**:
- Check `docs/concepts/` for theory
- Check `docs/guides/` for tutorials
- Check `docs/references/` for quick lookup

**External resources**:
- NumPy docs: https://numpy.org/doc/
- Pandas docs: https://pandas.pydata.org/docs/
- OpenAI docs: https://platform.openai.com/docs
- Anthropic docs: https://docs.anthropic.com

## 8. Check Your Python Demo Project

You have another learning project at `../python-demo`. The patterns there apply here:

```bash
# Your FastAPI patterns will be useful in Phase 5
cd ../python-demo
cat README.md
```

## Common Issues

**uv not found**:
```bash
# Ensure uv is in PATH
source ~/.bashrc  # or ~/.zshrc
```

**Import errors**:
```bash
# Make sure you've synced dependencies
uv sync
```

**API key errors** (Phase 2+):
```bash
# Check .env file exists and has valid keys
cat .env
```

## Next Steps

✅ **You're all set!** Start with:
```bash
cd phase1_foundations/01_numpy_basics
uv run python examples.py
```

**Remember**:
- Take your time, understand don't memorize
- Build every example yourself
- Document what you learn
- 1-2 hours daily is better than marathon sessions

**Happy learning! 🚀**
