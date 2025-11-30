# Prompt Engineering

Master the art of communicating with LLMs effectively.

## Why Prompt Engineering Matters

The same LLM can give terrible or amazing results depending on your prompt. Prompt engineering is the skill that makes the difference.

**Same question, different prompts:**
```
❌ "Summarize this"
   → Vague, inconsistent results

✅ "Summarize this document in 3 bullet points, focusing on key decisions"
   → Clear, actionable output
```

## Learning Objectives

By the end of this module, you will:
1. Write effective system prompts
2. Use few-shot learning with examples
3. Apply chain-of-thought reasoning
4. Control output format and structure
5. Handle edge cases and errors gracefully

## Key Concepts

### 1. Roles (System, User, Assistant)

```python
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
    {"role": "assistant", "content": "Use open() with a context manager..."}
]
```

- **System**: Sets behavior, persona, rules (invisible to user)
- **User**: The human's input
- **Assistant**: The LLM's response

### 2. Few-Shot Learning

Give examples to teach the pattern:

```python
prompt = """
Convert to JSON:
Input: John is 30 years old
Output: {"name": "John", "age": 30}

Input: Sarah is 25 years old
Output: {"name": "Sarah", "age": 25}

Input: Mike is 40 years old
Output:
"""
# LLM learns the pattern and continues correctly
```

### 3. Chain-of-Thought (CoT)

Make the LLM reason step by step:

```python
prompt = """
Solve this step by step:
Q: If a shirt costs $20 and is 25% off, what's the final price?

Let's think through this:
1. First, calculate the discount amount...
2. Then, subtract from original price...
"""
# Better accuracy for complex reasoning
```

### 4. Output Format Control

```python
prompt = """
Analyze this code and respond in JSON format:
{
    "has_bugs": true/false,
    "bugs": ["list of bugs"],
    "suggestions": ["list of improvements"]
}
"""
```

## Prompt Patterns for RAG

### Context + Question Pattern
```python
prompt = f"""
Context:
{retrieved_documents}

Question: {user_question}

Answer based only on the context above. If the answer isn't in the context, say "I don't have that information."
"""
```

### Citation Pattern
```python
prompt = f"""
Context:
[1] {doc1}
[2] {doc2}

Question: {question}

Answer the question and cite sources using [1], [2] format.
"""
```

## Module Structure

```
01_prompt_engineering/
├── README.md          # This file
├── examples.py        # Runnable examples (no API key needed!)
└── exercises.py       # Practice problems (coming soon)
```

## Quick Start

```bash
# run examples (works without API key - simulates responses)
uv run python phase2_llm_fundamentals/01_prompt_engineering/examples.py
```

## Topics Covered

1. **Basic Prompts** - Clear instructions
2. **System Prompts** - Setting behavior
3. **Few-Shot Learning** - Teaching by example
4. **Chain-of-Thought** - Step-by-step reasoning
5. **Output Formatting** - JSON, markdown, structured
6. **RAG Prompts** - Context + question patterns
7. **Error Handling** - Graceful fallbacks

## Connection to Your Journey

| Concept | Where You'll Use It |
|---------|---------------------|
| System prompts | Every LLM call |
| Few-shot | Teaching output format |
| Chain-of-thought | Complex reasoning tasks |
| RAG prompts | Phase 3 RAG systems |
| Output formatting | Structured responses |

## Next Steps

After this module:
1. **02_api_integration**: Connect to OpenAI/Anthropic APIs
2. **03_embeddings**: Generate embeddings programmatically
3. **Phase 3**: Build complete RAG systems

---

**Run the examples now:**
```bash
uv run python phase2_llm_fundamentals/01_prompt_engineering/examples.py
```