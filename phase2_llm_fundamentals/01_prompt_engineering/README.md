# Prompt Engineering

Master the art of communicating with LLMs effectively.

## Why Prompt Engineering Matters

The same LLM can give terrible or amazing results depending on your prompt. Prompt engineering is the skill that makes the difference.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROMPT QUALITY → OUTPUT QUALITY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ❌ VAGUE PROMPT                      ✅ ENGINEERED PROMPT                │
│   ┌───────────────────┐                ┌───────────────────────────────┐    │
│   │ "Summarize this"  │                │ "Summarize in 3 bullet points │    │
│   └─────────┬─────────┘                │  focusing on key decisions"   │    │
│             │                          └───────────────┬───────────────┘    │
│             ▼                                          ▼                    │
│   ┌───────────────────┐                ┌───────────────────────────────┐    │
│   │ Random length     │                │ • Decision 1: [specific]      │    │
│   │ Inconsistent focus│                │ • Decision 2: [specific]      │    │
│   │ May miss key info │                │ • Decision 3: [specific]      │    │
│   └───────────────────┘                └───────────────────────────────┘    │
│         ⚠️ Unreliable                           ✅ Consistent              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MESSAGE ROLES IN LLM CALLS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ SYSTEM (🎭 Invisible to User)                                       │    │ 
│  │ "You are a helpful coding assistant."                               │    │
│  │                                                                     │    │
│  │ Purpose: Sets persona, rules, behavior constraints                  │    │
│  │ Token cost: Counted but usually small                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ USER (👤 Human Input)                                               │    │
│  │ "How do I read a file in Python?"                                   │    │
│  │                                                                     │    │
│  │ Purpose: The actual question or request                             │    │
│  │ Token cost: Counted as input tokens                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ ASSISTANT (🤖 LLM Response)                                         │    │
│  │ "Use open() with a context manager..."                              │    │
│  │                                                                     │    │
│  │ Purpose: LLM's generated answer (or history in multi-turn)          │    │
│  │ Token cost: Counted as output tokens (more expensive!)              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEW-SHOT LEARNING PATTERN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   "Show me examples, and I'll learn the pattern"                            │
│                                                                             │
│   ┌──────────────────────────────────────┐                                  │
│   │ Example 1 (teaches format)           │                                  │
│   │ Input: John is 30 years old          │ ───┐                             │
│   │ Output: {"name": "John", "age": 30}  │    │                             │
│   └──────────────────────────────────────┘    │                             │
│                                               │  LLM learns:                │
│   ┌──────────────────────────────────────┐    │  - Extract name             │
│   │ Example 2 (reinforces pattern)       │    │  - Extract age              │
│   │ Input: Sarah is 25 years old         │ ───┼──► Format as JSON           │
│   │ Output: {"name": "Sarah", "age": 25} │  - Key names                     │
│   └──────────────────────────────────────┘    │  - Structure                │
│                                               │                             │
│   ┌──────────────────────────────────────┐    │                             │
│   │ NEW INPUT (LLM applies pattern)      │ ◄──┘                             │
│   │ Input: Mike is 40 years old          │                                  │
│   │ Output: ???                          │ ────► {"name": "Mike", "age": 40}│
│   └──────────────────────────────────────┘           ✅ Correct!            │
│                                                                             │
│   More examples = Better accuracy (but more tokens)                         │
│   2-3 examples usually sufficient for simple patterns                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHAIN-OF-THOUGHT REASONING                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WITHOUT CoT:                          WITH CoT:                           │
│   ┌─────────────────────┐               ┌─────────────────────────────────┐ │
│   │ Q: $20, 25% off,    │               │ Q: $20, 25% off, final price?   │ │
│   │    final price?     │               └────────────┬────────────────────┘ │
│   └──────────┬──────────┘                            │                      │
│              │                                       ▼                      │
│              │                          ┌─────────────────────────────────┐ │
│              │                          │ Step 1: Calculate discount      │ │
│              │                          │ 25% of $20 = $20 × 0.25 = $5    │ │
│              │                          └────────────┬────────────────────┘ │
│              │                                       │                      │
│              │                                       ▼                      │
│              │                          ┌─────────────────────────────────┐ │
│              │                          │ Step 2: Subtract from original  │ │
│              │                          │ $20 - $5 = $15                  │ │
│              │                          └────────────┬────────────────────┘ │
│              │                                       │                      │
│              ▼                                       ▼                      │
│       ┌──────────────┐                  ┌─────────────────────────────────┐ │
│       │ Answer: $16  │                  │ Answer: $15                     │ │
│       │     ❌       │                  │     ✅                          │ │
│       └──────────────┘                  └─────────────────────────────────┘ │
│   (jumped to answer,                    (showed work, caught errors)        │
│    made calculation error)                                                  │
│                                                                             │
│   KEY: "Let's think step by step" → Forces intermediate reasoning steps     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG PROMPT ASSEMBLY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User asks: "How do I deploy to production?"                               │
│                                                                             │
│   ┌──────────────────┐      ┌──────────────────┐      ┌────────────────┐    │
│   │    Retrieval     │      │   Prompt Builder │      │     LLM        │    │
│   │    (ChromaDB)    │ ───► │   (Assemble)     │ ───► │   (Generate)   │    │
│   └──────────────────┘      └──────────────────┘      └────────────────┘    │
│           │                          │                        │             │
│           ▼                          ▼                        ▼             │
│   ┌──────────────────┐      ┌──────────────────────┐ ┌────────────────┐     │
│   │ Retrieved chunks:│      │ "Context:            │ │ "Based on the  │     │
│   │ - deploy.md:15   │      │  [chunk1]            │ │  docs, you     │     │
│   │ - config.md:42   │      │  [chunk2]            │ │  should...     │     │
│   │                  │      │                      │ │  [grounded!]   │     │
│   │                  │      │  Question: How do I  │ │                │     │
│   │                  │      │  deploy to prod?     │ │                │     │
│   │                  │      │                      │ │                │     │
│   │                  │      │  Answer only from    │ │                │     │
│   │                  │      │  context above."     │ │                │     │
│   └──────────────────┘      └──────────────────────┘ └────────────────┘     │
│                                                                             │
│   KEY: The prompt template is critical for grounding answers in context     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
1. [02_api_integration](../02_api_integration/): Connect to OpenAI/Anthropic APIs
2. [03_embeddings](../03_embeddings/): Generate embeddings programmatically
3. [Phase 3](../../phase3_llm_applications/): Build complete RAG systems

---

**Run the examples now:**
```bash
uv run python phase2_llm_fundamentals/01_prompt_engineering/examples.py
```
