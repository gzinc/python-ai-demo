# Chains Module

**Purpose**: Learn LangChain's LCEL (LangChain Expression Language) for building multi-step LLM workflows

---

## Learning Objectives

By the end of this module, you will:
1. **LCEL Syntax**: Master the pipe operator (|) for chain composition
2. **Runnables**: Understand the Runnable interface (.invoke(), .stream(), .batch())
3. **Composition Patterns**: Parallel chains, passthrough, fallbacks
4. **Error Handling**: Retries, fallbacks, and graceful failure handling
5. **Debugging**: Trace execution and inspect intermediate results

---

## What is a Chain?

A **chain** connects multiple components (prompts, LLMs, parsers, tools) into a unified workflow:

```
Input → [Component 1] → [Component 2] → [Component 3] → Output
```

**Benefits**:
- **Composability**: Build complex workflows from simple components
- **Reusability**: Save and reuse chain definitions
- **Maintainability**: Modify individual components without rewriting logic
- **Observability**: Track intermediate steps and debug failures

---

## LCEL: Modern Chain Composition

**The pipe operator (`|`)** for composing components:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence")
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

# modern LCEL composition
chain = prompt | llm | parser

# use .invoke() not .run()
result = chain.invoke({"concept": "embeddings"})
```

**Why LCEL?**

**LCEL Benefits:**
- ✅ Concise pipe syntax (like Unix pipes)
- ✅ Built-in streaming support
- ✅ Native async execution
- ✅ Type-safe composition
- ✅ Advanced error handling (retries, fallbacks)
- ✅ Parallel and conditional execution
- ✅ Production-ready patterns

---

## LCEL Components

### Runnables (Base Interface)

All LCEL components implement the `Runnable` interface:

```python
# Core methods
result = runnable.invoke(input)           # Single execution
results = runnable.batch([input1, input2])  # Batch execution
for chunk in runnable.stream(input):      # Streaming
    print(chunk)

# Async versions
result = await runnable.ainvoke(input)
results = await runnable.abatch([input1, input2])
async for chunk in runnable.astream(input):
    print(chunk)
```

### Common Runnables

| Type | Purpose | Example |
|------|---------|---------|
| `ChatPromptTemplate` | Format prompts | `ChatPromptTemplate.from_messages(...)` |
| `ChatOpenAI` | LLM execution | `ChatOpenAI(model="gpt-4")` |
| `StrOutputParser` | Parse string output | `StrOutputParser()` |
| `RunnablePassthrough` | Pass data unchanged | `RunnablePassthrough()` |
| `RunnableLambda` | Custom function | `RunnableLambda(lambda x: x.upper())` |

---

## LCEL Patterns

### Pattern 1: Basic Chain (Prompt → LLM → Parser)

```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "AI"})
```

### Pattern 2: Parallel Chains (RunnableParallel)

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    "summary": summarize_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain
})

result = chain.invoke({"text": "Article content..."})
# {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

### Pattern 3: Branching Logic (RunnableBranch)

```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: len(x["text"]) < 100, short_text_chain),
    (lambda x: len(x["text"]) < 1000, medium_text_chain),
    long_text_chain  # default
)
```

### Pattern 4: Retry with Fallback

```python
from langchain_core.runnables import RunnableWithFallbacks

primary = prompt | gpt4_llm | parser
fallback = prompt | gpt35_llm | parser

chain = primary.with_fallbacks([fallback])
```

### Pattern 5: Passthrough + Side Information

```python
from langchain_core.runnables import RunnablePassthrough

chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
} | prompt | llm | parser

# Input: {"question": "What is RAG?"}
# Prompt gets: {"context": "...", "question": "What is RAG?"}
```

---

## Error Handling

### Strategy 1: Try/Except

```python
try:
    result = chain.invoke({"input": "..."})
except Exception as e:
    print(f"Chain failed: {e}")
    # Fallback logic
```

### Strategy 2: Fallback Chains

```python
chain = primary_chain.with_fallbacks([
    fallback_chain_1,
    fallback_chain_2
])
```

### Strategy 3: Retry Configuration

```python
from langchain_core.runnables import RunnableRetry

chain = (prompt | llm).with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1,
    wait_exponential_max=10
)
```

---

## Debugging Chains

### Strategy 1: Verbose Mode

```python
chain = prompt | llm | parser
result = chain.invoke({"input": "..."}, config={"verbose": True})
```

### Strategy 2: Intermediate Steps

```python
# Using RunnablePassthrough to inspect
from langchain_core.runnables import RunnablePassthrough

def debug_print(x):
    print(f"Intermediate: {x}")
    return x

chain = (
    prompt
    | RunnablePassthrough(func=debug_print)
    | llm
    | parser
)
```

### Strategy 3: LangSmith Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# All chain executions now traced in LangSmith
```

---

## Performance Optimization

### 1. Batch Processing

```python
inputs = [{"concept": c} for c in ["RAG", "embeddings", "agents"]]
results = chain.batch(inputs)  # Parallel execution
```

### 2. Streaming for UX

```python
for chunk in chain.stream({"concept": "embeddings"}):
    print(chunk, end="", flush=True)
```

### 3. Async for High Concurrency

```python
import asyncio

async def process():
    tasks = [chain.ainvoke({"concept": c}) for c in concepts]
    results = await asyncio.gather(*tasks)
    return results
```

---

## Run Examples

**📊 Visual Learning**: All practical demos include comprehensive ASCII diagrams showing chain composition, data flow, and execution patterns.

```bash
# Conceptual demos (no API key required)
uv run python -m phase7_frameworks.01_langchain_basics.03_chains.concepts

# Practical demos (requires OPENAI_API_KEY)
uv run python -m phase7_frameworks.01_langchain_basics.03_chains.practical
```

---

## Common Pitfalls

### 1. Forgetting Output Parsers

```python
# ❌ Bad: Returns AIMessage object
chain = prompt | llm
result = chain.invoke({"concept": "embeddings"})
# result.content → Need to extract manually

# ✅ Good: Parse to string
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"concept": "embeddings"})
# result → Direct string
```

### 2. Ignoring Streaming

```python
# ❌ Bad: Wait for full response
response = llm.invoke(messages)
print(response.content)

# ✅ Good: Stream for better UX
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

---

## When to Use Chains

### ✅ Good Use Cases

- **Multi-step LLM workflows**: Summarize → Analyze → Format
- **Standardized pipelines**: Reusable logic across applications
- **Complex composition**: Parallel branches, conditional logic
- **Production systems**: Need observability, retries, fallbacks

### ❌ When to Skip Chains

- **Single LLM call**: Just use `llm.invoke()` directly
- **Maximum control**: Custom logic doesn't fit chain patterns
- **Performance critical**: Chain overhead matters (microseconds, but measurable)

---

## Next Steps

After completing this module:
1. Practice LCEL syntax with various component combinations
2. Build a multi-step analysis chain (summarize → keywords → sentiment)
3. Move to Module 4 (Memory) to add conversation history to chains
4. Experiment with parallel chains and branching logic

---

## Resources

- [LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [Chain How-To Guides](https://python.langchain.com/docs/how_to/#chains)
- [Runnable Interface](https://python.langchain.com/docs/expression_language/interface)
- [LangSmith Tracing](https://docs.smith.langchain.com/)