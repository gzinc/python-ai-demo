# LLM Integration - Working with LangChain Chat Models

Learn how to integrate different LLM providers (OpenAI, Anthropic) through LangChain's unified interface.

## Learning Objectives

After completing this module, you will understand:

1. **Unified Chat Interface**: How LangChain abstracts different LLM providers
2. **Model Configuration**: Temperature, max_tokens, streaming, and other parameters
3. **Provider Switching**: How to swap between OpenAI and Anthropic seamlessly
4. **Streaming Responses**: Real-time token streaming for better UX
5. **Error Handling**: Retry logic, fallbacks, and provider-specific errors
6. **Model Capabilities**: Understanding what each provider excels at

## Why This Matters for AI Development

- **Provider Independence**: Switch LLM providers without rewriting application code
- **Cost Optimization**: Route requests to most cost-effective provider
- **Reliability**: Implement fallbacks when primary provider fails
- **Performance**: Use streaming for responsive user interfaces
- **Best Practices**: Learn model configuration patterns from production systems

## Files

### `concepts.py` - LLM Interface Patterns (No API Key Needed)

Run: `uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.concepts`

**No API required**: Shows interface construction and configuration only

Covers:
1. Chat model interface overview
2. Message types and structure
3. Configuration parameters explained
4. Streaming vs non-streaming patterns
5. Provider comparison matrix
6. Error handling strategies
7. Retry and fallback patterns
8. Token counting and cost estimation

### `practical.py` - Hands-On with Real LLMs (Requires API Keys)

Run: `uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.practical`

**Requires**: `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in `.env`

Covers:
1. ChatOpenAI basic usage
2. ChatAnthropic basic usage
3. Temperature and creativity control
4. Streaming responses in action
5. Provider switching patterns
6. Fallback chains (OpenAI → Anthropic)
7. Real-time token streaming
8. Cost tracking and optimization

## Key Concepts

### Unified Chat Interface

LangChain provides a consistent interface across all chat model providers:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Same interface, different providers
openai_llm = ChatOpenAI(model="gpt-4", temperature=0.7)
anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)

# Both use .invoke() with same message format
response = llm.invoke([HumanMessage(content="Hello!")])
```

### Configuration Parameters

| Parameter | Purpose | Range | Default |
|-----------|---------|-------|---------|
| `temperature` | Creativity vs consistency | 0.0-1.0 | 0.7 |
| `max_tokens` | Response length limit | 1-model_max | None |
| `streaming` | Real-time token streaming | True/False | False |
| `model` | Specific model version | Provider-specific | Latest |
| `timeout` | Request timeout | Seconds | 60 |
| `max_retries` | Retry failed requests | 0-10 | 2 |

### Streaming vs Non-Streaming

**Non-Streaming**: Wait for complete response
```python
response = llm.invoke(messages)  # Blocks until complete
print(response.content)
```

**Streaming**: Process tokens as they arrive
```python
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Provider Comparison

| Feature | OpenAI GPT-4 | Anthropic Claude-3.5 |
|---------|--------------|----------------------|
| Context Window | 128K tokens | 200K tokens |
| Streaming | Yes | Yes |
| Function Calling | Excellent | Excellent |
| Vision | Yes (GPT-4V) | Yes |
| Speed | Very Fast | Very Fast |
| Pricing | $$$ | $$$ |

## Integration with Previous Module

The prompts you learned in `01_prompts/` work seamlessly with these LLMs:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Combine prompts with LLMs using LCEL
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
chain = template | llm  # LCEL composition

response = chain.invoke({"question": "What are embeddings?"})
```

## Common Patterns

### Pattern 1: Provider Fallback
```python
from langchain_core.runnables import RunnableWithFallbacks

primary = ChatOpenAI(model="gpt-4")
fallback = ChatAnthropic(model="claude-3-5-sonnet-20241022")

llm = primary.with_fallbacks([fallback])
```

### Pattern 2: Cost-Based Routing
```python
# Use cheaper model for simple queries
simple_llm = ChatOpenAI(model="gpt-3.5-turbo")
complex_llm = ChatOpenAI(model="gpt-4")

# Route based on complexity
llm = simple_llm if is_simple_query else complex_llm
```

### Pattern 3: Streaming UI
```python
# Real-time display for better UX
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
print()  # newline at end
```

## Next Steps

After mastering LLM integration, you'll learn:
- **Module 03 - Chains**: Combining LLMs with prompts and parsers
- **Module 04 - Memory**: Adding conversation history to LLMs
- **Module 05 - RAG**: Connecting LLMs to your data
- **Module 06 - Agents**: Giving LLMs tools to use

## Exercises

1. **Provider Comparison**: Run the same prompt through both OpenAI and Anthropic, compare responses
2. **Streaming Demo**: Build a simple CLI that streams responses in real-time
3. **Fallback Chain**: Implement a fallback from GPT-4 to GPT-3.5-turbo on errors
4. **Cost Tracker**: Add logging to track token usage and estimated costs
5. **Temperature Experiment**: Test same prompt with temperature 0.0, 0.5, 1.0

## References

- [LangChain Chat Models](https://python.langchain.com/docs/integrations/chat/)
- [OpenAI Models](https://platform.openai.com/docs/models)
- [Anthropic Models](https://docs.anthropic.com/claude/docs/models-overview)
- [Streaming Guide](https://python.langchain.com/docs/expression_language/streaming)
