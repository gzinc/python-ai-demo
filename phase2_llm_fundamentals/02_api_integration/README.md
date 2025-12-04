# Module 2: API Integration - Connecting to Real LLMs

## Learning Objectives

By the end of this module, you will:
- Connect to OpenAI and Anthropic APIs
- Understand API request/response patterns
- Handle errors, rate limits, and retries
- Manage API keys securely
- Stream responses for better UX
- Track token usage and costs

## Prerequisites

- Phase 2 Module 1: Prompt Engineering
- API keys (at least one):
  - OpenAI: https://platform.openai.com/api-keys
  - Anthropic: https://console.anthropic.com/

## Key Concepts

### 1. API Architecture

```
Your Code → HTTP Request → LLM Provider → HTTP Response → Your Code
              ↓                              ↓
         - model name                   - generated text
         - messages                     - token usage
         - parameters                   - finish reason
```

### 2. Request Parameters

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `model` | Which LLM to use | gpt-4o, claude-sonnet-4-20250514 |
| `messages` | Conversation history | List of role/content dicts |
| `temperature` | Randomness (0=deterministic) | 0.0 - 1.0 |
| `max_tokens` | Response length limit | 100 - 4096 |
| `stream` | Get tokens as generated | True/False |

### 3. Token Economics

```
Input tokens:  System prompt + User message + Context
Output tokens: Model's response

Cost = (input_tokens × input_price) + (output_tokens × output_price)

Example (GPT-4o):
- Input: $2.50/1M tokens
- Output: $10.00/1M tokens
- 1000 token request + 500 token response ≈ $0.0075
```

### 4. Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| 401 | Invalid API key | Check key, regenerate if needed |
| 429 | Rate limit | Exponential backoff, wait and retry |
| 500 | Server error | Retry with backoff |
| Timeout | Slow response | Increase timeout, retry |

## AI Applications

### Why This Matters for LLM Development

1. **Foundation for Everything**: Every AI app calls LLMs via APIs
2. **Cost Control**: Understanding tokens = controlling spend
3. **User Experience**: Streaming makes apps feel responsive
4. **Reliability**: Proper error handling = production-ready apps
5. **Multi-Provider**: Switch between OpenAI/Anthropic for different tasks

### Real-World Patterns

```python
# Simple completion
response = client.chat.completions.create(...)

# Streaming for chat UIs
for chunk in client.chat.completions.create(stream=True, ...):
    print(chunk.choices[0].delta.content, end="")

# Batch processing with rate limiting
for item in items:
    response = process(item)
    time.sleep(0.1)  # respect rate limits
```

## Files in This Module

| File | Purpose |
|------|---------|
| `examples.py` | API integration patterns (works without keys using mocks) |
| `live_examples.py` | Real API calls (requires keys) |
| `exercises.py` | Practice problems |

## Setup

### Environment Variables

```bash
# Add to ~/.bashrc or .env file
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Verify Setup

```bash
# Check if keys are set
echo $OPENAI_API_KEY | head -c 10
echo $ANTHROPIC_API_KEY | head -c 10
```

## Running Examples

```bash
# Pattern examples (no API key needed)
uv run python phase2_llm_fundamentals/02_api_integration/examples.py

# Live API calls (requires keys)
uv run python phase2_llm_fundamentals/02_api_integration/live_examples.py
```

## Exercises

1. **Basic Call**: Make a simple API call and print the response
2. **Temperature Experiment**: Compare outputs at temp 0.0 vs 1.0
3. **Token Counter**: Track and display token usage
4. **Error Handler**: Implement retry logic with exponential backoff
5. **Streaming Chat**: Build a simple streaming chat loop

## Key Takeaways

1. **Always handle errors** - Networks fail, rate limits hit
2. **Track token usage** - Costs add up fast
3. **Use streaming for UX** - Users hate waiting
4. **Secure your keys** - Never commit to git
5. **Start with low temperature** - Increase only if needed

## Next Steps

After completing this module:
- → Module 3: Embeddings (vector representations)
- → Phase 3: Build RAG systems with real APIs
