# Module 2: Chat Interface - Build Conversational AI

## Learning Objectives

By the end of this module, you will:
- Implement conversation memory for multi-turn chats
- Manage context windows and token budgets
- Stream responses for better user experience
- Handle reference resolution across turns
- Build a complete chat interface

## Prerequisites

- [Phase 2](../../phase2_llm_fundamentals/) complete (API Integration with messages=[])
- [Phase 3 Module 1](../01_rag_system/) (RAG System)
- Understanding of token limits and context windows

## Chat vs Single-Turn Q&A

```
SINGLE-TURN (RAG):
┌─────────┐         ┌─────────┐         ┌─────────┐
│  User   │────────►│   LLM   │────────►│ Answer  │
│ Question│         │         │         │         │
└─────────┘         └─────────┘         └─────────┘
     No memory - each question is independent


MULTI-TURN (Chat):
┌─────────────────────────────────────────────────────────────────┐
│                    Conversation Memory                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Turn 1: User: "What is Python?"                                │
│          Assistant: "Python is a programming language..."       │
│                           │                                      │
│                           ▼                                      │
│  Turn 2: User: "What are its main features?"                    │
│          Assistant: "Python's main features include..."         │
│                      ↑                                           │
│                      │                                           │
│          LLM knows "its" refers to Python from Turn 1!          │
│                           │                                      │
│                           ▼                                      │
│  Turn 3: User: "Show me an example"                             │
│          Assistant: "Here's a Python example..."                │
│                      ↑                                           │
│                      │                                           │
│          LLM knows context from Turn 1 AND Turn 2!              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## The messages=[] Structure (Review from Phase 2)

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Message Format                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  messages = [                                                    │
│      {"role": "system", "content": "You are a helpful..."},     │
│      {"role": "user", "content": "What is Python?"},            │
│      {"role": "assistant", "content": "Python is..."},          │
│      {"role": "user", "content": "What are its features?"},     │
│      {"role": "assistant", "content": "The main features..."},  │
│      {"role": "user", "content": "Show me an example"},  ◄─ NEW │
│  ]                                                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ The ENTIRE conversation is sent to LLM on every turn!   │   │
│  │ This is how it "remembers" - it re-reads everything.    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Management Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│              Context Window Problem                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LLM Context Window: 8,000 tokens (example)                     │
│                                                                  │
│  Turn 1:    ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (500 tokens)     │
│  Turn 2:    ████████░░░░░░░░░░░░░░░░░░░░░░░░  (1000 tokens)    │
│  Turn 3:    ████████████░░░░░░░░░░░░░░░░░░░░  (1500 tokens)    │
│  Turn 10:   ████████████████████████████████  (8000 tokens)    │
│  Turn 11:   ████████████████████████████████▓▓ OVERFLOW!       │
│                                                                  │
│  Problem: Conversations grow unbounded, eventually overflow!    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

SOLUTIONS:

Strategy 1: SLIDING WINDOW (keep last N turns)
┌─────────────────────────────────────────────────────────────────┐
│  [sys] [u1] [a1] [u2] [a2] [u3] [a3] [u4] [a4] [u5] [a5]       │
│                          ▼                                       │
│  Keep last 6 messages:                                          │
│  [sys] [u3] [a3] [u4] [a4] [u5] [a5]                           │
│         ↑                                                        │
│    Older messages dropped                                        │
└─────────────────────────────────────────────────────────────────┘

Strategy 2: SUMMARIZATION (compress old context)
┌─────────────────────────────────────────────────────────────────┐
│  [sys] [u1] [a1] [u2] [a2] [u3] [a3] [u4] [a4] [u5] [a5]       │
│                          ▼                                       │
│  Summarize old turns:                                           │
│  [sys] [summary: "User asked about X, we discussed Y"]          │
│        [u4] [a4] [u5] [a5]                                      │
│         ↑                                                        │
│    Context preserved but compressed                              │
└─────────────────────────────────────────────────────────────────┘

Strategy 3: TOKEN BUDGET (count and trim)
┌─────────────────────────────────────────────────────────────────┐
│  Budget: 6000 tokens for history (leave 2000 for response)      │
│                                                                  │
│  Before adding new message:                                      │
│    while count_tokens(messages) > 6000:                         │
│        remove oldest non-system message                          │
└─────────────────────────────────────────────────────────────────┘
```

## Streaming Responses

```
┌─────────────────────────────────────────────────────────────────┐
│              Non-Streaming vs Streaming                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NON-STREAMING (wait for complete response):                    │
│                                                                  │
│  User: "Explain quantum computing"                              │
│  [.......... 5 seconds of nothing ..........]                   │
│  Assistant: "Quantum computing is a type of computation..."     │
│             (entire response appears at once)                    │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  STREAMING (token by token):                                    │
│                                                                  │
│  User: "Explain quantum computing"                              │
│  Assistant: "Quantum "                                          │
│  Assistant: "Quantum computing "                                │
│  Assistant: "Quantum computing is "                             │
│  Assistant: "Quantum computing is a "                           │
│  Assistant: "Quantum computing is a type "                      │
│  ... (continues token by token)                                 │
│                                                                  │
│  ✅ Better UX - user sees progress immediately                  │
│  ✅ Can cancel mid-response                                     │
│  ✅ Feels more interactive/natural                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Implementation:

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  # non-streaming                                                 │
│  response = client.chat.completions.create(                     │
│      model="gpt-4o-mini",                                       │
│      messages=messages,                                          │
│  )                                                               │
│  print(response.choices[0].message.content)                     │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  # streaming                                                     │
│  stream = client.chat.completions.create(                       │
│      model="gpt-4o-mini",                                       │
│      messages=messages,                                          │
│      stream=True,  # ◄── the key difference                     │
│  )                                                               │
│  for chunk in stream:                                            │
│      if chunk.choices[0].delta.content:                         │
│          print(chunk.choices[0].delta.content, end="")          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Reference Resolution

```
┌─────────────────────────────────────────────────────────────────┐
│              How LLM Resolves References                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User: "Tell me about Tesla stock"                              │
│  Assistant: "Tesla (TSLA) is currently trading at..."           │
│                                                                  │
│  User: "What about its competitors?"                            │
│              ↑                                                   │
│              └── "its" = Tesla (from context)                   │
│                                                                  │
│  User: "Compare them"                                           │
│              ↑                                                   │
│              └── "them" = Tesla and competitors                 │
│                                                                  │
│  User: "Which one is better?"                                   │
│              ↑                                                   │
│              └── "which one" = among the compared companies     │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  This works ONLY because the LLM sees the full conversation!   │
│  Without history, "its competitors" has no meaning.             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Files in This Module

| File | Purpose |
|------|---------|
| [chat_memory.py](chat_memory.py) | ChatMemory class with different memory strategies |
| [streaming.py](streaming.py) | Streaming utilities for both OpenAI and Anthropic |
| [examples.py](examples.py) | Interactive chat demos and usage patterns |

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Chat Module Structure                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    examples.py                           │   │
│  │                  (Interactive Demos)                     │   │
│  │                                                          │   │
│  │  • example_basic_chat() - Simple conversation           │   │
│  │  • example_memory_strategies() - Compare approaches     │   │
│  │  • example_streaming() - Real-time responses            │   │
│  │  • example_interactive() - REPL-style chat              │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                     │
│              ▼                             ▼                     │
│  ┌─────────────────────────┐   ┌─────────────────────────┐     │
│  │     chat_memory.py      │   │     streaming.py        │     │
│  │                         │   │                         │     │
│  │  • ChatMemory class     │   │  • stream_openai()      │     │
│  │  • sliding_window()     │   │  • stream_anthropic()   │     │
│  │  • summarize_history()  │   │  • StreamPrinter        │     │
│  │  • token_budget()       │   │  • collect_stream()     │     │
│  └─────────────────────────┘   └─────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Running Examples

```bash
# run comprehensive examples
uv run python phase3_llm_applications/02_chat_interface/examples.py

# run individual modules
uv run python phase3_llm_applications/02_chat_interface/chat_memory.py
uv run python phase3_llm_applications/02_chat_interface/streaming.py
```

## Exercises

1. **Basic Chat**: Build a simple chat loop with memory
2. **Memory Comparison**: Compare sliding window vs summarization
3. **Streaming Chat**: Implement streaming with typing effect
4. **Context-Aware Chat**: Handle "it", "that", "the previous" correctly
5. **Token Counter**: Build a token budget manager

## Key Takeaways

1. **No magic memory** - LLM re-reads entire conversation each turn
2. **Context limits** - Must manage conversation length
3. **Streaming = UX** - Makes chat feel responsive
4. **References work** - Because full context is always sent
5. **System prompt persists** - Always first in messages[]

## Next Steps

After completing this module:
- → [Module 3: Function Calling](../03_function_calling/) (tool integration)
- → [Phase 4: AI Agents](../../phase4_ai_agents/) (autonomous systems)