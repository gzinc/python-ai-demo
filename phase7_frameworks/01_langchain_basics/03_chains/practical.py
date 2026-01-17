"""
Chains - Hands-On Practice (Requires API Keys)

Demonstrates real chain execution with OpenAI/Anthropic, LCEL patterns, error handling,
and debugging techniques with actual LLM calls.

Run: uv run python -m phase7_frameworks.01_langchain_basics.03_chains.practical
"""

from inspect import cleandoc

from langchain_core.messages import HumanMessage, SystemMessage

from phase7_frameworks.utils import (
    check_api_keys,
    print_section,
    requires_both_keys,
    requires_openai,
)


# region Demo 1: Basic LCEL Chain


@requires_openai
def demo_basic_lcel_chain() -> None:
    """
    demonstrate basic prompt → llm → parser chain

    Basic LCEL Chain Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │           LCEL: Prompt | LLM | Parser Pipeline              │
    │                                                             │
    │  Three Concepts Processed Through Same Chain:               │
    │                                                             │
    │  Input 1: {"concept": "embeddings"}                         │
    │      │                                                      │
    │      ▼                                                      │
    │  ┌──────────────────────┐                                   │
    │  │ ChatPromptTemplate   │ "Explain {concept} in one..."     │
    │  └──────────┬───────────┘                                   │
    │             │ Formatted: "Explain embeddings in one..."     │
    │             ▼                                               │
    │  ┌──────────────────────┐                                   │
    │  │ ChatOpenAI (GPT-3.5) │ Process prompt                    │
    │  └──────────┬───────────┘                                   │
    │             │ AIMessage("Embeddings are...")                │
    │             ▼                                               │
    │  ┌──────────────────────┐                                   │
    │  │ StrOutputParser      │ Extract .content                  │
    │  └──────────┬───────────┘                                   │
    │             │                                               │
    │             ▼                                               │
    │  Output: "Embeddings are numerical representations..."      │
    │                                                             │
    │  LCEL Syntax:                                               │
    │     chain = prompt | llm | parser                           │
    │                ↑      ↑      ↑                              │
    │                │      │      └─ Extract string              │
    │                │      └──────── Generate response           │
    │                └───────────── Format template               │
    │                                                             │
    │  Same chain processes: embeddings → RAG → LCEL             │
    │                                                             │
    │  ✅ Benefit: Pipe operator (|) chains components            │
    │  ✅ Benefit: Reusable pipeline for multiple inputs          │
    │  ✅ Benefit: Clean, readable composition                    │
    │  ✅ Benefit: Type-safe data flow                            │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Basic LCEL Chain (Prompt → LLM → Parser)")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # components
    prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    parser = StrOutputParser()

    # LCEL composition
    chain = prompt | llm | parser

    print("## Chain Structure: prompt | llm | parser\n")

    concepts = ["embeddings", "RAG", "LCEL"]
    for concept in concepts:
        result = chain.invoke({"concept": concept})
        print(f"{concept:12s}: {result}")

    print("\n✓ LCEL syntax is concise and readable")


# endregion

# region Demo 2: Multi-Message Prompt Chain


@requires_openai
def demo_multi_message_chain() -> None:
    """
    demonstrate chain with system + user messages

    Multi-Message Chain Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       Multi-Message Templates: System + User Pattern        │
    │                                                             │
    │  Input: {role: "Python expert", task: "Explain lists"}      │
    │      │                                                      │
    │      ▼                                                      │
    │  ┌───────────────────────────────────────┐                  │
    │  │ ChatPromptTemplate.from_messages([    │                  │
    │  │   ("system", "You are a {role}"),     │                  │
    │  │   ("human", "{task}")                 │                  │
    │  │ ])                                    │                  │
    │  └──────────────┬────────────────────────┘                  │
    │                 │                                           │
    │                 ▼                                           │
    │  Formatted Messages:                                        │
    │  [SystemMessage("You are a Python expert"),                 │
    │   HumanMessage("Explain list comprehensions")]              │
    │                 │                                           │
    │                 ▼                                           │
    │  ┌────────────────────────┐                                 │
    │  │ ChatOpenAI processes   │                                 │
    │  │ with system context    │                                 │
    │  └──────────┬─────────────┘                                 │
    │             │                                               │
    │             ▼                                               │
    │  Response tailored to role (Python expert voice)            │
    │                                                             │
    │  Role Switching Example:                                    │
    │  • role="Python expert" → Technical, code-focused           │
    │  • role="data scientist" → Statistical, data-focused        │
    │  • Same template, different personalities                   │
    │                                                             │
    │  ✅ Benefit: System message sets assistant personality      │
    │  ✅ Benefit: Reusable template for different roles          │
    │  ✅ Benefit: Consistent behavior across sessions            │
    │  ✅ Benefit: Separate system context from user input        │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: Multi-Message Prompt Chain")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # system + user message template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise {role}"),
        ("human", "{task}")
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    print("## Multi-Message Chain:\n")

    tasks = [
        {"role": "Python expert", "task": "Explain list comprehensions"},
        {"role": "data scientist", "task": "Explain embeddings"},
    ]

    for task_data in tasks:
        result = chain.invoke(task_data)
        print(f"Role: {task_data['role']}")
        print(f"Task: {task_data['task']}")
        print(f"Response: {result}\n")


# endregion

# region Demo 3: Streaming Chain


@requires_openai
def demo_streaming_chain() -> None:
    """
    demonstrate real-time streaming with LCEL

    Streaming Chain Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         Streaming: Progressive Token-by-Token Output        │
    │                                                             │
    │  Chain: prompt | llm (streaming=True) | parser              │
    │                                                             │
    │  for chunk in chain.stream(input):                          │
    │      │                                                      │
    │      ▼                                                      │
    │  Chunk 1: "RAG"           ← Immediate                       │
    │  Chunk 2: " is"           ← 50ms later                      │
    │  Chunk 3: " Retrieval"    ← 100ms later                     │
    │  Chunk 4: "-Augmented"    ← 150ms later                     │
    │  Chunk 5: " Generation"   ← 200ms later                     │
    │  ...                                                        │
    │                                                             │
    │  User Experience:                                           │
    │  ┌──────────────────────────────────┐                       │
    │  │ Time 0ms:   "R"                  │                       │
    │  │ Time 50ms:  "RA"                 │                       │
    │  │ Time 100ms: "RAG"                │                       │
    │  │ Time 150ms: "RAG is"             │                       │
    │  │ ...progressive display...        │                       │
    │  └──────────────────────────────────┘                       │
    │                                                             │
    │  vs. Non-Streaming:                                         │
    │  ┌──────────────────────────────────┐                       │
    │  │ Time 0-500ms: ⏳ waiting...      │                       │
    │  │ Time 500ms:   Full response      │                       │
    │  └──────────────────────────────────┘                       │
    │                                                             │
    │  ✅ Benefit: Better perceived performance                   │
    │  ✅ Benefit: User sees immediate progress                   │
    │  ✅ Benefit: Can interrupt long generations                 │
    │  ✅ Benefit: Critical for chat UI experiences               │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Streaming Chain Execution")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template(
        "Explain {concept} in 2-3 sentences"
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    print("## Streaming Response:\n")
    print("Concept: RAG")
    print("Response: ", end="", flush=True)

    for chunk in chain.stream({"concept": "RAG"}):
        print(chunk, end="", flush=True)

    print("\n\n✓ Streaming provides progressive output for better UX")


# endregion

# region Demo 4: Parallel Chain Execution


@requires_openai
def demo_parallel_chains() -> None:
    """
    demonstrate parallel chain execution with RunnableParallel

    Parallel Chain Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      RunnableParallel: Concurrent Independent Chains        │
    │                                                             │
    │  Input: {text: "AI is transforming software development"}   │
    │             │                                               │
    │      ┌──────┴──────┐                                        │
    │      │             │                                        │
    │      ▼             ▼                                        │
    │  ┌─────────┐   ┌─────────┐                                 │
    │  │Summary  │   │Keywords │  (Run simultaneously)            │
    │  │Chain    │   │Chain    │                                 │
    │  └────┬────┘   └────┬────┘                                 │
    │       │             │                                       │
    │       │ "AI transforms software"                           │
    │       │             │ "AI, software, transform"            │
    │       │             │                                       │
    │       └──────┬──────┘                                       │
    │              │                                              │
    │              ▼                                              │
    │  Output: {                                                  │
    │    "summary": "AI transforms software",                     │
    │    "keywords": "AI, software, transform"                    │
    │  }                                                          │
    │                                                             │
    │  Parallel Execution vs Sequential:                          │
    │                                                             │
    │  Sequential:  Summary → Wait → Keywords → Wait             │
    │  Total time:  500ms + 500ms = 1000ms                        │
    │                                                             │
    │  Parallel:    Summary ┐                                     │
    │               Keywords┘ (Both at once)                      │
    │  Total time:  max(500ms, 500ms) = 500ms                     │
    │                                                             │
    │  Implementation:                                            │
    │     RunnableParallel({                                      │
    │         "summary": summary_chain,                           │
    │         "keywords": keyword_chain                           │
    │     })                                                      │
    │                                                             │
    │  ✅ Benefit: 2x speedup (or more) for independent chains    │
    │  ✅ Benefit: Single input, multiple analyses                │
    │  ✅ Benefit: Structured output with named results           │
    │  ✅ Benefit: Optimal for RAG with multiple retrievers       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Parallel Chain Execution")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=50)
    parser = StrOutputParser()

    # define multiple chains
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this in 5 words: {text}"
    )
    keyword_prompt = ChatPromptTemplate.from_template(
        "List 3 keywords for: {text}"
    )

    summary_chain = summary_prompt | llm | parser
    keyword_chain = keyword_prompt | llm | parser

    # parallel execution
    parallel_chain = RunnableParallel({
        "summary": summary_chain,
        "keywords": keyword_chain
    })

    print("## Parallel Chains: summary + keywords\n")

    text = "Artificial intelligence is transforming software development"
    result = parallel_chain.invoke({"text": text})

    print(f"Input: {text}\n")
    print(f"Summary:  {result['summary']}")
    print(f"Keywords: {result['keywords']}")

    print("\n✓ Parallel execution runs independent chains simultaneously")


# endregion

# region Demo 5: Passthrough Pattern for RAG


@requires_openai
def demo_passthrough_pattern() -> None:
    """
    demonstrate passthrough pattern for RAG-like workflows

    Passthrough Pattern (RAG):
    ┌─────────────────────────────────────────────────────────────┐
    │       RunnablePassthrough: Preserve + Transform Pattern     │
    │                                                             │
    │  Question: "What are embeddings?"                           │
    │      │                                                      │
    │      ▼                                                      │
    │  ┌────────────────────────────────────┐                     │
    │  │ {                                  │                     │
    │  │   "context": Passthrough → Retriever                     │
    │  │   "question": Passthrough (unchanged)                    │
    │  │ }                                  │                     │
    │  └──────────┬─────────────────────────┘                     │
    │             │                                               │
    │      ┌──────┴──────┐                                        │
    │      │             │                                        │
    │      ▼             ▼                                        │
    │  context:      question:                                    │
    │  "Retrieved:   "What are embeddings?"                       │
    │   Embeddings   (original preserved)                         │
    │   are vector                                               │
    │   representations"                                          │
    │      │             │                                        │
    │      └──────┬──────┘                                        │
    │             │                                               │
    │             ▼                                               │
    │  ┌──────────────────────────────────┐                       │
    │  │ ChatPromptTemplate:              │                       │
    │  │ "Answer using context:           │                       │
    │  │  Context: {context}              │                       │
    │  │  Question: {question}"           │                       │
    │  └──────────┬───────────────────────┘                       │
    │             │                                               │
    │             ▼                                               │
    │  Formatted prompt with both values → LLM → Answer           │
    │                                                             │
    │  Why RunnablePassthrough?                                   │
    │  • Preserves original input unchanged                       │
    │  • Allows parallel transformations                          │
    │  • Critical for RAG: context (retrieved) + question (raw)   │
    │  • Enables complex data routing                             │
    │                                                             │
    │  ✅ Benefit: Preserve input alongside transformations       │
    │  ✅ Benefit: Foundation for RAG pipelines                   │
    │  ✅ Benefit: Clean data routing in chains                   │
    │  ✅ Benefit: Combine processed + original data              │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Passthrough Pattern (RAG Simulation)")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI

    # simulate retriever (normally would query vector DB)
    def mock_retriever(query: str) -> str:
        """simulate document retrieval"""
        return f"Retrieved context: Embeddings are vector representations used in AI"

    # RAG prompt expecting context + question
    prompt = ChatPromptTemplate.from_template(cleandoc('''
        Answer the question using the context below.

        Context: {context}
        Question: {question}

        Answer:
    '''))

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    parser = StrOutputParser()

    # chain with passthrough pattern
    chain = {
        "context": RunnablePassthrough() | mock_retriever,  # process
        "question": RunnablePassthrough()                   # pass through
    } | prompt | llm | parser

    print("## RAG Pattern: context (retrieved) + question (passthrough)\n")

    question = "What are embeddings?"
    result = chain.invoke(question)

    print(f"Question: {question}")
    print(f"Answer: {result}")

    print("\n✓ Passthrough preserves original input alongside processed data")


# endregion

# region Demo 6: Fallback Chain


@requires_both_keys
def demo_fallback_chain() -> None:
    """
    demonstrate fallback between primary and secondary models

    Fallback Chain Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         Fallback Chain: Primary → Secondary Resilience      │
    │                                                             │
    │  Normal Execution (Primary succeeds):                       │
    │     Input → GPT-4 (Primary)                                 │
    │                  │                                          │
    │                  ✓ Success                                  │
    │                  │                                          │
    │                  ▼                                          │
    │     Output: High-quality GPT-4 response                     │
    │                                                             │
    │  Fallback Execution (Primary fails):                        │
    │     Input → GPT-4 (Primary)                                 │
    │                  │                                          │
    │                  ✗ Timeout / Error / Rate limit             │
    │                  │                                          │
    │                  ▼                                          │
    │     Automatic switch to Claude Haiku (Fallback)             │
    │                  │                                          │
    │                  ✓ Success                                  │
    │                  │                                          │
    │                  ▼                                          │
    │     Output: Claude Haiku response (still gets answer!)      │
    │                                                             │
    │  Configuration:                                             │
    │     primary_chain.with_fallbacks([fallback_chain])          │
    │                                                             │
    │  Failure Scenarios Handled:                                 │
    │  • Network timeouts                                         │
    │  • API rate limits                                          │
    │  • Provider outages                                         │
    │  • Model unavailability                                     │
    │  • Token/billing issues                                     │
    │                                                             │
    │  Use Cases:                                                 │
    │  • Production reliability (99.9% uptime)                    │
    │  • Multi-provider redundancy                                │
    │  • Cost optimization (cheap fallback for failures)          │
    │  • Geographic failover                                      │
    │                                                             │
    │  ✅ Benefit: Zero downtime from single provider failures    │
    │  ✅ Benefit: Automatic failover (no manual intervention)    │
    │  ✅ Benefit: Cost-effective (fallback only when needed)     │
    │  ✅ Benefit: Production-grade reliability                   │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 6: Fallback Chain (Reliability Pattern)")

    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence")
    parser = StrOutputParser()

    # primary: GPT-4 (expensive, high quality)
    primary_llm = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=100, timeout=5)
    primary_chain = prompt | primary_llm | parser

    # fallback: Claude Haiku (cheaper, fast)
    fallback_llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0.7,
        max_tokens=100
    )
    fallback_chain = prompt | fallback_llm | parser

    # combine with fallback
    chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

    print("## Fallback Chain: GPT-4 → Claude (if GPT-4 fails)\n")

    concepts = ["embeddings", "RAG"]
    for concept in concepts:
        try:
            result = chain_with_fallback.invoke({"concept": concept})
            print(f"{concept}: {result}")
        except Exception as e:
            print(f"{concept}: Both chains failed - {e}")

    print("\n✓ Fallback chains improve reliability and handle provider outages")


# endregion

# region Demo 7: Retry Configuration


@requires_openai
def demo_retry_chain() -> None:
    """
    demonstrate retry configuration for transient failures

    Retry Configuration Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      Retry Chain: Exponential Backoff with Jitter           │
    │                                                             │
    │  Attempt 1 (Immediate):                                     │
    │     Request → LLM API                                       │
    │                │                                            │
    │                ✗ Transient failure (network timeout)        │
    │                │                                            │
    │                ▼                                            │
    │     Wait: 1 second + random jitter (0-500ms)                │
    │                │                                            │
    │                ▼                                            │
    │  Attempt 2:                                                 │
    │     Retry → LLM API                                         │
    │                │                                            │
    │                ✗ Still failing (rate limit)                 │
    │                │                                            │
    │                ▼                                            │
    │     Wait: 2 seconds + random jitter (0-1000ms)              │
    │                │                                            │
    │                ▼                                            │
    │  Attempt 3 (Final):                                         │
    │     Retry → LLM API                                         │
    │                │                                            │
    │                ✓ Success!                                   │
    │                │                                            │
    │                ▼                                            │
    │     Return response (user never knew about failures)        │
    │                                                             │
    │  Exponential Backoff Strategy:                              │
    │     Attempt 1: 0s wait (immediate)                          │
    │     Attempt 2: 1s + jitter                                  │
    │     Attempt 3: 2s + jitter                                  │
    │     Attempt 4: 4s + jitter (if configured)                  │
    │     Pattern: 2^(attempt-1) seconds                          │
    │                                                             │
    │  Why Add Random Jitter?                                     │
    │     Without: All clients retry at same time → Thundering    │
    │              herd problem (overload server again)           │
    │     With:    Clients retry at random intervals → Spread     │
    │              load, better success rate                      │
    │                                                             │
    │  Configuration:                                             │
    │     chain.with_retry(                                       │
    │         stop_after_attempt=3,                               │
    │         wait_exponential_jitter=True                        │
    │     )                                                       │
    │                                                             │
    │  Handles Transient Failures:                                │
    │  • Network timeouts (temporary)                             │
    │  • Rate limits (brief overload)                             │
    │  • Server 5xx errors (transient)                            │
    │  • Connection drops                                         │
    │                                                             │
    │  ✅ Benefit: Automatic recovery from transient errors       │
    │  ✅ Benefit: No manual retry logic needed                   │
    │  ✅ Benefit: Exponential backoff prevents server overload   │
    │  ✅ Benefit: Jitter prevents thundering herd                │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 7: Retry Configuration")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    parser = StrOutputParser()

    # basic chain
    basic_chain = prompt | llm | parser

    # chain with retry (up to 3 attempts, exponential backoff)
    retry_chain = (prompt | llm | parser).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )

    print("## Retry Chain Configuration:\n")
    print("Strategy: 3 attempts with exponential backoff + jitter")
    print("Exponential backoff with random jitter prevents thundering herd\n")

    print("Executing with retry protection:")
    result = retry_chain.invoke({"concept": "embeddings"})
    print(f"Result: {result}")

    print("\n✓ Retry configuration handles transient failures automatically")


# endregion

# region Demo 8: Batch Processing


@requires_openai
def demo_batch_processing() -> None:
    """
    demonstrate batch processing for efficiency

    Batch Processing Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       Batch Processing: Parallel Execution for Efficiency   │
    │                                                             │
    │  Sequential Execution (Traditional):                        │
    │     Input 1: "embeddings" → LLM → Result 1 (500ms)          │
    │                                      │                      │
    │                                      ▼                      │
    │     Input 2: "RAG"        → LLM → Result 2 (500ms)          │
    │                                      │                      │
    │                                      ▼                      │
    │     Input 3: "fine-tuning"→ LLM → Result 3 (500ms)          │
    │                                                             │
    │     Total time: 500ms + 500ms + 500ms = 1500ms              │
    │                                                             │
    │  Batch Execution (Parallel):                                │
    │                  ┌──────────────────┐                       │
    │     Input 1 ────→│                  │                       │
    │     Input 2 ────→│  Batch Request   │                       │
    │     Input 3 ────→│  (Parallel)      │                       │
    │                  └────────┬─────────┘                       │
    │                           │                                 │
    │                     ┌─────┴─────┐                           │
    │                     │           │                           │
    │              Result 1    Result 2    Result 3               │
    │              (500ms - all at once)                          │
    │                                                             │
    │     Total time: max(500ms, 500ms, 500ms) = 500ms            │
    │     Speedup: 1500ms / 500ms = 3x faster! ⚡                  │
    │                                                             │
    │  Implementation:                                            │
    │     # Sequential (slow)                                     │
    │     results = [chain.invoke(item) for item in items]        │
    │                                                             │
    │     # Batch (fast)                                          │
    │     results = chain.batch(items)                            │
    │                                                             │
    │  LangChain Optimization:                                    │
    │  • Batches multiple requests into single API call           │
    │  • Provider processes requests in parallel                  │
    │  • Results returned in same order as inputs                 │
    │  • Automatic concurrency management                         │
    │                                                             │
    │  When to Use Batch:                                         │
    │  • Processing multiple similar inputs                       │
    │  • Bulk data analysis (100s-1000s of items)                 │
    │  • Report generation (many sections)                        │
    │  • Dataset labeling/classification                          │
    │  • Translation of multiple texts                            │
    │                                                             │
    │  ✅ Benefit: 2-10x speedup (depends on batch size)          │
    │  ✅ Benefit: Lower API costs (fewer round trips)            │
    │  ✅ Benefit: Maintains order (predictable results)          │
    │  ✅ Benefit: Automatic error handling per item              │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 8: Batch Processing")

    import time

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=50)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    concepts = [
        {"concept": "embeddings"},
        {"concept": "RAG"},
        {"concept": "fine-tuning"}
    ]

    # sequential execution
    print("## Sequential vs Batch Execution:\n")
    print("Sequential (one at a time):")
    start = time.time()
    sequential_results = [chain.invoke(c) for c in concepts]
    sequential_time = time.time() - start

    # batch execution
    print(f"✓ Completed in {sequential_time:.2f}s\n")
    print("Batch (parallel):")
    start = time.time()
    batch_results = chain.batch(concepts)
    batch_time = time.time() - start
    print(f"✓ Completed in {batch_time:.2f}s\n")

    speedup = sequential_time / batch_time if batch_time > 0 else 1
    print(f"Speedup: {speedup:.1f}x faster with batch processing")

    print("\n## Results:\n")
    for i, (concept_dict, result) in enumerate(zip(concepts, batch_results), 1):
        print(f"{i}. {concept_dict['concept']:15s}: {result}")


# endregion

# region Demo 9: Debugging with Verbose Mode


@requires_openai
def demo_verbose_debugging() -> None:
    """
    demonstrate verbose mode for debugging

    Verbose Debugging Mode:
    ┌─────────────────────────────────────────────────────────────┐
    │         Verbose Mode: Chain Execution Transparency          │
    │                                                             │
    │  Normal Execution (verbose=False):                          │
    │     chain.invoke(input)                                     │
    │           │                                                 │
    │           ▼                                                 │
    │     Final result only                                       │
    │     (Black box - no visibility)                             │
    │                                                             │
    │  Verbose Execution (verbose=True):                          │
    │     chain.invoke(input, config={"verbose": True})           │
    │           │                                                 │
    │           ▼                                                 │
    │     ┌─────────────────────────────────────┐                 │
    │     │ Step 1: PromptTemplate              │                 │
    │     │   Input: {concept: "LCEL"}          │                 │
    │     │   Output: "Explain LCEL briefly"    │                 │
    │     └─────────┬───────────────────────────┘                 │
    │               │                                             │
    │               ▼                                             │
    │     ┌─────────────────────────────────────┐                 │
    │     │ Step 2: ChatOpenAI                  │                 │
    │     │   Model: gpt-3.5-turbo              │                 │
    │     │   Temperature: 0.7                  │                 │
    │     │   Tokens: 50 max                    │                 │
    │     │   Response: AIMessage(...)          │                 │
    │     └─────────┬───────────────────────────┘                 │
    │               │                                             │
    │               ▼                                             │
    │     ┌─────────────────────────────────────┐                 │
    │     │ Step 3: StrOutputParser             │                 │
    │     │   Input: AIMessage                  │                 │
    │     │   Output: "LCEL is..."              │                 │
    │     └─────────┬───────────────────────────┘                 │
    │               │                                             │
    │               ▼                                             │
    │     Final result with full execution trace                  │
    │                                                             │
    │  Debugging Information Shown:                               │
    │  • Each chain step execution                                │
    │  • Input/output at each stage                               │
    │  • Component types being executed                           │
    │  • Timing information                                       │
    │  • Token usage (if available)                               │
    │  • Error locations (when failures occur)                    │
    │                                                             │
    │  Use Cases:                                                 │
    │  • Chain not working as expected                            │
    │  • Need to understand data transformation                   │
    │  • Debugging complex multi-step chains                      │
    │  • Learning how LCEL works internally                       │
    │  • Performance optimization                                 │
    │  • Validation of chain structure                            │
    │                                                             │
    │  ✅ Benefit: See exactly what each chain step does          │
    │  ✅ Benefit: Identify where failures occur                  │
    │  ✅ Benefit: Understand data transformations                │
    │  ✅ Benefit: Learning tool for LCEL internals               │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 9: Debugging with Verbose Mode")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("Explain {concept} briefly")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=50)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    print("## Verbose Mode (shows execution details):\n")

    result = chain.invoke(
        {"concept": "LCEL"},
        config={"verbose": True}
    )

    print(f"\nFinal Result: {result}")

    print("\n✓ Verbose mode helps debug chain execution flow")


# endregion

# region Demo 10: Custom Transformation


@requires_openai
def demo_custom_transformation() -> None:
    """
    demonstrate custom transformation with RunnableLambda

    Custom Transformation Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      RunnableLambda: Custom Transformations in Chains       │
    │                                                             │
    │  Standard Chain (No Custom Logic):                          │
    │     Prompt → LLM → StrOutputParser                          │
    │        │      │          │                                  │
    │        ▼      ▼          ▼                                  │
    │     Template  Model    String output                        │
    │                                                             │
    │  Chain with RunnableLambda:                                 │
    │     Prompt → LLM → StrOutputParser → RunnableLambda         │
    │        │      │          │                  │               │
    │        ▼      ▼          ▼                  ▼               │
    │     Template  Model    "1. AI\n         Custom function     │
    │                        2. ML\n         transforms output    │
    │                        3. NLP"                              │
    │                                           │                 │
    │                                           ▼                 │
    │                                    "📋 SUMMARY:             │
    │                                     1. AI                   │
    │                                     2. ML                   │
    │                                     3. NLP"                 │
    │                                                             │
    │  Custom Transformation Function:                            │
    │     def custom_transform(text: str) -> str:                 │
    │         # Add formatting                                    │
    │         return f"📋 SUMMARY:\n{text.upper()}"               │
    │                                                             │
    │  Integration:                                               │
    │     chain = (                                               │
    │         prompt                                              │
    │         | llm                                               │
    │         | parser                                            │
    │         | RunnableLambda(custom_transform)                  │
    │     )                                                       │
    │                                                             │
    │  What RunnableLambda Can Do:                                │
    │  • Format output (add headers, styles, emojis)              │
    │  • Parse and restructure data                               │
    │  • Filter/validate results                                  │
    │  • Call external APIs or databases                          │
    │  • Apply business logic                                     │
    │  • Transform between formats (JSON → CSV)                   │
    │  • Clean/sanitize output                                    │
    │  • Combine multiple results                                 │
    │                                                             │
    │  Use Cases:                                                 │
    │  • Format LLM output for specific UI needs                  │
    │  • Add metadata or timestamps                               │
    │  • Integrate with external systems                          │
    │  • Apply domain-specific transformations                    │
    │  • Validate and clean LLM responses                         │
    │                                                             │
    │  ✅ Benefit: Insert any Python function into chains         │
    │  ✅ Benefit: Full flexibility for custom logic              │
    │  ✅ Benefit: Composable with other LCEL components          │
    │  ✅ Benefit: Maintains streaming and async support          │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 10: Custom Transformation in Chain")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("List 3 {items}")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    parser = StrOutputParser()

    # custom transformation: convert to uppercase and add prefix
    def custom_transform(text: str) -> str:
        """add formatting to output"""
        return f"📋 SUMMARY:\n{text.upper()}"

    # chain with custom transformation
    chain = prompt | llm | parser | RunnableLambda(custom_transform)

    print("## Chain with Custom Transformation:\n")

    result = chain.invoke({"items": "AI concepts"})
    print(result)

    print("\n✓ RunnableLambda enables custom transformations in chains")


# endregion


def main() -> None:
    """run all practical demos"""
    print(cleandoc('''
        Chains - Practical Demos

        This module demonstrates real chain execution with LLM calls.
        Ensure you have API keys set in .env file.
    '''))

    has_openai, has_anthropic = check_api_keys()

    print("\n## API Key Status:")
    print(f"OPENAI_API_KEY: {'✓ Found' if has_openai else '✗ Missing'}")
    print(f"ANTHROPIC_API_KEY: {'✓ Found' if has_anthropic else '✗ Missing'}")

    if not (has_openai or has_anthropic):
        print("\n❌ No API keys found!")
        print("Set at least one API key in .env to run demos:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  ANTHROPIC_API_KEY=your-key-here")
        return

    demo_basic_lcel_chain()
    demo_multi_message_chain()
    demo_streaming_chain()
    demo_parallel_chains()
    demo_passthrough_pattern()
    demo_fallback_chain()
    demo_retry_chain()
    demo_batch_processing()
    demo_verbose_debugging()
    demo_custom_transformation()

    print("\n" + "=" * 70)
    print("  Practical demos complete! You've mastered chain composition.")
    print("=" * 70)


if __name__ == "__main__":
    main()
