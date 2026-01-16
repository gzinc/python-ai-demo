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
    """demonstrate basic prompt → llm → parser chain"""
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
    """demonstrate chain with system + user messages"""
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
    """demonstrate real-time streaming with LCEL"""
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
    """demonstrate parallel chain execution with RunnableParallel"""
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
    """demonstrate passthrough pattern for RAG-like workflows"""
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
    """demonstrate fallback between primary and secondary models"""
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
    """demonstrate retry configuration for transient failures"""
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
    """demonstrate batch processing for efficiency"""
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
    """demonstrate verbose mode for debugging"""
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
    """demonstrate custom transformation with RunnableLambda"""
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
