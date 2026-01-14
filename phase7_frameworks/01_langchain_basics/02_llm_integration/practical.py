"""
LLM Integration - Hands-On Practice (Requires API Keys)

Demonstrates real LLM integration with OpenAI and Anthropic using LangChain's
unified interface. Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env

Run: uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.practical
"""

import os
from inspect import cleandoc

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def check_api_keys() -> tuple[bool, bool]:
    """check which API keys are available"""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    return bool(openai_key), bool(anthropic_key)


# region Demo 1: ChatOpenAI Basic Usage


def demo_chatopenai_basic() -> None:
    """demonstrate basic ChatOpenAI usage"""
    print_section("Demo 1: ChatOpenAI Basic Usage")

    has_openai, _ = check_api_keys()
    if not has_openai:
        print("⚠️  OPENAI_API_KEY not found - skipping demo")
        print("Set OPENAI_API_KEY in .env to run this demo")
        return

    from langchain_openai import ChatOpenAI

    # Initialize chat model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # using cheaper model for demo
        temperature=0.7,
        max_tokens=100,
    )

    print("## Model Configuration:")
    print(f"Model: {llm.model_name}")
    print(f"Temperature: {llm.temperature}")
    print(f"Max Tokens: {llm.max_tokens}")

    # Single message
    print("\n## Single Message:")
    messages = [HumanMessage(content="What are embeddings in one sentence?")]
    response = llm.invoke(messages)
    print(f"Response: {response.content}")

    # Multi-turn conversation
    print("\n## Multi-Turn Conversation:")
    conversation = [
        SystemMessage(content="You are a concise Python expert"),
        HumanMessage(content="How do I read a file?"),
    ]
    response = llm.invoke(conversation)
    print(f"Response: {response.content[:200]}...")


# endregion

# region Demo 2: ChatAnthropic Basic Usage


def demo_chatanthropic_basic() -> None:
    """demonstrate basic ChatAnthropic usage"""
    print_section("Demo 2: ChatAnthropic Basic Usage")

    _, has_anthropic = check_api_keys()
    if not has_anthropic:
        print("⚠️  ANTHROPIC_API_KEY not found - skipping demo")
        print("Set ANTHROPIC_API_KEY in .env to run this demo")
        return

    from langchain_anthropic import ChatAnthropic

    # Initialize chat model
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",  # using cheaper model for demo
        temperature=0.7,
        max_tokens=100,
    )

    print("## Model Configuration:")
    print(f"Model: {llm.model}")
    print(f"Temperature: {llm.temperature}")
    print(f"Max Tokens: {llm.max_tokens}")

    # Single message
    print("\n## Single Message:")
    messages = [HumanMessage(content="What are embeddings in one sentence?")]
    response = llm.invoke(messages)
    print(f"Response: {response.content}")

    # Multi-turn conversation
    print("\n## Multi-Turn Conversation:")
    conversation = [
        SystemMessage(content="You are a concise Python expert"),
        HumanMessage(content="How do I read a file?"),
    ]
    response = llm.invoke(conversation)
    print(f"Response: {response.content[:200]}...")


# endregion

# region Demo 3: Temperature and Creativity Control


def demo_temperature_control() -> None:
    """demonstrate how temperature affects output"""
    print_section("Demo 3: Temperature and Creativity Control")

    has_openai, _ = check_api_keys()
    if not has_openai:
        print("⚠️  OPENAI_API_KEY not found - skipping demo")
        return

    from langchain_openai import ChatOpenAI

    prompt = "Write a two-sentence product description for AI-powered headphones"
    messages = [HumanMessage(content=prompt)]

    print(f"Prompt: {prompt}\n")

    temperatures = [0.0, 0.5, 1.0]
    for temp in temperatures:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temp, max_tokens=100)
        response = llm.invoke(messages)

        print(f"Temperature {temp}:")
        print(f"→ {response.content}\n")

    print("Notice how higher temperature produces more varied/creative responses")


# endregion

# region Demo 4: Streaming Responses


def demo_streaming() -> None:
    """demonstrate real-time streaming responses"""
    print_section("Demo 4: Streaming Responses")

    has_openai, _ = check_api_keys()
    if not has_openai:
        print("⚠️  OPENAI_API_KEY not found - skipping demo")
        return

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Explain what RAG is in 2-3 sentences"),
    ]

    print("## Streaming Response (tokens arrive in real-time):\n")
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")

    print("✓ Streaming provides better UX for interactive applications")


# endregion

# region Demo 5: Provider Switching


def demo_provider_switching() -> None:
    """demonstrate switching between providers seamlessly"""
    print_section("Demo 5: Provider Switching")

    has_openai, has_anthropic = check_api_keys()

    if not (has_openai and has_anthropic):
        print("⚠️  Both API keys needed for this demo")
        print("Set OPENAI_API_KEY and ANTHROPIC_API_KEY in .env")
        return

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # Same prompt for both providers
    messages = [
        SystemMessage(content="You are a concise AI expert"),
        HumanMessage(content="What are embeddings in one sentence?"),
    ]

    # OpenAI
    print("## OpenAI GPT-3.5:")
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    openai_response = openai_llm.invoke(messages)
    print(f"→ {openai_response.content}")

    # Anthropic
    print("\n## Anthropic Claude:")
    anthropic_llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022", temperature=0.7, max_tokens=100
    )
    anthropic_response = anthropic_llm.invoke(messages)
    print(f"→ {anthropic_response.content}")

    print("\n✓ Same interface, different providers - seamless switching!")


# endregion

# region Demo 6: Fallback Chains


def demo_fallback_chain() -> None:
    """demonstrate fallback from primary to secondary provider"""
    print_section("Demo 6: Fallback Chains")

    has_openai, has_anthropic = check_api_keys()

    if not (has_openai and has_anthropic):
        print("⚠️  Both API keys needed for this demo")
        print("Set OPENAI_API_KEY and ANTHROPIC_API_KEY in .env")
        return

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # Primary: GPT-4 (expensive, high quality)
    primary = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=100, timeout=5)

    # Fallback: Claude (different provider)
    fallback = ChatAnthropic(
        model="claude-3-5-haiku-20241022", temperature=0.7, max_tokens=100
    )

    # Chain with fallback
    llm_with_fallback = primary.with_fallbacks([fallback])

    print("## Fallback Chain: GPT-4 → Claude")
    print("Primary: GPT-4 (if it fails...)")
    print("Fallback: Claude-3.5-Haiku\n")

    messages = [HumanMessage(content="What are embeddings?")]

    try:
        response = llm_with_fallback.invoke(messages)
        print(f"Response: {response.content[:200]}...")
        print("\n✓ Request succeeded (primary or fallback)")
    except Exception as e:
        print(f"❌ Both providers failed: {e}")


# endregion

# region Demo 7: Token Usage Tracking


def demo_token_tracking() -> None:
    """demonstrate tracking token usage and costs"""
    print_section("Demo 7: Token Usage Tracking")

    has_openai, _ = check_api_keys()
    if not has_openai:
        print("⚠️  OPENAI_API_KEY not found - skipping demo")
        return

    from langchain.callbacks import get_openai_callback
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Explain embeddings in simple terms"),
    ]

    print("## Tracking Token Usage with Callbacks:\n")

    with get_openai_callback() as cb:
        response = llm.invoke(messages)

        print(f"Response: {response.content[:100]}...\n")

        print("Token Usage:")
        print(f"  Prompt tokens: {cb.prompt_tokens}")
        print(f"  Completion tokens: {cb.completion_tokens}")
        print(f"  Total tokens: {cb.total_tokens}")
        print(f"  Total cost: ${cb.total_cost:.6f}")

        # Calculate cost breakdown
        print("\nCost Breakdown (GPT-3.5-turbo pricing):")
        print(f"  Input cost:  ${cb.prompt_tokens / 1_000_000 * 0.50:.6f}")
        print(f"  Output cost: ${cb.completion_tokens / 1_000_000 * 1.50:.6f}")


# endregion

# region Demo 8: LCEL Integration


def demo_lcel_integration() -> None:
    """demonstrate LangChain Expression Language integration"""
    print_section("Demo 8: LCEL Integration with LLMs")

    has_openai, _ = check_api_keys()
    if not has_openai:
        print("⚠️  OPENAI_API_KEY not found - skipping demo")
        return

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Create a chain: prompt | llm | parser
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise AI expert"),
            ("human", "Explain {concept} in one sentence"),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    parser = StrOutputParser()

    # LCEL chain composition
    chain = template | llm | parser

    print("## LCEL Chain: Prompt | LLM | Parser\n")

    concepts = ["embeddings", "RAG", "fine-tuning"]
    for concept in concepts:
        result = chain.invoke({"concept": concept})
        print(f"{concept:15s}: {result}")

    print("\n✓ LCEL makes it easy to compose LLMs with prompts and parsers")


# endregion


def main() -> None:
    """run all practical demos"""
    print(cleandoc('''
        LLM Integration - Practical Demos

        This module demonstrates real API calls to OpenAI and Anthropic.
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

    demo_chatopenai_basic()
    demo_chatanthropic_basic()
    demo_temperature_control()
    demo_streaming()
    demo_provider_switching()
    demo_fallback_chain()
    demo_token_tracking()
    demo_lcel_integration()

    print("\n" + "=" * 70)
    print("  Practical demos complete! You've mastered LLM integration.")
    print("=" * 70)


if __name__ == "__main__":
    main()
