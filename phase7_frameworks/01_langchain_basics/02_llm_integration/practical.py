"""
LLM Integration - Hands-On Practice (Requires API Keys)

Demonstrates real LLM integration with OpenAI and Anthropic using LangChain's
unified interface. Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env

Run: uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.practical
"""

from inspect import cleandoc

from langchain_core.messages import HumanMessage, SystemMessage

from phase7_frameworks.utils import (
    check_api_keys,
    print_section,
    requires_anthropic,
    requires_both_keys,
    requires_openai,
)


# region Demo 1: ChatOpenAI Basic Usage


def demo_chatopenai_basic() -> None:
    """
    demonstrate basic ChatOpenAI usage

    ChatOpenAI Integration Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         ChatOpenAI: Unified OpenAI API Integration          │
    │                                                             │
    │  1. Model Initialization:                                   │
    │     ┌──────────────────────────────────────┐                │
    │     │ ChatOpenAI(                          │                │
    │     │   model="gpt-3.5-turbo",             │                │
    │     │   temperature=0.7,                   │                │
    │     │   max_tokens=100                     │                │
    │     │ )                                    │                │
    │     └──────────────┬───────────────────────┘                │
    │                    │                                        │
    │                    ▼                                        │
    │  2. Single Message Flow:                                    │
    │     [HumanMessage("What are embeddings?")]                  │
    │                    │                                        │
    │                    ▼                                        │
    │     llm.invoke(messages)                                    │
    │                    │                                        │
    │                    ▼                                        │
    │     AIMessage(content="Embeddings are...")                  │
    │                                                             │
    │  3. Multi-Turn Conversation:                                │
    │     ┌────────────────────────────────────┐                  │
    │     │ [SystemMessage("You are..."),      │                  │
    │     │  HumanMessage("How do I...?")]     │                  │
    │     └──────────────┬─────────────────────┘                  │
    │                    │                                        │
    │                    ▼                                        │
    │     llm.invoke(conversation)                                │
    │                    │                                        │
    │                    ▼                                        │
    │     AIMessage with contextual response                      │
    │                                                             │
    │  ✅ Benefit: Unified interface across OpenAI models         │
    │  ✅ Benefit: Simple configuration (model, temp, tokens)     │
    │  ✅ Benefit: Automatic message formatting                   │
    │  ✅ Benefit: Built-in retry logic and error handling        │
    └─────────────────────────────────────────────────────────────┘
    """
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
    """
    demonstrate basic ChatAnthropic usage

    ChatAnthropic Integration Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       ChatAnthropic: Unified Anthropic API Integration      │
    │                                                             │
    │  1. Model Initialization:                                   │
    │     ┌──────────────────────────────────────┐                │
    │     │ ChatAnthropic(                       │                │
    │     │   model="claude-3-5-haiku-20241022", │                │
    │     │   temperature=0.7,                   │                │
    │     │   max_tokens=100                     │                │
    │     │ )                                    │                │
    │     └──────────────┬───────────────────────┘                │
    │                    │                                        │
    │                    ▼                                        │
    │  2. Message Flow (Same as ChatOpenAI):                      │
    │     [HumanMessage("What are embeddings?")]                  │
    │                    │                                        │
    │                    ▼                                        │
    │     llm.invoke(messages)                                    │
    │                    │                                        │
    │                    ▼                                        │
    │     AIMessage(content="Embeddings are...")                  │
    │                                                             │
    │  Provider Interchangeability:                               │
    │  ┌────────────────┐         ┌────────────────┐              │
    │  │  ChatOpenAI    │         │ ChatAnthropic  │              │
    │  │  (GPT models)  │   ←→    │ (Claude models)│              │
    │  └────────────────┘         └────────────────┘              │
    │         ↓                            ↓                      │
    │    Same .invoke() interface                                 │
    │    Same message types                                       │
    │    Same response format                                     │
    │                                                             │
    │  ✅ Benefit: Identical interface to ChatOpenAI              │
    │  ✅ Benefit: Seamless provider switching                    │
    │  ✅ Benefit: No code changes needed for migration           │
    │  ✅ Benefit: Multi-provider fallback support                │
    └─────────────────────────────────────────────────────────────┘
    """
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


@requires_openai
def demo_temperature_control() -> None:
    """
    demonstrate how temperature affects output

    Temperature Control Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       Temperature: Controlling Output Randomness            │
    │                                                             │
    │  Same Prompt, Different Temperatures:                       │
    │                                                             │
    │  Temperature 0.0 (Deterministic):                           │
    │     ┌──────────────────────────────────────┐                │
    │     │ "AI-powered headphones use advanced  │                │
    │     │  noise cancellation. They adapt to   │                │
    │     │  your listening environment."        │                │
    │     └──────────────────────────────────────┘                │
    │     • Most likely tokens chosen every time                  │
    │     • Consistent, repeatable responses                      │
    │     • Best for: factual tasks, data extraction              │
    │                                                             │
    │  Temperature 0.5 (Balanced):                                │
    │     ┌──────────────────────────────────────┐                │
    │     │ "These AI headphones deliver smart   │                │
    │     │  noise cancellation with adaptive    │                │
    │     │  audio optimization."                │                │
    │     └──────────────────────────────────────┘                │
    │     • Moderate randomness in token selection                │
    │     • More variety, still coherent                          │
    │     • Best for: general chat, recommendations               │
    │                                                             │
    │  Temperature 1.0 (Creative):                                │
    │     ┌──────────────────────────────────────┐                │
    │     │ "Experience revolutionary AI audio   │                │
    │     │  that learns your preferences and    │                │
    │     │  transforms your listening journey." │                │
    │     └──────────────────────────────────────┘                │
    │     • High randomness in token selection                    │
    │     • Most creative and varied                              │
    │     • Best for: creative writing, brainstorming             │
    │                                                             │
    │  Temperature Scale:                                         │
    │  0.0 ────────── 0.5 ────────── 1.0 ────────── 2.0           │
    │  Deterministic   Balanced       Creative      Chaotic       │
    │                                                             │
    │  ✅ Benefit: Fine-tune creativity vs consistency            │
    │  ✅ Benefit: Task-specific optimization                     │
    │  ✅ Benefit: Control output variability                     │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Temperature and Creativity Control")


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


@requires_openai
def demo_streaming() -> None:
    """
    demonstrate real-time streaming responses

    Streaming Response Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │         Streaming: Real-Time Token-by-Token Delivery        │
    │                                                             │
    │  Non-Streaming (llm.invoke()):                              │
    │     User sends request                                      │
    │            │                                                │
    │            ▼                                                │
    │     ⏳ Wait for full response...                            │
    │            │                                                │
    │            ▼                                                │
    │     Complete response delivered at once                     │
    │     "RAG is Retrieval-Augmented Generation..."              │
    │                                                             │
    │  Streaming (llm.stream()):                                  │
    │     User sends request                                      │
    │            │                                                │
    │            ▼                                                │
    │     Token 1: "RAG"          ← Immediate                     │
    │     Token 2: " is"          ← 50ms                          │
    │     Token 3: " Retrieval"   ← 100ms                         │
    │     Token 4: "-Augmented"   ← 150ms                         │
    │     Token 5: " Generation"  ← 200ms                         │
    │     ...                                                     │
    │                                                             │
    │  Implementation:                                            │
    │     ┌──────────────────────────────────────┐                │
    │     │ for chunk in llm.stream(messages):   │                │
    │     │     print(chunk.content, end="")     │                │
    │     └──────────────────────────────────────┘                │
    │                                                             │
    │  User Experience Comparison:                                │
    │  Non-Streaming:  ⏳⏳⏳⏳ → Full response                  │
    │  Streaming:      R → RA → RAG → RAG is → ...                │
    │                  ↑ Feels faster, more interactive           │
    │                                                             │
    │  ✅ Benefit: Better perceived performance                   │
    │  ✅ Benefit: User sees progress immediately                 │
    │  ✅ Benefit: Can stop generation early                      │
    │  ✅ Benefit: Improved UX for long responses                 │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Streaming Responses")


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
    """
    demonstrate switching between providers seamlessly

    Provider Switching Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      Seamless Provider Switching: Same Interface            │
    │                                                             │
    │  Unified Message Interface:                                 │
    │     ┌────────────────────────────────────┐                  │
    │     │ messages = [                       │                  │
    │     │   SystemMessage("You are..."),     │                  │
    │     │   HumanMessage("What are...?")     │                  │
    │     │ ]                                  │                  │
    │     └──────────────┬─────────────────────┘                  │
    │                    │                                        │
    │         ┌──────────┴──────────┐                             │
    │         │                     │                             │
    │         ▼                     ▼                             │
    │  ┌────────────┐        ┌────────────┐                       │
    │  │  OpenAI    │        │ Anthropic  │                       │
    │  │  GPT-3.5   │        │ Claude 3.5 │                       │
    │  └─────┬──────┘        └─────┬──────┘                       │
    │        │                     │                              │
    │        ▼                     ▼                              │
    │  openai_llm.invoke()   anthropic_llm.invoke()               │
    │        │                     │                              │
    │        ▼                     ▼                              │
    │  AIMessage(...)         AIMessage(...)                      │
    │                                                             │
    │  Provider Comparison:                                       │
    │  ┌─────────────┬──────────────┬──────────────┐              │
    │  │ Feature     │ OpenAI       │ Anthropic    │              │
    │  ├─────────────┼──────────────┼──────────────┤              │
    │  │ Interface   │ ChatOpenAI   │ ChatAnthropic│              │
    │  │ invoke()    │ ✓ Same       │ ✓ Same       │              │
    │  │ stream()    │ ✓ Same       │ ✓ Same       │              │
    │  │ Messages    │ ✓ Same types │ ✓ Same types │              │
    │  └─────────────┴──────────────┴──────────────┘              │
    │                                                             │
    │  Migration Example:                                         │
    │     # Before: OpenAI                                        │
    │     llm = ChatOpenAI(model="gpt-3.5-turbo")                 │
    │                                                             │
    │     # After: Anthropic (only 1 line changes!)               │
    │     llm = ChatAnthropic(model="claude-3-5-haiku-20241022")  │
    │                                                             │
    │  ✅ Benefit: Zero code changes for provider switch          │
    │  ✅ Benefit: A/B test different providers easily            │
    │  ✅ Benefit: Multi-provider applications                    │
    │  ✅ Benefit: Vendor independence                            │
    └─────────────────────────────────────────────────────────────┘
    """
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
    """
    demonstrate fallback from primary to secondary provider

    Fallback Chain Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │        Fallback Chains: Resilient Multi-Provider Setup      │
    │                                                             │
    │  Request Flow with Fallback:                                │
    │                                                             │
    │     User Request                                            │
    │          │                                                  │
    │          ▼                                                  │
    │     ┌─────────────────┐                                     │
    │     │ Primary: GPT-4  │ ← Try first (high quality)          │
    │     │ timeout=5s      │                                     │
    │     └────────┬────────┘                                     │
    │              │                                              │
    │    ┌─────────┴─────────┐                                    │
    │    │                   │                                    │
    │    ▼                   ▼                                    │
    │ Success?           Timeout/Error?                           │
    │    │                   │                                    │
    │    │                   ▼                                    │
    │    │          ┌──────────────────┐                          │
    │    │          │ Fallback: Claude │ ← Different provider     │
    │    │          │ 3.5 Haiku        │                          │
    │    │          └────────┬─────────┘                          │
    │    │                   │                                    │
    │    │          ┌────────┴────────┐                           │
    │    │          │                 │                           │
    │    │          ▼                 ▼                           │
    │    │       Success?         Error?                          │
    │    │          │                 │                           │
    │    └──────────┴─────────────────┴─→ Return response/error   │
    │                                                             │
    │  Implementation:                                            │
    │     ┌──────────────────────────────────────┐                │
    │     │ primary = ChatOpenAI(...)            │                │
    │     │ fallback = ChatAnthropic(...)        │                │
    │     │                                      │                │
    │     │ llm = primary.with_fallbacks(        │                │
    │     │     [fallback]                       │                │
    │     │ )                                    │                │
    │     └──────────────────────────────────────┘                │
    │                                                             │
    │  Fallback Scenarios:                                        │
    │  • Primary timeout → Try fallback                           │
    │  • API rate limit → Switch provider                         │
    │  • Model unavailable → Use alternative                      │
    │  • Cost optimization → Cheap primary, quality fallback      │
    │                                                             │
    │  ✅ Benefit: Increased reliability (99.9%+ uptime)          │
    │  ✅ Benefit: Vendor independence and resilience             │
    │  ✅ Benefit: Cost optimization strategies                   │
    │  ✅ Benefit: Graceful degradation                           │
    └─────────────────────────────────────────────────────────────┘
    """
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


@requires_openai
def demo_token_tracking() -> None:
    """
    demonstrate tracking token usage and costs

    Token Usage Tracking Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       Token Tracking: Cost Monitoring and Optimization      │
    │                                                             │
    │  Request with Callback Tracking:                            │
    │                                                             │
    │     with get_openai_callback() as cb:                       │
    │          │                                                  │
    │          ▼                                                  │
    │     ┌─────────────────────────────┐                         │
    │     │ User Prompt (Input)         │                         │
    │     │ "Explain embeddings..."     │                         │
    │     └──────────┬──────────────────┘                         │
    │                │                                            │
    │                ▼                                            │
    │     Tokenization: 15 tokens                                 │
    │                │                                            │
    │                ▼                                            │
    │     ┌─────────────────────────────┐                         │
    │     │ LLM Processing              │                         │
    │     │ (GPT-3.5-turbo)             │                         │
    │     └──────────┬──────────────────┘                         │
    │                │                                            │
    │                ▼                                            │
    │     ┌───────────────────────────────┐                       │
    │     │ AI Response (Output)          │                       │
    │     │ "Embeddings are numerical..." │                       │
    │     └──────────┬────────────────────┘                       │
    │                │                                            │
    │                ▼                                            │
    │     Tokenization: 85 tokens                                 │
    │                │                                            │
    │                ▼                                            │
    │     Callback Records:                                       │
    │     ┌───────────────────────────────────┐                   │
    │     │ Prompt tokens:     15             │                   │
    │     │ Completion tokens: 85             │                   │
    │     │ Total tokens:      100            │                   │
    │     │                                   │                   │
    │     │ Cost Calculation:                 │                   │
    │     │ Input:  15 × $0.50/1M = $0.000008 │                   │
    │     │ Output: 85 × $1.50/1M = $0.000128 │                   │
    │     │ Total:                  $0.000136 │                   │
    │     └───────────────────────────────────┘                   │
    │                                                             │
    │  Cost Management:                                           │
    │  • Track per-request costs                                  │
    │  • Set budget alerts ($X per day/month)                     │
    │  • Compare provider costs (GPT-3.5 vs GPT-4 vs Claude)      │
    │  • Optimize prompt length (fewer input tokens)              │
    │  • Cache common responses                                   │
    │                                                             │
    │  ✅ Benefit: Real-time cost monitoring                      │
    │  ✅ Benefit: Budget control and alerts                      │
    │  ✅ Benefit: Provider cost comparison                       │
    │  ✅ Benefit: Optimization insights                          │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 7: Token Usage Tracking")


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


@requires_openai
def demo_lcel_integration() -> None:
    """
    demonstrate LangChain Expression Language integration

    LCEL Integration Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       LCEL: Composable LLM Chains with Pipe Operator        │
    │                                                             │
    │  Chain Composition: Prompt | LLM | Parser                   │
    │                                                             │
    │     Input: {"concept": "embeddings"}                        │
    │          │                                                  │
    │          ▼                                                  │
    │     ┌─────────────────────────────┐                         │
    │     │ ChatPromptTemplate          │                         │
    │     │ "Explain {concept} in       │                         │
    │     │  one sentence"              │                         │
    │     └──────────┬──────────────────┘                         │
    │                │                                            │
    │                │ Formatted prompt                           │
    │                ▼                                            │
    │     ┌─────────────────────────────┐                         │
    │     │ ChatOpenAI                  │                         │
    │     │ (gpt-3.5-turbo)             │                         │
    │     └──────────┬──────────────────┘                         │
    │                │                                            │
    │                │ AIMessage                                  │
    │                ▼                                            │
    │     ┌─────────────────────────────┐                         │
    │     │ StrOutputParser             │                         │
    │     │ Extract .content            │                         │
    │     └──────────┬──────────────────┘                         │
    │                │                                            │
    │                ▼                                            │
    │     Output: "Embeddings are numerical..."                   │
    │                                                             │
    │  LCEL Syntax:                                               │
    │     ┌──────────────────────────────────────┐                │
    │     │ chain = (                            │                │
    │     │     template                         │                │
    │     │     | llm                            │                │
    │     │     | parser                         │                │
    │     │ )                                    │                │
    │     │                                      │                │
    │     │ result = chain.invoke(input)         │                │
    │     └──────────────────────────────────────┘                │
    │                                                             │
    │  Multiple Concepts in Batch:                                │
    │     concepts = ["embeddings", "RAG", "fine-tuning"]         │
    │          │                                                  │
    │          ▼                                                  │
    │     Same chain processes all three                          │
    │          │                                                  │
    │          ▼                                                  │
    │     Three concise explanations                              │
    │                                                             │
    │  ✅ Benefit: Clean, readable composition                    │
    │  ✅ Benefit: Reusable chain components                      │
    │  ✅ Benefit: Type-safe data flow                            │
    │  ✅ Benefit: Easy to extend and modify                      │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 8: LCEL Integration with LLMs")


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


def show_menu(has_openai: bool, has_anthropic: bool) -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  LLM Integration - Practical Examples")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "ChatOpenAI Basic Usage", "simple OpenAI integration", 'openai'),
        ("2", "ChatAnthropic Basic Usage", "simple Anthropic integration", 'anthropic'),
        ("3", "Temperature and Creativity Control", "output randomness tuning", 'openai'),
        ("4", "Streaming Responses", "real-time token delivery", 'openai'),
        ("5", "Provider Switching", "seamless provider swapping", 'both'),
        ("6", "Fallback Chains", "multi-provider reliability", 'both'),
        ("7", "Token Usage Tracking", "cost monitoring and optimization", 'openai'),
        ("8", "LCEL Integration", "chain composition with LLMs", 'openai'),
    ]

    for num, name, desc, requires in demos:
        needs_api = requires in ('openai', 'both')
        needs_anthropic = requires in ('anthropic', 'both')

        has_required = (
            (not needs_api or has_openai) and
            (not needs_anthropic or has_anthropic)
        )

        api_marker = "🔑" if (needs_api or needs_anthropic) else "  "

        if not has_required:
            if requires == 'both':
                status = " ⚠️ (needs both API keys)"
            elif requires == 'openai':
                status = " ⚠️ (needs OpenAI key)"
            else:
                status = " ⚠️ (needs Anthropic key)"
        else:
            status = ""

        print(f"  {api_marker} [{num}] {name}")
        print(f"      {desc}{status}")
        print()

    if not (has_openai or has_anthropic):
        print("  ⚠️  At least one API key required")
        print("     Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 70)


def run_selected_demos(selections: str, has_openai: bool, has_anthropic: bool) -> bool:
    """run selected demos based on user input"""
    selections = selections.lower().strip()

    if selections == 'q':
        return False

    demo_map = {
        '1': ("ChatOpenAI Basic", demo_chatopenai_basic, 'openai'),
        '2': ("ChatAnthropic Basic", demo_chatanthropic_basic, 'anthropic'),
        '3': ("Temperature Control", demo_temperature_control, 'openai'),
        '4': ("Streaming", demo_streaming, 'openai'),
        '5': ("Provider Switching", demo_provider_switching, 'both'),
        '6': ("Fallback Chain", demo_fallback_chain, 'both'),
        '7': ("Token Tracking", demo_token_tracking, 'openai'),
        '8': ("LCEL Integration", demo_lcel_integration, 'openai'),
    }

    if selections == 'a':
        # run all demos
        for name, demo_func, requires in demo_map.values():
            if requires == 'openai' and not has_openai:
                print(f"\n⚠️  Skipping {name}: OpenAI API key required")
                continue
            if requires == 'anthropic' and not has_anthropic:
                print(f"\n⚠️  Skipping {name}: Anthropic API key required")
                continue
            if requires == 'both' and not (has_openai and has_anthropic):
                print(f"\n⚠️  Skipping {name}: Both API keys required")
                continue
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
    else:
        # parse comma-separated selections
        selected = [s.strip() for s in selections.split(',')]
        for sel in selected:
            if sel in demo_map:
                name, demo_func, requires = demo_map[sel]

                # check API key requirements
                if requires == 'openai' and not has_openai:
                    print(f"\n⚠️  Cannot run {name}: OpenAI API key required")
                    continue
                if requires == 'anthropic' and not has_anthropic:
                    print(f"\n⚠️  Cannot run {name}: Anthropic API key required")
                    continue
                if requires == 'both' and not (has_openai and has_anthropic):
                    print(f"\n⚠️  Cannot run {name}: Both API keys required")
                    continue

                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Error in {name}: {e}")
            else:
                print(f"⚠️  Invalid selection: {sel}")

    return True


def main() -> None:
    """run demonstrations with interactive menu"""
    has_openai, has_anthropic = check_api_keys()

    print("\n" + "=" * 70)
    print("  LLM Integration - Practical Examples")
    print("  Real API calls to OpenAI and Anthropic")
    print("=" * 70)
    print("\n## API Key Status:")
    print(f"OPENAI_API_KEY: {'✓ Found' if has_openai else '✗ Missing'}")
    print(f"ANTHROPIC_API_KEY: {'✓ Found' if has_anthropic else '✗ Missing'}")

    if not (has_openai or has_anthropic):
        print("\n❌ No API keys found!")
        print("Set at least one API key in .env to run demos:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  ANTHROPIC_API_KEY=your-key-here")
        return

    while True:
        show_menu(has_openai, has_anthropic)
        selection = input("\nSelect demos to run (comma-separated) or 'a' for all: ").strip()

        if not selection:
            continue

        if not run_selected_demos(selection, has_openai, has_anthropic):
            break

        print("\n" + "=" * 70)
        print("  Demos complete!")
        print("=" * 70)

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

    print("\n" + "=" * 70)
    print("  Thanks for exploring LLM integration!")
    print("  You've mastered real-world LLM API usage")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
