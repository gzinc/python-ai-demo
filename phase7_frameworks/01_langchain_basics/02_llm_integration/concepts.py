"""
LLM Integration - Conceptual Understanding (No API Key Needed)

Demonstrates LangChain's unified chat model interface and configuration patterns
without making actual API calls. Focus on understanding the abstraction layer,
message structure, and configuration options.

Run: uv run python -m phase7_frameworks.01_langchain_basics.02_llm_integration.concepts
"""

from inspect import cleandoc
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from common.demo_menu import Demo, MenuRunner


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


# region Demo 1: Unified Chat Interface Overview


def demo_unified_interface() -> None:
    """demonstrate the unified chat model interface concept"""
    print_section("Demo 1: Unified Chat Interface")

    print(cleandoc('''
        LangChain provides a consistent interface across all LLM providers:

        Key Interface Methods:
        - invoke(messages)  → single response
        - stream(messages)  → streaming response chunks
        - batch(messages)   → multiple requests in parallel
        - ainvoke()         → async single response
        - astream()         → async streaming

        All chat models inherit from BaseChatModel and implement these methods.
    '''))

    print("\n## Example Interface Signatures:")
    print(cleandoc('''
        # OpenAI
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)

        # Anthropic
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)

        # Both use identical invoke/stream methods:
        response = llm.invoke([HumanMessage(content="Hello")])
        for chunk in llm.stream([HumanMessage(content="Hello")]):
            print(chunk.content, end="")
    '''))

    print("\n## Why This Matters:")
    print("- Swap providers without changing application code")
    print("- Implement fallbacks transparently")
    print("- Test with different models easily")
    print("- Build provider-agnostic AI applications")


# endregion

# region Demo 2: Message Types and Structure


def demo_message_types() -> None:
    """demonstrate different message types in LangChain"""
    print_section("Demo 2: Message Types and Structure")

    print("## Message Types:")

    # System message
    system_msg = SystemMessage(content="You are a helpful AI assistant")
    print(f"\n1. SystemMessage:")
    print(f"   Type: {system_msg.type}")
    print(f"   Content: {system_msg.content}")
    print(f"   Purpose: Set behavior/context for conversation")

    # Human message
    human_msg = HumanMessage(content="What are embeddings?")
    print(f"\n2. HumanMessage:")
    print(f"   Type: {human_msg.type}")
    print(f"   Content: {human_msg.content}")
    print(f"   Purpose: User input/questions")

    # AI message
    ai_msg = AIMessage(content="Embeddings are vector representations...")
    print(f"\n3. AIMessage:")
    print(f"   Type: {ai_msg.type}")
    print(f"   Content: {ai_msg.content}")
    print(f"   Purpose: Assistant responses")

    print("\n## Message Attributes:")
    print(f"- type: {human_msg.type}")
    print(f"- content: {human_msg.content}")
    print(f"- additional_kwargs: {human_msg.additional_kwargs}")

    print("\n## Conversation Structure:")
    conversation = [
        SystemMessage(content="You are a Python expert"),
        HumanMessage(content="How do I read a file?"),
        AIMessage(content="Use `with open(file) as f:`"),
        HumanMessage(content="Can you show an example?"),
    ]

    print(f"\nConversation with {len(conversation)} messages:")
    for i, msg in enumerate(conversation):
        print(f"  {i+1}. {msg.type:10s}: {msg.content[:40]}...")


# endregion

# region Demo 3: Configuration Parameters Explained


def demo_configuration_parameters() -> None:
    """demonstrate LLM configuration parameters without API calls"""
    print_section("Demo 3: Configuration Parameters")

    print("## Core Parameters:\n")

    # temperature
    print("1. temperature (0.0 - 1.0):")
    print("   - Controls randomness/creativity")
    print("   - 0.0: Deterministic, consistent responses")
    print("   - 0.7: Balanced (default for most tasks)")
    print("   - 1.0: Maximum creativity, varied responses")
    print("   Usage: temperature=0.0 for factual tasks, 1.0 for creative writing")

    # max_tokens
    print("\n2. max_tokens:")
    print("   - Maximum response length")
    print("   - Limits generation to prevent excessive costs")
    print("   - None: No limit (use model's maximum)")
    print("   - 100: Short answers")
    print("   - 2000: Longer detailed responses")
    print("   Usage: Set based on expected response length")

    # model
    print("\n3. model:")
    print("   - Specific model version to use")
    print("   - OpenAI: 'gpt-4', 'gpt-3.5-turbo'")
    print("   - Anthropic: 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'")
    print("   Usage: Choose based on task complexity and cost")

    # streaming
    print("\n4. streaming:")
    print("   - Enable real-time token delivery")
    print("   - True: Tokens arrive progressively (better UX)")
    print("   - False: Wait for complete response (default)")
    print("   Usage: Enable for interactive chat interfaces")

    # timeout
    print("\n5. timeout:")
    print("   - Maximum wait time for response")
    print("   - Default: 60 seconds")
    print("   - Usage: Increase for complex queries, decrease for time-sensitive apps")

    # max_retries
    print("\n6. max_retries:")
    print("   - Number of retry attempts on failure")
    print("   - Default: 2")
    print("   - Usage: Increase for unreliable networks, 0 for fail-fast behavior")

    print("\n## Example Configuration Scenarios:")

    configs: dict[str, dict[str, Any]] = {
        "Factual Q&A": {"temperature": 0.0, "max_tokens": 500},
        "Creative Writing": {"temperature": 1.0, "max_tokens": 2000},
        "Code Generation": {"temperature": 0.2, "max_tokens": 1000},
        "Chat Interface": {"temperature": 0.7, "streaming": True},
        "Quick Summary": {"temperature": 0.5, "max_tokens": 150},
    }

    for scenario, config in configs.items():
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        print(f"- {scenario:20s}: {config_str}")


# endregion

# region Demo 4: Streaming vs Non-Streaming Patterns


def demo_streaming_patterns() -> None:
    """demonstrate streaming vs non-streaming patterns conceptually"""
    print_section("Demo 4: Streaming vs Non-Streaming Patterns")

    print("## Non-Streaming (Default):")
    print(cleandoc('''
        # Wait for complete response
        response = llm.invoke(messages)
        print(response.content)  # All tokens at once

        Pros:
        - Simple to implement
        - Easy to process complete response
        - Works well for batch processing

        Cons:
        - User waits for entire generation
        - Poor UX for long responses
        - No progress indication
    '''))

    print("\n## Streaming:")
    print(cleandoc('''
        # Process tokens as they arrive
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
        print()  # newline

        Pros:
        - Immediate feedback to user
        - Better perceived performance
        - Can process partial results
        - Lower latency to first token

        Cons:
        - More complex to implement
        - Harder to handle complete response
        - Requires buffer management
    '''))

    print("\n## Use Case Decision:")
    use_cases = [
        ("Interactive Chat", "Streaming", "Real-time user experience"),
        ("Batch Processing", "Non-Streaming", "Process complete results"),
        ("Code Generation", "Non-Streaming", "Need complete code block"),
        ("Live Assistant", "Streaming", "Natural conversation flow"),
        ("API Integration", "Non-Streaming", "Simple JSON processing"),
        ("Writing Aid", "Streaming", "Progressive text generation"),
    ]

    for use_case, pattern, reason in use_cases:
        print(f"- {use_case:20s}: {pattern:15s} ({reason})")


# endregion

# region Demo 5: Provider Comparison Matrix


def demo_provider_comparison() -> None:
    """compare different LLM provider characteristics"""
    print_section("Demo 5: Provider Comparison Matrix")

    print("## Provider Feature Comparison:\n")

    # Define providers and features
    providers = ["OpenAI GPT-4", "Anthropic Claude-3.5", "OpenAI GPT-3.5"]

    features = {
        "Context Window": ["128K tokens", "200K tokens", "16K tokens"],
        "Streaming": ["✓ Yes", "✓ Yes", "✓ Yes"],
        "Function Calling": ["✓ Excellent", "✓ Excellent", "✓ Good"],
        "Vision Support": ["✓ GPT-4V", "✓ Yes", "✗ No"],
        "Speed": ["Fast", "Fast", "Very Fast"],
        "Cost (per 1M tokens)": ["$30/$60", "$3/$15", "$0.50/$1.50"],
        "Best For": ["Complex tasks", "Long context", "High volume"],
    }

    # Print header
    print(f"{'Feature':<25s} | {' | '.join(f'{p:20s}' for p in providers)}")
    print("-" * 95)

    # Print features
    for feature, values in features.items():
        values_str = " | ".join(f"{v:20s}" for v in values)
        print(f"{feature:<25s} | {values_str}")

    print("\n## Choosing the Right Provider:")
    print("- GPT-4: Complex reasoning, vision tasks, function calling")
    print("- Claude-3.5: Very long context, nuanced understanding, safety")
    print("- GPT-3.5: High-volume simple tasks, cost optimization")

    print("\n## Cost Optimization Strategy:")
    print("1. Use GPT-3.5-turbo for simple queries (classification, extraction)")
    print("2. Use GPT-4 for complex reasoning (analysis, planning)")
    print("3. Use Claude for very long context (document analysis)")
    print("4. Implement routing logic based on query complexity")


# endregion

# region Demo 6: Error Handling Strategies


def demo_error_handling() -> None:
    """demonstrate error handling patterns for LLM calls"""
    print_section("Demo 6: Error Handling Strategies")

    print("## Common Error Types:\n")

    errors = [
        ("RateLimitError", "Too many requests", "Implement exponential backoff"),
        ("TimeoutError", "Request exceeded timeout", "Increase timeout or retry"),
        ("AuthenticationError", "Invalid API key", "Check environment variables"),
        ("InvalidRequestError", "Malformed request", "Validate input parameters"),
        ("ServiceUnavailableError", "Provider down", "Use fallback provider"),
        ("ContextLengthError", "Input too long", "Truncate or summarize input"),
    ]

    for error, cause, solution in errors:
        print(f"- {error:25s}: {cause:30s} → {solution}")

    print("\n## Error Handling Pattern (Conceptual):")
    print(cleandoc('''
        try:
            response = llm.invoke(messages)
        except RateLimitError:
            # Wait and retry with exponential backoff
            time.sleep(2 ** retry_count)
            response = llm.invoke(messages)
        except TimeoutError:
            # Use cached response or fallback
            response = get_cached_response() or fallback_llm.invoke(messages)
        except AuthenticationError:
            # Log and alert - requires manual intervention
            logger.error("Invalid API key")
            raise
        except Exception as e:
            # Catch-all with logging
            logger.error(f"Unexpected error: {e}")
            return default_response
    '''))

    print("\n## Retry Strategy:")
    print("1. Exponential Backoff: 1s, 2s, 4s, 8s...")
    print("2. Max Retries: Limit to 3-5 attempts")
    print("3. Circuit Breaker: Stop after multiple failures")
    print("4. Fallback: Switch to alternative provider")


# endregion

# region Demo 7: Retry and Fallback Patterns


def demo_retry_fallback() -> None:
    """demonstrate retry and fallback patterns conceptually"""
    print_section("Demo 7: Retry and Fallback Patterns")

    print("## Pattern 1: Automatic Retry")
    print(cleandoc('''
        # Built into LangChain
        llm = ChatOpenAI(
            model="gpt-4",
            max_retries=3,  # Retry up to 3 times
            timeout=60       # 60 second timeout
        )

        # Automatic retry on transient failures:
        # - Network issues
        # - Rate limits
        # - Temporary service issues
    '''))

    print("\n## Pattern 2: Provider Fallback")
    print(cleandoc('''
        from langchain_core.runnables import RunnableWithFallbacks

        # Primary: GPT-4 (best quality)
        primary = ChatOpenAI(model="gpt-4")

        # Fallback 1: Claude (different provider)
        fallback1 = ChatAnthropic(model="claude-3-5-sonnet-20241022")

        # Fallback 2: GPT-3.5 (cheaper, faster)
        fallback2 = ChatOpenAI(model="gpt-3.5-turbo")

        # Chain with fallbacks
        llm = primary.with_fallbacks([fallback1, fallback2])

        # Automatically tries:
        # 1. GPT-4 (if fails...)
        # 2. Claude (if fails...)
        # 3. GPT-3.5 (last resort)
    '''))

    print("\n## Pattern 3: Graceful Degradation")
    print(cleandoc('''
        # Try advanced model, fallback to simpler one
        try:
            response = gpt4_llm.invoke(messages)
        except Exception:
            # Degrade to faster/cheaper model
            response = gpt35_llm.invoke(messages)
            # Add disclaimer about degraded quality
            response.content = f"[Using fallback model] {response.content}"
    '''))

    print("\n## Decision Matrix:")
    scenarios = [
        ("Critical Production", "Multi-provider fallback + retry", "High"),
        ("Cost-Sensitive", "Single provider + retry", "Medium"),
        ("Development/Testing", "No fallback + fast fail", "Low"),
        ("High-Availability", "Multi-region + multi-provider", "Critical"),
    ]

    for scenario, strategy, reliability in scenarios:
        print(f"- {scenario:25s}: {strategy:35s} (Reliability: {reliability})")


# endregion

# region Demo 8: Token Counting and Cost Estimation


def demo_token_cost_estimation() -> None:
    """demonstrate token counting and cost estimation concepts"""
    print_section("Demo 8: Token Counting and Cost Estimation")

    print("## Token Basics:")
    print("- Token ≈ 4 characters (English)")
    print("- 1 word ≈ 1.3 tokens on average")
    print("- 100 tokens ≈ 75 words")

    print("\n## Counting Tokens (Conceptual):")
    print(cleandoc('''
        from langchain.callbacks import get_openai_callback

        # Track token usage
        with get_openai_callback() as cb:
            response = llm.invoke(messages)
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
            print(f"Total tokens: {cb.total_tokens}")
            print(f"Total cost: ${cb.total_cost:.4f}")
    '''))

    print("\n## Cost Estimation (2024 Pricing):")

    # Pricing per 1M tokens (input/output)
    pricing = {
        "GPT-4": (30.00, 60.00),
        "GPT-4 Turbo": (10.00, 30.00),
        "GPT-3.5 Turbo": (0.50, 1.50),
        "Claude-3 Opus": (15.00, 75.00),
        "Claude-3.5 Sonnet": (3.00, 15.00),
        "Claude-3 Haiku": (0.25, 1.25),
    }

    print(f"\n{'Model':<20s} | {'Input/1M':<10s} | {'Output/1M':<10s} | {'Est. 1K Query':<15s}")
    print("-" * 70)

    for model, (input_cost, output_cost) in pricing.items():
        # Estimate: 500 input tokens, 500 output tokens per query
        est_cost = (500 / 1_000_000 * input_cost) + (500 / 1_000_000 * output_cost)
        print(f"{model:<20s} | ${input_cost:<9.2f} | ${output_cost:<9.2f} | ${est_cost * 1000:<14.4f}")

    print("\n## Cost Optimization Tips:")
    print("1. Use cheaper models (GPT-3.5, Claude Haiku) for simple tasks")
    print("2. Reduce prompt length - be concise")
    print("3. Set max_tokens to prevent runaway costs")
    print("4. Cache responses when possible")
    print("5. Batch similar requests")
    print("6. Monitor usage with callbacks")

    print("\n## Example Cost Calculation:")
    print("Query: 500 input tokens + 500 output tokens")
    print("GPT-4: $0.030/1M * 500 + $0.060/1M * 500 = $0.045 per query")
    print("GPT-3.5: $0.0005/1M * 500 + $0.0015/1M * 500 = $0.001 per query")
    print("→ GPT-3.5 is 45x cheaper for this query")


# endregion



# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Unified Chat Interface", "unified chat interface", demo_unified_interface),
    Demo("2", "Message Types and Structure", "message types and structure", demo_message_types),
    Demo("3", "Configuration Parameters", "configuration parameters", demo_configuration_parameters),
    Demo("4", "Streaming vs Non-Streaming", "streaming vs non-streaming", demo_streaming_patterns),
    Demo("5", "Provider Comparison Matrix", "provider comparison matrix", demo_provider_comparison),
    Demo("6", "Error Handling Strategies", "error handling strategies", demo_error_handling),
    Demo("7", "Retry and Fallback Patterns", "retry and fallback patterns", demo_retry_fallback),
    Demo("8", "Token Counting and Cost", "token counting and cost", demo_token_cost_estimation),
]

# endregion

def main() -> None:
    """run demonstrations with interactive menu"""
    print("\n" + "=" * 70)
    print("  LLM Integration - Conceptual Understanding")
    print("  No API key required - demonstrates patterns only")
    print("=" * 70)

    
    runner = MenuRunner(DEMOS, title="TODO: Add title")
    runner.run()


if __name__ == "__main__":
    main()
