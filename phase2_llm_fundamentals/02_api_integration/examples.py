"""
Module: API Integration - Connecting to Real LLMs

This module demonstrates API integration patterns.
Examples work without API keys (using mocks to show patterns).
See live_examples.py for real API calls.

Run with: uv run python phase2_llm_fundamentals/02_api_integration/examples.py
"""

import os
import time
import json
from inspect import cleandoc
from typing import Generator
from dataclasses import dataclass


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# simulated response objects (mirrors real API structure)
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class ChatCompletion:
    id: str
    model: str
    choices: list[Choice]
    usage: TokenUsage


def simulate_api_call(
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 100,
) -> ChatCompletion:
    """simulate an API call - shows the exact structure you'll get from real APIs"""
    # in reality, this would be:
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )

    # simulated response matching OpenAI's structure
    user_content = messages[-1]["content"] if messages else ""

    # simple response simulation
    if "prime" in user_content.lower():
        response_text = "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    elif "capital" in user_content.lower():
        response_text = "The capital of France is Paris."
    else:
        response_text = f"This is a simulated response to: {user_content[:50]}..."

    # calculate approximate tokens (rough: 1 token ≈ 4 chars)
    prompt_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
    completion_tokens = len(response_text) // 4

    return ChatCompletion(
        id="chatcmpl-simulated",
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def example_basic_api_call() -> None:
    """demonstrate basic API call structure"""
    print_section("1. Basic API Call Structure")

    print("The core pattern for calling any LLM API:")
    print()

    # the messages format (universal across providers)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    print("Request:")
    print(f"  model: gpt-4o")
    print(f"  messages: {json.dumps(messages, indent=4)}")
    print()

    # make the call
    response = simulate_api_call(messages, model="gpt-4o")

    print("Response structure:")
    print(f"  id: {response.id}")
    print(f"  model: {response.model}")
    print(f"  content: {response.choices[0].message.content}")
    print(f"  finish_reason: {response.choices[0].finish_reason}")
    print(f"  tokens used: {response.usage.total_tokens}")


def example_real_openai_code() -> None:
    """show actual OpenAI code (for reference)"""
    print_section("2. Real OpenAI Code (Reference)")

    code = cleandoc('''
        # actual OpenAI code (requires: pip install openai)
        from openai import OpenAI

        client = OpenAI()  # uses OPENAI_API_KEY env var

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a haiku about programming"},
            ],
            temperature=0.7,
            max_tokens=100,
        )

        # access the response
        print(response.choices[0].message.content)
        print(f"Tokens used: {response.usage.total_tokens}")
    ''')
    print(code)


def example_real_anthropic_code() -> None:
    """show actual Anthropic code (for reference)"""
    print_section("3. Real Anthropic Code (Reference)")

    code = cleandoc('''
        # actual Anthropic code (requires: pip install anthropic)
        from anthropic import Anthropic

        client = Anthropic()  # uses ANTHROPIC_API_KEY env var

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a helpful assistant.",  # system is separate!
            messages=[
                {"role": "user", "content": "Write a haiku about programming"},
            ],
        )

        # access the response (slightly different structure)
        print(response.content[0].text)
        print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
    ''')
    print(code)

    print("\nKey difference: Anthropic separates 'system' from 'messages'")


def example_temperature_comparison() -> None:
    """demonstrate temperature effects"""
    print_section("4. Temperature Effects")

    print("Temperature controls randomness in outputs:")
    print("  0.0 = Deterministic (same input → same output)")
    print("  0.7 = Balanced (default, good for most tasks)")
    print("  1.0 = Creative (more variation, good for brainstorming)")
    print()

    messages = [{"role": "user", "content": "Give me a word that means 'happy'"}]

    print("Temperature 0.0 (deterministic):")
    print("  Run 1: 'joyful'")
    print("  Run 2: 'joyful'")
    print("  Run 3: 'joyful'")
    print()

    print("Temperature 1.0 (creative):")
    print("  Run 1: 'elated'")
    print("  Run 2: 'cheerful'")
    print("  Run 3: 'blissful'")
    print()

    print("Best practices:")
    print("  - Code generation: 0.0 (consistency)")
    print("  - Q&A / RAG: 0.0-0.3 (accuracy)")
    print("  - Creative writing: 0.7-1.0 (variety)")


def example_token_tracking() -> None:
    """demonstrate token usage and cost calculation"""
    print_section("5. Token Usage & Cost Tracking")

    messages = [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "Write a function to check if a number is prime."},
    ]

    response = simulate_api_call(messages, model="gpt-4o")

    print("Token breakdown:")
    print(f"  Prompt tokens:     {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total tokens:      {response.usage.total_tokens}")
    print()

    # cost calculation (GPT-4o prices as of 2024)
    input_price = 2.50 / 1_000_000  # $2.50 per 1M tokens
    output_price = 10.00 / 1_000_000  # $10.00 per 1M tokens

    input_cost = response.usage.prompt_tokens * input_price
    output_cost = response.usage.completion_tokens * output_price
    total_cost = input_cost + output_cost

    print("Cost calculation (GPT-4o):")
    print(f"  Input cost:  ${input_cost:.6f}")
    print(f"  Output cost: ${output_cost:.6f}")
    print(f"  Total cost:  ${total_cost:.6f}")
    print()

    print("At 1000 requests/day:")
    print(f"  Daily cost:   ${total_cost * 1000:.2f}")
    print(f"  Monthly cost: ${total_cost * 1000 * 30:.2f}")


def example_error_handling() -> None:
    """demonstrate error handling patterns"""
    print_section("6. Error Handling Patterns")

    code = cleandoc('''
        import time
        from openai import OpenAI, RateLimitError, APIError

        client = OpenAI()

        def call_with_retry(messages, max_retries=3):
            """call API with exponential backoff retry"""
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                    )
                    return response

                except RateLimitError:
                    # rate limited - wait and retry
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)

                except APIError as e:
                    # server error - might be temporary
                    if attempt < max_retries - 1:
                        print(f"API error: {e}. Retrying...")
                        time.sleep(1)
                    else:
                        raise

            raise Exception("Max retries exceeded")
    ''')
    print(code)

    print("\nCommon errors to handle:")
    print("  401 - Invalid API key (check your key)")
    print("  429 - Rate limit (wait and retry)")
    print("  500 - Server error (retry with backoff)")
    print("  Timeout - Slow response (increase timeout)")


def simulate_stream() -> Generator[str, None, None]:
    """simulate streaming response"""
    response = "Here's a simple Python function:\n\ndef greet(name):\n    return f'Hello, {name}!'"
    for char in response:
        yield char
        time.sleep(0.02)  # simulate network delay


def example_streaming() -> None:
    """demonstrate streaming responses"""
    print_section("7. Streaming Responses")

    print("Streaming shows tokens as they're generated (better UX):")
    print()

    code = cleandoc('''
        # real streaming code
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Write a greeting function"}],
            stream=True,  # enable streaming
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
    ''')
    print(code)
    print()

    print("Simulated stream output:")
    print("-" * 40)
    for char in simulate_stream():
        print(char, end="", flush=True)
    print()
    print("-" * 40)
    print()

    print("Why streaming matters:")
    print("  - Users see progress immediately")
    print("  - Feels faster (even if same total time)")
    print("  - Essential for chat interfaces")


def example_conversation_history() -> None:
    """demonstrate multi-turn conversation"""
    print_section("8. Multi-Turn Conversations")

    print("LLMs are stateless - you must send full history each time:")
    print()

    # conversation builds up
    conversation = []

    # turn 1
    conversation.append({"role": "user", "content": "My name is Alex."})
    conversation.append(
        {"role": "assistant", "content": "Nice to meet you, Alex! How can I help?"}
    )

    # turn 2
    conversation.append({"role": "user", "content": "What's my name?"})
    conversation.append(
        {"role": "assistant", "content": "Your name is Alex, as you told me earlier."}
    )

    print("Conversation history sent with each request:")
    for i, msg in enumerate(conversation):
        print(f"  {i+1}. [{msg['role']}]: {msg['content']}")

    print()
    print("Token implications:")
    print("  - History grows with each turn")
    print("  - Long conversations = more tokens = more cost")
    print("  - May need to summarize or truncate old messages")

    code = cleandoc('''
        # practical pattern
        MAX_HISTORY = 10  # keep last N messages

        def chat(user_message, history):
            history.append({"role": "user", "content": user_message})

            # truncate if too long
            messages_to_send = history[-MAX_HISTORY:]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_to_send,
            )

            assistant_message = response.choices[0].message.content
            history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
    ''')
    print()
    print("Truncation pattern:")
    print(code)


def example_provider_comparison() -> None:
    """compare OpenAI vs Anthropic patterns"""
    print_section("9. Provider Comparison: OpenAI vs Anthropic")

    print("OpenAI (GPT-4o):")
    print("-" * 40)
    openai_code = cleandoc('''
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi!"},
            ],
        )
        text = response.choices[0].message.content
    ''')
    print(openai_code)

    print("Anthropic (Claude):")
    print("-" * 40)
    anthropic_code = cleandoc('''
        from anthropic import Anthropic
        client = Anthropic()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,  # required!
            system="Be concise.",  # separate param
            messages=[
                {"role": "user", "content": "Hi!"},
            ],
        )
        text = response.content[0].text
    ''')
    print(anthropic_code)

    print("Key differences:")
    print("  1. System prompt: OpenAI=in messages, Anthropic=separate param")
    print("  2. max_tokens: Optional in OpenAI, required in Anthropic")
    print("  3. Response access: .message.content vs .content[0].text")
    print("  4. Model names: gpt-4o vs claude-sonnet-4-20250514")


def example_complete_pattern() -> None:
    """show a complete, production-ready pattern"""
    print_section("10. Complete Production Pattern")

    code = cleandoc('''
        import os
        from openai import OpenAI, RateLimitError
        import time

        class LLMClient:
            """production-ready LLM client with retry and tracking"""

            def __init__(self):
                self.client = OpenAI()
                self.total_tokens = 0
                self.total_cost = 0.0

            def chat(self, messages, model="gpt-4o", temperature=0.0, max_retries=3):
                """make API call with retry logic and cost tracking"""
                for attempt in range(max_retries):
                    try:
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                        )

                        # track usage
                        self.total_tokens += response.usage.total_tokens
                        self._update_cost(response.usage, model)

                        return response.choices[0].message.content

                    except RateLimitError:
                        wait = 2 ** attempt
                        print(f"rate limited, waiting {wait}s...")
                        time.sleep(wait)

                raise Exception("max retries exceeded")

            def _update_cost(self, usage, model):
                # simplified pricing
                if "gpt-4" in model:
                    cost = (usage.prompt_tokens * 2.5 + usage.completion_tokens * 10) / 1_000_000
                else:
                    cost = usage.total_tokens * 0.5 / 1_000_000
                self.total_cost += cost

            def get_stats(self):
                return {
                    "total_tokens": self.total_tokens,
                    "total_cost": f"${self.total_cost:.4f}",
                }

        # usage
        client = LLMClient()
        response = client.chat([{"role": "user", "content": "Hello!"}])
        print(response)
        print(client.get_stats())
    ''')
    print(code)


def main() -> None:
    """run all examples"""
    print("\n" + "=" * 60)
    print("  API Integration - Connecting to Real LLMs")
    print("  (Pattern examples - see live_examples.py for real calls)")
    print("=" * 60)

    example_basic_api_call()
    example_real_openai_code()
    example_real_anthropic_code()
    example_temperature_comparison()
    example_token_tracking()
    example_error_handling()
    example_streaming()
    example_conversation_history()
    example_provider_comparison()
    example_complete_pattern()

    print_section("Summary")
    print(cleandoc("""
        Key patterns learned:
          1. Messages format: [{"role": "...", "content": "..."}]
          2. Temperature: 0.0 for consistency, higher for creativity
          3. Token tracking: Monitor usage for cost control
          4. Error handling: Retry with exponential backoff
          5. Streaming: Better UX for chat interfaces
          6. History management: Send full context, truncate if needed

        Next: Run live_examples.py to make real API calls!
    """))


if __name__ == "__main__":
    main()
