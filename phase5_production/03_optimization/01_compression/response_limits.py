"""
Response limits - set appropriate max_tokens for responses.

WHY THIS MATTERS:
- Without max_tokens, LLM generates until it "feels done" (often 500-2000 tokens)
- Classification task needs 1 word, but LLM might explain its reasoning
- You pay for EVERY output token whether you need it or not

PRODUCTION EXAMPLE:
- Classification endpoint, 10,000 requests/day
- Without limit: avg 200 tokens/response = 2M output tokens = $20/day (gpt-4o)
- With limit=10: avg 5 tokens/response = 50K output tokens = $0.50/day
- Savings: $19.50/day = $585/month

Run with: uv run python -m phase5_production.03_optimization.01_compression.response_limits
"""
from dataclasses import dataclass
from inspect import cleandoc
from typing import Any
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TaskConfig:
    """configuration for a specific task type"""
    max_tokens: int
    temperature: float = 0.0
    stop_sequences: list[str] | None = None


class ResponseLimiter:
    """
    Production response limiter with task-specific configurations.

    Real-world usage:
        limiter = ResponseLimiter()
        config = limiter.get_config('classification')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[...],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=config.stop_sequences,
        )
    """

    TASK_CONFIGS: dict[str, TaskConfig] = {
        # classification: just need the label
        'classification': TaskConfig(
            max_tokens=10,
            temperature=0.0,
            stop_sequences=['\n', '.'],
        ),

        # sentiment: positive/negative/neutral
        'sentiment': TaskConfig(
            max_tokens=5,
            temperature=0.0,
        ),

        # extraction: structured data, moderate length
        'extraction': TaskConfig(
            max_tokens=500,
            temperature=0.0,
        ),

        # summarization: condensed but complete
        'summarization': TaskConfig(
            max_tokens=300,
            temperature=0.3,
        ),

        # qa: answer with some context
        'qa': TaskConfig(
            max_tokens=200,
            temperature=0.2,
        ),

        # code generation: needs more room
        'code': TaskConfig(
            max_tokens=1500,
            temperature=0.0,
        ),

        # chat: conversational, variable length
        'chat': TaskConfig(
            max_tokens=500,
            temperature=0.7,
        ),

        # default fallback
        'default': TaskConfig(
            max_tokens=500,
            temperature=0.3,
        ),
    }

    def get_config(self, task_type: str) -> TaskConfig:
        """get full config for task type"""
        return self.TASK_CONFIGS.get(task_type, self.TASK_CONFIGS['default'])

    def get_limit(self, task_type: str) -> int:
        """get just max_tokens for task type"""
        return self.get_config(task_type).max_tokens

    def apply_to_request(self, task_type: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Apply limits to an API request kwargs dict.

        Usage:
            kwargs = {'model': 'gpt-4o', 'messages': [...]}
            kwargs = limiter.apply_to_request('classification', kwargs)
            response = client.chat.completions.create(**kwargs)
        """
        config = self.get_config(task_type)
        request_kwargs['max_tokens'] = config.max_tokens
        request_kwargs['temperature'] = config.temperature
        if config.stop_sequences:
            request_kwargs['stop'] = config.stop_sequences
        return request_kwargs


def calculate_savings(
        requests_per_day: int,
        without_limit_tokens: int,
        with_limit_tokens: int,
        price_per_million: float = 10.0,  # gpt-4o output price
) -> dict[str, float]:
    """calculate cost savings from response limits"""
    without_cost = (requests_per_day * without_limit_tokens / 1_000_000) * price_per_million
    with_cost = (requests_per_day * with_limit_tokens / 1_000_000) * price_per_million

    return {
        'daily_without': without_cost,
        'daily_with': with_cost,
        'daily_savings': without_cost - with_cost,
        'monthly_savings': (without_cost - with_cost) * 30,
    }


# region Demo Functions

def demo_real_openai_comparison() -> None:
    """
    Real OpenAI demo showing token savings with response limits.

    Compares the SAME classification request:
    - Without max_tokens: LLM generates as much as it wants
    - With max_tokens=10: LLM stops after 10 tokens
    """
    from openai import OpenAI

    client = OpenAI()
    limiter = ResponseLimiter()

    test_text = "URGENT: You've won $1,000,000! Click here to claim your prize NOW!!!"

    system_prompt = cleandoc('''
        Classify the following text as one of: spam, ham, or unknown.
        Respond with ONLY the classification label.
    ''')

    print("=" * 70)
    print("  Real OpenAI Demo: Response Limits in Action")
    print("=" * 70)

    print(f"\nTest text: {test_text}\n")

    # 1. WITHOUT response limits (what most developers do)
    print("-" * 70)
    print("1. WITHOUT max_tokens (default behavior)")
    print("-" * 70)

    response_no_limit = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_text},
        ],
        temperature=0.0,
        # no max_tokens = LLM decides when to stop
    )

    result_no_limit = response_no_limit.choices[0].message.content
    tokens_no_limit = response_no_limit.usage.completion_tokens

    print(f"Response: {result_no_limit}")
    print(f"Tokens used: {tokens_no_limit}")

    # 2. WITH response limits (optimized)
    print("\n" + "-" * 70)
    print("2. WITH max_tokens=10 + stop sequences (optimized)")
    print("-" * 70)

    config = limiter.get_config('classification')

    response_limited = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_text},
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=config.stop_sequences,
    )

    result_limited = response_limited.choices[0].message.content
    tokens_limited = response_limited.usage.completion_tokens

    print(f"Response: {result_limited}")
    print(f"Tokens used: {tokens_limited}")

    # 3. cost comparison
    print("\n" + "-" * 70)
    print("3. COST COMPARISON (at 10,000 requests/day)")
    print("-" * 70)

    price_per_million = 0.60  # gpt-4o-mini output price

    cost_no_limit = (10_000 * tokens_no_limit / 1_000_000) * price_per_million
    cost_limited = (10_000 * tokens_limited / 1_000_000) * price_per_million
    savings_daily = cost_no_limit - cost_limited
    savings_monthly = savings_daily * 30

    print(f"Without limits: {tokens_no_limit} tokens × 10K = ${cost_no_limit:.2f}/day")
    print(f"With limits:    {tokens_limited} tokens × 10K = ${cost_limited:.2f}/day")
    print(f"Daily savings:  ${savings_daily:.2f}")
    print(f"Monthly savings: ${savings_monthly:.2f}")

    # 4. same result, fewer tokens
    print("\n" + "=" * 70)
    print(f"  Both give '{result_limited.strip()}' - but one costs {tokens_no_limit - tokens_limited} fewer tokens!")
    print("=" * 70)


def demo_multiple_tasks() -> None:
    """Demo response limits across different task types with real API calls."""
    from openai import OpenAI

    client = OpenAI()
    limiter = ResponseLimiter()

    print("\n" + "=" * 70)
    print("  Multiple Task Types Demo")
    print("=" * 70)

    tasks = [
        ('sentiment', "I love this product! Best purchase ever!", "Rate sentiment: positive, negative, or neutral"),
        ('classification', "Meeting tomorrow at 3pm in conference room B", "Classify: calendar, task, note, or other"),
        ('extraction', "Call John at 555-1234 about the project", "Extract: name and phone number as JSON"),
    ]

    for task_type, text, instruction in tasks:
        config = limiter.get_config(task_type)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=config.stop_sequences,
        )

        result = response.choices[0].message.content
        tokens = response.usage.completion_tokens

        print(f"\n{task_type.upper()} (max_tokens={config.max_tokens})")
        print(f"  Input: {text[:50]}...")
        print(f"  Output: {result}")
        print(f"  Tokens: {tokens}")

# endregion


if __name__ == "__main__":
    # run real OpenAI demos
    demo_real_openai_comparison()
    demo_multiple_tasks()

    print("\n" + "=" * 70)
    print("  Key insight: max_tokens is the EASIEST cost optimization")
    print("  Same answers, fewer tokens, lower bills")
    print("=" * 70)
