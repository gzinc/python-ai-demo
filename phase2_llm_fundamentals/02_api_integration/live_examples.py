"""
Module: Live API Integration - Real LLM Calls

This module makes actual API calls to OpenAI and/or Anthropic.
Requires API keys set as environment variables.

Setup:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."

Run with: uv run python phase2_llm_fundamentals/02_api_integration/live_examples.py
"""

import os
import sys
from inspect import cleandoc
from dotenv import load_dotenv

# load .env file (must be called before checking keys)
load_dotenv()


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check_openai_key() -> bool:
    """check if OpenAI key is available"""
    return bool(os.environ.get("OPENAI_API_KEY"))


def check_anthropic_key() -> bool:
    """check if Anthropic key is available"""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def run_openai_examples() -> None:
    """run OpenAI API examples"""
    print_section("OpenAI Examples")

    try:
        from openai import OpenAI

        client = OpenAI()

        # basic completion
        print("\n1. Basic Completion:")
        print("-" * 40)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # cheaper for testing
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": "What is Python in one sentence?"},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
        print(f"Total: {response.usage.total_tokens} tokens")

        # streaming example
        print("\n2. Streaming Response:")
        print("-" * 40)

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."},
            ],
            temperature=0.0,
            stream=True,
        )

        print("Streaming: ", end="")
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print()  # newline after stream

        # code generation
        print("\n3. Code Generation:")
        print("-" * 40)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python expert. Write clean, working code."},
                {"role": "user", "content": "Write a function to reverse a string. Just the code, no explanation."},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        print(response.choices[0].message.content)

    except Exception as e:
        print(f"OpenAI error: {e}")


def run_anthropic_examples() -> None:
    """run Anthropic API examples"""
    print_section("Anthropic Examples")

    try:
        from anthropic import Anthropic

        client = Anthropic()

        # basic completion
        print("\n1. Basic Completion:")
        print("-" * 40)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a helpful assistant. Be concise.",
            messages=[
                {"role": "user", "content": "What is Python in one sentence?"},
            ],
        )

        print(f"Response: {response.content[0].text}")
        print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

        # streaming example
        print("\n2. Streaming Response:")
        print("-" * 40)

        print("Streaming: ", end="")
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."},
            ],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print()

        # code generation
        print("\n3. Code Generation:")
        print("-" * 40)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a Python expert. Write clean, working code.",
            messages=[
                {"role": "user", "content": "Write a function to reverse a string. Just the code, no explanation."},
            ],
        )

        print(response.content[0].text)

    except Exception as e:
        print(f"Anthropic error: {e}")


def interactive_chat() -> None:
    """simple interactive chat loop"""
    print_section("Interactive Chat")

    # determine which provider to use
    if check_openai_key():
        provider = "openai"
        from openai import OpenAI
        client = OpenAI()
        print("Using OpenAI (gpt-4o-mini)")
    elif check_anthropic_key():
        provider = "anthropic"
        from anthropic import Anthropic
        client = Anthropic()
        print("Using Anthropic (claude-sonnet-4-20250514)")
    else:
        print("No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    print("Type 'quit' to exit\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})

            if provider == "openai":
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=history,
                    temperature=0.7,
                    max_tokens=500,
                )
                assistant_message = response.choices[0].message.content
                tokens = response.usage.total_tokens
            else:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=history,
                )
                assistant_message = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens

            history.append({"role": "assistant", "content": assistant_message})

            print(f"Assistant: {assistant_message}")
            print(f"  [{tokens} tokens used]\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def show_menu(has_openai: bool = True, has_anthropic: bool = True) -> None:
    """display interactive menu of available demos"""
    print("\n" + "=" * 60)
    print("  Live API Integration Examples")
    print("=" * 60)
    print("\nAvailable demos:")
    if has_openai:
        print("  1. OpenAI examples (basic, streaming, code generation)")
    if has_anthropic:
        print("  2. Anthropic examples (basic, streaming, code generation)")
    if has_openai or has_anthropic:
        print("  3. interactive chat session")
    print("\n  [a] Run all demos")
    print("  [q] Quit")
    print("=" * 60)


def run_selected_demos(selections: str, has_openai: bool = True, has_anthropic: bool = True) -> bool:
    """run selected demos based on user input"""
    demo_map = {}
    if has_openai:
        demo_map["1"] = ("OpenAI examples", run_openai_examples)
    if has_anthropic:
        demo_map["2"] = ("Anthropic examples", run_anthropic_examples)
    if has_openai or has_anthropic:
        demo_map["3"] = ("interactive chat", interactive_chat)

    if selections.lower() == "q":
        return False

    if selections.lower() == "a":
        print("\n🚀 Running all demos...\n")
        for name, func in demo_map.values():
            func()
        print("\n✅ All demos completed!")
        print_section("Key Observations")
        print(cleandoc("""
            - Response times vary (network latency + model inference)
            - Streaming feels more responsive
            - Token counts affect cost
            - Both providers have similar capabilities
        """))
        return True

    # parse comma-separated selections
    selected = [s.strip() for s in selections.split(",")]
    valid_selections = [s for s in selected if s in demo_map]

    if not valid_selections:
        valid_range = f"1-{len(demo_map)}"
        print(f"❌ Invalid selection. Please enter numbers ({valid_range}), 'a' for all, or 'q' to quit.")
        return True

    for selection in valid_selections:
        name, func = demo_map[selection]
        print(f"\n▶️  Running: {name}")
        func()

    print("\n✅ Selected demos completed!")
    return True


def main() -> None:
    """interactive demo runner"""
    # check for keys
    has_openai = check_openai_key()
    has_anthropic = check_anthropic_key()

    print("\n" + "=" * 60)
    print("  Live API Integration Examples")
    print("=" * 60)
    print("\nAPI Key Status:")
    print(f"  OpenAI:    {'✅ Found' if has_openai else '❌ Not set'}")
    print(f"  Anthropic: {'✅ Found' if has_anthropic else '❌ Not set'}")

    if not has_openai and not has_anthropic:
        print("\n⚠️  No API keys found!")
        print("Set environment variables:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("\nRun examples.py instead for pattern demonstrations.")
        sys.exit(1)

    while True:
        show_menu(has_openai, has_anthropic)
        choice = input("\nSelect demos (e.g., '1', '1,2', or 'a' for all): ").strip()
        should_continue = run_selected_demos(choice, has_openai, has_anthropic)
        if not should_continue:
            print("\n👋 Goodbye! Next: Move to Module 3 - Embeddings!\n")
            break

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye! Next: Move to Module 3 - Embeddings!\n")
            break


if __name__ == "__main__":
    main()
