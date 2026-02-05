"""
Streaming Module - Real-time response streaming for chat interfaces

This module provides utilities for streaming LLM responses:
1. OpenAI streaming
2. Anthropic streaming
3. Stream collection and processing
4. Typing effect simulation

Streaming Flow:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   API    │───►│  Stream  │───►│  Chunk   │───►│  Display │
│  Request │    │ Response │    │ Process  │    │  (live)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │
                     ▼
              Token by token!

Run with: uv run python phase3_llm_applications/02_chat_interface/streaming.py
"""

import os
import sys
import time
from typing import Generator, Optional, Callable
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class StreamChunk:
    """single chunk from streaming response"""
    content: str
    is_final: bool = False
    metadata: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# OPENAI STREAMING
# ─────────────────────────────────────────────────────────────


def stream_openai(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Generator[StreamChunk, None, None]:
    """
    stream response from OpenAI

    Usage:
        for chunk in stream_openai(messages):
            print(chunk.content, end="", flush=True)

    Streaming architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    OpenAI Streaming                          │
    │                                                              │
    │  client.chat.completions.create(stream=True)                │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │  Chunk 1: {"content": "The"}                        │   │
    │  │  Chunk 2: {"content": " answer"}                    │   │
    │  │  Chunk 3: {"content": " is"}                        │   │
    │  │  Chunk 4: {"content": "..."}                        │   │
    │  │  Chunk N: {"content": null, "finish_reason": "stop"}│   │
    │  └─────────────────────────────────────────────────────┘   │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # simulate streaming for demo without API key
        yield from _simulate_stream("OpenAI API key not configured. This is simulated streaming.")
        return

    try:
        from openai import OpenAI

        client = OpenAI()
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield StreamChunk(content=delta.content)

            # check for finish
            if chunk.choices[0].finish_reason:
                yield StreamChunk(
                    content="",
                    is_final=True,
                    metadata={"finish_reason": chunk.choices[0].finish_reason}
                )

    except Exception as e:
        yield StreamChunk(content=f"\n[Error: {e}]", is_final=True)


# ─────────────────────────────────────────────────────────────
# ANTHROPIC STREAMING
# ─────────────────────────────────────────────────────────────


def stream_anthropic(
    messages: list[dict],
    model: str = "claude-sonnet-4-20250514",
    system: str = "You are a helpful assistant.",
    max_tokens: int = 500,
) -> Generator[StreamChunk, None, None]:
    """
    stream response from Anthropic

    Note: Anthropic uses different message format
    - system prompt is separate parameter
    - messages don't include system role

    Anthropic streaming structure:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Anthropic Streaming                        │
    │                                                              │
    │  Event types:                                                │
    │  • message_start - beginning of response                    │
    │  • content_block_start - start of text block                │
    │  • content_block_delta - actual text content                │
    │  • content_block_stop - end of text block                   │
    │  • message_stop - end of response                           │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield from _simulate_stream("Anthropic API key not configured. This is simulated streaming.")
        return

    try:
        from anthropic import Anthropic

        client = Anthropic()

        # filter out system messages (Anthropic handles separately)
        api_messages = [m for m in messages if m["role"] != "system"]

        with client.messages.stream(
            model=model,
            system=system,
            messages=api_messages,
            max_tokens=max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield StreamChunk(content=text)

        yield StreamChunk(content="", is_final=True)

    except Exception as e:
        yield StreamChunk(content=f"\n[Error: {e}]", is_final=True)


# ─────────────────────────────────────────────────────────────
# STREAM UTILITIES
# ─────────────────────────────────────────────────────────────


def _simulate_stream(text: str) -> Generator[StreamChunk, None, None]:
    """simulate streaming for demo/testing"""
    words = text.split()
    for i, word in enumerate(words):
        yield StreamChunk(content=word + " ")
        time.sleep(0.05)  # simulate network delay
    yield StreamChunk(content="", is_final=True)


def collect_stream(stream: Generator[StreamChunk, None, None]) -> str:
    """
    collect streaming response into single string

    Useful when you need the complete response after streaming display
    """
    parts = []
    for chunk in stream:
        parts.append(chunk.content)
    return "".join(parts)


def stream_to_console(
    stream: Generator[StreamChunk, None, None],
    typing_effect: bool = False,
    typing_delay: float = 0.02,
) -> str:
    """
    stream response to console with optional typing effect

    Args:
        stream: streaming generator
        typing_effect: if True, add delay between characters
        typing_delay: seconds between characters (if typing_effect=True)

    Returns:
        complete response text
    """
    full_response = []

    for chunk in stream:
        if chunk.content:
            if typing_effect:
                for char in chunk.content:
                    print(char, end="", flush=True)
                    time.sleep(typing_delay)
            else:
                print(chunk.content, end="", flush=True)
            full_response.append(chunk.content)

    print()  # newline at end
    return "".join(full_response)


class StreamPrinter:
    """
    configurable stream printer with callbacks

    Usage:
        printer = StreamPrinter(on_chunk=my_callback)
        response = printer.print(stream)
    """

    def __init__(
        self,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        prefix: str = "",
        suffix: str = "\n",
    ):
        self.on_chunk = on_chunk
        self.on_complete = on_complete
        self.prefix = prefix
        self.suffix = suffix

    def print(self, stream: Generator[StreamChunk, None, None]) -> str:
        """print stream and return complete response"""
        if self.prefix:
            print(self.prefix, end="", flush=True)

        full_response = []

        for chunk in stream:
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response.append(chunk.content)

                if self.on_chunk:
                    self.on_chunk(chunk.content)

        print(self.suffix, end="")

        complete = "".join(full_response)
        if self.on_complete:
            self.on_complete(complete)

        return complete


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# region Demo Functions

def demo_simulated_streaming():
    """demonstrate streaming with simulated data"""
    print_section("Simulated Streaming")

    text = "This is a demonstration of streaming responses. Each word appears one at a time, simulating how LLM responses are streamed from the API."

    print("\n   Streaming response:")
    print("   ", end="")
    response = stream_to_console(_simulate_stream(text))


def demo_typing_effect():
    """demonstrate typing effect"""
    print_section("Typing Effect")

    text = "Watch each character appear like someone is typing..."

    print("\n   With typing effect:")
    print("   ", end="")
    stream_to_console(_simulate_stream(text), typing_effect=True, typing_delay=0.03)


def demo_stream_collection():
    """demonstrate collecting stream into string"""
    print_section("Stream Collection")

    text = "This stream will be collected into a single string."
    stream = _simulate_stream(text)

    # collect without printing
    collected = collect_stream(stream)

    print(f"\n   Collected text: '{collected.strip()}'")
    print(f"   Length: {len(collected)} characters")


def demo_stream_printer():
    """demonstrate StreamPrinter class"""
    print_section("StreamPrinter with Callbacks")

    chunks_received = []

    def on_chunk(text: str):
        chunks_received.append(text)

    def on_complete(full_text: str):
        print(f"\n   [Callback: received {len(chunks_received)} chunks]")

    printer = StreamPrinter(
        on_chunk=on_chunk,
        on_complete=on_complete,
        prefix="   Response: ",
        suffix="\n"
    )

    text = "Streaming with callbacks enabled for processing chunks."
    printer.print(_simulate_stream(text))


def demo_openai_streaming():
    """demonstrate actual OpenAI streaming (requires API key)"""
    print_section("OpenAI Streaming (Live API)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is Python in one sentence?"},
    ]

    print("\n   Question: What is Python in one sentence?")
    print("   Answer: ", end="")
    stream_to_console(stream_openai(messages, max_tokens=100))


def demo_anthropic_streaming():
    """demonstrate actual Anthropic streaming (requires API key)"""
    print_section("Anthropic Streaming (Live API)")

    messages = [
        {"role": "user", "content": "What is Python in one sentence?"},
    ]

    print("\n   Question: What is Python in one sentence?")
    print("   Answer: ", end="")
    stream_to_console(stream_anthropic(
        messages,
        system="You are a helpful assistant. Be concise.",
        max_tokens=100
    ))


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Streaming Module - Real-time Response Streaming")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Simulated Streaming", "word-by-word streaming simulation"),
        ("2", "Typing Effect", "character-by-character typing animation"),
        ("3", "Stream Collection", "collecting stream into single string"),
        ("4", "Stream Printer", "streaming with callbacks"),
        ("5", "OpenAI Streaming", "live OpenAI API streaming"),
        ("6", "Anthropic Streaming", "live Anthropic API streaming"),
    ]

    for num, name, desc in demos:
        print(f"   [{num}] {name}")
        print(f"      {desc}")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 70)


def run_selected_demos(selections: str) -> bool:
    """run selected demos based on user input"""
    selections = selections.strip().lower()

    if selections == 'q':
        return False

    demo_map = {
        '1': ('Simulated Streaming', demo_simulated_streaming),
        '2': ('Typing Effect', demo_typing_effect),
        '3': ('Stream Collection', demo_stream_collection),
        '4': ('Stream Printer', demo_stream_printer),
        '5': ('OpenAI Streaming', demo_openai_streaming),
        '6': ('Anthropic Streaming', demo_anthropic_streaming),
    }

    if selections == 'a':
        demos_to_run = list(demo_map.keys())
    else:
        demos_to_run = [s.strip() for s in selections.replace(',', ' ').split() if s.strip() in demo_map]

    if not demos_to_run:
        print("\n⚠️  invalid selection. please enter demo numbers, 'a' for all, or 'q' to quit")
        return True

    print(f"\n🚀 Running {len(demos_to_run)} demo(s)...\n")

    for demo_num in demos_to_run:
        name, func = demo_map[demo_num]
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n⚠️  demo interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ error in demo: {e}")
            continue

    print("\n✅ selected demos complete!")
    return True


def main():
    """run demonstrations with interactive menu"""
    while True:
        show_menu()

        try:
            selection = input("\n🎯 select demo(s) (e.g., '1', '1,3', or 'a' for all): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 goodbye!")
            break

        if not run_selected_demos(selection):
            print("\n👋 goodbye!")
            break

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 goodbye!")
            break

# endregion


if __name__ == "__main__":
    main()
