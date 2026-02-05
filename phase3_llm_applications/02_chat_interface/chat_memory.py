"""
Chat Memory Module - Conversation memory management for multi-turn chats

This module implements different strategies for managing conversation history:
1. Full history - keep everything (simple but hits token limits)
2. Sliding window - keep last N messages
3. Token budget - keep messages within token limit
4. Summarization - compress old messages into summary

Module Structure:
- schemas/        → Role enum, Message dataclass
- chat_memory.py → ChatMemory, SummarizingMemory (this file)
- streaming.py   → Streaming utilities
- engine.py      → ChatEngine orchestrator
- examples.py    → Demo functions

Memory Flow:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   Add    │───►│  Check   │───►│  Trim    │───►│  Return  │
│ Message  │    │  Limit   │    │ if needed│    │ messages │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

Run with: uv run python phase3_llm_applications/02_chat_interface/chat_memory.py
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from schemas import Role, Message


@dataclass
class ChatMemory:
    """
    conversation memory manager

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      ChatMemory                             │
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │   system    │  │  messages   │  │     strategy        │  │
    │  │   prompt    │  │    list     │  │  (window/budget)    │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    │         │                │                   │              │
    │         └────────────────┼───────────────────┘              │
    │                          ▼                                  │
    │                  ┌─────────────┐                            │
    │                  │ get_messages│ → API-ready format         │
    │                  └─────────────┘                            │
    └─────────────────────────────────────────────────────────────┘
    """

    system_prompt: str = "You are a helpful assistant."
    max_messages: int = 20  # for sliding window
    max_tokens: int = 4000  # for token budget
    strategy: Literal["full", "sliding_window", "token_budget"] = "sliding_window"
    messages: list[Message] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """add a user message to history"""
        self.messages.append(Message(role=Role.USER, content=content))
        self._apply_strategy()

    def add_assistant_message(self, content: str) -> None:
        """add an assistant message to history"""
        self.messages.append(Message(role=Role.ASSISTANT, content=content))
        self._apply_strategy()

    def _apply_strategy(self) -> None:
        """apply memory management strategy"""
        if self.strategy == "sliding_window":
            self._apply_sliding_window()
        elif self.strategy == "token_budget":
            self._apply_token_budget()
        # "full" strategy keeps everything

    def _apply_sliding_window(self) -> None:
        """
        keep only the last N messages

        Before: [u1, a1, u2, a2, u3, a3, u4, a4, u5, a5]
        After:  [u3, a3, u4, a4, u5, a5]  (if max_messages=6)
        """
        if len(self.messages) > self.max_messages:
            # keep the most recent messages
            self.messages = self.messages[-self.max_messages:]

    def _apply_token_budget(self) -> None:
        """
        remove oldest messages until under token budget

        Token counting:
        ┌─────────────────────────────────────────────────────────┐
        │  Approximate: 1 token ≈ 4 characters (for English)      │
        │  More accurate: use tiktoken library                    │
        └─────────────────────────────────────────────────────────┘
        """
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 1:
            # remove oldest message (preserve at least one for context)
            self.messages.pop(0)

    def _estimate_tokens(self) -> int:
        """
        estimate total tokens in conversation

        Simple estimation: ~4 chars per token
        For production, use tiktoken for accurate counting
        """
        total_chars = len(self.system_prompt)
        for msg in self.messages:
            total_chars += len(msg.content)
        return total_chars // 4

    def get_messages(self) -> list[dict]:
        """
        get messages in API format

        Returns:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """
        result = [{"role": "system", "content": self.system_prompt}]
        for msg in self.messages:
            result.append(msg.to_dict())
        return result

    def clear(self) -> None:
        """clear conversation history (keep system prompt)"""
        self.messages = []

    def get_turn_count(self) -> int:
        """get number of conversation turns (user-assistant pairs)"""
        return len([m for m in self.messages if m.role == Role.USER])

    def get_last_message(self) -> Optional[Message]:
        """get the most recent message"""
        return self.messages[-1] if self.messages else None

    def __len__(self) -> int:
        return len(self.messages)


# ─────────────────────────────────────────────────────────────
# ADVANCED: SUMMARIZATION MEMORY
# ─────────────────────────────────────────────────────────────


@dataclass
class SummarizingMemory(ChatMemory):
    """
    memory that summarizes old conversations

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   SummarizingMemory                         │
    │                                                             │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │                    messages                           │  │
    │  │  [old_summary] [recent messages...]                   │  │
    │  │       ↑                                               │  │
    │  │       │                                               │  │
    │  │   When messages exceed threshold,                     │  │
    │  │   old ones are summarized into this                   │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """

    summary: str = ""
    summarize_threshold: int = 10  # summarize when this many messages

    def _apply_strategy(self) -> None:
        """summarize old messages when threshold reached"""
        if len(self.messages) > self.summarize_threshold:
            self._summarize_old_messages()

    def _summarize_old_messages(self) -> None:
        """
        compress old messages into summary

        Note: In production, you'd use an LLM to generate the summary.
        This is a simplified placeholder.
        """
        # keep last 4 messages as recent context
        messages_to_summarize = self.messages[:-4]
        recent_messages = self.messages[-4:]

        # simple summary (in production, use LLM)
        topics = []
        for msg in messages_to_summarize:
            if msg.role == Role.USER:
                # extract key topics (simplified)
                words = msg.content.split()[:5]
                topics.append(" ".join(words))

        new_summary = f"Previous discussion covered: {'; '.join(topics)}"
        if self.summary:
            new_summary = f"{self.summary} | {new_summary}"

        self.summary = new_summary
        self.messages = recent_messages

    def get_messages(self) -> list[dict]:
        """get messages with summary prepended"""
        result = [{"role": "system", "content": self.system_prompt}]

        # add summary as context if exists
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[Conversation summary: {self.summary}]"
            })

        for msg in self.messages:
            result.append(msg.to_dict())

        return result


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# region Demo Functions

def demo_basic_memory():
    """demonstrate basic chat memory"""
    print_section("Basic Chat Memory")

    memory = ChatMemory(
        system_prompt="You are a helpful coding assistant.",
        strategy="full"
    )

    # simulate conversation
    memory.add_user_message("What is Python?")
    memory.add_assistant_message("Python is a high-level programming language known for its readability.")

    memory.add_user_message("What are its main features?")
    memory.add_assistant_message("Python's main features include dynamic typing, automatic memory management, and extensive libraries.")

    print(f"\nConversation turns: {memory.get_turn_count()}")
    print(f"Total messages: {len(memory)}")
    print(f"Estimated tokens: {memory._estimate_tokens()}")

    print("\n--- Messages for API ---")
    for msg in memory.get_messages():
        role = msg["role"]
        content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        print(f"  [{role}]: {content}")


def demo_sliding_window():
    """demonstrate sliding window strategy"""
    print_section("Sliding Window Strategy")

    memory = ChatMemory(
        system_prompt="You are a helpful assistant.",
        max_messages=4,  # keep only last 4 messages
        strategy="sliding_window"
    )

    # add many messages
    for i in range(1, 6):
        memory.add_user_message(f"Question {i}: What is topic {i}?")
        memory.add_assistant_message(f"Answer {i}: Topic {i} is about...")

    print(f"\nAdded 5 turns, keeping last {memory.max_messages} messages")
    print(f"Actual messages in memory: {len(memory)}")

    print("\n--- Remaining messages ---")
    for msg in memory.messages:
        print(f"  {msg}")


def demo_token_budget():
    """demonstrate token budget strategy"""
    print_section("Token Budget Strategy")

    memory = ChatMemory(
        system_prompt="You are a helpful assistant.",
        max_tokens=200,  # very low for demo
        strategy="token_budget"
    )

    # add messages until budget exceeded
    long_question = "Can you explain " + "something " * 20 + "?"
    long_answer = "Sure, " + "explanation " * 30 + "."

    memory.add_user_message(long_question)
    print(f"After user message: {memory._estimate_tokens()} tokens")

    memory.add_assistant_message(long_answer)
    print(f"After assistant message: {memory._estimate_tokens()} tokens")

    memory.add_user_message("Another question " * 10)
    print(f"After another message: {memory._estimate_tokens()} tokens")

    print(f"\nMessages remaining: {len(memory)}")
    print(f"Final token estimate: {memory._estimate_tokens()}")


def demo_summarizing_memory():
    """demonstrate summarizing memory"""
    print_section("Summarizing Memory")

    memory = SummarizingMemory(
        system_prompt="You are a helpful assistant.",
        summarize_threshold=6
    )

    # add messages to trigger summarization
    topics = ["Python basics", "data structures", "functions", "classes"]
    for topic in topics:
        memory.add_user_message(f"Tell me about {topic}")
        memory.add_assistant_message(f"Here's info about {topic}...")

    print(f"\nAdded {len(topics)} turns")
    print(f"Summary created: {memory.summary}")
    print(f"Recent messages kept: {len(memory.messages)}")


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Chat Memory Module - Conversation Memory Management")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Basic Memory", "full conversation history storage"),
        ("2", "Sliding Window", "keep only last N messages"),
        ("3", "Token Budget", "remove old messages within token limit"),
        ("4", "Summarizing Memory", "compress old messages into summary"),
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
        '1': ('Basic Memory', demo_basic_memory),
        '2': ('Sliding Window', demo_sliding_window),
        '3': ('Token Budget', demo_token_budget),
        '4': ('Summarizing Memory', demo_summarizing_memory),
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
