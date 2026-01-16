"""
Memory - Conceptual Understanding (No API Keys Required)

Demonstrates LangChain memory patterns using mock data and structural examples.
Focus on understanding memory types, integration patterns, and trade-offs.

Run: uv run python -m phase7_frameworks.01_langchain_basics.04_memory.concepts
"""

from inspect import cleandoc

from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langchain_core.messages import AIMessage, HumanMessage

from phase7_frameworks.utils import print_section


# region Demo 1: ConversationBufferMemory Basics


def demo_buffer_memory() -> None:
    """demonstrate basic buffer memory storing full conversation"""
    print_section("Demo 1: ConversationBufferMemory - Full History")

    memory = ConversationBufferMemory()

    print("## Simulating conversation:\n")

    # simulate 3 exchanges
    exchanges = [
        ("Hi, I'm Alice", "Hello Alice! How can I help you?"),
        ("What's the weather?", "I don't have access to weather data"),
        ("What's my name?", "Your name is Alice"),
    ]

    for user_msg, bot_msg in exchanges:
        memory.save_context({"input": user_msg}, {"output": bot_msg})
        print(f"User: {user_msg}")
        print(f"Bot:  {bot_msg}\n")

    # load memory
    print("## Memory Contents:")
    memory_vars = memory.load_memory_variables({})
    print(memory_vars["history"])

    print("\n✓ Buffer memory stores complete conversation history")


# endregion

# region Demo 2: ConversationBufferWindowMemory


def demo_window_memory() -> None:
    """demonstrate sliding window memory"""
    print_section("Demo 2: ConversationBufferWindowMemory - Sliding Window")

    memory = ConversationBufferWindowMemory(k=2)  # keep last 2 exchanges

    print("## Window size: k=2 (last 2 exchanges)\n")

    exchanges = [
        ("Message 1", "Response 1"),
        ("Message 2", "Response 2"),
        ("Message 3", "Response 3"),
        ("Message 4", "Response 4"),
    ]

    for i, (user_msg, bot_msg) in enumerate(exchanges, 1):
        memory.save_context({"input": user_msg}, {"output": bot_msg})
        print(f"Added exchange {i}:")
        print(f"  User: {user_msg}")
        print(f"  Bot:  {bot_msg}")

        # show current memory
        memory_vars = memory.load_memory_variables({})
        history = memory_vars["history"]
        num_lines = len([line for line in history.split("\n") if line.strip()])
        print(f"  Memory size: {num_lines} messages\n")

    print("## Final Memory (last 2 exchanges only):")
    print(memory.load_memory_variables({})["history"])

    print("\n✓ Window memory discards old messages, keeping only recent context")


# endregion

# region Demo 3: Memory with Message Objects


def demo_return_messages() -> None:
    """demonstrate returning messages as objects vs strings"""
    print_section("Demo 3: Memory Return Formats")

    # string format (default)
    print("## String Format (default):\n")
    memory_str = ConversationBufferMemory()
    memory_str.save_context({"input": "Hi"}, {"output": "Hello"})
    memory_str.save_context({"input": "Bye"}, {"output": "Goodbye"})

    result = memory_str.load_memory_variables({})
    print(f"Type: {type(result['history'])}")
    print(f"Content:\n{result['history']}\n")

    # message objects format
    print("## Message Objects Format:\n")
    memory_msgs = ConversationBufferMemory(return_messages=True)
    memory_msgs.save_context({"input": "Hi"}, {"output": "Hello"})
    memory_msgs.save_context({"input": "Bye"}, {"output": "Goodbye"})

    result = memory_msgs.load_memory_variables({})
    print(f"Type: {type(result['history'])}")
    print(f"Content: {result['history']}")

    print("\n✓ Memory can return string format or message objects")


# endregion

# region Demo 4: Custom Memory Keys


def demo_custom_keys() -> None:
    """demonstrate custom input/output/memory keys"""
    print_section("Demo 4: Custom Memory Keys")

    memory = ConversationBufferMemory(
        input_key="user_input",  # custom key for user messages
        output_key="bot_response",  # custom key for bot messages
        memory_key="chat_history",  # custom key for stored history
    )

    print("## Custom keys configured:")
    print("  input_key: 'user_input'")
    print("  output_key: 'bot_response'")
    print("  memory_key: 'chat_history'\n")

    # use custom keys
    memory.save_context(
        {"user_input": "What is Python?"},
        {"bot_response": "Python is a programming language"},
    )

    # retrieve with custom key
    result = memory.load_memory_variables({})
    print(f"Memory key: {list(result.keys())}")
    print(f"Content:\n{result['chat_history']}")

    print("\n✓ Custom keys allow flexible integration with different chain formats")


# endregion

# region Demo 5: Memory Token Counting


def demo_token_counting() -> None:
    """demonstrate token usage across memory types"""
    print_section("Demo 5: Token Usage Comparison")

    # simulate conversation
    exchanges = [
        ("Tell me about Python", "Python is a high-level programming language..."),
        ("What about Java?", "Java is a statically-typed language..."),
        ("Which is better?", "Both have strengths..."),
        ("For web development?", "Python with Django/Flask is popular..."),
        ("What about mobile?", "Java/Kotlin for Android, Swift for iOS..."),
    ]

    print("## Simulated conversation (5 exchanges):\n")

    # buffer memory (all messages)
    buffer_memory = ConversationBufferMemory()
    for user_msg, bot_msg in exchanges:
        buffer_memory.save_context({"input": user_msg}, {"output": bot_msg})

    buffer_history = buffer_memory.load_memory_variables({})["history"]
    buffer_chars = len(buffer_history)

    print(f"Buffer Memory:")
    print(f"  Messages: 10 (5 exchanges)")
    print(f"  Characters: {buffer_chars}")
    print(f"  Approximate tokens: ~{buffer_chars // 4}\n")

    # window memory (last 2 exchanges)
    window_memory = ConversationBufferWindowMemory(k=2)
    for user_msg, bot_msg in exchanges:
        window_memory.save_context({"input": user_msg}, {"output": bot_msg})

    window_history = window_memory.load_memory_variables({})["history"]
    window_chars = len(window_history)

    print(f"Window Memory (k=2):")
    print(f"  Messages: 4 (2 exchanges)")
    print(f"  Characters: {window_chars}")
    print(f"  Approximate tokens: ~{window_chars // 4}")
    print(f"  Savings: {100 - (window_chars / buffer_chars * 100):.1f}%\n")

    print("✓ Window memory significantly reduces token usage")


# endregion

# region Demo 6: Multi-User Memory Pattern


def demo_multi_user_memory() -> None:
    """demonstrate separate memory per user"""
    print_section("Demo 6: Multi-User Memory Pattern")

    # storage for user memories
    user_memories: dict[str, ConversationBufferMemory] = {}

    def get_memory(user_id: str) -> ConversationBufferMemory:
        """get or create memory for user"""
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory()
        return user_memories[user_id]

    print("## Simulating multi-user conversations:\n")

    # user 1 conversation
    alice_memory = get_memory("alice")
    alice_memory.save_context({"input": "My name is Alice"}, {"output": "Hello Alice!"})
    alice_memory.save_context(
        {"input": "I like Python"}, {"output": "Python is great!"}
    )

    print("Alice's memory:")
    print(f"  {alice_memory.load_memory_variables({})['history']}\n")

    # user 2 conversation
    bob_memory = get_memory("bob")
    bob_memory.save_context({"input": "My name is Bob"}, {"output": "Hello Bob!"})
    bob_memory.save_context({"input": "I like Java"}, {"output": "Java is powerful!"})

    print("Bob's memory:")
    print(f"  {bob_memory.load_memory_variables({})['history']}\n")

    # verify separation
    print("## Memory separation verified:")
    print(f"  Total users: {len(user_memories)}")
    print(f"  Alice messages: {len(alice_memory.chat_memory.messages)}")
    print(f"  Bob messages: {len(bob_memory.chat_memory.messages)}")

    print("\n✓ Each user maintains separate conversation context")


# endregion

# region Demo 7: Memory Clearing


def demo_memory_clearing() -> None:
    """demonstrate memory clearing strategies"""
    print_section("Demo 7: Memory Clearing Strategies")

    memory = ConversationBufferMemory()

    # add some messages
    memory.save_context({"input": "First message"}, {"output": "First response"})
    memory.save_context({"input": "Second message"}, {"output": "Second response"})

    print("## Before clearing:")
    print(f"  Messages: {len(memory.chat_memory.messages)}")
    print(f"  Content: {memory.load_memory_variables({})['history'][:50]}...\n")

    # clear memory
    memory.clear()

    print("## After memory.clear():")
    print(f"  Messages: {len(memory.chat_memory.messages)}")
    print(f"  Content: '{memory.load_memory_variables({})['history']}'\n")

    print("✓ Memory can be cleared to start fresh conversations")


# endregion

# region Demo 8: Memory Inspection


def demo_memory_inspection() -> None:
    """demonstrate memory debugging and inspection"""
    print_section("Demo 8: Memory Inspection and Debugging")

    memory = ConversationBufferMemory(return_messages=True)

    # add messages
    exchanges = [
        ("What is AI?", "AI is artificial intelligence"),
        ("Examples?", "Chatbots, image recognition, etc"),
    ]

    for user_msg, bot_msg in exchanges:
        memory.save_context({"input": user_msg}, {"output": bot_msg})

    print("## Inspection Methods:\n")

    # method 1: load_memory_variables
    print("1. load_memory_variables():")
    memory_vars = memory.load_memory_variables({})
    print(f"   Keys: {list(memory_vars.keys())}")
    print(f"   Messages count: {len(memory_vars['history'])}\n")

    # method 2: direct chat_memory access
    print("2. chat_memory.messages:")
    for i, msg in enumerate(memory.chat_memory.messages):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"   [{i}] {msg_type}: {msg.content[:40]}...")

    print("\n✓ Multiple ways to inspect memory state for debugging")


# endregion


def main() -> None:
    """run all conceptual demos"""
    print(
        cleandoc(
            """
        Memory - Conceptual Demos

        Understanding LangChain memory systems without requiring API keys.
        Focus on memory types, patterns, and trade-offs.
    """
        )
    )

    demo_buffer_memory()
    demo_window_memory()
    demo_return_messages()
    demo_custom_keys()
    demo_token_counting()
    demo_multi_user_memory()
    demo_memory_clearing()
    demo_memory_inspection()

    print("\n" + "=" * 70)
    print("  Conceptual demos complete! You understand memory fundamentals.")
    print("=" * 70)


if __name__ == "__main__":
    main()
