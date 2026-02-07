"""
Memory - Conceptual Understanding (No API Keys Required)

Demonstrates LangChain memory patterns using mock data and structural examples.
Focus on understanding memory types, integration patterns, and trade-offs.

Run: uv run python -m phase7_frameworks.01_langchain_basics.04_memory.concepts
"""

from inspect import cleandoc

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from common.demo_menu import Demo, MenuRunner
from common.util.utils import print_section


# region Demo 1: InMemoryChatMessageHistory Basics


def demo_buffer_memory() -> None:
    """demonstrate basic buffer memory storing full conversation"""
    print_section("Demo 1: InMemoryChatMessageHistory - Full History")

    # modern approach: use InMemoryChatMessageHistory
    history = InMemoryChatMessageHistory()

    print("## Simulating conversation:\n")

    # simulate 3 exchanges
    exchanges = [
        ("Hi, I'm Alice", "Hello Alice! How can I help you?"),
        ("What's the weather?", "I don't have access to weather data"),
        ("What's my name?", "Your name is Alice"),
    ]

    for user_msg, bot_msg in exchanges:
        history.add_user_message(user_msg)
        history.add_ai_message(bot_msg)
        print(f"User: {user_msg}")
        print(f"Bot:  {bot_msg}\n")

    # load memory - format messages as string
    print("## Memory Contents:")
    for msg in history.messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")

    print("\n✓ Chat history stores complete conversation history")


# endregion

# region Demo 2: Chat History with Sliding Window


def demo_window_memory() -> None:
    """demonstrate sliding window memory"""
    print_section("Demo 2: Chat History with Sliding Window")

    history = InMemoryChatMessageHistory()
    window_size = 4  # keep last 4 messages (2 exchanges)

    print(f"## Window size: {window_size} messages (2 exchanges)\n")

    exchanges = [
        ("Message 1", "Response 1"),
        ("Message 2", "Response 2"),
        ("Message 3", "Response 3"),
        ("Message 4", "Response 4"),
    ]

    for i, (user_msg, bot_msg) in enumerate(exchanges, 1):
        history.add_user_message(user_msg)
        history.add_ai_message(bot_msg)
        print(f"Added exchange {i}:")
        print(f"  User: {user_msg}")
        print(f"  Bot:  {bot_msg}")

        # show current window (last k messages)
        windowed_messages = history.messages[-window_size:]
        print(f"  Window size: {len(windowed_messages)} messages\n")

    print("## Final Memory (last 2 exchanges only):")
    windowed_messages = history.messages[-window_size:]
    for msg in windowed_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")

    print("\n✓ Window pattern keeps only recent messages, discarding old context")


# endregion

# region Demo 3: Memory with Message Objects


def demo_return_messages() -> None:
    """demonstrate message objects vs string formatting"""
    print_section("Demo 3: Message Formats")

    history = InMemoryChatMessageHistory()
    history.add_user_message("Hi")
    history.add_ai_message("Hello")
    history.add_user_message("Bye")
    history.add_ai_message("Goodbye")

    # message objects (always available)
    print("## Message Objects (native format):\n")
    print(f"Type: {type(history.messages)}")
    print(f"Content: {history.messages}\n")

    # string format (manual formatting)
    print("## String Format (formatted):\n")
    formatted = "\n".join([
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in history.messages
    ])
    print(f"Type: {type(formatted)}")
    print(f"Content:\n{formatted}")

    print("\n✓ Chat history stores message objects, format as needed")


# endregion

# region Demo 4: Custom Memory Keys


def demo_custom_keys() -> None:
    """demonstrate message metadata for custom attributes"""
    print_section("Demo 4: Message Metadata")

    history = InMemoryChatMessageHistory()

    print("## Adding messages with metadata:\n")

    # add messages with custom metadata
    history.add_message(
        HumanMessage(content="What is Python?", additional_kwargs={"user_id": "alice"})
    )
    history.add_message(
        AIMessage(
            content="Python is a programming language",
            additional_kwargs={"confidence": 0.95},
        )
    )

    print("Messages with metadata:")
    for msg in history.messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        metadata = msg.additional_kwargs
        print(f"  {role}: {msg.content}")
        print(f"    Metadata: {metadata}\n")

    print("✓ Messages can include custom metadata for tracking and filtering")


# endregion

# region Demo 5: Memory Token Counting


def demo_token_counting() -> None:
    """demonstrate token usage with full vs windowed history"""
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

    # full history
    full_history = InMemoryChatMessageHistory()
    for user_msg, bot_msg in exchanges:
        full_history.add_user_message(user_msg)
        full_history.add_ai_message(bot_msg)

    # calculate size
    full_content = "\n".join([msg.content for msg in full_history.messages])
    full_chars = len(full_content)

    print(f"Full History:")
    print(f"  Messages: {len(full_history.messages)} (5 exchanges)")
    print(f"  Characters: {full_chars}")
    print(f"  Approximate tokens: ~{full_chars // 4}\n")

    # windowed history (last 2 exchanges = 4 messages)
    window_size = 4
    windowed_messages = full_history.messages[-window_size:]
    windowed_content = "\n".join([msg.content for msg in windowed_messages])
    windowed_chars = len(windowed_content)

    print(f"Windowed History (last {window_size} messages):")
    print(f"  Messages: {len(windowed_messages)} (2 exchanges)")
    print(f"  Characters: {windowed_chars}")
    print(f"  Approximate tokens: ~{windowed_chars // 4}")
    print(f"  Savings: {100 - (windowed_chars / full_chars * 100):.1f}%\n")

    print("✓ Windowing significantly reduces token usage for long conversations")


# endregion

# region Demo 6: Multi-User Memory Pattern


def demo_multi_user_memory() -> None:
    """demonstrate separate memory per user"""
    print_section("Demo 6: Multi-User Memory Pattern")

    # storage for user chat histories
    user_histories: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(user_id: str) -> InMemoryChatMessageHistory:
        """get or create chat history for user"""
        if user_id not in user_histories:
            user_histories[user_id] = InMemoryChatMessageHistory()
        return user_histories[user_id]

    print("## Simulating multi-user conversations:\n")

    # user 1 conversation
    alice_history = get_history("alice")
    alice_history.add_user_message("My name is Alice")
    alice_history.add_ai_message("Hello Alice!")
    alice_history.add_user_message("I like Python")
    alice_history.add_ai_message("Python is great!")

    print("Alice's history:")
    for msg in alice_history.messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")
    print()

    # user 2 conversation
    bob_history = get_history("bob")
    bob_history.add_user_message("My name is Bob")
    bob_history.add_ai_message("Hello Bob!")
    bob_history.add_user_message("I like Java")
    bob_history.add_ai_message("Java is powerful!")

    print("Bob's history:")
    for msg in bob_history.messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")
    print()

    # verify separation
    print("## Memory separation verified:")
    print(f"  Total users: {len(user_histories)}")
    print(f"  Alice messages: {len(alice_history.messages)}")
    print(f"  Bob messages: {len(bob_history.messages)}")

    print("\n✓ Each user maintains separate conversation context")


# endregion

# region Demo 7: Memory Clearing


def demo_memory_clearing() -> None:
    """demonstrate clearing chat history"""
    print_section("Demo 7: Clearing Chat History")

    history = InMemoryChatMessageHistory()

    # add some messages
    history.add_user_message("First message")
    history.add_ai_message("First response")
    history.add_user_message("Second message")
    history.add_ai_message("Second response")

    print("## Before clearing:")
    print(f"  Messages: {len(history.messages)}")
    print(f"  First message: {history.messages[0].content}\n")

    # clear history
    history.clear()

    print("## After history.clear():")
    print(f"  Messages: {len(history.messages)}")
    print(f"  Content: {history.messages}\n")

    print("✓ Chat history can be cleared to start fresh conversations")


# endregion

# region Demo 8: Memory Inspection


def demo_memory_inspection() -> None:
    """demonstrate chat history debugging and inspection"""
    print_section("Demo 8: Chat History Inspection and Debugging")

    history = InMemoryChatMessageHistory()

    # add messages
    exchanges = [
        ("What is AI?", "AI is artificial intelligence"),
        ("Examples?", "Chatbots, image recognition, etc"),
    ]

    for user_msg, bot_msg in exchanges:
        history.add_user_message(user_msg)
        history.add_ai_message(bot_msg)

    print("## Inspection Methods:\n")

    # method 1: messages property
    print("1. history.messages:")
    print(f"   Type: {type(history.messages)}")
    print(f"   Count: {len(history.messages)}\n")

    # method 2: iterate and inspect
    print("2. Iterate and inspect messages:")
    for i, msg in enumerate(history.messages):
        msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"   [{i}] {msg_type}: {msg.content[:40]}...")
        print(f"       Type: {type(msg).__name__}")
        print(f"       Has metadata: {bool(msg.additional_kwargs)}")

    print("\n✓ Direct access to message objects for debugging and inspection")


# endregion


# region Demo Menu Configuration

DEMOS = [
    Demo("1", "Buffer Memory", "complete conversation history", demo_buffer_memory),
    Demo("2", "Window Memory", "sliding window with recent messages", demo_window_memory),
    Demo("3", "Return Messages", "message objects vs strings", demo_return_messages),
    Demo("4", "Message Metadata", "custom attributes and tracking", demo_custom_keys),
    Demo("5", "Token Counting", "memory size management", demo_token_counting),
    Demo("6", "Multi-User Memory", "per-user conversation isolation", demo_multi_user_memory),
    Demo("7", "Memory Clearing", "reset and cleanup operations", demo_memory_clearing),
    Demo("8", "Memory Inspection", "debugging and introspection", demo_memory_inspection),
]

# endregion


def main() -> None:
    """run demonstrations with interactive menu"""
    runner = MenuRunner(
        DEMOS,
        title="Memory - Conceptual Examples",
        subtitle="No API key required - demonstrates patterns only"
    )
    runner.run()

    print("\n" + "=" * 70)
    print("  Thanks for exploring LangChain memory!")
    print("  You understand memory types, patterns, and trade-offs")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
