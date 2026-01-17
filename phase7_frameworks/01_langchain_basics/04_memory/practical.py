"""
Memory - Hands-On Practice (Requires API Keys)

Demonstrates modern memory integration with OpenAI/Anthropic using LangChain's
RunnableWithMessageHistory pattern (LangChain 1.0+ compatible).

Run: uv run python -m phase7_frameworks.01_langchain_basics.04_memory.practical
"""

from inspect import cleandoc

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import Field

from phase7_frameworks.utils import (
    check_api_keys,
    print_section,
    requires_both_keys,
    requires_openai,
)


# region Demo 1: Basic Conversation with Buffer Memory


@requires_openai
def demo_buffer_memory_conversation() -> None:
    """
    demonstrate basic conversation with full history using modern API

    Buffer Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       Buffer Memory: Full Conversation History Storage      │
    │                                                             │
    │  Turn 1:                                                    │
    │     User: "Hi, I'm learning about LangChain memory"         │
    │           │                                                 │
    │           ▼                                                 │
    │     ┌──────────────────────────────────┐                    │
    │     │ Memory Store (session: user1)    │                    │
    │     │ Messages: []  (empty)            │                    │
    │     └──────────┬───────────────────────┘                    │
    │                │                                            │
    │                ▼                                            │
    │     Prompt: System + History (empty) + User input           │
    │                │                                            │
    │                ▼                                            │
    │     LLM: "Great! LangChain memory helps..."                 │
    │                │                                            │
    │                ▼                                            │
    │     Memory Store Updates:                                   │
    │     [HumanMessage("Hi, I'm learning..."),                   │
    │      AIMessage("Great! LangChain memory...")]               │
    │                                                             │
    │  Turn 2:                                                    │
    │     User: "What did I say I was learning about?"            │
    │           │                                                 │
    │           ▼                                                 │
    │     ┌──────────────────────────────────┐                    │
    │     │ Memory Store (session: user1)    │                    │
    │     │ Messages: [2 from turn 1]        │                    │
    │     └──────────┬───────────────────────┘                    │
    │                │                                            │
    │                ▼                                            │
    │     Prompt: System + History (turn 1) + New input           │
    │                │                                            │
    │                ▼                                            │
    │     LLM: "You said LangChain memory" (knows context!)       │
    │                                                             │
    │  Memory Growth Over Time:                                   │
    │     Turn 1: 2 messages                                      │
    │     Turn 2: 4 messages                                      │
    │     Turn 3: 6 messages                                      │
    │     Turn N: 2*N messages                                    │
    │     → Unlimited growth (no pruning)                         │
    │                                                             │
    │  RunnableWithMessageHistory Integration:                    │
    │     chain_with_memory = RunnableWithMessageHistory(         │
    │         chain,                                              │
    │         get_session_history,  # callback to get history     │
    │         input_messages_key="input",                         │
    │         history_messages_key="chat_history"                 │
    │     )                                                       │
    │                                                             │
    │  ✅ Benefit: Perfect recall (all context retained)          │
    │  ✅ Benefit: Simple implementation                          │
    │  ⚠️  Caution: Unbounded token growth over time              │
    │  ⚠️  Caution: May exceed context limits in long chats       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 1: Buffer Memory with RunnableWithMessageHistory")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    # create prompt with history placeholder
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # build chain
    chain = prompt | llm | StrOutputParser()

    # session store (simulates buffer memory)
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # wrap chain with memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("## Multi-turn conversation:\n")

    # turn 1
    response = chain_with_memory.invoke(
        {"input": "Hi, I'm learning about LangChain memory"},
        config={"configurable": {"session_id": "user1"}},
    )
    print(f"User: Hi, I'm learning about LangChain memory")
    print(f"Bot:  {response}\n")

    # turn 2 (bot remembers context)
    response = chain_with_memory.invoke(
        {"input": "What did I say I was learning about?"},
        config={"configurable": {"session_id": "user1"}},
    )
    print(f"User: What did I say I was learning about?")
    print(f"Bot:  {response}\n")

    # inspect memory
    print("## Memory Contents:")
    print(f"Messages stored: {len(store['user1'].messages)}")

    print("\n✓ RunnableWithMessageHistory enables context-aware conversations")


# endregion

# region Demo 2: Window Memory for Recent Context


@requires_openai
def demo_window_memory_conversation() -> None:
    """
    demonstrate sliding window memory using custom session history

    Window Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │      Window Memory: Sliding Window with Fixed Size (k=2)    │
    │                                                             │
    │  Turn 1: "My name is Alice"                                 │
    │     Memory: [H:"Alice", A:"Nice to meet you"]               │
    │     Window size: 2 messages (1 exchange)                    │
    │                                                             │
    │  Turn 2: "I live in New York"                               │
    │     Memory: [H:"Alice", A:"Nice...", H:"New York", A:"..."] │
    │     Window size: 4 messages (2 exchanges) ← MAX             │
    │                                                             │
    │  Turn 3: "I like Python programming"                        │
    │     Before: [H:"Alice", A:"Nice...", H:"NY", A:"..."]       │
    │     After:  [H:"NY", A:"...", H:"Python", A:"Great!"]       │
    │             ↑ Oldest exchange dropped to maintain k=2       │
    │                                                             │
    │  Turn 4: "What's my name?"                                  │
    │     Memory has: [H:"NY", H:"Python", H:"What's my name?"]   │
    │     LLM response: "I don't recall" ← Name outside window!   │
    │                                                             │
    │  Window Mechanism:                                          │
    │     ┌─────────────────────────────────┐                     │
    │     │ k=2 → Keep last 2 exchanges     │                     │
    │     │     → 2 * 2 = 4 messages max    │                     │
    │     └─────────────────────────────────┘                     │
    │                                                             │
    │     When messages > k*2:                                    │
    │         messages = messages[-(k*2):]  # Keep last k*2       │
    │                                                             │
    │  Sliding Window Visualization:                              │
    │     [Old Old Old New New] → Window moves right →            │
    │      ↑ Drop    ↑ Keep                                       │
    │                                                             │
    │  Implementation:                                            │
    │     class WindowChatMessageHistory(ChatMessageHistory):     │
    │         k: int = 2  # exchanges to keep                     │
    │                                                             │
    │         def add_message(self, message):                     │
    │             super().add_message(message)                    │
    │             if len(self.messages) > self.k * 2:             │
    │                 self.messages = self.messages[-(self.k*2):] │
    │                                                             │
    │  ✅ Benefit: Bounded memory (no token explosion)            │
    │  ✅ Benefit: Focus on recent context                        │
    │  ⚠️  Caution: Forgets older information                     │
    │  ⚠️  Caution: May lose important early context              │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 2: Window Memory Pattern")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # custom window-based history
    class WindowChatMessageHistory(ChatMessageHistory):
        """chat history with sliding window"""

        k: int = Field(default=2, description="keep last k exchanges (k*2 messages)")

        def add_message(self, message) -> None:
            super().add_message(message)
            # keep only last k*2 messages (k exchanges)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k * 2) :]

    store: dict[str, WindowChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowChatMessageHistory(k=2)
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("## Window size: k=2 (last 2 exchanges)\n")

    conversations = [
        "My name is Alice",
        "I live in New York",
        "I like Python programming",
        "What's my name?",  # should remember (within window)
        "Where do I live?",  # might not remember (outside window)
    ]

    for i, user_msg in enumerate(conversations, 1):
        response = chain_with_memory.invoke(
            {"input": user_msg}, config={"configurable": {"session_id": "user1"}}
        )
        print(f"[{i}] User: {user_msg}")
        print(f"    Bot:  {response}\n")

    print(f"✓ Window memory keeps last 2 exchanges in history")


# endregion

# region Demo 3: Summary Memory Pattern


@requires_openai
def demo_summary_memory_pattern() -> None:
    """
    demonstrate summary-based memory using LLM compression

    Summary Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │    Summary Memory: LLM-Based Compression for Long Chats     │
    │                                                             │
    │  Turn 1-3: Normal conversation (buffer mode)                │
    │     Memory: [H:"Alice", A:"Nice...", H:"NY", A:"...",       │
    │              H:"Python", A:"Great!"]                        │
    │     Total: 6 messages (3 exchanges)                         │
    │                                                             │
    │  Turn 4: Trigger threshold (>6 messages) → Summarize        │
    │     ┌─────────────────────────────────────────┐             │
    │     │ LLM Summary Request:                    │             │
    │     │ "Summarize this conversation:           │             │
    │     │  Human: My name is Alice                │             │
    │     │  AI: Nice to meet you, Alice!           │             │
    │     │  Human: I live in New York              │             │
    │     │  AI: New York is a great city!          │             │
    │     │  Human: I love Python programming       │             │
    │     │  AI: Python is great for AI!"           │             │
    │     └─────────────┬───────────────────────────┘             │
    │                   ▼                                         │
    │     Summary: "Alice from NY, loves Python"                  │
    │                   │                                         │
    │                   ▼                                         │
    │     Memory: [Summary] + [Recent 2 exchanges]                │
    │           = 1 summary + 4 messages (compressed!)            │
    │                                                             │
    │  Compression Workflow:                                      │
    │     1. Detect: len(messages) > threshold                    │
    │     2. Summarize: Send old messages to LLM                  │
    │     3. Compress: Replace old messages with summary          │
    │     4. Retain: Keep recent messages for context             │
    │     5. Continue: New messages added to compressed base      │
    │                                                             │
    │  Memory Growth Over Time:                                   │
    │     Turns 1-3:  6 messages  (normal)                        │
    │     Turn 4:     5 messages  (1 summary + 4 recent)          │
    │     Turns 5-6:  7 messages  (1 summary + 6 recent)          │
    │     Turn 7:     5 messages  (2 summaries + 4 recent)        │
    │     Pattern:    Periodic compression, bounded growth        │
    │                                                             │
    │  ✅ Benefit: Maintains context over very long conversations │
    │  ✅ Benefit: Bounded token usage (controlled growth)        │
    │  ✅ Benefit: Preserves key facts in compressed form         │
    │  ⚠️  Caution: Uses LLM tokens for summarization             │
    │  ⚠️  Caution: May lose nuance/details in compression        │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 3: Summary Memory Pattern")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    # summary-generating chain
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the conversation history into key facts about the user",
            ),
            ("human", "{history}"),
        ]
    )

    summary_chain = summary_prompt | llm | StrOutputParser()

    # conversation chain
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Here's what you know:\n{summary}"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    print("## Building conversation with summary compression:\n")

    # manual summary management for demonstration
    conversation_log = []
    summary = "No previous context"

    exchanges = [
        "I'm a Python developer",
        "I work on AI applications",
        "I use LangChain for chatbots",
    ]

    for msg in exchanges:
        # get response
        response = chain.invoke({"summary": summary, "input": msg})
        print(f"User: {msg}")
        print(f"Bot:  {response[:50]}...\n")

        # log conversation
        conversation_log.append(f"User: {msg}\nBot: {response}")

        # update summary
        if len(conversation_log) >= 2:
            history_text = "\n\n".join(conversation_log)
            summary = summary_chain.invoke({"history": history_text})

    print("## Generated Summary:")
    print(f"{summary}\n")

    # test summary works
    response = chain.invoke({"summary": summary, "input": "What do I work on?"})
    print(f"User: What do I work on?")
    print(f"Bot:  {response}")

    print("\n✓ Summary pattern compresses history while preserving key information")


# endregion

# region Demo 4: Adaptive Memory Pattern


@requires_openai
def demo_adaptive_memory_pattern() -> None:
    """
    demonstrate adaptive memory that switches strategies

    Adaptive Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │   Adaptive Memory: Keep First 2 + Recent 4 (max_messages=6) │
    │                                                             │
    │  Turn 1: "I'm Alice, a software engineer"                   │
    │     Memory: [H:"Alice engineer", A:"Nice..."]               │
    │     Status: 2 messages (below threshold)                    │
    │                                                             │
    │  Turn 2: "I specialize in machine learning"                 │
    │     Memory: [H:"Alice engineer", A:"Nice...",               │
    │              H:"ML specialization", A:"Great..."]           │
    │     Status: 4 messages (below threshold)                    │
    │                                                             │
    │  Turn 3: "I've worked with TensorFlow and PyTorch"          │
    │     Memory: [H:"Alice engineer", A:"Nice...",               │
    │              H:"ML specialization", A:"Great...",           │
    │              H:"TF & PyTorch", A:"Impressive..."]           │
    │     Status: 6 messages (at threshold)                       │
    │                                                             │
    │  Turn 4: "Recently I've been exploring LangChain"           │
    │     Before: [M1, M2, M3, M4, M5, M6] (6 messages)           │
    │     Trigger: len(messages) > 6 → ADAPTIVE TRIM              │
    │             │                                               │
    │             ▼                                               │
    │     ┌────────────────────────────────────┐                  │
    │     │ Adaptive Logic:                    │                  │
    │     │ Keep: messages[:2] (first 2)       │                  │
    │     │     + messages[-(6-2):] (last 4)   │                  │
    │     │                                    │                  │
    │     │ = M1, M2 (important context)       │                  │
    │     │ + M5, M6, M7, M8 (recent context)  │                  │
    │     └────────────────────────────────────┘                  │
    │             │                                               │
    │             ▼                                               │
    │     After: [H:"Alice engineer", A:"Nice...",  ← First 2     │
    │             H:"TF & PyTorch", A:"Impressive...",            │
    │             H:"LangChain", A:"Great!"]  ← Recent 4          │
    │     Status: 6 messages (trimmed middle, kept ends)          │
    │                                                             │
    │  Why This Strategy?                                         │
    │     • First messages: Often contain important setup         │
    │       (names, goals, context that shouldn't be lost)        │
    │     • Recent messages: Current topic needs fresh context    │
    │     • Middle messages: Transitional, less critical          │
    │                                                             │
    │  Implementation:                                            │
    │     self.messages = (                                       │
    │         self.messages[:2] +  # keep first 2                 │
    │         self.messages[-(max_messages - 2):]  # keep last 4  │
    │     )                                                       │
    │                                                             │
    │  ✅ Benefit: Retains initial context (names, setup)         │
    │  ✅ Benefit: Maintains recent conversation flow             │
    │  ✅ Benefit: Bounded memory (never exceeds max_messages)    │
    │  ⚠️  Caution: Loses middle conversation details             │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 4: Adaptive Memory Pattern")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    class AdaptiveChatMessageHistory(ChatMessageHistory):
        """chat history that adapts based on size"""

        max_messages: int = Field(
            default=6, description="maximum messages before trimming"
        )

        def add_message(self, message) -> None:
            super().add_message(message)
            # trim if exceeding limit
            if len(self.messages) > self.max_messages:
                # keep first 2 and last (max_messages - 2)
                self.messages = (
                    self.messages[:2] + self.messages[-(self.max_messages - 2) :]
                )

    store: dict[str, AdaptiveChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = AdaptiveChatMessageHistory(max_messages=6)
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("## Adaptive memory (keeps first 2 + recent 4 messages):\n")

    exchanges = [
        "I'm Alice, a software engineer",
        "I specialize in machine learning",
        "I've worked with TensorFlow and PyTorch",
        "Recently I've been exploring LangChain",
        "What's my background?",
    ]

    for i, msg in enumerate(exchanges, 1):
        response = chain_with_memory.invoke(
            {"input": msg}, config={"configurable": {"session_id": "user1"}}
        )
        print(f"[{i}] User: {msg}")
        print(f"    Bot:  {response[:60]}...\n")

    print(f"✓ Adaptive memory keeps important context while managing size")


# endregion

# region Demo 5: Multi-Session Memory


@requires_openai
def demo_multi_session_memory() -> None:
    """
    demonstrate separate memory per user/session

    Multi-Session Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │     Multi-Session: Separate Memory Stores Per User          │
    │                                                             │
    │  Shared Memory Store (Dictionary):                          │
    │     store = {                                               │
    │         "alice": ChatMessageHistory(),                      │
    │         "bob": ChatMessageHistory(),                        │
    │         "charlie": ChatMessageHistory(),                    │
    │     }                                                       │
    │                                                             │
    │  Alice's Conversation (session_id="alice"):                 │
    │     [Turn 1] User: "My name is Alice"                       │
    │             ↓                                               │
    │     ┌────────────────────────────┐                          │
    │     │ store["alice"].messages:  │                           │
    │     │ [H:"My name is Alice",    │                           │
    │     │  A:"Nice to meet you!"]   │                           │
    │     └────────────────────────────┘                          │
    │                                                             │
    │  Bob's Conversation (session_id="bob"):                     │
    │     [Turn 1] User: "My name is Bob"                         │
    │             ↓                                               │
    │     ┌────────────────────────────┐                          │
    │     │ store["bob"].messages:    │                           │
    │     │ [H:"My name is Bob",      │                           │
    │     │  A:"Hello Bob!"]          │                           │
    │     └────────────────────────────┘                          │
    │                                                             │
    │  Complete Isolation:                                        │
    │     When Alice asks: "What's my name?"                      │
    │         → Looks up store["alice"]                           │
    │         → Sees H:"My name is Alice"                         │
    │         → Responds: "Your name is Alice"                    │
    │                                                             │
    │     When Bob asks: "What's my name?"                        │
    │         → Looks up store["bob"]                             │
    │         → Sees H:"My name is Bob"                           │
    │         → Responds: "Your name is Bob"                      │
    │                                                             │
    │  Session ID Routing:                                        │
    │     config={"configurable": {"session_id": "alice"}}        │
    │         ↓                                                   │
    │     get_session_history("alice")                            │
    │         ↓                                                   │
    │     return store["alice"]                                   │
    │                                                             │
    │  Use Cases:                                                 │
    │     • Multi-user chat applications                          │
    │     • Customer support systems (one chat per customer)      │
    │     • Educational apps (one context per student)            │
    │     • Personal assistants (user-specific memory)            │
    │                                                             │
    │  ✅ Benefit: Complete conversation isolation per user       │
    │  ✅ Benefit: Scalable (add users dynamically)               │
    │  ✅ Benefit: Simple session management pattern              │
    │  ⚠️  Caution: Memory grows with number of active sessions   │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 5: Multi-Session Memory Management")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # shared store across all sessions
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("## Separate memory per user:\n")

    # alice's conversation
    response = chain_with_memory.invoke(
        {"input": "My name is Alice"}, config={"configurable": {"session_id": "alice"}}
    )
    print(f"[Alice] User: My name is Alice")
    print(f"[Alice] Bot:  {response}\n")

    # bob's conversation
    response = chain_with_memory.invoke(
        {"input": "My name is Bob"}, config={"configurable": {"session_id": "bob"}}
    )
    print(f"[Bob] User: My name is Bob")
    print(f"[Bob] Bot:  {response}\n")

    # verify separation
    response = chain_with_memory.invoke(
        {"input": "What's my name?"},
        config={"configurable": {"session_id": "alice"}},
    )
    print(f"[Alice] User: What's my name?")
    print(f"[Alice] Bot:  {response}\n")

    print(f"## Session Store:")
    print(f"  Total sessions: {len(store)}")
    print(f"  Alice messages: {len(store['alice'].messages)}")
    print(f"  Bob messages: {len(store['bob'].messages)}")

    print("\n✓ Each session maintains separate conversation context")


# endregion

# region Demo 6: Custom System Prompt with Memory


@requires_openai
def demo_memory_with_custom_prompt() -> None:
    """
    demonstrate memory with customized system prompts

    Custom System Prompt with Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │   Memory + Custom Persona: Python Expert with History       │
    │                                                             │
    │  Prompt Template Structure:                                 │
    │     ┌─────────────────────────────────────────┐             │
    │     │ 1. System Message (Persona):            │             │
    │     │    "You are a Python expert who gives   │             │
    │     │     concise, practical advice"          │             │
    │     │                                         │             │
    │     │ 2. Chat History (MessagesPlaceholder):  │             │
    │     │    [H:"What are list comprehensions?",  │             │
    │     │     A:"Concise syntax for lists...",    │             │
    │     │     H:"Give me an example",             │             │
    │     │     A:"[x*2 for x in range(5)]"]        │             │
    │     │                                         │             │
    │     │ 3. Current User Input:                  │             │
    │     │    "Can I use conditions?"              │             │
    │     └─────────────────────────────────────────┘             │
    │                                                             │
    │  Combined Prompt Sent to LLM:                               │
    │     System: "You are a Python expert..."                    │
    │     Human: "What are list comprehensions?"                  │
    │     AI: "Concise syntax for creating lists..."              │
    │     Human: "Give me an example"                             │
    │     AI: "[x*2 for x in range(5)]"                           │
    │     Human: "Can I use conditions?"                          │
    │         ↓                                                   │
    │     LLM processes WITH persona + full history               │
    │         ↓                                                   │
    │     AI: "Yes! Use: [x for x in range(10) if x%2==0]"        │
    │                                                             │
    │  Persona Examples:                                          │
    │     • "You are a Python expert..." → Technical advice       │
    │     • "You are a friendly tutor..." → Patient explanations  │
    │     • "You are a code reviewer..." → Critical analysis      │
    │     • "You are a DevOps engineer..." → Deployment focus     │
    │                                                             │
    │  Why Combine Memory + Custom Prompts?                       │
    │     1. Consistency: Persona maintained across conversation  │
    │     2. Context: History informs specialized responses       │
    │     3. Quality: Domain expertise + conversation flow        │
    │                                                             │
    │  Implementation:                                            │
    │     prompt = ChatPromptTemplate.from_messages([             │
    │         ("system", custom_persona),                         │
    │         MessagesPlaceholder(variable_name="chat_history"),  │
    │         ("human", "{input}"),                               │
    │     ])                                                      │
    │                                                             │
    │  ✅ Benefit: Specialized assistant with conversation memory │
    │  ✅ Benefit: Consistent persona across entire conversation  │
    │  ✅ Benefit: Context-aware expert responses                 │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 6: Memory with Custom System Message")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    # custom expert persona
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Python expert who gives concise, practical advice",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("## Custom system message: Python expert persona\n")

    # conversation
    response = chain_with_memory.invoke(
        {"input": "What are list comprehensions?"},
        config={"configurable": {"session_id": "user1"}},
    )
    print(f"User: What are list comprehensions?")
    print(f"Bot:  {response}\n")

    response = chain_with_memory.invoke(
        {"input": "Give me an example"},
        config={"configurable": {"session_id": "user1"}},
    )
    print(f"User: Give me an example")
    print(f"Bot:  {response}")

    print("\n✓ Memory works with custom prompts and system messages")


# endregion

# region Demo 7: Multi-Provider Memory


@requires_both_keys
def demo_multi_provider_memory() -> None:
    """demonstrate memory portability across providers

    Multi-Provider Memory Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │   Provider-Agnostic Memory: OpenAI ⇄ Anthropic Shared Store │
    │                                                             │
    │  Shared Memory Store (Session: "shared"):                   │
    │     store = {                                               │
    │         "shared": ChatMessageHistory()                      │
    │     }                                                       │
    │     ↓ Both providers access same memory ↓                   │
    │                                                             │
    │  Turn 1 - OpenAI Chain:                                     │
    │     User: "I prefer Python over Java"                       │
    │         ↓                                                   │
    │     ┌─────────────────────────────────────┐                 │
    │     │ OpenAI Chain (GPT-3.5):             │                 │
    │     │ Prompt: System + History + Input    │                 │
    │     │ Response: "Great choice! Python..." │                 │
    │     └────────────┬────────────────────────┘                 │
    │                  ▼                                          │
    │     store["shared"].messages:                               │
    │     [H:"I prefer Python over Java",                         │
    │      A:"Great choice! Python is..."]                        │
    │                                                             │
    │  Turn 2 - Anthropic Chain (Same Memory!):                   │
    │     User: "What language do I prefer?"                      │
    │         ↓                                                   │
    │     ┌────────────────────────────────────┐                  │
    │     │ Anthropic Chain (Claude):          │                  │
    │     │ Prompt: System + History + Input   │                  │
    │     │ History includes OpenAI exchange!  │                  │
    │     │ [H:"Python over Java", A:"Great"]  │                  │
    │     │ Response: "You prefer Python"      │                  │
    │     └────────────────────────────────────┘                  │
    │                                                             │
    │  Key Insight: Memory is LLM-Independent                     │
    │     ┌───────────────────────────────────┐                   │
    │     │ Memory Store (Just Messages):     │                   │
    │     │ • HumanMessage("Python > Java")   │                   │
    │     │ • AIMessage("Great choice!")      │                   │
    │     │ • HumanMessage("What language?")  │                   │
    │     │ • AIMessage("You prefer Python")  │                   │
    │     │                                   │                   │
    │     │ ✓ No provider-specific data       │                   │
    │     │ ✓ Standard message format         │                   │
    │     │ ✓ Works with ANY LLM              │                   │
    │     └───────────────────────────────────┘                   │
    │                                                             │
    │  Use Cases:                                                 │
    │     • Fallback patterns: OpenAI → Claude if unavailable     │
    │     • A/B testing: Same conversation, different models      │
    │     • Cost optimization: Switch based on query complexity   │
    │     • Hybrid systems: Combine strengths of multiple LLMs    │
    │                                                             │
    │  Implementation Pattern:                                    │
    │     # Same get_session_history for all chains               │
    │     openai_with_memory = RunnableWithMessageHistory(        │
    │         openai_chain, get_session_history, ...)             │
    │     anthropic_with_memory = RunnableWithMessageHistory(     │
    │         anthropic_chain, get_session_history, ...)          │
    │     # Both access store["shared"]                           │
    │                                                             │
    │  ✅ Benefit: Switch providers without losing context        │
    │  ✅ Benefit: Single memory for hybrid LLM systems           │
    │  ✅ Benefit: Seamless provider fallback capability          │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 7: Memory Across Multiple Providers")

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # shared memory store
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    print("## Shared memory across OpenAI and Anthropic:\n")

    # conversation with OpenAI
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    openai_chain = prompt | openai_llm | StrOutputParser()
    openai_with_memory = RunnableWithMessageHistory(
        openai_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = openai_with_memory.invoke(
        {"input": "I prefer Python over Java"},
        config={"configurable": {"session_id": "shared"}},
    )
    print(f"[OpenAI] User: I prefer Python over Java")
    print(f"[OpenAI] Bot:  {response}\n")

    # switch to Anthropic with same memory
    anthropic_llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022", temperature=0.7, max_tokens=100
    )
    anthropic_chain = prompt | anthropic_llm | StrOutputParser()
    anthropic_with_memory = RunnableWithMessageHistory(
        anthropic_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = anthropic_with_memory.invoke(
        {"input": "What language do I prefer?"},
        config={"configurable": {"session_id": "shared"}},
    )
    print(f"[Anthropic] User: What language do I prefer?")
    print(f"[Anthropic] Bot:  {response}")

    print("\n✓ Memory is provider-agnostic and works across different LLMs")


# endregion

# region Demo 8: Memory Persistence Pattern


@requires_openai
def demo_memory_persistence() -> None:
    """demonstrate memory save/load pattern

    Memory Persistence Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │   Memory Persistence: Save → Exit → Reload → Resume         │
    │                                                             │
    │  Session 1: Build Conversation                              │
    │     User: "My project is about AI agents"                   │
    │         ↓                                                   │
    │     LLM: "Fascinating! AI agents can..."                    │
    │         ↓                                                   │
    │     Memory: [H:"My project...", A:"Fascinating..."]         │
    │                                                             │
    │  Save to Disk (JSON):                                       │
    │     ┌──────────────────────────────────────┐                │
    │     │ Serialize Messages:                  │                │
    │     │ messages_data = [                    │                │
    │     │   {                                  │                │
    │     │     "type": "human",                 │                │
    │     │     "content": "My project is..."    │                │
    │     │   },                                 │                │
    │     │   {                                  │                │
    │     │     "type": "ai",                    │                │
    │     │     "content": "Fascinating..."      │                │
    │     │   }                                  │                │
    │     │ ]                                    │                │
    │     │                                      │                │
    │     │ json.dump(messages_data, file)       │                │
    │     └──────────────┬───────────────────────┘                │
    │                    ▼                                        │
    │     File: /tmp/memory_demo.json                             │
    │     [{"type":"human","content":"My project..."},            │
    │      {"type":"ai","content":"Fascinating..."}]              │
    │                                                             │
    │  Exit Application → Time passes → Restart                   │
    │                                                             │
    │  Session 2: Load Conversation                               │
    │     Load from disk:                                         │
    │         with open("/tmp/memory_demo.json") as f:            │
    │             loaded_data = json.load(f)                      │
    │                                                             │
    │     Reconstruct Messages:                                   │
    │         new_history = ChatMessageHistory()                  │
    │         for msg_data in loaded_data:                        │
    │             if msg_data["type"] == "human":                 │
    │                 new_history.add(HumanMessage(...))          │
    │             else:                                           │
    │                 new_history.add(AIMessage(...))             │
    │                                                             │
    │     Update Store:                                           │
    │         store["user1"] = new_history                        │
    │                                                             │
    │  Resume Conversation:                                       │
    │     User: "What is my project about?"                       │
    │         ↓                                                   │
    │     Memory: [H:"My project...", A:"Fascinating..."]         │
    │             ↑ Restored from disk!                           │
    │         ↓                                                   │
    │     LLM: "Your project is about AI agents"                  │
    │     ✓ Conversation context fully restored                   │
    │                                                             │
    │  Production Patterns:                                       │
    │     • Redis: session_id → serialized messages               │
    │     • PostgreSQL: user_id + session_id → messages table     │
    │     • MongoDB: {user_id, session_id, messages[]}            │
    │     • S3: session files for cold storage                    │
    │                                                             │
    │  ✅ Benefit: Survive application restarts                   │
    │  ✅ Benefit: Long-term conversation storage                 │
    │  ✅ Benefit: Share conversations across devices             │
    │  ⚠️  Caution: Need serialization/deserialization logic      │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("Demo 8: Memory Persistence Pattern")

    import json
    from pathlib import Path

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("## Session 1: Build conversation\n")

    # build conversation
    response = chain_with_memory.invoke(
        {"input": "My project is about AI agents"},
        config={"configurable": {"session_id": "user1"}},
    )
    print(f"User: My project is about AI agents")
    print(f"Bot:  {response[:60]}...\n")

    # save memory
    temp_file = Path("/tmp/memory_demo.json")
    messages_data = [
        {"type": msg.type, "content": msg.content}
        for msg in store["user1"].messages
    ]

    with open(temp_file, "w") as f:
        json.dump(messages_data, f)

    print(f"✓ Memory saved to {temp_file}")
    print(f"  Messages: {len(messages_data)}\n")

    # simulate session 2 - clear and reload
    print("## Session 2: Load conversation\n")

    # load memory
    with open(temp_file, "r") as f:
        loaded_data = json.load(f)

    new_history = ChatMessageHistory()
    for msg_data in loaded_data:
        if msg_data["type"] == "human":
            new_history.add_message(HumanMessage(content=msg_data["content"]))
        else:
            new_history.add_message(AIMessage(content=msg_data["content"]))

    # create new store with loaded history
    store["user1"] = new_history

    # test restored memory
    response = chain_with_memory.invoke(
        {"input": "What is my project about?"},
        config={"configurable": {"session_id": "user1"}},
    )
    print(f"User: What is my project about?")
    print(f"Bot:  {response}")

    # cleanup
    temp_file.unlink()

    print("\n✓ Memory can be persisted and restored across sessions")


# endregion


def main() -> None:
    """run all practical demos"""
    print(
        cleandoc(
            """
        Memory - Practical Demos (Modern API)

        This module demonstrates memory integration using RunnableWithMessageHistory,
        the modern LangChain 1.0+ pattern that replaces deprecated ConversationChain.

        Ensure you have API keys set in .env file.
    """
        )
    )

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

    demo_buffer_memory_conversation()
    demo_window_memory_conversation()
    demo_summary_memory_pattern()
    demo_adaptive_memory_pattern()
    demo_multi_session_memory()
    demo_memory_with_custom_prompt()
    demo_multi_provider_memory()
    demo_memory_persistence()

    print("\n" + "=" * 70)
    print("  Practical demos complete! You've mastered modern memory patterns.")
    print("=" * 70)


if __name__ == "__main__":
    main()
