# Memory Module

**Purpose**: Learn LangChain's memory systems for building conversational AI with context retention

---

## Learning Objectives

By the end of this module, you will:
1. **Buffer Memory**: Store and retrieve full conversation history
2. **Window Memory**: Maintain sliding windows of recent messages
3. **Summary Memory**: Use LLM-generated conversation summaries
4. **Memory Integration**: Connect memory to chains for conversational workflows
5. **Memory Comparison**: Understand trade-offs between memory types

---

## Key Concepts

### What is Memory?

**Memory** allows LLMs to maintain context across multiple interactions:

```
User: "What's the weather?"
Bot:  "It's sunny today"
User: "What about tomorrow?"  ← Needs context from previous exchange
Bot:  "Tomorrow will be rainy"
```

**Without Memory**: Each message is independent, bot loses context
**With Memory**: Bot remembers conversation history and provides coherent responses

---

## Memory Types

### 1. ConversationBufferMemory

**Stores**: Complete conversation history (all messages)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Hi, I'm Alice"}, {"output": "Hello Alice!"})
memory.save_context({"input": "What's my name?"}, {"output": "Your name is Alice"})

print(memory.load_memory_variables({}))
# Output: Full conversation history
```

**Pros**:
- Complete context retention
- No information loss
- Simple implementation

**Cons**:
- Token usage grows unbounded
- Expensive for long conversations
- May hit context limits

**Use Cases**:
- Short conversations
- High-value interactions requiring full context
- Debugging and development

---

### 2. ConversationBufferWindowMemory

**Stores**: Sliding window of K most recent message pairs

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # Keep last 2 exchanges
memory.save_context({"input": "Message 1"}, {"output": "Response 1"})
memory.save_context({"input": "Message 2"}, {"output": "Response 2"})
memory.save_context({"input": "Message 3"}, {"output": "Response 3"})

print(memory.load_memory_variables({}))
# Output: Only messages 2 and 3 (last 2 exchanges)
```

**Pros**:
- Bounded token usage
- Recent context always available
- Good balance for most use cases

**Cons**:
- Loses older context
- May forget important earlier information
- Fixed window size

**Use Cases**:
- Chat interfaces
- Customer support bots
- Most conversational AI applications

---

### 3. ConversationSummaryMemory

**Stores**: LLM-generated summaries of conversation history

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationSummaryMemory(llm=llm)

memory.save_context({"input": "Tell me about Python"}, {"output": "Python is..."})
memory.save_context({"input": "What about Java?"}, {"output": "Java is..."})

print(memory.load_memory_variables({}))
# Output: "The human asked about Python and Java. The AI explained..."
```

**Pros**:
- Compact representation
- Retains key information
- Scales to long conversations

**Cons**:
- Requires LLM calls (cost)
- Summary quality varies
- May lose nuanced details

**Use Cases**:
- Very long conversations
- Multi-session interactions
- When context must scale indefinitely

---

### 4. ConversationSummaryBufferMemory

**Stores**: Recent messages + summary of older messages

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100  # When exceeded, summarize oldest messages
)
```

**Pros**:
- Best of both worlds (recent detail + old summary)
- Adaptive to conversation length
- Controlled token usage

**Cons**:
- Most complex implementation
- Requires LLM for summaries
- More expensive than simple strategies

**Use Cases**:
- Production chatbots
- Multi-turn technical support
- Long-running conversations

---

## Memory Variables

### Input/Output Keys

Memory systems need to know which keys to track:

```python
memory = ConversationBufferMemory(
    input_key="user_input",      # Track this key as user messages
    output_key="bot_response",   # Track this key as bot messages
    memory_key="chat_history"    # Store history under this key
)
```

### Return Messages vs Strings

```python
# Return as string (default)
memory = ConversationBufferMemory()
memory.load_memory_variables({})
# → {"history": "Human: Hi\nAI: Hello"}

# Return as message objects
memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})
# → {"history": [HumanMessage("Hi"), AIMessage("Hello")]}
```

---

## Memory Integration with Chains

### Basic Integration

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory()

chain = ConversationChain(llm=llm, memory=memory)

# First interaction
response = chain.run("Hi, I'm Alice")
# → "Hello Alice! How can I help you?"

# Second interaction (remembers context)
response = chain.run("What's my name?")
# → "Your name is Alice"
```

### LCEL Integration

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

# Add memory wrapper
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

---

## Memory Comparison

| Memory Type | Token Usage | Context Retention | Cost | Best For |
|------------|-------------|-------------------|------|----------|
| **Buffer** | Grows unbounded | Complete | Low | Short conversations |
| **Window** | Fixed (k messages) | Recent only | Low | Most use cases |
| **Summary** | Bounded | Key points | High (LLM calls) | Long conversations |
| **Summary Buffer** | Adaptive | Recent + summary | Medium | Production chatbots |

---

## Phase 3 vs LangChain

### Your ChatMemory Class (Phase 3)

```python
# phase3_llm_applications/02_chat/memory.py
class ChatMemory:
    def __init__(self, strategy: str = "buffer", max_messages: int = 10):
        self.strategy = strategy
        self.max_messages = max_messages
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if self.strategy == "sliding_window":
            self.messages = self.messages[-self.max_messages:]
```

### LangChain Equivalent

```python
# Using LangChain
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10)  # Same sliding window
memory.save_context({"input": content}, {"output": response})
```

**Key Differences**:
- LangChain: Built-in serialization, chain integration, multiple strategies
- Your Implementation: Custom logic, explicit control, learning value

**When to Use LangChain**:
- ✅ Production applications needing robust memory
- ✅ Complex conversational workflows
- ✅ Multiple memory strategies

**When to Use Custom**:
- ✅ Learning and understanding fundamentals
- ✅ Specific requirements not covered by LangChain
- ✅ Performance-critical applications

---

## Common Patterns

### 1. Multi-User Memory

```python
# Store separate memory per user
user_memories = {}

def get_memory(user_id: str) -> ConversationBufferMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory()
    return user_memories[user_id]
```

### 2. Memory Persistence

```python
# Save memory to disk
import json

def save_memory(memory: ConversationBufferMemory, filename: str):
    messages = memory.chat_memory.messages
    with open(filename, 'w') as f:
        json.dump([msg.dict() for msg in messages], f)

def load_memory(filename: str) -> ConversationBufferMemory:
    with open(filename, 'r') as f:
        data = json.load(f)
    memory = ConversationBufferMemory()
    for msg in data:
        # Reconstruct message objects
        ...
    return memory
```

### 3. Memory Clearing

```python
# Clear memory when conversation resets
memory.clear()

# Or create new memory instance
memory = ConversationBufferMemory()
```

---

## Debugging Memory

### Inspect Memory State

```python
# View current memory contents
print(memory.load_memory_variables({}))

# View underlying messages
print(memory.chat_memory.messages)

# Check token count (if using summary)
if hasattr(memory, 'buffer'):
    print(f"Tokens: {len(memory.buffer)}")
```

### Common Issues

**Issue**: Memory not updating
```python
# Make sure to call save_context after each interaction
memory.save_context({"input": user_msg}, {"output": bot_response})
```

**Issue**: Token limits exceeded
```python
# Switch to window or summary memory
memory = ConversationBufferWindowMemory(k=5)  # Smaller window
```

**Issue**: Lost context
```python
# Check memory key matches prompt placeholder
memory = ConversationBufferMemory(memory_key="chat_history")
# Prompt must have: MessagesPlaceholder(variable_name="chat_history")
```

---

## Exercises

1. **Buffer Memory**: Build a chatbot that remembers your name
2. **Window Memory**: Create a customer support bot with 3-message context
3. **Summary Memory**: Implement a long conversation summarizer
4. **Comparison**: Compare token usage across all memory types
5. **Integration**: Build a conversational chain with memory

---

## Production Memory Patterns

### Database-Backed Storage

**PostgreSQL (Most Common):**
```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="postgresql://user:pass@localhost/chatdb"
    )
```

**Redis (Fast Sessions):**
```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=3600  # auto-expire after 1 hour
    )
```

### Multi-Tier Architecture

Production applications typically use:
1. **Redis** - Fast cache for active sessions
2. **PostgreSQL** - Durable storage for recent conversations
3. **S3/Archive** - Long-term storage for compliance

### Scalability Patterns

- **User-based sharding**: Partition by user ID for horizontal scaling
- **Session TTL**: Auto-expire inactive sessions to manage memory
- **Async writes**: Non-blocking database writes for better performance
- **Compliance**: GDPR/CCPA deletion and anonymization support

### Storage Comparison

| Storage | Speed | Durability | Cost | Best For |
|---------|-------|------------|------|----------|
| In-Memory Dict | ⚡ Fastest | ❌ Volatile | Free | Development |
| Redis | ⚡ Very Fast | ⚠️ Configurable | $$ | Active sessions |
| PostgreSQL | ✓ Fast | ✅ Durable | $ | Production |
| DynamoDB/Firestore | ✓ Fast | ✅ Durable | $$$ | Serverless |

---

## Implementation Notes

### Pydantic Custom Memory Classes

When extending `ChatMessageHistory` (a Pydantic BaseModel), declare fields properly:

```python
from pydantic import Field

class WindowChatMessageHistory(ChatMessageHistory):
    """custom memory with sliding window"""

    # ✅ Correct: Pydantic field declaration
    k: int = Field(default=2, description="window size")

    # ❌ Wrong: Setting in __init__ causes "object has no field" error
    # def __init__(self, k: int = 2):
    #     super().__init__()
    #     self.k = k  # ValueError!
```

---

## Key Takeaways

✅ **Memory enables context**: LLMs need memory for coherent multi-turn conversations
✅ **Multiple strategies**: Buffer, Window, Summary each have trade-offs
✅ **Token awareness**: Choose memory type based on conversation length and cost
✅ **Chain integration**: Memory works seamlessly with LangChain chains
✅ **Production ready**: LangChain memory is battle-tested for real applications

---

## Next Steps

After mastering memory:
- **RAG Module**: Apply memory to retrieval-augmented generation
- **Agents Module**: Use memory in agent workflows
- **LangGraph**: Build stateful multi-agent systems with memory

---

## Run Examples

```bash
# Conceptual demos (no API key required)
uv run python -m phase7_frameworks.01_langchain_basics.04_memory.concepts

# Practical demos (requires OPENAI_API_KEY)
uv run python -m phase7_frameworks.01_langchain_basics.04_memory.practical
```