# Agents & Tools Module

**Purpose**: Learn LangChain's agent patterns and compare them to your Phase 4 implementations

---

## Learning Objectives

By the end of this module, you will:
1. **Tool Creation**: Use `@tool` decorator for custom tool implementation
2. **Agent Patterns**: Understand `create_agent` and ReAct reasoning pattern
3. **Tool Integration**: Combine multiple tools in agent workflows
4. **Memory Integration**: Add conversational memory to agents
5. **Phase 4 Comparison**: See how LangChain compares to your custom agents

---

## Key Concepts

### What is an Agent?

**Agent** = LLM + Tools + Reasoning Loop

```
User: "What's the weather in Tokyo and convert 25°C to Fahrenheit?"

Agent thinks:
1. I need current weather → use weather_tool("Tokyo")
2. Got 25°C → use calculator_tool("25 * 9/5 + 32")
3. Combine results → respond to user

Agent Response: "Tokyo is 25°C (77°F) and sunny"
```

**Key Difference from Chains**:
- **Chain**: Fixed sequence (A → B → C)
- **Agent**: Dynamic reasoning (LLM decides which tools to use and when)

---

### ReAct Pattern (Reasoning + Acting)

**ReAct** = Reason about what to do, then Act with tools

```
Thought: I need to search for information about LangChain
Action: web_search
Action Input: "LangChain framework overview"
Observation: [search results...]

Thought: Now I have enough information to answer
Action: Final Answer
Action Input: "LangChain is a framework for..."
```

**Pattern**:
1. **Thought**: LLM reasons about next step
2. **Action**: LLM selects a tool
3. **Action Input**: LLM provides tool parameters
4. **Observation**: Tool execution result
5. **Repeat** until "Final Answer" action

---

## Phase 4 vs LangChain

### Your ReActAgent (Phase 4)

```python
# phase4_ai_agents/01_react_agent/agent.py
class ReActAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.max_iterations = 5

    def run(self, task: str) -> str:
        for i in range(self.max_iterations):
            # 1. Generate thought and action
            response = self.llm.generate(prompt)
            thought, action, action_input = self.parse(response)

            # 2. Execute tool
            tool = self.registry.get(action)
            observation = tool.execute(action_input)

            # 3. Check if done
            if action == "Final Answer":
                return observation

        return "Max iterations reached"
```

### LangChain Equivalent

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 1. Define tools
@tool
def web_search(query: str) -> str:
    """search the web for information"""
    return search_api(query)

# 2. Create agent (returns executable CompiledStateGraph)
agent = create_agent(model=llm, tools=[web_search])

# 3. Execute
result = agent.invoke({"messages": [HumanMessage(content="search for LangChain")]})
```

**Key Differences**:
- **LangChain**: Built-in tool registry, error handling, memory integration
- **Your Implementation**: Custom logic, explicit control, learning value

---

## Tool Types

### 1. Function-Based Tools (`@tool` decorator)

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> float:
    """evaluate mathematical expressions"""
    return eval(expression)

# Tool metadata automatically extracted:
# - name: "calculator"
# - description: "evaluate mathematical expressions"
# - args_schema: inferred from function signature
```

### 2. Class-Based Tools (`BaseTool`)

```python
from langchain.tools import BaseTool
from pydantic import Field

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "search the web for current information"
    api_key: str = Field(..., description="API key for search service")

    def _run(self, query: str) -> str:
        """execute search"""
        return self.search_api.query(query, api_key=self.api_key)

    async def _arun(self, query: str) -> str:
        """async execution"""
        return await self.search_api.aquery(query, api_key=self.api_key)
```

### 3. Pre-built Tools (LangChain Community)

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun()

tools = [search, wiki]
```

---

## Agent Patterns

### 1. Basic React Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [calculator, web_search]

# create_agent returns executable CompiledStateGraph
agent = create_agent(model=llm, tools=tools)

result = agent.invoke({"messages": [HumanMessage(content="What's 25 * 4 + 10?")]})
```

### 2. Agent with Memory

```python
from langchain_core.messages import HumanMessage

# LangGraph agents maintain conversation history automatically
agent = create_agent(model=llm, tools=tools)

# First interaction
result1 = agent.invoke({
    "messages": [HumanMessage(content="My name is Alice")]
})

# Second interaction - include previous messages for context
all_messages = result1["messages"] + [HumanMessage(content="What's my name?")]
result2 = agent.invoke({"messages": all_messages})
# → "Your name is Alice"
```

**Note**: LangGraph manages state automatically. For persistent memory across sessions, use checkpointers (covered in practical.py).

### 3. Multi-Tool Agent

```python
tools = [
    web_search,      # Current information
    calculator,      # Math operations
    wikipedia,       # Knowledge base
    weather,         # Weather data
]

agent = create_agent(model=llm, tools=tools)

# Agent decides which tools to use
result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in Paris? Convert temperature to Fahrenheit.")]
})
```

---

## Tool Error Handling

### Graceful Degradation

```python
@tool
def risky_operation(query: str) -> str:
    """operation that might fail"""
    try:
        return external_api.call(query)
    except APIError as e:
        # return error message to agent
        return f"Error: {str(e)}. Please try a different approach."

# agent will see the error and adapt its strategy
agent = create_agent(model=llm, tools=[risky_operation])

# execute with error handling
try:
    result = agent.invoke({"messages": [HumanMessage(content="task")]})
except Exception as e:
    print(f"Agent execution failed: {e}")
```

### Timeout Configuration

```python
from langchain_openai import ChatOpenAI

# configure model with timeout
llm = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=30,  # 30 second timeout for API calls
)

agent = create_agent(model=llm, tools=tools)
```

---

## Agent Prompts

### Default ReAct Prompt

```python
from langchain import hub

# Load default React prompt
react_prompt = hub.pull("hwchase17/react")

# Prompt structure:
"""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""
```

### Custom System Message

```python
from langchain_openai import ChatOpenAI

# configure model with custom system message
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={
        "system": "You are a helpful assistant. Always explain your reasoning before using tools."
    }
)

agent = create_agent(model=llm, tools=tools)
```

**Note**: LangGraph's `create_agent` uses built-in prompting. For advanced prompt customization, build custom graphs with LangGraph directly.

---

## Common Patterns

### 1. Tool Result Validation

```python
@tool
def validate_email(email: str) -> str:
    """validate email address format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if re.match(pattern, email):
        return f"✓ Valid email: {email}"
    else:
        return f"✗ Invalid email format: {email}"
```

### 2. Stateful Tools

```python
class DatabaseTool(BaseTool):
    name: str = "database_query"
    description: str = "query the database"
    connection: Any = None

    def __init__(self):
        super().__init__()
        self.connection = create_db_connection()

    def _run(self, query: str) -> str:
        cursor = self.connection.cursor()
        cursor.execute(query)
        return str(cursor.fetchall())

    def __del__(self):
        if self.connection:
            self.connection.close()
```

### 3. Tool Composition

```python
@tool
def multi_step_research(topic: str) -> str:
    """perform comprehensive research on a topic"""
    # Compose multiple tools
    web_results = web_search(topic)
    wiki_results = wikipedia(topic)

    # Synthesize
    summary = f"""
    Web Search: {web_results}
    Wikipedia: {wiki_results}
    """
    return summary
```

---

## Agent Types Comparison

| Agent Type | Use Case | Complexity | Phase 4 Equivalent |
|------------|----------|------------|-------------------|
| **ReAct** | General reasoning + actions | Medium | Your ReActAgent |
| **OpenAI Functions** | Structured tool calls | Low | FunctionCallingAgent |
| **Conversational** | Chat with tools + memory | Medium | ChatAgent with memory |
| **Plan-and-Execute** | Complex multi-step tasks | High | PlanExecuteAgent |
| **Self-Ask** | Research with citations | Medium | ResearchAgent |

---

## Debugging Agents

### Stream Agent Execution

```python
from langchain_core.messages import HumanMessage

agent = create_agent(model=llm, tools=tools)

# stream agent execution to see intermediate steps
for event in agent.stream({"messages": [HumanMessage(content="search for LangChain")]}):
    print(event)
    print("---")

# output shows:
# {'agent': {'messages': [AIMessage(...)]}}
# ---
# {'tools': {'messages': [ToolMessage(...)]}}
# ---
```

### Callbacks

```python
from langchain.callbacks import StdOutCallbackHandler

agent = create_agent(model=llm, tools=tools)

# execute with detailed logging
result = agent.invoke(
    {"messages": [HumanMessage(content="question")]},
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

### Inspect Agent State

```python
# check tool calls
print(f"Tools available: {[tool.name for tool in tools]}")

# inspect result messages
result = agent.invoke({"messages": [HumanMessage(content="question")]})
messages = result["messages"]

for msg in messages:
    print(f"{type(msg).__name__}: {msg.content[:100]}")
```

---

## Production Patterns

### Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_second: int):
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@tool
@rate_limit(calls_per_second=5)
def api_call(query: str) -> str:
    """rate-limited API call"""
    return external_api.query(query)
```

### Caching Tool Results

```python
from functools import lru_cache

@tool
@lru_cache(maxsize=100)
def cached_search(query: str) -> str:
    """search with caching"""
    return expensive_search_api(query)
```

### Timeout Protection

```python
from langchain_openai import ChatOpenAI

# configure model with timeout
llm = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=60,  # 60 second timeout for LLM calls
)

agent = create_agent(model=llm, tools=tools)

# additional timeout at invocation level
from langchain_core.runnables import RunnableConfig

result = agent.invoke(
    {"messages": [HumanMessage(content="task")]},
    config=RunnableConfig(timeout=60)  # overall execution timeout
)
```

---

## Exercises

1. **Basic Tool**: Create a `@tool` for temperature conversion (C ↔ F)
2. **Multi-Tool Agent**: Build agent with calculator + web search
3. **Custom Tool**: Implement `BaseTool` for file operations
4. **Agent Memory**: Add conversation memory to agent
5. **Error Handling**: Implement graceful degradation for API failures
6. **Phase 4 Migration**: Convert your Phase 4 agent to LangChain
7. **Tool Composition**: Create a research tool that uses multiple sub-tools
8. **Production Agent**: Add rate limiting, caching, and timeouts

---

## Common Pitfalls

### 1. Tool Description Quality

```python
# ❌ Bad: Vague description
@tool
def process(data: str) -> str:
    """process data"""
    return data.upper()

# ✅ Good: Specific description
@tool
def uppercase_converter(text: str) -> str:
    """convert text to uppercase letters. useful for formatting names or titles"""
    return text.upper()
```

### 2. Infinite Loops

```python
# ❌ Bad: No timeout protection
agent = create_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [HumanMessage(content="task")]})

# ✅ Good: Set timeouts
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", request_timeout=30)
agent = create_agent(model=llm, tools=tools)

result = agent.invoke(
    {"messages": [HumanMessage(content="task")]},
    config=RunnableConfig(recursion_limit=5, timeout=30)  # prevent infinite loops
)
```

### 3. Tool Overload

```python
# ❌ Bad: Too many similar tools (confuses LLM)
tools = [search1, search2, search3, search4]  # All do similar things

# ✅ Good: One clear tool per function
tools = [web_search, calculator, weather]  # Distinct purposes
```

---

## Key Takeaways

✅ **Agents are dynamic**: LLM decides tool usage at runtime
✅ **ReAct is powerful**: Combines reasoning with action execution
✅ **Tools need good descriptions**: LLM relies on descriptions to choose tools
✅ **Error handling matters**: Production agents need robust error handling
✅ **LangChain simplifies**: Compared to Phase 4, less boilerplate code
✅ **But custom has value**: Your Phase 4 code teaches core concepts

---

## Next Steps

After mastering agents:
- **LangGraph Module**: Multi-agent workflows and state machines
- **Production Deployment**: FastAPI + agent endpoints (Phase 5)
- **Advanced Patterns**: Plan-and-execute, self-ask agents
- **Real Projects**: Build production agent applications

---

## Run Examples

**📊 Visual Learning**: All practical demos include comprehensive ASCII diagrams showing agent workflows, tool interactions, and execution patterns.

```bash
# Conceptual demos (no API key required)
uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.concepts

# Practical demos (requires OPENAI_API_KEY)
uv run python -m phase7_frameworks.01_langchain_basics.06_agents_tools.practical
```

---

## Resources

- [LangChain Agents Docs](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [Agent Types Guide](https://python.langchain.com/docs/modules/agents/agent_types/)
