# Module 3: Function Calling - Let LLMs Use Your Tools

## Learning Objectives

By the end of this module, you will:
- Understand how function calling works in LLMs
- Define tool schemas for OpenAI and Anthropic
- Execute functions based on LLM decisions
- Handle errors and validation
- Build multi-tool orchestration patterns

## Prerequisites

- [Phase 2](../../phase2_llm_fundamentals/) complete (API Integration)
- [Phase 3 Module 1](../01_rag_system/) (RAG System)
- [Phase 3 Module 2](../02_chat_interface/) (Chat Interface)

## What is Function Calling?

```
┌─────────────────────────────────────────────────────────────────┐
│                  Function Calling Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WITHOUT Function Calling:                                      │
│  ┌─────────┐         ┌─────────┐         ┌─────────┐            │
│  │  User   │────────►│   LLM   │────────►│  Text   │            │
│  │ "What's │         │         │         │ "I can't│            │
│  │ weather"│         │         │         │  check" │            │
│  └─────────┘         └─────────┘         └─────────┘            │
│                                                                 │
│  WITH Function Calling:                                         │
│  ┌─────────┐         ┌─────────┐         ┌─────────┐            │
│  │  User   │────────►│   LLM   │────────►│ "Call   │            │
│  │ "What's │         │ decides │         │ weather │            │
│  │ weather"│         │ to call │         │  API"   │            │
│  └─────────┘         └─────────┘         └────┬────┘            │
│                                               │                 │
│                                               ▼                 │
│                                         ┌─────────┐             │
│                      ┌──────────────────│  Your   │             │
│                      │   Result: 72°F   │  Code   │             │
│                      │                  │ (tools) │             │
│                      ▼                  └─────────┘             │
│                ┌─────────┐                                      │
│                │   LLM   │────────►  "It's 72°F and sunny"      │
│                │ formats │                                      │
│                │ response│                                      │
│                └─────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The Key Insight

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   LLM DOES NOT execute functions!                               │
│                                                                 │
│   It only DECIDES which function to call and with what args.    │
│   YOUR CODE executes the function and returns results.          │
│                                                                 │
│   ┌─────────────┐        ┌─────────────┐        ┌────────────┐  │
│   │    LLM      │        │  Your Code  │        │  External  │  │
│   │  "I want to │───────►│  executes   │───────►│   APIs/    │  │
│   │   call X"   │        │  function X │        │  Services  │  │
│   └─────────────┘        └─────────────┘        └────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Function Definition Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                   OpenAI Tool Schema                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  {                                                              │
│    "type": "function",                                          │
│    "function": {                                                │
│      "name": "get_weather",         ◄── function identifier     │
│      "description": "Get current    ◄── helps LLM decide when   │
│                      weather for        to use this function    │
│                      a location",                               │
│      "parameters": {                ◄── JSON Schema for args    │
│        "type": "object",                                        │
│        "properties": {                                          │
│          "location": {                                          │
│            "type": "string",                                    │
│            "description": "City name, e.g., 'London'"           │
│          },                                                     │
│          "unit": {                                              │
│            "type": "string",                                    │
│            "enum": ["celsius", "fahrenheit"],                   │
│            "description": "Temperature unit"                    │
│          }                                                      │
│        },                                                       │
│        "required": ["location"]     ◄── required parameters     │
│      }                                                          │
│    }                                                            │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   Function Calling Loop                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Send message + available tools                         │
│  ─────────────────────────────────────                          │
│  messages = [{"role": "user", "content": "Weather in Tokyo?"}]  │
│  tools = [weather_tool, search_tool, calculator_tool]           │
│                      │                                          │
│                      ▼                                          │
│  Step 2: LLM decides to call a function                         │
│  ──────────────────────────────────────                         │
│  response.choices[0].message.tool_calls = [                     │
│    {                                                            │
│      "id": "call_abc123",                                       │
│      "function": {                                              │
│        "name": "get_weather",                                   │
│        "arguments": '{"location": "Tokyo", "unit": "celsius"}'  │
│      }                                                          │
│    }                                                            │
│  ]                                                              │
│                      │                                          │
│                      ▼                                          │
│  Step 3: YOUR CODE executes the function                        │
│  ───────────────────────────────────────                        │
│  result = get_weather(location="Tokyo", unit="celsius")         │
│  # result = {"temp": 22, "condition": "cloudy"}                 │
│                      │                                          │
│                      ▼                                          │
│  Step 4: Send result back to LLM                                │
│  ───────────────────────────────────                            │
│  messages.append({                                              │
│    "role": "tool",                                              │
│    "tool_call_id": "call_abc123",                               │
│    "content": '{"temp": 22, "condition": "cloudy"}'             │
│  })                                                             │
│                      │                                          │
│                      ▼                                          │
│  Step 5: LLM generates final response                           │
│  ────────────────────────────────────                           │
│  "The weather in Tokyo is 22°C and cloudy."                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## OpenAI vs Anthropic

```
┌─────────────────────────────────────────────────────────────────┐
│                   API Differences                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OPENAI                          ANTHROPIC                      │
│  ──────                          ─────────                      │
│  tools=[...]                     tools=[...]                    │
│  tool_choice="auto"              tool_choice={"type": "auto"}   │
│                                                                 │
│  Response structure:             Response structure:            │
│  message.tool_calls[0]           content[0] (type="tool_use")   │
│    .function.name                  .name                        │
│    .function.arguments             .input                       │
│    .id                             .id                          │
│                                                                 │
│  Send result:                    Send result:                   │
│  role="tool"                     role="user" with               │
│  tool_call_id=...                tool_result content block      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Tool Orchestration

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM Can Call Multiple Tools                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "Compare weather in Tokyo and London"                    │
│                      │                                          │
│                      ▼                                          │
│  LLM decides to call TWO functions:                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  tool_calls = [                                          │   │
│  │    {"name": "get_weather", "args": {"location": "Tokyo"}}│   │
│  │    {"name": "get_weather", "args": {"location": "London"}}   │
│  │  ]                                                       │   │
│  └──────────────────────────────────────────────────────────┘   │ 
│                      │                                          │
│         ┌────────────┴────────────┐                             │
│         ▼                         ▼                             │
│  ┌─────────────┐          ┌─────────────┐                       │
│  │ get_weather │          │ get_weather │                       │
│  │   Tokyo     │          │   London    │                       │
│  └──────┬──────┘          └──────┬──────┘                       │
│         │                        │                              │
│         ▼                        ▼                              │
│      22°C                     15°C                              │
│         │                        │                              │
│         └────────────┬───────────┘                              │
│                      ▼                                          │
│  LLM: "Tokyo is 22°C while London is 15°C, so Tokyo             │
│        is 7 degrees warmer."                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling

```
┌─────────────────────────────────────────────────────────────────┐
│                   Error Handling Pattern                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  What can go wrong:                                             │
│  ─────────────────                                              │
│  1. LLM calls function that doesn't exist                       │
│  2. LLM passes invalid arguments                                │
│  3. Function execution fails (API down, timeout)                │
│  4. Function returns unexpected format                          │
│                                                                 │
│  Solution: Wrap execution in try/catch, return errors to LLM    │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  try:                                                           │
│      result = execute_function(name, args)                      │
│      return {"success": True, "data": result}                   │
│  except ValidationError as e:                                   │
│      return {"success": False, "error": f"Invalid args: {e}"}   │
│  except Exception as e:                                         │
│      return {"success": False, "error": f"Execution failed: {e}"│
│                                                                 │
│  The LLM can then:                                              │
│  • Retry with different arguments                               │
│  • Try a different approach                                     │
│  • Inform user about the limitation                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files in This Module

| File | Purpose |
|------|---------|
| [schemas/](schemas/) | Data classes (Tool, ToolParameter, ToolResult) |
| [registry.py](registry.py) | ToolRegistry - manages available tools |
| [executor.py](executor.py) | ToolExecutor - safe execution with error handling |
| [common_tools.py](common_tools.py) | Tool factories + implementations |
| [engine.py](engine.py) | FunctionCallingEngine - orchestrates everything |
| [examples.py](examples.py) | Practical demos |

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Function Calling Module Structure                │
│                    (Java-like separation)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    examples.py                          │    │
│  │                  (Demo functions only)                  │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     engine.py                           │    │
│  │              (FunctionCallingEngine)                    │    │
│  │         Orchestrates: LLM ↔ Registry ↔ Executor         │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐          │
│  │ registry.py │   │ executor.py  │   │common_tools.py│         │
│  │             │   │              │   │              │          │
│  │ToolRegistry │   │ ToolExecutor │   │create_*_tool │          │
│  │• register() │   │• execute()   │   │get_weather() │          │
│  │• to_openai_ │   │• validate    │   │calculate()   │          │
│  │  format()   │   │• error wrap  │   │web_search()  │          │
│  └──────┬──────┘   └──────┬───────┘   └──────────────┘          │
│         │                 │                                     │
│         └────────┬────────┘                                     │
│                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     schemas/                             │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │    │
│  │  │  tool.py    │  │tool_result.py│  │  __init__.py   │  │    │
│  │  │             │  │              │  │                │  │    │
│  │  │ Tool        │  │ ToolResult   │  │ exports all    │  │    │
│  │  │ ToolParam   │  │ • success    │  │ dataclasses    │  │    │
│  │  │ • to_openai │  │ • data/error │  │                │  │    │
│  │  │ • to_anthro │  │ • to_message │  │                │  │    │
│  │  └─────────────┘  └──────────────┘  └────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Java Equivalents:**
| Python | Java |
|--------|------|
| schemas/tool.py | Tool.java (POJO) |
| schemas/tool_result.py | ToolResult.java (Response wrapper) |
| registry.py | ToolRegistry.java (Service registry) |
| executor.py | ToolExecutor.java (Command executor) |
| common_tools.py | ToolFactory.java |
| engine.py | FunctionCallingService.java (Facade) |

## Running Examples

```bash
# run all examples
uv run python phase3_llm_applications/03_function_calling/examples.py

# test imports (verify structure)
cd phase3_llm_applications/03_function_calling
uv run python -c "from models import Tool, ToolResult; print('OK')"
uv run python -c "from engine import FunctionCallingEngine; print('OK')"
```

## Key Takeaways

1. **LLM decides, you execute** - LLM only outputs function name + args
2. **Good descriptions matter** - Help LLM know when to use each tool
3. **Validate everything** - Don't trust LLM-generated arguments blindly
4. **Return errors gracefully** - Let LLM retry or adapt
5. **This enables agents** - Function calling is the foundation for AI agents

## Connection to AI Agents

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Function Calling is the FOUNDATION of AI Agents!               │
│                                                                 │
│  Agent = LLM + Tools + Loop                                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      Agent Loop                          │   │
│  │                                                          │   │
│  │  while not done:                                         │   │
│  │      response = llm.chat(messages, tools=available_tools)│   │
│  │      if response.has_tool_calls:                         │   │
│  │          results = execute_tools(response.tool_calls)    │   │
│  │          messages.append(results)                        │   │
│  │      else:                                               │   │
│  │          done = True  # LLM gave final answer            │   │
│  │          return response.content                         │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Phase 4 will build on this!                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps

After completing this module:
- → [Phase 4: AI Agents](../../phase4_ai_agents/) (autonomous systems with tools)
