# Module 2: Tool Use

Real, practical tools for AI agents.

## Learning Objectives

After this module, you'll understand:
- How to define tools with typed parameters
- The BaseTool interface pattern (like Java interfaces)
- Tool registry as a service container
- Creating file, web, and HTTP tools
- Mock vs real implementations for testing

## Key Concepts

### Tool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ToolRegistry                            │
│  (Service container - holds all tools)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ ReadFileTool │  │ WebSearchTool│  │ HttpGetTool  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                   ┌──────────────┐                          │
│                   │   BaseTool   │  ← Abstract interface    │
│                   └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Java Equivalents

| Python | Java |
|--------|------|
| `BaseTool` (ABC) | `interface Tool` |
| `ToolRegistry` | `@Component class ToolRegistry` |
| `ToolResult.ok(data)` | `ToolResult.success(data)` |
| `ToolResult.fail(error)` | `ToolResult.failure(error)` |
| `@property def name` | `String getName()` |
| `def execute(**kwargs)` | `ToolResult execute(Map<String, Object> args)` |

## File Structure

```
02_tool_use/
├── schemas/
│   ├── __init__.py
│   ├── tool.py          # ToolParameter, ToolDefinition, ToolResult
│   └── config.py        # ToolConfig settings
├── tools/
│   ├── __init__.py
│   ├── base.py          # BaseTool, ToolRegistry
│   ├── file_tools.py    # ReadFile, WriteFile, ListDirectory
│   ├── web_search.py    # WebSearchTool (mock + real)
│   └── http_tool.py     # HttpGetTool (mock + real)
├── tool_agent.py        # Agent with real tools
├── examples.py          # Demonstrations
└── README.md
```

## Quick Start

```bash
# Run examples (no API key needed)
cd phase4_ai_agents/02_tool_use
uv run python examples.py

# Run tool agent (needs API key)
OPENAI_API_KEY=your-key uv run python tool_agent.py
```

## Creating a Tool (3 Steps)

### Step 1: Extend BaseTool

```python
from tools.base import BaseTool
from schemas.tool import ToolDefinition, ToolParameter, ToolResult

class MyTool(BaseTool):
    pass
```

### Step 2: Define Metadata

```python
@property
def name(self) -> str:
    return "my_tool"

@property
def definition(self) -> ToolDefinition:
    return ToolDefinition(
        name=self.name,
        description="What this tool does",
        parameters=[
            ToolParameter(
                name="input",
                param_type="string",
                description="The input value",
                required=True,
            ),
        ],
    )
```

### Step 3: Implement Logic

```python
def execute(self, input: str, **kwargs) -> ToolResult:
    try:
        result = do_something(input)
        return ToolResult.ok(result)
    except Exception as e:
        return ToolResult.fail(str(e))
```

## Tools Included

### File Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file contents | `path` |
| `write_file` | Write to file | `path`, `content` |
| `list_directory` | List directory | `path` |

### Web Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `web_search` | Search the web | `query`, `max_results` |
| `http_get` | Make HTTP GET | `url` |

## Mock Mode

Tools support mock mode for testing without APIs:

```python
# Mock mode (default) - returns fake but realistic data
search = WebSearchTool(use_mock=True)

# Real mode - needs API key
search = WebSearchTool(use_mock=False, api_key="...")
```

## Security Features

### Directory Restrictions

```python
# Only allow access to specific directories
read_tool = ReadFileTool(
    allowed_directories=["/tmp", "/home/user/data"]
)

# This will fail:
read_tool.execute(path="/etc/passwd")
# → ToolResult.fail("Access denied: ...")
```

### Domain Restrictions

```python
# Only allow specific domains
http_tool = HttpGetTool(
    allowed_domains=["api.github.com", "jsonplaceholder.typicode.com"]
)
```

## Integration with Agent

```python
from tool_agent import ToolAgent

# Create agent with tools
agent = ToolAgent(
    use_mock=True,  # mock web/http tools
    provider="openai",
)

# Run task
result = agent.run("Search for Python tutorials and summarize")
print(result.answer)
```

## Exercises

1. **Create a Calculator Tool**
   - Operations: add, subtract, multiply, divide
   - Handle division by zero

2. **Create a Date/Time Tool**
   - Get current time
   - Format dates
   - Calculate time differences

3. **Extend HTTP Tool**
   - Add POST support
   - Add headers parameter
   - Add timeout parameter

4. **Build a File Search Tool**
   - Search for files by pattern
   - Search file contents

## Key Patterns

### Factory Methods

```python
# Instead of: ToolResult(success=True, data=x, error=None)
# Use factory methods:
ToolResult.ok(data)      # success
ToolResult.fail(error)   # failure
```

### Service Registry

```python
# Register tools once
registry = ToolRegistry()
registry.register(ReadFileTool())
registry.register(WebSearchTool())

# Use anywhere
result = registry.execute("read_file", path="/tmp/test.txt")
```

### Callable Tools

```python
# Tools can be called directly
tool = ReadFileTool()
result = tool(path="/tmp/test.txt")  # uses __call__
# Same as: tool.execute(path="/tmp/test.txt")
```

## Connection to Module 1

Module 1 (Simple Agent) used mock tools:
```python
tools = {
    "get_weather": lambda city: '{"temp": 18, "conditions": "rainy"}',
}
```

Module 2 provides real tools:
```python
registry = ToolRegistry()
registry.register(WebSearchTool())
registry.register(ReadFileTool())
# These actually search/read!
```

## Next: Module 3 (Multi-Agent)

Module 3 will show:
- Multiple agents working together
- Agent-to-agent communication
- Task delegation patterns
- Orchestration strategies

## Further Reading

- [CONCEPTS.md](../CONCEPTS.md) - Core agent concepts and terminology
