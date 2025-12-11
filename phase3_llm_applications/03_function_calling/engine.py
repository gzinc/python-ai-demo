"""
FunctionCallingEngine - Orchestrates tool calling with LLM

Similar to a Service class in Java that coordinates multiple components.
This is the main class you'd use in your application.
"""

import os
from typing import Optional

from schemas import Tool
from registry import ToolRegistry
from executor import ToolExecutor
from common_tools import get_weather


class FunctionCallingEngine:
    """
    complete function calling engine

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  FunctionCallingEngine                      │
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │   OpenAI    │  │   Tool      │  │     Tool            │  │
    │  │   Client    │  │  Registry   │  │    Executor         │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    │         │                │                   │              │
    │         └────────────────┼───────────────────┘              │
    │                          ▼                                  │
    │                    ┌─────────────┐                          │
    │                    │    chat()   │                          │
    │                    │  • send msg │                          │
    │                    │  • handle   │                          │
    │                    │    tools    │                          │
    │                    │  • loop     │                          │
    │                    └─────────────┘                          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant with access to tools.",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.registry = ToolRegistry()
        self.executor = ToolExecutor()
        self.messages: list[dict] = []

        # check for API key
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def register_tool(
        self,
        tool: Tool,
        function: callable,
        validator: Optional[callable] = None
    ) -> None:
        """register a tool with its implementation"""
        tool.function = function
        self.registry.register(tool)
        self.executor.register(tool.name, function, validator)

    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        send message and handle any tool calls

        The Loop:
        ┌─────────────────────────────────────────────────────────┐
        │  while has_tool_calls and iterations < max:             │
        │      1. Send messages to LLM                            │
        │      2. If LLM wants to call tools:                     │
        │         - Execute each tool                             │
        │         - Add results to messages                       │
        │         - Continue loop                                 │
        │      3. If no tool calls:                               │
        │         - Return LLM's text response                    │
        └─────────────────────────────────────────────────────────┘
        """
        if not self.api_key:
            return self._simulate_response(user_message)

        from openai import OpenAI
        client = OpenAI()

        # add user message
        self.messages.append({"role": "user", "content": user_message})

        # get tools in OpenAI format
        tools = self.registry.to_openai_format()

        iterations = 0
        while iterations < max_iterations:
            iterations += 1

            # call LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.messages
                ],
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )

            message = response.choices[0].message

            # check for tool calls
            if message.tool_calls:
                # add assistant message with tool calls
                self.messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # execute each tool call
                for tool_call in message.tool_calls:
                    result = self.executor.execute(
                        name=tool_call.function.name,
                        args=tool_call.function.arguments,
                        tool_call_id=tool_call.id,
                    )

                    # add tool result to messages
                    self.messages.append(result.to_openai_message())

                # continue loop to get final response
                continue

            else:
                # no tool calls - LLM gave final response
                assistant_content = message.content or ""
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                return assistant_content

        return "Max iterations reached. Please try again."

    def _simulate_response(self, user_message: str) -> str:
        """simulate response when no API key (for demo)"""
        user_lower = user_message.lower()

        if "weather" in user_lower:
            # simulate weather tool call
            if "tokyo" in user_lower:
                result = get_weather("Tokyo")
            elif "london" in user_lower:
                result = get_weather("London")
            else:
                result = get_weather("Unknown City")
            return f"[Simulated] The weather in {result['location']} is {result['temperature']}°C and {result['condition']}."

        elif any(op in user_lower for op in ["+", "-", "*", "/", "calculate", "what is"]):
            # extract numbers and simulate calculation
            return "[Simulated] I would use the calculator tool to compute this."

        else:
            return f"[Simulated] I would process: '{user_message}' (No API key configured)"

    def reset(self) -> None:
        """clear conversation history"""
        self.messages = []
