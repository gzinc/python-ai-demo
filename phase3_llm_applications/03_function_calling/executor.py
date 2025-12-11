"""
ToolExecutor - Safe execution of LLM-requested function calls

Similar to a Command Executor or Task Runner in Java.
Handles validation, error handling, and result formatting.
"""

from typing import Any, Callable, Optional
import json

from models import ToolResult


class ToolExecutor:
    """
    safe executor for LLM-requested function calls

    Usage:
        executor = ToolExecutor()
        executor.register("get_weather", get_weather_fn)
        executor.register("calculate", calculate_fn)

        # when LLM requests a tool call
        result = executor.execute(
            name="get_weather",
            args={"location": "Tokyo"},
            tool_call_id="call_abc123"
        )

        # add result to messages
        messages.append(result.to_openai_message())

    Error Handling:
    ┌─────────────────────────────────────────────────────────────┐
    │  ToolExecutor catches ALL errors and returns them as        │
    │  ToolResult with success=False. This lets the LLM:          │
    │  • Retry with different arguments                           │
    │  • Try a different approach                                 │
    │  • Inform the user about limitations                        │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.functions: dict[str, Callable] = {}
        self.validators: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        function: Callable,
        validator: Optional[Callable] = None
    ) -> None:
        """
        register a function that can be called by LLM

        Args:
            name: function name (must match tool schema)
            function: the actual function to execute
            validator: optional function to validate args before execution
        """
        self.functions[name] = function
        if validator:
            self.validators[name] = validator

    def execute(
        self,
        name: str,
        args: dict,
        tool_call_id: Optional[str] = None
    ) -> ToolResult:
        """
        safely execute a function

        Steps:
        1. Check if function exists
        2. Parse args if string (LLM sometimes returns JSON string)
        3. Validate args if validator registered
        4. Execute function in try/except
        5. Return formatted result
        """
        # step 1: check function exists
        if name not in self.functions:
            return ToolResult(
                success=False,
                error=f"unknown function: {name}. Available: {list(self.functions.keys())}",
                tool_call_id=tool_call_id,
            )

        # step 2: parse args if string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                return ToolResult(
                    success=False,
                    error=f"invalid JSON arguments: {e}",
                    tool_call_id=tool_call_id,
                )

        # step 3: validate args if validator exists
        if name in self.validators:
            try:
                validation_error = self.validators[name](args)
                if validation_error:
                    return ToolResult(
                        success=False,
                        error=f"validation failed: {validation_error}",
                        tool_call_id=tool_call_id,
                    )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"validation error: {e}",
                    tool_call_id=tool_call_id,
                )

        # step 4: execute function
        try:
            result = self.functions[name](**args)
            return ToolResult(
                success=True,
                data=result,
                tool_call_id=tool_call_id,
            )
        except TypeError as e:
            # likely wrong arguments
            return ToolResult(
                success=False,
                error=f"argument error: {e}",
                tool_call_id=tool_call_id,
            )
        except Exception as e:
            # any other error
            return ToolResult(
                success=False,
                error=f"execution failed: {type(e).__name__}: {e}",
                tool_call_id=tool_call_id,
            )

    def execute_multiple(
        self,
        tool_calls: list[dict]
    ) -> list[ToolResult]:
        """
        execute multiple tool calls (for parallel tool calling)

        Args:
            tool_calls: list of {"name": "...", "args": {...}, "id": "..."}

        Returns:
            list of ToolResult in same order
        """
        results = []
        for call in tool_calls:
            result = self.execute(
                name=call.get("name", ""),
                args=call.get("args", {}),
                tool_call_id=call.get("id"),
            )
            results.append(result)
        return results
