"""
ReAct Agent - Reasoning and Acting in a loop.

The ReAct pattern:
1. THINK - reason about what to do next
2. ACT - call a tool or finish
3. OBSERVE - process the result
4. Repeat until task complete
"""

import re
from typing import Callable

from dotenv import load_dotenv

from .schemas import AgentState, AgentAction, AgentResult, AgentConfig

load_dotenv()


# prompts for the ReAct loop
REACT_SYSTEM_PROMPT = """You are an AI agent that solves tasks step by step using available tools.

For each step, respond in this EXACT format:

THOUGHT: [Your reasoning about what to do next]
ACTION: [tool_name(param="value", param2="value2")]

When you have enough information to answer the task, use:
THOUGHT: [Why you're ready to finish]
ACTION: finish(answer="your complete answer here")

Available tools:
{tool_descriptions}

Rules:
1. ALWAYS start with THOUGHT explaining your reasoning
2. ALWAYS provide exactly ONE action after your thought
3. Use finish(answer="...") ONLY when you have enough information
4. Never make up information - use tools to get real data
5. Keep answers helpful and complete
"""

REACT_USER_PROMPT = """Task: {task}

{history}

Now provide your next THOUGHT and ACTION:"""


class LLMClient:
    """Handles LLM API calls for OpenAI and Anthropic."""

    def __init__(self, provider: str, model: str, temperature: float):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy initialization of API client."""
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            else:
                from anthropic import Anthropic
                self._client = Anthropic()
        return self._client

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return response text."""
        client = self._get_client()

        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        else:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text


class ResponseParser:
    """Parses LLM text responses into structured AgentAction objects."""

    def parse(self, response_text: str) -> AgentAction:
        """Parse LLM response into an AgentAction."""
        thought = self._extract_thought(response_text)
        action_string = self._extract_action_string(response_text)

        if action_string is None:
            return AgentAction(thought=thought, is_final=False)

        if action_string.startswith("finish"):
            return self._parse_finish_action(thought, action_string)

        return self._parse_tool_call(thought, action_string)

    def _extract_thought(self, response_text: str) -> str:
        """Extract THOUGHT from response."""
        pattern = r"THOUGHT:\s*(.+?)(?=ACTION:|$)"
        match = re.search(pattern, response_text, re.DOTALL)
        return match.group(1).strip() if match else "No thought provided"

    def _extract_action_string(self, response_text: str) -> str | None:
        """Extract ACTION from response."""
        pattern = r"ACTION:\s*(.+?)$"
        match = re.search(pattern, response_text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _parse_finish_action(self, thought: str, action_string: str) -> AgentAction:
        """Parse finish(answer="...") action."""
        pattern = r'finish\s*\(\s*answer\s*=\s*["\'](.+?)["\']\s*\)'
        match = re.match(pattern, action_string, re.DOTALL)

        if match:
            return AgentAction(
                thought=thought,
                tool_name="finish",
                tool_args={"answer": match.group(1)},
                is_final=True,
            )
        return AgentAction(thought=thought, is_final=False)

    def _parse_tool_call(self, thought: str, action_string: str) -> AgentAction:
        """Parse tool_name(args) action."""
        tool_pattern = r'(\w+)\s*\((.+?)\)'
        tool_match = re.match(tool_pattern, action_string, re.DOTALL)

        if not tool_match:
            return AgentAction(thought=thought, is_final=False)

        tool_name = tool_match.group(1)
        arguments = self._parse_arguments(tool_match.group(2))

        return AgentAction(
            thought=thought,
            tool_name=tool_name,
            tool_args=arguments,
            is_final=False,
        )

    def _parse_arguments(self, arguments_string: str) -> dict[str, str]:
        """Parse key="value" pairs into a dictionary."""
        arguments = {}
        pattern = r'(\w+)\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(pattern, arguments_string):
            arguments[match.group(1)] = match.group(2)
        return arguments


class ReActAgent:
    """
    ReAct agent that reasons and acts in a loop.

    The main orchestrator that coordinates LLM calls and tool execution.

    Usage:
        agent = ReActAgent(tools, tool_descriptions, config)
        result = agent.run("What's the weather in Tokyo?")
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str]],
        tool_descriptions: dict[str, str],
        config: AgentConfig | None = None,
    ):
        self.tools = tools
        self.tool_descriptions = tool_descriptions
        self.config = config if config is not None else AgentConfig()

        self.state = AgentState.PENDING
        self.actions: list[AgentAction] = []

        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
        )
        self.parser = ResponseParser()

    def run(self, task: str) -> AgentResult:
        """Run the agent on a task until completion."""
        self.state = AgentState.RUNNING
        self.actions = []

        if self.config.verbose:
            self._print_header(task)

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # THINK
            self.state = AgentState.THINKING
            try:
                action = self._think(task)
            except Exception as error:
                return self._handle_error(error, iteration)

            if self.config.verbose:
                self._print_action(action)

            # ACT
            self.state = AgentState.ACTING
            observation = self._act(action)
            action.observation = observation

            if self.config.verbose and not action.is_final:
                print(f"OBSERVATION: {observation}")

            # OBSERVE
            self.state = AgentState.OBSERVING
            self.actions.append(action)

            if action.is_final:
                return self._handle_success(action, iteration)

        return self._handle_timeout()

    def _think(self, task: str) -> AgentAction:
        """Call LLM to decide what to do next."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task)
        response_text = self.llm_client.call(system_prompt, user_prompt)
        return self.parser.parse(response_text)

    def _act(self, action: AgentAction) -> str:
        """Execute the chosen tool."""
        if action.is_final:
            return "Task completed"

        if action.tool_name not in self.tools:
            return f"Error: Unknown tool '{action.tool_name}'. Available: {list(self.tools.keys())}"

        try:
            tool_function = self.tools[action.tool_name]
            result = tool_function(**action.tool_args)
            return str(result)
        except Exception as error:
            return f"Error executing {action.tool_name}: {error}"

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        lines = [f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()]
        lines.append('- finish(answer="..."): Complete the task with final answer')
        tool_descriptions_text = "\n".join(lines)
        return REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions_text)

    def _build_user_prompt(self, task: str) -> str:
        """Build user prompt with task and history."""
        history_text = self._format_history()
        return REACT_USER_PROMPT.format(task=task, history=history_text)

    def _format_history(self) -> str:
        """Format action history for the prompt."""
        if not self.actions:
            return "No previous actions."

        lines = []
        for i, action in enumerate(self.actions):
            lines.append(f"Step {i + 1}:")
            lines.append(f"THOUGHT: {action.thought}")
            lines.append(f"ACTION: {action.action_str}")
            if action.observation:
                lines.append(f"OBSERVATION: {action.observation}")
            lines.append("")
        return "\n".join(lines)

    def _handle_success(self, action: AgentAction, iteration: int) -> AgentResult:
        """Handle successful completion."""
        self.state = AgentState.FINISHED
        answer = action.tool_args.get("answer", "")

        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"  FINAL ANSWER: {answer}")
            print(f"  Completed in {iteration + 1} iteration(s)")
            print("=" * 60)

        return AgentResult(
            answer=answer,
            actions=self.actions,
            iterations=iteration + 1,
            success=True,
        )

    def _handle_error(self, error: Exception, iteration: int) -> AgentResult:
        """Handle LLM error."""
        self.state = AgentState.ERROR
        return AgentResult(
            answer="",
            actions=self.actions,
            iterations=iteration + 1,
            success=False,
            error=f"LLM error: {error}",
        )

    def _handle_timeout(self) -> AgentResult:
        """Handle max iterations reached."""
        self.state = AgentState.TIMEOUT
        if self.config.verbose:
            print(f"\n⚠️  Max iterations ({self.config.max_iterations}) reached!")

        return AgentResult(
            answer="Task could not be completed within iteration limit",
            actions=self.actions,
            iterations=self.config.max_iterations,
            success=False,
            error="Max iterations reached",
        )

    def _print_header(self, task: str) -> None:
        """Print task header."""
        print(f"\n{'=' * 60}")
        print(f"  Agent Task: {task}")
        print("=" * 60)

    def _print_action(self, action: AgentAction) -> None:
        """Print thought and action."""
        print(f"THOUGHT: {action.thought}")
        print(f"ACTION: {action.action_str}")
