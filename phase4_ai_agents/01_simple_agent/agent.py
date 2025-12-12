"""
ReAct Agent - Reasoning and Acting in a loop

This module implements the ReAct pattern where an agent:
1. THINKS about what to do next
2. ACTS by calling a tool or finishing
3. OBSERVES the result
4. Repeats until task complete

Module Structure (Java-like separation):
- LLMClient         → handles OpenAI/Anthropic API calls
- ResponseParser    → parses LLM text responses into structured data
- ReActAgent        → orchestrates the think-act-observe loop

Run with: uv run python phase4_ai_agents/01_simple_agent/agent.py
"""

import json
import re
from typing import Callable

from dotenv import load_dotenv

from schemas import AgentState, AgentAction, AgentResult, AgentConfig

# load environment variables from .env file (like reading .properties in Java)
load_dotenv()


# ─────────────────────────────────────────────────────────────
# PROMPTS (Constants - like static final String in Java)
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# LLM CLIENT (Like a Service class in Java)
# ─────────────────────────────────────────────────────────────


class LLMClient:
    """
    handles LLM API calls for different providers

    Java equivalent: This is like a Service class that wraps external API calls.
    Similar to how you might have an OpenAIService or AnthropicService in Java.
    """

    def __init__(self, provider: str, model: str, temperature: float):
        """
        initialize the LLM client

        Args:
            provider: "openai" or "anthropic"
            model: model name (e.g., "gpt-4o-mini")
            temperature: creativity level (0.0 = deterministic, 1.0 = creative)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # client will be created on first use (lazy initialization)
        # this is like @Lazy in Spring
        self._client = None

    def _create_client(self):
        """
        create the actual API client

        we import here (not at top) to avoid loading heavy libraries
        until they're actually needed - this speeds up startup
        """
        if self.provider == "openai":
            # import only when needed
            from openai import OpenAI
            return OpenAI()
        else:
            from anthropic import Anthropic
            return Anthropic()

    def _get_client(self):
        """get or create client (lazy initialization pattern)"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """
        call the LLM and return the response text

        Args:
            system_prompt: instructions for the AI
            user_prompt: the user's message/task

        Returns:
            the AI's response as a string
        """
        client = self._get_client()

        if self.provider == "openai":
            return self._call_openai(client, system_prompt, user_prompt)
        else:
            return self._call_anthropic(client, system_prompt, user_prompt)

    def _call_openai(self, client, system_prompt: str, user_prompt: str) -> str:
        """call OpenAI API"""
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        # extract text from response object
        # response.choices is a list, we want the first one
        # .message.content is the actual text
        return response.choices[0].message.content

    def _call_anthropic(self, client, system_prompt: str, user_prompt: str) -> str:
        """call Anthropic API"""
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # anthropic returns content as a list of blocks
        # we want the first block's text
        return response.content[0].text


# ─────────────────────────────────────────────────────────────
# RESPONSE PARSER (Like a Parser/Converter class in Java)
# ─────────────────────────────────────────────────────────────


class ResponseParser:
    """
    parses LLM text responses into structured AgentAction objects

    Java equivalent: This is like a Parser or Converter class.
    It takes unstructured text and converts it to a typed object.

    The LLM returns text like:
        THOUGHT: I need to check the weather
        ACTION: get_weather(city="Tokyo")

    This parser extracts: thought="I need to check the weather",
                         tool_name="get_weather", tool_args={"city": "Tokyo"}
    """

    def parse(self, response_text: str) -> AgentAction:
        """
        parse LLM response into an AgentAction

        Args:
            response_text: raw text from LLM

        Returns:
            AgentAction with thought, tool_name, tool_args, etc.
        """
        # step 1: extract the THOUGHT part
        thought = self._extract_thought(response_text)

        # step 2: extract the ACTION part
        action_string = self._extract_action_string(response_text)

        # if no action found, return action with just the thought
        if action_string is None:
            return AgentAction(thought=thought, is_final=False)

        # step 3: check if it's a finish action
        if action_string.startswith("finish"):
            return self._parse_finish_action(thought, action_string)

        # step 4: parse as a tool call
        return self._parse_tool_call(thought, action_string)

    def _extract_thought(self, response_text: str) -> str:
        """
        extract the THOUGHT part from the response

        looks for text between "THOUGHT:" and "ACTION:" (or end of string)

        regex explanation:
        - THOUGHT:      -> literal text "THOUGHT:"
        - (backslash)s* -> zero or more whitespace characters
        - (.+?)         -> capture group: one or more characters (non-greedy)
        - (?=ACTION:|$) -> look ahead: stop at "ACTION:" or end of string
        """
        pattern = r"THOUGHT:\s*(.+?)(?=ACTION:|$)"

        # re.DOTALL makes . match newlines too
        match = re.search(pattern, response_text, re.DOTALL)

        if match:
            # .group(1) gets the first capture group (the part in parentheses)
            # .strip() removes leading/trailing whitespace
            return match.group(1).strip()
        else:
            return "No thought provided"

    def _extract_action_string(self, response_text: str) -> str | None:
        """
        extract the ACTION part from the response

        looks for text after "ACTION:" until end of line

        regex explanation:
        - ACTION:  -> literal text "ACTION:"
        - (backslash)s*  -> zero or more whitespace
        - (.+?)    -> capture group: the action text
        - $        -> end of line (with MULTILINE flag, this is end of any line)
        """
        pattern = r"ACTION:\s*(.+?)$"

        # re.MULTILINE makes $ match end of any line, not just end of string
        match = re.search(pattern, response_text, re.MULTILINE)

        if match:
            return match.group(1).strip()
        else:
            return None

    def _parse_finish_action(self, thought: str, action_string: str) -> AgentAction:
        """
        parse a finish action like: finish(answer="The weather is rainy")

        regex explanation:
        - finish       -> literal "finish"
        - (bksl)s*     -> optional whitespace
        - (bksl)(      -> literal opening parenthesis
        - answer       -> "answer" with optional whitespace around it
        - =            -> literal equals sign
        - ["']         -> opening quote (single or double)
        - (.+?)        -> capture group: the answer text (non-greedy)
        - ["']         -> closing quote
        - (bksl))      -> closing parenthesis
        """
        pattern = r'finish\s*\(\s*answer\s*=\s*["\'](.+?)["\']\s*\)'

        match = re.match(pattern, action_string, re.DOTALL)

        if match:
            answer = match.group(1)
            return AgentAction(
                thought=thought,
                tool_name="finish",
                tool_args={"answer": answer},
                is_final=True,  # this marks the task as complete
            )
        else:
            # couldn't parse finish action, return incomplete action
            return AgentAction(thought=thought, is_final=False)

    def _parse_tool_call(self, thought: str, action_string: str) -> AgentAction:
        """
        parse a tool call like: get_weather(city="Tokyo")

        returns AgentAction with tool_name and tool_args
        """
        # step 1: extract tool name and arguments string
        # pattern: word characters followed by (anything)
        # example: "get_weather(city="Tokyo")" → groups: ("get_weather", 'city="Tokyo"')
        tool_pattern = r'(\w+)\s*\((.+?)\)'
        tool_match = re.match(tool_pattern, action_string, re.DOTALL)

        if not tool_match:
            # couldn't parse tool call
            return AgentAction(thought=thought, is_final=False)

        tool_name = tool_match.group(1)       # e.g., "get_weather"
        arguments_string = tool_match.group(2)  # e.g., 'city="Tokyo"'

        # step 2: extract individual arguments
        # pattern matches: key="value" or key='value'
        # example: city="Tokyo" → groups: ("city", "Tokyo")
        arguments = self._parse_arguments(arguments_string)

        return AgentAction(
            thought=thought,
            tool_name=tool_name,
            tool_args=arguments,
            is_final=False,
        )

    def _parse_arguments(self, arguments_string: str) -> dict[str, str]:
        """
        parse argument string into a dictionary

        example input: 'city="Tokyo", country="Japan"'
        example output: {"city": "Tokyo", "country": "Japan"}

        regex explanation:
        - (bksl)w+     -> capture group 1: the key (word characters)
        - (bksl)s*=(bksl)s* -> equals sign with optional whitespace
        - ["']         -> opening quote (single or double)
        - ([^"']+)     -> capture group 2: the value (anything except quotes)
        - ["']         -> closing quote
        """
        arguments = {}
        pattern = r'(\w+)\s*=\s*["\']([^"\']+)["\']'

        # re.finditer returns all matches (like Java's Matcher.find() in a loop)
        for match in re.finditer(pattern, arguments_string):
            key = match.group(1)    # e.g., "city"
            value = match.group(2)  # e.g., "Tokyo"
            arguments[key] = value

        return arguments


# ─────────────────────────────────────────────────────────────
# REACT AGENT (The main orchestrator class)
# ─────────────────────────────────────────────────────────────


class ReActAgent:
    """
    ReAct agent that reasons and acts in a loop

    This is the main class that orchestrates the agent behavior.
    It uses LLMClient for API calls and ResponseParser for parsing.

    Java equivalent: This is like a Controller or Service class that
    coordinates other components to achieve a goal.

    The ReAct loop:
    ┌─────────────────────────────────────────────────────────────┐
    │  Task ──► THINK ──► ACT ──► OBSERVE ───┐                    │
    │             ▲                          │                    │
    │             └──────────────────────────┘                    │
    │                  (until finish or timeout)                  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str]],
        tool_descriptions: dict[str, str],
        config: AgentConfig | None = None,
    ):
        """
        initialize the agent

        Args:
            tools: dictionary mapping tool name to function
                   example: {"get_weather": get_weather_function}
            tool_descriptions: dictionary mapping tool name to description
                   example: {"get_weather": "Get weather for a city"}
            config: agent configuration (optional, uses defaults if not provided)
        """
        # store the tools and descriptions
        self.tools = tools
        self.tool_descriptions = tool_descriptions

        # use provided config or create default
        # this is like: config != null ? config : new AgentConfig()
        if config is not None:
            self.config = config
        else:
            self.config = AgentConfig()

        # initialize state
        self.state = AgentState.PENDING
        self.actions: list[AgentAction] = []  # history of actions taken

        # create helper objects (like dependency injection)
        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
        )
        self.parser = ResponseParser()

    def run(self, task: str) -> AgentResult:
        """
        run the agent on a task until completion

        this is the main entry point - like a "execute()" method

        Args:
            task: the task to complete (e.g., "What's the weather in Tokyo?")

        Returns:
            AgentResult with the answer and execution history
        """
        # reset state for new run
        self.state = AgentState.RUNNING
        self.actions = []

        # print header if verbose mode
        if self.config.verbose:
            self._print_header(task)

        # main loop: iterate until max_iterations
        for iteration_number in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n--- Iteration {iteration_number + 1} ---")

            # STEP 1: THINK - call LLM to get thought and action
            self.state = AgentState.THINKING
            try:
                action = self._think(task)
            except Exception as error:
                return self._handle_error(error, iteration_number)

            if self.config.verbose:
                self._print_thought_and_action(action)

            # STEP 2: ACT - execute the tool
            self.state = AgentState.ACTING
            observation = self._act(action)
            action.observation = observation

            if self.config.verbose and not action.is_final:
                print(f"OBSERVATION: {observation}")

            # STEP 3: OBSERVE - record and check if done
            self.state = AgentState.OBSERVING
            self.actions.append(action)

            # check if agent decided to finish
            if action.is_final:
                return self._handle_success(action, iteration_number)

        # reached max iterations without finishing
        return self._handle_timeout()

    def _think(self, task: str) -> AgentAction:
        """
        THINK step: call LLM to decide what to do next

        builds prompts, calls LLM, parses response
        """
        # build the prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task)

        # call LLM
        response_text = self.llm_client.call(system_prompt, user_prompt)

        # parse response into structured action
        action = self.parser.parse(response_text)

        return action

    def _act(self, action: AgentAction) -> str:
        """
        ACT step: execute the chosen tool

        Args:
            action: the action to execute

        Returns:
            observation (result of tool execution)
        """
        # if this is a finish action, no tool to execute
        if action.is_final:
            return "Task completed"

        # check if tool exists
        if action.tool_name not in self.tools:
            available_tools = list(self.tools.keys())
            return f"Error: Unknown tool '{action.tool_name}'. Available: {available_tools}"

        # execute the tool
        try:
            tool_function = self.tools[action.tool_name]
            # **action.tool_args unpacks the dict as keyword arguments
            # like calling: tool_function(city="Tokyo")
            result = tool_function(**action.tool_args)
            return str(result)
        except Exception as error:
            return f"Error executing {action.tool_name}: {error}"

    def _build_system_prompt(self) -> str:
        """build the system prompt with tool descriptions"""
        tool_descriptions_text = self._format_tool_descriptions()
        return REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions_text)

    def _build_user_prompt(self, task: str) -> str:
        """build the user prompt with task and history"""
        history_text = self._format_history()
        return REACT_USER_PROMPT.format(task=task, history=history_text)

    def _format_tool_descriptions(self) -> str:
        """format tools as a bullet list for the prompt"""
        lines = []

        # add each tool
        for tool_name, description in self.tool_descriptions.items():
            lines.append(f"- {tool_name}: {description}")

        # add the built-in finish tool
        lines.append('- finish(answer="..."): Complete the task with final answer')

        # join with newlines
        return "\n".join(lines)

    def _format_history(self) -> str:
        """format action history for the prompt"""
        if len(self.actions) == 0:
            return "No previous actions."

        lines = []
        for index, action in enumerate(self.actions):
            step_number = index + 1
            lines.append(f"Step {step_number}:")
            lines.append(f"THOUGHT: {action.thought}")
            lines.append(f"ACTION: {action.action_str}")
            if action.observation is not None:
                lines.append(f"OBSERVATION: {action.observation}")
            lines.append("")  # blank line between steps

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # Result handling methods
    # ─────────────────────────────────────────────────────────

    def _handle_success(self, action: AgentAction, iteration_number: int) -> AgentResult:
        """handle successful completion"""
        self.state = AgentState.FINISHED

        if self.config.verbose:
            answer = action.tool_args.get("answer", "")
            print(f"\n{'=' * 60}")
            print(f"  FINAL ANSWER: {answer}")
            print(f"  Completed in {iteration_number + 1} iteration(s)")
            print("=" * 60)

        return AgentResult(
            answer=action.tool_args.get("answer", ""),
            actions=self.actions,
            iterations=iteration_number + 1,
            success=True,
        )

    def _handle_error(self, error: Exception, iteration_number: int) -> AgentResult:
        """handle LLM error"""
        self.state = AgentState.ERROR
        return AgentResult(
            answer="",
            actions=self.actions,
            iterations=iteration_number + 1,
            success=False,
            error=f"LLM error: {error}",
        )

    def _handle_timeout(self) -> AgentResult:
        """handle max iterations reached"""
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
        """print task header"""
        print(f"\n{'=' * 60}")
        print(f"  Agent Task: {task}")
        print("=" * 60)

    def _print_thought_and_action(self, action: AgentAction) -> None:
        """print thought and action"""
        print(f"THOUGHT: {action.thought}")
        print(f"ACTION: {action.action_str}")


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────


def main():
    """quick demo of ReActAgent"""
    print("=" * 60)
    print("  ReActAgent Demo")
    print("=" * 60)

    # define simple tools (these are mock implementations)
    def get_weather(city: str) -> str:
        """mock weather API - returns fake weather data"""
        weather_data = {
            "tokyo": {"temp": 18, "conditions": "rainy", "humidity": 85},
            "paris": {"temp": 22, "conditions": "sunny", "humidity": 45},
            "new york": {"temp": 15, "conditions": "cloudy", "humidity": 60},
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            data = weather_data[city_lower]
            return json.dumps(data)
        return f"Weather data not available for {city}"

    def calculate(expression: str) -> str:
        """safe calculator - evaluates math expressions"""
        try:
            # only allow safe characters (numbers and basic operators)
            allowed_characters = set("0123456789+-*/.() ")
            for char in expression:
                if char not in allowed_characters:
                    return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as error:
            return f"Error: {error}"

    # create the tools dictionary
    tools = {
        "get_weather": get_weather,
        "calculate": calculate,
    }

    # create tool descriptions (shown to the LLM)
    tool_descriptions = {
        "get_weather": 'get_weather(city="city name") - Get weather for a city',
        "calculate": 'calculate(expression="math expr") - Calculate a math expression',
    }

    # create agent with configuration
    config = AgentConfig(max_iterations=5, verbose=True)
    agent = ReActAgent(
        tools=tools,
        tool_descriptions=tool_descriptions,
        config=config,
    )

    # run a task
    result = agent.run("What's the weather in Tokyo? Should I bring an umbrella?")

    # print final results
    print(f"\nResult: {result.answer}")
    print(f"Success: {result.success}")
    print(f"Tool calls: {result.total_tool_calls}")


if __name__ == "__main__":
    main()