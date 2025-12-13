"""
Multi-Agent Orchestrator - Coordinates specialist agents to complete tasks.

The orchestrator:
1. Receives a complex task from the user
2. Breaks it down into subtasks
3. Delegates subtasks to specialist agents
4. Collects and synthesizes results
5. Returns a final answer

Key Pattern: Agent-as-Tool
    Specialist agents are registered in the ToolRegistry, allowing
    the orchestrator to call them exactly like regular tools.

Run with: uv run python phase4_ai_agents/03_multi_agent/orchestrator.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

# enable imports from this package
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from schemas import TeamConfig, TeamResult, DelegationResult
from agents import (
    ToolRegistry,
    ToolResult,
    BaseSpecialist,
    ResearchAgent,
    AnalysisAgent,
    WriterAgent,
)


# =============================================================================
# Orchestrator Prompts
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are a team orchestrator. You coordinate specialist agents to complete complex tasks.

Your available specialists:
{specialist_descriptions}

For each step, respond in this exact format:

THOUGHT: [Your reasoning about what to do next - which specialist to use and why]
ACTION: [specialist_name(task="the specific subtask to delegate")]

OR when you have enough information:

THOUGHT: [Your reasoning about why you're done]
ACTION: finish(answer="your final synthesized answer")

Rules:
1. Break complex tasks into subtasks for specialists
2. Use the right specialist for each subtask
3. Synthesize results from multiple specialists when needed
4. Call finish() when you have enough information for the final answer
5. Be efficient - don't call specialists unnecessarily"""

ORCHESTRATOR_USER_PROMPT = """Task: {task}

{history}

What's your next step?"""


# =============================================================================
# Orchestrator Class
# =============================================================================


class MultiAgentOrchestrator:
    """
    Coordinates multiple specialist agents to complete complex tasks.

    The orchestrator runs a loop where it:
    1. Thinks about what to do next
    2. Delegates to a specialist OR finishes
    3. Observes the result
    4. Repeats until done or max iterations reached

    Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                    MultiAgentOrchestrator                      │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │                   ReAct-style Loop                       │  │
    │  │  THINK: "I need to research this topic first"            │  │
    │  │  ACTION: research_agent(task="...")                      │  │
    │  │  OBSERVE: [specialist result]                            │  │
    │  │  ... repeat ...                                          │  │
    │  │  ACTION: finish(answer="...")                            │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                           │                                    │
    │                           ▼                                    │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │                    ToolRegistry                          │  │
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │  │
    │  │  │  Research   │ │  Analysis   │ │   Writer    │         │  │
    │  │  │   Agent     │ │   Agent     │ │   Agent     │         │  │
    │  │  └─────────────┘ └─────────────┘ └─────────────┘         │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────────┘

    Usage:
        orchestrator = MultiAgentOrchestrator()
        result = orchestrator.run("Research AI trends and write a summary")
        print(result.answer)
    """

    def __init__(
        self,
        config: TeamConfig | None = None,
        specialists: list[BaseSpecialist] | None = None,
        use_mock: bool = True,
    ):
        """
        Initialize orchestrator with config and specialist agents.

        Args:
            config: team configuration (uses defaults if None)
            specialists: list of specialist agents (uses defaults if None)
            use_mock: if True, use mock responses instead of real LLM
        """
        self.config = config or TeamConfig()
        self.use_mock = use_mock
        self.registry = ToolRegistry()

        # register specialist agents
        if specialists is None:
            specialists = [
                ResearchAgent(use_mock=use_mock, verbose=self.config.verbose),
                AnalysisAgent(use_mock=use_mock, verbose=self.config.verbose),
                WriterAgent(use_mock=use_mock, verbose=self.config.verbose),
            ]

        for specialist in specialists:
            self.registry.register(specialist)

        # track delegations
        self._delegations: list[DelegationResult] = []
        self._llm_client = None

    def _get_specialist_descriptions(self) -> str:
        """format specialist descriptions for prompt"""
        lines = []
        for defn in self.registry.get_all_definitions():
            lines.append(f"- {defn.name}: {defn.description}")
        return "\n".join(lines)

    def _build_history(self) -> str:
        """format delegation history for prompt"""
        if not self._delegations:
            return "No actions taken yet."

        lines = ["Previous actions:"]
        for i, d in enumerate(self._delegations, 1):
            lines.append(f"\n{i}. THOUGHT: Delegating to {d.agent_name}")
            lines.append(f"   ACTION: {d.agent_name}(task=\"{d.task}\")")
            lines.append(f"   OBSERVATION: {d.result[:200]}...")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple[str, str, dict]:
        """
        Parse orchestrator response into thought, action, and args.

        Returns:
            (thought, action_name, action_args)
        """
        # extract THOUGHT
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # extract ACTION
        action_match = re.search(r"ACTION:\s*(\w+)\((.+?)\)", response, re.DOTALL)
        if not action_match:
            return thought, "", {}

        action_name = action_match.group(1)
        args_str = action_match.group(2)

        # parse arguments (simple key="value" format)
        args = {}
        for match in re.finditer(r'(\w+)=["\'](.+?)["\']', args_str):
            args[match.group(1)] = match.group(2)

        return thought, action_name, args

    def _mock_orchestrator_response(self, task: str, iteration: int) -> str:
        """
        Generate mock orchestrator response for testing.

        Simulates orchestrator decision-making without LLM.
        """
        task_lower = task.lower()

        # first iteration: always research
        if iteration == 1:
            return f"""THOUGHT: I need to start by researching this topic to gather information.
ACTION: research_agent(task="Research {task}")"""

        # second iteration: analyze the research
        if iteration == 2:
            return f"""THOUGHT: Now I have research findings. I should analyze them to extract insights.
ACTION: analysis_agent(task="Analyze the research findings about {task}")"""

        # third iteration: write summary
        if iteration == 3:
            return f"""THOUGHT: I have research and analysis. Now I should synthesize this into a final answer.
ACTION: writer_agent(task="Write a comprehensive summary of the findings about {task}")"""

        # fourth iteration: finish
        return f"""THOUGHT: I have gathered research, completed analysis, and prepared a written summary. I now have enough information to provide a complete answer.
ACTION: finish(answer="Based on my team's work: The research specialist found key information about {task}. The analysis specialist identified important patterns and insights. The writer specialist synthesized everything into a comprehensive summary. The task has been completed successfully with input from all specialists.")"""

    def _call_llm(self, task: str, iteration: int) -> str:
        """
        Call LLM to get orchestrator's next action.

        In mock mode, returns simulated response.
        In real mode, calls the configured LLM provider.
        """
        if self.use_mock:
            return self._mock_orchestrator_response(task, iteration)

        # real LLM call (lazy initialization)
        if self._llm_client is None:
            self._llm_client = self._create_llm_client()

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            specialist_descriptions=self._get_specialist_descriptions()
        )
        user_prompt = ORCHESTRATOR_USER_PROMPT.format(
            task=task,
            history=self._build_history(),
        )

        return self._llm_client.chat(system_prompt, user_prompt)

    def _create_llm_client(self):
        """create LLM client based on config"""
        # inline LLM client to keep module self-contained
        if self.config.provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            class OpenAIClient:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model

                def chat(self, system: str, user: str) -> str:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.0,
                    )
                    return response.choices[0].message.content or ""

            return OpenAIClient(client, self.config.orchestrator_model)

        elif self.config.provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            class AnthropicClient:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model

                def chat(self, system: str, user: str) -> str:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return response.content[0].text

            return AnthropicClient(client, self.config.orchestrator_model)

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def run(self, task: str) -> TeamResult:
        """
        Execute a task using the specialist team.

        Args:
            task: the complex task to complete

        Returns:
            TeamResult with answer and delegation trace
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"  🎯 ORCHESTRATOR: Starting task")
            print(f"{'='*60}")
            print(f"  Task: {task}")
            print(f"  Specialists: {', '.join(self.registry.list_tools())}")

        # reset state
        self._delegations = []
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            if self.config.verbose:
                print(f"\n--- Iteration {iteration} ---")

            # get orchestrator's decision
            response = self._call_llm(task, iteration)
            thought, action_name, action_args = self._parse_response(response)

            if self.config.verbose:
                print(f"  💭 THOUGHT: {thought[:100]}...")
                print(f"  🎬 ACTION: {action_name}({action_args})")

            # check for finish action
            if action_name == "finish":
                answer = action_args.get("answer", "Task completed.")
                if self.config.verbose:
                    print(f"\n{'='*60}")
                    print(f"  ✅ ORCHESTRATOR: Task complete!")
                    print(f"{'='*60}")

                return TeamResult(
                    answer=answer,
                    delegations=self._delegations,
                    iterations=iteration,
                    success=True,
                )

            # delegate to specialist
            if action_name not in self.registry:
                if self.config.verbose:
                    print(f"  ⚠️ Unknown specialist: {action_name}")
                continue

            subtask = action_args.get("task", task)
            result = self.registry.execute(action_name, task=subtask)

            # record delegation
            delegation = DelegationResult(
                agent_name=action_name,
                task=subtask,
                result=result.to_observation(),
                success=result.success,
                error=result.error or "",
            )
            self._delegations.append(delegation)

            # check delegation limit
            if len(self._delegations) >= self.config.max_delegations:
                if self.config.verbose:
                    print(f"\n  ⚠️ Max delegations ({self.config.max_delegations}) reached")
                break

        # reached max iterations
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"  ⏱️ ORCHESTRATOR: Max iterations reached")
            print(f"{'='*60}")

        return TeamResult(
            answer="Task incomplete: reached maximum iterations.",
            delegations=self._delegations,
            iterations=iteration,
            success=False,
            error="Max iterations reached",
        )


# =============================================================================
# Quick Demo
# =============================================================================


def _demo():
    """quick demo of the orchestrator"""
    print("\n" + "=" * 60)
    print("  Multi-Agent Orchestrator Demo")
    print("=" * 60)

    orchestrator = MultiAgentOrchestrator(use_mock=True)

    task = "Research AI trends in 2024 and write a summary report"
    result = orchestrator.run(task)

    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)
    print(f"\nAnswer: {result.answer}")
    print(f"\nDelegations: {result.delegation_summary()}")
    print(f"Iterations: {result.iterations}")
    print(f"Success: {result.success}")


if __name__ == "__main__":
    _demo()
