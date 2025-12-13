"""
Research Agent - Specialist for finding information.

This agent is called when the orchestrator needs to gather facts,
search for data, or find information about a topic.

Example tasks:
    - "Find information about Python async programming"
    - "Research the population of Tokyo"
    - "Look up the latest AI trends"
"""

from inspect import cleandoc

from schemas import AgentProfile
from .base_specialist import BaseSpecialist


# default profile for research specialist
RESEARCH_PROFILE = AgentProfile(
    name="research_agent",
    role="Research Specialist",
    capabilities=[
        "web search",
        "data gathering",
        "fact finding",
        "information retrieval",
    ],
    system_prompt=cleandoc("""
        You are a research specialist. Your job is to:
        1. Find accurate, relevant information about the given topic
        2. Gather facts and data from reliable sources
        3. Present findings in a clear, organized manner
        4. Cite sources when possible

        Focus on accuracy over speed. If you're unsure, say so.
    """).strip(),
)


class ResearchAgent(BaseSpecialist):
    """
    Specialist agent for research and information gathering.

    In mock mode, returns simulated research results.
    In real mode, would use web search tools and LLM reasoning.

    Usage:
        agent = ResearchAgent()  # uses default profile
        result = agent.execute(task="research Python frameworks")
    """

    def __init__(
        self,
        profile: AgentProfile | None = None,
        use_mock: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            profile=profile or RESEARCH_PROFILE,
            use_mock=use_mock,
            verbose=verbose,
        )

    def _execute_mock(self, task: str) -> str:
        """
        Return mock research results.

        Generates realistic but fake research output based on the task.
        """
        task_lower = task.lower()

        if "python" in task_lower:
            return cleandoc("""
                Research findings on Python:

                1. **Overview**: Python is a high-level programming language known for readability and versatility.

                2. **Key Facts**:
                   - Created by Guido van Rossum in 1991
                   - Currently at version 3.12 (as of 2024)
                   - Used in web development, data science, AI/ML, automation

                3. **Popular Frameworks**:
                   - Web: Django, FastAPI, Flask
                   - Data: Pandas, NumPy, Polars
                   - AI/ML: PyTorch, TensorFlow, scikit-learn

                4. **Trends**: Growing adoption in AI/ML applications, async programming with asyncio.

                Sources: python.org, Stack Overflow Developer Survey 2024
            """).strip()

        elif "ai" in task_lower or "artificial intelligence" in task_lower:
            return cleandoc("""
                Research findings on AI:

                1. **Current State**: AI has seen rapid advancement, especially in large language models (LLMs).

                2. **Key Technologies**:
                   - Transformers architecture (2017)
                   - GPT series from OpenAI
                   - Claude from Anthropic
                   - Open source models (Llama, Mistral)

                3. **Applications**:
                   - Code generation and assistance
                   - Content creation
                   - Data analysis
                   - Customer service automation

                4. **Trends**: Multi-modal models, AI agents, smaller efficient models.

                Sources: arxiv.org, OpenAI blog, Anthropic research
            """).strip()

        elif "weather" in task_lower:
            return cleandoc("""
                Research findings on weather:

                1. **Current Conditions**: Variable by location, recommend checking local forecast.

                2. **Key Weather Services**:
                   - National Weather Service (US)
                   - Met Office (UK)
                   - Japan Meteorological Agency
                   - OpenWeatherMap API

                3. **Weather Patterns**: Climate varies significantly by region and season.

                Sources: weather.gov, metoffice.gov.uk
            """).strip()

        else:
            topic = task.replace("research", "").replace("find", "").strip()
            return cleandoc(f"""
                Research findings on "{topic}":

                1. **Overview**: {topic} is a topic that requires further investigation.

                2. **Key Points**:
                   - Multiple perspectives exist on this topic
                   - Further research recommended for specific details
                   - Consider consulting domain experts

                3. **Recommended Sources**:
                   - Academic databases (Google Scholar)
                   - Industry publications
                   - Expert interviews

                Note: This is a preliminary research summary. More specific questions will yield more detailed results.
            """).strip()
