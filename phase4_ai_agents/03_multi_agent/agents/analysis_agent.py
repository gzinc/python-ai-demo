"""
Analysis Agent - Specialist for analyzing data and information.

This agent is called when the orchestrator needs to analyze data,
identify patterns, draw conclusions, or provide insights.

Example tasks:
    - "Analyze the research findings and identify key trends"
    - "Compare Python vs Java for enterprise development"
    - "Evaluate the pros and cons of microservices"
"""

from inspect import cleandoc

from schemas import AgentProfile
from .base_specialist import BaseSpecialist


# default profile for analysis specialist
ANALYSIS_PROFILE = AgentProfile(
    name="analysis_agent",
    role="Analysis Specialist",
    capabilities=[
        "data analysis",
        "pattern recognition",
        "comparison",
        "insight generation",
        "evaluation",
    ],
    system_prompt=cleandoc("""
        You are an analysis specialist. Your job is to:
        1. Examine data and information critically
        2. Identify patterns, trends, and relationships
        3. Draw logical conclusions from evidence
        4. Provide balanced, objective analysis

        Be thorough but concise. Support conclusions with evidence.
    """).strip(),
)


class AnalysisAgent(BaseSpecialist):
    """
    Specialist agent for analysis and insight generation.

    In mock mode, returns simulated analysis results.
    In real mode, would use LLM reasoning to analyze provided data.

    Usage:
        agent = AnalysisAgent()
        result = agent.execute(task="analyze the market trends")
    """

    def __init__(
        self,
        profile: AgentProfile | None = None,
        use_mock: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            profile=profile or ANALYSIS_PROFILE,
            use_mock=use_mock,
            verbose=verbose,
        )

    def _execute_mock(self, task: str) -> str:
        """
        Return mock analysis results.

        Generates realistic but fake analysis output based on the task.
        """
        task_lower = task.lower()

        if "compare" in task_lower or "vs" in task_lower:
            return cleandoc("""
                Comparative Analysis:

                **Comparison Framework**:
                | Aspect | Option A | Option B |
                |--------|----------|----------|
                | Performance | High | Moderate |
                | Ease of Use | Moderate | High |
                | Ecosystem | Extensive | Growing |
                | Cost | Low | Medium |

                **Key Findings**:
                1. Both options have distinct strengths
                2. Choice depends on specific use case requirements
                3. Option A better for performance-critical applications
                4. Option B better for rapid development

                **Recommendation**: Consider hybrid approach or evaluate based on team expertise.
            """).strip()

        elif "trend" in task_lower or "pattern" in task_lower:
            return cleandoc("""
                Trend Analysis:

                **Identified Patterns**:
                1. **Upward Trend**: Increasing adoption over past 3 years
                2. **Cyclical Pattern**: Peaks during Q4, dips in Q1
                3. **Emerging Shift**: Movement toward automated solutions

                **Statistical Summary**:
                - Growth rate: ~15% YoY
                - Market penetration: 45%
                - User satisfaction: 4.2/5.0

                **Implications**:
                - Continue current strategy with minor adjustments
                - Invest in automation capabilities
                - Monitor competitor responses

                **Confidence Level**: Medium-High (based on available data)
            """).strip()

        elif "pro" in task_lower and "con" in task_lower:
            return cleandoc("""
                Pros and Cons Analysis:

                **Advantages** ✅:
                1. Improved efficiency and productivity
                2. Better scalability for growing needs
                3. Cost savings in the long term
                4. Enhanced user experience

                **Disadvantages** ❌:
                1. Initial learning curve
                2. Upfront investment required
                3. Potential integration challenges
                4. Dependency on external factors

                **Risk Assessment**:
                - Low risk: Technical implementation
                - Medium risk: Organizational adoption
                - High risk: Market timing

                **Net Assessment**: Benefits outweigh costs for most use cases.
            """).strip()

        elif "evaluat" in task_lower or "assess" in task_lower:
            return cleandoc("""
                Evaluation Analysis:

                **Assessment Criteria**:
                1. Effectiveness: 8/10
                2. Efficiency: 7/10
                3. Feasibility: 9/10
                4. Impact: 8/10

                **Detailed Findings**:
                - Strengths: Clear value proposition, proven methodology
                - Weaknesses: Resource requirements, timeline constraints
                - Opportunities: Market expansion, partnership potential
                - Threats: Competitive pressure, regulatory changes

                **Overall Score**: 8.0/10 (Recommended with conditions)

                **Conditions for Success**:
                1. Adequate resource allocation
                2. Stakeholder alignment
                3. Clear success metrics
            """).strip()

        else:
            # generic analysis response
            return cleandoc(f"""
                Analysis of "{task}":

                **Analytical Framework Applied**: SWOT + Quantitative Assessment

                **Key Observations**:
                1. Multiple factors influence this topic
                2. Data suggests moderate complexity
                3. Further analysis may be beneficial

                **Preliminary Conclusions**:
                - The subject merits attention based on initial analysis
                - Recommend gathering additional data points
                - Consider multiple perspectives before final decision

                **Next Steps**:
                1. Gather more specific data
                2. Consult domain experts
                3. Run scenario analysis

                **Confidence Level**: Medium (preliminary analysis)
            """).strip()
