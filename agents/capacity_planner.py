"""
Capacity Planner - specialist for hiring forecasts and workload planning.

This agent excels at:
- Analyzing hiring velocity and throughput
- Forecasting hiring capacity and goal attainment
- Assessing pipeline backlog and coverage
- Planning recruiter workload and resource needs
"""

from __future__ import annotations

from google.adk.agents import Agent

from config import build_thinking_planner, get_config
from core.specialist_schema import (
    build_generate_content_config,
    build_structured_output_callback,
)
from agents.prompts import get_agent_prompt
from tools.adk_tools import (
    compute_goal_attainment_tool,
    get_planning_context_tool,
    multi_query_tool,
    query_metrics_tool,
    request_specialist_handoff_tool,
    select_visualization_tool,
)
from tools.knowledge_tools import (
    benchmark_comparison_tool,
    benchmark_data_tool,
)


def create_capacity_planner(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the Capacity Planner agent.

    This specialist handles questions about:
    - Hiring velocity ("How many people are we hiring per month?")
    - Capacity forecasting ("Will we hit our Q4 hiring goal?")
    - Pipeline coverage ("Do we have enough candidates in the pipeline?")
    - Workload planning ("How many reqs can our team handle?")

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["capacity_planner"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("capacity_planner", version=prompt_version)

    return Agent(
        name="capacity_planner",
        model=resolved_model,
        description=(
            "Capacity planning specialist. Analyzes hiring velocity, pipeline coverage, "
            "and forecasting. Use for questions about hiring goals, throughput projections, "
            "recruiter workload, pipeline backlog, and whether the team will meet hiring targets."
        ),
        instruction=instruction,
        tools=[
            query_metrics_tool,
            multi_query_tool,
            select_visualization_tool(),
            benchmark_comparison_tool,
            benchmark_data_tool,
            get_planning_context_tool,
            compute_goal_attainment_tool,
            request_specialist_handoff_tool,
        ],
        planner=build_thinking_planner("capacity_planner"),
        before_model_callback=build_structured_output_callback("capacity_planner"),
        generate_content_config=build_generate_content_config(),
    )
