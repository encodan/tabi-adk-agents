"""
Pipeline Analyst - specialist for pipeline health and bottleneck analysis.

This agent excels at:
- Identifying bottlenecks in the hiring pipeline
- Analyzing time spent in each interview stage
- Diagnosing process efficiency issues
- Breaking down time-to-hire by stage
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
    distribution_analysis_tool,
    distribution_query_tool,
    multi_query_tool,
    query_metrics_tool,
    request_specialist_handoff_tool,
    select_visualization_tool,
)
from tools.knowledge_tools import (
    benchmark_comparison_tool,
    benchmark_data_tool,
)


def create_pipeline_analyst(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the Pipeline Analyst agent.

    This specialist handles questions about:
    - Pipeline bottlenecks ("Where are candidates getting stuck?")
    - Stage duration analysis ("How long in each stage?")
    - Time to fill breakdown ("Why is hiring slow?")
    - Hiring velocity and process efficiency

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["pipeline_analyst"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("pipeline_analyst", version=prompt_version)

    return Agent(
        name="pipeline_analyst",
        model=resolved_model,
        description=(
            "Pipeline health specialist. Analyzes interview stage bottlenecks, "
            "time-to-hire breakdowns, stage durations, and hiring velocity. "
            "Use for questions about where candidates get stuck, slow stages, "
            "pipeline efficiency, or process flow issues."
        ),
        instruction=instruction,
        tools=[
            query_metrics_tool,
            multi_query_tool,
            select_visualization_tool(),
            benchmark_comparison_tool,
            benchmark_data_tool,
            distribution_query_tool,
            distribution_analysis_tool,
            request_specialist_handoff_tool,
        ],
        planner=build_thinking_planner("pipeline_analyst"),
        before_model_callback=build_structured_output_callback("pipeline_analyst"),
        generate_content_config=build_generate_content_config(),
    )
