"""
Sourcing Strategist - specialist for source channel optimization.

This agent excels at:
- Analyzing source ROI and channel effectiveness
- Comparing source quality vs. quantity metrics
- Identifying underperforming and underutilized sources
- Recommending sourcing budget allocation
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
    multi_query_tool,
    query_metrics_tool,
    request_specialist_handoff_tool,
    select_visualization_tool,
)
from tools.knowledge_tools import (
    benchmark_comparison_tool,
    benchmark_data_tool,
    best_practice_tool,
)


def create_sourcing_strategist(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the Sourcing Strategist agent.

    This specialist handles questions about:
    - Source ROI ("Which sources give us the best hires?")
    - Channel performance ("How is LinkedIn performing?")
    - Source quality vs. volume ("Which sources have the highest hire rate?")
    - Sourcing strategy ("Where should we focus our recruiting efforts?")

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["sourcing_strategist"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("sourcing_strategist", version=prompt_version)

    return Agent(
        name="sourcing_strategist",
        model=resolved_model,
        description=(
            "Sourcing strategy specialist. Analyzes recruitment source effectiveness, "
            "channel ROI, and source quality metrics. Use for questions about which "
            "sources perform best, source hire rates, referral vs. job board comparison, "
            "and sourcing budget optimization."
        ),
        instruction=instruction,
        tools=[
            query_metrics_tool,
            multi_query_tool,
            select_visualization_tool(),
            benchmark_comparison_tool,
            benchmark_data_tool,
            best_practice_tool,
            request_specialist_handoff_tool,
        ],
        planner=build_thinking_planner("sourcing_strategist"),
        before_model_callback=build_structured_output_callback("sourcing_strategist"),
        generate_content_config=build_generate_content_config(),
    )
