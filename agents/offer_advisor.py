"""
Offer Advisor - specialist for offer stage performance and acceptance optimization.

This agent excels at:
- Analyzing offer acceptance rates
- Identifying offer stage bottlenecks
- Understanding time-to-offer-decision patterns
- Providing recommendations to improve offer close rates
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
)


def create_offer_advisor(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the Offer Advisor agent.

    This specialist handles questions about:
    - Offer acceptance rates ("What's our offer acceptance rate?")
    - Offer timing ("How long do candidates take to decide?")
    - Offer outcomes ("How many offers were declined last quarter?")
    - Closing strategies ("Why are we losing candidates at the offer stage?")

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["offer_advisor"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("offer_advisor", version=prompt_version)

    return Agent(
        name="offer_advisor",
        model=resolved_model,
        description=(
            "Offer stage specialist. Analyzes offer acceptance rates, time-to-decision, "
            "and offer outcomes. Use for questions about offer performance, why candidates "
            "decline offers, offer timing optimization, and strategies to improve close rates."
        ),
        instruction=instruction,
        tools=[
            query_metrics_tool,
            multi_query_tool,
            select_visualization_tool(),
            benchmark_comparison_tool,
            benchmark_data_tool,
            request_specialist_handoff_tool,
        ],
        planner=build_thinking_planner("offer_advisor"),
        before_model_callback=build_structured_output_callback("offer_advisor"),
        generate_content_config=build_generate_content_config(),
    )
