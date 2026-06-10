"""
General Analyst - fallback for non-specialist queries.

This agent handles:
- Basic metric lookups
- Source/channel performance analysis
- General hiring rates and volumes
- Exploratory data questions
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
    list_knowledge_tool,
)


def create_general_analyst(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the General Analyst agent.

    This agent handles general recruitment metrics questions that
    don't fall into specialist domains like pipeline analysis.

    Handles questions about:
    - Basic metric lookups ("What's our hire rate?")
    - Source performance ("Which sources are best?")
    - Hiring volumes ("How many applications this month?")
    - General trends and comparisons

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["general_analyst"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("general_analyst", version=prompt_version)

    return Agent(
        name="general_analyst",
        model=resolved_model,
        description=(
            "General recruitment analyst. Handles basic metrics queries, "
            "source/channel analysis, hiring rates, volumes, and exploratory "
            "questions. Use when the question doesn't fit a specialist domain "
            "like pipeline analysis."
        ),
        instruction=instruction,
        tools=[
            query_metrics_tool,
            multi_query_tool,
            select_visualization_tool(),
            benchmark_comparison_tool,
            benchmark_data_tool,
            best_practice_tool,
            list_knowledge_tool,
            request_specialist_handoff_tool,
        ],
        planner=build_thinking_planner("general_analyst"),
        before_model_callback=build_structured_output_callback("general_analyst"),
        generate_content_config=build_generate_content_config(),
    )
