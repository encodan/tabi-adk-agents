"""
Interviewing Coach - specialist for interview process efficiency.

This agent excels at:
- Analyzing interview stage duration and efficiency
- Identifying interview process bottlenecks
- Optimizing interview scheduling and flow
- Improving candidate experience through faster processes
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


def create_interviewing_coach(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the Interviewing Coach agent.

    This specialist handles questions about:
    - Interview efficiency ("How long is our interview process?")
    - Stage optimization ("Which interview stages are slowest?")
    - Process improvement ("How can we speed up interviews?")
    - Interviewer effectiveness ("Are we scheduling interviews quickly enough?")

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["interviewing_coach"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("interviewing_coach", version=prompt_version)

    return Agent(
        name="interviewing_coach",
        model=resolved_model,
        description=(
            "Interview process specialist. Analyzes interview stage efficiency, "
            "scheduling patterns, and process optimization. Use for questions about "
            "interview duration, stage-by-stage timing, interviewer capacity, "
            "and how to run a faster, better interview process."
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
        planner=build_thinking_planner("interviewing_coach"),
        before_model_callback=build_structured_output_callback("interviewing_coach"),
        generate_content_config=build_generate_content_config(),
    )
