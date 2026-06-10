"""
Data Scientist - specialist for statistical analysis and predictive modeling.

This agent excels at:
- Predictive modeling (offer acceptance, time-to-hire, candidate success)
- Anomaly detection (flagging unusual metrics, sudden changes)
- Statistical significance testing (hypothesis tests, confidence intervals)
- Advanced segmentation (cohort analysis, clustering, pattern identification)
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
    distribution_analysis_tool,
    distribution_query_tool,
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
from tools.statistical_tools import (
    anomaly_detection_tool,
    cohort_analysis_tool,
    prediction_tool,
    statistical_test_tool,
)


def create_data_scientist(
    prompt_version: str = "v3.1",
    model: str | None = None,
) -> Agent:
    """
    Create the Data Scientist agent.

    This specialist handles questions about:
    - Predictive modeling ("Will we hit our target?", "What's the offer acceptance probability?")
    - Anomaly detection ("Is this change significant?", "Are there any outliers?")
    - Statistical testing ("Is this difference real?", "What's the confidence level?")
    - Cohort analysis ("How do Q1 hires compare to Q2?", "What patterns exist?")

    Args:
        prompt_version: Version of the system prompt to use
        model: Gemini model to use for this agent (None = use config default)

    Returns:
        Configured ADK Agent instance
    """
    cfg = get_config().models.agents["data_scientist"]
    resolved_model = model or cfg.model

    instruction = get_agent_prompt("data_scientist", version=prompt_version)

    return Agent(
        name="data_scientist",
        model=resolved_model,
        description=(
            "Data science and statistical analysis specialist. Performs predictive "
            "modeling for offer acceptance and time-to-hire, anomaly detection for "
            "unusual metrics, statistical significance testing for A/B comparisons, "
            "and advanced cohort segmentation. Use for questions about predictions, "
            "whether changes are significant, identifying outliers, or pattern discovery."
        ),
        instruction=instruction,
        tools=[
            query_metrics_tool,
            multi_query_tool,
            select_visualization_tool(),
            benchmark_comparison_tool,
            benchmark_data_tool,
            statistical_test_tool,
            anomaly_detection_tool,
            prediction_tool,
            cohort_analysis_tool,
            distribution_query_tool,
            distribution_analysis_tool,
            get_planning_context_tool,
            compute_goal_attainment_tool,
            request_specialist_handoff_tool,
        ],
        planner=build_thinking_planner("data_scientist"),
        before_model_callback=build_structured_output_callback("data_scientist"),
        generate_content_config=build_generate_content_config(),
    )
