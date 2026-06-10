"""
Knowledge tools for ADK agents.

Provides tool functions that wrap the knowledge loader API, allowing agents
to dynamically query benchmarks and best practices at runtime.

NOTE (public showcase): the production knowledge pack (a research-curated YAML
benchmark/best-practice corpus loaded via ``tabi_analytics.knowledge.loader``)
is excluded from this repository. The loader API is reproduced below as a small
synthetic stub so these tools stay callable end-to-end. Benchmark values are
illustrative, not sourced.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool
from google.genai import types

from tools.tool_tracer import trace_tool

# ---------------------------------------------------------------------------
# [public-repo stub] Synthetic recruitment-benchmark knowledge base.
# Stands in for the excluded ``tabi_analytics.knowledge.loader`` API.
# ---------------------------------------------------------------------------

_SYNTHETIC_BENCHMARKS: dict[str, dict[str, Any]] = {
    "time_to_hire": {
        "overall": {"median": 30.0, "p25": 21.0, "p75": 44.0, "unit": "days"},
        "interpretation": "Lower is better. ~30 days is a healthy median; >44 days warrants attention.",
    },
    "time_to_fill": {
        "overall": {"median": 36.0, "p25": 25.0, "p75": 52.0, "unit": "days"},
        "interpretation": "Org-side metric (post to hire). ~36 days is typical.",
    },
    "conversion_rates": {
        "overall": {"median": 18.0, "p25": 12.0, "p75": 28.0, "unit": "percent"},
        "interpretation": "Application-to-hire conversion. Healthy pipelines sit near 18%.",
    },
    "offer_acceptance": {
        "overall": {"median": 80.0, "p25": 70.0, "p75": 90.0, "unit": "percent"},
        "interpretation": "Acceptance below ~70% suggests comp or timing issues.",
    },
    "cost_per_hire": {
        "overall": {"median": 4200.0, "p25": 2800.0, "p75": 6500.0, "unit": "usd"},
        "interpretation": "Blended cost per hire across sources.",
    },
    "quality_of_hire": {
        "overall": {"median": 75.0, "p25": 60.0, "p75": 88.0, "unit": "score"},
        "interpretation": "Composite 90-day quality score; higher is better.",
    },
    "source_performance": {
        "overall": {"median": 22.0, "p25": 14.0, "p75": 34.0, "unit": "percent"},
        "interpretation": "Hire rate by channel; referrals typically top-quartile.",
    },
}

_SYNTHETIC_BEST_PRACTICES: dict[str, str] = {
    "sourcing": (
        "# Sourcing Best Practices\n\n"
        "## Key Principles\n"
        "- Diversify channels: referrals convert best, but balance with inbound + outbound.\n"
        "- Nurture passive candidates with a lightweight talent community.\n"
        "## Common Pitfalls\n"
        "- Over-reliance on a single job board.\n"
        "## Metrics to Track\n"
        "- Source-of-hire, channel conversion, cost per source.\n"
    ),
    "dei": (
        "# DEI in Hiring\n\n"
        "## Key Principles\n"
        "- Structure interviews and use consistent rubrics to reduce bias.\n"
        "## Interview Best Practices\n"
        "- Diverse panels; standardized question banks.\n"
        "## Metrics to Track\n"
        "- Funnel pass-through by demographic cohort.\n"
    ),
    "interview_process": (
        "# Interview Process Optimization\n\n"
        "## Key Principles\n"
        "- Minimize loops; aim for a tight, well-scheduled process.\n"
        "## Common Pitfalls\n"
        "- Scheduling delays that stretch time-in-stage.\n"
        "## Metrics to Track\n"
        "- Interviews per hire, stage cycle time.\n"
    ),
}


def list_benchmarks() -> list[str]:
    """Return the available synthetic benchmark categories."""
    return list(_SYNTHETIC_BENCHMARKS.keys())


def list_best_practices() -> list[str]:
    """Return the available synthetic best-practice topics."""
    return list(_SYNTHETIC_BEST_PRACTICES.keys())


def get_benchmark(
    category: str,
    metric: str = "",
    segment: str | None = None,
    segment_value: str | None = None,
) -> dict[str, Any] | None:
    """Return a copy of the benchmark record for ``category`` (or ``None``)."""
    record = _SYNTHETIC_BENCHMARKS.get(category)
    return dict(record) if record else None


def get_benchmark_interpretation(category: str, metric: str = "") -> str | None:
    """Return the human-readable interpretation for a benchmark category."""
    record = _SYNTHETIC_BENCHMARKS.get(category)
    return record.get("interpretation") if record else None


def compare_to_benchmark(
    category: str,
    value: float,
    metric: str = "",
    segment: str | None = None,
    segment_value: str | None = None,
) -> dict[str, Any]:
    """Compare ``value`` against the synthetic benchmark band for ``category``."""
    record = _SYNTHETIC_BENCHMARKS.get(category)
    if record is None:
        return {
            "error": f"No benchmark data found for category: {category}",
            "available_categories": list_benchmarks(),
        }
    overall = record["overall"]
    p25, median, p75 = overall["p25"], overall["median"], overall["p75"]
    # For most recruitment metrics lower is better; for acceptance/quality higher is.
    higher_is_better = category in {
        "offer_acceptance",
        "quality_of_hire",
        "conversion_rates",
        "source_performance",
    }
    if higher_is_better:
        if value >= p75:
            percentile, assessment = "top_quartile", "good"
        elif value >= p25:
            percentile, assessment = "median_range", "average"
        else:
            percentile, assessment = "bottom_quartile", "concerning"
    else:
        if value <= p25:
            percentile, assessment = "top_quartile", "good"
        elif value <= p75:
            percentile, assessment = "median_range", "average"
        else:
            percentile, assessment = "bottom_quartile", "concerning"
    return {
        "category": category,
        "value": value,
        "percentile_estimate": percentile,
        "benchmark_median": median,
        "benchmark_p25": p25,
        "benchmark_p75": p75,
        "assessment": assessment,
        "context": (
            f"A value of {value} for {category} is {assessment} "
            f"(median benchmark {median} {overall.get('unit', '')})."
        ),
    }


def get_all_benchmarks_summary() -> dict[str, Any]:
    """Return a compact median snapshot across all synthetic benchmarks."""
    return {
        name: {"median": rec["overall"]["median"], "unit": rec["overall"].get("unit")}
        for name, rec in _SYNTHETIC_BENCHMARKS.items()
    }


def get_best_practice(topic: str) -> str | None:
    """Return the full synthetic best-practice guide for ``topic``."""
    return _SYNTHETIC_BEST_PRACTICES.get(topic)


def get_best_practice_section(topic: str, section: str) -> str | None:
    """Return a single ``## section`` from a best-practice guide, if present."""
    content = _SYNTHETIC_BEST_PRACTICES.get(topic)
    if content is None:
        return None
    marker = f"## {section}"
    if marker not in content:
        return None
    tail = content.split(marker, 1)[1]
    # Stop at the next H2 heading.
    next_h2 = tail.find("\n## ")
    body = tail if next_h2 == -1 else tail[:next_h2]
    return (marker + body).strip()


@trace_tool("get_benchmark_comparison")
def get_benchmark_comparison(
    category: str,
    value: float,
    metric: str = "",
    industry: str = "",
    role_level: str = "",
) -> dict:
    """
    Compare a customer's metric value against industry benchmarks.

    Use this tool to contextualize customer data by comparing it to industry
    standards. This helps determine if a metric is performing well, average,
    or needs attention.

    Args:
        category: Benchmark category. Available categories:
            - "time_to_fill": Days from job posting to hire (org-side)
            - "time_to_hire": Days from application to hire (candidate-side)
            - "conversion_rates": Stage conversion percentages
            - "offer_acceptance": Offer acceptance rate
            - "cost_per_hire": Hiring costs by source/method
            - "quality_of_hire": Quality indicators
            - "source_performance": Source channel effectiveness
        value: The customer's metric value to compare
        metric: Specific metric within category (for multi-metric files like
            conversion_rates: "application_to_screen", "interview_to_offer", etc.)
        industry: Industry segment for industry-specific benchmarks:
            "technology", "healthcare", "finance", "hospitality", etc.
        role_level: Role level for role-specific benchmarks:
            "entry_level", "mid_level", "senior", "executive"

    Returns:
        Dictionary with comparison results:
        - percentile_estimate: "top_quartile", "median_range", or "bottom_quartile"
        - benchmark_median: The median benchmark value
        - benchmark_p25: 25th percentile (top performers)
        - benchmark_p75: 75th percentile (underperformers)
        - assessment: "good", "average", or "concerning"
        - context: Human-readable explanation
        - interpretation: Guidance on what good/concerning means

    Examples:
        # Compare time to hire against overall benchmark
        get_benchmark_comparison("time_to_hire", 52)
        # Returns: {"assessment": "concerning", "benchmark_median": 27.5, ...}

        # Compare against technology industry benchmark
        get_benchmark_comparison("time_to_hire", 35, industry="technology")
        # Returns: {"assessment": "average", ...}

        # Compare offer acceptance rate
        get_benchmark_comparison("offer_acceptance", 78)
    """
    # Map industry/role_level to loader's segment/segment_value
    segment = None
    segment_value = None
    if industry:
        segment = "by_industry"
        segment_value = industry
    elif role_level:
        segment = "by_role_level"
        segment_value = role_level

    result = compare_to_benchmark(
        category=category,
        value=value,
        metric=metric,
        segment=segment,
        segment_value=segment_value,
    )

    # Add interpretation if available
    interpretation = get_benchmark_interpretation(category, metric)
    if interpretation:
        result["interpretation"] = interpretation

    return result


@trace_tool("get_benchmark_data")
def get_benchmark_data(
    category: str,
    metric: str = "",
    industry: str = "",
    role_level: str = "",
) -> dict:
    """
    Retrieve raw benchmark data for a category.

    Use this tool when you need to display benchmark ranges to the user or
    explain what "good" looks like for a metric without comparing a specific value.

    Args:
        category: Benchmark category (see get_benchmark_comparison for options)
        metric: Specific metric within category (optional)
        industry: Industry segment (optional)
        role_level: Role level (optional)

    Returns:
        Dictionary with benchmark data including:
        - overall: {median, p25, p75, unit}
        - by_industry: Industry-specific benchmarks (if available)
        - by_role_level: Role-level benchmarks (if available)
        - interpretation: Guidance on interpreting values
        - metadata: Data sources, year, confidence level

    Examples:
        # Get all time to hire benchmarks
        get_benchmark_data("time_to_hire")

        # Get technology industry benchmarks
        get_benchmark_data("time_to_hire", industry="technology")

        # Get conversion rate benchmarks
        get_benchmark_data("conversion_rates")
    """
    segment = None
    segment_value = None
    if industry:
        segment = "by_industry"
        segment_value = industry
    elif role_level:
        segment = "by_role_level"
        segment_value = role_level

    data = get_benchmark(category, metric, segment, segment_value)

    if data is None:
        return {
            "error": f"No benchmark data found for category: {category}",
            "available_categories": list_benchmarks(),
        }

    # Add interpretation if available
    interpretation = get_benchmark_interpretation(category, metric)
    if interpretation:
        data["interpretation"] = interpretation

    return data


@trace_tool("get_best_practice_guidance")
def get_best_practice_guidance(
    topic: str,
    section: str = "",
) -> dict:
    """
    Retrieve best practice guidance for recommendations.

    Use this tool when providing recommendations to the user. The guidance
    is research-backed and provides actionable advice.

    Args:
        topic: The best practice topic. Available topics:
            - "sourcing": Multi-channel sourcing, passive candidates, employer branding
            - "dei": Diversity, equity, and inclusion in hiring
            - "interview_process": Interview process optimization
        section: Optional section heading to extract (e.g., "Key Principles",
            "Common Pitfalls", "Metrics to Track"). If not provided, returns
            the full guide.

    Returns:
        Dictionary with:
        - content: The guidance content (markdown format)
        - topic: The requested topic
        - section: The requested section (if any)
        - available_topics: List of all available topics

    Examples:
        # Get full sourcing guide
        get_best_practice_guidance("sourcing")

        # Get specific section
        get_best_practice_guidance("dei", section="Interview Best Practices")
    """
    available_topics = list_best_practices()

    if topic not in available_topics:
        return {
            "error": f"Topic '{topic}' not found",
            "available_topics": available_topics,
            "content": None,
        }

    if section:
        content = get_best_practice_section(topic, section)
        if content is None:
            # Fall back to full content
            content = get_best_practice(topic)
            return {
                "content": content,
                "topic": topic,
                "section": None,
                "note": f"Section '{section}' not found, returning full guide",
                "available_topics": available_topics,
            }
    else:
        content = get_best_practice(topic)

    return {
        "content": content,
        "topic": topic,
        "section": section,
        "available_topics": available_topics,
    }


@trace_tool("list_available_knowledge")
def list_available_knowledge() -> dict:
    """
    List all available knowledge resources.

    Use this tool to discover what benchmarks and best practices are
    available for reference.

    Returns:
        Dictionary with:
        - benchmarks: List of benchmark categories
        - best_practices: List of best practice topics
        - benchmark_summary: Quick overview of key metrics
    """
    return {
        "benchmarks": list_benchmarks(),
        "best_practices": list_best_practices(),
        "benchmark_summary": get_all_benchmarks_summary(),
    }


# =============================================================================
# FunctionTool wrappers with explicit schemas
# =============================================================================
# The ADK auto-generates JSON Schema from Python type hints, which can produce
# constructs like `anyOf` that the Gemini API doesn't support. These wrappers
# provide explicit schemas to avoid compatibility issues.


def _create_function_tool_with_schema(
    func,
    name: str,
    description: str,
    parameters: dict,
) -> FunctionTool:
    """Create a FunctionTool with explicit schema override."""

    class ExplicitSchemaFunctionTool(FunctionTool):
        """FunctionTool that uses explicit schema."""

        def _get_declaration(self) -> types.FunctionDeclaration | None:
            schema = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    key: _convert_property(value) for key, value in parameters["properties"].items()
                },
                required=parameters.get("required", []),
            )
            return types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=schema,
            )

    return ExplicitSchemaFunctionTool(func=func)


def _convert_property(prop: dict) -> types.Schema:
    """Convert a JSON Schema property to Gemini Schema type."""
    prop_type = prop.get("type", "string")

    type_mapping = {
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    schema_type = type_mapping.get(prop_type, types.Type.STRING)

    kwargs = {
        "type": schema_type,
        "description": prop.get("description"),
    }

    if "enum" in prop:
        kwargs["enum"] = prop["enum"]

    if prop_type == "array" and "items" in prop:
        kwargs["items"] = _convert_property(prop["items"])

    return types.Schema(**kwargs)


# Tool schema definitions
_BENCHMARK_COMPARISON_SCHEMA = {
    "properties": {
        "category": {
            "type": "string",
            "description": "Benchmark category: time_to_fill, time_to_hire, conversion_rates, offer_acceptance, cost_per_hire, quality_of_hire, source_performance",
        },
        "value": {
            "type": "number",
            "description": "The customer's metric value to compare",
        },
        "metric": {
            "type": "string",
            "description": "Specific metric within category (optional)",
        },
        "industry": {
            "type": "string",
            "description": "Industry segment: technology, healthcare, finance, hospitality, etc.",
        },
        "role_level": {
            "type": "string",
            "description": "Role level: entry_level, mid_level, senior, executive",
        },
    },
    "required": ["category", "value"],
}

_BENCHMARK_DATA_SCHEMA = {
    "properties": {
        "category": {
            "type": "string",
            "description": "Benchmark category: time_to_fill, time_to_hire, conversion_rates, offer_acceptance, cost_per_hire, quality_of_hire, source_performance",
        },
        "metric": {
            "type": "string",
            "description": "Specific metric within category (optional)",
        },
        "industry": {
            "type": "string",
            "description": "Industry segment: technology, healthcare, finance, hospitality, etc.",
        },
        "role_level": {
            "type": "string",
            "description": "Role level: entry_level, mid_level, senior, executive",
        },
    },
    "required": ["category"],
}

_BEST_PRACTICE_SCHEMA = {
    "properties": {
        "topic": {
            "type": "string",
            "description": "Best practice topic: sourcing, dei, interview_process",
        },
        "section": {
            "type": "string",
            "description": "Optional section heading to extract",
        },
    },
    "required": ["topic"],
}

_LIST_KNOWLEDGE_SCHEMA = {
    "properties": {},
    "required": [],
}


# Create wrapped tool instances
benchmark_comparison_tool = _create_function_tool_with_schema(
    func=get_benchmark_comparison,
    name="get_benchmark_comparison",
    description="Compare a customer's metric value against industry benchmarks to determine if performance is good, average, or concerning.",
    parameters=_BENCHMARK_COMPARISON_SCHEMA,
)

benchmark_data_tool = _create_function_tool_with_schema(
    func=get_benchmark_data,
    name="get_benchmark_data",
    description="Retrieve raw benchmark data for a category to display ranges and explain what good performance looks like.",
    parameters=_BENCHMARK_DATA_SCHEMA,
)

best_practice_tool = _create_function_tool_with_schema(
    func=get_best_practice_guidance,
    name="get_best_practice_guidance",
    description="Retrieve research-backed best practice guidance for providing recommendations on sourcing, DEI, or interview processes.",
    parameters=_BEST_PRACTICE_SCHEMA,
)

list_knowledge_tool = _create_function_tool_with_schema(
    func=list_available_knowledge,
    name="list_available_knowledge",
    description="List all available benchmarks and best practice topics.",
    parameters=_LIST_KNOWLEDGE_SCHEMA,
)

# Re-exported from the statistical-tools module for convenience so callers that
# reach for the analytical knowledge surface find anomaly/cohort tooling here
# too. (In the production tree these live in ``statistical_tools``; the public
# showcase ships them as mock stand-ins — see ``tools.statistical_tools``.)
from tools.statistical_tools import (  # noqa: E402
    anomaly_detection_tool,
    cohort_analysis_tool,
)

__all__ = [
    "benchmark_comparison_tool",
    "benchmark_data_tool",
    "best_practice_tool",
    "list_knowledge_tool",
    "anomaly_detection_tool",
    "cohort_analysis_tool",
]
