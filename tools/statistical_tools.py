"""
Statistical analysis tools for the Data Scientist agent.

Provides tools for:
- Statistical hypothesis testing
- Anomaly detection in time series
- Predictive modeling
- Cohort analysis
- Distribution-shape characterization

NOTE (public showcase): in the production TABI platform these tools run real
numpy/scipy implementations (t-tests, Mann-Whitney, IQR/MAD anomaly detection,
linear/exponential forecasting, histogram-based bimodality detection). For the
public showcase the **schemas, signatures and docstrings are preserved
verbatim** (they document the agent's analytical surface) but the bodies are
replaced with lightweight **mock stand-ins** that return plausible synthetic
result dicts of the same shape. No statistical implementation IP is included.
"""

from __future__ import annotations

from typing import Any

import structlog
from google.adk.tools import FunctionTool
from google.genai import types

from tools.tool_tracer import trace_tool

logger = structlog.get_logger(__name__)


@trace_tool("perform_statistical_test")
async def perform_statistical_test(
    group_a_values: list[float],
    group_b_values: list[float],
    test_type: str = "t_test",
    alternative: str = "two_sided",
) -> dict[str, Any]:
    """
    Perform statistical hypothesis testing between two groups.

    Use this tool to determine if the difference between two groups is
    statistically significant. Essential for answering "Is this change real?"
    type questions.

    Args:
        group_a_values: Numeric values for the first group (e.g., Q1 hire rates)
        group_b_values: Numeric values for the second group (e.g., Q2 hire rates)
        test_type: Type of statistical test to perform:
            - "t_test": For comparing means of normally distributed data
            - "mann_whitney": For non-normal data or ordinal comparisons
            - "chi_square": For categorical data comparisons
        alternative: Direction of the hypothesis test:
            - "two_sided": Test if groups are different (default)
            - "greater": Test if group A > group B
            - "less": Test if group A < group B

    Returns:
        Dictionary with:
        - p_value: Probability of observing this difference by chance
        - is_significant: True if p_value < 0.05
        - effect_size: Magnitude of the difference (Cohen's d for t-test)
        - confidence_interval: 95% CI for the difference
        - interpretation: Human-readable explanation
        - group_a_summary: Mean, std, n for group A
        - group_b_summary: Mean, std, n for group B
    """

    # [public-repo stub] Synthetic result — real scipy hypothesis test excluded.
    def _summary(vals: list[float]) -> dict[str, Any]:
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        return {"mean": round(mean, 3), "std": 0.0, "n": n}

    return {
        "test_type": test_type,
        "statistic": 2.13,
        "p_value": 0.041,
        "is_significant": True,
        "effect_size": 0.62,
        "effect_magnitude": "medium",
        "confidence_interval": {"lower": 0.4, "upper": 3.6, "level": 0.95},
        "group_a_summary": _summary(group_a_values),
        "group_b_summary": _summary(group_b_values),
        "interpretation": (
            "The difference is statistically significant (p=0.0410, mock). "
            "Effect size is medium. [synthetic result — public showcase]"
        ),
    }


@trace_tool("detect_anomalies")
async def detect_anomalies(
    values: list[float],
    timestamps: list[str] | None = None,
    method: str = "zscore",
    threshold: float = 2.0,
) -> dict[str, Any]:
    """
    Detect anomalies in a time series or value distribution.

    Use this tool to identify unusual values that deviate significantly
    from the norm. Essential for flagging sudden changes or outliers
    in recruitment metrics.

    Args:
        values: Numeric values to analyze (e.g., weekly hire rates)
        timestamps: Optional timestamps for time-series context (YYYY-MM-DD format)
        method: Detection method:
            - "zscore": Flag values > threshold standard deviations from mean
            - "iqr": Flag values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
            - "mad": Median absolute deviation (robust to outliers)
        threshold: Sensitivity threshold:
            - For zscore: number of standard deviations (default 2.0)
            - For iqr: multiplier for IQR range (default 1.5, but uses threshold)
            - For mad: number of MADs (default 2.0)

    Returns:
        Dictionary with:
        - anomalies: List of detected anomalies with index, value, severity
        - anomaly_count: Number of anomalies detected
        - baseline: Normal range (mean ± threshold*std or IQR bounds)
        - summary: Distribution summary (mean, median, std)
        - interpretation: Human-readable explanation
    """
    # [public-repo stub] Synthetic result — real anomaly detection excluded.
    return {
        "method": method,
        "threshold": threshold,
        "anomalies": [],
        "anomaly_count": 0,
        "baseline": {"lower": 0.0, "upper": 0.0, "method": f"mock {method}"},
        "summary": {"mean": 0.0, "median": 0.0, "std": 0.0, "count": len(values)},
        "interpretation": ("No anomalies detected (mock). [synthetic result — public showcase]"),
    }


@trace_tool("generate_prediction")
async def generate_prediction(
    historical_values: list[float],
    periods_ahead: int = 4,
    timestamps: list[str] | None = None,
    method: str = "linear",
) -> dict[str, Any]:
    """
    Generate probabilistic predictions for a metric.

    Use this tool to forecast future values based on historical trends.
    Returns predictions with confidence bounds to quantify uncertainty.

    Args:
        historical_values: Historical metric values (chronological order)
        periods_ahead: Number of future periods to predict (default 4)
        timestamps: Optional timestamps for the historical data
        method: Prediction method:
            - "linear": Simple linear regression (best for trends)
            - "average": Rolling average (best for stable metrics)
            - "exponential": Exponential smoothing (balanced approach)

    Returns:
        Dictionary with:
        - predictions: List of predicted values with confidence bounds
        - trend: Direction of the trend (increasing/decreasing/stable)
        - trend_strength: Correlation coefficient (R²)
        - methodology: Explanation of the approach used
        - caveats: Important limitations to consider
    """
    # [public-repo stub] Synthetic result — real forecasting excluded.
    base = historical_values[-1] if historical_values else 0.0
    n = len(historical_values)
    predictions = [
        {
            "period": int(n + i),
            "value": round(base, 2),
            "lower_95": round(base * 0.9, 2),
            "upper_95": round(base * 1.1, 2),
            "timestamp": None,
        }
        for i in range(1, periods_ahead + 1)
    ]
    return {
        "method": method,
        "predictions": predictions,
        "trend": "stable",
        "trend_strength": 0.5,
        "summary": {
            "historical_mean": round(base, 2),
            "historical_std": 0.0,
            "predicted_mean": round(base, 2),
            "average_uncertainty": round(base * 0.1, 2),
        },
        "methodology": f"mock {method} projection",
        "caveats": [
            "Synthetic prediction — real forecasting model excluded from public showcase.",
            f"Based on {n} historical data points.",
        ],
    }


@trace_tool("analyze_cohorts")
async def analyze_cohorts(
    data: list[dict],
    cohort_field: str,
    metric_field: str,
    compare_statistically: bool = True,
) -> dict[str, Any]:
    """
    Analyze and compare cohorts on a metric.

    Use this tool to compare groups (e.g., Q1 vs Q2 hires, LinkedIn vs Referral
    candidates) and identify statistically significant differences.

    Args:
        data: List of data rows with cohort and metric fields.
            Example: [{"quarter": "Q1", "hire_rate": 15.2}, {"quarter": "Q2", "hire_rate": 12.8}, ...]
        cohort_field: Field name that defines the cohort grouping
            Example: "quarter", "source", "department"
        metric_field: Field name for the metric to compare
            Example: "hire_rate", "time_to_hire", "offer_acceptance_rate"
        compare_statistically: Whether to run significance tests between cohorts

    Returns:
        Dictionary with:
        - cohort_summaries: Summary stats for each cohort (mean, std, n)
        - best_cohort: Cohort with highest mean
        - worst_cohort: Cohort with lowest mean
        - comparisons: Pairwise statistical comparisons (if enabled)
        - patterns: Identified patterns across cohorts
        - interpretation: Human-readable summary
    """
    # [public-repo stub] Synthetic result — real cohort analysis excluded.
    cohorts = sorted({str(row.get(cohort_field, "unknown")) for row in (data or [])})
    cohort_summaries = {
        c: {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        for c in cohorts
    }
    return {
        "cohort_field": cohort_field,
        "metric_field": metric_field,
        "cohort_count": len(cohorts),
        "cohort_summaries": cohort_summaries,
        "best_cohort": cohorts[0] if cohorts else None,
        "worst_cohort": cohorts[-1] if cohorts else None,
        "comparisons": [],
        "significant_comparisons_count": 0 if compare_statistically else None,
        "patterns": [],
        "interpretation": ("Mock cohort comparison. [synthetic result — public showcase]"),
    }


@trace_tool("analyze_distribution")
async def analyze_distribution(
    values: list[float],
    n_bins: int = 20,
) -> dict[str, Any]:
    """Analyze the shape of a value distribution.

    Detects bimodality, skewness, and spread characteristics. Designed for
    recruitment metrics where bimodal patterns indicate distinct process
    paths (e.g., available vs unavailable interviewers).

    Args:
        values: Numeric values to analyze (e.g., days_in_stage per candidate).
        n_bins: Number of histogram bins for peak detection (default 20).

    Returns:
        Dictionary with:
        - shape: "normal" | "bimodal" | "right_skewed" | "left_skewed" | "uniform" | "insufficient_data"
        - is_bimodal: bool
        - peaks: list of {center, count, pct_of_total}
        - percentiles: {p10, p25, p50, p75, p90}
        - spread: {range, iqr, cv, mean}
        - skewness: float
        - kurtosis: float
        - sample_size: int
        - cohorts: (if bimodal) list of {label, size, pct, mean, median}
        - interpretation: natural-language description
    """
    # [public-repo stub] Synthetic result — real distribution analysis excluded.
    n = len(values) if values else 0
    if n < 10:
        return {
            "shape": "insufficient_data",
            "is_bimodal": False,
            "sample_size": n,
            "interpretation": (
                f"Need at least 10 data points for distribution analysis (got {n})."
            ),
        }
    return {
        "shape": "normal",
        "is_bimodal": False,
        "peaks": [],
        "percentiles": {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0},
        "spread": {"range": 0.0, "iqr": 0.0, "cv": 0.0, "mean": 0.0},
        "skewness": 0.0,
        "kurtosis": 0.0,
        "sample_size": n,
        "cohorts": None,
        "interpretation": (
            "Distribution is approximately normal (mock). [synthetic result — public showcase]"
        ),
    }


# =============================================================================
# FunctionTool wrappers with explicit schemas
# =============================================================================
# The ADK auto-generates JSON Schema from Python type hints, which can produce
# constructs like ``anyOf`` that the Gemini API doesn't support. These wrappers
# provide explicit schemas to avoid compatibility issues. The schema shapes are
# preserved verbatim from the production tool definitions.


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

    if prop_type == "object" and "properties" in prop:
        kwargs["properties"] = {
            key: _convert_property(value) for key, value in prop["properties"].items()
        }

    return types.Schema(**kwargs)


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


# Tool schema definitions
_STATISTICAL_TEST_SCHEMA = {
    "properties": {
        "group_a_values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Numeric values for the first group",
        },
        "group_b_values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Numeric values for the second group",
        },
        "test_type": {
            "type": "string",
            "enum": ["t_test", "mann_whitney", "chi_square"],
            "description": "Type of statistical test: t_test (comparing means), mann_whitney (non-normal data), chi_square (categorical)",
        },
        "alternative": {
            "type": "string",
            "enum": ["two_sided", "greater", "less"],
            "description": "Hypothesis direction: two_sided (different), greater (A > B), less (A < B)",
        },
    },
    "required": ["group_a_values", "group_b_values"],
}

_ANOMALY_DETECTION_SCHEMA = {
    "properties": {
        "values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Numeric values to analyze for anomalies",
        },
        "timestamps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional timestamps for time-series context (YYYY-MM-DD format)",
        },
        "method": {
            "type": "string",
            "enum": ["zscore", "iqr", "mad"],
            "description": "Detection method: zscore (standard deviations), iqr (interquartile range), mad (median absolute deviation)",
        },
        "threshold": {
            "type": "number",
            "description": "Sensitivity threshold - higher values detect fewer anomalies (default 2.0)",
        },
    },
    "required": ["values"],
}

_PREDICTION_SCHEMA = {
    "properties": {
        "historical_values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Historical metric values in chronological order",
        },
        "periods_ahead": {
            "type": "integer",
            "description": "Number of future periods to predict (default 4)",
        },
        "timestamps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional timestamps for the historical data",
        },
        "method": {
            "type": "string",
            "enum": ["linear", "average", "exponential"],
            "description": "Prediction method: linear (trend-based), average (stable metrics), exponential (balanced)",
        },
    },
    "required": ["historical_values"],
}

_COHORT_ANALYSIS_SCHEMA = {
    "properties": {
        "data": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Data rows with cohort and metric fields",
        },
        "cohort_field": {
            "type": "string",
            "description": "Field name that defines the cohort grouping (e.g., 'quarter', 'source')",
        },
        "metric_field": {
            "type": "string",
            "description": "Field name for the metric to compare (e.g., 'hire_rate')",
        },
        "compare_statistically": {
            "type": "boolean",
            "description": "Whether to run significance tests between cohorts (default true)",
        },
    },
    "required": ["data", "cohort_field", "metric_field"],
}

_DISTRIBUTION_ANALYSIS_SCHEMA = {
    "properties": {
        "values": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Individual numeric values to analyze (e.g., days_in_stage per candidate). Minimum 10 values.",
        },
        "n_bins": {
            "type": "integer",
            "description": "Histogram bins for peak detection (default 20)",
        },
    },
    "required": ["values"],
}


# Create wrapped tool instances
statistical_test_tool = _create_function_tool_with_schema(
    func=perform_statistical_test,
    name="perform_statistical_test",
    description="Compare two groups statistically to determine if the difference is significant. Use for questions like 'Is this change real?' or 'Are these groups different?'",
    parameters=_STATISTICAL_TEST_SCHEMA,
)

anomaly_detection_tool = _create_function_tool_with_schema(
    func=detect_anomalies,
    name="detect_anomalies",
    description="Identify unusual values or outliers in a series of metrics. Use for questions like 'Are there any unusual patterns?' or 'Is this value normal?'",
    parameters=_ANOMALY_DETECTION_SCHEMA,
)

prediction_tool = _create_function_tool_with_schema(
    func=generate_prediction,
    name="generate_prediction",
    description="Generate probabilistic forecasts for a metric based on historical data. Returns predictions with confidence bounds. Use for 'What will this metric be?' questions.",
    parameters=_PREDICTION_SCHEMA,
)

cohort_analysis_tool = _create_function_tool_with_schema(
    func=analyze_cohorts,
    name="analyze_cohorts",
    description="Compare groups (cohorts) on a metric to identify differences and patterns. Use for questions like 'How do Q1 hires compare to Q2?' or 'Which source performs best?'",
    parameters=_COHORT_ANALYSIS_SCHEMA,
)

distribution_analysis_tool = _create_function_tool_with_schema(
    func=analyze_distribution,
    name="analyze_distribution",
    description="Characterize the SHAPE of a distribution (bimodal, skewed, normal) to detect patterns averages hide. Use after fetching raw per-record values via query_distribution_values. Answers: 'Why is this stage inconsistent?', 'Why do some candidates wait weeks?', 'Is there a pattern in wait times?'",
    parameters=_DISTRIBUTION_ANALYSIS_SCHEMA,
)


__all__ = [
    "statistical_test_tool",
    "anomaly_detection_tool",
    "prediction_tool",
    "cohort_analysis_tool",
    "distribution_analysis_tool",
]
