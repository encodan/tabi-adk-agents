"""Tool for fetching individual-level values for distribution analysis.

The semantic layer aggregates by design, so it cannot return one row per
candidate. In production this tool wraps a raw-row query through a
tenant-scoped backend endpoint (per-tenant scoping, authenticated,
server-side column whitelist).

NOTE (public showcase): the proprietary semantic layer is excluded; this tool
resolves through ``tools.mock_semantic_layer.SemanticLayerTool``
(``query_distribution_values``), which returns synthetic / empty distribution
samples. The schema, signature and guardrails are preserved verbatim.

The returned values list is intended to be passed straight into
``analyze_distribution`` from ``statistical_tools.py``.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from google.adk.tools import FunctionTool
from google.genai import types

from tools.tool_context import get_tool_context
from tools.tool_tracer import record_tool_call

logger = structlog.get_logger(__name__)

# Whitelist of sources agents may request. Must match the server-side
# whitelist in the backend endpoint — duplicated here to reject obvious
# misuse before a network round-trip.
ALLOWED_SOURCES: frozenset[str] = frozenset(
    {
        "stage_duration",
        # `time_to_fill` here is the org-side opening metric (fct_openings.days_to_fill),
        # not the candidate-side `time_to_hire` MetricFlow metric. Do not unify.
        "time_to_fill",
        "offer_decision_time",
    }
)

# Hard cap on values per call — guards against the distribution tool
# OOM-ing on large tenants. The server also enforces this.
MAX_VALUES: int = 5000

# Default sample size. Distribution shape (bimodality, skew, percentiles)
# converges well below MAX_VALUES and every call scans BigQuery, so the
# default stays conservative. Callers can override up to MAX_VALUES.
DEFAULT_VALUES: int = 1000


async def query_distribution_values(
    source: str,
    stage_name: str | None = None,
    department_name: str | None = None,
    source_name: str | None = None,
    job_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = DEFAULT_VALUES,
) -> dict[str, Any]:
    """Fetch raw individual-record values for distribution analysis.

    Pairs with ``analyze_distribution``: this tool returns the values,
    the analyzer characterizes the shape.

    Args:
        source: Which metric to pull raw values for. One of:
            ``stage_duration``, ``time_to_fill``, ``offer_decision_time``.
        stage_name: Required when ``source='stage_duration'``.
        department_name: Optional department filter.
        source_name: Optional recruitment-source filter.
        job_name: Optional job-title filter.
        start_date: Optional ISO-8601 lower bound on the fact's primary date.
        end_date: Optional ISO-8601 upper bound on the fact's primary date.
        limit: Max values to return. Defaults to ``DEFAULT_VALUES`` (1000);
            capped at ``MAX_VALUES`` (5000).

    Returns:
        ``{"success": True, "values": [float, ...], "sample_size": int,
        "source": str}`` on success, or ``{"success": False,
        "error": {"type": str, "message": str}}`` on failure.
    """
    trace_args = {
        "source": source,
        "stage_name": stage_name,
        "department_name": department_name,
        "source_name": source_name,
        "job_name": job_name,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }

    if source not in ALLOWED_SOURCES:
        resp = {
            "success": False,
            "error": {
                "type": "invalid_source",
                "message": f"source must be one of {sorted(ALLOWED_SOURCES)}",
            },
        }
        record_tool_call(
            "query_distribution_values",
            trace_args,
            result=resp,
            result_summary=f"error: invalid_source {source!r}",
        )
        return resp

    if source == "stage_duration" and not stage_name:
        resp = {
            "success": False,
            "error": {
                "type": "missing_parameter",
                "message": "stage_name is required when source='stage_duration'",
            },
        }
        record_tool_call(
            "query_distribution_values",
            trace_args,
            result=resp,
            result_summary="error: missing_parameter stage_name",
        )
        return resp

    ctx = get_tool_context()
    if ctx is None or ctx.tool_instance is None:
        resp = {
            "success": False,
            "error": {
                "type": "not_configured",
                "message": "Tool context not initialized for this session",
            },
        }
        record_tool_call(
            "query_distribution_values",
            trace_args,
            result=resp,
            result_summary="error: not_configured",
        )
        return resp

    safe_limit = min(max(1, limit), MAX_VALUES)
    filters = {
        "stage_name": stage_name,
        "department_name": department_name,
        "source_name": source_name,
        "job_name": job_name,
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        response = await ctx.tool_instance.query_distribution_values(
            source=source,
            filters=filters,
            limit=safe_limit,
        )
    except httpx.HTTPStatusError as e:
        logger.warning(
            "distribution_query_http_error",
            source=source,
            status=e.response.status_code,
        )
        resp = {
            "success": False,
            "error": {
                "type": "http_error",
                "message": f"API returned {e.response.status_code}",
            },
        }
        record_tool_call(
            "query_distribution_values",
            trace_args,
            result=resp,
            result_summary=f"error: http {e.response.status_code}",
        )
        return resp
    except httpx.HTTPError as e:
        logger.warning("distribution_query_network_error", source=source, error=str(e))
        resp = {
            "success": False,
            "error": {"type": "network_error", "message": str(e)},
        }
        record_tool_call(
            "query_distribution_values",
            trace_args,
            result=resp,
            result_summary=f"error: network {e}",
        )
        return resp

    sample_size = response.get("sample_size") if isinstance(response, dict) else None
    record_tool_call(
        "query_distribution_values",
        trace_args,
        result=response,
        result_summary=f"success, sample_size={sample_size}",
    )
    return response


_DISTRIBUTION_QUERY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "source": types.Schema(
            type=types.Type.STRING,
            enum=sorted(ALLOWED_SOURCES),
            description=(
                "Which metric to pull raw per-record values for. "
                "stage_duration = days_in_stage per candidate (requires stage_name); "
                "time_to_fill = days_to_fill per filled opening; "
                "offer_decision_time = days_to_decision per offer."
            ),
        ),
        "stage_name": types.Schema(
            type=types.Type.STRING,
            description="Stage name (required when source='stage_duration').",
        ),
        "department_name": types.Schema(
            type=types.Type.STRING, description="Optional department filter."
        ),
        "source_name": types.Schema(
            type=types.Type.STRING, description="Optional recruitment-source filter."
        ),
        "job_name": types.Schema(type=types.Type.STRING, description="Optional job-title filter."),
        "start_date": types.Schema(
            type=types.Type.STRING,
            description="Optional ISO-8601 lower bound (YYYY-MM-DD).",
        ),
        "end_date": types.Schema(
            type=types.Type.STRING,
            description="Optional ISO-8601 upper bound (YYYY-MM-DD).",
        ),
        "limit": types.Schema(
            type=types.Type.INTEGER,
            description=(
                f"Max values to return. Default {DEFAULT_VALUES}, capped at {MAX_VALUES}."
            ),
        ),
    },
    required=["source"],
)


class _DistributionQueryFunctionTool(FunctionTool):
    def _get_declaration(self) -> types.FunctionDeclaration | None:
        return types.FunctionDeclaration(
            name="query_distribution_values",
            description=(
                "Fetch raw individual-record values (not aggregated) so the "
                "distribution shape can be analyzed. Pair with "
                "analyze_distribution. Use when investigating inconsistency, "
                "wait-time variability, or 'why do some candidates experience "
                "X while others don't' questions."
            ),
            parameters=_DISTRIBUTION_QUERY_SCHEMA,
        )


distribution_query_tool = _DistributionQueryFunctionTool(func=query_distribution_values)
