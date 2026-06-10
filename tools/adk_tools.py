"""
ADK-compatible tool wrappers for semantic layer access.

These tools wrap the semantic layer for use with Google ADK agents. The tool
functions are designed to be passed directly to ADK Agent definitions.

Includes query batching optimization for parallel execution of independent
queries.

------------------------------------------------------------------------------
PUBLIC SHOWCASE NOTE
------------------------------------------------------------------------------
In the production TABI platform these tools resolve through a proprietary
**MetricFlow semantic layer over per-tenant BigQuery** (governed metric
definitions, multi-tenant isolation, a dbt-managed catalog with per-metric
cache classes, and a shared ``QueryExecutor`` with TTL caching). That data
layer is TABI's commercial moat and is **excluded** from this public repo.

This curated copy preserves the engineering surface that's worth showing —
the hand-authored Gemini ``FunctionDeclaration`` schemas, the wrapper
signatures + docstrings, the chart-by-reference handle contract, and the
``asyncio.gather`` parallel-query structure — but swaps the execution layer
for ``tools.mock_semantic_layer.SemanticLayerTool`` (deterministic synthetic
data). Proprietary cache / dbt-metadata / visualization
helpers are replaced with clearly-commented local stubs. No proprietary
query, caching, or charting logic is included.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from google.adk.tools import FunctionTool
from google.genai import types

from config import get_config
from core.handoff import VALID_SPECIALISTS
from core.ids import generate_ulid
from core.logging_config import get_metric_trace_logger
from core.spans import query_execute_span
from core.spans import set_attrs as set_span_attrs
from core.spans import set_payload as set_span_payload
from tools.distribution_query_tool import (
    distribution_query_tool,  # noqa: F401
)
from tools.handle_registry import (
    get_turn_handle_registry,
)
from tools.mock_semantic_layer import SemanticLayerTool
from tools.planning_tools import (
    compute_goal_attainment as _compute_goal_attainment,
)
from tools.planning_tools import (
    get_planning_context as _get_planning_context,
)
from tools.statistical_tools import distribution_analysis_tool  # noqa: F401
from tools.tool_context import (
    ToolContext,
    _tool_context_var,
)
from tools.tool_context import (
    get_query_batcher as _get_query_batcher,
)
from tools.tool_context import (
    get_tool_context as _get_tool_context,
)
from tools.tool_context import (
    get_tool_instance as _get_tool_instance,
)
from tools.tool_tracer import record_tool_call as _record_to_tracer

if TYPE_CHECKING:
    from models.planning import PlanningContext

logger = structlog.get_logger(__name__)
metric_trace_logger = get_metric_trace_logger()


# ===========================================================================
# [public-repo stub] proprietary query_executor / dbt_metadata excluded
# ---------------------------------------------------------------------------
# The production tree imports a shared QueryExecutor (TTL caching over the
# semantic layer) and a dbt metadata catalog (per-metric cache classes). Both
# are excluded. The trivial stand-ins below keep the call sites below working
# against the mock data layer: TTLs are a flat constant, cache-key/strip/
# metadata helpers are simple, and ``QueryExecutor`` runs the mock queries
# sequentially (the mock backend has no network latency to amortise).
# ===========================================================================

DEFAULT_CACHE_TTL_SECONDS = 300
OPERATIONAL_TTL_SECONDS = 300
REFERENCE_TTL_SECONDS = 300
CACHE_CLASS_OPERATIONAL = "operational"
CACHE_CLASS_REFERENCE = "reference"

_PER_TURN_KEYS: frozenset[str] = frozenset({"query_id", "cache_metadata", "query_result_id"})


def strip_per_turn_keys(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``result`` without per-turn provenance keys."""
    return {k: v for k, v in result.items() if k not in _PER_TURN_KEYS}


def _build_cache_metadata(inserted_at: float, ttl_seconds: float, ttl_class: str) -> dict[str, Any]:
    """Render the per-result cache envelope (ISO-8601 UTC ``cached_at``)."""
    return {
        "cached_at": datetime.fromtimestamp(inserted_at, tz=UTC).isoformat(timespec="seconds"),
        "ttl_class": ttl_class,
        "ttl_seconds": ttl_seconds,
    }


def _shared_make_cache_key(customer_id: str | None, function_args: dict[str, Any]) -> str:
    """Deterministic cache key from customer id + canonicalized query args."""
    import hashlib
    import json

    payload = json.dumps(
        {"customer_id": customer_id, "args": function_args}, sort_keys=True, default=str
    )
    return "ck_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


async def handle_metric_query_tool_call(
    tool_instance: SemanticLayerTool, args: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch a single metric query to the (mock) semantic layer.

    Production routes this through the MetricFlow HTTP client; here it calls
    the mock ``SemanticLayerTool.query_metrics`` and normalises the envelope.
    """
    return await tool_instance.query_metrics(
        metrics=args.get("metrics", []),
        group_by=args.get("group_by"),
        time_granularity=args.get("time_granularity"),
        time_range=args.get("time_range"),
        filters=args.get("filters"),
        order_by=args.get("order_by"),
        limit=args.get("limit", 100),
    )


class _QueryExecutorStats:
    def __init__(self) -> None:
        self.durations_ms: list[float] = []


class QueryExecutor:
    """[public-repo stub] Minimal sequential executor over the mock layer.

    Mirrors the public method surface the deterministic-plan path uses
    (``execute`` + ``stats.durations_ms``). The proprietary version batches,
    caches per-customer, and amortises BigQuery latency; the mock backend
    resolves synchronously so this stand-in just iterates.
    """

    def __init__(
        self,
        tool: SemanticLayerTool,
        customer_id: str | None = None,
        cache: dict[str, Any] | None = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        on_result: Callable[[dict[str, Any]], Any] | None = None,
        ttl_resolver: Callable[[dict[str, Any]], tuple[float, str]] | None = None,
    ) -> None:
        self.tool = tool
        self.customer_id = customer_id
        self.on_result = on_result
        self.stats = _QueryExecutorStats()

    async def execute(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        start = time.perf_counter()
        results: list[dict[str, Any]] = []
        for idx, q in enumerate(queries):
            result = await handle_metric_query_tool_call(self.tool, q)
            result.setdefault("query_index", idx)
            if result.get("success") and self.on_result is not None:
                qid = await self.on_result(result)
                if qid is not None:
                    result["query_id"] = qid
            results.append(result)
        self.stats.durations_ms.append((time.perf_counter() - start) * 1000.0)
        return results


# ===========================================================================
# [public-repo stub] proprietary visualization / chart helpers
# ---------------------------------------------------------------------------
# Production renders Vega-Lite charts through a visualization service, which
# is excluded. The stand-ins below let ``propose_chart`` / ``generate_chart``
# remain defined and return benign mock chart references.
# ===========================================================================

# Narrow chart vocabulary surfaced to the model (matches production schema).
ChartType = str  # the Literal lives in the excluded services/chart_types.py
_CHART_TYPE_VALUES = ["bar", "line", "area", "scatter", "pie", "recruitment-funnel"]
ALLOWED_CHART_TYPES: frozenset[str] = frozenset(_CHART_TYPE_VALUES)


def classify_field(rows: list[dict[str, Any]], field_name: str) -> str:
    """[public-repo stub] Heuristic field classifier for the handle registry.

    The production classifier lives in the excluded visualization service.
    This lightweight version inspects the first non-null value.
    """
    for row in rows:
        val = row.get(field_name)
        if val is None:
            continue
        if isinstance(val, bool):
            return "nominal"
        if isinstance(val, (int, float)):
            return "quantitative"
        sval = str(val)
        if len(sval) >= 7 and sval[:4].isdigit() and sval[4] in "-/":
            return "temporal"
        return "nominal"
    return "nominal"


class ChartResult:
    """[public-repo stub] Mock chart descriptor."""

    def __init__(
        self,
        chart_id: str,
        chart_type: str,
        title: str | None,
        x_field: str,
        y_field: str,
        data_points: int,
    ) -> None:
        self.chart_id = chart_id
        self.chart_type = chart_type
        self.title = title
        self.x_field = x_field
        self.y_field = y_field
        self.data_points = data_points


class _DataAnalysis:
    def __init__(self, dimensions: list[str], measures: list[str]) -> None:
        self.dimensions = dimensions
        self.measures = measures


class _MockVisualizationService:
    """[public-repo stub] Stand-in for the excluded visualization service."""

    def analyze_data(self, rows: list[dict[str, Any]], intent: str) -> _DataAnalysis:
        if not rows:
            return _DataAnalysis([], [])
        dims: list[str] = []
        measures: list[str] = []
        for key, val in rows[0].items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                measures.append(key)
            else:
                dims.append(key)
        return _DataAnalysis(dims, measures)

    def create_chart(
        self,
        data: list[dict[str, Any]],
        intent: str,
        chart_type: str | None = None,
        x_field: str | None = None,
        y_field: str | None = None,
        color_field: str | None = None,
        title: str | None = None,
    ) -> ChartResult:
        analysis = self.analyze_data(data, intent)
        x = x_field or (analysis.dimensions[0] if analysis.dimensions else "category")
        y = y_field or (analysis.measures[0] if analysis.measures else "value")
        return ChartResult(
            chart_id=f"chart_{generate_ulid()}",
            chart_type=chart_type or "bar",
            title=title or intent,
            x_field=x,
            y_field=y,
            data_points=len(data),
        )


def get_visualization_service() -> _MockVisualizationService:
    return _MockVisualizationService()


async def _generate_chart(
    data: list[dict],
    chart_type: str,
    x_field: str,
    y_field: str,
    title: str | None = None,
    color_field: str | None = None,
    size_field: str | None = None,
    horizontal: bool = False,
) -> str:
    """[public-repo stub] Return a benign ```chart markdown block reference.

    Production emits a full Vega-Lite spec from the visualization service.
    """
    chart_id = f"chart_{generate_ulid()}"
    return f'```chart\n{{"chart_id": "{chart_id}", "type": "{chart_type}", "mock": true}}\n```'


# ===========================================================================
# [public-repo stub] hand-authored Gemini schema for the metric-query tool
# ---------------------------------------------------------------------------
# Production keeps this in the excluded ``tools.semantic_layer_tool``. The
# schema itself is product surface (it teaches the model the metric/dimension
# vocabulary), so it's reproduced here verbatim and backed by the mock layer.
# ===========================================================================

SEMANTIC_LAYER_TOOL_DEFINITION: dict[str, Any] = {
    "name": "query_recruitment_metrics",
    "description": (
        "Query recruitment metrics from the TABI semantic layer. Primary tool "
        "for all recruitment analytics — governed metric access with PII "
        "guardrails. Returns aggregated rows for the requested metrics, "
        "optionally sliced by dimension and time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Metric names to query (required). Volume: total_applications, "
                    "hired_count, total_rejections, active_applications. Rate: "
                    "hire_rate, rejection_rate, offer_acceptance_rate. Time: "
                    "time_to_hire, avg_days_in_stage, avg_time_to_offer_decision."
                ),
            },
            "group_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Dimensions to group by. Source: application__source_name, "
                    "application__source_type. Job: application__job_name, "
                    "application__job_status. Stage: stage_transition__stage_name. "
                    "Time: metric_time__day, metric_time__week, metric_time__month, "
                    "metric_time__quarter, metric_time__year."
                ),
            },
            "time_granularity": {
                "type": "string",
                "description": "Time grain for results (day, week, month, quarter, year)",
            },
            "time_range": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
                "description": "Date range filter (YYYY-MM-DD)",
            },
            "filters": {
                "type": "array",
                "description": "Advanced filters as list of {dimension, operator, value} dicts",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": [
                                "=",
                                "!=",
                                ">",
                                "<",
                                ">=",
                                "<=",
                                "in",
                                "not_in",
                                "between",
                                "like",
                            ],
                        },
                        "value": {"type": "string"},
                    },
                },
            },
            "order_by": {
                "type": "string",
                "description": "Column to sort by (prefix with - for descending)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return (1-10000, default 100)",
            },
        },
        "required": ["metrics"],
    },
}


# ===========================================================================
# [public-repo stub] hand-authored Gemini schema for the legacy chart tool
# (production keeps this in the excluded ``tools.chart_tool``).
# ===========================================================================

CHART_TOOL_DEFINITION: dict[str, Any] = {
    "name": "generate_chart",
    "description": (
        "Generate a Vega-Lite chart from query data. Returns a ```chart markdown "
        "block string to include directly in your response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "List of data rows from a previous metric query result",
                "items": {"type": "object"},
            },
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "area", "scatter", "pie"],
                "description": "Chart type",
            },
            "x_field": {"type": "string", "description": "Field name for x-axis"},
            "y_field": {"type": "string", "description": "Field name for y-axis"},
            "title": {"type": "string", "description": "Optional chart title"},
            "color_field": {"type": "string", "description": "Optional field for color encoding"},
            "size_field": {"type": "string", "description": "Optional field for size (scatter)"},
            "horizontal": {"type": "boolean", "description": "Horizontal bar orientation"},
        },
        "required": ["data", "chart_type", "x_field", "y_field"],
    },
}


def get_dbt_metadata() -> Any:  # pragma: no cover - stubbed out
    """[public-repo stub] The dbt metadata catalog is excluded.

    Only ``_classify_ttl`` referenced it (to pick a per-metric cache class).
    In the mock world every metric uses the flat reference TTL.
    """

    class _NoCatalog:
        def metric_cache_class(self, metric: str) -> str:
            return CACHE_CLASS_REFERENCE

    return _NoCatalog()


def _classify_ttl(args: dict[str, Any]) -> tuple[float, str]:
    """Resolve the TTL class for a single metric query.

    In production each metric carries a ``meta.cache_class`` from the dbt
    catalog and the *shortest* TTL wins. The public mock has no catalog, so
    everything resolves to the flat reference TTL.
    """
    metrics = args.get("metrics") or []
    if not metrics:
        return REFERENCE_TTL_SECONDS, CACHE_CLASS_REFERENCE

    md = get_dbt_metadata()
    for m in metrics:
        if md.metric_cache_class(m) == CACHE_CLASS_OPERATIONAL:
            return OPERATIONAL_TTL_SECONDS, CACHE_CLASS_OPERATIONAL
    return REFERENCE_TTL_SECONDS, CACHE_CLASS_REFERENCE


# Tool call trace for evaluation (list of dicts with tool_name, arguments, timestamp, etc.)
_tool_trace_var: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "_tool_trace", default=None
)


# Per-turn keyed query-results map. ContextVar, not a session attribute, so
# ``asyncio.gather``-spawned tool calls all see the same dict (copy_context
# captures this at task creation), and overlapping turns on the same session
# stay isolated. The lock guards concurrent appends from gathered tools.
_turn_query_results: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("_turn_query_results", default=None)
)
_turn_query_results_lock: contextvars.ContextVar[asyncio.Lock | None] = contextvars.ContextVar(
    "_turn_query_results_lock", default=None
)

TurnQueryResultsToken = tuple[
    contextvars.Token[dict[str, dict[str, Any]] | None],
    contextvars.Token[asyncio.Lock | None],
]


def set_turn_query_results(
    keyed_results: dict[str, dict[str, Any]] | None,
) -> TurnQueryResultsToken:
    """Bind ``keyed_results`` as the per-turn query-results dict.

    Returns a ``(dict_token, lock_token)`` pair for ``reset_turn_query_results``
    at turn end. A fresh ``asyncio.Lock`` is installed alongside the dict so
    concurrent ``asyncio.gather`` tool calls serialise their appends.
    """
    dict_token = _turn_query_results.set(keyed_results)
    lock_token = _turn_query_results_lock.set(asyncio.Lock() if keyed_results is not None else None)
    return dict_token, lock_token


def reset_turn_query_results(token: TurnQueryResultsToken) -> None:
    """Restore the previous bindings set by ``set_turn_query_results``."""
    dict_token, lock_token = token
    _turn_query_results.reset(dict_token)
    _turn_query_results_lock.reset(lock_token)


def _summarise_intent(
    metrics: list[str],
    group_by: list[str] | None,
    time_range: dict[str, str] | str | None,
) -> str:
    """Short, deterministic phrase used as the registry intent + chart title hint."""
    parts = [", ".join(metrics)]
    if group_by:
        parts.append(f"by {', '.join(group_by)}")
    if time_range:
        if isinstance(time_range, dict):
            start = time_range.get("start_date")
            end = time_range.get("end_date")
            if start and end:
                parts.append(f"over {start} to {end}")
        else:
            parts.append(f"over {time_range}")
    return " ".join(parts)


def _register_query_handle(
    *,
    rows: list[dict[str, Any]],
    function_args: dict[str, Any],
    cache_key: str | None,
) -> str | None:
    """Register ``rows`` in the per-turn handle registry, returning the id.

    Returns ``None`` when no turn-scoped registry is bound (unit tests
    bypassing ``AgentSession``) or ``rows`` is empty.
    """
    if not rows:
        return None
    registry = get_turn_handle_registry()
    if registry is None:
        return None
    metrics = function_args.get("metrics") or []
    return registry.register(
        rows=rows,
        query_intent=_summarise_intent(
            metrics,
            function_args.get("group_by"),
            function_args.get("time_range"),
        ),
        classify_field=classify_field,
        metric=metrics[0] if len(metrics) == 1 else None,
        cache_key=cache_key,
    )


async def _append_query_result(result: dict[str, Any]) -> str | None:
    """Store a query result under a new ULID-based id and return the id."""
    keyed = _turn_query_results.get()
    if keyed is None:
        return None
    # ULIDs are monotonic-in-time; prefix so logs are skimmable.
    query_id = f"q_{generate_ulid()}"
    lock = _turn_query_results_lock.get()
    if lock is None:
        # No lock bound (test harness) — best-effort write.
        keyed[query_id] = result
    else:
        async with lock:
            keyed[query_id] = result
    return query_id


def set_tool_trace(trace_list: list[dict[str, Any]] | None) -> None:
    """Set or clear the tool trace list for the current session (called by AgentSession)."""
    _tool_trace_var.set(trace_list)


def _record_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    duration_ms: float | None = None,
    result_summary: str | None = None,
    result: Any = None,
) -> None:
    """Record a tool call to both the legacy per-session dict trace and the
    eval-harness ``ToolTrace`` ContextVar when one is bound.
    """
    trace = _tool_trace_var.get()
    if trace is not None:
        trace.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "result_summary": result_summary[:200] if result_summary else None,
            }
        )

    _record_to_tracer(
        tool_name,
        arguments,
        duration_ms=duration_ms,
        result=result,
        result_summary=result_summary,
    )


def configure_tools(
    api_base_url: str,
    customer_id: str,
    api_key: str | None = None,
    internal_api_key: str | None = None,
    timeout: float = 30.0,
    enable_batching: bool | None = None,
    batch_window_ms: float | None = None,
    max_batch_size: int | None = None,
    tool_factory: Callable[..., SemanticLayerTool] | None = None,
    planning_contexts: dict[int, PlanningContext] | None = None,
    current_year_provider: Callable[[], int] | None = None,
) -> ToolContext:
    """
    Configure the semantic layer tool for all ADK agents.

    Creates a session-scoped ToolContext and sets it as the current context
    via contextvars. Must be called before using any ADK agents.

    Args:
        api_base_url: Base URL for the TABI API (e.g., "http://localhost:8000")
        customer_id: Tenant/customer identifier for multi-tenant isolation
        api_key: Optional API key for authentication
        internal_api_key: Optional internal key for /metric-query access
        timeout: Request timeout in seconds
        enable_batching: Enable query batching for parallel execution.
            If None, uses config default.
        batch_window_ms: Time window for collecting queries to batch.
            If None, uses config default (50ms).
        max_batch_size: Maximum queries per batch. If None, uses config default.
        tool_factory: Optional callable used to construct the SemanticLayerTool
            instance. Default ``None`` constructs the (mock) ``SemanticLayerTool``.

    Returns:
        The created ToolContext instance for this session.
    """
    # Get batching config from centralized config if not provided
    config = get_config()
    if enable_batching is None:
        enable_batching = config.query_batching.enabled
    if batch_window_ms is None:
        batch_window_ms = config.query_batching.batch_window_ms
    if max_batch_size is None:
        max_batch_size = config.query_batching.max_batch_size

    tool_config = {
        "api_base_url": api_base_url,
        "customer_id": customer_id,
        "api_key": api_key,
        "timeout": timeout,
        "enable_batching": enable_batching,
        "batch_window_ms": batch_window_ms,
        "max_batch_size": max_batch_size,
    }

    tool_instance = (tool_factory or SemanticLayerTool)(
        api_base_url=api_base_url,
        customer_id=customer_id,
        api_key=api_key,
        internal_api_key=internal_api_key,
        timeout=timeout,
    )

    # In the public showcase the mock data layer resolves synchronously, so the
    # batcher is left unset — the parallel path in
    # ``query_multiple_recruitment_metrics`` (asyncio.gather) is preserved and
    # remains the visible optimisation.
    query_batcher = None
    if enable_batching:
        logger.debug("Query batching requested; mock layer runs queries directly")
    else:
        logger.debug("Query batching disabled")

    ctx_kwargs: dict[str, Any] = {
        "tool_instance": tool_instance,
        "tool_config": tool_config,
        "query_batcher": query_batcher,
    }
    if planning_contexts is not None:
        ctx_kwargs["planning_contexts"] = planning_contexts
    if current_year_provider is not None:
        ctx_kwargs["current_year_provider"] = current_year_provider
    ctx = ToolContext(**ctx_kwargs)
    _tool_context_var.set(ctx)

    logger.info(
        "ADK tools configured for customer %s at %s",
        customer_id,
        api_base_url,
    )

    return ctx


async def _execute_metric_query(args: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single metric query (raw, without caching).
    """
    tool_instance = _get_tool_instance()
    if tool_instance is None:
        return {
            "success": False,
            "error": {
                "type": "configuration_error",
                "message": "Tools not configured.",
            },
        }
    return await handle_metric_query_tool_call(tool_instance, args)


def restore_tools_context(ctx: ToolContext) -> None:
    """Restore a previously created ToolContext into the current async task.

    ContextVars are scoped per-task: when a session is reused across HTTP
    requests the new request runs in a fresh task where the contextvar is
    unset. Calling this at the start of each ``ask`` / ``ask_streaming``
    re-establishes the session's tool context so tools can find it.
    """
    _tool_context_var.set(ctx)


def get_tool_instance() -> SemanticLayerTool | None:
    """Get the configured tool instance (for testing/debugging)."""
    return _get_tool_instance()


def _make_cache_key(function_args: dict[str, Any]) -> str:
    """ContextVar-aware wrapper that reads ``customer_id`` from the active tool context."""
    ctx = _get_tool_context()
    customer_id = ctx.tool_config.get("customer_id") if ctx else None
    return _shared_make_cache_key(customer_id, function_args)


def _get_cached_result(cache_key: str) -> dict[str, Any] | None:
    """Get a cached result if it exists and hasn't expired.

    On hit, stamps a fresh ``cache_metadata`` envelope so the consumer sees
    the same shape as on a cache miss.
    """
    ctx = _get_tool_context()
    if ctx is None:
        return None
    if cache_key in ctx.query_cache:
        result, inserted_at, ttl_seconds, ttl_class = ctx.query_cache[cache_key]
        if time.time() - inserted_at < ttl_seconds:
            return {
                **result,
                "cache_metadata": _build_cache_metadata(inserted_at, ttl_seconds, ttl_class),
            }
        del ctx.query_cache[cache_key]
    return None


def _cache_result(
    cache_key: str,
    result: dict[str, Any],
    *,
    ttl_seconds: float | None = None,
    ttl_class: str = "unclassified",
) -> None:
    """Cache a query result with timestamp + TTL.

    Stores a stripped copy so a later mutation of the caller's ``result`` dict
    (e.g. stamping ``cache_metadata`` or ``query_id``) cannot leak into the
    cache via shared reference.
    """
    ctx = _get_tool_context()
    if ctx is not None:
        ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_CACHE_TTL_SECONDS
        ctx.query_cache[cache_key] = (strip_per_turn_keys(result), time.time(), ttl, ttl_class)


def clear_query_cache() -> None:
    """Clear the query result cache. Call between sessions if needed."""
    ctx = _get_tool_context()
    if ctx is not None:
        count = len(ctx.query_cache)
        ctx.query_cache.clear()
        if count > 0:
            logger.debug("Cleared %d cached query results", count)


async def execute_queries_directly(
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Execute metric queries directly, bypassing the ADK tool framework.

    Used by the deterministic query-plan path to run pre-defined queries
    without an LLM call. ContextVar-aware shim around :class:`QueryExecutor`
    that pulls the tool instance, customer id, and shared query cache from the
    active ``ToolContext`` and binds the per-turn provenance hook.

    Returns:
        Results sorted by ``query_index``, each carrying ``success``, ``data``,
        ``columns``, and (when a per-turn keyed-results store is bound)
        ``query_id``.

    Raises:
        RuntimeError: If tools are not configured (call ``configure_tools()`` first).
    """
    ctx = _get_tool_context()
    if ctx is None or ctx.tool_instance is None:
        raise RuntimeError("Tools not configured. Call configure_tools() first.")

    executor = QueryExecutor(
        tool=ctx.tool_instance,
        customer_id=ctx.tool_config.get("customer_id"),
        cache=ctx.query_cache,
        cache_ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
        on_result=_append_query_result,
        ttl_resolver=_classify_ttl,
    )
    # ``tabi.query.execute``: this path runs *outside* ADK (no ``execute_tool``
    # span), so this manual span is the SOLE source for the viewer's Query Plan
    # / Tools panels on the deterministic path. Structural attrs (metric names,
    # dims, counts) are always set; raw specs/rows are dev/eval-gated.
    with query_execute_span() as _qspan:
        results = await executor.execute(queries)
        if _qspan.is_recording():
            set_span_attrs(
                _qspan,
                {
                    "tabi.query.count": len(queries),
                    "tabi.query.metrics": sorted(
                        {m for q in queries for m in (q.get("metrics") or [])}
                    ),
                    "tabi.query.group_by": sorted(
                        {g for q in queries for g in (q.get("group_by") or [])}
                    ),
                    "tabi.query.row_count": sum(
                        len(r.get("data") or []) for r in results if r.get("success")
                    ),
                    "tabi.query.success_count": sum(1 for r in results if r.get("success")),
                },
            )
            set_span_payload(_qspan, "tabi.query.specs", queries)
            set_span_payload(_qspan, "tabi.query.results", results)

    # Register each successful result in the per-turn handle registry so
    # ``propose_chart`` can resolve rows by ``query_result_id``.
    for query, result in zip(queries, results, strict=True):
        if not result.get("success"):
            continue
        rows = result.get("data") or []
        if not rows:
            continue
        handle_id = _register_query_handle(
            rows=rows,
            function_args=query,
            cache_key=_make_cache_key(query),
        )
        if handle_id is not None:
            result["query_result_id"] = handle_id

    # Mirror the LLM-loop path's trace entry shape so eval traces are uniform
    # across routing paths. ``QueryExecutor.execute`` appends exactly one
    # batch-wall-clock duration per call — take the last entry.
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    duration_ms = executor.stats.durations_ms[-1] if executor.stats.durations_ms else None
    _record_tool_call(
        tool_name="query_multiple_recruitment_metrics",
        arguments={"queries": queries},
        duration_ms=duration_ms,
        result_summary=(
            f"success=True, queries={len(queries)}, successful={successful}, failed={failed}"
        ),
        result={
            "success": True,
            "results": results,
            "total_queries": len(queries),
            "successful_queries": successful,
            "failed_queries": failed,
            "execution_time_ms": duration_ms,
        },
    )

    return results


async def cleanup_tools() -> None:
    """
    Clean up tool resources (close HTTP client, clear cache, flush batcher).

    Should be called when the agent session ends.
    """
    ctx = _tool_context_var.get()
    if ctx is not None:
        await ctx.cleanup()
        _tool_context_var.set(None)


def get_batcher_stats() -> dict[str, Any] | None:
    """
    Get query batcher statistics for monitoring.

    Returns:
        Dictionary with batch statistics, or None if batching is disabled.
    """
    query_batcher = _get_query_batcher()
    if query_batcher is None:
        return None

    stats = query_batcher.stats
    return {
        "total_batches": stats.total_batches,
        "total_queries": stats.total_queries,
        "avg_batch_size": stats.avg_batch_size,
        "batching_rate_pct": stats.batching_rate,
        "avg_execution_time_ms": stats.avg_execution_time_ms,
    }


async def query_recruitment_metrics(
    metrics: list[str],
    group_by: list[str] | None = None,
    time_granularity: str | None = None,
    time_range: dict[str, str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    order_by: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Query recruitment metrics from the TABI semantic layer.

    This is the primary tool for all recruitment analytics agents. It provides
    governed access to metrics with PII guardrails.

    Args:
        metrics: List of metric names to query (required).
            Volume metrics: total_applications, hired_count, total_rejections, active_applications
            Rate metrics: hire_rate, rejection_rate, offer_acceptance_rate
            Time metrics: time_to_hire, avg_days_in_stage, avg_time_to_offer_decision
            Source slicing: group total_applications / hired_count / hire_rate /
                time_to_hire by application__source_name (no separate source_* metrics)
            Trailing metrics: applications_last_30_days, hires_last_90_days

        group_by: Dimensions to group results by.
            Source: application__source_name, application__source_type
            Job: application__job_name, application__job_status
            Stage: stage_transition__stage_name
            Time: metric_time__day, metric_time__week, metric_time__month, metric_time__quarter, metric_time__year

        time_granularity: Time grain for results (day, week, month, quarter, year)

        time_range: Date range filter with start_date and end_date (YYYY-MM-DD format)

        filters: Advanced filters as list of dicts with dimension, operator, values

        order_by: Column to sort by (prefix with - for descending)

        limit: Maximum rows to return (1-10000, default 100)

    Returns:
        On success: {"success": True, "data": [...], "columns": [...], "metadata": {...}}
        On error: {"success": False, "error": {"type": ..., "message": ..., "suggestion": ...}}
    """
    tool_instance = _get_tool_instance()
    query_batcher = _get_query_batcher()

    if tool_instance is None:
        return {
            "success": False,
            "error": {
                "type": "configuration_error",
                "message": "Tools not configured. Call configure_tools() first.",
                "suggestion": "Ensure the agent session is properly initialized.",
            },
        }

    function_args = {
        "metrics": metrics,
        "group_by": group_by,
        "time_granularity": time_granularity,
        "time_range": time_range,
        "filters": filters,
        "order_by": order_by,
        "limit": limit,
    }

    # Remove None values to avoid sending unnecessary params
    function_args = {k: v for k, v in function_args.items() if v is not None}

    # Check cache first to avoid duplicate API calls
    cache_key = _make_cache_key(function_args)
    cached_result = _get_cached_result(cache_key)
    if cached_result is not None:
        logger.info(
            "Cache HIT for query_recruitment_metrics | metrics=%s | group_by=%s",
            function_args.get("metrics"),
            function_args.get("group_by"),
        )
        handle_id = _register_query_handle(
            rows=cached_result.get("data") or [],
            function_args=function_args,
            cache_key=cache_key,
        )
        if handle_id is not None:
            cached_result["query_result_id"] = handle_id
        return cached_result

    logger.info("Invoking query_recruitment_metrics tool (cache miss)")

    trace_id = str(uuid.uuid4())[:8]
    metric_trace_logger.info(
        "METRIC_QUERY_START",
        event_type="METRIC_QUERY_START",
        trace_id=trace_id,
        metrics=function_args.get("metrics"),
        group_by=function_args.get("group_by"),
        time_range=function_args.get("time_range"),
        filters=function_args.get("filters"),
        time_granularity=function_args.get("time_granularity"),
        order_by=function_args.get("order_by"),
        limit=function_args.get("limit", 100),
        batching_enabled=query_batcher is not None,
    )

    # Use query batcher if enabled (for parallel execution); otherwise direct.
    if query_batcher is not None:
        result = await query_batcher.submit(function_args)
    else:
        result = await handle_metric_query_tool_call(tool_instance, function_args)

    # Cache successful results
    if result.get("success"):
        ttl_seconds, ttl_class = _classify_ttl(function_args)
        _cache_result(cache_key, result, ttl_seconds=ttl_seconds, ttl_class=ttl_class)
        result["cache_metadata"] = _build_cache_metadata(time.time(), ttl_seconds, ttl_class)
        qid = await _append_query_result(result)
        if qid is not None:
            result["query_id"] = qid
        handle_id = _register_query_handle(
            rows=result.get("data") or [],
            function_args=function_args,
            cache_key=cache_key,
        )
        if handle_id is not None:
            result["query_result_id"] = handle_id

    # Record tool call for evaluation tracing
    _record_tool_call(
        tool_name="query_recruitment_metrics",
        arguments=function_args,
        result_summary=f"success={result.get('success')}, rows={len(result.get('data', []))}",
        result=result,
    )

    # Log the query result for verification
    if result.get("success"):
        data = result.get("data", [])
        metric_trace_logger.info(
            "METRIC_QUERY_RESULT",
            event_type="METRIC_QUERY_RESULT",
            trace_id=trace_id,
            success=True,
            row_count=len(data),
            columns=result.get("columns", []),
            sample_rows=data[:5],
            full_data=data,
            metadata=result.get("metadata", {}),
        )
    else:
        metric_trace_logger.warning(
            "METRIC_QUERY_RESULT",
            event_type="METRIC_QUERY_RESULT",
            trace_id=trace_id,
            success=False,
            error=result.get("error", {}),
        )

    return result


async def query_multiple_recruitment_metrics(
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute multiple recruitment metric queries in parallel.

    Use this tool when you need multiple independent metrics. All queries
    execute simultaneously, significantly reducing total latency compared to
    calling query_recruitment_metrics multiple times sequentially.

    Args:
        queries: List of query specifications, each with:
            - metrics: List of metric names (required)
            - group_by: Optional list of dimensions to group by
            - time_granularity: Optional time grain (day, week, month, etc.)
            - time_range: Optional dict with start_date/end_date (YYYY-MM-DD)
            - filters: Optional list of filter dicts
            - order_by: Optional column to sort by
            - limit: Optional max rows (default 100)

    Returns:
        On success: {
            "success": True,
            "results": [
                {"query_index": 0, "success": True, "data": [...], ...},
                {"query_index": 1, "success": True, "data": [...], ...},
                ...
            ],
            "total_queries": N,
            "successful_queries": M,
            "failed_queries": K
        }
        On error: {"success": False, "error": {...}}

    Example:
        queries = [
            {"metrics": ["hire_rate"], "time_range": {"start_date": "2025-01-01", "end_date": "2026-01-01"}},
            {"metrics": ["time_to_hire"], "group_by": ["application__job_name"]},
            {"metrics": ["total_applications"], "time_granularity": "month"}
        ]
        result = await query_multiple_recruitment_metrics(queries)
    """
    tool_instance = _get_tool_instance()

    if tool_instance is None:
        return {
            "success": False,
            "error": {
                "type": "configuration_error",
                "message": "Tools not configured. Call configure_tools() first.",
                "suggestion": "Ensure the agent session is properly initialized.",
            },
        }

    if not queries or not isinstance(queries, list):
        return {
            "success": False,
            "error": {
                "type": "validation_error",
                "message": "queries must be a non-empty list of query specifications",
            },
        }

    logger.info(
        "Executing %d queries in parallel via query_multiple_recruitment_metrics",
        len(queries),
    )

    # Prepare all queries with proper structure
    prepared_queries = []
    for idx, query in enumerate(queries):
        if not isinstance(query, dict) or "metrics" not in query:
            logger.warning(
                "Skipping invalid query at index %d: missing 'metrics' field",
                idx,
            )
            continue

        function_args = {
            "metrics": query["metrics"],
            "group_by": query.get("group_by"),
            "time_granularity": query.get("time_granularity"),
            "time_range": query.get("time_range"),
            "filters": query.get("filters"),
            "order_by": query.get("order_by"),
            "limit": query.get("limit", 100),
        }

        # Remove None values
        function_args = {k: v for k, v in function_args.items() if v is not None}
        prepared_queries.append((idx, function_args))

    if not prepared_queries:
        return {
            "success": False,
            "error": {
                "type": "validation_error",
                "message": "No valid queries found in the queries list",
            },
        }

    trace_id = str(uuid.uuid4())[:8]
    metric_trace_logger.info(
        "MULTI_QUERY_START",
        event_type="MULTI_QUERY_START",
        trace_id=trace_id,
        query_count=len(prepared_queries),
    )

    start_time = time.perf_counter()

    # Execute queries in parallel (the headline optimization — independent
    # queries fan out via asyncio.gather rather than running sequentially).
    async def execute_single_query(idx_and_args):
        idx, args = idx_and_args

        # Check cache for each query
        cache_key = _make_cache_key(args)
        cached_result = _get_cached_result(cache_key)
        if cached_result is not None:
            logger.debug("Cache HIT for query %d", idx)
            handle_id = _register_query_handle(
                rows=cached_result.get("data") or [],
                function_args=args,
                cache_key=cache_key,
            )
            if handle_id is not None:
                cached_result["query_result_id"] = handle_id
            return idx, cached_result

        result = await handle_metric_query_tool_call(tool_instance, args)

        if result.get("success"):
            ttl_seconds, ttl_class = _classify_ttl(args)
            _cache_result(cache_key, result, ttl_seconds=ttl_seconds, ttl_class=ttl_class)
            result["cache_metadata"] = _build_cache_metadata(time.time(), ttl_seconds, ttl_class)
            handle_id = _register_query_handle(
                rows=result.get("data") or [],
                function_args=args,
                cache_key=cache_key,
            )
            if handle_id is not None:
                result["query_result_id"] = handle_id

        return idx, result

    # Create tasks explicitly to ensure they're scheduled concurrently
    tasks = [execute_single_query(q) for q in prepared_queries]
    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    execution_time_ms = (time.perf_counter() - start_time) * 1000

    # Process results
    query_results = []
    successful_count = 0
    failed_count = 0

    for result_item in results:
        if isinstance(result_item, Exception):
            failed_count += 1
            query_results.append(
                {
                    "query_index": -1,
                    "success": False,
                    "error": {
                        "type": "execution_error",
                        "message": str(result_item),
                    },
                }
            )
            logger.error("Query failed with exception: %s", result_item)
        else:
            idx, result = result_item
            query_results.append(
                {
                    "query_index": idx,
                    **result,
                }
            )
            if result.get("success"):
                successful_count += 1
            else:
                failed_count += 1

    # Sort by query_index to maintain order
    query_results.sort(key=lambda x: x.get("query_index", -1))

    for result in query_results:
        if result.get("success"):
            qid = await _append_query_result(result)
            if qid is not None:
                result["query_id"] = qid

    logger.info(
        "Multi-query completed: %d queries in %.1fms (%.1fms avg), %d successful, %d failed",
        len(prepared_queries),
        execution_time_ms,
        execution_time_ms / len(prepared_queries) if prepared_queries else 0,
        successful_count,
        failed_count,
    )

    metric_trace_logger.info(
        "MULTI_QUERY_RESULT",
        event_type="MULTI_QUERY_RESULT",
        trace_id=trace_id,
        total_queries=len(prepared_queries),
        successful_queries=successful_count,
        failed_queries=failed_count,
        execution_time_ms=execution_time_ms,
    )

    response = {
        "success": True,
        "results": query_results,
        "total_queries": len(prepared_queries),
        "successful_queries": successful_count,
        "failed_queries": failed_count,
        "execution_time_ms": execution_time_ms,
    }

    # Record tool call for evaluation tracing
    _record_tool_call(
        tool_name="query_multiple_recruitment_metrics",
        arguments={"queries": [args for _, args in prepared_queries]},
        duration_ms=execution_time_ms,
        result_summary=f"success=True, queries={len(prepared_queries)}, "
        f"successful={successful_count}, failed={failed_count}",
        result=response,
    )

    return response


async def get_available_metrics() -> dict[str, Any]:
    """
    Get the list of available metrics and dimensions.

    Useful for discovery and debugging. Returns the semantic layer's metric
    catalog.

    Returns:
        On success: {"success": True, "metrics": [...], "dimensions": [...]}
        On error: {"success": False, "error": {...}}
    """
    tool_instance = _get_tool_instance()
    if tool_instance is None:
        return {
            "success": False,
            "error": {
                "type": "configuration_error",
                "message": "Tools not configured. Call configure_tools() first.",
                "suggestion": "Ensure the agent session is properly initialized.",
            },
        }

    try:
        result = await tool_instance.get_available_metrics()
        response = {"success": True, **result}
        _record_tool_call(
            tool_name="get_available_metrics",
            arguments={},
            result_summary=f"metrics={len(result.get('metrics', []))}",
            result=response,
        )
        return response
    except Exception as e:
        logger.exception("Error getting available metrics")
        error_response = {
            "success": False,
            "error": {
                "type": "internal_error",
                "message": str(e),
                "suggestion": "Check API connectivity and try again.",
            },
        }
        _record_tool_call(
            tool_name="get_available_metrics",
            arguments={},
            result_summary=f"error: {e}",
            result=error_response,
        )
        return error_response


async def request_specialist_handoff(
    target_agent: str,
    reason: str,
    context_summary: str,
) -> dict[str, Any]:
    """
    Request handoff to another specialist agent.

    Use this tool when your analysis reveals a need for expertise outside your
    specialty. The orchestrator will automatically invoke the target specialist
    with your findings as context.

    This enables collaborative multi-agent responses where specialists build on
    each other's insights to provide comprehensive answers.

    Args:
        target_agent: Name of the specialist to hand off to.
            Valid options:
            - pipeline_analyst: For bottleneck analysis, stage duration, pipeline health
            - general_analyst: For basic metrics, hiring volumes, trends
            - sourcing_strategist: For source/channel performance, ROI analysis
            - offer_advisor: For offer acceptance, decline analysis, closing strategies
            - interviewing_coach: For interview process efficiency, scheduling
            - capacity_planner: For hiring velocity, forecasting, pipeline coverage

        reason: Brief explanation of why this handoff is needed.
            Example: "Bottleneck analysis reveals significant delays at offer stage,
            requiring Offer Advisor expertise to diagnose acceptance issues."

        context_summary: Key findings from your analysis to share with the target agent.
            Include relevant data points, observations, and any hypotheses. This
            helps the target agent build on your work rather than starting fresh.

    Returns:
        Acknowledgment that the handoff was requested. The orchestrator will
        evaluate and invoke the target specialist.

    Example:
        If you're the Pipeline Analyst and discover offer stage issues:

        request_specialist_handoff(
            target_agent="offer_advisor",
            reason="Pipeline shows 40% drop-off at offer stage, need offer acceptance analysis",
            context_summary="Analysis shows: avg 8 days in offer stage (benchmark: 3 days), "
                          "Engineering roles have 35% offer decline rate. Recommend investigating "
                          "compensation competitiveness and offer timing."
        )
    """
    # Validate target agent
    if target_agent not in VALID_SPECIALISTS:
        return {
            "status": "error",
            "error": f"Invalid target_agent: {target_agent}. "
            f"Valid options: {', '.join(sorted(VALID_SPECIALISTS))}",
        }

    # Log the handoff request
    logger.info(
        "Handoff requested to %s. Reason: %s",
        target_agent,
        reason[:100],
    )

    # Return structured response that orchestrator will recognize
    response = {
        "status": "handoff_requested",
        "target": target_agent,
        "reason": reason,
        "context": context_summary,
        "note": "The orchestrator will invoke the target specialist with your context.",
    }
    _record_tool_call(
        tool_name="request_specialist_handoff",
        arguments={"target_agent": target_agent, "reason": reason},
        result_summary=f"handoff_to={target_agent}",
        result=response,
    )
    return response


async def generate_chart(
    data: list[dict],
    chart_type: str,
    x_field: str,
    y_field: str,
    title: str | None = None,
    color_field: str | None = None,
    size_field: str | None = None,
    horizontal: bool = False,
) -> str:
    """
    Generate a Vega-Lite chart visualization from query data.

    Use this tool after querying recruitment metrics to create visual
    representations. The chart will be rendered inline in the chat response.

    IMPORTANT: This tool returns a ```chart markdown block as a string. You MUST
    include this returned string DIRECTLY in your response text - the frontend
    will parse and render it as an interactive chart.

    Args:
        data: List of data rows from a previous metric query result.
            This should be the "data" array from query_recruitment_metrics.
            Example: [{"source": "LinkedIn", "hire_rate": 12.5}, ...]

        chart_type: Type of chart to generate.
            - "bar": For comparing categories or distributions
            - "line": For time series and trends
            - "area": For cumulative time series
            - "scatter": For relationships between metrics
            - "pie": For proportions of a whole

        x_field: Field name for x-axis (or category for pie charts).
            For time series: use the time field (e.g., "metric_time__month")
            For comparisons: use the category (e.g., "application__source_name")

        y_field: Field name for y-axis (or value for pie charts).
            The metric being measured (e.g., "hire_rate", "time_to_hire")

        title: Optional chart title. Recommended for clarity.

        color_field: Optional field for color encoding.
            Adds a second dimension (e.g., color by department).

        size_field: Optional field for size in scatter plots.

        horizontal: For bar charts - use horizontal orientation.
            Useful when category labels are long.

    Returns:
        A ```chart markdown block string ready to include in your response.
        On error, returns an error message string.
    """
    logger.info(
        "Generating chart: type=%s, x=%s, y=%s, title=%s, data_rows=%d",
        chart_type,
        x_field,
        y_field,
        title,
        len(data) if data else 0,
    )
    if not data:
        logger.warning("Chart data is EMPTY - no bars will render!")

    result = await _generate_chart(
        data=data,
        chart_type=chart_type,
        x_field=x_field,
        y_field=y_field,
        title=title,
        color_field=color_field,
        size_field=size_field,
        horizontal=horizontal,
    )
    _record_tool_call(
        tool_name="generate_chart",
        arguments={
            "chart_type": chart_type,
            "x_field": x_field,
            "y_field": y_field,
            "title": title,
            "color_field": color_field,
            "size_field": size_field,
            "horizontal": horizontal,
            "data_rows": len(data) if data else 0,
        },
        result_summary=f"chart_type={chart_type}, rows={len(data) if data else 0}",
    )
    return result


async def propose_chart(
    query_result_id: str,
    intent: str,
    chart_type: ChartType | None = None,
    x_field: str | None = None,
    y_field: str | None = None,
    color_field: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create a chart from a previously-retrieved query result.

    The chart-by-reference contract — the model passes the handle returned by
    ``query_recruitment_metrics`` (the ``query_result_id`` field), the server
    resolves the rows in-process. The model never re-serializes data.

    Args:
        query_result_id: The handle returned by ``query_recruitment_metrics``
            (field ``query_result_id``). DO NOT echo back row data.
        intent: One-line description of what the chart shows (used for title
            + telemetry).
        chart_type: Override the server's recommendation. One of: bar, line,
            area, scatter, pie, recruitment-funnel.
        x_field / y_field / color_field / title: Override server defaults.

    Returns:
        On success: ``{success: True, chart_id, chart_reference}``.
        On scalar data: ``{success: False, reason: "scalar", skip_visualization: True}``.
        On unknown handle: ``{success: False, reason: "unknown_handle", available_ids: [...]}``.
    """
    trace_args = {
        "query_result_id": query_result_id,
        "intent": intent,
        "chart_type": chart_type,
        "x_field": x_field,
        "y_field": y_field,
        "color_field": color_field,
        "title": title,
    }

    def _trace(summary: str, response: dict[str, Any]) -> dict[str, Any]:
        """Stamp the call into the tool trace and return the response."""
        _record_tool_call(
            tool_name="propose_chart",
            arguments=trace_args,
            result_summary=summary,
            result=response,
        )
        return response

    registry = get_turn_handle_registry()
    if registry is None:
        return _trace(
            "error: no_registry",
            {
                "success": False,
                "reason": "no_registry",
                "skip_visualization": True,
                "message": (
                    "Visualization registry not bound — this tool requires an active turn scope."
                ),
            },
        )

    handle = registry.resolve(query_result_id)
    if handle is None:
        return _trace(
            f"error: unknown_handle {query_result_id!r}",
            {
                "success": False,
                "reason": "unknown_handle",
                "available_ids": registry.list_ids(),
                "message": (
                    "No query result registered for that handle. Call "
                    "query_recruitment_metrics first and pass the returned "
                    "`query_result_id` here."
                ),
            },
        )

    if chart_type is not None and chart_type not in ALLOWED_CHART_TYPES:
        return _trace(
            f"error: invalid chart_type {chart_type!r}",
            {
                "success": False,
                "reason": "invalid_chart_type",
                "error": (
                    f"unknown chart_type: {chart_type!r}. Allowed: {sorted(ALLOWED_CHART_TYPES)}"
                ),
            },
        )

    viz = get_visualization_service()
    # Scalar short-circuit — structured refusal so the model knows to write
    # text only rather than burning a chart slot on a single value.
    analysis = viz.analyze_data(handle.rows, intent=intent)
    if not analysis.dimensions and not analysis.measures:
        return _trace(
            "skip: scalar",
            {
                "success": False,
                "reason": "scalar",
                "skip_visualization": True,
                "message": "Data is scalar; respond in text only with the value.",
            },
        )

    try:
        result: ChartResult = viz.create_chart(
            data=handle.rows,
            intent=intent or handle.query_intent,
            chart_type=chart_type,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
            title=title,
        )
    except ValueError as e:
        logger.warning("propose_chart validation error: %s", e)
        return _trace(
            f"error: {e}",
            {"success": False, "reason": "validation_error", "error": str(e)},
        )
    except Exception as e:
        logger.exception("Error creating chart from handle")
        return _trace(
            f"error: {e}",
            {
                "success": False,
                "reason": "internal_error",
                "error": f"Failed to create chart: {e}",
            },
        )

    return _trace(
        f"chart_id={result.chart_id}, type={result.chart_type}",
        {
            "success": True,
            "chart_id": result.chart_id,
            "chart_type": result.chart_type,
            "title": result.title,
            "x_field": result.x_field,
            "y_field": result.y_field,
            "data_points": result.data_points,
            "chart_reference": f"[chart:{result.chart_id}]",
        },
    )


# Tool definition for propose_chart — the chart-by-reference replacement for
# create_visualization. Payload is ~7 short scalars regardless of row count,
# eliminating the ``[{}]`` / truncated-data class of failures.
PROPOSE_CHART_TOOL_DEFINITION = {
    "name": "propose_chart",
    "description": (
        "Create a chart from a previously-retrieved query result. "
        "Pass the `query_result_id` returned by query_recruitment_metrics "
        "(or query_multiple_recruitment_metrics) — the server resolves the "
        "rows in-process. DO NOT pass raw row data. "
        "Returns a chart_id; include `chart_reference` in your response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_result_id": {
                "type": "string",
                "description": (
                    "Handle returned in the `query_result_id` field of a "
                    "previous metric-query tool call. Required."
                ),
            },
            "intent": {
                "type": "string",
                "description": (
                    "One-line description of what the chart shows "
                    "(used for title + telemetry). Required."
                ),
            },
            "chart_type": {
                "type": "string",
                # Sourced from the shared chart vocabulary so a future change
                # flows to the tool schema automatically.
                "enum": list(_CHART_TYPE_VALUES),
                "description": (
                    "Chart type (optional - auto-detected if not specified). "
                    "Use 'recruitment-funnel' for hiring-funnel / stage drop-off visuals."
                ),
            },
            "x_field": {
                "type": "string",
                "description": "X-axis field (optional - auto-detected)",
            },
            "y_field": {
                "type": "string",
                "description": "Y-axis field (optional - auto-detected)",
            },
            "color_field": {
                "type": "string",
                "description": "Field for color encoding (optional)",
            },
            "title": {
                "type": "string",
                "description": "Chart title (recommended)",
            },
        },
        "required": ["query_result_id", "intent"],
    },
}


class ChartByReferenceBaseTool(FunctionTool):
    """Shared JSON-Schema → Gemini Schema converter for chart tools.

    Carries the ``_convert_property`` helper that :class:`ProposeChartFunctionTool`
    uses to translate a JSON-Schema tool definition into the
    ``google.genai.types.Schema`` Gemini wants. Subclasses supply
    ``_get_declaration``.
    """

    def _convert_property(self, prop: dict) -> types.Schema:
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
            kwargs["items"] = self._convert_property(prop["items"])

        if prop_type == "object" and "properties" in prop:
            kwargs["properties"] = {
                key: self._convert_property(value) for key, value in prop["properties"].items()
            }
            if "required" in prop:
                kwargs["required"] = prop["required"]

        return types.Schema(**kwargs)


class SemanticLayerFunctionTool(FunctionTool):
    """
    Custom FunctionTool that uses explicit schema to avoid Gemini API compatibility issues.

    The default FunctionTool auto-generates schema from Python type hints, which
    can produce constructs like `anyOf` and `additionalProperties` that the
    Gemini API doesn't support. This class overrides the declaration generation
    to use our explicit SEMANTIC_LAYER_TOOL_DEFINITION schema.
    """

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        """Override to use explicit schema instead of auto-generated one."""
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                key: self._convert_property(value)
                for key, value in SEMANTIC_LAYER_TOOL_DEFINITION["parameters"]["properties"].items()
            },
            required=SEMANTIC_LAYER_TOOL_DEFINITION["parameters"].get("required", []),
        )

        return types.FunctionDeclaration(
            name=SEMANTIC_LAYER_TOOL_DEFINITION["name"],
            description=SEMANTIC_LAYER_TOOL_DEFINITION["description"],
            parameters=schema,
        )

    def _convert_property(self, prop: dict) -> types.Schema:
        """Convert a JSON Schema property to Gemini Schema type."""
        prop_type = prop.get("type", "string")

        # Map JSON Schema types to Gemini types
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

        # Handle enum
        if "enum" in prop:
            kwargs["enum"] = prop["enum"]

        # Handle array items
        if prop_type == "array" and "items" in prop:
            kwargs["items"] = self._convert_property(prop["items"])

        # Handle object properties
        if prop_type == "object" and "properties" in prop:
            kwargs["properties"] = {
                key: self._convert_property(value) for key, value in prop["properties"].items()
            }
            if "required" in prop:
                kwargs["required"] = prop["required"]

        return types.Schema(**kwargs)


class ChartFunctionTool(FunctionTool):
    """
    Custom FunctionTool for chart generation with explicit schema.

    The default FunctionTool auto-generates schema from Python type hints, which
    produces `additionalProperties` for dict[str, Any] parameters that the
    Gemini API doesn't support. This class uses our explicit
    CHART_TOOL_DEFINITION schema.
    """

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        """Override to use explicit chart tool schema."""
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                key: self._convert_property(value)
                for key, value in CHART_TOOL_DEFINITION["parameters"]["properties"].items()
            },
            required=CHART_TOOL_DEFINITION["parameters"].get("required", []),
        )

        return types.FunctionDeclaration(
            name=CHART_TOOL_DEFINITION["name"],
            description=CHART_TOOL_DEFINITION["description"],
            parameters=schema,
        )

    def _convert_property(self, prop: dict) -> types.Schema:
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
            kwargs["items"] = self._convert_property(prop["items"])

        if prop_type == "object" and "properties" in prop:
            kwargs["properties"] = {
                key: self._convert_property(value) for key, value in prop["properties"].items()
            }
            if "required" in prop:
                kwargs["required"] = prop["required"]

        return types.Schema(**kwargs)


# Multi-query tool definition schema
MULTI_QUERY_TOOL_DEFINITION = {
    "name": "query_multiple_recruitment_metrics",
    "description": (
        "Execute multiple recruitment metric queries in parallel. "
        "Use this when you need multiple independent metrics - all queries execute "
        "simultaneously, significantly reducing total latency compared to calling "
        "query_recruitment_metrics multiple times sequentially."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": "List of query specifications to execute in parallel",
                "items": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of metric names (required)",
                        },
                        "group_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional dimensions to group by",
                        },
                        "time_granularity": {
                            "type": "string",
                            "description": "Optional time grain (day, week, month, quarter, year)",
                        },
                        "time_range": {
                            "type": "object",
                            "properties": {
                                "start_date": {
                                    "type": "string",
                                    "description": "Start date (YYYY-MM-DD)",
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "End date (YYYY-MM-DD)",
                                },
                            },
                            "description": "Optional date range filter",
                        },
                        "filters": {
                            "type": "array",
                            "description": "Optional advanced filters",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dimension": {"type": "string"},
                                    "operator": {
                                        "type": "string",
                                        "enum": [
                                            "=",
                                            "!=",
                                            ">",
                                            "<",
                                            ">=",
                                            "<=",
                                            "in",
                                            "not_in",
                                            "between",
                                            "like",
                                        ],
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "Filter value as string. For 'between': JSON array like '[min, max]'. For 'like': pattern with % wildcards. For 'in'/'not_in': JSON array like '[\"val1\", \"val2\"]'. For comparison operators: simple string value.",
                                    },
                                },
                            },
                        },
                        "order_by": {
                            "type": "string",
                            "description": "Optional column to sort by (prefix with - for descending)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional max rows (default 100)",
                        },
                    },
                    # Note: "required" field removed - Gemini Schema doesn't support it on nested objects
                    # Validation is handled in query_multiple_recruitment_metrics() function
                },
            }
        },
        "required": ["queries"],
    },
}


class MultiQueryFunctionTool(FunctionTool):
    """
    Custom FunctionTool for multi-query execution with explicit schema.
    """

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        """Override to use explicit multi-query tool schema."""
        try:
            schema = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    key: self._convert_property(value)
                    for key, value in MULTI_QUERY_TOOL_DEFINITION["parameters"][
                        "properties"
                    ].items()
                },
                required=MULTI_QUERY_TOOL_DEFINITION["parameters"].get("required", []),
            )

            return types.FunctionDeclaration(
                name=MULTI_QUERY_TOOL_DEFINITION["name"],
                description=MULTI_QUERY_TOOL_DEFINITION["description"],
                parameters=schema,
            )
        except Exception as e:
            logger.error(
                "Error creating multi-query tool declaration: %s",
                e,
                exc_info=True,
            )
            raise

    def _convert_property(self, prop: dict, is_nested: bool = False) -> types.Schema:
        """Convert a JSON Schema property to Gemini Schema type.

        Args:
            prop: JSON Schema property definition
            is_nested: True if this is a nested object (e.g., array items, nested objects)
                      Nested objects cannot have 'required' fields in Gemini Schema
        """
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
            # Array items are nested - create a copy without 'required' field
            # Gemini Schema API doesn't support 'required' on nested object schemas
            items_prop = {k: v for k, v in prop["items"].items() if k != "required"}
            kwargs["items"] = self._convert_property(items_prop, is_nested=True)

        if prop_type == "object" and "properties" in prop:
            kwargs["properties"] = {
                key: self._convert_property(value, is_nested=True)
                for key, value in prop["properties"].items()
            }
            # Only set 'required' for top-level objects, not nested ones
            # Gemini Schema API doesn't support 'required' on nested object schemas
            if "required" in prop and not is_nested:
                kwargs["required"] = prop["required"]

        return types.Schema(**kwargs)


# Create the wrapped tool instances for use by agents
query_metrics_tool = SemanticLayerFunctionTool(func=query_recruitment_metrics)
multi_query_tool = MultiQueryFunctionTool(func=query_multiple_recruitment_metrics)
chart_tool = ChartFunctionTool(func=generate_chart)  # Legacy generate_chart wrapper


class ProposeChartFunctionTool(ChartByReferenceBaseTool):
    """``propose_chart`` declaration with the chart-by-reference schema."""

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                key: self._convert_property(value)
                for key, value in PROPOSE_CHART_TOOL_DEFINITION["parameters"]["properties"].items()
            },
            required=PROPOSE_CHART_TOOL_DEFINITION["parameters"].get("required", []),
        )
        return types.FunctionDeclaration(
            name=PROPOSE_CHART_TOOL_DEFINITION["name"],
            description=PROPOSE_CHART_TOOL_DEFINITION["description"],
            parameters=schema,
        )


propose_chart_tool = ProposeChartFunctionTool(func=propose_chart)
request_specialist_handoff_tool = FunctionTool(func=request_specialist_handoff)


def select_visualization_tool() -> FunctionTool:
    """Return the canonical chart tool for specialist agents.

    Historically gated by ``chart_by_reference_enabled`` (P1 kill-switch); now
    unconditional since ``create_visualization`` was deleted in P2. Kept as a
    function so agent factories continue to call a single hook rather than
    importing :data:`propose_chart_tool` directly.
    """
    return propose_chart_tool


# Planning-context tools.
# Wrap the bare functions defined in planning_tools.py so they're callable from
# ADK agents. Schemas are inferred from the type hints — both signatures are
# simple enough that the default FunctionTool introspection is correct.
get_planning_context_tool = FunctionTool(func=_get_planning_context)
compute_goal_attainment_tool = FunctionTool(func=_compute_goal_attainment)
