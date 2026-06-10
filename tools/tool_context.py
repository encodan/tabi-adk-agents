"""Session-scoped tool context shared by ADK tools.

Lives in its own module so individual tools (e.g. ``distribution_query_tool``)
can read the active ``ToolContext`` without importing ``adk_tools`` and
forming a circular dependency.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from tools.mock_semantic_layer import SemanticLayerTool

if TYPE_CHECKING:
    from models.planning import PlanningContext

logger = structlog.get_logger(__name__)


# [public-repo stub] The production QueryBatcher (parallel-execution window
# over the proprietary semantic layer) is excluded. The mock data backend
# resolves synchronously, so the public showcase never constructs a batcher —
# this minimal stand-in only keeps the ToolContext type annotation and the
# ``cleanup`` flush/stat path importable.
class QueryBatcher:  # pragma: no cover - mock stand-in
    """No-op stand-in for the excluded production query batcher."""

    async def flush(self) -> None:
        return None

    @property
    def stats(self) -> Any:
        class _Stats:
            total_batches = 0
            total_queries = 0
            avg_batch_size = 0.0
            batching_rate = 0.0
            avg_execution_time_ms = 0.0

        return _Stats()


@dataclass(frozen=True)
class GoalAttainmentInvocation:
    """One ``compute_goal_attainment`` call recorded for replay/verification.

    Lives in ``tool_context`` (not ``planning_tools``) so the verifier in
    ``response_validator`` can import it without pulling in the ContextVar
    machinery, keeping ``planning_tools → tool_context`` one-directional.
    """

    source_query_id: str
    args: dict[str, Any]
    result: dict[str, Any]


def _default_current_year_provider() -> int:
    return datetime.now(UTC).year


@dataclass
class ToolContext:
    """Session-scoped tool state. One instance per AgentSession.

    Replaces module-level globals to prevent cross-tenant data leaks
    when multiple sessions run concurrently.
    """

    tool_instance: SemanticLayerTool
    tool_config: dict[str, Any]
    query_batcher: QueryBatcher | None = None
    # Cache value is ``(result, inserted_at, ttl_seconds, ttl_class)`` —
    # per-entry TTL lets metric-class differentiation (operational vs
    # reference) coexist in one store; the class is preserved so the
    # consumer can label freshness without a reverse lookup.
    query_cache: dict[str, tuple[dict[str, Any], float, float, str]] = field(default_factory=dict)
    # Pre-loaded per-year planning contexts, one Postgres read at session
    # create covers every configured year. Tools resolve multi-year queries
    # via sync dict lookups — no DB roundtrip, no analytics→API callback.
    planning_contexts: dict[int, PlanningContext] = field(default_factory=dict)
    # Re-evaluated on each tool call (not memoised) so a long-lived session
    # crossing midnight on Dec 31 picks up the new year on the next call.
    # Tests inject a deterministic year via ``lambda: 2026``.
    current_year_provider: Callable[[], int] = field(default=_default_current_year_provider)

    async def cleanup(self) -> None:
        """Clean up tool resources (close HTTP client, clear cache, flush batcher)."""
        if self.query_batcher is not None:
            try:
                await self.query_batcher.flush()
            except Exception as e:
                logger.warning("query_batcher_flush_failed", error=str(e))

            stats = self.query_batcher.stats
            if stats.total_batches > 0:
                logger.info(
                    "query_batcher_stats",
                    batches=stats.total_batches,
                    queries=stats.total_queries,
                    avg_batch_size=round(stats.avg_batch_size, 2),
                    multi_query_rate_pct=round(stats.batching_rate, 1),
                    avg_execution_time_ms=round(stats.avg_execution_time_ms, 1),
                )

        if self.tool_instance is not None:
            await self.tool_instance.close()

        count = len(self.query_cache)
        self.query_cache.clear()
        if count > 0:
            logger.debug("query_cache_cleared", cached_results=count)

        logger.debug("adk_tools_cleaned_up")


# Session-scoped tool context via contextvars (async-safe, per-task isolation)
_tool_context_var: contextvars.ContextVar[ToolContext | None] = contextvars.ContextVar(
    "_tool_context", default=None
)


def get_tool_context() -> ToolContext | None:
    """Get the current session's tool context."""
    return _tool_context_var.get()


def get_tool_instance() -> SemanticLayerTool | None:
    """Get the current session's tool instance."""
    ctx = _tool_context_var.get()
    return ctx.tool_instance if ctx else None


def get_query_batcher() -> QueryBatcher | None:
    """Get the current session's query batcher."""
    ctx = _tool_context_var.get()
    return ctx.query_batcher if ctx else None


def get_active_planning_context() -> PlanningContext | None:
    """Resolve the session's :class:`PlanningContext` for the current year.

    Returns ``None`` when no session is bound (test paths) or the tenant
    has no target for the current year — both cases leave query-plan
    augmentation on its existing LIKE-based behaviour. Shared by
    ``execution_engine`` and ``specialist_runner`` so the two call sites
    can't drift.
    """
    ctx = _tool_context_var.get()
    if ctx is None or not ctx.planning_contexts:
        return None
    year = ctx.current_year_provider()
    return ctx.planning_contexts.get(year)
