"""Mocked-mode SemanticLayerTool for the private live-eval runner.

`MockSemanticLayerTool` is a `SemanticLayerTool` subclass that overrides every
async HTTP-touching method to read rows from the active `GoldenExample`'s
`mock_query_results` instead of dispatching to the API. The HTTP client opened
by the parent's `__init__` is never used; `close()` is overridden so teardown
doesn't touch httpx either.

Wired by the live-eval runner when `tool_mode="mocked"`. The active example is
bound on `_active_example` (a ContextVar) before each `session.ask()` and
unbound in a `finally` so binding cannot leak across examples.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from tools.mock_semantic_layer import SemanticLayerTool

if TYPE_CHECKING:
    from evaluation.golden_dataset import GoldenExample


# Bound by LiveEvalRunner per example. Read by the mock tool's query methods.
# ContextVar (not a class attribute) so the binding flows through
# ``asyncio.gather``-spawned tool calls without explicit propagation.
_active_example: contextvars.ContextVar[GoldenExample | None] = contextvars.ContextVar(
    "tabi_eval_active_example", default=None
)


@contextmanager
def bind_active_example(example: GoldenExample) -> Iterator[None]:
    """Bind ``example`` as the active mock-tool source for the duration of the
    block. Always pairs the ``set``/``reset`` so a binding cannot leak across
    examples — including when the wrapped code raises.
    """
    token = _active_example.set(example)
    try:
        yield
    finally:
        _active_example.reset(token)


class MockSemanticLayerTool(SemanticLayerTool):
    """Returns rows from the active GoldenExample's `mock_query_results`
    instead of issuing HTTP. Used by LiveEvalRunner when tool_mode='mocked'.

    Subclasses SemanticLayerTool so callers (`handle_metric_query_tool_call`,
    `QueryExecutor`) get the same envelope shape — only the data source
    changes. The parent's httpx.AsyncClient is skipped (one would otherwise
    leak per session across a 90-example run), so `close()` is also a no-op.
    """

    def __init__(
        self,
        api_base_url: str,
        customer_id: str,
        api_key: str | None = None,
        internal_api_key: str | None = None,
        timeout: float = 30.0,
        retry_config: Any = None,
        circuit_breaker: Any = None,
    ):
        # The public-repo mock ``SemanticLayerTool`` has a simple, I/O-free
        # ``__init__`` (no CircuitBreaker/RetryConfig — those live in the
        # excluded proprietary semantic_layer_tool). Delegate to it and pass the
        # supported args through; ``retry_config`` / ``circuit_breaker`` are
        # accepted-and-ignored by the mock parent. The parent already sets
        # ``self._client = None`` (no httpx client is ever opened).
        super().__init__(
            api_base_url=api_base_url,
            customer_id=customer_id,
            api_key=api_key,
            internal_api_key=internal_api_key,
            timeout=timeout,
            retry_config=retry_config,
            circuit_breaker=circuit_breaker,
        )

    async def query_metrics(
        self,
        metrics: list[str],
        group_by: list[str] | None = None,
        time_granularity: str | None = None,
        time_range: dict[str, str] | None = None,
        filters: list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        example = _active_example.get()
        rows = _rows_for_metrics(example, metrics)
        return {
            "success": True,
            "data": rows,
            "columns": list(rows[0].keys()) if rows else [],
            "metadata": {"source": "mock", "metrics": metrics},
        }

    async def query_distribution_values(
        self,
        source: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5000,
    ) -> dict[str, Any]:
        # Distribution values aren't covered by golden mock_query_results;
        # return empty so the agent narrates "no data" rather than the call
        # silently falling through to the parent's HTTP path.
        return {"success": True, "values": [], "sample_size": 0, "source": source}

    async def get_available_metrics(self) -> dict[str, Any]:
        # Static empty snapshot — agents don't depend on discovery during
        # eval; the deterministic / adaptive plan paths drive metric choice
        # from the classifier sub_intent, not from this catalog.
        return {"metrics": [], "dimensions": []}

    async def close(self) -> None:  # parent opens an httpx client we never used
        return None


def _rows_for_metrics(example: GoldenExample | None, metrics: list[str]) -> list[dict[str, Any]]:
    """Pull rows from `example.mock_query_results` for the requested metrics.

    Exact metric-name match first: this preserves precise behaviour for
    multi-row goldens where distinct queries in a plan want distinct rows.

    Fallback: if nothing matches but the example *does* carry curated rows,
    serve the full rowset. The deterministic/adaptive planners now emit
    broader metric baskets (and the classifier may pick a sub_intent whose
    plan names metrics differently) than the single curated metric each
    golden was authored with. The mocked-mode contract is "the
    agent is judged on what it does with the data it's handed" — plan
    correctness is scored independently by `eval_query_plan`, so starving
    the agent of its curated data on a name mismatch is the wrong signal.

    Returns an empty list — never raises — only when no example is bound or
    the example genuinely has no `mock_query_results`.
    """
    if example is None or not example.mock_query_results:
        return []
    wanted = set(metrics)
    exact = [r for r in example.mock_query_results if r.get("metric") in wanted]
    if exact:
        return exact
    return list(example.mock_query_results)


__all__ = ["MockSemanticLayerTool", "_active_example"]
