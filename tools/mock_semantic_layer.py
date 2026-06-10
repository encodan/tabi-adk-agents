"""Mock semantic layer — synthetic recruitment data for the public showcase.

In the production TABI platform, recruitment metrics resolve through a
proprietary **MetricFlow semantic layer** over per-tenant **BigQuery** datasets
(governed metric definitions, multi-tenant isolation, dbt-managed marts). That
layer is TABI's commercial moat and is intentionally **excluded** from this
public repository.

This module is a drop-in stand-in that exposes the same public surface the ADK
``FunctionTool`` wrappers call (``query_metrics`` /
``query_distribution_values`` / ``get_available_metrics`` / ``close``) but
returns **deterministic synthetic data** instead of dispatching to MetricFlow or
BigQuery. It lets the agent layer, the grounding validator, and the eval harness
run end-to-end on mock data — exactly the "real (or mock) data" the contest
guide blesses — without leaking any proprietary query logic.

The numbers below are illustrative fixtures for a fictional company. They are
*not* real customer data. The shape of each envelope (``success`` / ``data`` /
``columns`` / ``metadata``) mirrors the production tool so downstream code is
byte-compatible.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic metric catalog + fixtures
# ---------------------------------------------------------------------------
# A small, plausible slice of recruitment-analytics metrics. Production exposes
# 50+ MetricFlow metrics; this is a representative subset for the showcase.
SYNTHETIC_METRICS: dict[str, dict[str, Any]] = {
    "time_to_hire": {"label": "Time to Hire (days)", "unit": "days", "value": 42.0},
    "time_in_stage": {"label": "Time in Stage (days)", "unit": "days", "value": 8.5},
    "pipeline_conversion_rate": {
        "label": "Pipeline Conversion Rate",
        "unit": "ratio",
        "value": 0.18,
    },
    "offer_acceptance_rate": {"label": "Offer Acceptance Rate", "unit": "ratio", "value": 0.74},
    "hires": {"label": "Hires", "unit": "count", "value": 128.0},
    "applications": {"label": "Applications", "unit": "count", "value": 4210.0},
    "interviews_per_hire": {"label": "Interviews per Hire", "unit": "ratio", "value": 6.2},
    "recruiter_capacity": {"label": "Hires per Recruiter / Month", "unit": "ratio", "value": 5.0},
}

# Per-stage funnel fixture (drives the pipeline_analyst bottleneck story).
SYNTHETIC_STAGE_FUNNEL: list[dict[str, Any]] = [
    {
        "stage_funnel__stage_name": "Application Review",
        "time_in_stage": 3.1,
        "pipeline_conversion_rate": 0.42,
    },
    {
        "stage_funnel__stage_name": "Phone Screen",
        "time_in_stage": 5.4,
        "pipeline_conversion_rate": 0.55,
    },
    {
        "stage_funnel__stage_name": "Technical Interview",
        "time_in_stage": 14.8,
        "pipeline_conversion_rate": 0.38,
    },
    {"stage_funnel__stage_name": "Onsite", "time_in_stage": 9.2, "pipeline_conversion_rate": 0.61},
    {"stage_funnel__stage_name": "Offer", "time_in_stage": 6.0, "pipeline_conversion_rate": 0.74},
]

# Per-source comparison fixture (drives sourcing_strategist).
SYNTHETIC_SOURCE_BREAKDOWN: list[dict[str, Any]] = [
    {"application__source_name": "Referral", "hires": 41, "offer_acceptance_rate": 0.86},
    {"application__source_name": "LinkedIn", "hires": 38, "offer_acceptance_rate": 0.71},
    {"application__source_name": "Job Board", "hires": 29, "offer_acceptance_rate": 0.63},
    {"application__source_name": "Agency", "hires": 20, "offer_acceptance_rate": 0.69},
]


class SemanticLayerTool:
    """Synthetic-data stand-in for the proprietary MetricFlow/BigQuery tool.

    Public surface mirrors the production ``SemanticLayerTool`` so the ADK
    ``FunctionTool`` wrappers, ``QueryExecutor`` shims, and the eval mock
    subclass can use it unchanged. All methods are async to match the real
    (HTTP-backed) signatures; none of them perform I/O here.
    """

    def __init__(
        self,
        api_base_url: str = "mock://semantic-layer",
        customer_id: str = "demo-tenant",
        api_key: str | None = None,
        internal_api_key: str | None = None,
        timeout: float = 30.0,
        retry_config: Any = None,
        circuit_breaker: Any = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.customer_id = customer_id
        self.api_key = api_key
        self.internal_api_key = internal_api_key
        self.timeout = timeout
        self._client = None  # No HTTP client in the mock.

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
        """Return synthetic rows for the requested metrics.

        Honours ``group_by`` for the two dimensions the fixtures cover (stage
        funnel + source breakdown); otherwise returns one scalar row per metric.
        """
        rows = self._rows_for(metrics, group_by)
        return {
            "success": True,
            "data": rows,
            "columns": list(rows[0].keys()) if rows else [],
            "metadata": {
                "source": "mock_semantic_layer",
                "metrics": metrics,
                "group_by": group_by or [],
                "note": "synthetic fixtures — MetricFlow/BigQuery excluded from public repo",
            },
        }

    def _rows_for(self, metrics: list[str], group_by: list[str] | None) -> list[dict[str, Any]]:
        gb = group_by or []
        if any("stage" in g for g in gb):
            return [
                {**r, **{m: r.get(m, SYNTHETIC_METRICS.get(m, {}).get("value")) for m in metrics}}
                for r in SYNTHETIC_STAGE_FUNNEL
            ]
        if any("source" in g for g in gb):
            return [
                {**r, **{m: r.get(m, SYNTHETIC_METRICS.get(m, {}).get("value")) for m in metrics}}
                for r in SYNTHETIC_SOURCE_BREAKDOWN
            ]
        # Scalar: one row, one column per requested metric.
        row: dict[str, Any] = {}
        for m in metrics:
            spec = SYNTHETIC_METRICS.get(m)
            row[m] = spec["value"] if spec else 0.0
        return [row] if row else []

    async def query_distribution_values(
        self,
        source: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5000,
    ) -> dict[str, Any]:
        """Distribution discovery — empty in the mock (agents narrate "no data")."""
        return {"success": True, "values": [], "sample_size": 0, "source": source}

    async def get_available_metrics(self) -> dict[str, Any]:
        """Static catalog snapshot from the synthetic fixtures."""
        return {
            "metrics": [{"name": k, **v} for k, v in SYNTHETIC_METRICS.items()],
            "dimensions": [
                "stage_funnel__stage_name",
                "application__source_name",
                "job__department_name",
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"customer_id": self.customer_id, "source": "mock_semantic_layer"}

    async def close(self) -> None:
        return None


__all__ = ["SemanticLayerTool", "SYNTHETIC_METRICS"]
