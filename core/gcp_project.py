"""Single GCP project-id resolver shared across analytics + api.

Returns ``None`` (never raises) so callers that rely on exporter ADC
auto-detect keep working when the project is unset. Lives in
``tabi_analytics`` (not ``tabi_core``, which ``analytics`` doesn't depend on);
``api`` reaches it via its ``tabi-analytics`` dependency.
"""

from __future__ import annotations

import os

__all__ = ["resolve_gcp_project"]


def resolve_gcp_project() -> str | None:
    """First of ``GCP_PROJECT_ID`` / ``GOOGLE_CLOUD_PROJECT`` / ``PROJECT_ID``,
    or ``None``. Not cached — env is read live so callers (and tests) see
    changes; the cost is a few dict lookups, dwarfed by per-turn LLM latency.
    """
    return (
        os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    )
