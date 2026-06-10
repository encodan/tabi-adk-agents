"""Eval-run mirror writer.

Mirrors evalset invocations into the private platform's eval-mirror store.
That store is authoritative for downstream reads (a trace-projection service
and an evalset-export job); the JSONL artifact written to
``analytics/eval_runs/{run_id}/`` is a human-inspection sidecar.

**Ordering contract**: PG commit first, JSONL append second. A crash between
the two leaves PG authoritative + a missing JSONL line, which the CI
``scores_table()`` gate surfaces as a row-count mismatch — safe and visible.
The reverse ordering would silently emit JSONL lines with no PG row.

Three transaction scopes per run — a single 30-minute txn would pin an MVCC
snapshot and bloat WAL. A crash between cases leaves N complete cases + an
``in_flight`` run header; dashboards filter that status out, so partial runs
are inert (not corrupting). Cleanup is deferred to a follow-up janitor.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    # [public-repo stub] proprietary db.postgres excluded — the async asyncpg
    # connection-pool type is only referenced as a type hint here. ``Any``
    # keeps the writer signature faithful without pulling in the control-plane
    # DB layer (the PG mirror is best-effort and disabled in the public repo).
    from typing import Any as AnalyticsPool

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# git_sha resolution
# ---------------------------------------------------------------------------


def resolve_git_sha() -> str | None:
    """Resolve the current commit SHA for traceability.

    Order: ``git rev-parse HEAD`` → ``GITHUB_SHA`` env var → ``None``.

    Returns ``None`` rather than a sentinel like ``"unknown"`` because a NULL
    is queryable and the cross-run regression query already filters
    ``WHERE git_sha IS NOT NULL``. An ``"unknown"`` string would silently
    pollute diff outputs.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out:
            return out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    env_sha = os.environ.get("GITHUB_SHA", "").strip()
    if env_sha:
        return env_sha
    logger.warning("eval_run_no_git_sha")
    return None


# ---------------------------------------------------------------------------
# Per-case payload
# ---------------------------------------------------------------------------


@dataclass
class CaseArtifacts:
    """The non-score data captured per case.

    Stored in a separate artifact table so the hot per-result table stays
    narrow. JSONB blobs (potentially MB-sized for narrative responses) live here.
    """

    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    tool_responses: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    drive_seconds: float | None = None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class EvalRunWriter:
    """Three-scope writer for an evalset invocation.

    Usage:
        async with EvalRunWriter(pool, run_id=..., evalset_path=..., ...) as w:
            for case in cases:
                await w.write_case(...)

    A crash inside the ``with`` block leaves the run header at
    ``status='in_flight'`` and ``finished_at IS NULL``; dashboards filter
    those out, so partial runs are inert.

    ``artifacts_dir`` is the JSONL sidecar directory; the writer creates
    ``{artifacts_dir}/{run_id}/summary.json`` and ``results.jsonl``. PG is
    authoritative — the sidecar is a human-inspection artifact and the CI
    gate input. Set ``artifacts_dir=None`` to skip the sidecar entirely.
    """

    def __init__(
        self,
        pool: AnalyticsPool,
        *,
        run_id: str,
        evalset_path: Path,
        tenant_id: str | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        self._pool = pool
        self._run_id = run_id
        self._evalset_path = evalset_path
        self._tenant_id = tenant_id
        self._artifacts_dir = artifacts_dir
        self._run_dir: Path | None = None
        self._jsonl_path: Path | None = None
        self._git_sha: str | None = None
        self._started_at: datetime | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    async def __aenter__(self) -> EvalRunWriter:
        await self._open_run()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._close_run()

    async def _open_run(self) -> None:
        self._git_sha = resolve_git_sha()
        self._started_at = datetime.now(UTC)

        await self._pool.execute(
            """
            INSERT INTO eval_runs
                (run_id, evalset_path, git_sha, started_at, tenant_id, status)
            VALUES ($1, $2, $3, $4, $5, 'in_flight')
            """,
            self._run_id,
            str(self._evalset_path),
            self._git_sha,
            self._started_at,
            self._tenant_id,
        )

        if self._artifacts_dir is not None:
            self._run_dir = self._artifacts_dir / self._run_id
            self._run_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self._run_dir / "results.jsonl"
            # Touch the file so subsequent appends are deterministic.
            self._jsonl_path.touch()
            summary_path = self._run_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": self._run_id,
                        "evalset_path": str(self._evalset_path),
                        "git_sha": self._git_sha,
                        "started_at": self._started_at.isoformat(),
                        "tenant_id": self._tenant_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    async def write_case(
        self,
        *,
        eval_id: str,
        metric_scores: dict[str, float | None],
        thresholds: dict[str, float | None] | None = None,
        passed: bool | None,
        errored: bool,
        error_message: str | None,
        rationale: str | None = None,
        artifacts: CaseArtifacts | None = None,
        source_case_json: dict[str, Any] | None = None,
    ) -> None:
        """Insert one case's metric rows + artifact row in a single short txn.

        Ordering: PG commit first; JSONL append second. A crash between leaves
        PG authoritative with a missing JSONL line, which surfaces as a
        scores_table row-count mismatch — safe and visible. The reverse
        ordering silently emits orphan JSONL rows.

        ``source_case_json`` is the full originating ``EvalCase`` model_dump
        — user-content + expected final-response + intermediateData if any —
        captured at write time so a downstream judge-rerun service can
        reconstruct ``InferenceResult`` without
        re-reading ``evalset_path`` from disk. ``None`` writes the column
        default ``{}``; replay-time code raises ``MissingSourceCaseError``
        on such pre-snapshot rows.
        """
        thresholds = thresholds or {}
        artifacts = artifacts or CaseArtifacts()

        # Per-case transaction: metric rows + artifact row commit atomically.
        async with self._pool.transaction() as conn:
            for metric_name, score in metric_scores.items():
                await conn.execute(
                    """
                    INSERT INTO eval_results
                        (run_id, eval_id, metric_name, score, threshold,
                         passed, rationale, errored, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (run_id, eval_id, metric_name) DO UPDATE
                        SET score = EXCLUDED.score,
                            threshold = EXCLUDED.threshold,
                            passed = EXCLUDED.passed,
                            rationale = EXCLUDED.rationale,
                            errored = EXCLUDED.errored,
                            error_message = EXCLUDED.error_message
                    """,
                    self._run_id,
                    eval_id,
                    metric_name,
                    score,
                    thresholds.get(metric_name),
                    passed,
                    rationale,
                    errored,
                    error_message,
                )
            await conn.execute(
                """
                INSERT INTO eval_case_artifacts
                    (run_id, eval_id, tool_uses, tool_responses,
                     final_response, drive_seconds, source_case_json)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7::jsonb)
                ON CONFLICT (run_id, eval_id) DO UPDATE
                    SET tool_uses = EXCLUDED.tool_uses,
                        tool_responses = EXCLUDED.tool_responses,
                        final_response = EXCLUDED.final_response,
                        drive_seconds = EXCLUDED.drive_seconds,
                        source_case_json = EXCLUDED.source_case_json
                """,
                self._run_id,
                eval_id,
                json.dumps(artifacts.tool_uses),
                json.dumps(artifacts.tool_responses),
                artifacts.final_response,
                artifacts.drive_seconds,
                json.dumps(source_case_json or {}),
            )

        # JSONL sidecar — best-effort. If this fails, PG is still authoritative;
        # surface the failure for operator hand-append.
        if self._jsonl_path is not None:
            try:
                with self._jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "eval_id": eval_id,
                                "metric_scores": metric_scores,
                                "thresholds": thresholds,
                                "passed": passed,
                                "errored": errored,
                                "error_message": error_message,
                                "drive_seconds": artifacts.drive_seconds,
                            },
                            default=str,
                        )
                        + "\n"
                    )
            except OSError as exc:
                logger.warning(
                    "eval_writer_jsonl_append_failed",
                    run_id=self._run_id,
                    eval_id=eval_id,
                    error=str(exc),
                )

    async def _close_run(self) -> None:
        await self._pool.execute(
            """
            UPDATE eval_runs
            SET status = 'completed', finished_at = NOW()
            WHERE run_id = $1
            """,
            self._run_id,
        )
