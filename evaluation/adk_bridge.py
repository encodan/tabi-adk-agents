"""ADK-scorer bridge.

The pinned ``google-adk==1.33.0`` wheel cannot point ``AgentEvaluator.evaluate()``
at our router/``AgentSession`` pipeline (the verify-on-wheel resolution: it builds
its own ``LocalEvalService(root_agent=…)`` over a single root agent with a hardcoded
``app_name="test_app"``). This module honours that *intent* instead:

1. Load the declarative ``eval_config.json`` criteria → typed
   ``EvalMetric`` list with the **pinned** ``config.EVAL_JUDGE_MODEL`` judge.
2. Load ``*.evalset.json`` (canonical ADK schema) via ADK's own model so alias
   casing is ADK's problem, not ours.
3. Drive each eval case's ``userContent`` through the **real**
   ``AgentSession.ask()`` (one session per case, so router multi-turn entity
   carryover is exercised) and capture the live tool trajectory.
4. Re-implement the **exact** ``agent_error`` bucketing: a cap-hit /
   salvaged / raised / timed-out case is marked an example-level *error* and is
   **excluded** from metric scoring — never scored as a hallucination — exactly
   as the legacy ``LiveEvalRunner`` does today.
5. Score the non-errored cases with the prebuilt ADK metric evaluators
   (``hallucinations_v1`` / ``safety_v1`` / ``final_response_match_v2`` /
   ``tool_trajectory_avg_score``) via ``LocalEvalService.evaluate()``.

ALL ``google.adk.evaluation`` imports are lazy (function-local). The metric
scorers pull in ``pandas`` / ``vertexai`` (the ``google-adk[eval]`` extra, not
in the default analytics venv) — importing this module in the fast
``make test-analytics`` suite must stay cheap. The scoring entrypoint is
``integration``-marked and only runs in the credentialed CI gate.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from config import EVAL_JUDGE_MODEL
from evaluation.eval_writer import CaseArtifacts, EvalRunWriter

# ---------------------------------------------------------------------------
# [public-repo stub] Proprietary pipeline modules excluded from this showcase.
#
# In the monorepo the chat-eval path drives the REAL ``AgentSession`` pipeline
# and the narrative path drives ``StorytellingService`` — both are the
# commercial core and are intentionally NOT vendored here. The ADK-eval WIRING
# (LocalEvalService construction, the four metric scorers, eval_config loading,
# the pinned-judge fail-closed guard, agent_error bucketing) is the showcase
# value and remains fully faithful below. The functions that need a live
# pipeline (``score_evalset`` / ``score_storytelling_evalset`` /
# ``_build_session``) are kept verbatim so the wiring is visible, but they
# reference the stubs below and will raise if invoked without the proprietary
# layer — the fast (non-integration) tests that exercise the scorer wiring do
# not need them.
#
# Excluded originals:
#   - core.session.APP_NAME / AgentSession  (full pipeline orchestrator)
#   - utils.time_periods.extract_time_period
#   - models.story_models.Story / StoryConfig
#   - services.storytelling_service.StorytellingService
#   - db.postgres.AnalyticsPool             (PG trace-viewer mirror)
#   - tabi_api.services.tenant_service.TenantContext
#   - tabi_api.routes.metrics.get_metricflow_service
# ---------------------------------------------------------------------------

# APP_NAME is a plain constant in the excluded ``core.session``; inline its
# canonical value so the ADK eval-set manager keying stays identical.
APP_NAME = "tabi_analytics"


def extract_time_period(_question: str) -> Any:  # [public-repo stub]
    """proprietary utils.time_periods excluded — returns ``None`` (the narrative
    driver only passes this through to the excluded StorytellingService)."""
    return None


if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime in fast suite
    from google.adk.evaluation.eval_case import EvalCase, Invocation
    from google.adk.evaluation.eval_metrics import EvalMetric
    from google.adk.evaluation.eval_set import EvalSet

    # [public-repo stub] proprietary types referenced only as hints below.
    from typing import Any as Story
    from typing import Any as StorytellingService
    from typing import Any as TenantContext

logger = structlog.get_logger(__name__)

# Artifact locations: in the showcase the evalsets + eval_config live alongside
# this module under ``evaluation/`` (the monorepo kept them under
# ``analytics/tests/eval/``).
_EVAL_DIR = Path(__file__).resolve().parent
EVALSETS_DIR = _EVAL_DIR / "evalsets"
EVAL_CONFIG_PATH = _EVAL_DIR / "eval_config.json"

# Per-turn wall-clock cap for the real pipeline drive. Mirrors the legacy
# LiveEvalRunner default; env-tunable for the slower reasoning tier.
_DEFAULT_ASK_TIMEOUT_SECONDS = 120.0


@dataclass
class CaseScore:
    """One eval case's outcome. ``errored`` ⇒ unscored, not failed (per the
    agent_error bucketing rule; see ``EvalRunResult.ok``)."""

    eval_id: str
    errored: bool
    error_message: str | None = None
    passed: bool | None = None
    metric_scores: dict[str, float | None] = field(default_factory=dict)
    drive_seconds: float | None = None
    """Wall-clock seconds for the real-pipeline drive of this case
    (``_drive_case``). Recorded for every case — errored or scored — so
    latency comparisons (e.g. the A/B harness) have a per-case number.
    Not part of the pass decision; the gate ignores it."""


@dataclass
class EvalRunResult:
    """Aggregate result for one evalset. ``ok`` is the gate signal."""

    eval_set_id: str
    cases: list[CaseScore]

    @property
    def errored_ids(self) -> list[str]:
        return [c.eval_id for c in self.cases if c.errored]

    @property
    def scored(self) -> list[CaseScore]:
        return [c for c in self.cases if not c.errored]

    @property
    def ok(self) -> bool:
        """Green iff every *scored* case passed. Errored (bucketed) cases do
        not fail the gate — but a wholly-errored run is not evidence and the
        caller asserts ``scored`` is non-empty."""
        return all(c.passed for c in self.scored)

    def scores_table(self) -> str:
        """The evidence artifact — paste before declaring pass."""
        lines = [f"evalset={self.eval_set_id}", f"{'eval_id':<44} {'status':<10} metrics"]
        for c in self.cases:
            if c.errored:
                status = "ERROR/skip"
                detail = c.error_message or ""
            else:
                status = "PASS" if c.passed else "FAIL"
                detail = ", ".join(
                    f"{k}={'-' if v is None else f'{v:.3f}'}" for k, v in c.metric_scores.items()
                )
            lines.append(f"{c.eval_id:<44} {status:<10} {detail}")
        return "\n".join(lines)


def load_advisory_metrics(config_path: Path | None = None) -> set[str]:
    """Return the names of metrics declared *advisory* in eval_config.json.

    Advisory metrics are scored and logged for visibility but do NOT
    contribute to the case's pass decision and are NOT fail-closed when
    unevaluated. The threshold field is preserved (never lowered) — only
    the metric's *gating role* changes. This implements the documented
    metric-rationale decision (`hallucinations_v1` reclassified as advisory
    after a discrepancy-class diagnostic run proved no
    fabrication residual exists; the metric's failure mode is the
    strict-judge-vs-analytical-synthesis tension intrinsic to an
    interpretive analytics product, not an agent defect).
    """
    path = config_path or EVAL_CONFIG_PATH
    raw = json.loads(path.read_text())
    advisory = raw.get("advisory_metrics", []) or []
    if not isinstance(advisory, list) or not all(isinstance(n, str) for n in advisory):
        raise ValueError(
            "eval_config.json advisory_metrics must be a list[str] (names of "
            "metrics declared as advisory per the eval-config design)."
        )
    return set(advisory)


def _finalize_pass(
    metric_scores: dict[str, float | None],
    metrics: list[Any],  # list[EvalMetric] at runtime; quoted to avoid eager ADK import
    advisory_metrics: set[str],
) -> tuple[bool, str | None]:
    """Fail-closed pass decision for a *scored* case.

    A scored case passes iff every **blocking** metric (i.e. every metric
    in ``metrics`` not in ``advisory_metrics``) produced a non-``None``
    score that meets its configured threshold. Advisory metrics are
    visible in ``metric_scores`` but are neither fail-closed nor
    threshold-gated here — the threshold-lowering ban stays in
    force; only the gating *role* of advisory metrics changes.

    Two fail modes, each with a distinct, actionable reason string:
      - any blocking metric is ``None`` → fail-closed (a
        silently-skipped/errored judge is not evidence).
      - any blocking metric is non-``None`` but below its threshold → fail
        with the per-metric score-vs-threshold breakdown.
    """
    blocking = [m for m in metrics if m.metric_name not in advisory_metrics]

    missing = sorted(m.metric_name for m in blocking if metric_scores.get(m.metric_name) is None)
    if missing:
        return False, (
            f"blocking metrics not evaluated — fail-closed: {missing}. "
            "A silently-skipped/errored judge is not evidence."
        )

    failing = sorted(
        f"{m.metric_name}={metric_scores[m.metric_name]:.3f}<{m.threshold}"
        for m in blocking
        if metric_scores[m.metric_name] < m.threshold
    )
    if failing:
        return False, f"blocking metrics below threshold: {failing}"

    return True, None


# ---------------------------------------------------------------------------
# eval_config.json -> typed EvalMetric list
# ---------------------------------------------------------------------------


def load_eval_metrics(config_path: Path | None = None) -> list[EvalMetric]:
    """Map the declarative ``eval_config.json`` criteria to concrete ADK
    criterion classes (the ``EvalConfig.criteria`` union does not
    self-discriminate ``BaseCriterion`` subclasses from JSON, so the mapping is
    explicit and type-checked here — see the eval_config.json comment).

    Fail-closed if the config tries to override the pinned judge: a scorer must
    not be able to silently drift to a weaker/retired judge.
    """
    from google.adk.evaluation.eval_metrics import (
        EvalMetric,
        HallucinationsCriterion,
        JudgeModelOptions,
        LlmAsAJudgeCriterion,
        PrebuiltMetrics,
        ToolTrajectoryCriterion,
    )
    from google.genai import types as genai_types

    from core.specialist_schema import build_generate_content_config

    path = config_path or EVAL_CONFIG_PATH
    raw = json.loads(path.read_text())
    criteria: dict[str, Any] = raw["criteria"]

    def _req(name: str, spec: dict[str, Any], key: str) -> Any:
        """Required eval_config.json key with an actionable miss (the file is
        hand-edited; a bare KeyError gives no hint which criterion is malformed)."""
        if key not in spec:
            raise ValueError(
                f"eval_config.json criterion {name!r} is missing required key {key!r} — add it."
            )
        return spec[key]

    def _judge_opts(name: str, spec: dict[str, Any]) -> JudgeModelOptions:
        opts = _req(name, spec, "judge_model_options")
        configured = opts.get("judge_model")
        if configured != EVAL_JUDGE_MODEL:
            raise ValueError(
                f"eval_config.json judge_model={configured!r} != pinned "
                f"config.EVAL_JUDGE_MODEL={EVAL_JUDGE_MODEL!r}. The judge id is "
                "pinned so a scorer cannot drift to a weaker/retired model and "
                "mask a groundedness regression. Repin in "
                "config.py, never override here."
            )
        jm_config = opts.get("judge_model_config") or {}
        return JudgeModelOptions(
            judge_model=EVAL_JUDGE_MODEL,
            judge_model_config=build_generate_content_config(
                extra=genai_types.GenerateContentConfig(**jm_config),
            ),
            num_samples=int(opts.get("num_samples", 1)),
        )

    # Anchor the metric-name dispatch to ADK's own enum (not hand-typed
    # literals): an ADK bump that renames a metric then fails fast here / in
    # test_adk_eval, instead of silently mismatching (ADK ships ~weekly).
    metrics: list[EvalMetric] = []
    for name, spec in criteria.items():
        if name == PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value:
            threshold = float(_req(name, spec, "threshold"))
            metrics.append(
                EvalMetric(
                    metric_name=name,
                    threshold=threshold,
                    criterion=ToolTrajectoryCriterion(
                        threshold=threshold,
                        match_type=ToolTrajectoryCriterion.MatchType[
                            spec.get("match_type", "IN_ORDER")
                        ],
                    ),
                )
            )
        elif name == PrebuiltMetrics.HALLUCINATIONS_V1.value:
            threshold = float(_req(name, spec, "threshold"))
            metrics.append(
                EvalMetric(
                    metric_name=name,
                    threshold=threshold,
                    criterion=HallucinationsCriterion(
                        threshold=threshold,
                        judge_model_options=_judge_opts(name, spec),
                        evaluate_intermediate_nl_responses=bool(
                            spec.get("evaluate_intermediate_nl_responses", False)
                        ),
                    ),
                )
            )
        elif name in (
            PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value,
            PrebuiltMetrics.SAFETY_V1.value,
        ):
            threshold = float(_req(name, spec, "threshold"))
            metrics.append(
                EvalMetric(
                    metric_name=name,
                    threshold=threshold,
                    criterion=LlmAsAJudgeCriterion(
                        threshold=threshold,
                        judge_model_options=_judge_opts(name, spec),
                    ),
                )
            )
        else:  # pragma: no cover - guards a future criteria key added without a mapping
            raise ValueError(
                f"eval_config.json criterion {name!r} has no concrete-criterion "
                "mapping in adk_bridge.load_eval_metrics — add one explicitly."
            )
    return metrics


def load_evalset(path: Path) -> EvalSet:
    """Parse a ``*.evalset.json`` via ADK's own model (handles alias casing)."""
    from google.adk.evaluation.eval_set import EvalSet

    return EvalSet.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Real-pipeline drive -> ADK Invocation / InferenceResult
# ---------------------------------------------------------------------------


def _resolve_ask_timeout() -> float:
    raw = os.environ.get("TABI_EVAL_ASK_TIMEOUT_SECONDS")
    try:
        return float(raw) if raw else _DEFAULT_ASK_TIMEOUT_SECONDS
    except ValueError:
        return _DEFAULT_ASK_TIMEOUT_SECONDS


def _build_session(tenant_id: str) -> Any:
    """Construct a real ``AgentSession``. Mirrors ``LiveEvalRunner._build_session``
    (same env-var fallbacks) so the bridge drives the production pipeline, not a
    mock — the whole point of the verify-on-wheel resolution.

    [public-repo stub] The proprietary ``core.session.AgentSession`` (the full
    chat pipeline orchestrator) is excluded from this showcase, so the live
    chat-eval drive path raises here. The ADK-eval WIRING above
    (``load_eval_metrics`` / ``_finalize_pass`` / the ``score_evalset`` scoring
    loop) is exercised by the fast, non-integration tests without a live
    session."""
    raise NotImplementedError(
        "core.session.AgentSession is proprietary and excluded from the public "
        "showcase repo; the live ADK-scorer drive path is not runnable here. "
        "See the [public-repo stub] note at the top of this module."
    )


def _user_text(invocation: Invocation) -> str:
    parts = invocation.user_content.parts if invocation.user_content else []
    return "".join(p.text or "" for p in (parts or []))


async def _drive_case(eval_case: EvalCase, tenant_id: str) -> tuple[list[Invocation], str | None]:
    """Drive every turn of one eval case through ONE real ``AgentSession``.

    Returns ``(actual_invocations, error_message)``. ``error_message`` is set —
    re-implementing ``AgentSession.get_last_agent_error()`` semantics exactly
    (the agent_error bucketing rule) — when ANY turn salvages (cap-hit /
    agent_error), raises, or times
    out. The caller buckets such a case as an example-level *error* and skips
    metric scoring for it (a cap-hit must never count as a hallucination).
    """
    from google.adk.evaluation.eval_case import IntermediateData, Invocation
    from google.genai import types as genai_types

    from tools.tool_tracer import capture_tool_trace

    timeout_s = _resolve_ask_timeout()
    actual: list[Invocation] = []
    turns = eval_case.conversation or []

    try:
        async with _build_session(tenant_id) as session:
            for idx, expected in enumerate(turns):
                question = _user_text(expected)
                with capture_tool_trace() as trace:
                    response_text = await asyncio.wait_for(session.ask(question), timeout=timeout_s)
                # Re-implement the exact two-channel union read (the agent_error
                # bucketing rule): salvage
                # / cap-hit on ANY turn => the whole case is bucketed `error`.
                if session.get_last_agent_error():
                    return actual, (
                        "AgentCapExceeded: specialist hit the cap / salvaged on turn "
                        f"{idx + 1} ({question!r}) — bucketed example-level error, "
                        "factuality/safety skipped (agent_error bucketing rule)"
                    )
                tool_uses = [
                    genai_types.FunctionCall(name=tc.tool_name, args=dict(tc.arguments or {}))
                    for tc in trace.to_tool_calls()
                ]
                # hallucinations_v1 grounds the response against the
                # Invocation's tool_responses (its judge {context}). Passing
                # only tool_uses starves it — every figure reads as
                # `unsupported` and the spec's core groundedness metric is
                # structurally unpassable (fix the harness, not the
                # threshold). FunctionResponse.response must be a dict.
                tool_responses = [
                    genai_types.FunctionResponse(
                        name=name,
                        response=result if isinstance(result, dict) else {"result": result},
                    )
                    for name, result in trace.call_results()
                    if result is not None
                ]
                # Follow-up: hallucinations_v1 builds its judge {context} as
                # "Developer instructions:\n{...}" from
                # ``invocation.app_details.get_developer_instructions()``. The
                # specialists legitimately cite benchmarks baked into their
                # system prompt (e.g. pipeline_analyst_v3_1.txt's stage-
                # benchmark table) — on the deterministic-query-plan path the
                # knowledge tools aren't called, so those figures come *only*
                # from the prompt. Without app_details the judge is blind to
                # the prompt and marks grounded benchmark statements
                # `unsupported`. Reconstruct the executed specialist's resolved
                # instruction and attach it (fix the harness, not the agent /
                # threshold). Fail-soft: never break the eval over this.
                # Compound-turn fix: drive the attachment off
                # ``session.last_specialists_invoked`` — the real list of
                # specialists that contributed to the turn — instead of the
                # last (possibly synthesised) ``SpecialistResponse``. Compound
                # turns carry ``synthesized_response.agent_name == "synthesis"``,
                # which the prior single-name path filtered out, so app_details
                # ended up ``None`` and BOTH specialists' prompt-grounded
                # benchmark sentences read as ``unsupported`` to the judge.
                # ``last_specialists_invoked`` never contains the "synthesis"
                # sentinel (turn-state populated only by the specialist runner,
                # see ``specialist_runner.py``), so single-specialist turns
                # keep the one-entry behaviour and compound turns attach both
                # prompts. ``hallucinations_v1`` iterates the dict and joins
                # entries with ``"\n\n"`` (hallucinations_v1.py:436), which is
                # exactly the union the compound case needs. Errored turns
                # now attach app_details too — benign, since
                # ``get_last_agent_error()`` above buckets such turns as
                # example-level errors and metric scoring is skipped (the
                # agent_error bucketing rule).
                app_details = None
                try:
                    invoked = session.last_specialists_invoked
                    if invoked:
                        from google.adk.evaluation.app_details import (
                            AgentDetails,
                            AppDetails,
                        )

                        from agents.prompts import get_agent_prompt

                        agent_details: dict[str, AgentDetails] = {
                            agent_name: AgentDetails(
                                name=agent_name,
                                instructions=get_agent_prompt(
                                    agent_name, version=session.prompt_version
                                ),
                                tool_declarations=[],
                            )
                            for agent_name in invoked
                        }
                        app_details = AppDetails(agent_details=agent_details)
                except Exception:  # noqa: BLE001 - diagnostics must not fail the gate
                    logger.warning(
                        "adk_eval_app_details_skipped", eval_id=eval_case.eval_id, exc_info=True
                    )
                    app_details = None
                actual.append(
                    Invocation(
                        invocation_id=expected.invocation_id,
                        user_content=expected.user_content,
                        final_response=genai_types.Content(
                            role="model",
                            parts=[genai_types.Part(text=response_text or "")],
                        ),
                        intermediate_data=IntermediateData(
                            tool_uses=tool_uses,
                            tool_responses=tool_responses,
                        ),
                        app_details=app_details,
                        creation_timestamp=time.time(),
                    )
                )
                # Sanctioned regeneration / diagnosis aid (OFF by default —
                # not set in CI). The evalset goldens pin the real
                # deterministic query_multiple_recruitment_metrics args
                # (Issue 2); the evalset `description` instructs maintainers to
                # REGENERATE them from a fresh dump if query_plans.py changes.
                # Set TABI_EVAL_DUMP_DETAIL=1 locally to emit each case's
                # tool_uses (args to re-pin) plus (final_response,
                # tool_responses == judge {context}) for hallucinations_v1
                # per-sentence replay. Kept env-gated so normal gate runs stay
                # clean.
                if os.environ.get("TABI_EVAL_DUMP_DETAIL"):
                    logger.info(
                        "adk_eval_case_detail",
                        eval_id=eval_case.eval_id,
                        turn=idx + 1,
                        response_text=response_text or "",
                        tool_responses=json.dumps(
                            [fr.model_dump(exclude_none=True) for fr in tool_responses]
                        ),
                        tool_uses=json.dumps(
                            [fc.model_dump(exclude_none=True) for fc in tool_uses]
                        ),
                    )
    except TimeoutError:
        return actual, f"TimeoutError: session.ask exceeded {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001 - any agent failure is an example-level error
        logger.warning("adk_eval_case_failed", eval_id=eval_case.eval_id, exc_info=True)
        return actual, f"{type(exc).__name__}: {exc}"

    return actual, None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _noop_root_agent() -> Any:
    """``LocalEvalService`` requires a ``root_agent`` but the evaluate-only path
    (we supply pre-generated inferences) never invokes inference. A no-op agent
    keeps us from constructing a real LLM agent just to satisfy the signature."""
    from collections.abc import AsyncGenerator

    from google.adk.agents.base_agent import BaseAgent

    class _NoOpRootAgent(BaseAgent):
        async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Any, None]:
            raise RuntimeError(
                "adk_bridge._NoOpRootAgent must never run inference — the bridge "
                "supplies pre-generated inferences and only uses the scorer path."
            )
            yield  # pragma: no cover - makes this an async generator

    return _NoOpRootAgent(name="tabi_eval_noop_root")


async def score_evalset(evalset_path: Path, tenant_id: str | None = None) -> EvalRunResult:
    """Drive the real pipeline for every case in ``evalset_path`` then score the
    non-errored cases with the ADK metric evaluators. Errored cases are bucketed
    (per the agent_error bucketing rule) and reported but not scored.

    ``tenant_id`` resolves from the arg, else ``TABI_EVAL_TENANT_ID``, else the
    ``seed_realistic_2yr`` legacy default. The default name is historical: the
    legacy harness ran ``MockSemanticLayerTool`` so it never resolved a real
    control-plane customer. Driving the REAL pipeline (the verify-on-wheel
    resolution)
    requires an actual ``customers`` row — the credentialed CI gate sets
    ``TABI_EVAL_TENANT_ID`` to the real tenant (mirrors ``_build_session``'s
    env-fallback pattern); never hardcode a customer id at a call site."""
    tenant_id = tenant_id or os.environ.get("TABI_EVAL_TENANT_ID") or "seed_realistic_2yr"
    from google.adk.evaluation.base_eval_service import (
        EvaluateConfig,
        EvaluateRequest,
        InferenceResult,
        InferenceStatus,
    )
    from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from google.adk.evaluation.local_eval_service import LocalEvalService

    eval_set = load_evalset(evalset_path)
    metrics = load_eval_metrics()
    advisory_metrics = load_advisory_metrics()

    # ADK's InMemoryEvalSetsManager.__init__ has no return annotation, so the
    # call is `no-untyped-call` under disallow_untyped_calls — not a real type
    # error (ADK ships py.typed; do not "re-enable once stubs land").
    manager = InMemoryEvalSetsManager()  # type: ignore[no-untyped-call]
    manager.create_eval_set(app_name=APP_NAME, eval_set_id=eval_set.eval_set_id)
    for ec in eval_set.eval_cases:
        manager.add_eval_case(app_name=APP_NAME, eval_set_id=eval_set.eval_set_id, eval_case=ec)

    cases: dict[str, CaseScore] = {}
    inference_results: list[InferenceResult] = []
    # Deliberately SEQUENTIAL (mirrors LiveEvalRunner.run_suite). Do NOT
    # asyncio.gather: each case is a live multi-turn AgentSession.ask() against
    # Gemini + the API — fanning out N× spikes QPS into 429/cap-hit salvage
    # (silently inflating the errored bucket) and interleaves the per-
    # conversation structlog scoping triage relies on.
    for ec in eval_set.eval_cases:
        drive_start = time.perf_counter()
        actual, err = await _drive_case(ec, tenant_id)
        drive_seconds = time.perf_counter() - drive_start
        if err is not None:
            cases[ec.eval_id] = CaseScore(
                eval_id=ec.eval_id,
                errored=True,
                error_message=err,
                drive_seconds=drive_seconds,
            )
            logger.info("adk_eval_case_bucketed_error", eval_id=ec.eval_id, reason=err)
            continue
        cases[ec.eval_id] = CaseScore(
            eval_id=ec.eval_id, errored=False, drive_seconds=drive_seconds
        )
        inference_results.append(
            InferenceResult(
                app_name=APP_NAME,
                eval_set_id=eval_set.eval_set_id,
                eval_case_id=ec.eval_id,
                # No real ADK session (evaluate-only path); the eval id is a
                # stable, unique stand-in the required field needs.
                session_id=ec.eval_id,
                inferences=actual,
                status=InferenceStatus.SUCCESS,
            )
        )

    if inference_results:
        service = LocalEvalService(root_agent=_noop_root_agent(), eval_sets_manager=manager)
        request = EvaluateRequest(
            inference_results=inference_results,
            evaluate_config=EvaluateConfig(eval_metrics=metrics),
        )
        async for result in service.evaluate(evaluate_request=request):
            cs = cases.get(result.eval_id)
            if cs is None:
                # ADK ships ~weekly; an eval_id we never submitted means a
                # contract change. Fail fast and loud rather than KeyError mid
                # -loop or silently dropping a score (mirrors the enum-anchored
                # dispatch discipline in load_eval_metrics).
                raise RuntimeError(
                    f"LocalEvalService.evaluate returned unknown eval_id "
                    f"{result.eval_id!r} (submitted: {sorted(cases)}). "
                    "ADK result/inference contract changed — investigate the "
                    "pinned google-adk wheel."
                )
            for m in result.overall_eval_metric_results:
                cs.metric_scores[m.metric_name] = m.score
            cs.passed, fail_reason = _finalize_pass(
                cs.metric_scores,
                metrics,
                advisory_metrics,
            )
            if fail_reason:
                cs.error_message = fail_reason
                logger.warning(
                    "adk_eval_metric_not_evaluated_fail_closed",
                    eval_id=cs.eval_id,
                    reason=fail_reason,
                )

    return EvalRunResult(eval_set_id=eval_set.eval_set_id, cases=list(cases.values()))


# ---------------------------------------------------------------------------
# Narrative-storytelling evalset driver
# ---------------------------------------------------------------------------
#
# ``score_evalset`` above drives ``AgentSession.ask()`` — the chat-router /
# specialist pipeline. The chat-narrative path is *separate*: it lives in
# ``StorytellingService.generate_story(story_type="narrative")`` and is
# invoked from ``ChatService`` (api/.../services/chat_service.py) when the
# request's mode is ``"narrative"``, bypassing the router entirely.
#
# ``narrative_chat.evalset.json`` was curated against
# ``StorytellingService`` outputs; driving it through ``score_evalset`` would
# score specialist chat output against storytelling-narrative references AND
# fail ``tool_trajectory_avg_score`` (expected ``toolUses=[]``; specialists
# emit tool calls). Both are the wrong measurement.
#
# This driver scores the narrative path directly. The narrative pipeline
# was consolidated onto the ADK ``StorytellingAgent`` runtime; this
# function is the post-consolidation regression gate for the storytelling
# runtime-convergence design.


def _narrative_text_from_story(story: Story) -> str:
    """Synthesize the judge-input text from a ``Story``.

    Matches the curation rule the evalset description records:
    *"Goldens curated by reading the dump's executive_summary + per-slide
    narrative + key_metrics labels for each question"*. We concatenate the
    executive summary with each slide's narrative — the substantive text
    the references were written against. Slide order is preserved (the
    ``Story.slides`` list is the authoritative ordering)."""
    parts: list[str] = []
    if story.executive_summary:
        parts.append(story.executive_summary)
    for slide in story.slides:
        if slide.narrative:
            parts.append(slide.narrative)
    return "\n\n".join(parts)


def _build_storytelling_tenant_context(tenant_id: str) -> TenantContext:
    """Construct the ``TenantContext`` ``StorytellingService.fetch_metrics_data``
    needs. A per-tenant BQ dataset name is derived as ``c_<lower(tenant_id)>_data``.

    [public-repo stub] ``tabi_api.services.tenant_service.TenantContext`` is
    proprietary and excluded from this showcase, so the narrative-eval path is
    not runnable here. The structure is preserved for reference; the project id
    is read from env (no hardcoded project)."""
    raise NotImplementedError(
        "tabi_api.services.tenant_service.TenantContext is proprietary and "
        "excluded from the public showcase repo; the narrative storytelling "
        "eval path is not runnable here."
    )
    # Original (preserved for reference; uses env-resolved project id):
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get(  # type: ignore[unreachable]
        "GOOGLE_CLOUD_PROJECT", "your-gcp-project"
    )
    dataset = f"c_{tenant_id.lower()}_data"
    profile_path = Path(f"/tmp/adk_bridge_storytelling_profiles/{tenant_id}")
    profile_path.mkdir(parents=True, exist_ok=True)
    return TenantContext(
        customer_id=tenant_id,
        project_id=project_id,
        dataset=dataset,
        profile_path=profile_path,
    )


async def _drive_storytelling_case(
    eval_case: EvalCase,
    tenant_id: str,
    service: StorytellingService,
    tenant_context: TenantContext,
) -> tuple[list[Invocation], str | None]:
    """Drive every turn of one narrative case through
    ``StorytellingService.generate_story(story_type="narrative")``.

    Returns ``(actual_invocations, error_message)``. ``error_message`` is
    set on timeout or any exception during ``generate_story`` — the caller
    buckets such a case as example-level *error* and skips metric scoring
    (same agent_error bucketing discipline as ``_drive_case``).

    ``tool_uses`` / ``tool_responses`` are deliberately empty for every
    invocation: the narrative pipeline does NOT fan ADK tool calls (it
    queries MetricFlow directly via the planner and synthesizes via either
    a single ``generate_content`` call or a single ``Runner.run_async``
    invocation). ``app_details`` is ``None`` for the same reason — there
    are no per-specialist instructions to attach for ``hallucinations_v1``
    to ground against."""
    from types import SimpleNamespace

    from google.adk.evaluation.eval_case import IntermediateData, Invocation
    from google.genai import types as genai_types

    # [public-repo stub] proprietary models.story_models.StoryConfig excluded.
    # ``StoryConfig`` is a plain request DTO threaded into the excluded
    # ``StorytellingService.generate_story``; the fast-suite stub service
    # ignores it. ``SimpleNamespace`` preserves the attribute-carrying shape so
    # the driver stays faithful without vendoring the story models.
    StoryConfig = SimpleNamespace

    timeout_s = _resolve_ask_timeout()
    actual: list[Invocation] = []
    turns = eval_case.conversation or []

    try:
        for expected in turns:
            question = _user_text(expected)
            story_config = StoryConfig(
                story_type="narrative",
                question=question,
                time_period=extract_time_period(question),
            )
            story = await asyncio.wait_for(
                service.generate_story(
                    config=story_config,
                    customer_id=tenant_id,
                    tenant_context=tenant_context,
                ),
                timeout=timeout_s,
            )
            actual.append(
                Invocation(
                    invocation_id=expected.invocation_id,
                    user_content=expected.user_content,
                    final_response=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text=_narrative_text_from_story(story))],
                    ),
                    intermediate_data=IntermediateData(
                        tool_uses=[],
                        tool_responses=[],
                    ),
                    app_details=None,
                    creation_timestamp=time.time(),
                )
            )
    except TimeoutError:
        return actual, f"TimeoutError: generate_story exceeded {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001 - any storytelling failure is an example-level error
        logger.warning(
            "adk_eval_storytelling_case_failed",
            eval_id=eval_case.eval_id,
            exc_info=True,
        )
        return actual, f"{type(exc).__name__}: {exc}"

    return actual, None


def _final_response_text(invocations: list[Invocation]) -> str | None:
    """Pull the final-turn narrative text out of a case's invocations.

    The mirror stores it in ``eval_case_artifacts.final_response`` so the
    judge-replay UI and the trace detail view can show what the model
    actually produced without re-driving the pipeline. Returns ``None`` when
    there are no invocations (errored-before-first-turn) or no text parts.
    """
    if not invocations:
        return None
    content = invocations[-1].final_response
    if content is None or not content.parts:
        return None
    text = "".join(p.text or "" for p in content.parts)
    return text or None


def _eval_runs_artifacts_dir() -> Path:
    """The JSONL sidecar root under the eval output dir.

    Resolved relative to this file (CWD-independent) so the gate writes to the
    same place whether invoked from the repo root or a package subdir.
    """
    return Path(__file__).resolve().parents[3] / "eval_runs"


async def _open_eval_mirror(
    stack: AsyncExitStack, evalset_path: Path, tenant_id: str | None
) -> EvalRunWriter | None:
    """Best-effort: open the trace-viewer PG mirror writer for this run.

    Returns ``None`` (and logs) when Postgres is unreachable. The storytelling
    gate's stdout / ``scores_table()`` output must never depend on the
    control-plane DB being up — the credentialed CI gate runs without it.
    Postgres is authoritative for the trace-viewer *reads*, not for the gate.

    A dedicated pool (``min=1, max=3`` per ``DatabaseConfig`` defaults — the
    eval writer is single-runner, one short txn per case) is opened here and
    closed on stack unwind; the writer's ``__aexit__`` (run-close UPDATE) runs
    before ``pool.close`` because the stack unwinds LIFO.
    """
    from core.ids import generate_ulid

    try:
        # Import inside the try so an env without asyncpg (the fast
        # `make test-analytics` suite doesn't install it) degrades the mirror
        # to a no-op rather than raising ModuleNotFoundError up through the gate.
        #
        # [public-repo stub] proprietary db.postgres.AnalyticsPool excluded —
        # the trace-viewer PG mirror is a monorepo-only sidecar. Raising here
        # makes ``_open_eval_mirror`` degrade to ``None`` (mirror disabled),
        # exactly the best-effort "Postgres unreachable" path the original
        # handles. The scorer output never depends on the mirror.
        raise ModuleNotFoundError(
            "db.postgres.AnalyticsPool is proprietary and excluded from the "
            "public showcase repo; the eval PG mirror is disabled here."
        )

        pool = AnalyticsPool.from_env()  # type: ignore[unreachable]  # noqa: F821
        await pool.connect()
    except Exception as exc:  # noqa: BLE001 — DB-down / no-asyncpg must not fail the gate
        logger.warning("eval_mirror_pool_unavailable", error=str(exc))
        return None
    stack.push_async_callback(pool.close)

    run_id = generate_ulid()
    try:
        writer = await stack.enter_async_context(
            EvalRunWriter(
                pool,
                run_id=run_id,
                evalset_path=evalset_path,
                tenant_id=tenant_id,
                artifacts_dir=_eval_runs_artifacts_dir(),
            )
        )
    except Exception as exc:  # noqa: BLE001 — a failed run-open must not fail the gate
        logger.warning("eval_mirror_open_failed", error=str(exc), run_id=run_id)
        return None
    logger.info("eval_mirror_opened", run_id=run_id, evalset_path=str(evalset_path))
    return writer


async def score_storytelling_evalset(
    evalset_path: Path, tenant_id: str | None = None
) -> EvalRunResult:
    """Narrative-evalset gate entrypoint.

    Drive every case in ``evalset_path`` through
    ``StorytellingService.generate_story(story_type="narrative")``. Score
    the non-errored cases with the prebuilt ADK metric evaluators, *minus*
    ``tool_trajectory_avg_score`` — the narrative pipeline does not fan
    ADK tool calls, so the metric is structurally meaningless on this
    path. Filtering at the driver layer (rather than via an eval_config
    override) keeps the declarative config the chat-eval source of truth
    and follows the eval-config design's "change the metric's gating role,
    not the threshold" pattern. This is **not** a threshold-lowering: the metric
    measures nothing here, so removing it from the pass decision is the
    only honest call.

    Tenant id resolves identically to ``score_evalset`` (arg → env var →
    seed default) — never hardcode a customer id at a call site.

    Narrative is consolidated onto the ADK runtime, so no flag
    flip is needed — ``generate_story`` always uses ``_generate_via_adk``.
    The ADK App is prewarmed before the first case so the build cost is
    paid once, not on the first billed request.

    NOTE (public showcase): the narrative path drives the proprietary
    MetricFlow service + ``StorytellingService``, both excluded from this
    repository, so this live gate is not runnable here. The ADK wiring below
    is preserved verbatim as a reference for how ``LocalEvalService`` is driven
    over the narrative evalset; ``score_evalset``'s drive path is likewise
    stubbed (it needs the proprietary ``AgentSession`` — see
    ``_build_session``), but its scoring/aggregation wiring is exercised by
    the fast tests in ``tests/test_adk_eval.py``."""
    raise NotImplementedError(
        "score_storytelling_evalset depends on the proprietary MetricFlow "
        "service + StorytellingService, which are excluded from the public "
        "showcase. See score_evalset for the mock-runnable specialist eval."
    )
    tenant_id = tenant_id or os.environ.get("TABI_EVAL_TENANT_ID") or "seed_realistic_2yr"
    from google.adk.evaluation.base_eval_service import (
        EvaluateConfig,
        EvaluateRequest,
        InferenceResult,
        InferenceStatus,
    )
    from google.adk.evaluation.eval_metrics import PrebuiltMetrics
    from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from google.adk.evaluation.local_eval_service import LocalEvalService

    # [public-repo stub] The MetricFlow service (``tabi_api.routes.metrics.
    # get_metricflow_service``) and ``services.storytelling_service.
    # StorytellingService`` are proprietary and excluded from this showcase, so
    # the live narrative-eval gate is not runnable here. The scoring/wiring
    # below (LocalEvalService over the same four metric scorers, minus the
    # meaningless ``tool_trajectory_avg_score``) is preserved verbatim so the
    # narrative-path eval design stays visible.
    raise NotImplementedError(
        "services.storytelling_service.StorytellingService and the MetricFlow "
        "service are proprietary and excluded from the public showcase repo; "
        "the live narrative storytelling eval gate is not runnable here. The "
        "chat-eval design is shown by score_evalset; the narrative scoring "
        "wiring below is preserved for reference."
    )

    eval_set = load_evalset(evalset_path)
    all_metrics = load_eval_metrics()
    # See docstring: tool_trajectory is meaningless for the narrative path
    # (every expected toolUse is empty; every actual is empty). Excluding
    # it from both the LocalEvalService request AND _finalize_pass so the
    # pass decision is symmetric.
    tool_trajectory_name = PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
    narrative_metrics = [m for m in all_metrics if m.metric_name != tool_trajectory_name]
    advisory_metrics = load_advisory_metrics()

    mf_service = get_metricflow_service()  # noqa: F821 — unreachable reference code (see raise above)
    await mf_service.prewarm()

    service = StorytellingService(
        use_semantic_layer=True,
        metricflow_service=mf_service,
    )
    # Mirror the API lifespan: pay the ADK App + plugins build cost once,
    # not on the first case.
    await service.prewarm_adk_app()
    tenant_context = _build_storytelling_tenant_context(tenant_id)

    manager = InMemoryEvalSetsManager()  # type: ignore[no-untyped-call]
    manager.create_eval_set(app_name=APP_NAME, eval_set_id=eval_set.eval_set_id)
    for ec in eval_set.eval_cases:
        manager.add_eval_case(app_name=APP_NAME, eval_set_id=eval_set.eval_set_id, eval_case=ec)

    cases: dict[str, CaseScore] = {}
    inference_results: list[InferenceResult] = []
    # Per-case capture for the trace-viewer mirror (eval_case_artifacts):
    # snapshot the final narrative + the originating case so judge
    # replay can reconstruct without re-reading evalset_path. Tool data is
    # empty on the narrative path (see _drive_storytelling_case docstring).
    artifacts_by_id: dict[str, CaseArtifacts] = {}
    source_json_by_id: dict[str, dict[str, Any]] = {}

    # The mirror writer is best-effort (None when PG is down) and wraps the
    # whole scoring block so its run-close UPDATE always fires on exit.
    async with AsyncExitStack() as stack:
        writer = await _open_eval_mirror(stack, evalset_path, tenant_id)

        # Sequential per-case (same QPS / 429 reasoning as score_evalset).
        # Per-case MetricFlow cache clears match the A/B harness discipline
        # (an earlier harness lesson): same 5 questions → same planner-generated queries
        # → cache hits would mask first-call cost on later cases.
        for ec in eval_set.eval_cases:
            await mf_service.clear_cache()
            mf_service.clear_sql_plan_cache()
            drive_start = time.perf_counter()
            actual, err = await _drive_storytelling_case(ec, tenant_id, service, tenant_context)
            drive_seconds = time.perf_counter() - drive_start
            artifacts_by_id[ec.eval_id] = CaseArtifacts(
                tool_uses=[],
                tool_responses=[],
                final_response=_final_response_text(actual),
                drive_seconds=drive_seconds,
            )
            source_json_by_id[ec.eval_id] = ec.model_dump()
            if err is not None:
                cases[ec.eval_id] = CaseScore(
                    eval_id=ec.eval_id,
                    errored=True,
                    error_message=err,
                    drive_seconds=drive_seconds,
                )
                logger.info(
                    "adk_eval_storytelling_case_bucketed_error",
                    eval_id=ec.eval_id,
                    reason=err,
                )
                continue
            cases[ec.eval_id] = CaseScore(
                eval_id=ec.eval_id, errored=False, drive_seconds=drive_seconds
            )
            inference_results.append(
                InferenceResult(
                    app_name=APP_NAME,
                    eval_set_id=eval_set.eval_set_id,
                    eval_case_id=ec.eval_id,
                    # No real ADK session (evaluate-only path); same eval_id-as-
                    # session_id stand-in score_evalset uses (see comment there).
                    session_id=ec.eval_id,
                    inferences=actual,
                    status=InferenceStatus.SUCCESS,
                )
            )

        if inference_results:
            eval_service = LocalEvalService(
                root_agent=_noop_root_agent(), eval_sets_manager=manager
            )
            request = EvaluateRequest(
                inference_results=inference_results,
                evaluate_config=EvaluateConfig(eval_metrics=narrative_metrics),
            )
            async for result in eval_service.evaluate(evaluate_request=request):
                cs = cases.get(result.eval_id)
                if cs is None:
                    raise RuntimeError(
                        f"LocalEvalService.evaluate returned unknown eval_id "
                        f"{result.eval_id!r} (submitted: {sorted(cases)}). "
                        "ADK result/inference contract changed — investigate the "
                        "pinned google-adk wheel."
                    )
                for m in result.overall_eval_metric_results:
                    cs.metric_scores[m.metric_name] = m.score
                cs.passed, fail_reason = _finalize_pass(
                    cs.metric_scores,
                    narrative_metrics,
                    advisory_metrics,
                )
                if fail_reason:
                    cs.error_message = fail_reason
                    logger.warning(
                        "adk_eval_storytelling_metric_not_evaluated_fail_closed",
                        eval_id=cs.eval_id,
                        reason=fail_reason,
                    )

        # Mirror every case (errored + scored) into Postgres. Errored cases
        # have empty metric_scores → no eval_results rows, but still get an
        # eval_case_artifacts row recording the error + source snapshot.
        if writer is not None:
            threshold_map: dict[str, float | None] = {
                m.metric_name: m.threshold for m in narrative_metrics
            }
            for cs in cases.values():
                await writer.write_case(
                    eval_id=cs.eval_id,
                    metric_scores=cs.metric_scores,
                    thresholds=threshold_map,
                    passed=cs.passed,
                    errored=cs.errored,
                    error_message=cs.error_message,
                    artifacts=artifacts_by_id.get(cs.eval_id)
                    or CaseArtifacts(drive_seconds=cs.drive_seconds),
                    source_case_json=source_json_by_id.get(cs.eval_id),
                )

        return EvalRunResult(eval_set_id=eval_set.eval_set_id, cases=list(cases.values()))
