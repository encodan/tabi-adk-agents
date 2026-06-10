"""ADK-scorer eval gate.

Two tiers, deliberately:

* **Fast (default suite)** — schema-drift lock on the ``*.evalset.json`` files,
  full ``eval_config.json`` → typed ``EvalMetric`` mapping, the pinned-judge
  fail-closed guard, and the ``agent_error`` bucketing aggregation.
  These import only the pandas-free ADK models, so they run in the default
  ``pytest`` suite (the private platform's ``make test-analytics``) and
  protect the artifacts on every PR.

* **``integration``-marked** — the real gate: drives the live
  ``AgentSession.ask()`` pipeline and scores with the ADK metric evaluators
  (``hallucinations_v1`` / ``safety_v1`` / ``final_response_match_v2`` /
  ``tool_trajectory_avg_score``). Excluded from the fast suite (needs the
  ``google-adk[eval]`` extra + GCP credentials + ``TABI_FLASH_MODEL`` + a live
  API). On the private platform this is ``make test-analytics-adk-eval``, the
  required pre-merge gate; a green run with the scorer silently skipped
  is NOT evidence. Not runnable in this showcase repo — the drive
  path needs the proprietary ``AgentSession`` (see the [public-repo stub]
  notes in ``evaluation/adk_bridge.py``) — but the gate's structure, fail-
  closed guards, and scoring wiring are preserved verbatim.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import EVAL_JUDGE_MODEL
from evaluation.adk_bridge import (
    EVAL_CONFIG_PATH,
    EVALSETS_DIR,
    CaseScore,
    EvalRunResult,
    load_eval_metrics,
    load_evalset,
)

EVALSET_PATHS = sorted(EVALSETS_DIR.glob("*.evalset.json"))

# Evalsets whose references describe ``StorytellingService`` narrative output
# (not chat-router / specialist output). The direct-path deletion gate drives
# these through ``score_storytelling_evalset`` instead of ``score_evalset``
# because the narrative pipeline lives in
# ``StorytellingService.generate_story(story_type="narrative")``, NOT in
# ``AgentSession.ask()``. See the module-header comment in
# ``adk_bridge.score_storytelling_evalset`` for the path discussion.
_NARRATIVE_EVALSET_NAMES = {"narrative_chat.evalset.json"}
CHAT_EVALSET_PATHS = [p for p in EVALSET_PATHS if p.name not in _NARRATIVE_EVALSET_NAMES]
STORYTELLING_EVALSET_PATHS = [p for p in EVALSET_PATHS if p.name in _NARRATIVE_EVALSET_NAMES]


# ---------------------------------------------------------------------------
# Fast: artifact schema + config mapping (pandas-free, runs every PR)
# ---------------------------------------------------------------------------


def test_evalsets_exist():
    assert EVALSET_PATHS, f"no *.evalset.json under {EVALSETS_DIR}"


@pytest.mark.parametrize("path", EVALSET_PATHS, ids=lambda p: p.name)
def test_evalset_parses_against_pinned_adk_schema(path: Path):
    """ADK ships ~weekly; this fails fast if a bump changes the evalset schema
    so the artifacts can't silently rot until the credentialed gate runs."""
    eval_set = load_evalset(path)
    assert eval_set.eval_cases, f"{path.name} has no eval cases"
    for ec in eval_set.eval_cases:
        # Either a scripted conversation or a generative scenario, never empty.
        assert ec.conversation or ec.conversation_scenario, ec.eval_id


def test_eval_config_maps_to_four_typed_metrics():
    """All four criteria map to their concrete ADK criterion classes
    with the IN_ORDER trajectory match and the pinned judge."""
    from google.adk.evaluation.eval_metrics import (
        HallucinationsCriterion,
        LlmAsAJudgeCriterion,
        PrebuiltMetrics,
        ToolTrajectoryCriterion,
    )

    metrics = {m.metric_name: m for m in load_eval_metrics()}
    # Anchored to the wheel's enum, not a copy of the literals under test.
    assert set(metrics) == {
        PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
        PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value,
        PrebuiltMetrics.HALLUCINATIONS_V1.value,
        PrebuiltMetrics.SAFETY_V1.value,
    }

    traj = metrics["tool_trajectory_avg_score"].criterion
    assert isinstance(traj, ToolTrajectoryCriterion)
    assert traj.match_type is ToolTrajectoryCriterion.MatchType.IN_ORDER

    assert isinstance(metrics["hallucinations_v1"].criterion, HallucinationsCriterion)
    assert isinstance(metrics["final_response_match_v2"].criterion, LlmAsAJudgeCriterion)
    assert isinstance(metrics["safety_v1"].criterion, LlmAsAJudgeCriterion)

    for name in ("final_response_match_v2", "hallucinations_v1", "safety_v1"):
        opts = metrics[name].criterion.judge_model_options
        assert opts.judge_model == EVAL_JUDGE_MODEL, (name, opts.judge_model)
        # Deterministic, reproducible judge.
        assert opts.judge_model_config.temperature == 0.0


def test_load_eval_metrics_fails_closed_on_judge_override(tmp_path: Path):
    """A scorer must not be able to drift to a weaker/retired judge by
    editing eval_config.json — the loader fail-closes."""
    cfg = json.loads(EVAL_CONFIG_PATH.read_text())
    cfg["criteria"]["safety_v1"]["judge_model_options"]["judge_model"] = "gemini-2.0-flash-lite"
    bad = tmp_path / "eval_config.json"
    bad.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="pinned"):
        load_eval_metrics(bad)


def test_load_eval_metrics_actionable_error_on_missing_key(tmp_path: Path):
    """A hand-edited eval_config.json missing a required key names the offending
    criterion + key, not a bare KeyError."""
    cfg = json.loads(EVAL_CONFIG_PATH.read_text())
    del cfg["criteria"]["hallucinations_v1"]["threshold"]
    bad = tmp_path / "eval_config.json"
    bad.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="hallucinations_v1.*'threshold'"):
        load_eval_metrics(bad)


# ---------------------------------------------------------------------------
# Fast: agent_error bucketing aggregation — no GCP
# ---------------------------------------------------------------------------


def test_errored_case_is_bucketed_not_failed():
    """A cap-hit / salvaged case is *unscored*, not a failure: it must not drag
    the gate red and must not be scored as a hallucination."""
    result = EvalRunResult(
        eval_set_id="t",
        cases=[
            CaseScore(eval_id="ok", errored=False, passed=True, metric_scores={"safety_v1": 1.0}),
            CaseScore(eval_id="cap", errored=True, error_message="AgentCapExceeded: …"),
        ],
    )
    assert result.errored_ids == ["cap"]
    assert [c.eval_id for c in result.scored] == ["ok"]
    assert result.ok is True  # the errored case did not turn the gate red
    assert "ERROR/skip" in result.scores_table()
    assert "cap" not in {c.eval_id for c in result.scored}


def test_failed_scored_case_turns_gate_red():
    result = EvalRunResult(
        eval_set_id="t",
        cases=[CaseScore(eval_id="x", errored=False, passed=False, metric_scores={})],
    )
    assert result.ok is False


def _metric(name: str, threshold: float):
    """Minimal stand-in for ADK ``EvalMetric`` — ``_finalize_pass`` only
    reads ``metric_name`` and ``threshold`` off each entry."""
    return SimpleNamespace(metric_name=name, threshold=threshold)


_METRICS = [
    _metric("tool_trajectory_avg_score", 1.0),
    _metric("final_response_match_v2", 0.6),
    _metric("hallucinations_v1", 0.6),
    _metric("safety_v1", 0.8),
]


def test_finalize_pass_fails_closed_on_unevaluated_blocking_metric():
    """By the fail-closed rule, a scored case must NOT pass if any *blocking*
    metric errored / was never evaluated (None or missing). A silently-skipped
    judge is not evidence."""
    from evaluation.adk_bridge import _finalize_pass

    # All present + every blocking score ≥ threshold → pass.
    ok, reason = _finalize_pass(
        {
            "tool_trajectory_avg_score": 1.0,
            "final_response_match_v2": 0.8,
            "hallucinations_v1": 0.7,
            "safety_v1": 1.0,
        },
        _METRICS,
        advisory_metrics=set(),
    )
    assert ok is True and reason is None

    # A blocking judge metric errored (None) → fail-closed.
    ok, reason = _finalize_pass(
        {
            "tool_trajectory_avg_score": 1.0,
            "final_response_match_v2": None,
            "hallucinations_v1": 0.7,
            "safety_v1": 1.0,
        },
        _METRICS,
        advisory_metrics=set(),
    )
    assert ok is False and reason is not None and "final_response_match_v2" in reason

    # A blocking metric is entirely missing → fail-closed.
    ok, reason = _finalize_pass({"safety_v1": 1.0}, _METRICS, advisory_metrics=set())
    assert ok is False and "hallucinations_v1" in reason and "final_response_match_v2" in reason


def test_finalize_pass_fails_when_blocking_below_threshold():
    """Threshold enforcement is now owned by ``_finalize_pass`` (was ADK's
    ``final_eval_status``). Each blocking metric must meet its configured
    threshold; reason cites every failing metric with score vs threshold."""
    from evaluation.adk_bridge import _finalize_pass

    ok, reason = _finalize_pass(
        {
            "tool_trajectory_avg_score": 1.0,
            "final_response_match_v2": 0.4,  # below 0.6
            "hallucinations_v1": 0.7,
            "safety_v1": 1.0,
        },
        _METRICS,
        advisory_metrics=set(),
    )
    assert ok is False
    assert reason is not None and "final_response_match_v2=0.400<0.6" in reason


def test_finalize_pass_advisory_metric_does_not_gate():
    """An advisory metric is scored and logged but does NOT
    block — neither below-threshold nor ``None`` on an advisory metric
    fails the case. The threshold field is preserved (never lowered); only
    the metric's gating *role* changes."""
    from evaluation.adk_bridge import _finalize_pass

    advisory = {"hallucinations_v1"}

    # hallucinations_v1 below threshold but advisory → still pass.
    ok, reason = _finalize_pass(
        {
            "tool_trajectory_avg_score": 1.0,
            "final_response_match_v2": 0.8,
            "hallucinations_v1": 0.3,
            "safety_v1": 1.0,
        },
        _METRICS,
        advisory_metrics=advisory,
    )
    assert ok is True and reason is None

    # hallucinations_v1 None (judge unavailable) but advisory → still pass;
    # the fail-closed rule only applies to *blocking* metrics.
    ok, reason = _finalize_pass(
        {
            "tool_trajectory_avg_score": 1.0,
            "final_response_match_v2": 0.8,
            "hallucinations_v1": None,
            "safety_v1": 1.0,
        },
        _METRICS,
        advisory_metrics=advisory,
    )
    assert ok is True and reason is None

    # A *blocking* None still fail-closes — advisory exclusion is scoped.
    ok, reason = _finalize_pass(
        {
            "tool_trajectory_avg_score": 1.0,
            "final_response_match_v2": 0.8,
            "hallucinations_v1": 0.3,
            "safety_v1": None,
        },
        _METRICS,
        advisory_metrics=advisory,
    )
    assert ok is False and reason is not None and "safety_v1" in reason


def test_load_advisory_metrics_reads_config():
    """eval_config.json's ``advisory_metrics`` array is the source of truth
    for the advisory classification; loader returns a ``set[str]``.

    Both ``hallucinations_v1`` (strict-judge-vs-synthesis tension)
    and ``safety_v1`` (pinned-judge invariant silently violated by
    ADK 1.33 SafetyEvaluatorV1) are reclassified advisory."""
    from evaluation.adk_bridge import load_advisory_metrics

    advisory = load_advisory_metrics()
    assert advisory == {"hallucinations_v1", "safety_v1"}


def test_load_advisory_metrics_rejects_malformed(tmp_path):
    """A typo in eval_config.json's ``advisory_metrics`` must fail loudly
    rather than silently treating it as empty (which would resurrect
    blocking behaviour without leaving a trace)."""
    from evaluation.adk_bridge import load_advisory_metrics

    bad = tmp_path / "eval_config.json"
    bad.write_text('{"advisory_metrics": "hallucinations_v1", "criteria": {}}')
    with pytest.raises(ValueError, match="advisory_metrics"):
        load_advisory_metrics(bad)


# ---------------------------------------------------------------------------
# Fast: _drive_case attaches per-specialist app_details (compound-turn fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drive_case_unions_app_details_across_invoked_specialists(monkeypatch):
    """Compound turns must attach EVERY invoked specialist's developer
    instructions so ``hallucinations_v1``'s judge has the prompt context it
    grounds benchmark statements against. The pre-fix single-name path
    (``get_last_specialist_response().agent_name``) returned ``"synthesis"``
    for compound turns and was filtered out — app_details ended up ``None``
    and both specialists' prompt-grounded benchmark sentences read as
    ``unsupported`` (load-bearing cause of compound ``hallucinations_v1
    ≈ 0.16–0.35`` in an earlier A/B re-run). The fix drives off
    ``session.last_specialists_invoked``; this test pins the resulting
    ``AppDetails.agent_details`` keys + per-agent instructions so a future
    regression to the single-name shape can't slip past the fast suite."""
    from contextlib import contextmanager

    pytest.importorskip(
        "google.adk.evaluation.eval_case",
        reason="ADK eval models needed (default fast suite has them)",
    )
    from google.adk.evaluation.eval_case import EvalCase, Invocation
    from google.genai import types as genai_types

    from evaluation import adk_bridge

    invoked = ["pipeline_analyst", "capacity_planner"]
    version = "v3_1"

    class _StubSession:
        prompt_version = version
        last_specialists_invoked = invoked

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def ask(self, _question):
            return "compound answer"

        def get_last_agent_error(self):
            return None

    monkeypatch.setattr(adk_bridge, "_build_session", lambda tenant_id: _StubSession())

    @contextmanager
    def _stub_tracer():
        yield SimpleNamespace(to_tool_calls=lambda: [], call_results=lambda: [])

    monkeypatch.setattr("tools.tool_tracer.capture_tool_trace", _stub_tracer)
    monkeypatch.setattr(
        "agents.prompts.get_agent_prompt",
        lambda agent_name, version: f"INSTR[{agent_name}:{version}]",
    )

    case = EvalCase(
        eval_id="t",
        conversation=[
            Invocation(
                invocation_id="t-1",
                user_content=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="bottlenecks vs goals?")],
                ),
            )
        ],
    )

    actual, error = await adk_bridge._drive_case(case, tenant_id="t-ten")

    assert error is None
    assert len(actual) == 1
    invocation = actual[0]
    assert invocation.app_details is not None, (
        "compound-turn app_details must be attached so hallucinations_v1's "
        "judge has the per-specialist instructions for grounding"
    )
    assert set(invocation.app_details.agent_details) == set(invoked)
    for name in invoked:
        assert invocation.app_details.agent_details[name].instructions == f"INSTR[{name}:{version}]"


# ---------------------------------------------------------------------------
# The real gate: live pipeline + ADK scorers. Preserved verbatim from the
# private platform's `make test-analytics-adk-eval` pre-merge gate; not runnable
# here (the drive path needs the proprietary AgentSession — see adk_bridge).
# ---------------------------------------------------------------------------

_MISSING_CREDS = not (
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    or os.environ.get("GCP_SERVICE_ACCOUNT")
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("evalset_path", CHAT_EVALSET_PATHS, ids=lambda p: p.name)
async def test_evalset_scores_green(evalset_path: Path):
    """One ADK-scorer pytest green beside the existing harness. Drives the
    REAL specialist pipeline and scores it. The fail-closed prerequisite
    is enforced, not skipped: ``TABI_FLASH_MODEL`` MUST be exported (the LLM
    judge fail-closes without it) and at least one scored case must exist — a
    wholly-errored/skipped run is NOT evidence.

    Scoped to ``CHAT_EVALSET_PATHS`` (excludes narrative_chat) because the
    storytelling-narrative evalset references curated against
    ``StorytellingService.generate_story`` output cannot validly score
    ``AgentSession.ask()`` output. The narrative evalset is driven by
    ``test_storytelling_evalset_scores_green`` instead."""
    pytest.importorskip(
        "pandas",
        reason="google-adk[eval] extra not installed — this live gate runs on the "
        "private platform (its drive path needs the proprietary AgentSession; "
        "see the [public-repo stub] notes in evaluation/adk_bridge.py)",
    )
    if _MISSING_CREDS:
        pytest.skip("no GCP credentials — this gate runs in the credentialed CI job")
    assert os.environ.get("TABI_FLASH_MODEL"), (
        "TABI_FLASH_MODEL must be exported or the LLM judge fail-closes and a "
        "green run is not evidence"
    )

    from evaluation.adk_bridge import score_evalset

    result = await score_evalset(evalset_path)
    print("\n" + result.scores_table())  # evidence — pasted into the PR

    assert result.scored, (
        f"{result.eval_set_id}: every case bucketed as error "
        f"({result.errored_ids}) — not evidence, the pipeline is broken"
    )
    assert result.ok, (
        f"{result.eval_set_id} regressed: "
        + "; ".join(f"{c.eval_id} {c.metric_scores}" for c in result.scored if not c.passed)
        + " — fix the agent (instructions/tools), NOT the threshold"
    )


# ---------------------------------------------------------------------------
# Narrative storytelling driver — shape (fast) + gate (integration)
# ---------------------------------------------------------------------------


def test_storytelling_evalset_paths_includes_narrative_chat():
    """The direct-path deletion gate is the ONLY consumer of the storytelling
    driver — if narrative_chat is silently renamed / removed (e.g. by a future
    evalset rebuild that misses the rule), the gate would parametrize over
    nothing and the fast suite would never notice. Pin the membership so a
    rename is caught at PR time, not at the credentialed gate run."""
    names = {p.name for p in STORYTELLING_EVALSET_PATHS}
    assert "narrative_chat.evalset.json" in names, (
        "narrative_chat.evalset.json missing from STORYTELLING_EVALSET_PATHS — "
        "rename / rebuild broke the deletion-gate parametrize"
    )
    assert "narrative_chat.evalset.json" not in {p.name for p in CHAT_EVALSET_PATHS}, (
        "narrative_chat must NOT also be in CHAT_EVALSET_PATHS — it would be "
        "driven through both score_evalset (wrong path) and "
        "score_storytelling_evalset (right path), wasting a billed gate run"
    )


@pytest.mark.asyncio
async def test_drive_storytelling_case_synthesizes_narrative_text_from_story() -> None:
    """The judge-input text fed to ``final_response_match_v2`` /
    ``hallucinations_v1`` must be ``executive_summary`` + each slide's
    ``narrative`` joined with ``\\n\\n`` — that's exactly the curation rule
    the evalset description records (*"executive_summary + per-slide
    narrative + key_metrics labels"*). A silent regression here (e.g.
    feeding just ``executive_summary``, or losing slide order) would
    drift the references away from the inputs without surfacing in any
    schema test. Pinned in the fast suite so a refactor of the synthesis
    helper can't silently rot the gate."""
    pytest.importorskip(
        "google.adk.evaluation.eval_case",
        reason="ADK eval models needed (default fast suite has them)",
    )
    from google.adk.evaluation.eval_case import EvalCase, Invocation
    from google.genai import types as genai_types

    from evaluation import adk_bridge

    # [public-repo stub] proprietary models.story_models.Story / Slide excluded.
    # ``_narrative_text_from_story`` reads only ``story.executive_summary`` and
    # ``story.slides[*].narrative``; a SimpleNamespace mirrors that minimal
    # shape so the synthesis assertion below stays faithful without vendoring
    # the story models.
    story = SimpleNamespace(
        executive_summary="EXEC SUMMARY paragraph.",
        slides=[
            SimpleNamespace(narrative="SLIDE 1 narrative."),
            SimpleNamespace(narrative="SLIDE 2 narrative."),
        ],
    )

    class _StubService:
        async def generate_story(
            self, *, config: object, customer_id: str, tenant_context: object
        ) -> object:
            return story

    case = EvalCase(
        eval_id="t",
        conversation=[
            Invocation(
                invocation_id="t-1",
                user_content=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="Why is our time-to-hire increasing?")],
                ),
            )
        ],
    )

    actual, error = await adk_bridge._drive_storytelling_case(
        case,
        tenant_id="t-ten",
        service=_StubService(),
        tenant_context=SimpleNamespace(customer_id="t-ten"),
    )

    assert error is None
    assert len(actual) == 1
    invocation = actual[0]
    assert invocation.final_response is not None
    text = invocation.final_response.parts[0].text
    assert text == "EXEC SUMMARY paragraph.\n\nSLIDE 1 narrative.\n\nSLIDE 2 narrative.", (
        f"synthesis must be exec_summary + per-slide narratives joined with \\n\\n; got {text!r}"
    )
    assert invocation.intermediate_data.tool_uses == [], (
        "narrative pipeline does NOT fan ADK tool calls — tool_uses must be empty "
        "(matches the evalset's deliberately-empty toolUses, see evalset description)"
    )
    assert invocation.intermediate_data.tool_responses == []
    assert invocation.app_details is None, (
        "no per-specialist instructions to attach for hallucinations_v1 — the "
        "narrative pipeline has no sub-agents to ground against"
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("evalset_path", STORYTELLING_EVALSET_PATHS, ids=lambda p: p.name)
async def test_storytelling_evalset_scores_green(evalset_path: Path):
    """Narrative-evalset regression gate.

    Drives ``narrative_chat.evalset.json`` through
    ``StorytellingService.generate_story(story_type="narrative")`` — the
    ADK runtime that narrative was consolidated onto. Scores
    ``final_response_match_v2`` (blocking @0.6) plus the advisory
    ``hallucinations_v1`` / ``safety_v1`` per ``eval_config.json``;
    excludes ``tool_trajectory_avg_score`` (the narrative pipeline does
    not fan ADK tool calls — the metric is structurally meaningless on
    this path; see ``score_storytelling_evalset`` docstring).

    Per the storytelling runtime-convergence gate: if judge
    factuality does not hold on at least 4 of 5 cases, **do not delete
    the direct path** — investigate first. Same fail-closed prerequisites
    as ``test_evalset_scores_green`` (``TABI_FLASH_MODEL`` required;
    wholly-errored run is not evidence)."""
    pytest.importorskip(
        "pandas",
        reason="google-adk[eval] extra not installed — this live gate runs on the "
        "private platform (its drive path needs the proprietary AgentSession; "
        "see the [public-repo stub] notes in evaluation/adk_bridge.py)",
    )
    if _MISSING_CREDS:
        pytest.skip("no GCP credentials — this gate runs in the credentialed CI job")
    assert os.environ.get("TABI_FLASH_MODEL"), (
        "TABI_FLASH_MODEL must be exported or the LLM judge fail-closes and a "
        "green run is not evidence"
    )

    from evaluation.adk_bridge import score_storytelling_evalset

    result = await score_storytelling_evalset(evalset_path)
    print("\n" + result.scores_table())  # evidence — pasted into the PR

    assert result.scored, (
        f"{result.eval_set_id}: every case bucketed as error "
        f"({result.errored_ids}) — not evidence, the pipeline is broken"
    )
    assert result.ok, (
        f"{result.eval_set_id} regressed: "
        + "; ".join(f"{c.eval_id} {c.metric_scores}" for c in result.scored if not c.passed)
        + " — DO NOT delete the direct path (deletion gate); investigate first"
    )
