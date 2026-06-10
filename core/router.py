"""
LLM classifier router for fast agent routing using Gemini structured output.

Uses a lightweight Gemini model to classify user queries into agents,
sub-intents, and entities in a single call (~200-400ms).
Replaces the previous embeddings-based semantic router.

Supports both single-agent and multi-agent routing patterns.
"""

import asyncio
import calendar
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import structlog
from google.genai import types as genai_types

from config import (
    ALL_SUB_INTENTS,
    GROUPING_DIMENSIONS,
    SPECIALIST_AGENTS,
    get_config,
    get_genai_client,
    reset_genai_client,
)
from core.logging_config import log_timing
from core.spans import router_classify_span
from core.spans import set_attrs as set_span_attrs
from core.specialist_schema import build_generate_content_config

logger = structlog.get_logger(__name__)


@dataclass
class RouteResult:
    """Result of query classification."""

    agents: list[str]
    """List of agent names to invoke (may be single or multiple)."""

    confidence: float
    """Classification confidence (0.0-1.0)."""

    sub_intent: str | None = None
    """Sub-intent within the agent (e.g., 'bottleneck', 'overview')."""

    entities: dict[str, Any] | None = None
    """Extracted entities: department, source, job, time range, granularity, group_by."""

    inherited_fields: list[str] = field(default_factory=list)
    """Entity field names carried over from the prior turn (e.g. ['department', 'time_range'])."""

    chart_likely: bool = True
    """Hint to two-pass orchestration that this turn is likely to want a
    chart. Pass-1 includes ``propose_chart`` in its tool list when ``True``;
    drops it (saving one LLM call on scalar prompts) when ``False``.

    Initial value: unconditionally ``True`` for the entire P2 rollout. The
    classifier-driven heuristic that would set this to ``False`` for purely
    scalar prompts is deferred to P3 (router chart-likely heuristic) so we
    can calibrate against post-rollout sub-intent distribution. Until then
    this field is a no-op on retrieval correctness — pass 2 still emits
    ``chart: null`` for genuinely scalar prompts because the synthesis
    schema permits it."""

    @property
    def is_multi_agent(self) -> bool:
        """True if this query requires multiple specialists."""
        return len(self.agents) > 1

    @property
    def single_agent(self) -> str | None:
        """Get the single agent name, or None if multi-agent."""
        if len(self.agents) == 1:
            return self.agents[0]
        return None


def get_router_threshold() -> float:
    """Get router threshold from config."""
    return get_config().model.router_threshold


# ---------------------------------------------------------------------------
# Classifier response schema (Gemini structured output)
# ---------------------------------------------------------------------------

# Single source of truth for job-title abbreviation expansion. Rendered into
# both the Gemini schema description and the system prompt so the two
# surfaces can't drift (expansion correctness is load-bearing
# for filter row-count).
JOB_ABBREVIATION_HINTS: str = (
    "SDR → 'Sales Development Representative', "
    "BDR → 'Business Development Representative', "
    "PM → 'Product Manager', "
    "EM → 'Engineering Manager', "
    "SWE → 'Software Engineer'"
)

# Entity fields eligible for carryover from the prior turn. Also rendered into
# the Gemini schema enum on `inherited_fields` below so the runtime and the
# structured-output contract share one source of truth.
INHERITABLE_FIELDS: frozenset[str] = frozenset(
    {"department", "source", "time_range", "time_granularity", "job"}
)

CLASSIFIER_RESPONSE_SCHEMA = genai_types.Schema(
    type="OBJECT",
    properties={
        "agent": genai_types.Schema(
            type="STRING",
            enum=list(SPECIALIST_AGENTS),
            description="The specialist agent best suited to answer this question.",
        ),
        "sub_intent": genai_types.Schema(
            type="STRING",
            description="The specific analysis type within the agent's domain.",
            enum=list(ALL_SUB_INTENTS),
        ),
        "confidence": genai_types.Schema(
            type="NUMBER",
            description="Classification confidence from 0.0 to 1.0.",
        ),
        "is_compound": genai_types.Schema(
            type="BOOLEAN",
            description="True if the question requires multiple specialist agents.",
        ),
        "secondary_agents": genai_types.Schema(
            type="ARRAY",
            items=genai_types.Schema(type="STRING"),
            description="Additional agents needed for compound queries. Empty if not compound.",
        ),
        "inherited_fields": genai_types.Schema(
            type="ARRAY",
            items=genai_types.Schema(
                type="STRING",
                enum=sorted(INHERITABLE_FIELDS),
            ),
            description=(
                "Entity fields copied from the prior turn's entities. "
                "Empty list if there is no prior turn or no inheritance applies."
            ),
        ),
        "entities": genai_types.Schema(
            type="OBJECT",
            properties={
                "department": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    description="Department name mentioned (e.g., 'Engineering', 'Sales').",
                ),
                "source": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    description="Recruiting source/channel (e.g., 'LinkedIn', 'Referral').",
                ),
                "job": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    description=(
                        "Job title or role name mentioned (e.g., 'Data Engineer', "
                        "'Product Manager'). Expand common abbreviations to their "
                        f"full form: {JOB_ABBREVIATION_HINTS}. "
                        "Do NOT extract if the user is asking about a department or team "
                        "('Engineering', 'Sales') rather than a specific role."
                    ),
                ),
                "start_date": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    description="Start date in ISO format if a time range is mentioned.",
                ),
                "end_date": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    description="End date in ISO format if a time range is mentioned.",
                ),
                "time_granularity": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    enum=["day", "week", "month", "quarter", "year"],
                    description="Time granularity if mentioned (e.g., 'monthly' → 'month').",
                ),
                "group_by": genai_types.Schema(
                    type="STRING",
                    nullable=True,
                    enum=list(GROUPING_DIMENSIONS),
                    description=(
                        "MetricFlow dimension for group-by if user says "
                        "'by department', 'by source', etc."
                    ),
                ),
            },
        ),
    },
    required=[
        "agent",
        "sub_intent",
        "confidence",
        "is_compound",
        "secondary_agents",
        "entities",
        "inherited_fields",
    ],
)


# ---------------------------------------------------------------------------
# Classifier system prompt
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """\
You are a query classifier for a recruitment analytics platform.
Classify the user's question into exactly one specialist agent and sub-intent.
Today's date is {today}.

## Agents and Sub-Intents

### pipeline_analyst
Analyses hiring pipeline bottlenecks, stage durations, and pipeline health.
- bottleneck: Where candidates are getting stuck, what's slowing hiring down
- stage_analysis: How long candidates spend in each stage, stage-level metrics
- pipeline_health: Overall pipeline efficiency, health checks, pass-through rates
- recruitment_funnel: Show the hiring funnel / pipeline shape / stage-by-stage drop-off (volumes per stage, trapezoid visual)

Disambiguation: if the user names "funnel", "drop-off", or asks to "see / show / visualise" the pipeline shape, pick `recruitment_funnel`. If they name "bottleneck", "stuck", "slow", or "why is hiring taking so long", pick `bottleneck`.

### general_analyst
Basic hiring metrics, volumes, rates, and trend analysis.
- overview: Summary metrics — application counts, hire counts, hire rates
- breakdown: Metrics grouped by department, source, or job
- trends: Metrics over time — monthly, quarterly, year-over-year

### sourcing_strategist
Recruiting source/channel performance and ROI.
- source_comparison: Compare sources head-to-head (LinkedIn vs referral, etc.)
- source_quality: Which sources produce the best candidates (highest hire rate, fastest fill)

### offer_advisor
Offer acceptance, decline analysis, and closing strategies.
- acceptance_analysis: Offer acceptance rates, time to offer decision
- decline_analysis: Why candidates decline, decline rates by segment

### interviewing_coach
Interview process efficiency and stage optimisation.
- interview_efficiency: How long the interview process takes, scheduling gaps
- stage_optimization: Which stages are slowest, where to optimise

### capacity_planner
Hiring velocity, forecasting, and pipeline coverage.
- velocity: Hires per month, application volume trends
- forecast: Will we hit hiring goals, projected outcomes
- coverage: Pipeline depth, enough candidates in flight
- goal_attainment: Whether the team is on track to hit a stated or implied hiring target (e.g. "can we hit our target of 40 engineer hires this year?"). Distinct from `forecast` (general projection without a stated target) and `velocity` (descriptive throughput).

### data_scientist
Statistical analysis, predictions, anomaly detection, cohort analysis.
- statistical_analysis: Significance tests, correlations, comparisons
- prediction: Forecasting, predictive modelling
- anomaly_detection: Unusual patterns, outliers, deviations
- goal_attainment: Predictive goal-attainment question grounded in a stated hiring target — overlaps with capacity_planner's variant. Prefer capacity_planner when the framing emphasises planning/headcount; pick data_scientist only when the question is explicitly framed as a prediction with confidence/uncertainty.

## Entity Extraction

Extract entities ONLY when explicitly mentioned in the question:
- department: The team or department name (e.g., "Engineering", "Sales", "Product")
- source: The recruiting channel (e.g., "LinkedIn", "Indeed", "Referral")
- job: The specific job title or role (e.g., "Data Engineer", "Product Manager").
  Expand abbreviations: {job_hints}.
  Prefer department when the user names a team ("Engineering"), prefer job
  when they name a role ("Data Engineer"). If they mention both, extract both —
  the augmentation layer will use job and drop department.
- time_range: ALWAYS resolve relative phrases to ABSOLUTE ISO dates using today's \
date ({today}). Set BOTH `start_date` and `end_date` (YYYY-MM-DD).
  - "last month" / "previous month" → first/last day of the previous calendar month.
  - "this month" → first day of the current calendar month → today.
  - "last quarter" / "previous quarter" → first/last day of the PREVIOUS calendar \
quarter. NOT "the past 3 months". NOT "this year". Quarters are Q1=Jan–Mar, \
Q2=Apr–Jun, Q3=Jul–Sep, Q4=Oct–Dec.
  - "this quarter" / "current quarter" → first day of the current calendar quarter \
→ today.
  - "last year" / "previous year" → Jan 1 / Dec 31 of the previous calendar year.
  - "this year" / "ytd" → Jan 1 of the current year → today.
  - "Q1 2026" / "Q3 2025" → first/last day of that named quarter.
  - Worked example: today is {today}. "last quarter" → start_date / end_date for \
the calendar quarter immediately before the one containing {today}. Compute it \
yourself; do NOT default to the full year.
- time_granularity: "monthly" → "month", "quarterly" → "quarter", etc.
- group_by: "by department" → "job__department_name", "by source" → "application__source_name", \
"by job/role" → "application__job_name", "by stage" → "stage_transition__stage_name"

## Confidence Scoring

- 1.0: Exact match to a sub-intent's domain, no ambiguity
- 0.8-0.9: Clear match with minor ambiguity
- 0.6-0.7: Reasonable match but could fit another agent
- 0.3-0.5: Unclear, multiple agents could handle this
- 0.1-0.2: Off-topic or too vague to classify

## Prior-Turn Entity Carryover

If a PRIOR TURN block is present, decide which of its entities to carry into
the current question by populating `inherited_fields`. For each inherited
field, ALSO copy its value into the current `entities` object.

Inherit a prior-turn entity when the current question is a refinement of the
prior turn — a breakdown, narrower slice, or related metric on the same
subject — and the current question does not contradict that entity.

Do NOT inherit when any of the following apply:
- The current question introduces a different subject (a new department, a
  new metric family unrelated to the prior turn).
- The current question uses reset language: "across all", "overall",
  "company-wide", "for everyone", "every department".
- The current question explicitly contradicts a prior filter (e.g. prior
  `department=Engineering`, current mentions "Sales").
- The current `group_by` axis equals a prior filter's dimension (e.g.
  `group_by=job__department_name` when prior had `department=Engineering`).
  Group-by is the axis of the new question, not a pre-filter.

NEVER inherit `group_by`. It is always re-derived from the current question.

Only `department`, `source`, `job`, `time_range`, and `time_granularity`
are eligible for inheritance. If there is no PRIOR TURN, return an empty
`inherited_fields` list.

## Compound Queries

Set is_compound=true when the question genuinely requires analysis from \
multiple specialist domains. List the primary agent first, secondary agents \
in secondary_agents.

## Worked Examples

Use these as anchors for ambiguous routing — match the SHAPE of the user's \
question, not surface keywords.

Example: "Why are we losing so many candidates in the later stages?"
→ primary_agent: pipeline_analyst
  sub_intent: bottleneck
  secondary_agents: []
  is_compound: false
  confidence: 0.85
  reasoning: late-stage loss is pipeline stage analysis; offer-decline \
investigation only kicks in if pipeline_analyst hands off.

Example: "What sources give us the best offer-acceptance rate?"
→ primary_agent: sourcing_strategist
  sub_intent: source_quality
  secondary_agents: [offer_advisor]
  is_compound: true
  confidence: 0.80
  reasoning: blends source comparison with offer outcomes — both domains \
contribute material findings.

Example: "Show me the hiring funnel for engineering."
→ primary_agent: pipeline_analyst
  sub_intent: recruitment_funnel
  secondary_agents: []
  is_compound: false
  confidence: 0.92
  reasoning: explicit "show me ... funnel" maps to recruitment_funnel; \
filter is `department=Engineering`.

Example: "Will we hit our Q2 hiring goal?"
→ primary_agent: capacity_planner
  sub_intent: goal_attainment
  secondary_agents: []
  is_compound: false
  confidence: 0.94
  reasoning: explicit goal-attainment ask with a stated target, capacity_planner's domain.

Example: "Can we hit our target of 40 engineer hires this year at current recruiter capacity?"
→ primary_agent: capacity_planner
  sub_intent: goal_attainment
  secondary_agents: []
  is_compound: false
  confidence: 0.92
  reasoning: explicit target ("40"), explicit role ("engineer"), explicit capacity framing — goal_attainment is the right sub-intent for verifier dispatch.

Example: "Is the dip in our hire rate this quarter actually significant?"
→ primary_agent: data_scientist
  sub_intent: statistical_analysis
  secondary_agents: []
  is_compound: false
  confidence: 0.90
  reasoning: "actually significant" signals statistical-significance test, \
not a general-analyst lookup.
"""


# Version namespace for the classifier prompt cache. Bump
# this only when the prompt body materially changes — not when adding new
# sub-intents (those flow through the schema, not the prompt). The prompt-
# cache key is ``("classifier", CLASSIFIER_PROMPT_VERSION)``.
CLASSIFIER_PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_classification_cache: dict[str, tuple[RouteResult, float]] = {}

# Dedup concurrent classifier calls for the same query
_inflight: dict[str, asyncio.Future[dict[str, Any] | None]] = {}

# MetricFlow group_by dimension → entity field it would conflict with.
# An inherited filter on an entity being used as the breakdown axis would
# silently collapse the group — every group_by dimension for a carryover-
# eligible entity must have an entry here.
_DIMENSION_TO_FIELD: dict[str, str] = {
    "job__department_name": "department",
    "application__source_name": "source",
    "application__job_name": "job",
}

# Cache the formatted system prompt per date to avoid re-formatting per call
_cached_prompt: tuple[str, str] | None = None  # (date_str, formatted_prompt)


def _get_system_prompt() -> str:
    """Return the formatted system prompt, cached per calendar date."""
    global _cached_prompt
    today = date.today().isoformat()
    if _cached_prompt is not None and _cached_prompt[0] == today:
        return _cached_prompt[1]
    prompt = CLASSIFIER_SYSTEM_PROMPT.format(
        today=today,
        job_hints=JOB_ABBREVIATION_HINTS,
    )
    _cached_prompt = (today, prompt)
    return prompt


# Patterns for deterministic relative-time normalisation. Order matters:
# more specific phrases first so "Q1 2026" beats a stray "in 2026", and a
# phrase like "this quarter" wins over "this year" when both happen to
# substring-match.
_RELATIVE_TIME_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("quarter_explicit", re.compile(r"\bq([1-4])\s*(\d{4})\b", re.IGNORECASE)),
    ("year_explicit", re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE)),
    (
        "last_quarter",
        re.compile(r"\b(?:last|previous|prior)\s+quarter\b", re.IGNORECASE),
    ),
    (
        "this_quarter",
        re.compile(r"\b(?:this|current)\s+quarter\b", re.IGNORECASE),
    ),
    ("last_month", re.compile(r"\b(?:last|previous|prior)\s+month\b", re.IGNORECASE)),
    ("this_month", re.compile(r"\b(?:this|current)\s+month\b", re.IGNORECASE)),
    ("last_year", re.compile(r"\b(?:last|previous|prior)\s+year\b", re.IGNORECASE)),
    (
        "this_year",
        re.compile(r"\b(?:this|current)\s+year\b|\bytd\b|\byear[\s-]to[\s-]date\b", re.IGNORECASE),
    ),
]


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    last_day = calendar.monthrange(year, end_month)[1]
    return date(year, start_month, 1), date(year, end_month, last_day)


def _resolve_relative_time(
    kind: str,
    match: re.Match[str],
    today: date,
) -> tuple[date, date] | None:
    if kind == "quarter_explicit":
        quarter = int(match.group(1))
        year = int(match.group(2))
        return _quarter_bounds(year, quarter)

    if kind == "year_explicit":
        year = int(match.group(1))
        return date(year, 1, 1), date(year, 12, 31)

    if kind == "last_quarter":
        current_q = (today.month - 1) // 3 + 1
        if current_q == 1:
            return _quarter_bounds(today.year - 1, 4)
        return _quarter_bounds(today.year, current_q - 1)

    if kind == "this_quarter":
        current_q = (today.month - 1) // 3 + 1
        start, _ = _quarter_bounds(today.year, current_q)
        return start, today

    if kind == "last_month":
        first_of_this_month = today.replace(day=1)
        last_of_prev = first_of_this_month - timedelta(days=1)
        return last_of_prev.replace(day=1), last_of_prev

    if kind == "this_month":
        return today.replace(day=1), today

    if kind == "last_year":
        year = today.year - 1
        return date(year, 1, 1), date(year, 12, 31)

    if kind == "this_year":
        return date(today.year, 1, 1), today

    return None


def _normalize_time_range_from_question(
    question: str,
    today: date,
) -> dict[str, str] | None:
    """Resolve a relative time phrase in ``question`` to absolute ISO dates.

    The classifier prompt instructs the model to do this conversion, but
    smaller models occasionally widen "last quarter" to the full current
    year. This helper provides a deterministic fallback for the common
    phrases. Returns None when no recognised phrase appears or when the
    question contains conflicting phrases (e.g. "this quarter vs last
    quarter") — in those cases the classifier's own extraction is left
    untouched.
    """
    matches: list[tuple[date, date]] = []
    for kind, pattern in _RELATIVE_TIME_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        bounds = _resolve_relative_time(kind, match, today)
        if bounds is None:
            continue
        if any(bounds == prior for prior in matches):
            continue
        matches.append(bounds)
        if len(matches) > 1:
            return None

    if not matches:
        return None

    start, end = matches[0]
    return {"start_date": start.isoformat(), "end_date": end.isoformat()}


def _normalize_query(query: str) -> str:
    """Normalize a query for cache lookup — lowercase, strip, collapse whitespace."""
    return " ".join(query.lower().split())


def _prior_turn_hash(prior_turn: dict[str, Any] | None) -> str:
    """Stable short hash of the prior turn for cache-key inclusion."""
    if not prior_turn:
        return ""
    payload = json.dumps(
        {
            "q": _normalize_query(prior_turn.get("question", "")),
            "e": prior_turn.get("entities") or {},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _render_prior_turn_block(prior_turn: dict[str, Any] | None) -> str:
    """Render the PRIOR TURN block for the classifier prompt, or empty string."""
    if not prior_turn:
        return ""
    entities = {k: v for k, v in (prior_turn.get("entities") or {}).items() if v is not None}
    return (
        "PRIOR TURN:\n"
        f'  Question: "{prior_turn.get("question", "")}"\n'
        f"  Extracted entities: {json.dumps(entities, default=str)}\n\n"
    )


async def _classify_query(
    query: str,
    prior_turn: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Classify a query using Gemini structured output.

    Returns raw classifier response dict, or None on error.
    Deduplicates concurrent calls for the same (normalized query, prior turn)
    tuple.
    """
    client = get_genai_client()
    norm = _normalize_query(query)
    inflight_key = f"{norm}|{_prior_turn_hash(prior_turn)}"

    # Deduplicate: if another coroutine is already classifying this query, await it
    if inflight_key in _inflight:
        return await _inflight[inflight_key]

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any] | None] = loop.create_future()
    _inflight[inflight_key] = future

    config = get_config()
    model = config.model.classifier_model

    prior_block = _render_prior_turn_block(prior_turn)
    contents = f"{prior_block}CURRENT QUESTION: {query}" if prior_block else query

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=build_generate_content_config(
                extra=genai_types.GenerateContentConfig(
                    system_instruction=_get_system_prompt(),
                    response_mime_type="application/json",
                    response_schema=CLASSIFIER_RESPONSE_SCHEMA,
                    temperature=0.0,
                ),
            ),
        )
        if not response.text:
            # Empty/None text typically means Vertex blocked the call (safety,
            # recitation, or token-budget). Surface the finish_reason so a
            # safety-pin-induced regression is
            # debuggable without trawling generic exception traces. Falls
            # through to the coordinator path identically to other failures.
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            logger.warning(
                "classifier.empty_response",
                finish_reason=str(finish_reason) if finish_reason is not None else None,
                model=model,
            )
            future.set_result(None)
            return None
        result = json.loads(response.text)
        future.set_result(result)
        return result
    except Exception:
        logger.exception("Classifier call failed, falling back to coordinator")
        future.set_result(None)
        return None
    finally:
        _inflight.pop(inflight_key, None)


def _entity_carryover_enabled() -> bool:
    """Feature flag: disable prior-turn carryover without a redeploy."""
    return os.environ.get("ENTITY_CARRYOVER_ENABLED", "true").lower() != "false"


async def get_fast_route(
    query: str,
    threshold: float | None = None,
    prior_turn: dict[str, Any] | None = None,
) -> RouteResult | None:
    """``tabi.router.classify`` span wrapper around :func:`_get_fast_route_impl`.

    The classifier is a pre-runner direct-genai call with no ADK span.
    This thin wrapper records it as a span carrying the route outcome
    (structural attrs only); the impl is unchanged. No-op when no provider.
    """
    with router_classify_span() as span:
        result = await _get_fast_route_impl(query, threshold, prior_turn)
        if result is not None:
            set_span_attrs(
                span,
                {
                    "tabi.sub_intent": result.sub_intent,
                    "tabi.route_confidence": result.confidence,
                    "tabi.primary_agent": result.single_agent
                    or (result.agents[0] if result.agents else None),
                    "tabi.is_compound": result.is_multi_agent,
                },
            )
        return result


async def _get_fast_route_impl(
    query: str,
    threshold: float | None = None,
    prior_turn: dict[str, Any] | None = None,
) -> RouteResult | None:
    """Classify a user query into an agent route using LLM classification.

    Args:
        query: The raw user question.
        threshold: Confidence threshold; falls back to config default.
        prior_turn: Optional `{"question": str, "entities": dict}` describing
            the most recent prior user turn whose route entities are eligible
            for carryover. When provided, the classifier is instructed to
            decide per-field whether to inherit.
    Returns RouteResult if confidence exceeds threshold, None otherwise.
    Falls back to None (coordinator path) on any error.
    """
    if threshold is None:
        threshold = get_router_threshold()

    if not _entity_carryover_enabled():
        prior_turn = None

    model_config = get_config().model
    cache_key = f"{_normalize_query(query)}|{_prior_turn_hash(prior_turn)}"

    cached = _classification_cache.get(cache_key)
    if cached is not None:
        result, cached_at = cached
        if (time.monotonic() - cached_at) < model_config.classifier_cache_ttl:
            if result.confidence >= threshold:
                return result
            return None
        else:
            _classification_cache.pop(cache_key, None)

    start = time.perf_counter()
    raw = await _classify_query(query, prior_turn=prior_turn)
    duration_ms = int((time.perf_counter() - start) * 1000)

    if raw is None:
        log_timing(logger, "query_classification", duration_ms, status="error")
        return None

    # Flatten time range for downstream compatibility
    entities = raw.get("entities", {})
    time_range = None
    if entities.get("start_date") and entities.get("end_date"):
        time_range = {
            "start_date": entities["start_date"],
            "end_date": entities["end_date"],
        }

    # Deterministic override for common relative-time phrases. The
    # classifier is instructed to compute these but small models sometimes
    # widen "last quarter" to the full current year. The helper returns
    # ``None`` for ambiguous or unrecognised phrases, in which case the
    # classifier's own extraction stands.
    deterministic_range = _normalize_time_range_from_question(query, date.today())
    if deterministic_range is not None and deterministic_range != time_range:
        logger.info(
            "router.time_range_override",
            classifier=time_range,
            deterministic=deterministic_range,
        )
        time_range = deterministic_range

    parsed_entities = {
        "department": entities.get("department"),
        "source": entities.get("source"),
        "job": entities.get("job"),
        "time_range": time_range,
        "time_granularity": entities.get("time_granularity"),
        "group_by": entities.get("group_by"),
    }

    agent = raw["agent"]
    is_compound = raw.get("is_compound", False)
    secondary = raw.get("secondary_agents", [])
    agents = [agent] + [a for a in secondary if a != agent] if is_compound else [agent]

    # Carryover applies to single-agent turns only (see spec non-goals).
    inherited_fields: list[str] = []
    if prior_turn and not is_compound:
        raw_inherited = raw.get("inherited_fields") or []
        prior_entities = prior_turn.get("entities") or {}
        group_by = parsed_entities.get("group_by")
        conflict_field = _DIMENSION_TO_FIELD.get(group_by) if group_by else None
        for field_name in raw_inherited:
            if field_name not in INHERITABLE_FIELDS:
                continue
            if field_name == conflict_field:
                continue
            # Backfill: the classifier prompt instructs it to copy inherited
            # values into `entities`, but it sometimes lists the field without
            # populating the value. Fall back to the prior turn's value.
            if parsed_entities.get(field_name) is None:
                prior_value = prior_entities.get(field_name)
                if prior_value is None:
                    # Nothing to inherit — don't claim inheritance of a null.
                    continue
                parsed_entities[field_name] = prior_value
            inherited_fields.append(field_name)
        if (
            conflict_field
            and parsed_entities.get(conflict_field)
            and (prior_entities.get(conflict_field) == parsed_entities.get(conflict_field))
        ):
            # Drop the filter value so the query plan doesn't filter on what is now the breakdown axis.
            parsed_entities[conflict_field] = None

    result = RouteResult(
        agents=agents,
        confidence=raw["confidence"],
        sub_intent=raw["sub_intent"],
        entities=parsed_entities,
        inherited_fields=inherited_fields,
    )

    if len(_classification_cache) >= model_config.classifier_cache_max_size:
        oldest_key = min(
            _classification_cache,
            key=lambda k: _classification_cache[k][1],
        )
        _classification_cache.pop(oldest_key, None)
    _classification_cache[cache_key] = (result, time.monotonic())

    log_timing(
        logger,
        "query_classification",
        duration_ms,
        agent=agent,
        sub_intent=raw["sub_intent"],
        confidence=raw["confidence"],
        is_compound=is_compound,
    )

    if inherited_fields:
        logger.info(
            "entity_carryover",
            inherited_fields=inherited_fields,
            agent=result.single_agent,
            prior_agent=prior_turn.get("agent") if prior_turn else None,
        )

    if result.confidence >= threshold:
        return result
    return None


def clear_classification_cache() -> None:
    """Clear the classification result cache. Call between sessions if needed."""
    global _classification_cache
    count = len(_classification_cache)
    _classification_cache = {}
    if count > 0:
        logger.debug("Cleared %d cached classification results", count)


def reset_router() -> None:
    """Reset the router state (useful for testing)."""
    global _cached_prompt
    _cached_prompt = None
    _inflight.clear()
    reset_genai_client()
    clear_classification_cache()
