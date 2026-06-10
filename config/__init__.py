"""
Centralized configuration management for TABI analytics.

Loads configuration from environment variables with validation.
Supports different model configurations and agent settings.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from dotenv import load_dotenv
from google.genai import types as genai_types

if TYPE_CHECKING:
    from google.adk.planners.base_planner import BasePlanner
    from google.genai import Client as GenaiClient


logger = structlog.get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


# Available Gemini models for agents. Override the mid-tier Flash default via
# ``TABI_FLASH_MODEL``. Any substitute must appear here so the boot-time
# validator accepts it.
AVAILABLE_MODELS = [
    "gemini-3-flash-preview",  # Gemini 3 Flash — retained ONLY as the
    # pinned EVAL_JUDGE_MODEL (see below); no longer a serving slot default.
    "gemini-3.1-flash-preview",  # Gemini 3.1 Flash — mid tier
    "gemini-3.1-flash-lite",  # Gemini 3.1 Flash Lite GA — classifier routing (lowest latency)
    "gemini-3.1-pro-preview",  # Gemini 3.1 Pro
    "gemini-3.5-flash",  # Gemini 3.5 Flash — mid-tier serving default
]


# Models past the provider's lifecycle end. Catches the one case
# ``AgentModelConfig.__post_init__`` can't: an id retired but still listed in
# ``AVAILABLE_MODELS``. CI asserts ``RETIRED_MODELS ∩ AVAILABLE_MODELS == ∅``.
# A model can be withdrawn provider-side and start returning NOT_FOUND on every
# turn while still listed in AVAILABLE_MODELS — the silent-degradation case
# RETIRED_MODELS exists to catch.
RETIRED_MODELS: frozenset[str] = frozenset(
    {
        "gemini-3-pro-preview",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-3.1-flash-lite-preview",
    }
)


# Eval-judge pinning: the ADK
# `final_response_match_v2` / `hallucinations_v1` / `safety_v1` scorers run an
# LLM-as-judge. The judge id is **pinned** (not env-tunable) so a scorer can't
# silently drift to a retired/weaker model and mask a groundedness regression —
# the exact failure these metrics exist to catch. It must be in AVAILABLE_MODELS
# so the model-lifecycle validator (`_assert_no_retired_models`) and
# `AgentModelConfig.__post_init__` keep it from pointing at a retired id.
#
# JUDGE_TIER_MODELS is the explicit one-entry exempt allow-list: this id is
# deliberately *neither* classifier-tier *nor* Interactions-API-eligible (not a
# serving model — not gemini-3.5-flash, the mid-tier serving default, nor any
# gemini-3.1-* slot; not the gemini-3.1-flash-lite classifier), so the
# three-way eligibility partition CI test (added alongside
# `interactions_api_eligible()`) would false-fail it without this exemption.
# Adding a *second* judge id must be a conscious edit here — the set must never
# be widened to skip a real specialist model's eligibility decision.
EVAL_JUDGE_MODEL = "gemini-3-flash-preview"
JUDGE_TIER_MODELS: frozenset[str] = frozenset({EVAL_JUDGE_MODEL})


# Safety-pinning design — Vertex AI safety thresholds pinned across model upgrades.
# Vertex Gemini defaults drift between model versions (3.x defaults to
# BLOCK_ONLY_HIGH for most categories); pinning here makes behaviour
# deterministic. Threaded into every Agent/GenerateContentConfig site via
# ``core.specialist_schema.build_generate_content_config`` — never reach for
# the constant directly at a call site, or the anti-drift introspection test
# in ``tests/test_safety_settings_pin.py`` can be sidestepped.
#
# Categories cover the full text-modality set (JAILBREAK is the load-bearing
# one for a prompt-injection-defence spec; CIVIC_INTEGRITY surfaces in
# protected-class / EEO recruitment queries; image-modality categories are
# omitted because the platform produces no image content). Omitting a
# category would silently inherit the model-version default — by pinning
# every available text category, drift can only happen if a new category is
# added to the SDK, and ``test_safety_settings_pin`` will fail loudly.
#
# Thresholds are uniform at BLOCK_MEDIUM_AND_ABOVE for the four core text
# categories. SEXUALLY_EXPLICIT was initially pinned strictest
# (BLOCK_LOW_AND_ABOVE) but harassment-complaint summaries / parental-leave
# discussions / sex-discrimination case law are exactly the recruitment
# content that trips the LOW threshold; matching the others avoids that
# false-positive class. CIVIC_INTEGRITY is BLOCK_ONLY_HIGH because EEO and
# protected-class language is in-distribution for recruitment analytics.
#
# If the eval suite flags false positives on any category, downgrade to
# BLOCK_ONLY_HIGH per the safety-pinning design.
SAFETY_SETTINGS: list[genai_types.SafetySetting] = [
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_JAILBREAK,
        threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    genai_types.SafetySetting(
        category=genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold=genai_types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]


def _resolve_flash_model() -> str:
    """Resolve the mid-tier Flash model name. Validates against ``AVAILABLE_MODELS``
    so a typo in ``TABI_FLASH_MODEL`` fails fast with an actionable message
    rather than bubbling out of ``AgentModelConfig.__post_init__`` later."""
    model = os.getenv("TABI_FLASH_MODEL", "gemini-3.5-flash")
    if model not in AVAILABLE_MODELS:
        raise ConfigurationError(
            f"TABI_FLASH_MODEL={model!r} is not a recognised model. "
            f"Must be one of: {', '.join(AVAILABLE_MODELS)}"
        )
    return model


def _getenv_with_alias(
    new_key: str,
    old_key: str | None,
    default: str | None = None,
) -> str | None:
    """Read ``new_key``, fall back to ``old_key`` with a deprecation warning."""
    val = os.getenv(new_key)
    if val is not None:
        return val
    if old_key is not None:
        val = os.getenv(old_key)
        if val is not None:
            logger.warning(
                "config.env_alias_deprecated",
                old=old_key,
                new=new_key,
            )
            return val
    return default


def _normalise_prompt_version(value: str) -> str:
    """Translate the env-var underscore form (``v3_1``) to the canonical
    dot-separated form (``v3.1``). POSIX env vars containing dots are awkward
    to quote, so the env surface accepts underscores."""
    if not value:
        return value
    head = value[:1]
    tail = value[1:]
    if head == "v" and "_" in tail and "." not in tail:
        return head + tail.replace("_", ".")
    return value


# Models that require global location (not us-central1)
GLOBAL_LOCATION_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
]


@dataclass
class AgentModelConfig:
    """Per-role model configuration — model name + fixed thinking budget + generation params.

    ``thinking_budget`` meanings:
    - 0: thinking disabled (cheapest, fastest)
    - N > 0: fixed budget (reasoning depth is bounded)
    - -1: dynamic (model decides) — allowed but avoided in defaults because
      per-turn cost becomes unpredictable.
    """

    model: str
    thinking_budget: int = 0
    temperature: float = 0.2
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.model not in AVAILABLE_MODELS:
            raise ConfigurationError(
                f"Invalid model: {self.model!r}. Must be one of: {', '.join(AVAILABLE_MODELS)}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError(
                f"Invalid temperature: {self.temperature}. Must be between 0.0 and 2.0"
            )
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ConfigurationError(
                f"Invalid max_output_tokens: {self.max_output_tokens}. Must be at least 1"
            )


# Legacy alias.
AgentThinkingConfig = AgentModelConfig


# Specialist agent names for env var loading
SPECIALIST_AGENTS = [
    "pipeline_analyst",
    "general_analyst",
    "sourcing_strategist",
    "offer_advisor",
    "interviewing_coach",
    "capacity_planner",
    "data_scientist",
]

# Sub-intents per agent — defines the full routing taxonomy.
# Used by the classifier schema, query plans, and exemplars.
AGENT_SUB_INTENTS: dict[str, list[str]] = {
    "pipeline_analyst": [
        "bottleneck",
        "stage_analysis",
        "pipeline_health",
        "recruitment_funnel",
    ],
    "general_analyst": ["overview", "breakdown", "trends"],
    "sourcing_strategist": ["source_comparison", "source_quality"],
    "offer_advisor": ["acceptance_analysis", "decline_analysis"],
    "interviewing_coach": ["interview_efficiency", "stage_optimization"],
    "capacity_planner": ["velocity", "forecast", "coverage", "goal_attainment"],
    "data_scientist": [
        "statistical_analysis",
        "prediction",
        "anomaly_detection",
        "goal_attainment",
    ],
}

ALL_SUB_INTENTS: list[str] = [si for subs in AGENT_SUB_INTENTS.values() for si in subs]

# MetricFlow dimensions available for user-facing group-by operations.
# Shared between the classifier schema and query plan augmentation.
GROUPING_DIMENSIONS: list[str] = [
    "job__department_name",
    "application__source_name",
    "application__job_name",
    "stage_transition__stage_name",
    # Stage funnel dimensions (fct_stage_funnel)
    "stage_funnel__stage_name",
    "stage_funnel__department_name",
    "stage_funnel__source_name",
    "stage_funnel__job_name",
    # Opening dimensions (fct_openings)
    "opening__job_name",
    "opening__department_name",
]


# Per-agent model defaults — sane boot config when no env vars are set.
# Override any slot via ``TABI_MODEL_<AGENT>`` / ``TABI_THINKING_<AGENT>``.
def _default_agent_models() -> dict[str, AgentModelConfig]:
    flash = _resolve_flash_model()
    pro = "gemini-3.1-pro-preview"
    return {
        "pipeline_analyst": AgentModelConfig(pro, 4096),
        "data_scientist": AgentModelConfig(pro, 8192),
        "sourcing_strategist": AgentModelConfig(flash, 3072),
        "capacity_planner": AgentModelConfig(flash, 3072),
        "offer_advisor": AgentModelConfig(flash, 2048),
        "interviewing_coach": AgentModelConfig(flash, 2048),
        "general_analyst": AgentModelConfig(flash, 1024),
    }


# Legacy export kept for callers importing ``DEFAULT_AGENT_THINKING``.
DEFAULT_AGENT_THINKING = _default_agent_models()
DEFAULT_AGENT_MODELS = DEFAULT_AGENT_THINKING


@dataclass
class ModelTieringConfig:
    """Tiered model selection — one ``AgentModelConfig`` per role.

    Legacy flat fields (``default_model``, ``synthesis_model``, etc.) are
    exposed as read-only aliases so downstream code can migrate incrementally.
    """

    agents: dict[str, AgentModelConfig]
    """Specialist agent name → model/thinking config."""

    synthesis: dict[str, AgentModelConfig]
    """Context → synthesis model. Keys: "single" (one specialist) and "multi"
    (orchestrator-merged multi-agent response)."""

    classifier: AgentModelConfig
    """Lightweight classifier for sub-intent routing."""

    handoff_synthesis: AgentModelConfig
    """Model/config for the handoff-chain synthesis (see handoff.py)."""

    storytelling: AgentModelConfig
    """Model for long-form narrative generation."""

    default: AgentModelConfig
    """Fallback model for unrecognised agents, the coordinator, and any call
    site not covered by a more specific slot."""

    router_threshold: float = 0.6
    """Minimum confidence score for fast routing (0.0–1.0)."""

    classifier_cache_ttl: float = 600.0
    """TTL in seconds for classifier result cache."""

    classifier_cache_max_size: int = 100
    """Max entries in classifier result cache."""

    thinking_enabled: bool = True
    """Master toggle for thinking budgets. When False, all agents use thinking=0."""

    secondary_agent_weight: float = 0.5
    """Relevance weight for secondary agents in handoff synthesis.
    Primary uses ``route.confidence``; secondaries use it multiplied by this
    weight. Bounded to [0, 1]."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.router_threshold <= 1.0:
            raise ConfigurationError(
                f"Invalid router threshold: {self.router_threshold}. Must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.secondary_agent_weight <= 1.0:
            raise ConfigurationError(
                f"Invalid secondary_agent_weight: {self.secondary_agent_weight}. "
                "Must be between 0.0 and 1.0"
            )
        if "single" not in self.synthesis or "multi" not in self.synthesis:
            raise ConfigurationError(
                "ModelTieringConfig.synthesis must define both 'single' and 'multi' entries"
            )

    def get_agent_thinking(self, agent_name: str) -> int:
        """Get thinking budget for a specific agent (respects the master toggle)."""
        if not self.thinking_enabled:
            return 0
        cfg = self.agents.get(agent_name)
        return cfg.thinking_budget if cfg else 0

    def get_agent_model(self, agent_name: str) -> AgentModelConfig:
        """Return the agent's config, falling back to ``default`` when unmapped."""
        return self.agents.get(agent_name) or self.default

    # Legacy flat-field aliases. New code should read structured fields
    # (``config.models.default.model``) instead.
    @property
    def default_model(self) -> str:
        return self.default.model

    @property
    def synthesis_model(self) -> str:
        return self.synthesis["multi"].model

    @property
    def storytelling_model(self) -> str:
        return self.storytelling.model

    @property
    def classifier_model(self) -> str:
        return self.classifier.model

    @property
    def agent_thinking(self) -> dict[str, AgentModelConfig]:
        return self.agents


# Legacy alias for callers importing ``ModelConfig``.
ModelConfig = ModelTieringConfig


# Single source of truth for every model-bearing slot ``ModelTieringConfig``
# resolves to. Consumed by the retired-id guard and the lifecycle test's
# count + rejection matrix — a new slot not registered here fails that test's
# count assertion rather than silently widening the retired-id bypass. The
# third tuple element is the dedicated injecting env var, or ``None`` for
# slots with no dedicated var (the env-var matrix filters those out).
def _all_resolved_model_slots(
    models: ModelTieringConfig,
) -> Iterator[tuple[str, AgentModelConfig, str | None]]:
    """Yield ``(slot_name, resolved_config, injecting_env_var | None)`` for
    all 13 model-bearing slots, in a stable order."""
    for agent_name, cfg in models.agents.items():
        yield f"agents[{agent_name}]", cfg, f"TABI_MODEL_{agent_name.upper()}"
    yield "default", models.default, "TABI_MODEL_DEFAULT"
    # Flash-hardwired; only injectable via TABI_FLASH_MODEL (no dedicated var).
    yield "synthesis[single]", models.synthesis["single"], None
    yield "synthesis[multi]", models.synthesis["multi"], "TABI_MODEL_SYNTHESIS"
    yield "classifier", models.classifier, "TABI_MODEL_CLASSIFIER"
    yield "handoff_synthesis", models.handoff_synthesis, "TABI_MODEL_HANDOFF_SYNTHESIS"
    yield "storytelling", models.storytelling, "TABI_MODEL_STORYTELLING"


def _assert_no_retired_models(models: ModelTieringConfig) -> None:
    """Fail-closed if any resolved slot selected a retired model id. Only
    fires for ids still in ``AVAILABLE_MODELS`` (``__post_init__`` already
    rejected absent ones); the distinct "retired" wording lets the lifecycle
    test attribute the failure to this path.
    """
    for slot_name, cfg, env_var in _all_resolved_model_slots(models):
        if cfg.model in RETIRED_MODELS:
            via = f" (set via {env_var})" if env_var else ""
            raise ConfigurationError(
                f"Model {cfg.model!r} for slot {slot_name!r}{via} is retired "
                f"(past Google's lifecycle end; see RETIRED_MODELS). "
                f"Pick a current model from: {', '.join(AVAILABLE_MODELS)}"
            )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str
    """Log level (DEBUG, INFO, WARNING, ERROR)."""

    subdir: str
    """Subdirectory for log files."""

    project_root: str | None = None
    """Project root directory for logs."""


@dataclass
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) configuration."""

    enabled: bool
    """Whether RAG features are enabled."""

    location: str
    """GCP region for Vertex AI RAG Engine."""

    min_similarity: float
    """Minimum similarity score for retrieval (0.0-1.0)."""

    default_top_k: int
    """Default number of results to retrieve."""

    min_rating_to_index: int
    """Minimum feedback rating to index (1-5)."""

    min_response_length: int
    """Minimum response length to index (characters)."""

    def __post_init__(self):
        """Validate RAG configuration."""
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigurationError(
                f"Invalid min_similarity: {self.min_similarity}. Must be between 0.0 and 1.0"
            )

        if self.default_top_k < 1:
            raise ConfigurationError(
                f"Invalid default_top_k: {self.default_top_k}. Must be at least 1"
            )

        if not 1 <= self.min_rating_to_index <= 5:
            raise ConfigurationError(
                f"Invalid min_rating_to_index: {self.min_rating_to_index}. Must be between 1 and 5"
            )


@dataclass
class QueryBatchingConfig:
    """Configuration for query batching optimization.

    Query batching collects independent metric queries that arrive within
    a short time window and executes them in parallel using asyncio.gather().
    This reduces total latency when the LLM makes multiple tool calls.
    """

    enabled: bool
    """Whether query batching is enabled."""

    batch_window_ms: float
    """Time to wait for additional queries before executing (milliseconds).
    Default 50ms is imperceptible but allows batching of concurrent queries."""

    max_batch_size: int
    """Maximum queries per batch. Triggers immediate execution when reached."""

    def __post_init__(self):
        """Validate query batching configuration."""
        if self.batch_window_ms < 0:
            raise ConfigurationError(
                f"Invalid batch_window_ms: {self.batch_window_ms}. Must be non-negative"
            )

        if self.batch_window_ms > 1000:
            raise ConfigurationError(
                f"batch_window_ms too high: {self.batch_window_ms}. "
                "Maximum is 1000ms to avoid excessive latency"
            )

        if self.max_batch_size < 1:
            raise ConfigurationError(
                f"Invalid max_batch_size: {self.max_batch_size}. Must be at least 1"
            )


@dataclass
class FeatureFlags:
    """Runtime feature flags."""

    structured_output_enabled: bool = False
    """Env: TABI_STRUCTURED_OUTPUT_ENABLED. Structured-output kill-switch:
    when on, specialists emit Gemini-validated ``SpecialistResponse`` JSON and
    the session emits ``claim`` / ``chart`` / ``confidence`` SSE events ahead
    of the prose. When off, the legacy free-form path is preserved
    byte-identically."""

    chart_by_reference_enabled: bool = True
    """Env: TABI_CHART_BY_REFERENCE_ENABLED. **Deprecated no-op** since the
    two-pass synthesis change; scheduled for removal after the GA soak.
    ``propose_chart`` is now the only chart tool (the two-pass change deleted
    ``create_visualization``) and the prompt rewrite that references
    ``query_result_id`` is unconditional, so every code path needs the handle
    registry populated — handle registration in ``_register_query_handle`` no
    longer reads this flag. Setting it to ``False`` has no effect; it remains
    in config solely to preserve env var compatibility during the soak. See
    the chart-by-reference design note."""

    two_pass_synthesis_enabled: bool = True
    """Env: TABI_TWO_PASS_SYNTHESIS_ENABLED. Kill-switch (now on by
    default). Specialist invocation splits into two LLM calls — pass 1
    retrieves data (tools=AUTO, drains the loop), pass 2 synthesizes prose
    + a typed ``ChartIntent`` under ``response_schema = AnswerEnvelope``
    with tools forbidden (``function_calling_config.mode = NONE``). This
    removes the failure modes that produce salvages (tool-cap exhaustion,
    malformed chart payloads, scalar-loop pathology) rather than patching
    them. The legacy single-pass branch is still present in
    ``_run_specialist_collect`` as an emergency fallback; planned removal
    after a stable production soak. See the two-pass-synthesis design
    note."""

    bq_agent_analytics_enabled: bool = False
    """Env: TABI_BQ_AGENT_ANALYTICS_ENABLED. Attach ADK's
    ``BigQueryAgentAnalyticsPlugin`` so prompt-cache hit rate is measured.
    Off until the Terraform dataset + privacy review land, so the ``App``
    refactor merges independently of the BQ dependency. Capture is
    metadata-only regardless (see ``build_session_plugins``)."""

    tenant_scope_guardrail_enabled: bool = True
    """Env: TABI_TENANT_SCOPE_GUARDRAIL_ENABLED. Tenant-scope guardrail —
    fail-closed tenant-scope assertion in ``GuardrailPlugin.before_tool_callback``.
    Always-on by design; the flag exists **only** so the
    bucketing-independent isolation comparison can flip it off to demonstrate
    the guardrail-off regression (cross-tenant rows returned + answer
    synthesized over them). Do not turn this off in production; the API auth
    middleware is the primary tenant boundary but this is defense-in-depth and
    the cheap last line in a documented threat model."""

    def __post_init__(self) -> None:
        # ``chart_by_reference_enabled`` is a deprecated no-op (see field
        # docstring), but the combination ``two_pass=True + cbr=False`` is
        # still rejected at construction. The check no longer guards a
        # correctness bug — handle registration is unconditional now — but
        # the combo signals confused intent (operator believes the kill-
        # switch is live), so we fail loud rather than accept inconsistent
        # config. Drop this check when the flag itself is removed.
        if self.two_pass_synthesis_enabled and not self.chart_by_reference_enabled:
            raise ConfigurationError(
                "TABI_CHART_BY_REFERENCE_ENABLED is a deprecated no-op; "
                "setting it to false while TABI_TWO_PASS_SYNTHESIS_ENABLED=true "
                "indicates a misunderstanding of the live config. Either "
                "unset TABI_CHART_BY_REFERENCE_ENABLED or set it to true."
            )


@functools.lru_cache(maxsize=1)
def _calibration_passed() -> bool:
    """Return True iff ``evaluation/baseline.yaml`` reports
    ``calibration.passed: true``.

    Cached for the process lifetime — baseline.yaml is committed, so changes
    require a restart.
    """
    try:
        import yaml

        # parents[1] = the repo root (this file lives at <root>/config/).
        baseline_path = Path(__file__).resolve().parents[1] / "evaluation" / "baseline.yaml"
        data = yaml.safe_load(baseline_path.read_text())
    except Exception:
        # Missing file / YAML module / parse error — fail safe (gate regen off).
        return False
    if not isinstance(data, dict):
        return False
    calibration = data.get("calibration")
    if not isinstance(calibration, dict):
        return False
    return bool(calibration.get("passed", False))


@dataclass
class ValidationConfig:
    """Post-response validation policy — two-threshold decision model
    (soft→annotate, hard→regenerate) plus a gated regeneration budget."""

    enabled: bool = True
    """Master toggle. When False, validation never runs (kill-switch)."""

    soft_threshold: float = 0.05
    """Relative deviation above which a claim is annotated with its source value."""

    hard_threshold: float = 0.20
    """Relative deviation above which we trigger one regeneration attempt."""

    absolute_tolerance: float = 0.5
    """Absolute tolerance for small values (< 10). Below this we treat the
    claim as matching regardless of relative deviation."""

    log_level: str = "warning"
    """Log level for validation events (debug|info|warning|error)."""

    max_regen: int = 0
    """Regeneration budget. Resolved in ``from_env`` to ``1`` iff validation is
    enabled and structured output has shipped, otherwise ``0`` (annotate-only).
    Override explicitly with ``TABI_VALIDATION_MAX_REGEN`` (e.g. set to ``0`` to
    force annotate-only). The calibration gate was dropped — confidence
    calibration governs *confidence-as-discriminator*, not deviation-driven
    regen, which keys off hard_threshold (≥20% off) and works regardless."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.soft_threshold <= 1.0:
            raise ConfigurationError(
                f"Invalid soft_threshold: {self.soft_threshold}. Must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.hard_threshold <= 1.0:
            raise ConfigurationError(
                f"Invalid hard_threshold: {self.hard_threshold}. Must be between 0.0 and 1.0"
            )
        if self.hard_threshold < self.soft_threshold:
            raise ConfigurationError(
                f"hard_threshold ({self.hard_threshold}) must be >= "
                f"soft_threshold ({self.soft_threshold})"
            )
        if self.absolute_tolerance < 0:
            raise ConfigurationError(
                f"Invalid absolute_tolerance: {self.absolute_tolerance}. Must be non-negative"
            )
        valid_log_levels = ("debug", "info", "warning", "error")
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log_level: {self.log_level}. "
                f"Must be one of: {', '.join(valid_log_levels)}"
            )

    # Legacy alias — pre-Phase-2 code reads ``deviation_threshold``.
    @property
    def deviation_threshold(self) -> float:
        return self.soft_threshold


# Legacy alias.
ResponseValidationConfig = ValidationConfig


@dataclass
class ContextAssemblyConfig:
    """Configuration for context assembly service."""

    enabled: bool
    """Whether context assembly is enabled (env: CONTEXT_ASSEMBLY_ENABLED)."""

    token_budget: int
    """Total token budget for assembled context (env: CONTEXT_TOKEN_BUDGET)."""

    include_benchmarks: bool
    """Whether to auto-select benchmarks (env: CONTEXT_INCLUDE_BENCHMARKS)."""

    include_exemplars: bool
    """Whether to include query exemplars (env: CONTEXT_INCLUDE_EXEMPLARS)."""

    def __post_init__(self) -> None:
        if self.token_budget < 100:
            raise ConfigurationError(
                f"Invalid token_budget: {self.token_budget}. Must be at least 100"
            )


@dataclass
class LearningMemoryConfig:
    """Configuration for the learning memory / correction system."""

    enabled: bool
    """Feature flag — disabled until tested (env: LEARNING_MEMORY_ENABLED)."""

    min_confidence: float
    """Minimum confidence to store a correction (env: CORRECTION_MIN_CONFIDENCE)."""

    auto_promote_confidence: float
    """Confidence threshold for auto-promotion (env: CORRECTION_AUTO_PROMOTE_CONFIDENCE)."""

    auto_promote_min_count: int
    """Min count before auto-promote (env: CORRECTION_AUTO_PROMOTE_MIN_COUNT)."""

    quality_gate_min_samples: int
    """Minimum sample size for daily batch promotion (env: CORRECTION_QUALITY_GATE_MIN_SAMPLES)."""

    dictionary_cache_ttl: int
    """Tenant dictionary in-memory cache TTL in seconds (env: TENANT_DICTIONARY_CACHE_TTL)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError(
                f"Invalid min_confidence: {self.min_confidence}. Must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.auto_promote_confidence <= 1.0:
            raise ConfigurationError(
                f"Invalid auto_promote_confidence: {self.auto_promote_confidence}. "
                "Must be between 0.0 and 1.0"
            )
        if self.auto_promote_min_count < 1:
            raise ConfigurationError(
                f"Invalid auto_promote_min_count: {self.auto_promote_min_count}. Must be at least 1"
            )
        if self.quality_gate_min_samples < 1:
            raise ConfigurationError(
                f"Invalid quality_gate_min_samples: {self.quality_gate_min_samples}. "
                "Must be at least 1"
            )
        if self.dictionary_cache_ttl < 0:
            raise ConfigurationError(
                f"Invalid dictionary_cache_ttl: {self.dictionary_cache_ttl}. Must be non-negative"
            )


AVAILABLE_PROMPT_VERSIONS = ["v3.1"]

# The canonical prompt version — what production runs and what every environment
# should resolve to by default. Kept as a single symbol so the dataclass default,
# the ``from_env`` env fallback, the deploy env, and the eval workflow never
# disagree: an environment silently resolving a different prompt version than
# the one the eval gate scored is a class of drift this symbol exists to
# prevent (older capacity_planner templates lacked the planning surface, so a
# silent version drift emitted ungrounded prose and paid a full retry pass —
# ~2x LLM calls/latency — with no error). Bump this when the canonical
# template moves.
CANONICAL_PROMPT_VERSION = "v3.1"
assert CANONICAL_PROMPT_VERSION in AVAILABLE_PROMPT_VERSIONS


@dataclass
class PromptCacheConfig:
    """Configuration for Gemini context caching of stable prompt prefixes.
    The cache holds tenant-agnostic content only; tenant data
    travels in the per-turn ``contents`` argument."""

    enabled: bool = True
    """Master toggle (env: ``TABI_PROMPT_CACHE_ENABLED``). When False,
    ``PromptCacheManager.get_or_create`` returns ``None`` and callers fall
    back to inline prompts (rollback lever)."""

    ttl_seconds: int = 1800
    """TTL for cached entries. 30 minutes — sized for analytics-exploration
    sessions where users drill into the same conversation context across many
    turns. Cache key includes ``agent`` + ``prompt_version`` (see ``_stable_prefix``)
    so a prompt-version bump correctly invalidates entries; pure-TTL extensions
    are safe. Minimum 60s (Gemini server-side floor); previous default was 300s
    (5 min) which left cache-hit opportunities on the table for the common
    multi-turn pattern. Env override: ``TABI_PROMPT_CACHE_TTL_SECONDS``
    (currently read only via ``PromptCacheConfig.__init__``)."""

    def __post_init__(self) -> None:
        if self.ttl_seconds < 60:
            raise ConfigurationError(
                f"Invalid prompt cache ttl_seconds: {self.ttl_seconds}. "
                "Must be at least 60 (Gemini server-side minimum)."
            )


@dataclass
class SelfCritiqueConfig:
    """Self-critique pass configuration.

    A second LLM call reviews high-stakes specialist responses against the
    underlying query results. Triggered selectively (multi-agent, data_scientist,
    low-confidence, data-insufficient, annotated) so cost stays bounded —
    one critique per qualifying turn at most.
    """

    enabled: bool = True
    """Master toggle (env: ``TABI_SELF_CRITIQUE_ENABLED``)."""

    model: str = field(default_factory=_resolve_flash_model)
    """Critic model. Resolved at config-load via ``TABI_FLASH_MODEL``, so the
    same default works whether or not the new Flash model has shipped."""

    thinking_budget: int = 2048
    """Tuned to keep critique latency inside the +1.5s P50 budget.
    Higher budgets give better catch rates but trade off against the SLO."""

    timeout_seconds: float = 30.0
    """Hard cap; the executor returns ``_SHIP_ON_FAILURE`` on timeout so a
    failure to critique never fails a turn that already has a valid response."""

    confidence_threshold: float = 0.7
    """Confidence below this triggers critique — but only when calibration
    has passed. See ``confidence_is_calibrated``."""

    confidence_is_calibrated: bool = field(default_factory=_calibration_passed)
    """Mirrors ``evaluation/baseline.yaml -> calibration.passed``.
    When False, the confidence-gated branch of ``should_critique`` is skipped
    entirely (other triggers still fire). Same flag drives the frontend badge.
    Read once per process via ``_calibration_passed`` (lru_cached). Can be
    forced in a single environment via ``TABI_SELF_CRITIQUE_CALIBRATED``
    without editing the committed baseline file."""

    def __post_init__(self) -> None:
        if self.model not in AVAILABLE_MODELS:
            raise ConfigurationError(
                f"Invalid self-critique model: {self.model!r}. Must be one of: "
                f"{', '.join(AVAILABLE_MODELS)}"
            )
        if self.thinking_budget < 0:
            raise ConfigurationError(
                f"Invalid thinking_budget: {self.thinking_budget}. Must be non-negative"
            )
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                f"Invalid timeout_seconds: {self.timeout_seconds}. Must be positive"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError(
                f"Invalid confidence_threshold: {self.confidence_threshold}. "
                "Must be between 0.0 and 1.0"
            )


@dataclass
class Config:
    """Main configuration container for TABI analytics."""

    models: ModelTieringConfig
    logging: LoggingConfig
    rag: RAGConfig
    query_batching: QueryBatchingConfig
    validation: ValidationConfig
    context_assembly: ContextAssemblyConfig
    learning_memory: LearningMemoryConfig
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    prompt_cache: PromptCacheConfig = field(default_factory=PromptCacheConfig)
    self_critique: SelfCritiqueConfig = field(default_factory=SelfCritiqueConfig)
    environment: str = "development"
    prompt_version: str = CANONICAL_PROMPT_VERSION
    fast_path_handoff_enabled: bool = True
    """Gate the fast-path dispatch loop. When False, handoffs on the fast
    path are dropped (logged as warning) and only the first specialist runs
    — matching pre-feature behavior. Env: ENABLE_FAST_PATH_HANDOFF."""
    handoff_synthesis: Literal["target", "llm"] = "target"
    """How to merge source + target specialist outputs into one voice.
    - ``target``: target specialist is the synthesizer (zero extra LLM calls).
    - ``llm``: dedicated synthesis call via the configured synthesis model.
    Env: HANDOFF_SYNTHESIS."""

    def __post_init__(self):
        """Validate top-level configuration."""
        if self.prompt_version not in AVAILABLE_PROMPT_VERSIONS:
            raise ConfigurationError(
                f"Invalid prompt version: {self.prompt_version}. "
                f"Must be one of: {', '.join(AVAILABLE_PROMPT_VERSIONS)}"
            )
        if self.handoff_synthesis not in ("target", "llm"):
            raise ConfigurationError(
                f"Invalid handoff_synthesis: {self.handoff_synthesis}. Must be 'target' or 'llm'"
            )

    # Legacy field aliases. New code should read ``config.models`` /
    # ``config.validation`` directly.
    @property
    def model(self) -> ModelTieringConfig:
        return self.models

    @property
    def response_validation(self) -> ValidationConfig:
        return self.validation

    @classmethod
    def from_env(cls, env_file: str | None = None) -> Config:
        """
        Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file. If None, looks for .env in current directory.

        Returns:
            Config instance with all settings loaded.

        Raises:
            ConfigurationError: If required configuration is missing or invalid.
        """
        # Load .env file if it exists
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Determine environment and prompt version
        environment = os.getenv("TABI_ENV", "development").lower()
        prompt_version = _normalise_prompt_version(
            os.getenv("PROMPT_VERSION", CANONICAL_PROMPT_VERSION)
        )

        # Load thinking budget configuration
        thinking_enabled = os.getenv("GEMINI_THINKING_ENABLED", "true").lower() == "true"

        # Per-agent tiering. ``TABI_MODEL_X`` is the canonical env name;
        # ``GEMINI_MODEL_X`` / ``GEMINI_THINKING_X`` are aliased for one release.
        agent_models: dict[str, AgentModelConfig] = {}
        for agent_name, base_cfg in DEFAULT_AGENT_MODELS.items():
            env_key = agent_name.upper()
            model_override = _getenv_with_alias(f"TABI_MODEL_{env_key}", f"GEMINI_MODEL_{env_key}")
            thinking_override = _getenv_with_alias(
                f"TABI_THINKING_{env_key}", f"GEMINI_THINKING_{env_key}"
            )
            temperature_override = os.getenv(f"TABI_TEMPERATURE_{env_key}")
            agent_models[agent_name] = AgentModelConfig(
                model=model_override or base_cfg.model,
                thinking_budget=(
                    int(thinking_override)
                    if thinking_override is not None
                    else base_cfg.thinking_budget
                ),
                temperature=(
                    float(temperature_override)
                    if temperature_override is not None
                    else base_cfg.temperature
                ),
                max_output_tokens=base_cfg.max_output_tokens,
            )

        # Named slots. Legacy ``GEMINI_*_MODEL`` env vars remain readable.
        flash = _resolve_flash_model()
        pro = "gemini-3.1-pro-preview"

        default_model_name = (
            _getenv_with_alias("TABI_MODEL_DEFAULT", "GEMINI_DEFAULT_MODEL") or flash
        )
        default_slot = AgentModelConfig(model=default_model_name, thinking_budget=0)

        synthesis_multi_model = (
            _getenv_with_alias("TABI_MODEL_SYNTHESIS", "GEMINI_SYNTHESIS_MODEL") or pro
        )
        synthesis: dict[str, AgentModelConfig] = {
            "single": AgentModelConfig(model=flash, thinking_budget=2048),
            "multi": AgentModelConfig(model=synthesis_multi_model, thinking_budget=4096),
        }

        classifier_model_name = (
            _getenv_with_alias("TABI_MODEL_CLASSIFIER", "GEMINI_CLASSIFIER_MODEL")
            or "gemini-3.1-flash-lite"
        )
        classifier_slot = AgentModelConfig(
            model=classifier_model_name,
            thinking_budget=0,
            temperature=0.0,
        )

        handoff_model_name = os.getenv("TABI_MODEL_HANDOFF_SYNTHESIS", pro)
        handoff_slot = AgentModelConfig(model=handoff_model_name, thinking_budget=2048)

        storytelling_model_name = (
            _getenv_with_alias("TABI_MODEL_STORYTELLING", "GEMINI_STORYTELLING_MODEL") or pro
        )
        storytelling_slot = AgentModelConfig(model=storytelling_model_name, thinking_budget=4096)

        secondary_weight_raw = os.getenv("TABI_SECONDARY_AGENT_WEIGHT")
        secondary_agent_weight = (
            float(secondary_weight_raw) if secondary_weight_raw is not None else 0.5
        )

        models_config = ModelTieringConfig(
            agents=agent_models,
            synthesis=synthesis,
            classifier=classifier_slot,
            handoff_synthesis=handoff_slot,
            storytelling=storytelling_slot,
            default=default_slot,
            router_threshold=float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.6")),
            classifier_cache_ttl=float(os.getenv("CLASSIFIER_CACHE_TTL", "600.0")),
            classifier_cache_max_size=int(os.getenv("CLASSIFIER_CACHE_MAX_SIZE", "100")),
            thinking_enabled=thinking_enabled,
            secondary_agent_weight=secondary_agent_weight,
        )

        # Validate the fully-resolved config (after every slot, incl.
        # storytelling, is built) so a future slot can't widen the retired-id
        # bypass undetected.
        _assert_no_retired_models(models_config)

        # Load logging configuration
        logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            subdir="analytics",
            project_root=os.getenv("TABI_PROJECT_ROOT"),
        )

        # Load RAG configuration
        rag = RAGConfig(
            enabled=os.getenv("RAG_ENABLED", "false").lower() == "true",
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
            min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.7")),
            default_top_k=int(os.getenv("RAG_DEFAULT_TOP_K", "3")),
            min_rating_to_index=int(os.getenv("RAG_MIN_RATING_TO_INDEX", "4")),
            min_response_length=int(os.getenv("RAG_MIN_RESPONSE_LENGTH", "200")),
        )

        # Load query batching configuration
        query_batching = QueryBatchingConfig(
            enabled=os.getenv("QUERY_BATCHING_ENABLED", "true").lower() == "true",
            batch_window_ms=float(os.getenv("QUERY_BATCHING_WINDOW_MS", "50.0")),
            max_batch_size=int(os.getenv("QUERY_BATCHING_MAX_SIZE", "10")),
        )

        # ``RESPONSE_VALIDATION_*`` env vars are aliased for one release.
        validation_enabled_raw = _getenv_with_alias(
            "TABI_VALIDATION_ENABLED", "RESPONSE_VALIDATION_ENABLED", "true"
        )
        soft_threshold_raw = _getenv_with_alias(
            "TABI_VALIDATION_SOFT_THRESHOLD",
            "VALIDATION_DEVIATION_THRESHOLD",
            "0.05",
        )
        hard_threshold_raw = os.getenv("TABI_VALIDATION_HARD_THRESHOLD", "0.20")
        feature_flags = FeatureFlags(
            structured_output_enabled=(
                os.getenv("TABI_STRUCTURED_OUTPUT_ENABLED", "false").lower() == "true"
            ),
            chart_by_reference_enabled=(
                os.getenv("TABI_CHART_BY_REFERENCE_ENABLED", "true").lower() == "true"
            ),
            two_pass_synthesis_enabled=(
                os.getenv("TABI_TWO_PASS_SYNTHESIS_ENABLED", "true").lower() == "true"
            ),
            bq_agent_analytics_enabled=(
                os.getenv("TABI_BQ_AGENT_ANALYTICS_ENABLED", "false").lower() == "true"
            ),
            tenant_scope_guardrail_enabled=(
                os.getenv("TABI_TENANT_SCOPE_GUARDRAIL_ENABLED", "true").lower() == "true"
            ),
        )

        validation_enabled = (validation_enabled_raw or "true").lower() == "true"
        max_regen_override = os.getenv("TABI_VALIDATION_MAX_REGEN")
        validation_config = ValidationConfig(
            enabled=validation_enabled,
            soft_threshold=float(soft_threshold_raw or "0.05"),
            hard_threshold=float(hard_threshold_raw),
            absolute_tolerance=float(os.getenv("VALIDATION_ABSOLUTE_TOLERANCE", "0.5")),
            log_level=os.getenv("VALIDATION_LOG_LEVEL", "warning"),
            max_regen=(
                int(max_regen_override)
                if max_regen_override is not None
                else (1 if validation_enabled and feature_flags.structured_output_enabled else 0)
            ),
        )

        # Load context assembly configuration
        context_assembly = ContextAssemblyConfig(
            enabled=os.getenv("CONTEXT_ASSEMBLY_ENABLED", "true").lower() == "true",
            token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "1200")),
            include_benchmarks=os.getenv("CONTEXT_INCLUDE_BENCHMARKS", "true").lower() == "true",
            include_exemplars=os.getenv("CONTEXT_INCLUDE_EXEMPLARS", "true").lower() == "true",
        )

        # Load learning memory configuration
        learning_memory = LearningMemoryConfig(
            enabled=os.getenv("LEARNING_MEMORY_ENABLED", "false").lower() == "true",
            min_confidence=float(os.getenv("CORRECTION_MIN_CONFIDENCE", "0.5")),
            auto_promote_confidence=float(os.getenv("CORRECTION_AUTO_PROMOTE_CONFIDENCE", "0.8")),
            auto_promote_min_count=int(os.getenv("CORRECTION_AUTO_PROMOTE_MIN_COUNT", "2")),
            quality_gate_min_samples=int(os.getenv("CORRECTION_QUALITY_GATE_MIN_SAMPLES", "3")),
            dictionary_cache_ttl=int(os.getenv("TENANT_DICTIONARY_CACHE_TTL", "300")),
        )

        prompt_cache = PromptCacheConfig(
            enabled=os.getenv("TABI_PROMPT_CACHE_ENABLED", "true").lower() == "true",
            # Default must stay in sync with ``PromptCacheConfig.ttl_seconds`` default.
            ttl_seconds=int(os.getenv("TABI_PROMPT_CACHE_TTL_SECONDS", "1800")),
        )

        # Self-critique. Calibration normally comes from
        # baseline.yaml; ``TABI_SELF_CRITIQUE_CALIBRATED`` lets ops force the
        # flag in a single environment without editing the committed file.
        calibrated_override = os.getenv("TABI_SELF_CRITIQUE_CALIBRATED")
        if calibrated_override is None:
            confidence_is_calibrated = _calibration_passed()
        else:
            confidence_is_calibrated = calibrated_override.lower() == "true"
        self_critique = SelfCritiqueConfig(
            enabled=os.getenv("TABI_SELF_CRITIQUE_ENABLED", "true").lower() == "true",
            model=_resolve_flash_model(),
            thinking_budget=int(os.getenv("TABI_SELF_CRITIQUE_THINKING_BUDGET", "2048")),
            timeout_seconds=float(os.getenv("TABI_SELF_CRITIQUE_TIMEOUT_SECONDS", "30.0")),
            confidence_threshold=float(os.getenv("TABI_SELF_CRITIQUE_CONFIDENCE_THRESHOLD", "0.7")),
            confidence_is_calibrated=confidence_is_calibrated,
        )

        fast_path_handoff_enabled = os.getenv("ENABLE_FAST_PATH_HANDOFF", "true").lower() == "true"
        handoff_synthesis_raw = os.getenv("HANDOFF_SYNTHESIS", "target").lower()
        handoff_synthesis: Literal["target", "llm"]
        if handoff_synthesis_raw == "llm":
            handoff_synthesis = "llm"
        elif handoff_synthesis_raw == "target":
            handoff_synthesis = "target"
        else:
            raise ConfigurationError(
                f"Invalid HANDOFF_SYNTHESIS={handoff_synthesis_raw!r}. Must be 'target' or 'llm'"
            )

        return cls(
            models=models_config,
            logging=logging,
            rag=rag,
            query_batching=query_batching,
            validation=validation_config,
            context_assembly=context_assembly,
            learning_memory=learning_memory,
            feature_flags=feature_flags,
            prompt_cache=prompt_cache,
            self_critique=self_critique,
            environment=environment,
            prompt_version=prompt_version,
            fast_path_handoff_enabled=fast_path_handoff_enabled,
            handoff_synthesis=handoff_synthesis,
        )


def build_thinking_planner(agent_name: str) -> BasePlanner | None:
    """Build an ADK BuiltInPlanner with the agent's thinking budget.

    Returns None if thinking is disabled for this agent (budget=0).
    ADK requires thinking config via the ``planner`` parameter, not
    ``generate_content_config``.
    """
    from google.adk.planners import BuiltInPlanner
    from google.genai import types

    budget = get_config().models.get_agent_thinking(agent_name)
    if budget == 0:
        return None
    return BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=budget),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get the application configuration (cached)."""
    cfg = Config.from_env()
    # Surface load-bearing resolved config at startup so an environment's actual
    # behavior is one grep away — not inferred from latency. The prompt_version
    # line exists specifically because a silent prompt-version drift can double
    # capacity-planner latency with no error.
    logger.info(
        "config_summary",
        environment=cfg.environment,
        prompt_version=cfg.prompt_version,
        prompt_version_is_canonical=cfg.prompt_version == CANONICAL_PROMPT_VERSION,
        handoff_synthesis=cfg.handoff_synthesis,
        fast_path_handoff_enabled=cfg.fast_path_handoff_enabled,
    )
    if cfg.prompt_version != CANONICAL_PROMPT_VERSION:
        # Loud, but non-fatal — a deliberate A/B on a non-canonical
        # PROMPT_VERSION is legitimate; we just refuse to let it be silent.
        logger.warning(
            "config.prompt_version_non_canonical",
            prompt_version=cfg.prompt_version,
            canonical=CANONICAL_PROMPT_VERSION,
            hint=(
                "Non-canonical capacity_planner prompts lack the planning surface "
                "and trigger the goal-attainment retry loop (~2x LLM calls). Set "
                "PROMPT_VERSION to the canonical value or bump CANONICAL_PROMPT_VERSION."
            ),
        )
    # Surface the resolved regen budget so the gated kill-switch is visible
    # without grepping baseline.yaml.
    logger.info(
        "validation.max_regen.resolved",
        max_regen=cfg.validation.max_regen,
        structured_output_enabled=cfg.feature_flags.structured_output_enabled,
    )
    if cfg.self_critique.enabled and not cfg.self_critique.confidence_is_calibrated:
        # Operators need to know which regime is in force without grepping
        # baseline.yaml (calibration-fallback regime).
        logger.info(
            "self_critique.calibration_fallback_active",
            confidence_threshold=cfg.self_critique.confidence_threshold,
        )
    return cfg


def clear_config_cache() -> None:
    """Clear the configuration cache. Useful for testing."""
    get_config.cache_clear()
    # ``_calibration_passed`` may be monkeypatched in tests — tolerate a
    # non-lru_cache replacement rather than fail with AttributeError.
    cache_clear = getattr(_calibration_passed, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


# ---------------------------------------------------------------------------
# Shared genai client factory
# ---------------------------------------------------------------------------

_genai_client: GenaiClient | None = None


# Backstop ordering: explicit genai-client transport timeout
# slotted between ``TURN_BUDGET_SECONDS`` (~90s) and Cloud Run's request
# timeout (``var.cloud_run_api_timeout``, default 120s). Without an
# explicit value the genai SDK falls through to an effectively-unbounded
# default for the httpx transport (``BaseApiClient._use_google_auth_async``
# sets ``max_allowed_time=float('inf')`` when ``http_options.timeout is
# None``), which means a hung Gemini call wouldn't surface a transport
# error until Cloud Run 503s the whole request — the watchdog's
# salvage-before-503 guarantee would have no transport-layer safety net
# underneath it. 105s default gives the watchdog (90s) ~15s to fire and
# salvage gracefully while still beating Cloud Run (120s) to the 503.
# Env override: ``TABI_GENAI_TRANSPORT_TIMEOUT_SECONDS``.
#
# Backstop-ordering invariant: ``TURN_BUDGET_SECONDS <
# T_GENAI_TRANSPORT_TIMEOUT_SECONDS < resolved var.cloud_run_api_timeout``.
# Pinned by the platform's turn-budget backstop test (not vendored here).
def resolve_timeout_env(env_var: str, default: float) -> float:
    """Read a float-seconds value from ``env_var`` with a default fallback,
    logging a structured warning on parse failure (shared
    helper). Used by all the timeout constants
    (``TURN_BUDGET_SECONDS``, ``BRANCH_BUDGET_SECONDS``,
    ``T_GENAI_TRANSPORT_TIMEOUT_SECONDS``) so error handling stays
    consistent and a bad env-var bump fails the same way everywhere."""
    raw = os.environ.get(env_var)
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("invalid_timeout_env", env_var=env_var, raw=raw)
    return default


T_GENAI_TRANSPORT_TIMEOUT_SECONDS = resolve_timeout_env(
    "TABI_GENAI_TRANSPORT_TIMEOUT_SECONDS", 105.0
)


def get_genai_client() -> GenaiClient:
    """Return a shared genai.Client singleton for Vertex AI.

    Lazily initialized on first call. All analytics and API services
    that need a Gemini client should use this instead of creating their own.

    Configures explicit ``http_options.timeout`` (backstop
    ordering) so a hung Gemini call surfaces as a transport error before
    Cloud Run 503s the request — preserves the watchdog's
    salvage-before-503 guarantee on the transport layer.
    """
    global _genai_client
    if _genai_client is None:
        from google import genai
        from google.genai.types import HttpOptions

        # NB: ``HttpOptions.timeout`` is in MILLISECONDS (google-genai 1.55+
        # field metadata). Convert from seconds at the call site so the
        # ``T_GENAI_TRANSPORT_TIMEOUT_SECONDS`` constant stays in the same
        # unit as ``TURN_BUDGET_SECONDS`` / ``BRANCH_BUDGET_SECONDS`` /
        # ``var.cloud_run_api_timeout`` for the backstop-ordering test.
        _genai_client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            http_options=HttpOptions(timeout=int(T_GENAI_TRANSPORT_TIMEOUT_SECONDS * 1000)),
        )
    return _genai_client


def reset_genai_client() -> None:
    """Reset the genai client singleton. Useful for testing."""
    global _genai_client
    _genai_client = None
