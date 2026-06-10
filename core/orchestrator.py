"""
Orchestrator for multi-agent query execution.

Coordinates multiple specialist agents for complex queries that require
expertise from more than one domain. Handles:
- Sequential specialist execution with state sharing
- Automatic handoff processing
- Response synthesis across specialists
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import get_config, get_genai_client
from core.handoff import (
    MAX_HANDOFF_DEPTH,
    HandoffRequest,
    collect_tool_response,
    detect_handoff,
    format_tool_outputs,
)
from core.logging_config import get_metric_trace_logger, log_timing
from core.salvage_payload import (
    bind_salvage_payload_scope,
    get_pending_salvage_payload,
)
from core.salvage_plugin import (
    CAUSE_WATCHDOG,
    SALVAGE_TEXT_BRANCH_TIMEOUT,
    record_salvage_signal,
)
from core.specialist_schema import (
    SpecialistResponse,
    build_fallback_response,
    build_generate_content_config,
    parse_specialist_response,
)
from core.turn_error_sink import TurnErrorSink, bind_turn_error_sink
from core.usage_tracker import GeminiUsageTracker

# ---------------------------------------------------------------------------
# [public-repo stub] proprietary core.specialist_runner and core.synthesizer
# are excluded from this showcase. The orchestrator below demonstrates the
# multi-agent coordination logic (routing, fan-out to specialists, synthesis
# dispatch, handoff handling). The single-pass specialist runner and the
# cross-specialist synthesizer live in the private package. These minimal
# stubs preserve the names + signatures the orchestrator references so the
# file py_compiles and reads cleanly, without shipping proprietary internals.
# ---------------------------------------------------------------------------

# core.specialist_runner stubs ---------------------------------------------
APP_NAME = "tabi_analytics"  # shared App name (runner + session service)
BRANCH_BUDGET_SECONDS = 45.0  # per-specialist branch wall-clock budget


def _extract_salvaged_chart_refs(_tool_responses: list[Any]) -> list[str]:
    """Stub: real impl pulls chart refs out of salvaged tool outputs."""
    return []


# core.synthesizer stub -----------------------------------------------------
class Synthesizer:
    """Stub for the cross-specialist response synthesizer.

    The real Synthesizer composes a single grounded answer from multiple
    specialist responses (multi-agent and streaming variants).
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

    async def synthesize_multi_agent(self, *_args: Any, **_kwargs: Any) -> str:
        raise NotImplementedError(
            "Synthesizer is proprietary and excluded from the public showcase."
        )

    async def synthesize_multi_agent_streaming(self, *_args: Any, **_kwargs: Any):
        raise NotImplementedError(
            "Synthesizer is proprietary and excluded from the public showcase."
        )
        yield  # pragma: no cover — marks this an async generator


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)
metric_trace_logger = get_metric_trace_logger()


# Config helpers (avoid hard failures on bad env values).
def _get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%s, defaulting to %s", name, raw_value, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%s, defaulting to %s", name, raw_value, default)
        return default


# Timeout for Gemini API calls (in seconds)
# Prevents application freeze if API hangs
GEMINI_API_TIMEOUT_SECONDS = 60.0

# When specialist responses are highly similar, skip synthesis for speed.
SYNTHESIS_OVERLAP_THRESHOLD = _get_env_float("SYNTHESIS_OVERLAP_THRESHOLD", 0.78)
SYNTHESIS_OVERLAP_MIN_CHARS = _get_env_int("SYNTHESIS_OVERLAP_MIN_CHARS", 120)

# Prefer these agents as data providers when sharing tool outputs.
DATA_PROVIDER_PREFERENCE = (
    "data_scientist",
    "general_analyst",
    "capacity_planner",
    "pipeline_analyst",
    "sourcing_strategist",
    "offer_advisor",
    "interviewing_coach",
)

# Normalize chart references to avoid false negatives in overlap checks.
_CHART_REF_PATTERN = re.compile(r"\[chart:[^\]]+\]", re.IGNORECASE)
_CHART_BLOCK_PATTERN = re.compile(r"```chart[\s\S]*?```", re.IGNORECASE)

# Key used to store handoff requests in session state
HANDOFF_REQUEST_KEY = "pending_handoff_request"


@dataclass
class SpecialistResult:
    """Result from a specialist agent execution.

    ``response`` carries the user-visible text. When structured output is
    enabled, ``structured`` additionally carries the parsed
    :class:`SpecialistResponse` so the engine and synthesiser can read
    typed claims, charts, and confidence values from the same shape on
    every execution path.
    """

    agent_name: str
    """Name of the specialist that produced this result."""

    response: str
    """The specialist's response text (``answer_markdown`` when structured)."""

    handoff_request: HandoffRequest | None = None
    """If set, the specialist requested a handoff."""

    tokens_used: int = 0
    """Tokens consumed by this specialist."""

    tool_responses: list[dict[str, Any]] = field(default_factory=list)
    """Tool outputs captured during this specialist run."""

    structured: SpecialistResponse | None = None
    """Parsed structured output. ``None`` when the structured-output flag
    is off, or when parsing failed and the free-form fallback fires."""


@dataclass
class OrchestrationResult:
    """Final result from multi-agent orchestration."""

    response: str
    """The synthesized or single-agent response."""

    specialists_invoked: list[str]
    """List of specialist names that contributed."""

    was_synthesized: bool
    """True if multiple specialists contributed and synthesis occurred."""

    specialist_results: list[SpecialistResult] = field(default_factory=list)
    """Individual results from each specialist."""


class MultiAgentOrchestrator:
    """
    Orchestrates multi-specialist query execution.

    Manages sequential execution of specialists, processes handoff requests,
    and synthesizes responses when multiple specialists contribute.

    Usage:
        orchestrator = MultiAgentOrchestrator(
            specialists={"pipeline_analyst": agent1, "capacity_planner": agent2},
            session_service=session_service,
            customer_id="customer_abc",
        )
        result = await orchestrator.execute(
            question="What's our bottleneck and how does it affect forecast?",
            initial_agents=["pipeline_analyst", "capacity_planner"],
        )
    """

    def __init__(
        self,
        specialists: dict[str, Agent],
        session_service: InMemorySessionService,
        session_id: str,
        customer_id: str,
        model: str | None = None,
        synthesis_model: str | None = None,
        app_name: str = APP_NAME,
        synthesizer: Synthesizer | None = None,
        usage_tracker: GeminiUsageTracker | None = None,
        get_specialist_app: Callable[..., App] | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            specialists: Map of agent names to Agent instances
            session_service: ADK session service for state management
            session_id: Current ADK session ID
            customer_id: Tenant identifier
            model: Model to use for specialists (None = use config default)
            synthesis_model: Model to use for synthesis (None = use config synthesis_model)
                            Using a faster model here can significantly reduce latency.
            app_name: ADK app name
            synthesizer: Optional shared :class:`Synthesizer`. When the
                orchestrator is built by :class:`AgentSession` the session
                owns one instance and threads it in so fast-path handoff
                synthesis and multi-agent synthesis roll up under the same
                usage counter. Standalone constructions (tests, scripts)
                pass ``None`` and the orchestrator builds a private one.
            usage_tracker: Optional shared :class:`GeminiUsageTracker`
                paired with ``synthesizer``. When ``synthesizer`` is
                provided, ``usage_tracker`` should be the tracker that
                synthesizer is bound to so external callers can read a
                single roll-up. Built fresh when omitted.
            get_specialist_app: The session's bound
                ``RunnerHost.get_specialist_app``, so the coordinator→
                specialist path resolves the same cached ``App`` (and thus
                plugin set + trace root) the router path uses. ``AgentSession``
                always threads it; ``None`` only for standalone/test
                constructions, which fall back to a self-contained App cache.
        """
        self.specialists = specialists
        self.session_service = session_service
        self.session_id = session_id
        self.customer_id = customer_id
        self.model = model if model is not None else get_config().model.default_model
        # Use a separate (typically faster) model for synthesis
        self.synthesis_model = (
            synthesis_model if synthesis_model is not None else get_config().model.synthesis_model
        )
        self.app_name = app_name
        # App(name=...) must equal create_session(app_name=...) or ADK raises
        # "Session not found"; both sides use the single APP_NAME constant.
        if self.app_name != APP_NAME:
            raise ValueError(
                f"MultiAgentOrchestrator.app_name={self.app_name!r} must be "
                f"{APP_NAME!r} — the single APP_NAME the App accessor and "
                f"create_session share."
            )
        self._get_specialist_app = get_specialist_app
        # Standalone/test fallback — only used when no accessor was threaded.
        self._standalone_apps: dict[str, App] = {}
        self._standalone_plugins: list[Any] | None = None

        # Track which agents have been invoked (prevents duplicates)
        self._invoked_agents: set[str] = set()
        # Set to True if any specialist this run fell into the salvage
        # path (timeout / empty response / cap). ``run_orchestrated_streaming``
        # reads this after the stream drains and propagates to
        # ``turn.agent_error`` so eval / chat_service see the signal.
        self._last_agent_error: bool = False
        self._handoff_count = 0
        self._usage_tracker = usage_tracker if usage_tracker is not None else GeminiUsageTracker()
        if synthesizer is not None:
            self._synthesizer = synthesizer
        else:
            self._synthesizer = Synthesizer(
                get_config().models,
                usage_tracker=self._usage_tracker,
            )

    async def execute(
        self,
        question: str,
        initial_agents: list[str],
    ) -> OrchestrationResult:
        """
        Execute multi-agent orchestration.

        Runs initial agents in parallel for better performance, then
        processes any handoff requests sequentially.

        Args:
            question: User's question
            initial_agents: List of agent names to invoke (in order)

        Returns:
            OrchestrationResult with synthesized response and metadata
        """
        start_time = time.perf_counter()

        self._invoked_agents.clear()
        self._last_agent_error = False
        self._handoff_count = 0

        specialist_results: list[SpecialistResult] = []

        # Filter to valid, unique agents
        valid_initial_agents = []
        for agent_name in initial_agents:
            if agent_name in self._invoked_agents:
                logger.debug("Skipping duplicate agent: %s", agent_name)
                continue
            if agent_name not in self.specialists:
                logger.warning("Unknown specialist requested: %s", agent_name)
                continue
            valid_initial_agents.append(agent_name)
            self._invoked_agents.add(agent_name)

        # Run initial agents (optionally with a data provider first)
        if valid_initial_agents:
            # Build context once for all initial agents (no previous results)
            initial_context = self._build_context(question, [])
            shared_tool_outputs: list[dict[str, Any]] = []
            remaining_agents: list[str] = []
            data_provider = self._select_data_provider(valid_initial_agents)
            use_data_provider = data_provider is not None and len(valid_initial_agents) > 1

            handoff_queue: list[str] = []

            if use_data_provider:
                logger.info("Using data provider %s for shared tool outputs", data_provider)
                provider_result = await self._run_specialist(data_provider, initial_context)
                specialist_results.append(provider_result)
                self._merge_tool_outputs(shared_tool_outputs, provider_result.tool_responses)

                if provider_result.handoff_request and self._handoff_count < MAX_HANDOFF_DEPTH:
                    target = provider_result.handoff_request.target_agent
                    if target not in self._invoked_agents and target not in handoff_queue:
                        logger.info(
                            "Processing handoff: %s -> %s (reason: %s)",
                            provider_result.agent_name,
                            target,
                            provider_result.handoff_request.reason,
                        )
                        handoff_queue.append(target)
                        self._handoff_count += 1

                remaining_agents = [
                    agent_name for agent_name in valid_initial_agents if agent_name != data_provider
                ]

                if remaining_agents:
                    shared_context = self._build_context_with_tool_outputs(
                        question,
                        specialist_results,
                        shared_tool_outputs,
                        avoid_tools=True,
                    )
                    parallel_start = time.perf_counter()
                    logger.info(
                        "Starting parallel specialist execution (shared data): %s",
                        remaining_agents,
                    )
                    initial_results = await asyncio.gather(
                        *[
                            self._run_specialist(agent_name, shared_context)
                            for agent_name in remaining_agents
                        ],
                        return_exceptions=True,
                    )
                    parallel_duration_ms = int((time.perf_counter() - parallel_start) * 1000)
                    logger.info(
                        "Parallel specialist execution completed in %dms (specialists: %s)",
                        parallel_duration_ms,
                        remaining_agents,
                    )
                else:
                    initial_results = []
            else:
                # Execute all initial agents concurrently
                parallel_start = time.perf_counter()
                logger.info("Starting parallel specialist execution: %s", valid_initial_agents)

                initial_results = await asyncio.gather(
                    *[
                        self._run_specialist(agent_name, initial_context)
                        for agent_name in valid_initial_agents
                    ],
                    return_exceptions=True,
                )

                parallel_duration_ms = int((time.perf_counter() - parallel_start) * 1000)
                logger.info(
                    "Parallel specialist execution completed in %dms (specialists: %s)",
                    parallel_duration_ms,
                    valid_initial_agents,
                )

            # Process results and collect handoff requests
            for i, result in enumerate(initial_results):
                if isinstance(result, Exception):
                    agent_name = (
                        remaining_agents[i] if use_data_provider else valid_initial_agents[i]
                    )
                    logger.exception("Error running specialist %s", agent_name)
                    specialist_results.append(
                        SpecialistResult(
                            agent_name=agent_name,
                            response=f"Error from {agent_name}: {result}",
                        )
                    )
                else:
                    specialist_results.append(result)
                    self._merge_tool_outputs(shared_tool_outputs, result.tool_responses)
                    # Queue handoff requests for sequential processing
                    if result.handoff_request and self._handoff_count < MAX_HANDOFF_DEPTH:
                        target = result.handoff_request.target_agent
                        if target not in self._invoked_agents and target not in handoff_queue:
                            logger.info(
                                "Processing handoff: %s -> %s (reason: %s)",
                                result.agent_name,
                                target,
                                result.handoff_request.reason,
                            )
                            handoff_queue.append(target)
                            self._handoff_count += 1

            # Process handoff requests SEQUENTIALLY (they need previous context)
            for agent_name in handoff_queue:
                if agent_name in self._invoked_agents:
                    continue
                if agent_name not in self.specialists:
                    logger.warning("Unknown handoff target: %s", agent_name)
                    continue

                self._invoked_agents.add(agent_name)

                # Build context with all previous results and shared tool outputs
                context = self._build_context_with_tool_outputs(
                    question,
                    specialist_results,
                    shared_tool_outputs,
                    avoid_tools=True,
                )
                result = await self._run_specialist(agent_name, context)
                specialist_results.append(result)
                self._merge_tool_outputs(shared_tool_outputs, result.tool_responses)

                # Handle nested handoffs (rare)
                if result.handoff_request and self._handoff_count < MAX_HANDOFF_DEPTH:
                    target = result.handoff_request.target_agent
                    if target not in self._invoked_agents:
                        handoff_queue.append(target)
                        self._handoff_count += 1

        # Synthesize if multiple specialists contributed
        if len(specialist_results) > 1:
            overlap, min_ratio = self._responses_overlap(specialist_results)
            if overlap:
                best = self._select_best_response(specialist_results)
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                log_timing(
                    logger,
                    "multi_agent_orchestration",
                    duration_ms,
                    specialists_count=len(specialist_results),
                    specialists=list(self._invoked_agents),
                    was_synthesized=False,
                    handoff_count=self._handoff_count,
                    skipped_synthesis=True,
                    overlap_ratio=min_ratio,
                )
                return OrchestrationResult(
                    response=best.response,
                    specialists_invoked=list(self._invoked_agents),
                    was_synthesized=False,
                    specialist_results=specialist_results,
                )

            synthesized = await self._synthesizer.synthesize_multi_agent(
                question, specialist_results
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration",
                duration_ms,
                specialists_count=len(specialist_results),
                specialists=list(self._invoked_agents),
                was_synthesized=True,
                handoff_count=self._handoff_count,
            )
            return OrchestrationResult(
                response=synthesized,
                specialists_invoked=list(self._invoked_agents),
                was_synthesized=True,
                specialist_results=specialist_results,
            )
        elif specialist_results:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration",
                duration_ms,
                specialists_count=1,
                specialists=list(self._invoked_agents),
                was_synthesized=False,
            )
            return OrchestrationResult(
                response=specialist_results[0].response,
                specialists_invoked=list(self._invoked_agents),
                was_synthesized=False,
                specialist_results=specialist_results,
            )
        else:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration",
                duration_ms,
                specialists_count=0,
                status="no_specialists",
            )
            return OrchestrationResult(
                response="No specialists were able to process this query.",
                specialists_invoked=[],
                was_synthesized=False,
                specialist_results=[],
            )

    async def execute_streaming(
        self,
        question: str,
        initial_agents: list[str],
        early_streaming: bool = True,
    ) -> AsyncIterator[str]:
        """
        Execute multi-agent orchestration with streaming.

        For multi-agent queries, streams responses as they become available.
        With early_streaming=True (default), yields the first specialist's
        response immediately for faster time-to-first-token, then optionally
        streams a synthesis refinement if other specialists add value.

        Args:
            question: User's question
            initial_agents: List of agent names to invoke
            early_streaming: If True, yield first specialist response immediately
                           instead of waiting for all specialists + synthesis.
                           Dramatically improves perceived latency.

        Yields:
            Text chunks of the final response
        """
        start_time = time.perf_counter()

        self._invoked_agents.clear()
        self._last_agent_error = False
        self._handoff_count = 0

        specialist_results: list[SpecialistResult] = []

        # Filter to valid, unique agents
        valid_initial_agents = []
        for agent_name in initial_agents:
            if agent_name in self._invoked_agents:
                continue
            if agent_name not in self.specialists:
                continue
            valid_initial_agents.append(agent_name)
            self._invoked_agents.add(agent_name)

        # Single agent case - just run it
        if len(valid_initial_agents) == 1:
            result = await self._run_specialist(
                valid_initial_agents[0],
                self._build_context(question, []),
            )
            yield result.response
            return

        # No agents case
        if not valid_initial_agents:
            yield "No specialists were able to process this query."
            return

        # Multi-agent case with early streaming
        if early_streaming and len(valid_initial_agents) > 1:
            async for chunk in self._execute_streaming_early(
                question, valid_initial_agents, start_time
            ):
                yield chunk
            return

        # Fall back to original behavior (wait for all, then synthesize)
        initial_context = self._build_context(question, [])
        shared_tool_outputs: list[dict[str, Any]] = []
        remaining_agents: list[str] = []
        data_provider = self._select_data_provider(valid_initial_agents)
        use_data_provider = data_provider is not None and len(valid_initial_agents) > 1

        handoff_queue: list[str] = []

        if use_data_provider:
            logger.info("Using data provider %s for shared tool outputs", data_provider)
            provider_result = await self._run_specialist(data_provider, initial_context)
            specialist_results.append(provider_result)
            self._merge_tool_outputs(shared_tool_outputs, provider_result.tool_responses)

            if provider_result.handoff_request and self._handoff_count < MAX_HANDOFF_DEPTH:
                target = provider_result.handoff_request.target_agent
                if target not in self._invoked_agents and target not in handoff_queue:
                    handoff_queue.append(target)
                    self._handoff_count += 1

            remaining_agents = [
                agent_name for agent_name in valid_initial_agents if agent_name != data_provider
            ]

            if remaining_agents:
                shared_context = self._build_context_with_tool_outputs(
                    question,
                    specialist_results,
                    shared_tool_outputs,
                    avoid_tools=True,
                )

                parallel_start = time.perf_counter()
                logger.info(
                    "Starting parallel specialist execution (shared data): %s",
                    remaining_agents,
                )

                initial_results = await asyncio.gather(
                    *[
                        self._run_specialist(agent_name, shared_context)
                        for agent_name in remaining_agents
                    ],
                    return_exceptions=True,
                )

                parallel_duration_ms = int((time.perf_counter() - parallel_start) * 1000)
                logger.info(
                    "Parallel specialist execution completed in %dms (specialists: %s)",
                    parallel_duration_ms,
                    remaining_agents,
                )
            else:
                initial_results = []
        else:
            parallel_start = time.perf_counter()
            logger.info("Starting parallel specialist execution: %s", valid_initial_agents)

            initial_results = await asyncio.gather(
                *[
                    self._run_specialist(agent_name, initial_context)
                    for agent_name in valid_initial_agents
                ],
                return_exceptions=True,
            )

            parallel_duration_ms = int((time.perf_counter() - parallel_start) * 1000)
            logger.info(
                "Parallel specialist execution completed in %dms (specialists: %s)",
                parallel_duration_ms,
                valid_initial_agents,
            )

        for i, result in enumerate(initial_results):
            if isinstance(result, Exception):
                agent_name = remaining_agents[i] if use_data_provider else valid_initial_agents[i]
                specialist_results.append(
                    SpecialistResult(
                        agent_name=agent_name,
                        response=f"Error from {agent_name}: {result}",
                    )
                )
            else:
                specialist_results.append(result)
                self._merge_tool_outputs(shared_tool_outputs, result.tool_responses)
                if result.handoff_request and self._handoff_count < MAX_HANDOFF_DEPTH:
                    target = result.handoff_request.target_agent
                    if target not in self._invoked_agents and target not in handoff_queue:
                        handoff_queue.append(target)
                        self._handoff_count += 1

        # Process handoffs sequentially
        for agent_name in handoff_queue:
            if agent_name in self._invoked_agents or agent_name not in self.specialists:
                continue
            self._invoked_agents.add(agent_name)
            context = self._build_context_with_tool_outputs(
                question,
                specialist_results,
                shared_tool_outputs,
                avoid_tools=True,
            )
            result = await self._run_specialist(agent_name, context)
            specialist_results.append(result)
            self._merge_tool_outputs(shared_tool_outputs, result.tool_responses)

        # Stream synthesis if multiple specialists contributed
        if len(specialist_results) > 1:
            overlap, min_ratio = self._responses_overlap(specialist_results)
            if overlap:
                best = self._select_best_response(specialist_results)
                yield best.response
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                log_timing(
                    logger,
                    "multi_agent_orchestration_streaming",
                    duration_ms,
                    specialists_count=len(specialist_results),
                    was_synthesized=False,
                    skipped_synthesis=True,
                    overlap_ratio=min_ratio,
                )
            else:
                async for chunk in self._synthesizer.synthesize_multi_agent_streaming(
                    question, specialist_results
                ):
                    yield chunk
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                log_timing(
                    logger,
                    "multi_agent_orchestration_streaming",
                    duration_ms,
                    specialists_count=len(specialist_results),
                    was_synthesized=True,
                )
        elif specialist_results:
            yield specialist_results[0].response
        else:
            yield "No specialists were able to process this query."

    async def _execute_streaming_early(
        self,
        question: str,
        agents: list[str],
        start_time: float,
    ) -> AsyncIterator[str]:
        """
        Execute specialists with early streaming and data provider pattern.

        Combines two optimizations:
        1. Data provider pattern: First agent fetches data, others reuse it
        2. Early streaming: Yield first response immediately for fast TTFT

        Flow:
        1. Select data provider from requested agents
        2. Run data provider first, capture tool outputs
        3. Stream data provider's response immediately (TTFT win)
        4. Run remaining agents with shared tool outputs (avoids duplicate calls)
        5. Stream synthesis addon if remaining agents have unique insights

        Args:
            question: User's question
            agents: List of agent names to invoke
            start_time: Start time for timing metrics

        Yields:
            Text chunks as they become available
        """
        initial_context = self._build_context(question, [])
        shared_tool_outputs: list[dict[str, Any]] = []

        # Select data provider - this agent runs first and shares its tool outputs
        data_provider = self._select_data_provider(agents)
        remaining_agents = [a for a in agents if a != data_provider]

        logger.info(
            "Early streaming with data provider: %s (remaining: %s)",
            data_provider,
            remaining_agents,
        )

        # Phase 1: Run data provider first and stream its response immediately.
        # ``remaining_agents_pending=True`` lets the salvage path know an addon
        # synthesis will carry the analysis — so it can emit chart refs without
        # the "I couldn't finalize..." apology that would otherwise mislead the
        # reader once a complete Additional insights section follows below.
        provider_result = await self._run_specialist(
            data_provider,
            initial_context,
            remaining_agents_pending=bool(remaining_agents),
        )
        self._merge_tool_outputs(shared_tool_outputs, provider_result.tool_responses)

        ttft_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "Early streaming: yielding data provider response from %s (TTFT: %dms, tools: %d)",
            data_provider,
            ttft_ms,
            len(provider_result.tool_responses),
        )

        # Stream data provider's response immediately
        yield provider_result.response

        # If no remaining agents, we're done
        if not remaining_agents:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration_streaming",
                duration_ms,
                specialists_count=1,
                was_synthesized=False,
                early_streaming=True,
                data_provider=data_provider,
            )
            return

        # Phase 2: Run remaining agents with shared tool outputs
        shared_context = self._build_context_with_tool_outputs(
            question,
            [provider_result],
            shared_tool_outputs,
            avoid_tools=True,
        )

        parallel_start = time.perf_counter()
        logger.info(
            "Early streaming: running remaining agents with shared data: %s",
            remaining_agents,
        )

        # Run remaining agents in parallel with shared context
        tasks = [
            asyncio.create_task(
                self._run_specialist(agent_name, shared_context),
                name=agent_name,
            )
            for agent_name in remaining_agents
        ]

        remaining_results: list[SpecialistResult] = []
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                remaining_results.append(result)
                self._merge_tool_outputs(shared_tool_outputs, result.tool_responses)
            except Exception as e:
                agent_name = "unknown"
                for task in tasks:
                    if task.done() and task.exception() is e:
                        agent_name = task.get_name()
                        break
                logger.exception("Error running specialist %s", agent_name)
                remaining_results.append(
                    SpecialistResult(
                        agent_name=agent_name,
                        response=f"Error from {agent_name}: {e}",
                    )
                )

        parallel_duration_ms = int((time.perf_counter() - parallel_start) * 1000)
        logger.info(
            "Early streaming: remaining agents completed in %dms",
            parallel_duration_ms,
        )

        # Phase 3: Check if remaining agents have unique insights to add
        all_results = [provider_result] + remaining_results

        # When the data provider salvaged to an empty response (no chart refs +
        # remaining_agents_pending=True branch in the multi-agent salvage), the
        # addon synthesizer would otherwise produce meta-commentary like
        # "X adds a layer of..." about a primary that doesn't exist. Promote
        # the best clean secondary as the primary answer instead so the user
        # sees the actual analysis, not commentary about it.
        if not provider_result.response.strip():
            clean_secondaries = [
                r
                for r in remaining_results
                if r.structured is not None
                and not r.structured.agent_error
                and r.structured.answer_markdown.strip()
            ]
            if clean_secondaries:
                best = max(
                    clean_secondaries,
                    key=lambda r: self._score_response(r.response),
                )
                logger.info(
                    "Early streaming: promoting %s as primary (data provider salvaged empty)",
                    best.agent_name,
                )
                yield best.response
                # Secondary saved the turn — clear the sticky error so the
                # "Try again" banner doesn't render. Mirrors the same clearing
                # done after addon synthesis in the non-promoted path below.
                self._last_agent_error = False
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                log_timing(
                    logger,
                    "multi_agent_orchestration_streaming",
                    duration_ms,
                    specialists_count=len(all_results),
                    was_synthesized=False,
                    promoted_secondary=best.agent_name,
                    early_streaming=True,
                    data_provider=data_provider,
                    shared_tools_count=len(shared_tool_outputs),
                )
                return
            # Empty primary AND no clean secondary — yield the generic apology
            # so the user gets feedback. Banner stays (sticky agent_error from
            # the salvage). This is the all-fail case.
            logger.info(
                "Early streaming: primary salvaged empty and no clean secondary — yielding apology"
            )
            yield (
                "I had trouble finalizing this response. "
                "Please try asking again, or rephrase your question."
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration_streaming",
                duration_ms,
                specialists_count=len(all_results),
                was_synthesized=False,
                all_failed=True,
                early_streaming=True,
                data_provider=data_provider,
                shared_tools_count=len(shared_tool_outputs),
            )
            return

        overlap, min_ratio = self._responses_overlap(all_results)

        if overlap:
            # Responses are similar enough - no synthesis needed
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration_streaming",
                duration_ms,
                specialists_count=len(all_results),
                was_synthesized=False,
                skipped_synthesis=True,
                overlap_ratio=min_ratio,
                early_streaming=True,
                data_provider=data_provider,
                shared_tools_count=len(shared_tool_outputs),
            )
            logger.info(
                "Early streaming: skipping synthesis (overlap=%.2f)",
                min_ratio,
            )
        else:
            # Other specialists have unique insights - stream a synthesis addon
            logger.info(
                "Early streaming: adding synthesis from %d additional specialists",
                len(remaining_results),
            )
            yield "\n\n---\n\n**Additional insights:**\n\n"

            async for chunk in self._synthesize_addon_streaming(
                question,
                provider_result,
                remaining_results,
            ):
                yield chunk

            # When the data provider salvaged but a later specialist produced a
            # complete answer, the addon synthesis carries the real response and
            # the "Try again" banner is misleading. Clear the sticky flag so
            # ``AgentSession.get_last_agent_error()`` reports clean for this turn.
            # Keep the banner for the all-failed case (no clean specialist below).
            if self._last_agent_error and self._has_clean_addon_specialist(remaining_results):
                logger.info(
                    "Early streaming: clearing agent_error — addon specialist produced clean answer",
                )
                self._last_agent_error = False

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "multi_agent_orchestration_streaming",
                duration_ms,
                specialists_count=len(all_results),
                was_synthesized=True,
                early_streaming=True,
                synthesis_type="addon",
                data_provider=data_provider,
                shared_tools_count=len(shared_tool_outputs),
            )

    async def _synthesize_addon_streaming(
        self,
        original_question: str,
        first_result: SpecialistResult,
        additional_results: list[SpecialistResult],
    ) -> AsyncIterator[str]:
        """
        Stream a synthesis addon that integrates insights from additional specialists.

        This is used in early streaming mode when the first specialist's response
        has already been yielded, but other specialists have unique insights to add.

        Args:
            original_question: User's original question
            first_result: The first specialist's result (already yielded)
            additional_results: Results from other specialists to integrate

        Yields:
            Text chunks of the synthesis addon
        """
        # Build a focused synthesis prompt for the addon
        prompt_parts = [
            "You are adding complementary insights to an existing analysis.\n\n",
            f"Original question: {original_question}\n\n",
            f"The {first_result.agent_name} already provided this analysis:\n",
            f"---\n{first_result.response}\n---\n\n",
            "Additional specialist perspectives:\n",
        ]

        for result in additional_results:
            prompt_parts.append(f"\n--- {result.agent_name} ---\n")
            prompt_parts.append(result.response)
            prompt_parts.append("\n")

        prompt_parts.append(
            "\n\nProvide a brief synthesis of the UNIQUE insights from the additional "
            "specialists that weren't already covered. Focus on:\n"
            "1. New information or perspectives not in the first analysis\n"
            "2. How these insights complement or extend the initial response\n"
            "3. Any additional recommendations\n\n"
            "Be concise - avoid repeating what was already said. "
            "If there's nothing substantially new, just say so briefly.\n"
            "Preserve any chart references exactly as they appear (e.g., [chart:chart_abc123])."
        )

        synthesis_prompt = "".join(prompt_parts)
        client = get_genai_client()

        try:
            response_stream = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=self.synthesis_model,
                    contents=synthesis_prompt,
                    config=build_generate_content_config(),
                ),
                timeout=GEMINI_API_TIMEOUT_SECONDS,
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except TimeoutError:
            logger.error("Synthesis addon timed out after %ss", GEMINI_API_TIMEOUT_SECONDS)
            # Fall back to listing the other responses
            for result in additional_results:
                yield f"\n**{result.agent_name}**: {result.response[:200]}..."

        except Exception as e:
            logger.exception("Error during synthesis addon")
            yield f"\n(Error synthesizing additional insights: {e})"

    def _build_context(
        self,
        original_question: str,
        previous_results: list[SpecialistResult],
    ) -> str:
        """Build context prompt including previous specialist findings."""
        if not previous_results:
            return original_question

        context_parts = [f"Original question: {original_question}\n"]
        context_parts.append("Previous specialist findings:\n")

        for result in previous_results:
            context_parts.append(f"--- {result.agent_name} ---\n")
            context_parts.append(result.response)
            context_parts.append("\n\n")

        context_parts.append("Building on the above findings, please provide your analysis.")

        return "".join(context_parts)

    def _build_context_with_tool_outputs(
        self,
        original_question: str,
        previous_results: list[SpecialistResult],
        tool_outputs: list[dict[str, Any]],
        avoid_tools: bool = False,
    ) -> str:
        context = self._build_context(original_question, previous_results)

        if tool_outputs:
            context = (
                f"{context}\n\nShared tool outputs (reuse these instead of calling tools):\n"
                f"{format_tool_outputs(tool_outputs)}"
            )

        if avoid_tools:
            context = (
                f"{context}\n\nUse the shared tool outputs above and avoid calling tools "
                "unless absolutely necessary."
            )

        return context

    def _merge_tool_outputs(
        self,
        shared: list[dict[str, Any]],
        new_outputs: list[dict[str, Any]],
    ) -> None:
        if not new_outputs:
            return
        seen = {self._tool_output_key(item) for item in shared}
        for item in new_outputs:
            key = self._tool_output_key(item)
            if key not in seen:
                shared.append(item)
                seen.add(key)

    def _tool_output_key(self, tool_output: dict[str, Any]) -> str:
        name = tool_output.get("name", "unknown_tool")
        response = tool_output.get("response")
        try:
            response_key = json.dumps(
                response,
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            response_key = str(response)
        return f"{name}:{response_key}"

    def _select_data_provider(self, agent_names: list[str]) -> str | None:
        for preferred in DATA_PROVIDER_PREFERENCE:
            if preferred in agent_names:
                return preferred
        return agent_names[0] if agent_names else None

    @staticmethod
    def _has_clean_addon_specialist(results: list[SpecialistResult]) -> bool:
        """True iff any specialist in ``results`` produced a complete, non-salvage
        answer. Used to decide whether to clear the sticky ``_last_agent_error``
        when the data provider salvaged but a later specialist saved the turn.
        Falls back to a free-form length check when structured output is off
        (``result.structured`` is ``None``)."""
        for result in results:
            structured = result.structured
            if structured is not None:
                if not structured.agent_error and structured.answer_markdown.strip():
                    return True
            elif result.response.strip():
                return True
        return False

    def _normalize_for_overlap(self, text: str) -> str:
        normalized = text.lower()
        normalized = _CHART_REF_PATTERN.sub("[chart:id]", normalized)
        normalized = _CHART_BLOCK_PATTERN.sub("```chart```", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _responses_overlap(
        self,
        results: list[SpecialistResult],
    ) -> tuple[bool, float]:
        if len(results) < 2:
            return (False, 0.0)

        normalized = [self._normalize_for_overlap(r.response) for r in results]
        if any(len(text) < SYNTHESIS_OVERLAP_MIN_CHARS for text in normalized):
            return (False, 0.0)

        min_ratio = 1.0
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                ratio = difflib.SequenceMatcher(None, normalized[i], normalized[j]).ratio()
                min_ratio = min(min_ratio, ratio)
                if ratio < SYNTHESIS_OVERLAP_THRESHOLD:
                    return (False, min_ratio)

        return (True, min_ratio)

    def _score_response(self, text: str) -> int:
        score = len(text)
        lowered = text.lower()
        if "[chart:" in lowered or "```chart" in lowered:
            score += 200
        return score

    def _mark_branch_error(self) -> None:
        """Mark this run as having taken a salvage path on at least one
        branch. ``run_orchestrated_streaming`` ORs this into
        ``turn.agent_error`` after the stream drains (the boolean channel
        of the two-channel union seam). Every code path that
        constructs a salvaged ``SpecialistResult`` must call this — the
        in-impl salvage block, the per-branch watchdog handler, and any
        future guardrail / retry-plugin path. Skipping it silently
        drops the run from the boolean channel."""
        self._last_agent_error = True

    def _select_best_response(
        self,
        results: list[SpecialistResult],
    ) -> SpecialistResult:
        return max(results, key=lambda result: self._score_response(result.response))

    def _resolve_specialist_app(self, agent_name: str) -> App:
        """Resolve the specialist ``App`` through the session's accessor when
        threaded — returning the same cached instance the router path uses, so
        plugin set and trace grouping stay consistent across paths.

        Standalone/test fallback (no accessor): a self-contained App cache
        keyed by ``agent_name`` with plugins built once.
        """
        if self._get_specialist_app is not None:
            return self._get_specialist_app(agent_name)
        # Standalone/test fallback: same get-or-create rule as
        # AgentSession.get_specialist_app, via the shared helper so the two
        # cannot drift (e.g. one caching per attempt).
        from core.app_factory import resolve_specialist_app
        from core.session_plugins import build_session_plugins

        if self._standalone_plugins is None:
            self._standalone_plugins = build_session_plugins()
        return resolve_specialist_app(
            app_cls=App,
            name=self.app_name,
            plugins=self._standalone_plugins,
            cache=self._standalone_apps,
            agent_name=agent_name,
            root_agent=None,
            find_agent=lambda name: self.specialists[name],
        )

    async def _run_specialist(
        self,
        agent_name: str,
        question: str,
        *,
        remaining_agents_pending: bool = False,
    ) -> SpecialistResult:
        """Run a single specialist agent inside a fresh per-branch
        :class:`TurnErrorSink` scope under a per-branch watchdog.

        Public entry point. Constructs a per-branch sink + binds it to
        the ``_current_turn_error_sink`` ContextVar, then runs the
        specialist inside ``asyncio.wait_for(BRANCH_BUDGET_SECONDS)``
        so one pathological branch is salvaged without eating the whole
        turn budget (nested bound: ``BRANCH_BUDGET_SECONDS`` ≤
        ``TURN_BUDGET_SECONDS``).

        On per-branch watchdog trip (``TimeoutError``):
        - The branch coroutine is cancelled mid-flight.
        - The shared salvage seam (:func:`record_salvage_signal`) is
          invoked with ``cause="watchdog"`` — Channel A mutates *this*
          branch's sink (which we hold the binding to), Channel B sets
          the payload ContextVar.
        - A salvage :class:`SpecialistResult` is constructed with
          ``structured.agent_error=True`` so the orchestrator's post-
          gather reduction sees this branch as salvaged. The sibling
          branches under the same ``asyncio.gather`` continue
          independently — the per-branch watchdog cancels only this
          branch, not the gather as a whole (the runner-owned turn
          watchdog at ``run_orchestrated_streaming`` is the one that
          cancels the gather).

        Per-branch implementation note (deliberate design choice):
        the per-branch watchdog could be plugin-owned
        (in ``SalvagePlugin.before_run_callback``). This implementation
        is **orchestrator-owned** — same outcome with less code,
        because the orchestrator naturally owns the per-branch boundary
        (one ``_run_specialist`` call per gathered branch); the
        "plugin-owned" alternative describes the *lifecycle* (per-branch,
        armed at branch entry, disarmed at branch exit), not a hard
        requirement on the owning class. The shared salvage seam is
        invoked identically; the channel writes are byte-identical to
        what a plugin-callback-based implementation would produce.

        The actual specialist-run body lives in :meth:`_run_specialist_impl`;
        this wrapper exists solely to bind the sink + watchdog without
        indenting ~460 lines of body.
        """
        sink = TurnErrorSink()
        # ``bind_salvage_payload_scope`` zeroes the Channel-B ContextVar
        # before this branch runs and resets it on exit, so sequential
        # ``_run_specialist`` invocations on the same task (handoff queue
        # processing in ``execute()``) cannot leak a prior branch's
        # payload into the next branch's ``get_pending_salvage_payload()``
        # read inside ``_run_specialist_impl``. The specialist runner
        # binds the same scope per-attempt; the orchestrator now mirrors
        # that discipline at the per-branch boundary.
        with bind_turn_error_sink(sink), bind_salvage_payload_scope():
            try:
                return await asyncio.wait_for(
                    self._run_specialist_impl(
                        agent_name,
                        question,
                        remaining_agents_pending=remaining_agents_pending,
                    ),
                    timeout=BRANCH_BUDGET_SECONDS,
                )
            except TimeoutError as e:
                # Per-branch watchdog tripped.
                # Invoke the shared salvage seam to write both channels
                # for this branch. The sink we bound above is still in
                # scope, so Channel A is mutated in place; Channel B's
                # payload is set via ContextVar. Return a salvage
                # SpecialistResult so the post-gather reduction sees
                # this branch as salvaged and the orchestrator's
                # set-AND-clear over ``_last_agent_error`` continues to
                # work unchanged (preserved verbatim).
                record_salvage_signal(
                    cause=CAUSE_WATCHDOG,
                    error=e,
                    extra_log_context={
                        "agent_name": agent_name,
                        "branch_budget_seconds": BRANCH_BUDGET_SECONDS,
                        "scope": "per_branch",
                    },
                )
                # Watchdog cancels ``_run_specialist_impl`` before its
                # own ``_mark_branch_error()`` fires; mirror it here so
                # ``run_orchestrated_streaming`` sees the flag.
                self._mark_branch_error()
                return SpecialistResult(
                    agent_name=agent_name,
                    response=SALVAGE_TEXT_BRANCH_TIMEOUT,
                    structured=build_fallback_response(
                        SALVAGE_TEXT_BRANCH_TIMEOUT,
                        agent_name=agent_name,
                        agent_error=True,
                    ),
                )

    async def _run_specialist_impl(
        self,
        agent_name: str,
        question: str,
        *,
        remaining_agents_pending: bool = False,
    ) -> SpecialistResult:
        """Actual specialist-run logic, wrapped by :meth:`_run_specialist`.

        Two-phase execution pattern:
        1. Phase 1: Execute the agent with SSE streaming to handle tool calls,
           detect handoffs, and capture any early text responses
        2. Phase 2: If no text response captured (common after tool calls),
           make a follow-up call with SSE streaming to get the model's text response

        This pattern is required because the Google ADK doesn't deliver the
        model's text response after tool execution in the same stream - the
        final event has content=present but parts=0.

        Using SSE streaming in both phases improves time-to-first-token and
        allows capturing text that may appear before tool calls.

        Args:
            remaining_agents_pending: True when this specialist is the data
                provider in the early-streaming flow and other specialists are
                about to run. Suppresses the "I couldn't finalize..." apology
                in the salvage path since the addon synthesis below will carry
                the analysis. Default ``False`` preserves prior behaviour on
                all other call sites (sequential dispatch, handoff target,
                single-agent path).
        """
        start_time = time.perf_counter()
        specialist = self.specialists[agent_name]

        logger.info("Running specialist: %s", agent_name)

        # Create a fresh ADK session per specialist to avoid contamination
        # from other specialists' events in the shared session
        specialist_session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.customer_id,
        )

        runner = Runner(
            app=self._resolve_specialist_app(agent_name),
            session_service=self.session_service,
        )

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=question)],
        )

        # Use SSE streaming for faster response capture. ``max_llm_calls=5``
        # mirrors the fast-path cap in the specialist runner — without
        # it the runner defaults to 500 and a single specialist can spin
        # tool-call loops indefinitely (e.g. flash-preview's repeated
        # ``request_specialist_handoff`` calls observed in repeated-handoff
        # incidents). The
        # below ``handoff_seen`` early-break is the primary guard; this is
        # the backstop.
        run_config = RunConfig(
            streaming_mode=StreamingMode.SSE,
            response_modalities=["TEXT"],
            max_llm_calls=5,
        )

        full_response_parts: list[str] = []
        handoff_request: HandoffRequest | None = None
        tool_was_called = False
        tool_responses: list[dict[str, Any]] = []
        handoff_seen = False
        # Diagnostic state, surfaced on the salvage path so a stuck-turn
        # investigation can distinguish "Vertex returned empty" from
        # "Vertex stalled" (events_seen == 0).
        events_seen = 0
        last_finish_reason: str | None = None

        try:
            # Wall-clock bounds enforced by the watchdog umbrella (see
            # the specialist runner's ``TURN_BUDGET_SECONDS`` docstring).
            event_iter = runner.run_async(
                user_id=self.customer_id,
                session_id=specialist_session.id,
                new_message=user_content,
                run_config=run_config,
            ).__aiter__()
            try:
                while True:
                    try:
                        event = await event_iter.__anext__()
                    except StopAsyncIteration:
                        break

                    events_seen += 1
                    finish_reason = getattr(
                        getattr(event, "actions", None), "finish_reason", None
                    ) or getattr(event, "finish_reason", None)
                    if finish_reason is not None:
                        last_finish_reason = str(finish_reason)

                    if not event.content or not event.content.parts:
                        continue

                    for part in event.content.parts:
                        # Capture text response incrementally (SSE streaming)
                        if hasattr(part, "text") and part.text:
                            # For streaming, we get cumulative text, so
                            # just keep latest
                            if not full_response_parts or part.text != full_response_parts[-1]:
                                full_response_parts = [part.text]
                                logger.debug(
                                    "[%s] Captured text: %d chars",
                                    agent_name,
                                    len(part.text),
                                )

                        # Track if tools were called
                        if hasattr(part, "function_call") and part.function_call:
                            tool_was_called = True
                            logger.debug(
                                "[%s] Tool called: %s",
                                agent_name,
                                part.function_call.name,
                            )

                        # Check for handoff request or tool output (shared with fast-path).
                        detected_handoff = detect_handoff(part, source_agent=agent_name)
                        if detected_handoff is not None:
                            # First handoff wins — matches the fast-path dispatch loop
                            # so both routes behave identically when a specialist
                            # emits more than one handoff call (prompt directive
                            # forbids it, but defend anyway).
                            if handoff_request is None:
                                handoff_request = detected_handoff
                                logger.info(
                                    "Handoff requested by %s to %s",
                                    agent_name,
                                    handoff_request.target_agent,
                                )
                                # Once a specialist has signalled "hand off", any
                                # further tool-calling is wasted work and (with
                                # flash-preview) becomes a tight loop on the
                                # handoff tool itself. Exit phase 1 immediately
                                # — the outer dispatch consumes ``handoff_request``
                                # and runs the target specialist.
                                handoff_seen = True
                        else:
                            tool_output = collect_tool_response(part)
                            if tool_output is not None:
                                tool_responses.append(tool_output)

                    if handoff_seen:
                        break
            finally:
                # Close the iterator so any HTTP stream / generator
                # holding state is released — under upstream cancellation
                # the generator needs an explicit ``aclose`` for clean
                # teardown of resources further down the stack.
                aclose = getattr(event_iter, "aclose", None)
                if aclose is not None:
                    with suppress(Exception):
                        await aclose()

            full_response = "".join(full_response_parts)

            # Phase 2: If tool was called but no text captured, get text response.
            # Skip when a handoff was detected — the dispatch chain will run
            # the target specialist; this specialist's text isn't shown.
            # No retry on Phase 2 — by this point Vertex has streamed Phase-1
            # events successfully, so a Phase-2 stall is a model-issue, not
            # an HTTP-layer hang.
            if tool_was_called and not full_response and not handoff_seen:
                phase2_start = time.perf_counter()
                logger.info("[%s] Phase 2: Getting text response after tool execution", agent_name)

                # Minimal continuation prompt - session already has question + tool results.
                # Phase 2 is for writing the analysis only — calling another tool here
                # (notably create_visualization) leaves the chart reference unreturned to
                # the user when the model stops without prose. Forbid further tool calls
                # explicitly; the Phase-2 tool-capture below is a defence-in-depth backstop.
                followup_content = types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Write your analysis now based on the data already retrieved. "
                                "Do not call any more tools. "
                                "If you previously called create_visualization, embed the returned "
                                "'reference' value (e.g., [chart:chart_abc123]) in your prose exactly as returned."
                            )
                        )
                    ],
                )

                # Phase 2 shares the per-branch / per-turn watchdog
                # umbrella with Phase 1; no per-event idle ``wait_for``
                # here either.
                phase2_parts: list[str] = []
                phase2_iter = runner.run_async(
                    user_id=self.customer_id,
                    session_id=specialist_session.id,
                    new_message=followup_content,
                    run_config=run_config,
                ).__aiter__()
                try:
                    while True:
                        try:
                            event = await phase2_iter.__anext__()
                        except StopAsyncIteration:
                            break

                        events_seen += 1
                        finish_reason = getattr(
                            getattr(event, "actions", None), "finish_reason", None
                        ) or getattr(event, "finish_reason", None)
                        if finish_reason is not None:
                            last_finish_reason = str(finish_reason)

                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    phase2_parts = [part.text]
                                    continue
                                # Capture any tool responses the model emits in Phase 2
                                # (e.g. a stray create_visualization call). Without this,
                                # a model that produces a chart-only Phase 2 leaves the
                                # chart reference invisible to the salvage path, and the
                                # user sees the generic "trouble finalizing" string
                                # instead of the chart-refs salvage.
                                tool_output = collect_tool_response(part)
                                if tool_output is not None:
                                    tool_responses.append(tool_output)
                            if event.is_final_response():
                                break
                finally:
                    aclose = getattr(phase2_iter, "aclose", None)
                    if aclose is not None:
                        with suppress(Exception):
                            await aclose()

                full_response = "".join(phase2_parts)
                phase2_duration_ms = int((time.perf_counter() - phase2_start) * 1000)
                logger.info(
                    "[%s] Phase 2 completed in %dms: %d chars",
                    agent_name,
                    phase2_duration_ms,
                    len(full_response),
                )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_timing(
                logger,
                "specialist_execution",
                duration_ms,
                status="error",
                agent=agent_name,
                error=str(e),
            )
            logger.exception("Error running specialist %s", agent_name)
            return SpecialistResult(
                agent_name=agent_name,
                response=f"Error from {agent_name}: {e}",
            )

        # Salvage when the model produced no usable text and no handoff
        # was detected. Covers timeout, empty-response, and "tool called
        # but no Phase-2 text" failures. When a handoff was detected the
        # target specialist will produce the user-visible text — salvaging
        # here would either be discarded by the dispatch chain or, in
        # llm-mode synthesis, confuse the synthesiser by introducing
        # fallback prose into the chain.
        #
        # By design: tenant-scope blocks force-fire regardless of model text
        # — see ``SalvagePayload.requires_force_salvage``.
        salvage_payload = get_pending_salvage_payload()
        force_salvage = salvage_payload is not None and salvage_payload.requires_force_salvage()
        used_salvage = False
        if (not full_response.strip() or force_salvage) and handoff_request is None:
            # Diagnostic dump for replay / root-cause analysis. Same fields
            # as the specialist runner's salvage diagnostic so the
            # ``tabi-logs`` query returns a uniform shape regardless of
            # which path the turn went down. See the documented security
            # rationale on why the
            # user message text is deliberately NOT included here — the
            # chat-pipeline log stream carries the prompt against the same
            # structlog-contextvar-bound ``turn_id`` / ``conversation_id``.
            logger.warning(
                "specialist.salvage_diagnostic",
                agent=agent_name,
                agent_model=getattr(specialist, "model", None),
                path="orchestrator",
                events_seen=events_seen,
                text_parts_count=len(full_response_parts),
                tool_outputs_count=len(tool_responses),
                tool_output_names=[str(o.get("tool", o.get("name", "?"))) for o in tool_responses],
                last_finish_reason=last_finish_reason,
            )

            salvaged_refs = _extract_salvaged_chart_refs(tool_responses)
            if remaining_agents_pending:
                # Addon synthesis is about to run and will carry the real
                # analysis. Emit only the chart refs (no preface, no apology)
                # so the message reads as: chart → "Additional insights" → addon.
                # When there are no chart refs, emit nothing — the addon fully
                # replaces this specialist's contribution.
                full_response = "\n".join(salvaged_refs) if salvaged_refs else ""
            elif salvaged_refs:
                full_response = (
                    "Here's the chart for that data:\n\n"
                    + "\n".join(salvaged_refs)
                    + "\n\nI couldn't finalize a written analysis this turn — "
                    "ask a follow-up if you'd like me to interpret the result."
                )
            else:
                full_response = (
                    "I had trouble finalizing this response. "
                    "Please try asking again, or rephrase your question."
                )
            used_salvage = True
            # Sticky for the whole multi-agent run — even if a later
            # specialist succeeds, the user-visible answer this turn
            # included salvage prose, so flag it.
            self._mark_branch_error()

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "[%s] Completed in %dms: %d chars%s",
            agent_name,
            duration_ms,
            len(full_response),
            " (with handoff)" if handoff_request else "",
        )

        log_timing(
            logger,
            "specialist_execution",
            duration_ms,
            agent=agent_name,
            has_handoff=handoff_request is not None,
            response_chars=len(full_response),
        )

        # Metric trace logging for debugging metric-to-answer flow
        metric_trace_logger.info(
            "AGENT_RESPONSE",
            event_type="AGENT_RESPONSE",
            agent_name=agent_name,
            response_length=len(full_response),
            response_text=full_response,
            duration_ms=duration_ms,
            has_handoff=handoff_request is not None,
        )

        # Tool-based handoff (above) wins over the structured
        # ``response.handoff`` so behaviour matches the fast-path rule in
        # the specialist-collection path.
        structured: SpecialistResponse | None = None
        if get_config().feature_flags.structured_output_enabled:
            if used_salvage:
                # Synthesised salvage text is plain markdown, not structured
                # JSON. Calling ``parse_specialist_response`` would emit a
                # spurious ``parse_failed`` warning before falling back;
                # build the fallback directly so dashboards' parse-failure
                # rate stays clean. ``agent_error=True`` is what eval reads
                # via ``AgentSession.get_last_agent_error()`` to skip
                # factuality / safety scoring for this turn.
                structured = build_fallback_response(
                    full_response, agent_name=agent_name, agent_error=True
                )
            else:
                structured = parse_specialist_response(full_response, agent_name=agent_name)
                full_response = structured.answer_markdown
                if handoff_request is None and structured.handoff is not None:
                    intent = structured.handoff
                    handoff_request = HandoffRequest(
                        source_agent=agent_name,
                        target_agent=intent.target_agent,
                        reason=intent.reason,
                        context_summary=intent.context_summary,
                    )

        return SpecialistResult(
            agent_name=agent_name,
            response=full_response,
            handoff_request=handoff_request,
            tool_responses=tool_responses,
            structured=structured,
        )
