"""``StorytellingAgent`` — ADK Agent that owns the chat-triggered
narrative/diagnostic LLM call site.

Previously, ``StorytellingService._generate_narrative`` issued a raw
``google.genai.Client.aio.models.generate_content(...)`` call that
sat outside ADK entirely. This wires the Agent under the shared
``App`` so the chat-triggered story_types (``narrative`` +
``diagnostic``) inherit the same plugin set every specialist already
does. A later consolidation moved narrative onto this path unconditionally.

Single agent for both chat-triggered types because they share the same
``_NARRATIVE_RESPONSE_SCHEMA`` constraint, the same thinking budget
(low — chat-narrative is latency-sensitive), and the same model slot
(``models.storytelling``). The per-story_type *system prompt* divergence
is preserved by passing the system prompt via the user-message
concatenation pattern the current direct-call path already uses, not by
splitting into two agents.

No tools — this is a one-shot structured-output call, not a tool-loop
agent. Tools would defeat the latency profile we're matching (raw
``generate_content`` is one call; an agent with tools could iterate).

No planner — chat-narrative is latency-sensitive; the current direct-call
path uses ``ThinkingLevel.LOW`` for the same reason. ``planner=None``
maps to that intent under ADK; raising thinking is a future-iteration
question if the A/B comparison ever surfaces a need.

Standalone /report story_types (``report``, ``summary``) are explicitly
NOT routed through this agent — consolidating those is future work. The flag check
in ``_generate_narrative`` also tests ``is_chat_triggered`` so the
non-chat paths stay on the direct call regardless of flag state.
"""

from __future__ import annotations

from google.adk.agents import Agent

from config import get_config
from core.specialist_schema import build_generate_content_config

# Imported lazily inside the factory to avoid importing
# ``storytelling_service`` (which pulls in a chunk of the analytics
# surface) at agent-registration time.


# Minimal system instruction. The per-story_type system prompt
# (``get_storytelling_prompt``) is concatenated into the user-message
# contents at call time, matching the current direct-call path — see
# ``StorytellingService._generate_narrative`` for the historical seam.
_STORYTELLING_INSTRUCTION = (
    "You generate structured data-story responses for the chat-narrative "
    "flow. The complete task-specific system prompt is provided as part "
    "of the user message; the response must match the JSON schema "
    "supplied at runtime."
)


def create_storytelling_agent(
    prompt_version: str = "v3.1",  # noqa: ARG001 — accepted for factory-signature parity
    model: str | None = None,
) -> Agent:
    """Create the ``StorytellingAgent``.

    Args:
        prompt_version: Accepted for signature parity with the specialist
            factories in ``core/session.py``'s registration loop, but
            unused — the storytelling system prompt comes from the
            ``storytelling_prompts`` registry (versioned independently)
            and is concatenated into the user message at call time.
        model: Optional Gemini model override. Defaults to the
            ``models.storytelling`` slot from ``Config``.

    Returns:
        Configured ADK ``Agent`` ready for registration under
        ``RunnerHost.get_specialist_app("storytelling")``.
    """
    # [public-repo stub] proprietary services.storytelling_service excluded
    _NARRATIVE_RESPONSE_SCHEMA = None  # type: ignore[assignment]

    resolved_model = model or get_config().models.storytelling.model
    return Agent(
        name="storytelling",
        model=resolved_model,
        description=(
            "Chat-narrative storytelling agent. One-shot structured-output "
            "call that turns prefetched metrics into a Story (executive "
            "summary + ordered slides). Driven by StorytellingService; "
            "not routable from the classifier."
        ),
        instruction=_STORYTELLING_INSTRUCTION,
        output_schema=_NARRATIVE_RESPONSE_SCHEMA,
        tools=[],
        planner=None,
        generate_content_config=build_generate_content_config(),
    )
