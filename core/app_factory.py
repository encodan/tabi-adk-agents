"""The single get-or-create rule for a session's ADK ``App`` objects.

Both run-path resolvers — ``AgentSession.get_specialist_app`` (router /
coordinator / single-pass) and ``MultiAgentOrchestrator``'s standalone/test
fallback — share one invariant:

* ``name`` is always :data:`APP_NAME` (must equal ``create_session(app_name=)``
  or ADK raises "Session not found");
* ``root_agent`` given → an *ephemeral* ``App`` wrapping that per-call agent
  (two-pass ``model_copy`` agents have distinct identities), reusing the
  session-scoped plugin instances — not cached;
* otherwise → exactly one ``App`` per ``agent_name``, cached and shared across
  every calling site and retry attempt.

That rule lived in two places before; a drift between them (e.g. one caching
per attempt) would silently fan out BigQuery clients / break trace grouping.
It is centralised here so there is one definition to test.

``app_cls`` is injected by the caller rather than imported here so each
module keeps its own ``App`` symbol — preserving the existing
``monkeypatch.setattr("...core.session.App", _CountingApp)`` test seam.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.adk.apps import App

__all__ = ["resolve_specialist_app"]


def resolve_specialist_app(
    *,
    app_cls: type[App],
    name: str,
    plugins: list[Any],
    cache: dict[str, App],
    agent_name: str,
    root_agent: Any | None,
    find_agent: Callable[[str], Any],
) -> App:
    """Apply the get-or-create rule above.

    Args:
        app_cls: the caller's ``App`` class (injected to keep the per-module
            monkeypatch seam intact).
        name: the App name — always :data:`APP_NAME` at every call site.
        plugins: the session-scoped plugin list (built once; shared by every
            App in the session, including ephemeral two-pass wrappers).
        cache: the caller's per-``agent_name`` App cache (mutated in place).
        agent_name: cache key; not asserted ∈ the 7 specialists so the
            storytelling seam registers through this same path.
        root_agent: when given, wrap this exact agent in a fresh, uncached App.
        find_agent: resolves ``agent_name`` → the agent to wrap when caching.
    """
    if root_agent is not None:
        return app_cls(name=name, root_agent=root_agent, plugins=plugins)
    app = cache.get(agent_name)
    if app is None:
        app = app_cls(name=name, root_agent=find_agent(agent_name), plugins=plugins)
        cache[agent_name] = app
    return app
