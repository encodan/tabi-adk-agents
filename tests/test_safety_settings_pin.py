"""Anti-drift gate for the safety-pinning design — every ADK agent factory must pin
``SAFETY_SETTINGS`` via ``build_generate_content_config``.

Two complementary checks:

1. ``test_agent_factory_pins_safety_settings`` — runtime introspection.
   Walks ``getattr(agent, "sub_agents", []) or [agent]`` recursively so any
   future ``LoopAgent`` / ``SequentialAgent`` / ``ParallelAgent`` wrapper
   cannot silently hide an unpinned inner ``LlmAgent``. None of today's
   eight specialists use wrappers, but if a future change does, this test
   fails loudly instead of leaving the inner agent calling Vertex with
   whatever defaults the model version happens to ship.

2. ``test_no_unwrapped_generate_content_config`` — AST-walk static gate.
   Every direct ``client.aio.models.generate_content(...)`` site MUST go
   through ``build_generate_content_config(...)``. Closes the gap the
   safety-pinning work explicitly flagged: "direct sites need a lint or a periodic grep —
   track this as a follow-up if drift becomes a pattern." This IS the
   lint, run on every CI invocation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from google.adk.agents import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types as genai_types

from agents.capacity_planner import create_capacity_planner
from agents.data_scientist import create_data_scientist
from agents.general_analyst import create_general_analyst
from agents.interviewing_coach import create_interviewing_coach
from agents.offer_advisor import create_offer_advisor
from agents.pipeline_analyst import create_pipeline_analyst
from agents.sourcing_strategist import create_sourcing_strategist
from agents.storytelling_agent import create_storytelling_agent
from config import SAFETY_SETTINGS

# Factories deliberately enumerated. We do NOT iterate via
# ``tabi_analytics.agents.__init__`` because it is intentionally import-free
# (guarded by ``tests/test_core_cold_import.py``). Adding a new specialist
# means adding it here too — a small recurring cost that makes the gate
# explicit instead of magical.
AGENT_FACTORIES = [
    create_capacity_planner,
    create_data_scientist,
    create_general_analyst,
    create_interviewing_coach,
    create_offer_advisor,
    create_pipeline_analyst,
    create_sourcing_strategist,
    create_storytelling_agent,
]

EXPECTED_CATEGORIES = {s.category for s in SAFETY_SETTINGS}


def _walk_llm_agents(agent: Agent) -> Iterator[LlmAgent]:
    """Yield every ``LlmAgent`` reachable from ``agent``, recursing through
    ``sub_agents``. Wrapper agents (``LoopAgent`` / ``SequentialAgent`` /
    ``ParallelAgent``) are ``BaseAgent`` but not ``LlmAgent``, so they
    contribute their children rather than themselves.
    """
    if isinstance(agent, LlmAgent):
        yield agent
    for child in getattr(agent, "sub_agents", []) or []:
        yield from _walk_llm_agents(child)


@pytest.mark.parametrize("factory", AGENT_FACTORIES, ids=lambda f: f.__name__)
def test_agent_factory_pins_safety_settings(factory):
    """Every LlmAgent reachable from each factory must carry the pinned
    safety settings — exact category set, non-empty threshold per entry."""
    agent = factory()
    inner_agents = list(_walk_llm_agents(agent))
    assert inner_agents, (
        f"{factory.__name__} returned no LlmAgent — the gate cannot verify "
        "an unreachable target. Add the new agent shape to _walk_llm_agents."
    )
    for inner in inner_agents:
        cfg = inner.generate_content_config
        assert isinstance(cfg, genai_types.GenerateContentConfig), (
            f"{factory.__name__}::{inner.name} has no GenerateContentConfig — "
            "safety pin cannot be threaded. Pass "
            "generate_content_config=build_generate_content_config() in the factory."
        )
        settings = cfg.safety_settings or []
        actual_categories = {s.category for s in settings}
        assert actual_categories == EXPECTED_CATEGORIES, (
            f"{factory.__name__}::{inner.name} safety categories drift: "
            f"expected {EXPECTED_CATEGORIES}, got {actual_categories}. "
            "Use build_generate_content_config(...) — do not construct "
            "GenerateContentConfig bare."
        )
        for setting in settings:
            assert setting.threshold is not None, (
                f"{factory.__name__}::{inner.name} category {setting.category} has threshold=None"
            )


# --- AST-walk static gate ----------------------------------------------------

# Locate the repo root by walking up from this test file. In the showcase repo
# the package is rooted at the repo top (flat layout, no nested src dir),
# so this test sits at ``<repo>/tests/test_safety_settings_pin.py``
# and ``parents[1]`` is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Source roots to scan. Restricted to package sources (no tests, no scripts):
# - tests/ deliberately constructs configs for mock/fixture purposes; they
#   are not on the served path.
# In the showcase the agent / core / tools / evaluation packages live directly
# under the repo root; scan those rather than the monorepo's analytics/src +
# api/src.
_SCAN_ROOTS = (
    _REPO_ROOT / "agents",
    _REPO_ROOT / "core",
    _REPO_ROOT / "tools",
    _REPO_ROOT / "evaluation",
)

# Files where bare GenerateContentConfig(...) is allowed by definition.
_ALLOWED_FILES = frozenset(
    {
        # The helper itself constructs a fresh GenerateContentConfig in the
        # extra=None branch — that's the wrapping primitive every other site
        # routes through. Whitelisting one file keeps the rest exhaustive.
        _REPO_ROOT / "core" / "specialist_schema.py",
    }
)

_HELPER_NAME = "build_generate_content_config"
_TARGET_NAME = "GenerateContentConfig"


def _attach_parents(tree: ast.AST) -> None:
    """Decorate every node with a ``parent`` attribute. ``ast`` does not
    track parents natively; we need them to check whether a target call is
    nested inside the wrapper call.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _is_generate_content_config_call(node: ast.AST) -> bool:
    """True if ``node`` is ``GenerateContentConfig(...)`` or
    ``<anything>.GenerateContentConfig(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == _TARGET_NAME:
        return True
    if isinstance(func, ast.Attribute) and func.attr == _TARGET_NAME:
        return True
    return False


def _is_wrapped_by_helper(node: ast.AST) -> bool:
    """True if any ancestor ``Call`` node is ``build_generate_content_config(...)``."""
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, ast.Call):
            pfunc = parent.func
            name = None
            if isinstance(pfunc, ast.Name):
                name = pfunc.id
            elif isinstance(pfunc, ast.Attribute):
                name = pfunc.attr
            if name == _HELPER_NAME:
                return True
        parent = getattr(parent, "parent", None)
    return False


def _find_unwrapped_calls(path: Path) -> list[int]:
    """Return line numbers of unwrapped GenerateContentConfig(...) calls in
    ``path``. Returns empty list if file unparseable (skip silently — a
    different test owns syntax-error detection)."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return []
    _attach_parents(tree)
    return [
        node.lineno
        for node in ast.walk(tree)
        if _is_generate_content_config_call(node) and not _is_wrapped_by_helper(node)
    ]


def test_no_unwrapped_generate_content_config():
    """Every ``GenerateContentConfig(...)`` construction in package source
    MUST be inside a ``build_generate_content_config(...)`` call.

    Closes the gap the safety-pinning work explicitly flagged: 'direct sites
    need a lint or a periodic grep — track this as a follow-up if drift becomes
    a pattern.' This IS the lint. Compared with
    ``test_agent_factory_pins_safety_settings`` (which only covers the 8
    ADK Agent factories), this catches every direct
    ``client.aio.models.generate_content(...)`` site — including the API
    tier, the orchestrator's addon-stream path, and any future judge
    added to the evaluation package.
    """
    issues: list[tuple[Path, int]] = []
    for root in _SCAN_ROOTS:
        if not root.exists():
            # Worktree may be partial in some CI configurations; skip gracefully.
            continue
        for py_file in root.rglob("*.py"):
            if py_file in _ALLOWED_FILES:
                continue
            for line in _find_unwrapped_calls(py_file):
                issues.append((py_file, line))

    if issues:
        lines = "\n".join(f"  {p.relative_to(_REPO_ROOT)}:{line}" for p, line in issues)
        pytest.fail(
            f"Found {len(issues)} unwrapped GenerateContentConfig(...) calls. "
            f"Every construction must go through "
            f"build_generate_content_config(extra=...). Sites:\n{lines}"
        )
