"""Canonical evalset path classification.

Single source of truth for which ``*.evalset.json`` files belong to the
chat-specialist family vs. the narrative-storytelling family. Imported by
the ADK eval test parameterisation, a downstream judge-rerun service
(narrative guard), and the judge-alignment dashboard.

Why a module-level constant rather than a literal-per-call-site: when a
future evalset is added or renamed, exactly one place needs editing — a
duplicated literal in three call sites silently desyncs (chat-eval gates
the new file while the narrative guard does not, or vice versa).
"""

from __future__ import annotations

# Narrative-storytelling evalsets — invocations driven via the narrative
# pipeline (excluded) rather than the chat router / specialist pipeline.
# The narrative pipeline does NOT fan ADK tool calls; ``tool_uses`` /
# ``tool_responses`` are empty by design. Keying narrative classification by
# filename keeps the test gates declarative and lets the judge-rerun
# guard fail loud + early on a path-mismatch.
_NARRATIVE_EVALSET_NAMES: frozenset[str] = frozenset({"narrative_chat.evalset.json"})


def is_narrative_evalset(path: str) -> bool:
    """True iff ``path`` resolves to a narrative-storytelling evalset.

    Accepts either a bare filename or a full path; only the basename is
    compared. A future caller passing an absolute path or a Path-like
    object stringified via ``str(...)`` works identically.
    """
    # str.rsplit handles both POSIX and Windows separators without needing
    # to materialise a ``Path`` — keeps this module dependency-free so it
    # can be imported from anywhere (tests, services, the ADK bridge).
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return name in _NARRATIVE_EVALSET_NAMES


__all__ = ["_NARRATIVE_EVALSET_NAMES", "is_narrative_evalset"]
