"""Prompt loader for specialist agents.

Reads agent prompt templates from agents/prompts/{agent_name}_{version}.txt.
Version strings normalise between dot (v3.1) and underscore (v3_1) forms so
both conventions work at the call site.

Usage::

    from agents.prompts import get_agent_prompt
    prompt = get_agent_prompt("pipeline_analyst", version="v3.1")
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _normalise_version(version: str) -> str:
    """Normalise version string: v3_1 -> v3.1, v3.1 -> v3.1."""
    if not version:
        return version
    head = version[:1]
    tail = version[1:]
    # Underscore form → dot form
    if head == "v" and "_" in tail and "." not in tail:
        return head + tail.replace("_", ".")
    return version


def _version_to_filename_suffix(version: str) -> str:
    """Convert version to filename suffix: v3.1 -> v3_1, v2 -> v2."""
    return version.replace(".", "_")


def _candidate_files(agent_name: str) -> list[Path]:
    """Return all prompt files for *agent_name*, sorted highest version first."""
    pattern = re.compile(rf"^{re.escape(agent_name)}_v(\d+)(?:_(\d+))?\.txt$")
    results: list[tuple[tuple[int, int], Path]] = []
    for p in PROMPTS_DIR.glob(f"{agent_name}_v*.txt"):
        m = pattern.match(p.name)
        if m:
            major = int(m.group(1))
            minor = int(m.group(2)) if m.group(2) else 0
            results.append(((major, minor), p))
    results.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in results]


def get_agent_prompt(agent_name: str, version: str = "v3.1") -> str:
    """Load the prompt template for *agent_name* at *version*.

    Normalises the version string (v3_1 and v3.1 are equivalent).
    Falls back to the highest available version when the exact file is missing.
    Raises ``FileNotFoundError`` when no prompt file exists for the agent at all.

    Args:
        agent_name: Agent identifier, e.g. ``"pipeline_analyst"``.
        version: Prompt version, e.g. ``"v3.1"`` or ``"v3_1"``.

    Returns:
        Raw template text (no variable substitution applied here; callers
        may call ``.format(today=...)`` if the template uses ``{today}``).
    """
    version = _normalise_version(version)
    suffix = _version_to_filename_suffix(version)
    exact = PROMPTS_DIR / f"{agent_name}_{suffix}.txt"
    if exact.exists():
        return exact.read_text()

    # Fall back to highest available version
    candidates = _candidate_files(agent_name)
    if candidates:
        return candidates[0].read_text()

    raise FileNotFoundError(
        f"No prompt template found for agent '{agent_name}' "
        f"(requested version '{version}'). "
        f"Expected files in: {PROMPTS_DIR}"
    )
