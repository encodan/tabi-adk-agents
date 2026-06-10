"""Tools for Gemini function calling (semantic-layer queries, charts, data sampling).

Deliberately **import-free** — see the ``core`` package docstring for the
circular-import rationale. Import tools from their submodules directly, e.g.
``from tools.adk_tools import configure_tools``. Do NOT re-add eager re-exports
here; ``tests/test_core_cold_import.py`` guards it.
"""
