"""Core components: routing, session orchestration, salvage, validation.

Deliberately **import-free**. Eager re-exports here — and in the sibling
``agents`` / ``tools`` / ``evaluation`` packages — previously created a
cold-import circular dependency (``core/__init__`` → ``core.session`` →
``agents.base`` → ``core.conversation_manager`` → back into a
partially-initialised ``core/__init__``). It was masked by import order
everywhere until a fresh entry point cold-imported through ``agents`` /
``tools`` first.

Import core components from their submodules directly, e.g.
``from tabi_analytics.core.session import AgentSession``. Do NOT re-add
eager ``from tabi_analytics.core.X import Y`` lines here — it reintroduces
the cycle. ``tests/test_core_cold_import.py`` is the regression guard.
"""
