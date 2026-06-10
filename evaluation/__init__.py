"""Evaluation pipeline — quality sampling, metric tracking, eval runner.

Deliberately **import-free** — see the ``tabi_analytics.core`` package
docstring for the circular-import rationale. The eager re-export of
``eval_runner`` here was specifically load-bearing in the cycle:
``tools.tool_tracer`` imports ``evaluation.evaluators``, so an eager
``evaluation/__init__`` dragged ``eval_runner`` → ``core.session`` →
``agents`` → back into ``tools``.

Import from submodules directly, e.g.
``from tabi_analytics.evaluation.eval_runner import EvalRunner``. Do NOT
re-add eager re-exports here; ``tests/test_core_cold_import.py`` guards it.
"""
