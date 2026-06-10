"""Gemini agents for recruitment analytics.

Deliberately **import-free** — circular-import rationale: agent factories
import from core and tools; keeping this package empty avoids transitive
import cycles at package-init time. Import agent factories directly, e.g.
``from agents.pipeline_analyst import create_pipeline_analyst``.
Do NOT re-add eager re-exports here.
"""
