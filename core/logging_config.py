"""
Logging configuration for the TABI Analytics module.

Thin wrapper around tabi_core.logging — all shared config lives there.
Metric trace logger is analytics-specific and configured here.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

import structlog

# [public-repo stub] proprietary tabi_core.log_redact / tabi_core.logging excluded.
# Minimal shims so logging configuration py_compiles and the metric-trace logger
# keeps working in the showcase. The real implementations live in the private
# tabi-core package (PII redaction processor, shared structlog config, log_timing).


def make_pii_redaction_processor(enabled: bool = True):  # type: ignore[no-untyped-def]
    """No-op structlog processor stub (real one scrubs PII from every event)."""

    def _processor(_logger, _method_name, event_dict):  # type: ignore[no-untyped-def]
        return event_dict

    return _processor


def redaction_enabled() -> bool:
    return True


def _get_log_directory(_service: str) -> str:
    return os.getenv("TABI_LOG_DIR", "logs")


def log_timing(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    """Stub: real impl emits a structured timing event (event_name=<name>)."""
    return None


def _configure(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    """Stub for tabi_core.logging.configure_logging."""
    return None


__all__ = ["configure_logging", "get_metric_trace_logger", "log_timing"]


def configure_logging(
    log_dir: str | None = None,
    log_level: str | None = None,
) -> None:
    """Configure structured logging for the analytics service."""
    if log_level is None:
        from config import get_config

        log_level = get_config().logging.level

    # Thread the OTel-reading log↔trace correlation processor
    # into the shared chain. Defined here (analytics declares
    # ``opentelemetry-api``) so ``tabi_core`` stays OTel-free. No-op when no
    # recording span is active (local dev / tests).
    # [public-repo stub] proprietary core.trace_correlation excluded — use a
    # no-op processor so logging still configures cleanly in the showcase.
    def make_gcp_trace_correlation_processor():  # type: ignore[no-untyped-def]
        def _processor(_logger, _method_name, event_dict):  # type: ignore[no-untyped-def]
            return event_dict

        return _processor

    _configure(
        service="analytics",
        log_dir=log_dir,
        log_level=log_level,
        # Built here so the GCP project / enabled flag are resolved once,
        # not per log line. Redactor is last — any processor after it would
        # emit raw fields that bypass redaction.
        extra_processors=[
            make_gcp_trace_correlation_processor(),
            make_pii_redaction_processor(enabled=redaction_enabled()),
        ],
    )

    # Configure analytics-specific metric trace logger
    _configure_metric_trace_logger(log_dir)


def get_metric_trace_logger() -> structlog.stdlib.BoundLogger:
    """
    Get the metric trace logger for debugging metric-to-answer flow.

    Active only when TABI_METRIC_TRACE=true. Otherwise events are
    silently discarded (never leaks to parent).
    """
    return structlog.get_logger("tabi_analytics.metric_trace")


def _configure_metric_trace_logger(log_dir: str | None) -> None:
    """Configure the metric trace logger with propagate=False always."""
    mt_logger = logging.getLogger("tabi_analytics.metric_trace")
    mt_logger.handlers.clear()
    mt_logger.propagate = False

    if os.getenv("TABI_METRIC_TRACE", "false").lower() == "true":
        mt_logger.setLevel(logging.DEBUG)
        is_production = os.getenv("TABI_ENV") == "production"

        shared_processors: list[structlog.types.Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            # ``format_exc_info`` materialises ``exc_info=True`` as a string
            # ``exception`` field BEFORE the redactor sees it. Matches the
            # main shared chain — otherwise PII inside a metric-trace
            # traceback would bypass redaction.
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # PII redaction — last in the chain so every preceding processor's
            # fields are scrubbed before the JSON/console renderer sees them.
            make_pii_redaction_processor(enabled=redaction_enabled()),
        ]

        if is_production:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    processor=structlog.processors.JSONRenderer(),
                    foreign_pre_chain=shared_processors,
                )
            )
            mt_logger.addHandler(handler)
        else:
            if log_dir is None:
                log_dir = _get_log_directory("analytics")
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            mt_file_handler = logging.handlers.RotatingFileHandler(
                log_path / "metric_trace.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            mt_file_handler.setLevel(logging.DEBUG)
            mt_file_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    processor=structlog.processors.JSONRenderer(),
                    foreign_pre_chain=shared_processors,
                )
            )
            mt_logger.addHandler(mt_file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    processor=structlog.dev.ConsoleRenderer(
                        colors=True,
                        # See rationale in ``tabi_core.logging.configure_logging``.
                        exception_formatter=structlog.dev.plain_traceback,
                    ),
                    foreign_pre_chain=shared_processors,
                )
            )
            mt_logger.addHandler(console_handler)
    else:
        # Disabled: high level + NullHandler = no log records created, no leaking
        mt_logger.setLevel(logging.CRITICAL + 1)
        mt_logger.addHandler(logging.NullHandler())
