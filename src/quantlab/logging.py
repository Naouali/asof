"""Structured logging.

Two rules matter here, both from the spec:

* Section 13 -- "Do not quietly substitute a different data source when one fails;
  fail loudly and log it." Ingestion degradation must be an explicit, greppable
  event, so :func:`degraded` exists and is the only sanctioned way to record it.
* Section 10 -- unavailable sources must produce a clear message, not a traceback.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

__all__ = ["configure_logging", "degraded", "get_logger"]

_configured = False


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog on top of stdlib logging. Idempotent.

    Two decisions worth stating:

    * structlog uses the *stdlib* logger factory, not its own ``PrintLogger``, so
      ``add_logger_name`` has a ``.name`` to read.
    * Records from third-party libraries (httpx, uvicorn, apscheduler) are rendered
      through the same ``ProcessorFormatter``. Without that, `docker compose logs`
      is a mix of JSON and bare strings, and any ingest-health monitor that parses
      the stream has to cope with both.
    """
    global _configured

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    # Replace rather than append: reconfiguring (a CLI callback after an earlier
    # import, or a test) must not stack duplicate handlers.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "urllib3", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name or "quantlab")
    return logger


def degraded(source: str, reason: str, **context: Any) -> None:
    """Record that a data source is unavailable or has fallen back.

    This never silently substitutes another source -- callers are expected to log
    through here and then *stop*, or to proceed only where the substitution is an
    explicit, configured decision. The event name is stable (`source.degraded`) so
    that ingest health monitoring can count it.
    """
    get_logger("quantlab.data").warning("source.degraded", source=source, reason=reason, **context)
