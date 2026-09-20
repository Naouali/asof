"""Logging is load-bearing: the worker's first action is to log.

A misconfigured processor chain raises inside the logging call itself, which in a
container reads as an immediate crash-loop with a traceback that points at
structlog rather than at the real cause. These tests exercise the chain rather
than just constructing it.
"""

from __future__ import annotations

import json

import pytest

from quantlab.logging import configure_logging, degraded, get_logger


@pytest.mark.parametrize("fmt", ["console", "json"])
def test_logging_actually_emits(fmt: str, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt=fmt)
    get_logger("quantlab.test").info("worker.start", interval=30.0)
    captured = capsys.readouterr().err
    assert "worker.start" in captured


def test_json_format_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """The container images default to JSON so `docker compose logs` is parseable."""
    configure_logging(level="INFO", fmt="json")
    get_logger("quantlab.test").info("job.finish", job="ingest-daily", returncode=0)
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "job.finish"
    assert record["job"] == "ingest-daily"
    assert record["logger"] == "quantlab.test"
    assert record["level"] == "info"
    assert "timestamp" in record


def test_reconfiguring_does_not_duplicate_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for _ in range(3):
        configure_logging(level="INFO", fmt="json")
    get_logger("quantlab.test").info("once")
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", fmt="json")
    logger = get_logger("quantlab.test")
    logger.info("suppressed")
    logger.warning("emitted")
    err = capsys.readouterr().err
    assert "suppressed" not in err
    assert "emitted" in err


def test_degraded_emits_a_stable_greppable_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec section 13: a failing source must fail loudly and be logged. Ingest
    health monitoring counts this exact event name, so it is pinned by a test."""
    configure_logging(level="INFO", fmt="json")
    degraded("binance", "HTTP 451 from this jurisdiction", symbol="BTCUSDT")
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "source.degraded"
    assert record["source"] == "binance"
    assert record["level"] == "warning"
    assert record["symbol"] == "BTCUSDT"


def test_exception_info_is_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    """An ingest failure must carry its traceback: spec section 13 requires loud
    failure, and a bare event name is not loud enough to debug from."""
    configure_logging(level="INFO", fmt="json")
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("quantlab.test").exception("ingest.failed")
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[0])
    assert record["event"] == "ingest.failed"
    assert record["level"] == "error"
    assert "ValueError: boom" in err
