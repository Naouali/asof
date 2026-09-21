"""Symbols a source could not serve, and when to ask again.

A ticker in a disclosure is whatever the filer typed. Most are real; a few are
misspelt, delisted, foreign, or were never tickers at all. Asking a price source
for those every night is rude to the source and slow for us, and quietly dropping
them would leave the lake looking complete when it is not.

So a symbol that fails is written down with the reason it failed and the day to
try it again, and the list is shown in the app. The wait doubles with each failed
attempt up to a cap: a ticker that was simply delisted is asked about rarely, and
one that failed because a server was down comes back quickly.

The file is state, not data. Deleting it costs one slow run, nothing more.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["Unpriceable", "UnpriceableSymbol"]

log = get_logger("quantlab.data.unpriceable")

FILE = "unpriceable_symbols.json"
#: The wait after a first failure, doubling with each one after it.
FIRST_WAIT_DAYS = 3
MAX_WAIT_DAYS = 90


@dataclass(frozen=True, slots=True)
class UnpriceableSymbol:
    source: str
    symbol: str
    reason: str
    attempts: int
    first_failed: dt.datetime
    last_tried: dt.datetime
    retry_after: dt.datetime

    @property
    def due(self) -> bool:
        return utcnow() >= self.retry_after

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "reason": self.reason,
            "attempts": self.attempts,
            "first_failed": self.first_failed.isoformat(),
            "last_tried": self.last_tried.isoformat(),
            "retry_after": self.retry_after.isoformat(),
        }


def _wait(attempts: int) -> dt.timedelta:
    return dt.timedelta(days=min(FIRST_WAIT_DAYS * 2 ** (attempts - 1), MAX_WAIT_DAYS))


class Unpriceable:
    """The record of symbols a source could not serve, kept in the state directory."""

    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / FILE
        self._held: dict[tuple[str, str], UnpriceableSymbol] = {}
        self._read()

    # ------------------------------------------------------------------ reading --
    def _read(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            # A damaged file is not worth failing an ingest over: it only means
            # every symbol is tried again, which is what a first run does anyway.
            log.warning("unpriceable.unreadable", path=str(self.path), error=repr(exc))
            return
        for entry in raw.get("symbols", []) if isinstance(raw, dict) else []:
            try:
                found = UnpriceableSymbol(
                    source=str(entry["source"]),
                    symbol=str(entry["symbol"]),
                    reason=str(entry["reason"]),
                    attempts=int(entry["attempts"]),
                    first_failed=dt.datetime.fromisoformat(entry["first_failed"]),
                    last_tried=dt.datetime.fromisoformat(entry["last_tried"]),
                    retry_after=dt.datetime.fromisoformat(entry["retry_after"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self._held[(found.source, found.symbol)] = found

    def all(self) -> list[UnpriceableSymbol]:
        return sorted(self._held.values(), key=lambda row: (row.source, row.symbol))

    def resting(self, source: str, symbols: list[str]) -> list[str]:
        """Those of ``symbols`` that failed recently and are not due to be retried."""
        return [
            symbol
            for symbol in symbols
            if (found := self._held.get((source, symbol))) is not None and not found.due
        ]

    # ------------------------------------------------------------------ writing --
    def failed(self, source: str, symbol: str, reason: str) -> None:
        now = utcnow()
        before = self._held.get((source, symbol))
        attempts = (before.attempts if before else 0) + 1
        self._held[(source, symbol)] = UnpriceableSymbol(
            source=source,
            symbol=symbol,
            reason=reason[:300],
            attempts=attempts,
            first_failed=before.first_failed if before else now,
            last_tried=now,
            retry_after=now + _wait(attempts),
        )

    def served(self, source: str, symbol: str) -> None:
        """It worked: forget the failure entirely."""
        self._held.pop((source, symbol), None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "written_at": utcnow().isoformat(timespec="seconds"),
            "symbols": [row.as_dict() for row in self.all()],
        }
        self.path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
