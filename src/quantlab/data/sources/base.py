"""The uniform source interface.

Every source module exposes one or more :class:`Source` subclasses with the same
signature::

    fetch(symbols, start, end) -> polars.DataFrame

conforming to a canonical schema from :mod:`quantlab.data.schemas`, plus the
metadata in :mod:`quantlab.data.catalogue` declaring update frequency, rate
limits, licence and known data quality issues.

The contract sources must honour:

* **Never invent a timestamp.** ``as_of`` is the instant the observation refers to
  and comes from the venue's calendar or from the venue's own bar-close field.
* **Set ``known_at`` honestly.** For a source that revises, it is the vintage or
  publication instant. For a source that does not, it is ``as_of``. Setting it to
  "now" for revisable data is correct and is *supposed* to make historical
  point-in-time queries return nothing -- that is the data telling you the truth.
* **Never return an empty frame to signal failure.** Empty means "this window
  genuinely has no observations". Failure raises.
* **Never fall back to another source.** Spec section 13.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.config import Settings, get_settings
from quantlab.data.catalogue import AssetClass, SourceSpec
from quantlab.data.http import HttpClient, SourceUnavailableError
from quantlab.data.schemas import DatasetSchema, empty_frame, get_schema
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["SOURCE_REGISTRY", "Source", "get_fetcher", "register"]

log = get_logger("quantlab.data.sources")

#: Every registered fetcher, keyed ``"<source>.<dataset>"``.
SOURCE_REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    """Class decorator adding a fetcher to the registry."""
    if cls.name in SOURCE_REGISTRY:
        raise RuntimeError(f"duplicate fetcher name {cls.name!r}")
    SOURCE_REGISTRY[cls.name] = cls
    return cls


def get_fetcher(name: str) -> type[Source]:
    try:
        return SOURCE_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(SOURCE_REGISTRY))
        raise KeyError(f"unknown fetcher {name!r}; known fetchers: {known}") from None


class Source(ABC):
    """Base class for one (source, dataset) pair."""

    #: Registry key, ``"<source>.<dataset>"``.
    name: ClassVar[str]
    #: The catalogue entry carrying licence, rate limits and caveats.
    spec: ClassVar[SourceSpec]
    #: Canonical dataset this fetcher produces.
    dataset: ClassVar[str]
    #: Lake partition.
    asset_class: ClassVar[AssetClass]
    #: Set when a source is known to be inaccessible for a reason no credential
    #: fixes -- an anti-bot challenge, a retired endpoint. Reported by
    #: :meth:`availability`, so tooling never advertises a source as ready when it
    #: is not.
    blocked_reason: ClassVar[str | None] = None
    #: Symbols to pull when a config names no explicit list. Kept small and liquid:
    #: a default that quietly ingests thousands of symbols is a default that gets
    #: the user rate-limited on their first run.
    default_symbols: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        client: HttpClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------- lifecycle --
    @property
    def client(self) -> HttpClient:
        if self._client is None:
            self._client = HttpClient(self.spec, settings=self.settings)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Source:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- schema --
    @property
    def schema(self) -> DatasetSchema:
        return get_schema(self.dataset)

    def availability(self) -> tuple[bool, str]:
        """Whether this fetcher can run right now, and why not if it cannot."""
        if self.blocked_reason is not None:
            return False, self.blocked_reason
        if self.spec.key_setting is None:
            return True, "keyless"
        if self.settings.secret_for(self.spec.key_setting):
            return True, f"QUANTLAB_{self.spec.key_setting.upper()} is set"
        return False, f"QUANTLAB_{self.spec.key_setting.upper()} is not set"

    def require_available(self) -> None:
        ok, reason = self.availability()
        if not ok:
            raise SourceUnavailableError(self.spec.key, reason)

    # ----------------------------------------------------------------- fetch --
    @abstractmethod
    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.datetime,
        end: dt.datetime,
    ) -> pl.DataFrame:
        """Fetch observations for ``symbols`` in ``[start, end]``.

        Returns a frame conforming to :attr:`schema`. Raises
        :class:`~quantlab.data.http.SourceError` on failure; never returns an
        empty frame to mean "something went wrong".
        """

    # --------------------------------------------------------------- helpers --
    def empty(self) -> pl.DataFrame:
        return empty_frame(self.dataset)

    def finalise(
        self, rows: list[dict[str, Any]], *, ingested_at: dt.datetime | None = None
    ) -> pl.DataFrame:
        """Stamp provenance onto raw rows and validate against the schema.

        Sources build plain dicts of their own columns plus ``symbol``, ``as_of``
        and ``known_at``; this fills in ``source``, ``dataset`` and ``ingested_at``
        and enforces the schema, so a source cannot file data under the wrong name
        or skip validation.
        """
        if not rows:
            return self.empty()

        stamp = ingested_at or utcnow()
        frame = pl.DataFrame(
            [
                {
                    **row,
                    "source": self.spec.key,
                    "dataset": self.dataset,
                    "ingested_at": stamp,
                }
                for row in rows
            ],
            infer_schema_length=None,
        )
        # Fill columns the source did not supply, so every fetcher does not have to
        # spell out the nullable ones.
        missing = {
            name: pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in self.schema.polars_schema.items()
            if name not in frame.columns
        }
        if missing:
            frame = frame.with_columns(list(missing.values()))
        return self.schema.validate(frame)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dataset={self.dataset!r})"
