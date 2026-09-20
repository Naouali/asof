"""Collecting source limitations into the tearsheet.

The caveats are not written here. They live in the source catalogue as structured
data, recorded once when a source is registered, and this module looks them up.
That is the whole design: a limitation discovered while writing a fetcher appears
automatically on every tearsheet built from data that fetcher produced, rather
than depending on whoever writes the report remembering it.
"""

from __future__ import annotations

from collections.abc import Iterable

from quantlab.data.catalogue import SOURCES, PitQuality, get_source

__all__ = ["caveats_for", "sources_behind"]


def caveats_for(source_keys: Iterable[str], *, limit: int | None = None) -> tuple[str, ...]:
    """Every recorded caveat for a set of sources, prefixed with the source name.

    Unknown keys raise rather than being skipped. A tearsheet that silently
    omitted the caveats of a source it could not identify would be worse than one
    that refuses to build: the reader would see a short, clean list and conclude
    the data was clean (spec section 13 -- never quietly substitute, fail loudly).
    """
    collected: list[str] = []
    for key in dict.fromkeys(source_keys):  # de-duplicated, order preserved
        spec = get_source(key)
        if spec.pit_quality is PitQuality.RESTATED:
            collected.append(
                f"{spec.name}: data is RESTATED, not point-in-time. The provider "
                "rewrites history, so a snapshot taken today does not reproduce "
                "what was knowable at the time. Any result built on it is an "
                "upper bound."
            )
        collected.extend(f"{spec.name}: {caveat}" for caveat in spec.caveats)
    return tuple(collected[:limit] if limit is not None else collected)


def sources_behind(datasets: Iterable[str]) -> tuple[str, ...]:
    """Every catalogue source that can supply any of these datasets.

    Used when a result does not record which source it actually read -- it is a
    superset, which means the caveat list errs toward showing a limitation that
    did not apply rather than hiding one that did.
    """
    wanted = set(datasets)
    return tuple(sorted(key for key, spec in SOURCES.items() if wanted.intersection(spec.datasets)))
