"""The numbers this app runs on, for the interface to print.

Every threshold the app applies is a constant somewhere in the code or a line in
the configuration. A sentence on screen that repeats one -- "actions of $25 million
and over", "published 90 days late" -- has to get it from here. A number typed
into the interface as well is a number that will one day disagree with the one
the app is actually using, and nothing would say so.
"""

from __future__ import annotations

from typing import Any

from quantlab.api import analytics, events, portfolios, queries
from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.ingest import load_plan
from quantlab.data.sources import usaspending
from quantlab.logging import get_logger

__all__ = ["rules"]

log = get_logger("quantlab.api.rules")

CONTRACTS_FETCHER = "usaspending.government_contracts"


def rules(settings: Settings) -> dict[str, Any]:
    min_action, contractors = _contracts_job(settings)
    return {
        "deadlines": {
            "congress_days": events.CONGRESS_DEADLINE_DAYS,
            "fund_days": events.FUND_DEADLINE_DAYS,
            "insider_business_days": events.INSIDER_DEADLINE_BUSINESS_DAYS,
        },
        "contracts": {
            "min_action_usd": min_action,
            "feed_min_usd": queries.FEED_CONTRACT_MIN_USD,
            "publication_lag_days": usaspending.PUBLICATION_LAG.days,
            "defense_embargo_days": usaspending.DEFENSE_EMBARGO.days,
            "companies": contractors,
            "listed_per_ticker": queries.TICKER_CONTRACTS,
            "on_chart": queries.CHART_CONTRACTS,
        },
        "feed": {
            "fund_changes_per_filing": events.FUND_CHANGES_PER_FILING,
            "fund_min_change_pct": round(events.FUND_MIN_CHANGE * 100),
        },
        "analytics": {
            "min_sample": analytics.MIN_SAMPLE,
            "max_lag_days": analytics.MAX_PLAUSIBLE_LAG_DAYS,
        },
        "portfolios": {"min_priced_share_pct": round(portfolios.MIN_PRICED_SHARE * 100)},
    }


def _contracts_job(settings: Settings) -> tuple[float, int | None]:
    """The smallest contract action the ingest keeps, and how many companies it
    covers -- both read from the configuration the ingest itself reads."""
    minimum = usaspending.DEFAULT_MIN_OBLIGATION
    path = settings.layout.configs / usaspending.CONTRACTORS_FILE
    try:
        plan = load_plan(settings.layout.configs / "ingest.yaml")
        job = next((job for job in plan.jobs if job.fetcher == CONTRACTS_FETCHER), None)
        if job is not None:
            minimum = float(job.options.get("min_obligation", minimum))
            if job.options.get("contractors"):
                path = settings.layout.configs.parent / str(job.options["contractors"])
    except (OSError, ValueError, KeyError) as exc:
        log.warning("rules.plan_unreadable", error=repr(exc))
    try:
        return minimum, len(usaspending.load_contractors(path))
    except SourceError as exc:
        log.warning("rules.contractors_unreadable", error=str(exc))
        return minimum, None
