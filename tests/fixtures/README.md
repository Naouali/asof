# Test fixtures

Recorded API payloads, so the unit suite runs offline and does not depend on a
third party's uptime or on today's market data.

| File | Provenance |
| --- | --- |
| `yahoo_chart_aapl_2014.json` | **Recorded live** 2026-09-20. May–June 2014, chosen because it contains both a dividend and the 7:1 split — the window that proves Yahoo restates prices retroactively. |
| `binance_klines_btcusdt_1d.json` | **Recorded live** 2026-09-20. Daily bars, 2024-01-01 → 2024-01-05. |
| `binance_funding_btcusdt.json` | **Recorded live** 2026-09-20. Funding payments over the same window; exercises the derived 8-hour interval. |
| `binance_exchangeinfo.json` | **Recorded live** 2026-09-20, trimmed to four symbols (the full payload is ~2 MB of filter definitions). One symbol is in `BREAK` status to exercise the delisting path. |
| `stooq_challenge.html` | **Recorded live** 2026-09-20. The JavaScript proof-of-work interstitial Stooq now serves instead of CSV. |
| `stooq_aapl_sample.csv` | Hand-constructed in Stooq's documented CSV format, since the live endpoint no longer serves CSV. Tests the parser, not the transport. |
| `cboe_vix_history.csv` | **Recorded live** 2026-09-20, trimmed. Carries the 1990 opening rows, the 2003 methodology boundary and the 2008 crisis, so the break detector has something to detect. |
| `cboe_vvix_history.csv` | **Recorded live** 2026-09-20, first twelve rows. Contains the unusable opening fortnight -- 71.73, a nine-day gap, then 15.71 -- which is what the reliability filter exists to drop. |
| `osap_signal_documentation.csv` | **Recorded live** 2026-09-20, first eight predictors, all 28 columns. Includes Sloan's accruals, the canonical published anomaly. |
| `fred_observations_dgs10.json` | **Recorded live** 2026-09-20. DGS10, 2024-01-01 → 2024-01-08. Contains a New Year's Day `"."`, FRED's encoding for a missing observation. |
| `alfred_observations_gdpc1.json` | **Recorded live** 2026-09-20. Real GDP across 2020 H1 over the full real-time window: 17 rows covering two quarters, 2020 Q1 alone carrying nine vintages from the advance estimate to a revision five years later. |

The two FRED fixtures were hand-constructed from the published response shape
until a key was configured, and have now been re-recorded. The parser needed no
change — the documented shape was the real one — but one test did: it asserted
exactly three vintages of 2020 Q1, because the invented payload had three. The
live payload has nine. That assertion was pinned to fabricated data and is now
pinned to a property that survives the next annual revision.
