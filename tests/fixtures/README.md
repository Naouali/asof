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
| `fred_observations_dgs10.json` | **Hand-constructed** from the documented response shape. No free FRED key was available when Milestone 2 was built. |
| `alfred_observations_gdpc1.json` | **Hand-constructed** from the documented vintage response shape. Models one GDP quarter revised twice — the case the vintage machinery exists for. |

The two hand-constructed FRED fixtures are the weakest link in this directory:
they assert that our parser handles the shape the documentation describes, not the
shape the API actually returns. Re-record both against the live API once a key is
configured, and delete this paragraph.
