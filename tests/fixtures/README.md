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
| `sec_submissions_aapl.json` | **Recorded live** 2026-09-20 from the EDGAR submissions API, trimmed from a thousand filings to seven: three Form 4s and one each of Form 3, 144, 10-Q and 8-K, so the form filter has something to reject. |
| `sec_form4_aapl.xml` | **Recorded live** 2026-09-20, unmodified. One real Form 4 carrying a 10b5-1 sale, an option exercise on both tables, shares withheld for tax, and a price given only in a footnote. Tests route it to all three filings with the trade date rewritten to each filing's own report period, so it never claims a trade later than the filing reporting it. |
| `sec_13f_index_brk.json`, `sec_13f_primary_brk.xml` | **Recorded live** 2026-09-20, unmodified: the directory listing and cover page of Berkshire Hathaway's June 2026 13F-HR. The listing is what shows the holdings table is named `56757.xml`. |
| `sec_13f_infotable_brk.xml` | **Recorded live** 2026-09-20, trimmed from 89 lines to 8: the six lines Ally Financial is split across, plus two of Apple's twelve. The cover page still declares 89, so fetching this fixture logs a cover-page mismatch -- correctly. |
| `house_fd_index_2026.xml` | **Recorded live** 2026-09-20 from the Clerk's `2026FD.zip`, trimmed from 1,664 entries to six: two electronic transaction reports, one scanned one, a candidate report, an annual report, and one of the 35 withdrawals that carry no filing date. Tests re-zip it with the byte-order mark the real file opens with. |
| `house_ptr_20034342.pdf` | **Recorded live** 2026-09-20, unmodified: a real two-page periodic transaction report. The only binary fixture, kept so one test goes through the PDF text layer for real -- that path is where the NUL-for-lowercase small caps come from. |
| `house_ptr_20034585.txt`, `house_ptr_20034202.txt` | Text **extracted 2026-09-20** from two more real reports with the same code the fetcher uses. The first holds a transaction with no asset-type code; the second a comment long enough to wrap into the next row. Stored as text to keep the repository light. |
| `senate_ptr_b999bc0e.html`, `senate_ptr_51455bcd.html` | Two real Senate periodic transaction reports, unmodified. **Not recorded here**: the Senate's site refuses connections from outside the United States, so these are the pages as saved on 2026-09-20 by a public third-party scraper (github.com/StrokeOfLuck/senate-ptr-scraper). The first holds two option purchases, an exchange naming two securities and a sale whose ticker appears only in the asset's name; the second is an amendment of fourteen lines. |
| `senate_ptr_index.json` | **Hand-constructed** in the shape of the site's search response (five cells a row, the fourth an HTML link), from the same scraper's code and its saved index, for the same reason. Three real reports: the two above and one filed on paper, whose filer the Senate indexes in capitals. To be re-recorded from the first run of the `senate-mirror` workflow. |

The two FRED fixtures were hand-constructed from the published response shape
until a key was configured, and have now been re-recorded. The parser needed no
change — the documented shape was the real one — but one test did: it asserted
exactly three vintages of 2020 Q1, because the invented payload had three. The
live payload has nine. That assertion was pinned to fabricated data and is now
pinned to a property that survives the next annual revision.
