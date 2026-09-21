# Limitations

What this data cannot tell you. Read this before you act on any number computed
from it.

This document is deliberately blunt. Free data is never clean, and the ways in
which it is dirty are exactly the ways that flatter a backtest.

## Contents

**Part I — what constrains all of it**

1. [The data is free, and you get what you pay for](#1-the-data-is-free-and-you-get-what-you-pay-for)
2. [The point-in-time guarantee has a hard edge](#2-the-point-in-time-guarantee-has-a-hard-edge)
3. [What "reproducible" means here](#3-what-reproducible-means-here)

**Part II — limitations of individual sources**

Positioning and fundamentals · Volatility indices and the anomaly catalogue ·
Option chains · Energy inventories · Disclosed trades · The app


# Part I — what constrains all of it

## 1. The data is free, and you get what you pay for

### Stooq is blocked, so Yahoo's adjustments are unverified

As of 2026-09-20 Stooq serves a JavaScript proof-of-work anti-bot interstitial
instead of CSV. QuantLab does not solve anti-bot challenges, so that source fails
loudly and is disabled in the shipped ingest plan.

The consequence is not merely "one fewer source". Stooq was the **independent
cross-check** on Yahoo's undocumented adjustment methodology. Without it, every
equity price in this system comes from one unofficial API whose adjustments nobody
outside Yahoo can verify, and a systematic error in those adjustments would be
invisible here.

### Yahoo restates prices for splits that have not happened yet

A May 2014 Apple close comes back as $21.12. The real figure was $591.48. Yahoo has
divided it by the 7:1 split that June **and** by the 4:1 split six years later.

Returns are unaffected, which is why most signals are safe. But every price-*level*
signal is wrong and nothing raises: penny-stock filters, nominal price momentum,
round-number effects, and any dollar-volume threshold computed from price times
volume. `adj_close` is worse — it is recomputed whenever Yahoo reprocesses a
corporate action, so a backtest re-run months later can produce different numbers
from identical code.

### Yahoo sometimes serves only 20 years of history, silently

Observed repeatedly on 2026-09-20: the identical SPY request returned 5462 bars
from 2005-01-03 on one call and 5030 bars from 2006-09-20 on another minutes later.
No error either time.

It cannot be detected from the response. When Yahoo truncates, it also reports
`firstTradeDate` as the start of the truncated range, so the payload is internally
consistent and every in-payload check passes. Requesting history in eight-year
windows helps sometimes and not always.

Two things limit the damage, and neither eliminates it:

- The lake is **append-only**, so a later truncated fetch never destroys history
  you already have, and ingest warns (`ingest.coverage_regression`) when a fetch
  returns less than the lake holds.
- On a **first** ingest there is nothing to compare against, so the loss is
  invisible.

**Practical advice: after your first ingest, check `quantlab data status` against
the span you expect, and re-run until it matches.** A backtest over 2006-2026
instead of 2005-2026 excludes the financial crisis, which is not a detail.

### Equity survivorship bias cannot be fully solved

Without paid CRSP, the universe of US equities that free sources will serve is
biased toward companies that still exist. Delisted names — the ones that went to
zero — are partly or entirely missing.

Mitigations in place: the lake is append-only, so every symbol observed is retained
after it delists; `binance.instruments` snapshots the traded crypto universe so its
history exists from the day collection starts; and the catalogue marks each affected
source `survivorship_biased` so a consumer cannot miss it.

**Residual bias remains, and it inflates long-leg returns and deflates short-leg
losses.** Treat cross-sectional equity results built on this data as an upper
bound.

### Restated fundamentals and macro data

SEC EDGAR gives genuine as-filed fundamentals — this is the strongest free dataset
here. Almost everything else does not. FRED serves the latest vintage of
every series; EIA and USDA revise without publishing an archive. For those, our
own download time is the only vintage, which only works from the day
collection starts. **History before the first ingest is restated**, and any signal
built on it is optimistic by an unknown amount.

### No free historical options data

There is none. The snapshot collector accumulates chains from the day it first
runs. **No options backtest before that date is possible.** Any result claiming one
is fabricated.

### Futures term structure is barely available

Carry and basis-momentum need at least two points on the curve. Free continuous
front-month series use an opaque roll rule and carry no second contract. The only
free route to a real curve is exchange settlement bulletins, whose history starts
when you start collecting and whose terms of service may forbid automated access.

### FX has no intraday path and no spreads

ECB reference rates are a single daily fix, not a tradable quote. FX research is
restricted to daily-or-slower work, and there is no free source of FX spreads.

## 2. The point-in-time guarantee has a hard edge

`Snapshot` makes look-ahead bias structurally unavailable **for data that is in the
lake**. It cannot help with the two harder cases:

- **Sources that revise without publishing vintages** (EIA, USDA, CoinGecko market
  caps). For these, `known_at` is our own download time, so history before the day
  you started collecting is restated, and the catalogue says so rather than
  pretending otherwise.
- **Your own knowledge.** The snapshot does not know which symbols you chose to
  ingest, and you chose them knowing which ones did well. A universe of fourteen
  ETFs that still exist in 2026 is a survivorship-biased universe no matter how
  correct the timestamps are.

## 3. What "reproducible" means here

Re-running ingest does **not** reproduce the lake you had: Yahoo restates adjusted closes,
EDGAR receives amended filings, and exchanges delist symbols. **If a result has to be reproducible, keep a
copy of the lake it was computed from.**

# Part II — limitations of individual sources

### Positioning and company fundamentals

**COT release dates are derived, not observed.** No CFTC payload carries one, and
the CFTC publishes no machine-readable release calendar — only the prose rule and
a warning that "holidays can change the COT release schedule". The derivation
here is the rule plus federal holiday arithmetic, rolled forward. It will be
wrong for any ad-hoc delay the CFTC did not announce in a form this code can
read, and the 2018–19 government shutdown is the known case: report dates run
through it unbroken while publication was suspended for weeks. A COT signal
backtested across that winter sees those reports about six weeks before they
actually appeared. There is no free machine-readable record of the catch-up
schedule, so this is not corrected — it is disclosed.

**COT positioning is weekly, and that caps what it can support.** Fifty-two
observations a year is a small sample for any parameter you fit, and the
categories are reclassified from time to time, which creates level shifts that
look exactly like signal. Normalise within a regime, not across one.

**EDGAR covers US registrants only.** No foreign private issuer that files 20-F
without XBRL, no ETF, no company that has deregistered. A universe built from
what EDGAR returns today is a universe of current filers, so the survivorship
problem is displaced rather than solved: the filings of a delisted company remain
in EDGAR, but you have to know its CIK to ask, and the ticker mapping only lists
live registrants.

**XBRL coverage is biased by sector, not random.** Measured over five large
filers: revenue, total assets, book equity and interest expense present for all
five; cost of goods sold for three. JPMorgan has none because banks do not report
one. Dropping filers with missing line items therefore drops financials, and a
profitability screen built this way is implicitly a screen against banks.

**Tagging practice changed around 2012**, and small filers tag inconsistently
throughout. A long backtest on EDGAR fundamentals is running on a materially
different data-generating process before and after that boundary, and the early
period is both thinner and tilted toward large, well-reported companies.

**Only one XBRL tag per metric is read.** Where a filer reports a concept under a
tag not in the precedence list, that metric is simply absent for them rather than
approximated from a near-neighbour. That is deliberate — the alternative is
silently summing things that are not the same — but it means coverage is a
function of the tag list, which is short.

**The fundamentals universe is fifteen companies.** Each company's facts payload
is several megabytes, so the default is small enough to ingest politely. Nothing
here has been validated at the scale a real cross-sectional equity strategy needs.

### Volatility indices and the anomaly catalogue

**VIX is not investable, and nothing in the lake stops you forgetting that.** It
is stored as an index level rather than a bar, so nothing downstream should mistake it for a
tradeable price series — but nothing stops a consumer doing so either. Every tradeable expression — futures, options, the ETPs —
carries a roll cost that spot does not, and over any long horizon they have
underperformed the index by a wide margin. A backtest that "holds VIX" is not a
strategy anyone could have run.

**The VIX series spans a methodology change.** CBOE moved to the model-free
variance-swap calculation on 2003-09-22. The pre-2003 history served today is a
back-cast under the new method; the index actually published then was VXO,
computed from OEX implied volatilities. A fetch spanning that date logs a warning
and the data is still one column, so a long-horizon study has to decide
deliberately whether to use it.

**The anomaly catalogue is a snapshot of one team's replication effort.** Chen
and Zimmermann's assessments are judgements, not measurements, and the 212
predictors are the ones they could find and code — not every anomaly ever
published, and certainly not every anomaly ever tried. The published t-statistic
is what the original paper reported, so it inherits whatever that paper's
specification choices were.

**The published t-stat distribution understates the search.** Median 4.0, with
2.7% below |t| = 2. That is not evidence the field is finding real effects; it is
evidence journals do not publish t-statistics below 2. The signals that were
tried and abandoned are absent by construction, so the distribution is a lower
bound on how hard the space has been searched — which means the multiple-testing
correction it motivates is also a lower bound.

**The catalogue stops in 2016.** Nothing published since is in it, and the
post-2016 literature is where the replication debate has been most active.

**OSAP's distribution channel is fragile by the source's own choice.** The files
live on Google Drive behind ids that change on re-upload, with no versioned URL,
no content hash and no API. The id is pinned and the payload shape is checked, so
a change fails loudly — but it will fail, and when it does the fix is a manual
re-pin rather than anything automatic.

### Option chains

**This dataset has no history and never will have.** No free source sells
historical option chains. The collector's first run is 2026-09-20, and nothing
before that date exists or can be obtained. Any options research is therefore
limited to the window since collection began, and that window only grows if the
job keeps running. A gap in it is permanent.

**A snapshot is not a path.** One observation a day, from a delayed file. Realised
gamma P&L, intraday hedging error, anything that needs the path between two
closes — none of it can be reconstructed from this however long it accumulates.

**Quote quality degrades sharply away from the money.** Measured on one SPX
chain of 20,236 quotes with open interest:

| moneyness | implied vol (median) | relative spread (median) |
| --- | --- | --- |
| near the money (0.9–1.1×) | 0.14 | 1.2% |
| below 0.5× spot | 0.59, max **4.49** | 7.1% |
| above 1.5× spot | 0.15 | 16.3% |

A 449% implied volatility on a deep in-the-money call is not a volatility, it is
an artefact of inverting a price that is essentially intrinsic value. Use the
near-the-money surface; treat the wings as indicative at best.

**The greeks are CBOE's.** Computed with a model, dividend assumption and rate
curve none of which is published with the numbers. They are stored because a free
field is worth keeping, and they should be recomputed before anything is traded
on them.

### Energy inventories

**Not point-in-time, and unfixably so.** EIA revises weekly inventories and
monthly production and serves only the current value. There is no ALFRED
equivalent — the number as first published is not archived anywhere, by EIA or by
anyone else. Every row is therefore restated, a backtest reading it has
look-ahead, and nothing in this codebase can remove it. The mitigations are to
lag the series well past the revision window, or to treat any result built on it
as an upper bound. A warning is logged on every fetch so this cannot be forgotten
quietly.

**Not run against the live API.** No free EIA key was configured, so the parser
is tested against a fixture hand-built from EIA's documented v2 envelope. That
asserts it handles the documented shape, not the actual one — the same debt FRED
carried until a key arrived. The release-date arithmetic, which is where the
look-ahead risk lives, needs no key and is fully tested.

### Disclosed trades

Insider, institutional and congressional data share one property that governs
everything else: **nobody here observes a trade.** Each row is a legal disclosure,
made days to months after the fact. `as_of` is when the trade happened and
`known_at` is when the world could first have known. Every "follow the smart
money" result that looks too good was built on the first.

**Most insider transactions mean nothing.** On a large issuer's Form 4s, grants
(A), option exercises (M) and shares withheld for tax (F) far outnumber
open-market trades. Of what remains, many sales are under 10b5-1 plans scheduled
months earlier: in six weeks of Apple and Nvidia filings in 2026, 12 of the 26
open-market sales were, and there was not one open-market purchase. The
informative subset is open-market *purchases* by officers and directors, and it
is small.

**Insider coverage is of issuers that still exist.** Tickers resolve through the
SEC's current map, so a delisted or acquired company cannot be requested, even
though its filings are still in EDGAR. The data is not survivorship-biased row by
row; the universe you can ask for is.

**A 13F is stale by construction and partial by design.** Positions are counted
at quarter end and disclosed up to 45 days later, so they are six weeks old on the
day they become knowable and nineteen weeks old the day before the next filing.
Only long positions in 13(f) securities appear: no shorts, no cash, no swaps, most
derivatives absent, and a put is reported at the value of the underlying shares.
A long-short fund is indistinguishable from a long-only one, and a manager that
turns its book over in days discloses nothing useful about what it holds now.

**A 13F can be incomplete when first filed.** Managers may omit positions under
confidential treatment and disclose them up to a year later in an amendment. A
restating amendment supersedes matching positions but cannot delete one: a holding
the amendment dropped stays visible from the original filing.

**The CUSIP-to-ticker bridge has holes.** `fails_to_deliver` is the only free
source pairing the two, and a security appears in it only on days it has a
settlement fail. Liquid large caps are covered; a security that always settles
cleanly is never seen. A few CUSIPs map to more than one ticker over time and a
few tickers to more than one CUSIP, so the join needs a rule -- the most frequent
pairing is a reasonable one -- and an unmatched holding is not a holding that does
not exist.

**Part of Congress is missing in both chambers.** Roughly one House report in
eight, and one Senate report in seven, is filed on paper: a scan that cannot be
read, and the members who file that way include some of the most active traders.
`congress_filings` lists every report, so the unread ones can be counted -- count
them before concluding anything about who trades.

**The Senate arrives second-hand.** Its site answers only connections from inside
the United States, so it is read by a daily job on a US-hosted runner, which
commits what it saw to the `senate-mirror` branch of this repository; the ingest
reads that. If the job stops, Senate data stops, and nothing in the lake says so
except its age -- check the data page. The Senate's index also does not say which
state a senator sits for, and an amended report re-lists the trades of the one it
corrects, so counting across both double-counts.

**Congressional reports carry a statutory restriction on use.** 5 U.S.C. 13107(c)
forbids using them for a commercial purpose other than news dissemination, among
other things. The Senate's site makes each visitor accept that; the House's does
not ask, and is covered by the same law. What that means for a product built on
this data is a question for a lawyer, not for this document.

**Congressional trade sizes are brackets.** "$1,001 - $15,000" is the entire
disclosure, and the brackets widen to "$5,000,001 - $25,000,000". Any dollar
aggregate is an order-of-magnitude statement. The ticker is whatever the member
typed and is not validated; the `asset` name is recovered from a PDF's text layer
and is best effort.

**The House parser is pattern-matching on a layout nobody promised to keep.** It
was checked against a spread of 2026 reports, where every transaction on every
page was recovered, and each run cross-checks parsed rows against the dates on the
page and logs a shortfall. It has not been run across earlier years, whose reports
may be laid out differently; expect the first backfill to log some.

**Fails-to-deliver balances are not short interest.** The SEC says so on the page
the data comes from. `quantity` is a cumulative balance, so summing across days
counts the same shares repeatedly, and fails arise from long sales and processing
delays as well as shorts.

**Overlapping re-ingests are stored more than once.** Incremental runs step back
over an overlap window, and for these sources the window is keyed to trade dates
while files arrive weeks later, so each new fails-to-deliver posting re-stores the
previous file once or twice. Point-in-time reads de-duplicate it; raw
`data query` without `--as-of` does not, and will show the repeats.

### The app

**It has no login.** Anyone who can reach the port can read everything in the
lake. The defaults keep it on this machine; nothing about it is safe to expose to
a network as it stands. Notes, alerts and saved views do not exist, because each
needs to know who is asking.

**The brief on a ticker page is arithmetic, not analysis.** Every sentence is a
count of rows shown lower on the same page -- how many insiders bought, how much,
how many House members traded. It cannot tell a meaningful purchase from a token
one, and it says nothing about whether any of it matters.

**"Routine" is a rule, and rules misfile things.** A sale outside a 10b5-1 plan is
shown as a choice, but plans were not flagged on the form before April 2023, so
earlier planned sales look discretionary. A purchase through an option exercise
is hidden with the rest of the option exercises.

**The feed shows each fund filing's four largest changes, not all of them.** The
rest are on the ticker pages of the securities concerned, for the securities the
CUSIP bridge can name. A change in a security the bridge cannot name has no
ticker and no page.

**The price move between trade and disclosure is close-to-close.** It uses the
last close on or before each date, from whatever price source the lake holds, and
ignores dividends. It is a rough answer to "what did I miss", not a return.

**The timeline is capped.** It draws the forty most recent disclosures and the
forty hidden trades that surfaced soonest, and says so when there were more.

**Names are tidied for display, imperfectly.** SEC names are turned round only
when that is clearly safe, so some people still appear as `LAST FIRST`. Two
people with the same name in the same district would be merged.
