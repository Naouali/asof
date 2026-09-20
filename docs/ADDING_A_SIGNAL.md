# Adding a signal

A signal here is a class that turns a point-in-time snapshot into scores. It
never returns weights: portfolio construction owns those, and a signal that sizes
its own positions has quietly taken over risk management (spec section 4).

This guide is the whole process, with the reasons for the parts that look like
bureaucracy. They are the parts that stop a signal being wrong in ways nobody
notices.

## 1. Decide what it needs, and whether the lake has it

```bash
quantlab data catalogue          # every source, and what it can and cannot do
quantlab data status             # what is actually in your lake
```

If the data is not there, the signal still gets written — and made to **raise**
`SignalUnavailableError` naming what is missing. It does not return an empty
cross-section. An empty cross-section looks exactly like a signal with no view,
and a strategy built on one trades nothing while appearing to work.

Check the source's point-in-time quality before you build on it. A source marked
`restated` rewrites its history, so today's snapshot does not reproduce what was
knowable a year ago, and a signal reading it has look-ahead that no amount of
care in the signal itself removes.

## 2. Write the spec first

```python
@register_signal
class MyBrilliantIdea(Signal):
    spec: ClassVar[SignalSpec] = SignalSpec(
        name="assetclass.short_name",
        asset_class=AssetClass.EQUITY,
        tier=2,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("ohlcv_daily",),
        rebalance=RebalanceFrequency.MONTHLY,
        expected_turnover_annual=1.2,
        evidence=EvidenceGrade.MIXED,
        reference="Author (Year), 'Title'",
        known_failure_modes="...",
        warmup_days=252,
    )
```

Two fields do real work and one of them is enforced.

`known_failure_modes` **must say something substantive** — the constructor
rejects anything under forty characters. Every signal in the literature has a
documented way of going wrong, and if you cannot name this one's you do not
understand it well enough to trade it. Write the mechanism, not a disclaimer:
*"crowded, so its worst days coincide with everyone else's"* beats *"past
performance is no guarantee"*.

`expected_turnover_annual` is the single most useful number for guessing whether
a signal survives costs before any of it is built. A signal with 600% turnover
needs roughly six times the gross edge of one with 100% to reach the same net.
Estimate it honestly; you will be measured against it.

`tier` is evidence quality, not interest. Tier 1 is replicated widely and
survives costs. Tier 2 is real but fragile — sensitive to construction, crowded,
or costly. Tier 3 is decayed, and is implemented anyway so it can be re-tested on
current data with that prior attached.

## 3. Compute from a snapshot, and only a snapshot

```python
def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
    bars = snapshot.ohlcv_daily(symbols=list(symbols))
    ...
    return self.finalise(snapshot, scores)
```

`Snapshot` is the look-ahead guard. It filters on `known_at`, refuses a window
that extends past its own instant, and re-checks every frame it returns. A signal
that reaches around it — reading the store directly, or taking a date as a
parameter and indexing past it — has defeated the one structural protection the
platform provides.

Three things to get right inside `compute`:

**Never join on a raw timestamp across datasets.** A daily bar is stamped at its
session close and a daily series at midnight, so an exact-instant join matches
nothing — and a positional join appears to work while regressing one series
against another shifted by every session either side is missing. Join on the
calendar date, explicitly.

**Distinguish absent from zero.** A missing line item is `None`, never `0.0`. A
missing SG&A treated as zero makes a company look more profitable than it is, and
the filers with missing items are systematically the small badly-reported ones —
so the error is a size tilt, not noise.

**Pick one reporting basis and stay on it.** EDGAR publishes the same metric over
3-, 6-, 9- and 12-month windows in a single filing. Taking the most recently
*filed* value per metric once gave nine-month revenue over an instantaneous
balance sheet, understating Apple's gross profitability by 16% and moving Walmart
from last place to first.

## 4. Test the sign before anything else

The most expensive bug in a signal is a reversed sign, because it does not look
like a bug. It backtests as a confident, slow loss — a strategy that
systematically pays a risk premium instead of earning it.

Write the test against the *theory*, not against a number you observed:

```python
def test_commercials_unusually_short_is_a_long_signal() -> None:
    """Keynes and Hicks: hedgers are net short, speculators take the other side
    and are paid to. So an unusually short commercial book means an unusually
    large premium accruing to the long."""
```

A test asserting `score == 0.0737` locks in whatever you wrote. A test asserting
the direction the theory requires catches you writing it backwards.

## 5. Register it

```python
# src/quantlab/signals/loader.py
SIGNAL_MODULES = (
    ...,
    "quantlab.signals.yourclass.your_signal",
)
```

Explicit, not scanned: adding a signal is a deliberate act, and an import error
should surface as an import error rather than as a mysteriously absent signal.

## 6. Run it, then distrust it

```bash
quantlab signals show assetclass.short_name     # the spec, as a reader sees it
quantlab signals run assetclass.short_name --as-of 2026-09-18
```

Then the part that matters:

```bash
quantlab validate anomalies --t-stat <yours>    # where you sit in the literature
quantlab report --config configs/your_run.yaml  # the tearsheet, which may refuse
```

Every backtest records itself as a trial and the count deflates the Sharpe it
reports. **There is no way to search quietly and still quote a deflated number**,
and that is deliberate. If you sweep twenty parameters, the twenty-first result
is deflated against twenty trials whether or not you mention them.

The tearsheet will refuse to render without a trial count and a capacity
estimate, and it orders its panels worst-first. A strategy that fails deflation
says so above its equity curve.

## What a good signal looks like when it is finished

- It raises rather than returning empty when its data is missing.
- Its failure modes name mechanisms you could check, not disclaimers.
- Its sign is pinned by a test that cites the theory.
- Its turnover estimate is close to what the backtest measures.
- Its tearsheet renders, and you have read the warnings rather than the Sharpe.

And the honest expectation, from the platform's own prior: **roughly half the
backtest Sharpe survives contact with reality**, and most candidate signals do
not survive deflation at all. The deflated Sharpe failing is the normal outcome,
not a sign that something has gone wrong.
