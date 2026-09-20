"""The `quantlab` command line interface.

One rule governs every command here: a command that cannot do its job **fails
loudly**. Nothing in this CLI silently substitutes a fallback data source, quietly
skips a failed ingest, or returns a plausible-looking zero (spec section 13).
Commands whose implementation is scheduled for a later milestone exit non-zero
with the milestone named, rather than returning an empty result that could be
mistaken for "no data".
"""

from __future__ import annotations

import math
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Annotated, NoReturn

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from quantlab import __version__
from quantlab.config import get_settings
from quantlab.data.catalogue import SOURCES, get_source
from quantlab.health import Status, run_checks, source_availability
from quantlab.logging import configure_logging, get_logger
from quantlab.runtime import Heartbeat, read_heartbeat
from quantlab.validation import expected_max_sharpe

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.backtest.config import RunConfig
    from quantlab.backtest.panel import Panel
    from quantlab.backtest.results import BacktestResult

app = typer.Typer(
    name="quantlab",
    help="Multi-asset quantitative research and backtesting platform.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,  # never print secrets in a traceback
)
worker_app = typer.Typer(name="worker", help="Long-running worker process.", no_args_is_help=True)
scheduler_app = typer.Typer(name="scheduler", help="Job scheduler.", no_args_is_help=True)
data_app = typer.Typer(name="data", help="Data lake and catalogue.", no_args_is_help=True)
costs_app = typer.Typer(
    name="costs", help="Transaction costs, impact and capacity.", no_args_is_help=True
)
signals_app = typer.Typer(name="signals", help="The signal library.", no_args_is_help=True)
validate_app = typer.Typer(
    name="validate", help="Overfitting statistics and the trial registry.", no_args_is_help=True
)
portfolio_app = typer.Typer(
    name="portfolio", help="Covariance estimation and portfolio construction.", no_args_is_help=True
)
risk_app = typer.Typer(
    name="risk", help="Factor attribution, risk models and controls.", no_args_is_help=True
)
app.add_typer(worker_app)
app.add_typer(scheduler_app)
app.add_typer(data_app)
app.add_typer(costs_app)
app.add_typer(signals_app)
app.add_typer(validate_app)
app.add_typer(portfolio_app)
app.add_typer(risk_app)

console = Console()
err_console = Console(stderr=True)
log = get_logger("quantlab.cli")

_STATUS_STYLE = {Status.OK: "green", Status.WARN: "yellow", Status.FAIL: "red"}


def _not_yet(command: str, milestone: int, what: str) -> NoReturn:
    """Exit non-zero for a command whose implementation lands in a later milestone."""
    err_console.print(
        f"[red]`quantlab {command}` is not implemented yet.[/red]\n"
        f"It lands in Milestone {milestone}: {what}.\n"
        "Failing rather than returning an empty result, so this can never be "
        "mistaken for a successful run that found nothing."
    )
    raise typer.Exit(code=2)


@app.callback()
def main(
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
    ] = None,
    log_format: Annotated[str | None, typer.Option("--log-format", help="console or json.")] = None,
) -> None:
    settings = get_settings()
    configure_logging(level=log_level or settings.log_level, fmt=log_format or settings.log_format)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


@app.command()
def init() -> None:
    """Create the data lake directory structure. Idempotent."""
    layout = get_settings().layout
    layout.ensure()
    console.print(f"[green]ok[/green] data root ready at {layout.data_root}")
    for directory in layout.all_data_dirs():
        console.print(f"  {directory}")


@app.command()
def doctor(
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero on warnings as well as failures.")
    ] = False,
) -> None:
    """Report whether this installation can do useful work, and what it cannot do."""
    checks = run_checks()
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for check in checks:
        table.add_row(
            check.name,
            f"[{_STATUS_STYLE[check.status]}]{check.status.value}[/]",
            check.detail + (f"\n[dim]{check.hint}[/dim]" if check.hint else ""),
        )
    console.print(table)

    failed = [c for c in checks if c.status is Status.FAIL]
    warned = [c for c in checks if c.status is Status.WARN]
    if failed:
        raise typer.Exit(code=1)
    if warned and strict:
        raise typer.Exit(code=1)


@data_app.command("catalogue")
def data_catalogue(
    source: Annotated[
        str | None, typer.Argument(help="Show full detail for one source key.")
    ] = None,
    caveats: Annotated[
        bool, typer.Option("--caveats", help="Print every caveat for every source.")
    ] = False,
) -> None:
    """List catalogued data sources, their availability and their known biases."""
    if source is not None:
        spec = get_source(source)
        console.print(f"[bold]{spec.name}[/bold]  ([cyan]{spec.key}[/cyan])")
        console.print(f"  url            {spec.url}")
        console.print(f"  asset classes  {', '.join(a.value for a in spec.asset_classes)}")
        console.print(f"  datasets       {', '.join(spec.datasets)}")
        console.print(f"  point-in-time  {spec.pit_quality.value}")
        console.print(f"  updates        {spec.update_frequency}")
        console.print(f"  reliability    {spec.reliability}")
        console.print(f"  rate limit     {spec.rate_limit}")
        console.print(f"  licence        {spec.licence}")
        console.print(f"  needs key      {spec.key_setting or 'no'}")
        if spec.cross_check_only:
            console.print("  [yellow]cross-check only -- not for primary ingest[/yellow]")
        if not spec.usable_in_signal_path:
            console.print("  [yellow]restated data -- blocked from the signal path[/yellow]")
        console.print("\n  [bold]caveats[/bold]")
        for caveat in spec.caveats:
            console.print(f"   - {caveat}")
        for note in spec.notes:
            console.print(f"  [dim]note: {note}[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("key")
    table.add_column("status")
    table.add_column("asset classes")
    table.add_column("point-in-time")
    table.add_column("caveats", justify="right")
    for item in source_availability():
        table.add_row(
            item.spec.key,
            "[green]available[/green]" if item.available else "[yellow]no key[/yellow]",
            ",".join(a.value for a in item.spec.asset_classes),
            item.spec.pit_quality.value,
            str(len(item.spec.caveats)),
        )
    console.print(table)
    console.print(f"[dim]{len(SOURCES)} sources. `quantlab data catalogue <key>` for detail.[/dim]")

    if caveats:
        for item in source_availability():
            console.print(f"\n[bold]{item.spec.key}[/bold]")
            for caveat in item.spec.caveats:
                console.print(f"  - {caveat}")


@data_app.command("ingest")
def data_ingest(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Ingest plan YAML.")] = None,
    fetcher: Annotated[
        list[str] | None,
        typer.Option("--fetcher", "-f", help="Run only these fetchers. Repeatable."),
    ] = None,
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental",
            help="Resume from what the lake already holds, minus each job's overlap window.",
        ),
    ] = False,
    end: Annotated[
        str | None, typer.Option("--end", help="Ingest up to this ISO date (default: today).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the windows that would be fetched.")
    ] = False,
) -> None:
    """Pull data into the lake.

    Exits non-zero if any job failed. Jobs skipped for a missing API key are not
    failures -- the platform is required to run with none -- but they are listed.
    """
    import datetime as date_module

    from quantlab.data.ingest import load_plan, run_plan

    settings = get_settings()
    path = config or (settings.layout.configs / "ingest.yaml")
    if not path.exists():
        err_console.print(f"[red]no ingest plan at {path}[/red]")
        raise typer.Exit(code=2)

    plan = load_plan(path).select(fetcher)
    result = run_plan(
        plan,
        settings=settings,
        incremental=incremental,
        end=date_module.date.fromisoformat(end) if end else None,
        dry_run=dry_run,
    )

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("fetcher")
    table.add_column("rows", justify="right")
    table.add_column("window")
    table.add_column("secs", justify="right")
    table.add_column("status", overflow="fold")
    for job in result.jobs:
        if not job.ok:
            status = f"[red]failed[/red] {job.error}"
        elif job.skipped_reason:
            status = f"[yellow]skipped[/yellow] {job.skipped_reason}"
        else:
            status = "[green]ok[/green]"
        table.add_row(
            job.fetcher,
            f"{job.rows:,}",
            f"{job.start.date()} → {job.end.date()}",
            f"{job.seconds:.1f}",
            status,
        )
    console.print(table)
    console.print(
        f"\n[bold]{result.rows:,}[/bold] rows from {len(result.jobs)} jobs "
        f"in {(result.finished_at - result.started_at).total_seconds():.1f}s"
    )

    if result.failures:
        err_console.print(
            f"[red]{len(result.failures)} job(s) failed.[/red] Nothing was substituted "
            "for the missing data; fix the cause and re-run."
        )
        raise typer.Exit(code=1)


@data_app.command("status")
def data_status() -> None:
    """Show what the lake holds and how stale it is."""
    from quantlab.data.store import Store

    stats = Store(get_settings().layout).stats()
    if not stats:
        console.print("[yellow]the lake is empty[/yellow] -- run `quantlab data ingest`")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("source", "dataset", "rows", "symbols", "span", "newest", "age", "size"):
        table.add_column(
            column, justify="right" if column in {"rows", "symbols", "size"} else "left"
        )
    for stat in stats:
        age = stat.staleness
        if age is None:
            age_text = "-"
        else:
            days = age.total_seconds() / 86400
            colour = "green" if days < 3 else "yellow" if days < 14 else "red"
            age_text = f"[{colour}]{days:.1f}d[/{colour}]"
        span = (
            f"{stat.first_as_of:%Y-%m-%d} → {stat.last_as_of:%Y-%m-%d}"
            if stat.first_as_of and stat.last_as_of
            else "-"
        )
        table.add_row(
            stat.source,
            stat.dataset,
            f"{stat.rows:,}",
            f"{stat.symbols:,}",
            span,
            f"{stat.last_known_at:%Y-%m-%d}" if stat.last_known_at else "-",
            age_text,
            f"{stat.bytes / 1e6:.1f} MB",
        )
    console.print(table)
    console.print(
        "[dim]`age` is time since the newest observation became knowable, not since "
        "we downloaded it.[/dim]"
    )


@data_app.command("fetchers")
def data_fetchers() -> None:
    """List every registered fetcher and whether it can run right now."""
    from quantlab.data.ingest import iter_fetchers

    settings = get_settings()
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("fetcher", "dataset", "asset class", "point-in-time", "status"):
        table.add_column(column)
    for name, cls in iter_fetchers():
        available, reason = cls(settings=settings).availability()
        table.add_row(
            name,
            cls.dataset,
            cls.asset_class.value,
            cls.spec.pit_quality.value,
            "[green]ready[/green]" if available else f"[yellow]{reason}[/yellow]",
        )
    console.print(table)


@data_app.command("query")
def data_query(
    sql: Annotated[str, typer.Argument(help="SQL over the lake.")],
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of", help="Run point-in-time: only data knowable at this date is visible."
        ),
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", "-d", help="Datasets the query reads (required with --as-of)."),
    ] = None,
) -> None:
    """Query the lake.

    Without --as-of this sees everything, including rows that were not knowable on
    any given date. That is fine for inspection and wrong for research: pass
    --as-of to run inside the point-in-time sandbox.
    """
    from quantlab.data.store import Store

    store = Store(get_settings().layout)
    if as_of is None:
        console.print(store.sql(sql))
        console.print("[dim]not point-in-time -- pass --as-of for research queries[/dim]")
        return

    if not dataset:
        err_console.print(
            "[red]--as-of requires --dataset[/red]: the point-in-time sandbox holds "
            "nothing by default, so it must be told which datasets to materialise."
        )
        raise typer.Exit(code=2)
    console.print(store.as_of(as_of).sql(sql, datasets=dataset))


def _money(value: float) -> str:
    """Currency with adaptive units, so a capacity curve spanning nine orders of
    magnitude does not print a column of zeroes."""
    for threshold, suffix in ((1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.1f}{suffix}"
    return f"${value:,.0f}"


@costs_app.command("estimate")
def costs_estimate(
    notional: Annotated[float, typer.Option("--notional", "-n", help="Order size in currency.")],
    adv: Annotated[float, typer.Option("--adv", help="Average daily traded value.")],
    volatility: Annotated[
        float, typer.Option("--vol", help="Daily return volatility as a decimal (0.02 = 2%).")
    ] = 0.02,
    spread_bps: Annotated[float, typer.Option("--spread", help="Quoted spread in bps.")] = 2.0,
    asset_class: Annotated[
        str, typer.Option("--asset-class", "-a", help="equity, futures, fx, crypto or credit.")
    ] = "equity",
    horizon_days: Annotated[
        float, typer.Option("--horizon", help="Days over which the order is worked.")
    ] = 1.0,
    compare_flat: Annotated[
        float | None,
        typer.Option("--compare-flat", help="Also show what a flat N-bps model would claim."),
    ] = None,
) -> None:
    """Cost one order, decomposed.

    The decomposition is the diagnostic: a strategy killed by spread needs a slower
    rebalance, one killed by impact needs less size, one killed by borrow needs a
    different short book. A single number tells you none of that.
    """
    from quantlab.costs import Instrument, Order, TransactionCostModel
    from quantlab.costs.base import FlatBpsCostModel
    from quantlab.data.catalogue import AssetClass

    try:
        klass = AssetClass(asset_class.lower())
    except ValueError:
        err_console.print(
            f"[red]unknown asset class {asset_class!r}[/red]; "
            f"known: {', '.join(a.value for a in AssetClass)}"
        )
        raise typer.Exit(code=2) from None

    instrument = Instrument(
        symbol="ORDER",
        asset_class=klass,
        adv_notional=adv,
        volatility_daily=volatility,
        spread_bps=spread_bps,
    )
    model = TransactionCostModel.for_asset_class(klass)
    breakdown = model.estimate(Order("ORDER", notional, horizon_days=horizon_days), instrument)

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("component")
    table.add_column("bps", justify="right")
    table.add_column("currency", justify="right")
    for label, value in (
        ("spread (half)", breakdown.spread_bps),
        ("impact paid", breakdown.impact_bps),
        ("  temporary", breakdown.temporary_impact_bps),
        ("  permanent (half paid)", breakdown.permanent_impact_bps / 2),
        ("commission", breakdown.commission_bps),
    ):
        table.add_row(label, f"{value:.3f}", f"{value * 1e-4 * notional:,.0f}")
    table.add_row(
        "[bold]total[/bold]",
        f"[bold]{breakdown.total_bps:.3f}[/bold]",
        f"[bold]{breakdown.currency(notional):,.0f}[/bold]",
    )
    console.print(table)

    participation = breakdown.detail["participation"]
    round_trip = model.round_trip_bps(
        Order("ORDER", notional, horizon_days=horizon_days), instrument
    )
    console.print(
        f"\nparticipation {participation:.2%} of ADV, "
        f"round trip {round_trip:.2f} bps over {horizon_days:g} day(s)"
    )
    if breakdown.detail.get("extrapolating"):
        console.print(
            "[yellow]beyond the square-root law's calibrated range (~10% of ADV); "
            "real impact is likely HIGHER than shown[/yellow]"
        )

    if compare_flat is not None:
        flat = FlatBpsCostModel(compare_flat, acknowledge_unrealistic=True)
        flat_bps = flat.estimate(Order("ORDER", notional), instrument).total_bps
        ratio = breakdown.total_bps / flat_bps if flat_bps else float("inf")
        console.print(
            f"\n[dim]a flat {compare_flat:.1f} bp model would claim "
            f"{flat_bps * 1e-4 * notional:,.0f} -- understating by {ratio:.1f}x. "
            "Flat costs are size-blind, which is why they imply unlimited capacity.[/dim]"
        )


@costs_app.command("capacity")
def costs_capacity(
    alpha_bps: Annotated[float, typer.Option("--alpha", help="Gross alpha in bps per rebalance.")],
    turnover: Annotated[
        float, typer.Option("--turnover", help="One-way turnover per rebalance, as a fraction.")
    ],
    rebalances: Annotated[float, typer.Option("--rebalances", help="Rebalances per year.")] = 12.0,
    names: Annotated[
        int, typer.Option("--names", help="Instruments in the traded universe.")
    ] = 100,
    adv: Annotated[float, typer.Option("--adv", help="Average daily value per name.")] = 50e6,
    volatility: Annotated[float, typer.Option("--vol", help="Daily volatility.")] = 0.02,
    spread_bps: Annotated[float, typer.Option("--spread", help="Quoted spread in bps.")] = 5.0,
    horizon_days: Annotated[
        float, typer.Option("--horizon", help="Days over which each rebalance is worked.")
    ] = 1.0,
    holding_bps: Annotated[
        float, typer.Option("--holding", help="Annual financing and borrow, in bps.")
    ] = 0.0,
) -> None:
    """Break-even AUM: where the strategy's own trading eats its edge.

    Spec section 5: a strategy with no capacity number is not a finished strategy.
    """
    from quantlab.costs import CapacityModel, UniverseLiquidity

    universe = UniverseLiquidity.equal_weight(
        [f"N{i}" for i in range(names)],
        adv_notional=adv,
        volatility_daily=volatility,
        spread_bps=spread_bps,
    )
    result = CapacityModel().solve(
        universe,
        gross_alpha_bps_per_rebalance=alpha_bps,
        turnover_per_rebalance=turnover,
        rebalances_per_year=rebalances,
        horizon_days=horizon_days,
        holding_cost_bps_annual=holding_bps,
    )

    if not result.is_viable:
        err_console.print(f"[red]{result.summary()}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{result.summary()}[/bold]\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("AUM", justify="right")
    table.add_column("gross bp/yr", justify="right")
    table.add_column("cost bp/yr", justify="right")
    table.add_column("net bp/yr", justify="right")
    # Show the decade or so either side of the break-even. The full curve spans
    # $1k to $1tn, and printing the part where cost rounds to zero tells no one
    # anything.
    break_even = result.break_even_aum or 0.0
    rows = result.curve.filter(
        (pl.col("aum") >= break_even / 1e3) & (pl.col("aum") <= break_even * 30)
    )
    for row in rows.iter_rows(named=True):
        net = row["net_bps_annual"]
        table.add_row(
            _money(row["aum"]),
            f"{row['gross_bps_annual']:.0f}",
            f"{row['cost_bps_annual']:.1f}",
            f"[{'green' if net > 0 else 'red'}]{net:.0f}[/]",
        )
    console.print(table)
    console.print(
        "[dim]Equal-weighted homogeneous universes overstate capacity: a real "
        "universe's thin names cost disproportionately more.[/dim]"
    )


@validate_app.command("trials")
def validate_trials(
    family: Annotated[
        str | None, typer.Argument(help="Show the individual trials in one family.")
    ] = None,
) -> None:
    """Show what the platform has counted.

    Every backtest records itself here, and this count is what deflates your
    Sharpe ratios. Spec section 7: manual honesty about trial counts does not work,
    so the platform counts whether you want it to or not.
    """
    from quantlab.validation import TrialRegistry

    registry = TrialRegistry(get_settings().layout.state)
    sets = registry.summary()
    if not sets:
        console.print(
            "[yellow]no trials recorded yet[/yellow] -- run a backtest and it will record itself"
        )
        return

    if family is not None:
        trial_set = registry.family(family)
        if trial_set.count == 0:
            err_console.print(
                f"[red]no trials in family {family!r}[/red]; known families: "
                f"{', '.join(registry.families())}"
            )
            raise typer.Exit(code=2)
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        for column in ("config", "name", "Sharpe", "bars", "recorded"):
            table.add_column(column, justify="right" if column in {"Sharpe", "bars"} else "left")
        for item in sorted(trial_set.trials, key=lambda t: -t.sharpe_annual):
            table.add_row(
                item.config_hash,
                item.name,
                f"{item.sharpe_annual:.2f}",
                f"{item.observations:,}",
                f"{item.recorded_at:%Y-%m-%d %H:%M}",
            )
        console.print(table)
        console.print(f"\n[dim]{trial_set.describe()}[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("family")
    table.add_column("trials", justify="right")
    table.add_column("best Sharpe", justify="right")
    table.add_column("deflation bar", justify="right")
    for trial_set in sorted(sets, key=lambda s: -s.count):
        best = trial_set.best
        bar = expected_max_sharpe(max(1, trial_set.count), trial_set.sharpe_variance()) * math.sqrt(
            252
        )
        table.add_row(
            trial_set.family,
            str(trial_set.count),
            f"{best.sharpe_annual:.2f}" if best else "-",
            f"{bar:.2f}",
        )
    console.print(table)
    console.print(
        "[dim]`deflation bar` is the annualised Sharpe the best of that many "
        "worthless strategies would be expected to show. Beating it is the minimum "
        "for a result to mean anything.[/dim]"
    )


@validate_app.command("library")
def validate_library(
    min_years: Annotated[
        float, typer.Option("--min-years", help="Ignore families with a shorter sample.")
    ] = 1.0,
) -> None:
    """Empirical-Bayes luck adjustment across every signal family.

    The most uncomfortable number the platform produces. If the cross-sectional
    variance of your t-statistics is near 1, the spread of your results is exactly
    what chance produces and the correct conclusion is that nothing has been found.
    """
    from quantlab.validation import TrialRegistry, luck_adjust, t_statistic

    registry = TrialRegistry(get_settings().layout.state)
    statistics: dict[str, float] = {}
    for trial_set in registry.summary():
        best = trial_set.best
        if best is None:
            continue
        years = best.observations / 252.0
        if years < min_years:
            continue
        statistics[trial_set.family] = t_statistic(best.sharpe_annual, years)

    if len(statistics) < 5:
        err_console.print(
            f"[yellow]only {len(statistics)} families with at least {min_years} "
            "year(s) of data.[/yellow] The luck adjustment needs at least 5 to say "
            "anything; below that it tells you about your sample size, not your "
            "library. Build more signals first."
        )
        raise typer.Exit(code=2)

    adjustment = luck_adjust(statistics)
    console.print(adjustment.describe())
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("family")
    table.add_column("raw t", justify="right")
    table.add_column("shrunk t", justify="right")
    table.add_column("survives", justify="right")
    for name, raw, shrunk in adjustment.ranked():
        survives = abs(shrunk) >= 2.0
        table.add_row(
            name,
            f"{raw:.2f}",
            f"{shrunk:.2f}",
            "[green]yes[/green]" if survives else "[red]no[/red]",
        )
    console.print(table)


@validate_app.command("sharpe")
def validate_sharpe(
    sharpe: Annotated[float, typer.Argument(help="Observed annualised Sharpe ratio.")],
    years: Annotated[float, typer.Option("--years", help="Length of the sample.")],
    trials: Annotated[int, typer.Option("--trials", help="Configurations tried to find it.")] = 1,
    skew: Annotated[float, typer.Option("--skew", help="Return skewness.")] = 0.0,
    kurtosis: Annotated[float, typer.Option("--kurtosis", help="Excess kurtosis.")] = 0.0,
    trial_sharpe_sd: Annotated[
        float, typer.Option("--trial-sd", help="Annualised SD of the trials' Sharpes.")
    ] = 0.7,
) -> None:
    """Deflate a Sharpe ratio by hand, for a result that came from elsewhere.

    For a published number, or a backtest run outside the platform. Runs inside it
    are deflated automatically against the recorded trial count.
    """
    from quantlab.validation import (
        HaircutSchedule,
        deflated_sharpe_ratio,
        minimum_backtest_length,
        probabilistic_sharpe_ratio,
    )

    bars = max(2, int(years * 252))
    per_bar = sharpe / math.sqrt(252)
    variance = (trial_sharpe_sd / math.sqrt(252)) ** 2

    probabilistic = probabilistic_sharpe_ratio(
        per_bar, observations=bars, skewness=skew, excess_kurtosis=kurtosis
    )
    deflated = deflated_sharpe_ratio(
        per_bar,
        observations=bars,
        trials=trials,
        sharpe_variance=variance,
        skewness=skew,
        excess_kurtosis=kurtosis,
    )
    bar = expected_max_sharpe(trials, variance) * math.sqrt(252)
    needed = minimum_backtest_length(abs(sharpe) or 1e-6, trials)
    haircuts = HaircutSchedule()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("observed Sharpe", f"{sharpe:.2f} over {years:.1f} years")
    table.add_row("after haircuts", f"{haircuts.apply(sharpe):.2f}  ({haircuts.describe()})")
    table.add_row("trials", f"{trials}")
    table.add_row("deflation bar", f"{bar:.2f} annualised")
    table.add_row("probabilistic Sharpe", f"{probabilistic:.3f}")
    table.add_row("deflated Sharpe", f"{deflated:.3f}")
    table.add_row("sample needed", f"{needed:.1f} years")
    console.print(table)

    if deflated >= 0.95:
        console.print("\n[green]survives deflation at the 0.95 threshold.[/green]")
    else:
        console.print(
            f"\n[red]does not survive deflation.[/red] After {trials} trial(s), a "
            f"Sharpe of {sharpe:.2f} over {years:.1f} years is consistent with having "
            "found the luckiest member of the search."
        )
        raise typer.Exit(code=1)


@signals_app.command("list")
def signals_list(
    tier: Annotated[int | None, typer.Option("--tier", help="Show one tier only.")] = None,
) -> None:
    """The signal library, ordered by evidence quality.

    Tiering is by how well the effect is supported, not by how interesting it is.
    Tier 3 signals are implemented so a decayed effect can be re-tested on current
    data -- with the evidence that it decayed attached.
    """
    from quantlab.signals import SIGNAL_REGISTRY

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("tier", "signal", "asset class", "evidence", "output", "turnover"):
        table.add_column(column, justify="right" if column in {"tier", "turnover"} else "left")

    colour = {"strong": "green", "mixed": "yellow", "decayed": "red"}
    for name, cls in sorted(SIGNAL_REGISTRY.items(), key=lambda kv: (kv[1].spec.tier, kv[0])):
        spec = cls.spec
        if tier is not None and spec.tier != tier:
            continue
        table.add_row(
            str(spec.tier),
            name,
            spec.asset_class.value,
            f"[{colour[spec.evidence.value]}]{spec.evidence.value}[/]",
            spec.output.value.replace("_", " "),
            f"{spec.expected_turnover_annual:.0%}",
        )
    console.print(table)
    console.print(
        "[dim]`quantlab signals show <name>` for the reference and the failure "
        "modes. Every signal declares how it is known to go wrong.[/dim]"
    )


@signals_app.command("show")
def signals_show(
    name: Annotated[str, typer.Argument(help="Signal name, as `signals list` prints it.")],
) -> None:
    """Everything a reader of this signal's Sharpe ratio should know about it."""
    from quantlab.signals import get_signal

    try:
        spec = get_signal(name).spec
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    console.print(f"[bold]{spec.name}[/bold]  (tier {spec.tier}, {spec.evidence.value} evidence)")
    console.print(f"  asset class   {spec.asset_class.value}")
    console.print(f"  output        {spec.output.value.replace('_', ' ')}")
    console.print(f"  datasets      {', '.join(spec.required_datasets)}")
    console.print(
        f"  rebalance     {spec.rebalance.value}, ~{spec.expected_turnover_annual:.0%} "
        "turnover a year"
    )
    console.print(f"  warm-up       {spec.warmup_days} bars")
    console.print(f"\n  [bold]reference[/bold]\n   {spec.reference}")
    console.print(f"\n  [bold]known failure modes[/bold]\n   {spec.known_failure_modes}")
    if spec.notes:
        console.print(f"\n  [dim]{spec.notes}[/dim]")
    if spec.evidence.value == "decayed":
        console.print(
            "\n[red]This effect has decayed since publication.[/red] It is implemented "
            "so it can be re-tested on current data, not because it is expected to work."
        )


@signals_app.command("run")
def signals_run(
    name: Annotated[str, typer.Argument(help="Signal to compute.")],
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")],
    symbols: Annotated[
        list[str] | None, typer.Option("--symbol", "-s", help="Instrument. Repeatable.")
    ] = None,
) -> None:
    """Compute a signal's current scores, through a point-in-time snapshot."""
    from quantlab.data.store import Store
    from quantlab.signals import get_signal
    from quantlab.signals.base import SignalUnavailableError

    try:
        signal_class = get_signal(name)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    store = Store(get_settings().layout)
    snapshot = store.as_of(as_of)
    universe = symbols or list(store.symbols(signal_class.spec.required_datasets[0]))
    if not universe:
        err_console.print(
            "[red]no instruments[/red] -- pass --symbol, or ingest the datasets this "
            f"signal needs: {', '.join(signal_class.spec.required_datasets)}"
        )
        raise typer.Exit(code=2)

    try:
        scores = signal_class().compute(snapshot, universe)
    except SignalUnavailableError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if scores.height == 0:
        console.print(
            f"[yellow]no scores at {as_of}[/yellow] -- every instrument is inside the "
            f"{signal_class.spec.warmup_days}-bar warm-up window, or has no data."
        )
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("symbol")
    table.add_column("score", justify="right")
    for row in scores.sort("score", descending=True).iter_rows(named=True):
        colour = "green" if row["score"] > 0 else "red"
        table.add_row(row["symbol"], f"[{colour}]{row['score']:+.4f}[/]")
    console.print(table)
    console.print(
        f"[dim]{signal_class.spec.output.value.replace('_', ' ')}; point-in-time as of "
        f"{snapshot.as_of:%Y-%m-%d}. Scores are not weights -- portfolio construction "
        "owns those.[/dim]"
    )


def _load_and_run(
    config: Path, *, record_trial: bool = True
) -> tuple[RunConfig, Panel, BacktestResult]:
    """Config to (run config, panel, result). Shared by `backtest` and `report`.

    Both commands must build the panel identically -- a tearsheet describing a
    slightly different run than the one the user just looked at would be worse
    than no tearsheet.
    """
    from quantlab.backtest import Panel, VectorisedBacktest
    from quantlab.backtest.config import load_run_config
    from quantlab.backtest.prepare import panel_frame
    from quantlab.data.store import Store

    if not config.exists():
        err_console.print(f"[red]no run config at {config}[/red]")
        raise typer.Exit(code=2)

    run = load_run_config(config)
    if not run.weights_path.exists():
        err_console.print(
            f"[red]no weights file at {run.weights_path}[/red]\n"
            "A backtest needs target weights: a parquet or csv with columns "
            "symbol, as_of, weight, dated on the bar they are COMPUTED on -- the "
            "engine applies the execution lag itself.\n"
            "Generate them from the signal library (`quantlab signals list`), or "
            "write the file directly."
        )
        raise typer.Exit(code=2)

    snapshot = Store(get_settings().layout).as_of(run.as_of)
    bars = snapshot.frame(
        run.dataset, symbols=list(run.symbols), start=run.start, source=run.source
    )
    if bars.height == 0:
        err_console.print(
            f"[red]no data for those symbols at as-of {run.as_of}[/red]\n"
            "Run `quantlab data ingest` first, or widen the window."
        )
        raise typer.Exit(code=2)

    keep = [
        c
        for c in ("symbol", "as_of", "close", "open", "volume", "high", "low")
        if c in bars.columns
    ]
    frame = panel_frame(
        bars.select(keep),
        window=run.liquidity_window,
        spread_bps=run.spread_bps,
        estimate_spreads=run.estimate_spreads,
    )
    panel = Panel.from_frame(frame, timing=run.engine.execution, staleness=run.engine.staleness)

    weights = (
        pl.read_parquet(run.weights_path)
        if run.weights_path.suffix == ".parquet"
        else pl.read_csv(run.weights_path, try_parse_dates=True)
    )
    try:
        result = VectorisedBacktest(run.engine, run.cost_model()).run(
            panel, weights, name=run.name, record_trial=record_trial
        )
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
    return run, panel, result


@app.command()
def backtest(
    config: Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the equity curve here.")
    ] = None,
) -> None:
    """Run a backtest from a config.

    Reads market data through a point-in-time snapshot fixed by the config's
    `as_of`, so re-running later sees the same data rather than whatever the lake
    has learned since.
    """
    run, _panel, result = _load_and_run(config)

    console.print(result.summary())
    console.print()
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("cost component")
    table.add_column("bp/yr", justify="right")
    for component, value in sorted(
        result.cost_decomposition_bps_annual().items(), key=lambda kv: -kv[1]
    ):
        table.add_row(component, f"{value:.1f}")
    console.print(table)
    console.print(
        f"\n[dim]point-in-time as of {run.as_of}; "
        f"{result.data_quality['symbols']} instruments, "
        f"{result.data_quality['forward_filled_cells']:,} forward-filled cells[/dim]"
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.curve.write_parquet(output)
        console.print(f"[green]wrote[/green] {output}")


@validate_app.command("anomalies")
def validate_anomalies(
    t_stat: Annotated[
        float | None,
        typer.Option("--t-stat", help="Your own t-statistic, to place against the literature."),
    ] = None,
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")] = "2026-01-01",
) -> None:
    """The published anomaly literature, and where your result sits in it.

    Chen & Zimmermann catalogued every cross-sectional equity predictor they could
    find in a published paper, with the t-statistic the original authors reported.
    That distribution is the trial count of the whole field, and it is visibly
    truncated at the significance threshold: almost nothing below |t| = 2 is ever
    published, so the sample says nothing about how many signals were tried and
    discarded.

    The practical consequence is that a t-statistic is not evidence on its own.
    Harvey, Liu and Zhu argue a new anomaly needs |t| above roughly 3.0 to survive
    the multiple testing the literature has already done; a quarter of published
    predictors do not clear that bar.
    """
    from quantlab.data.store import Store

    snapshot = Store(get_settings().layout).as_of(as_of)
    frame = snapshot.frame("anomaly_catalogue")
    if frame.height == 0:
        err_console.print(
            "[red]no anomaly catalogue in the lake[/red] -- ingest it first:\n"
            "  quantlab data ingest -f open_asset_pricing.anomaly_catalogue"
        )
        raise typer.Exit(code=2)

    predictors = frame.filter(pl.col("category") == "Predictor")
    stats = predictors.select(pl.col("published_t_stat")).drop_nulls()["published_t_stat"]
    if stats.len() == 0:
        err_console.print("[red]the catalogue carries no published t-statistics[/red]")
        raise typer.Exit(code=2)

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("the published literature")
    table.add_column("", justify="right")
    table.add_row("catalogued signals", f"{frame.height:,}")
    table.add_row("  of which predictors", f"{predictors.height:,}")
    table.add_row(
        "  of which placebos", f"{frame.filter(pl.col('category') == 'Placebo').height:,}"
    )
    values = np.abs(stats.to_numpy().astype(float))
    median = float(np.median(values))
    low, high = (float(x) for x in np.percentile(values, [10, 90]))
    below_two = float((values < 2.0).mean())
    below_three = float((values < 3.0).mean())

    table.add_row("published t-statistics", f"{stats.len():,}")
    table.add_row("  median", f"{median:.2f}")
    table.add_row("  10th / 90th percentile", f"{low:.2f} / {high:.2f}")
    table.add_row("  share below |t| = 2.0", f"{below_two:.1%}")
    table.add_row("  share below |t| = 3.0", f"{below_three:.1%}")
    console.print(table)

    console.print(
        "\n[dim]Almost nothing below |t| = 2 appears, because that is where journals "
        "stop accepting papers. The distribution therefore describes what survived "
        "publication, not what was tried.[/dim]"
    )

    if t_stat is not None:
        share = float((values < abs(t_stat)).mean())
        console.print(
            f"\nA t-statistic of [bold]{t_stat:.2f}[/bold] is larger than "
            f"[bold]{share:.0%}[/bold] of published predictors."
        )
        if abs(t_stat) < 3.0:
            console.print(
                "[yellow]Below the |t| > 3.0 bar[/yellow] that Harvey, Liu and Zhu "
                "argue a new anomaly needs to survive the multiple testing the "
                f"literature has already done. {below_three:.0%} of "
                "published predictors are below it too, which is the problem rather "
                "than the reassurance."
            )
        else:
            console.print(
                "[green]Clears the |t| > 3.0 bar.[/green] That is necessary and not "
                "sufficient: it says nothing about how many configurations you tried, "
                "which is what `quantlab validate trials` counts."
            )


@app.command()
def report(
    config: Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the tearsheet here (.html, .md or .txt)."),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="text, markdown or html.")] = "text",
    sources: Annotated[
        list[str] | None,
        typer.Option("--source", help="Catalogue source key whose caveats apply. Repeatable."),
    ] = None,
) -> None:
    """Run a backtest and render its tearsheet.

    The tearsheet refuses to render a Sharpe ratio without its deflated value and
    trial count, and refuses to render at all without a capacity estimate (spec
    section 13). Both are derived from the run itself rather than supplied, so
    there is nothing to leave out.

    Panels are ordered worst-first, so a strategy that fails deflation says so
    above its equity curve rather than beneath it.
    """
    from quantlab.reporting import Tearsheet, TearsheetError, caveats_for, sources_behind
    from quantlab.reporting.capacity import capacity_for
    from quantlab.reporting.render import to_html, to_markdown, to_text

    if fmt not in {"text", "markdown", "html"}:
        err_console.print(f"[red]unknown format {fmt!r}[/red]; use text, markdown or html")
        raise typer.Exit(code=2)

    run, panel, result = _load_and_run(config)

    # A superset when the run did not record its source: better to show a caveat
    # that did not apply than to hide one that did.
    keys = sources or ([run.source] if run.source else list(sources_behind([run.dataset])))
    try:
        caveats = caveats_for(keys)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    try:
        capacity = capacity_for(result, panel)
        sheet = Tearsheet.build(result, capacity=capacity, caveats=caveats)
    except (TearsheetError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    rendered = {"text": to_text, "markdown": to_markdown, "html": to_html}[fmt](sheet)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        console.print(f"[green]wrote[/green] {output}  ({len(rendered):,} bytes)")
    else:
        console.print(rendered, highlight=False, markup=False)

    # Exit non-zero when the tearsheet says the result is not evidence, so a
    # scripted sweep cannot treat a disqualified strategy as a success.
    if sheet.is_disqualified:
        raise typer.Exit(code=1)


paper_app = typer.Typer(
    name="paper", help="Paper trading: the loop, the book, and decay.", no_args_is_help=True
)
app.add_typer(paper_app)


@paper_app.command("run")
def paper_run(
    signal: Annotated[str, typer.Option("--signal", "-s", help="Signal to trade.")],
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")],
    symbols: Annotated[
        list[str] | None, typer.Option("--symbol", help="Instrument. Repeatable.")
    ] = None,
    strategy: Annotated[
        str | None, typer.Option("--strategy", help="Book name. Defaults to the signal.")
    ] = None,
    equity: Annotated[
        float, typer.Option("--equity", help="Starting equity, first run only.")
    ] = 1e6,
    gross: Annotated[float, typer.Option("--gross", help="Target gross exposure.")] = 1.0,
) -> None:
    """Run one paper-trading cycle.

    No orders leave this machine. The platform simulates execution and leaves a
    clean interface where a live adapter would attach (spec section 1); fills are
    charged the backtest's own cost model, because a paper book that filled at
    mid would beat its own backtest for no reason.

    The cycle is idempotent per as-of date: a repeat is refused rather than
    absorbed, so re-running after a failure is safe.
    """
    from quantlab.data.store import Store
    from quantlab.paper import PaperBroker, PaperState, PaperTradingLoop
    from quantlab.signals import get_signal

    try:
        signal_class = get_signal(signal)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    layout = get_settings().layout
    book = strategy or signal
    state = PaperState.for_strategy(layout.state, book)

    store = Store(layout)
    universe = symbols or list(store.symbols(signal_class.spec.required_datasets[0]))
    if not universe:
        err_console.print("[red]no instruments[/red] -- pass --symbol, or ingest first")
        raise typer.Exit(code=2)

    # Resume the book from its last recorded state rather than starting fresh.
    previous = state.latest()
    broker = PaperBroker(cash=float(previous["cash"]) if previous else equity)
    if previous:
        broker.shares = {p["symbol"]: float(p["shares"]) for p in previous["positions"]}

    loop = PaperTradingLoop(signal_class(), broker, state, symbols=universe, gross_exposure=gross)
    from quantlab.data.store import _as_utc

    outcome = loop.run_once(store, _as_utc(as_of, boundary="end"))

    if outcome.skipped:
        console.print(f"[yellow]skipped[/yellow] -- {outcome.reason}")
        raise typer.Exit(code=0)

    record = outcome.record
    console.print(
        f"[bold]{book}[/bold] at {record.as_of[:10]}: equity "
        f"${record.equity:,.0f}, gross {record.gross_exposure:.2f}x, net "
        f"{record.net_exposure:+.2f}x, {len(record.positions)} positions"
    )
    if record.fills:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("symbol")
        table.add_column("shares", justify="right")
        table.add_column("price", justify="right")
        table.add_column("cost bp", justify="right")
        for fill in sorted(record.fills, key=lambda f: -abs(float(f["shares"]))):
            table.add_row(
                str(fill["symbol"]),
                f"{float(fill['shares']):+,.1f}",
                f"{float(fill['reference_price']):,.2f}",
                f"{float(fill['slippage_bps']):.1f}",
            )
        console.print(table)
        console.print(
            f"\ntraded ${record.traded_notional:,.0f}, costs ${record.costs_paid:,.0f} "
            f"({record.costs_paid / max(record.traded_notional, 1) / 1e-4:.1f} bp)"
        )
    if record.unfilled:
        console.print(
            f"[yellow]{len(record.unfilled)} target(s) unfilled[/yellow]: "
            + ", ".join(f"{u['symbol']} ({u['reason']})" for u in record.unfilled)
        )


@paper_app.command("book")
def paper_book(
    strategy: Annotated[str, typer.Option("--strategy", "-s", help="Book name.")],
) -> None:
    """Show a paper book's current state and its history."""
    from quantlab.paper import PaperState

    state = PaperState.for_strategy(get_settings().layout.state, strategy)
    history = state.history()
    if not history:
        err_console.print(
            f"[red]no paper history for {strategy!r}[/red] -- run a cycle first:\n"
            f"  quantlab paper run --signal {strategy} --as-of <date>"
        )
        raise typer.Exit(code=2)

    latest = history[-1]
    console.print(
        f"[bold]{strategy}[/bold]: {len(history)} cycles, "
        f"{latest['as_of'][:10]} latest, equity ${latest['equity']:,.0f}"
    )
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("symbol")
    table.add_column("shares", justify="right")
    table.add_column("mark", justify="right")
    table.add_column("weight", justify="right")
    for position in sorted(latest["positions"], key=lambda p: -abs(float(p["weight"]))):
        table.add_row(
            str(position["symbol"]),
            f"{float(position['shares']):+,.1f}",
            f"{float(position['mark']):,.2f}",
            f"{float(position['weight']):+.2%}",
        )
    console.print(table)
    total_costs = sum(float(r["costs_paid"]) for r in history)
    total_traded = sum(float(r["traded_notional"]) for r in history)
    console.print(
        f"\n[dim]cumulative: traded ${total_traded:,.0f}, costs ${total_costs:,.0f}"
        f"{f' ({total_costs / total_traded / 1e-4:.1f} bp)' if total_traded else ''}[/dim]"
    )


@paper_app.command("decay")
def paper_decay(
    strategy: Annotated[str, typer.Option("--strategy", "-s", help="Book name.")],
    backtest_sharpe: Annotated[
        float, typer.Option("--backtest-sharpe", help="The Sharpe this was launched on.")
    ],
    haircut: Annotated[
        float, typer.Option("--haircut", help="Fraction of backtest Sharpe expected live.")
    ] = 0.53,
) -> None:
    """Has the strategy decayed, or is it too early to tell?

    The second question usually has an answer and is the one people skip. The
    comparison is against the *haircut* backtest Sharpe, not the raw one:
    delivering half the backtest is what the platform's prior predicts, so
    comparing against the printed number would declare decay on every strategy
    that behaved exactly as expected.
    """
    from quantlab.paper import PaperState, assess_decay

    state = PaperState.for_strategy(get_settings().layout.state, strategy)
    _dates, equity = state.equity_curve()
    if len(equity) < 3:
        err_console.print(
            f"[red]{len(equity)} cycle(s) recorded[/red] -- a decay assessment needs "
            "a return series, which needs at least three."
        )
        raise typer.Exit(code=2)

    values = np.array(equity, dtype=float)
    returns = np.diff(values) / values[:-1]
    # Inferred from the cycles, not assumed daily: a fortnightly book annualised
    # by the square root of 252 reports a Sharpe three times too large.
    periods = state.periods_per_year()
    report = assess_decay(
        returns,
        strategy=strategy,
        backtest_sharpe=backtest_sharpe,
        haircut=haircut,
        periods_per_year=periods,
    )

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("backtest Sharpe", f"{report.backtest_sharpe:.2f}")
    table.add_row("expected after haircut", f"{report.expected_sharpe:.2f}")
    table.add_row("live Sharpe", f"{report.live_sharpe:.2f}")
    table.add_row("observations", f"{report.observations:,}")
    table.add_row("periods a year (inferred)", f"{periods:.0f}")
    table.add_row("t-statistic", f"{report.t_statistic:+.2f}")
    table.add_row("observations to conclude", f"{report.observations_needed:,}")
    console.print(table)
    console.print(f"\n{report.verdict()}")

    if report.has_decayed:
        raise typer.Exit(code=1)


@worker_app.command("run")
def worker_run(
    interval: Annotated[
        float, typer.Option("--interval", help="Seconds between heartbeats.")
    ] = 30.0,
) -> None:
    """Run the worker process.

    The worker exists so that ad-hoc jobs (`make shell`, `make backtest`) have a
    warm container to execute in and so the stack has a liveness signal. It holds
    no queue of its own yet; distributed job execution is out of scope for v1.
    """
    settings = get_settings()
    settings.layout.ensure()
    heartbeat = Heartbeat(settings.layout.state, "worker", stale_after_seconds=interval * 3)

    stopping = False

    def _stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        log.info("worker.signal", signal=signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log.info("worker.start", data_root=str(settings.data_root), interval=interval)
    heartbeat.beat(state="starting")
    while not stopping:
        heartbeat.beat(state="idle")
        # Sleep in short slices so SIGTERM is honoured promptly and `docker compose
        # down` does not have to wait out a full interval before killing us.
        deadline = time.monotonic() + interval
        while not stopping and time.monotonic() < deadline:
            time.sleep(0.5)
    log.info("worker.stop")


@worker_app.command("healthcheck")
def worker_healthcheck() -> None:
    """Assert the worker heartbeat is fresh. Used by the Docker HEALTHCHECK."""
    _heartbeat_healthcheck("worker")


@scheduler_app.command("run")
def scheduler_run(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Schedule YAML.")] = None,
) -> None:
    """Run the job scheduler."""
    from quantlab.scheduler import run_scheduler

    get_settings().layout.ensure()
    run_scheduler(config)


@scheduler_app.command("healthcheck")
def scheduler_healthcheck() -> None:
    """Assert the scheduler heartbeat is fresh. Used by the Docker HEALTHCHECK."""
    _heartbeat_healthcheck("scheduler")


@scheduler_app.command("list")
def scheduler_list(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Schedule YAML.")] = None,
) -> None:
    """Show the configured jobs without running them."""
    from quantlab.scheduler import load_schedule

    settings = get_settings()
    path = config or (settings.layout.configs / "schedule.yaml")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("job")
    table.add_column("cron (UTC)")
    table.add_column("enabled")
    table.add_column("command", overflow="fold")
    for job in load_schedule(path):
        table.add_row(
            job.name,
            job.cron,
            "[green]yes[/green]" if job.enabled else "[dim]no[/dim]",
            " ".join(job.command),
        )
    console.print(table)


def _heartbeat_healthcheck(name: str) -> None:
    settings = get_settings()
    status = read_heartbeat(settings.layout.state / f"{name}.heartbeat.json", name)
    if status.healthy:
        console.print(status.describe())
        return
    err_console.print(status.describe())
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())


# --------------------------------------------------------------------- portfolio --
def _returns_frame(symbols: list[str], as_of: str, lookback: int) -> pl.DataFrame:
    """Daily **total** returns for a universe, through a point-in-time snapshot.

    Returned as a long frame carrying its own ``as_of`` column so that anything
    joining against it joins on the date. Symbols are dropped, loudly, if they
    lack a complete history over the window: padding a short history with zeros
    would understate both the volatility and the correlation of the instruments
    with least data, which is the opposite of conservative.

    **Total return, not price return.** Returns are computed from ``adj_close``,
    which carries dividends. Using the raw close instead drops the dividend yield
    from the return series, and in an attribution that missing yield reappears as
    negative alpha: SPY over the last five years attributes to -1.45% a year on
    price returns (t = -2.3, comfortably "significant") and to +0.02% a year
    (t = 0.04) on total returns. The second number is the true one. A short book
    would show the same error with the sign reversed, as manufactured alpha.

    The cost of this choice is that ``adj_close`` is **restated**: the provider
    rewrites the whole history each time a dividend is paid, so it is not
    point-in-time and a snapshot taken today does not reproduce what the series
    looked like a year ago. That is acceptable for measuring realised exposure
    over a fixed historical window, which is what these commands do. It is not
    acceptable in the backtest engine, which is why that path uses the raw close
    with corporate actions applied on their own dates instead.
    """
    from quantlab.data.store import Store

    snapshot = Store(get_settings().layout).as_of(as_of)
    bars = snapshot.ohlcv_daily(symbols=symbols)
    if bars.height == 0:
        err_console.print(
            f"[red]no daily bars for any of {', '.join(symbols)} at {as_of}[/red] -- "
            "ingest them first: quantlab data ingest"
        )
        raise typer.Exit(code=2)

    price = "adj_close" if "adj_close" in bars.columns else None
    if price is None:
        err_console.print(
            "[red]these bars carry no adj_close[/red], so only price returns are "
            "available. Dividends would be missing from every return, and in an "
            "attribution that shows up as negative alpha roughly equal to the "
            "dividend yield. Refusing rather than reporting a number that is wrong "
            "by a known amount."
        )
        raise typer.Exit(code=2)

    wide = bars.sort("as_of").pivot(index="as_of", on="symbol", values=price).tail(lookback + 1)
    names = [c for c in wide.columns if c != "as_of"]
    complete = [c for c in names if wide[c].null_count() == 0]
    dropped = sorted(set(names) - set(complete))
    if dropped:
        err_console.print(
            f"[yellow]dropped {len(dropped)}[/yellow] with gaps over the window: "
            f"{', '.join(dropped)}"
        )
    if not complete:
        err_console.print("[red]no instrument has a complete history over the window[/red]")
        raise typer.Exit(code=2)
    if wide.height < 4:
        err_console.print(f"[red]only {wide.height} bars in the window[/red]")
        raise typer.Exit(code=2)

    # The return dated T is the move from T-1 to T, so it drops the first row.
    return wide.select(
        pl.col("as_of"),
        *[(pl.col(c) / pl.col(c).shift(1) - 1.0).alias(c) for c in complete],
    ).slice(1)


def _returns_matrix(symbols: list[str], as_of: str, lookback: int) -> tuple[np.ndarray, list[str]]:
    frame = _returns_frame(symbols, as_of, lookback)
    names = [c for c in frame.columns if c != "as_of"]
    if len(names) < 2:
        err_console.print("[red]need at least two instruments with a complete history[/red]")
        raise typer.Exit(code=2)
    return frame.select(names).to_numpy(), names


@portfolio_app.command("covariance")
def portfolio_covariance(
    symbols: Annotated[list[str], typer.Option("--symbol", "-s", help="Instrument. Repeatable.")],
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")],
    lookback: Annotated[int, typer.Option("--lookback", help="Trading days of history.")] = 504,
) -> None:
    """Compare covariance estimators on a real universe.

    The number to look at is the condition number. A sample covariance matrix on a
    wide universe is routinely conditioned in the millions, and an optimiser
    inverting it is amplifying estimation error rather than using information --
    which is where confident, enormous, meaningless positions come from.
    """
    from quantlab.portfolio import ewma_covariance, ledoit_wolf_covariance, sample_covariance

    returns, names = _returns_matrix(symbols, as_of, lookback)
    estimates = [
        sample_covariance(returns),
        ewma_covariance(returns),
        ledoit_wolf_covariance(returns, target="constant_correlation"),
        ledoit_wolf_covariance(returns, target="identity"),
    ]

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("estimator")
    table.add_column("shrinkage", justify="right")
    table.add_column("condition", justify="right")
    table.add_column("mean vol", justify="right")
    for estimate in estimates:
        table.add_row(
            estimate.method,
            f"{estimate.shrinkage:.3f}",
            f"{estimate.condition:,.0f}",
            f"{estimate.volatilities().mean():.1%}",
        )
    console.print(table)
    console.print(
        f"\n{len(names)} instruments, {returns.shape[0]:,} observations "
        f"({estimates[0].observations_per_parameter:.1f} per free parameter)."
    )
    if not estimates[0].is_usable:
        console.print(
            "[yellow]The sample estimate is effectively singular.[/yellow] Shrink it, "
            "or use fewer instruments -- do not invert it."
        )


@portfolio_app.command("build")
def portfolio_build(
    symbols: Annotated[list[str], typer.Option("--symbol", "-s", help="Instrument. Repeatable.")],
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")],
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help="equal, inverse-vol, risk-parity or mean-variance.",
        ),
    ] = "risk-parity",
    lookback: Annotated[int, typer.Option("--lookback", help="Trading days of history.")] = 504,
    max_position: Annotated[
        float, typer.Option("--max-position", help="Cap on any one weight.")
    ] = 0.25,
    risk_aversion: Annotated[
        float, typer.Option("--risk-aversion", help="Mean-variance only.")
    ] = 5.0,
) -> None:
    """Construct a portfolio, and report the risk each position actually carries.

    The weights are the less interesting half of the output. Risk contributions
    are what a book is actually exposed to, and an equal-weighted portfolio of
    correlated assets routinely puts most of its risk in one place while looking
    perfectly diversified on the weights.

    Expected returns for ``mean-variance`` are **not** estimated from the sample.
    Sample means are so noisy that optimising on them reliably produces a worse
    portfolio than equal weighting; this command uses a flat prior, so what you
    see is the risk model's view alone.
    """
    from quantlab.portfolio import (
        Constraints,
        equal_weight,
        inverse_volatility,
        ledoit_wolf_covariance,
        mean_variance,
        risk_contributions,
        risk_parity,
    )

    returns, names = _returns_matrix(symbols, as_of, lookback)
    estimate = ledoit_wolf_covariance(returns)
    matrix = estimate.matrix

    limits = Constraints(max_position=max_position, min_position=0.0, net_range=(1.0, 1.0))
    try:
        if method == "equal":
            allocation = equal_weight(len(names))
        elif method == "inverse-vol":
            allocation = inverse_volatility(matrix)
        elif method == "risk-parity":
            allocation = risk_parity(matrix)
        elif method == "mean-variance":
            allocation = mean_variance(
                np.zeros(len(names)), matrix, risk_aversion=risk_aversion, constraints=limits
            )
        else:
            err_console.print(
                f"[red]unknown method {method!r}[/red]; known: equal, inverse-vol, "
                "risk-parity, mean-variance"
            )
            raise typer.Exit(code=2)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    contributions = risk_contributions(allocation.weights, matrix)
    shares = contributions / contributions.sum()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("symbol")
    table.add_column("weight", justify="right")
    table.add_column("vol", justify="right")
    table.add_column("risk share", justify="right")
    volatilities = estimate.volatilities()
    order = np.argsort(allocation.weights)[::-1]
    for i in order:
        table.add_row(
            names[i],
            f"{allocation.weights[i]:+.2%}",
            f"{volatilities[i]:.1%}",
            f"{shares[i]:.1%}",
        )
    console.print(table)

    console.print(
        f"\n{allocation.method}: {allocation.effective_positions:.1f} effective positions "
        f"out of {len(names)}; portfolio volatility "
        f"{math.sqrt(float(allocation.weights @ matrix @ allocation.weights) * 252):.1%}."
    )
    if not allocation.converged:
        console.print("[yellow]the optimiser did not converge[/yellow] -- treat with suspicion")
    violations = limits.violations(allocation.weights)
    if violations:
        console.print(f"[yellow]constraints not met:[/yellow] {'; '.join(violations)}")
    console.print(f"[dim]covariance: {estimate.describe()}[/dim]")


# -------------------------------------------------------------------------- risk --
@risk_app.command("attribute")
def risk_attribute(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Instrument to attribute.")],
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")],
    lookback: Annotated[int, typer.Option("--lookback", help="Trading days of history.")] = 1260,
    factors: Annotated[
        list[str] | None,
        typer.Option("--factor", "-f", help="Factor series symbol in the lake. Repeatable."),
    ] = None,
) -> None:
    """Answer the question a risk model exists for: is this secretly just beta?

    Returns are joined to the factors **on the date**, and the risk-free rate is
    subtracted from the instrument so that both sides are excess returns over the
    same rate. Getting either wrong loads the mismatch straight onto the
    intercept, which is precisely the number being tested.

    Standard errors are Newey-West, not OLS. Returns are autocorrelated, and OLS
    standard errors on autocorrelated data come out too small -- inflating every
    t-statistic in the direction of finding alpha that is not there.
    """
    from quantlab.data.store import Store
    from quantlab.risk import attribute

    wanted = tuple(factors or ("KF_MKT_RF", "KF_SMB", "KF_HML", "KF_MOM"))
    snapshot = Store(get_settings().layout).as_of(as_of)
    series = snapshot.series(symbols=[*wanted, "KF_RF"])
    if series.height == 0:
        err_console.print(
            f"[red]none of {', '.join(wanted)} is in the lake[/red] -- ingest the "
            "academic factor datasets first:\n"
            "  quantlab data ingest -f ken_french.series_observations"
        )
        raise typer.Exit(code=2)

    factor_wide = series.sort("as_of").pivot(index="as_of", on="symbol", values="value")
    available = [name for name in wanted if name in factor_wide.columns]
    if not available:
        err_console.print(
            f"[red]no usable factor columns among {', '.join(wanted)}[/red]; "
            f"the lake has {', '.join(sorted(set(factor_wide.columns) - {'as_of'}))}"
        )
        raise typer.Exit(code=2)
    if "KF_RF" not in factor_wide.columns:
        err_console.print(
            "[red]KF_RF is not in the lake[/red] -- without the risk-free rate the "
            "instrument cannot be put on the same excess-return footing as the "
            "factors, and the mismatch would land on the alpha"
        )
        raise typer.Exit(code=2)

    returns = _returns_frame([symbol], as_of, lookback)
    if symbol not in returns.columns:
        err_console.print(f"[red]{symbol} has no complete history over the window[/red]")
        raise typer.Exit(code=2)

    # An inner join on the calendar date, not the instant: a daily bar is stamped
    # at its session close (20:00 UTC for a US listing) while a daily series is
    # stamped at midnight, so joining on the raw timestamp matches nothing.
    # Aligning them positionally instead would silently regress one series
    # against another shifted by every session either side is missing.
    joined = (
        returns.select(pl.col("as_of").dt.date().alias("date"), pl.col(symbol).alias("_asset"))
        .join(
            factor_wide.select(
                pl.col("as_of").dt.date().alias("date"),
                pl.col("KF_RF"),
                *[pl.col(c) for c in available],
            ),
            on="date",
            how="inner",
        )
        .drop_nulls()
        .sort("date")
    )
    if joined.height < 60:
        err_console.print(
            f"[red]only {joined.height} dates are common to the instrument and the "
            f"factors[/red] -- too few to identify {len(available)} loadings and an "
            "intercept"
        )
        raise typer.Exit(code=2)

    excess = (joined["_asset"] - joined["KF_RF"]).to_numpy()
    result = attribute(excess, joined.select(available).to_numpy(), factor_names=tuple(available))
    first, last = joined["date"].item(0), joined["date"].item(-1)
    console.print(
        f"[bold]{symbol}[/bold] excess of KF_RF, {joined.height:,} common dates "
        f"({first:%Y-%m-%d} to {last:%Y-%m-%d})"
    )
    console.print(result.describe())


@risk_app.command("pca")
def risk_pca(
    symbols: Annotated[list[str], typer.Option("--symbol", "-s", help="Instrument. Repeatable.")],
    as_of: Annotated[str, typer.Option("--as-of", help="Point-in-time date.")],
    lookback: Annotated[int, typer.Option("--lookback", help="Trading days of history.")] = 504,
    n_factors: Annotated[int, typer.Option("--factors", help="Components to keep.")] = 3,
) -> None:
    """Extract statistical risk factors when no factor returns are available.

    A principal component is a direction of variance, not an economic exposure.
    The first one is almost always "the market" whether or not anyone measured it;
    the rest have no names, and naming them is how a risk model becomes a story.
    """
    from quantlab.risk import pca_risk_model

    returns, names = _returns_matrix(symbols, as_of, lookback)
    try:
        model = pca_risk_model(returns, n_factors=n_factors)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("symbol")
    for k in range(model.n_factors):
        table.add_column(f"PC{k + 1}", justify="right")
    table.add_column("specific vol", justify="right")
    for i, name in enumerate(names):
        table.add_row(
            name,
            *[f"{model.loadings[i, k]:+.3f}" for k in range(model.n_factors)],
            f"{math.sqrt(model.specific_variance[i] * 252):.1%}",
        )
    console.print(table)
    console.print(f"\n{model.describe()}")
    if model.first_factor_share > 0.5:
        console.print(
            f"[yellow]PC1 carries {model.first_factor_share:.0%} of the variance.[/yellow] "
            "A strategy loading on it is a market bet however its signal is described."
        )
