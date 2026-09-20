"""The `quantlab` command line interface.

One rule governs every command here: a command that cannot do its job **fails
loudly**. Nothing in this CLI silently substitutes a fallback data source, quietly
skips a failed ingest, or returns a plausible-looking zero (spec section 13).
Commands whose implementation is scheduled for a later milestone exit non-zero
with the milestone named, rather than returning an empty result that could be
mistaken for "no data".
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from quantlab import __version__
from quantlab.config import get_settings
from quantlab.data.catalogue import SOURCES, get_source
from quantlab.health import Status, run_checks, source_availability
from quantlab.logging import configure_logging, get_logger
from quantlab.runtime import Heartbeat, read_heartbeat

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
app.add_typer(worker_app)
app.add_typer(scheduler_app)
app.add_typer(data_app)

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
    incremental: Annotated[
        bool,
        typer.Option("--incremental", help="Update since the last cursor instead of a full pull."),
    ] = False,
) -> None:
    """Pull data into the lake."""
    mode = "incremental" if incremental else "full"
    _not_yet(
        f"data ingest ({mode})", 2, "the data layer -- store, PIT machinery, FRED/Stooq/Binance"
    )


@app.command()
def backtest(
    config: Annotated[Path, typer.Option("--config", "-c", help="Strategy YAML.")],
) -> None:
    """Run a backtest from a strategy config."""
    _not_yet(f"backtest --config {config}", 4, "the vectorised backtest engine")


@app.command()
def paper() -> None:
    """Run one paper-trading cycle."""
    _not_yet("paper", 11, "the paper-trading loop")


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
