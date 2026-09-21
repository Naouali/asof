import type { DataHealth } from "../lib/api";
import { bytes, plural, shortDay, stampDay } from "../lib/format";
import { useApi, useTitle } from "../lib/hooks";
import { Problem } from "./FeedPage";

const DISCLOSURE_SETS = new Set(["insider_transactions", "institutional_holdings", "congress_trades", "congress_filings", "fails_to_deliver"]);

function freshness(age: number | null): { label: string; tone: "fresh" | "aging" | "stale" } {
  if (age === null) return { label: "Unknown", tone: "stale" };
  const label = age < 1 ? "Today" : `${plural(Math.round(age), "day")} ago`;
  return { label, tone: age < 4 ? "fresh" : age < 21 ? "aging" : "stale" };
}

export function DataPage({ today }: { today: string }) {
  const { data, error } = useApi<DataHealth>("/api/data", {});
  useTitle("Data health");
  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  const sets = [...data.datasets].sort((a, b) => Number(DISCLOSURE_SETS.has(b.dataset)) - Number(DISCLOSURE_SETS.has(a.dataset)) || a.dataset.localeCompare(b.dataset));
  const run = data.runs[0];
  const coverage = data.unpriceable.wanted > 0 ? Math.round((data.unpriceable.priced / data.unpriceable.wanted) * 100) : null;

  return (
    <div className="page page--data">
      <main className="page__main">
        <header className="pagehead">
          <div>
            <h1>Data health</h1>
            <p className="pagehead__lede">What the lake holds right now, and how old the newest thing in it is. This page is never read through the as-of date: it describes the machinery, not the market.</p>
          </div>
        </header>

        <section aria-labelledby="priced-title">
          <div className="sectionhead">
            <h2 id="priced-title">Tickers the lake can price</h2>
          </div>
          <p className="note">
            {coverage === null
              ? "No disclosure names a ticker yet."
              : `${data.unpriceable.priced.toLocaleString("en-US")} of the ${data.unpriceable.wanted.toLocaleString("en-US")} tickers named in disclosures have prices here, ${coverage}%. Every return in this app — a portfolio's chart, the price move before a disclosure — is measured on those and on no others.`}
            {data.unpriceable.refused_total > 0 &&
              ` ${plural(data.unpriceable.refused_total, "ticker")} were asked for and refused: usually misspelt, delisted, or never a ticker. They are retried on the dates below.`}
          </p>
          {data.unpriceable.refused.length > 0 && (
            <ul className="refused">
              {data.unpriceable.refused.map((row) => (
                <li key={row.symbol} title={row.reason}>
                  <strong>{row.symbol}</strong>
                  <span className="row__sub">retry {shortDay(row.retry_after, today)}</span>
                </li>
              ))}
              {data.unpriceable.refused_total > data.unpriceable.refused.length && (
                <li className="refused__more">and {(data.unpriceable.refused_total - data.unpriceable.refused.length).toLocaleString("en-US")} more</li>
              )}
            </ul>
          )}
        </section>

        <section aria-labelledby="sets-title">
          <div className="sectionhead">
            <h2 id="sets-title">Datasets</h2>
          </div>
          {sets.length === 0 ? (
            <div className="empty">
              <h2>The lake is empty.</h2>
              <p>
                Run <code>quantlab data ingest</code>, then reload.
              </p>
            </div>
          ) : (
            <div className="ledger ledger--data">
              <div className="ledger__head" aria-hidden="true">
                <span>Dataset</span>
                <span>Source</span>
                <span className="ledger__num">Rows</span>
                <span>Covers</span>
                <span>Newest became public</span>
                <span className="ledger__num">Size</span>
              </div>
              <ul className="ledger__rows">
                {sets.map((set) => {
                  const fresh = freshness(set.age_days);
                  return (
                    <li key={`${set.source}.${set.dataset}`} className="row row--data">
                      <strong>{set.dataset.replaceAll("_", " ")}</strong>
                      <span className="row__sub">{set.source}</span>
                      <span className="ledger__num">{set.rows.toLocaleString("en-US")}</span>
                      <span>{set.first && set.last ? `${shortDay(set.first, today)} to ${shortDay(set.last, today)}` : ""}</span>
                      <span className={`fresh fresh--${fresh.tone}`}>
                        <span className="fresh__dot" aria-hidden="true" />
                        {fresh.label}
                      </span>
                      <span className="ledger__num">{bytes(set.bytes)}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </section>

        <section aria-labelledby="unread-title">
          <div className="sectionhead">
            <h2 id="unread-title">Congressional reports that could not be read</h2>
          </div>
          <p className="note">
            {data.unread_reports.length === 0
              ? "None in the period the lake covers."
              : `${plural(data.unread_reports.length, "report")} filed on paper. Each is a scan with no text, so the trades in it are missing from the feed, not zero.`}
          </p>
          {data.unread_reports.length > 0 && (
            <div className="ledger ledger--unread">
              <ul className="ledger__rows">
                {data.unread_reports.map((report) => (
                  <li key={report.id} className="row row--unreadlist">
                    <strong>{report.actor}</strong>
                    <span className="row__sub">{report.role}</span>
                    <span>Filed {shortDay(report.disclosed_on, today)}</span>
                    {report.source_url && (
                      <a href={report.source_url} target="_blank" rel="noreferrer">
                        Open the scan
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </main>

      <aside className="page__aside">
        <section>
          <h2>Last ingest</h2>
          {run?.started_at ? (
            <>
              <p>
                Started {stampDay(run.started_at)}
                {run.incremental ? ", incremental" : ", full"}.
              </p>
              <ul className="runs">
                {(run.jobs ?? []).map((job) => (
                  <li key={job.fetcher}>
                    <span>{job.fetcher}</span>
                    <strong className={job.ok ? "" : "runs__failed"}>{!job.ok ? "Failed" : job.skipped_reason ? "Skipped" : `${job.rows.toLocaleString("en-US")} rows`}</strong>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p>No ingest has been recorded yet.</p>
          )}
        </section>
        <section>
          <h2>How data gets here</h2>
          <p>
            This app only reads. The scheduler runs <code>quantlab data ingest --incremental</code> every morning; you can run it by hand at any time.
          </p>
        </section>
      </aside>
    </div>
  );
}
