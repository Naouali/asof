import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { LagBar } from "../components/LagBar";
import { Legend, Mark } from "../components/Mark";
import { PriceChart } from "../components/PriceChart";
import type { TickerResponse } from "../lib/api";
import { plural, shares as formatShares, dollars, shortDay, signedPercent } from "../lib/format";
import { useApi, useAsOf } from "../lib/hooks";
import { actorHref } from "../lib/links";
import { Problem } from "./FeedPage";

export function TickerPage() {
  const { ticker = "" } = useParams();
  const { asOf, search } = useAsOf();
  const { data, error, loading } = useApi<TickerResponse>(`/api/tickers/${encodeURIComponent(ticker)}`, { as_of: asOf });
  const [noise, setNoise] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const events = useMemo(() => (data?.events ?? []).filter((event) => noise || !event.noise), [data, noise]);
  useEffect(() => setSelected(null), [ticker, asOf]);

  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  const today = data.today;
  const picked = events.find((event) => event.id === selected) ?? events[0] ?? null;
  const last = data.prices[data.prices.length - 1];
  const hiddenNoise = data.events.length - data.events.filter((event) => !event.noise).length;
  const oldest = data.holders.reduce((age, holder) => Math.max(age, holder.age_days), 0);

  return (
    <div className={`page page--ticker${loading ? " page--stale" : ""}`}>
      <main className="page__main">
        <header className="pagehead">
          <div>
            <Link className="crumb" to={`/${search}`}>
              Back to the feed
            </Link>
            <div className="tickerhead">
              <h1 className="tickerhead__symbol">{data.ticker}</h1>
              {data.name && <p className="tickerhead__name">{data.name}</p>}
            </div>
            {last && (
              <p className="pagehead__lede">
                Closed at ${last.close.toFixed(2)} on {shortDay(last.date, today)}.
              </p>
            )}
          </div>
        </header>

        <p className="brief">{data.brief.join(" ")}</p>

        {data.prices.length > 1 ? (
          <section className="panel" aria-label="Price and disclosures">
            <h2 className="panel__title">Daily close, with each disclosure drawn from the trade to the day it became public</h2>
            <Legend chart />
            <PriceChart prices={data.prices} events={events} selected={picked?.id ?? null} onSelect={setSelected} today={today} />
            {picked && (
              <div className="readout" aria-live="polite">
                <p>
                  <strong>{picked.actor}</strong> {picked.verb.toLowerCase()} {picked.size} on {shortDay(picked.traded_on, today)}. It became public on {shortDay(picked.disclosed_on, today)},{" "}
                  {plural(picked.lag_days, "day")} later.
                </p>
                {picked.price_move_pct !== null && (
                  <p className="readout__move">
                    Price moved <strong>{signedPercent(picked.price_move_pct)}</strong> before anyone outside knew
                  </p>
                )}
              </div>
            )}
          </section>
        ) : (
          <div className="empty">
            <h2>No price history for {data.ticker} in the lake.</h2>
            <p>
              Disclosures are listed below without a chart. To draw one, add <code>{data.ticker}</code> to the <code>yahoo.ohlcv_daily</code> job in <code>configs/ingest.yaml</code> and run{" "}
              <code>quantlab data ingest -f yahoo.ohlcv_daily</code>.
            </p>
          </div>
        )}

        <section aria-labelledby="disclosures-title">
          <div className="sectionhead">
            <h2 id="disclosures-title">Every disclosure naming {data.ticker}</h2>
            <label className="check">
              <input type="checkbox" checked={noise} onChange={(event) => setNoise(event.target.checked)} />
              <span>Include grants, option exercises and pre-scheduled sales{hiddenNoise > 0 && !noise ? ` (${hiddenNoise} hidden)` : ""}</span>
            </label>
          </div>
          {events.length === 0 ? (
            <p className="note">Nothing in the last year. {hiddenNoise > 0 ? "Routine filings are hidden; tick the box to see them." : ""}</p>
          ) : (
            <div className="ledger ledger--ticker">
              <div className="ledger__head" aria-hidden="true">
                <span>Who</span>
                <span>What they did</span>
                <span className="ledger__num">Price move</span>
                <span>Trade to disclosure</span>
              </div>
              <ul className="ledger__rows">
                {events.map((event) => (
                  <li key={event.id} className={`row row--ticker${event.id === picked?.id ? " row--on" : ""}${event.noise ? " row--noise" : ""}`}>
                    <button type="button" className="row__select" aria-pressed={event.id === picked?.id} aria-label={`Show ${event.actor} on the chart`} onClick={() => setSelected(event.id)} />
                    <div className="row__who">
                      <Link to={actorHref(event, search)}>{event.actor}</Link>
                      {event.role && <div className="row__sub">{event.role}</div>}
                    </div>
                    <div className="row__what">
                      <Mark kind={event.kind} direction={event.direction} />
                      <div>
                        <p>
                          <strong>{event.verb}</strong> {event.size}
                        </p>
                        {event.detail && <p className="row__sub">{event.detail}</p>}
                      </div>
                    </div>
                    <div className="ledger__num">{event.price_move_pct === null ? "" : signedPercent(event.price_move_pct)}</div>
                    <LagBar event={event} today={today} />
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {data.holders.length > 0 && (
          <section aria-labelledby="funds-title">
            <div className="sectionhead">
              <h2 id="funds-title">Tracked funds that reported holding {data.ticker}</h2>
            </div>
            <p className="stale">
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
                <path d="M8 4.5v4l2.5 1.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
              <span>
                These positions were counted up to <strong>{oldest} days ago</strong>. Funds report once a quarter, up to 45 days late, and never report shorts.
              </span>
            </p>
            <div className="ledger ledger--funds">
              <div className="ledger__head" aria-hidden="true">
                <span>Fund</span>
                <span className="ledger__num">Shares</span>
                <span className="ledger__num">Change on the quarter</span>
                <span className="ledger__num">Value then</span>
                <span className="ledger__num">Counted on</span>
              </div>
              <ul className="ledger__rows">
                {data.holders.map((holder) => (
                  <li key={holder.manager_id} className="row row--fund">
                    <Link to={`/?${new URLSearchParams({ ...(asOf ? { asof: asOf } : {}), actor: holder.manager_id })}`}>{holder.manager}</Link>
                    <span className="ledger__num">{formatShares(holder.shares)}</span>
                    <strong className="ledger__num">
                      {holder.change === null ? "First report held" : holder.change === 0 ? "No change" : `${holder.change > 0 ? "+" : "−"}${formatShares(Math.abs(holder.change))}`}
                    </strong>
                    <span className="ledger__num">{dollars(holder.value_usd)}</span>
                    <span className="ledger__num">{shortDay(holder.period, today)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>

      <aside className="page__aside">
        {data.fails_to_deliver && (
          <section>
            <h2>Settlement</h2>
            <p>
              <strong>{formatShares(data.fails_to_deliver.quantity)} shares</strong> had failed to deliver on {shortDay(data.fails_to_deliver.settled_on, today)}, posted by the SEC on{" "}
              {shortDay(data.fails_to_deliver.posted_on, today)}. This is a settlement balance, not short interest.
            </p>
          </section>
        )}
        <section>
          <h2>What this page can't see</h2>
          <dl className="blind">
            <div>
              <dt>Senate trades.</dt>
              <dd>Only the House is covered.</dd>
            </div>
            <div>
              <dt>House reports filed on paper.</dt>
              <dd>Any of them may name {data.ticker}.</dd>
            </div>
            <div>
              <dt>Funds outside the tracked list.</dt>
              <dd>Holdings are read for the managers in the ingest plan, not for every fund.</dd>
            </div>
            <div>
              <dt>Anything a fund has done since its last quarter end.</dt>
              <dd>And any short position, ever.</dd>
            </div>
          </dl>
        </section>
      </aside>
    </div>
  );
}
