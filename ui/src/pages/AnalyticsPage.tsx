import { Link, useSearchParams } from "react-router-dom";

import { LagBar } from "../components/LagBar";
import { Mark } from "../components/Mark";
import type { ContractsAnalytics, ContractTotal, FilerRecord, PriceMovesResponse, Spread, TrackRecords, TradedResponse } from "../lib/api";
import { dollars, monthName, plural, roundDollars, shortDay, signedPercent } from "../lib/format";
import { useApi, useAsOf, useRules, useTitle } from "../lib/hooks";
import { actorHref, hasPortfolio, portfolioHref, tickerHref } from "../lib/links";
import { Problem } from "./FeedPage";

type Tab = "traded" | "moves" | "money" | "following";

const TABS: { key: Tab; label: string }[] = [
  { key: "traded", label: "What is being traded" },
  { key: "moves", label: "Price moved before you knew" },
  { key: "money", label: "Government money" },
  { key: "following", label: "Who is worth following" },
];
const KINDS = [
  { key: "", label: "Everyone" },
  { key: "insider", label: "Insiders" },
  { key: "congress", label: "Congress" },
  { key: "fund", label: "Funds" },
];
const WINDOWS = [30, 90, 365];

/** Three questions asked of the whole lake. Each tab is one screen, and all of them follow the as-of date. */
export function AnalyticsPage() {
  const [params, setParams] = useSearchParams();
  const asked = params.get("tab");
  const tab: Tab = TABS.some((entry) => entry.key === asked) ? (asked as Tab) : "traded";
  useTitle(`Analytics, ${(TABS.find((entry) => entry.key === tab)?.label ?? "").toLowerCase()}`);

  const setParam = (key: string, value: string | null) =>
    setParams((current) => {
      const next = new URLSearchParams(current);
      if (value === null) next.delete(key);
      else next.set(key, value);
      return next;
    });

  return (
    <div className="analytics">
      <div className="deskbar">
        <div className="deskbar__title">
          <h1>Analytics</h1>
        </div>
        <div className="deskbar__controls">
          <div className="pills" role="group" aria-label="Analysis">
            {TABS.map((entry) => (
              <button key={entry.key} type="button" aria-pressed={tab === entry.key} onClick={() => setParam("tab", entry.key === "traded" ? null : entry.key)}>
                {entry.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      {tab === "traded" && <Traded kind={params.get("kind") ?? ""} days={Number(params.get("days")) || 90} setParam={setParam} />}
      {tab === "moves" && <Moves days={Number(params.get("days")) || 365} setParam={setParam} />}
      {tab === "money" && <Money />}
      {tab === "following" && <Following horizon={Number(params.get("horizon")) || 90} setParam={setParam} />}
    </div>
  );
}

function Window({ days, setParam }: { days: number; setParam: (key: string, value: string | null) => void }) {
  return (
    <label className="select">
      <span className="visually-hidden">Window</span>
      <select value={days} onChange={(event) => setParam("days", event.target.value)}>
        {WINDOWS.map((value) => (
          <option key={value} value={value}>
            {value === 365 ? "Last year" : `Last ${value} days`}
          </option>
        ))}
      </select>
    </label>
  );
}

// -------------------------------------------------------------------- traded --
function Traded({ kind, days, setParam }: { kind: string; days: number; setParam: (key: string, value: string | null) => void }) {
  const { asOf, search } = useAsOf();
  const { data, error, loading } = useApi<TradedResponse>("/api/analytics/traded", { as_of: asOf, days, kind: kind || null });
  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  const top = data.tickers[0];
  const widest = Math.max(1, ...data.tickers.map((row) => Math.max(row.buyers, row.sellers)));
  const busiest = Math.max(1, ...data.people.map((person) => person.buys + person.sells));
  const headline = !top
    ? "Nobody disclosed a trade by choice in this window."
    : `${top.ticker} drew the most people: ${plural(top.sellers, "seller")} and ${plural(top.buyers, "buyer")} among ${data.trades.toLocaleString("en-US")} disclosed trades.`;

  return (
    <div className={`analytics__body${loading ? " analytics__body--stale" : ""}`}>
      <div className="analytics__lead">
        <p className="brief">{headline}</p>
        <div className="analytics__controls">
          <div className="pills" role="group" aria-label="Who disclosed">
            {KINDS.map((entry) => (
              <button key={entry.key} type="button" aria-pressed={kind === entry.key} onClick={() => setParam("kind", entry.key || null)}>
                {entry.label}
              </button>
            ))}
          </div>
          <Window days={days} setParam={setParam} />
        </div>
      </div>

      <div className="analytics__grid">
        <section className="panel" aria-labelledby="traded-tickers">
          <h2 className="panel__title" id="traded-tickers">
            People on each side, by ticker
          </h2>
          <p className="note">
            Counted in people, not dollars: Congress discloses a range and a fund a position, so a dollar total across them would be invented. Grants, option exercises and pre-scheduled sales are left out.
            {data.tickers_total > data.tickers.length ? ` The ${data.tickers.length} busiest of ${data.tickers_total.toLocaleString("en-US")} tickers.` : ""}
          </p>
          <div className="sides" role="table" aria-label="Buyers and sellers by ticker">
            <div className="sides__head" role="row">
              <span role="columnheader">Ticker</span>
              <span role="columnheader" className="sides__label sides__label--sell">
                Sold
              </span>
              <span role="columnheader" className="sides__label">
                Bought
              </span>
            </div>
            {data.tickers.map((row) => (
              <Link key={row.ticker} className="sides__row" role="row" to={tickerHref(row.ticker, search)} title={`${row.ticker}: ${plural(row.sellers, "person", "people")} sold, ${plural(row.buyers, "person", "people")} bought`}>
                <span role="cell" className="sides__ticker">
                  <strong>{row.ticker}</strong>
                  {row.name && <span>{row.name}</span>}
                </span>
                <span role="cell" className="sides__half sides__half--sell">
                  <span className="sides__count">{row.sellers || ""}</span>
                  <span className="sides__bar sides__bar--sell" style={{ width: `${(row.sellers / widest) * 100}%` }} />
                </span>
                <span role="cell" className="sides__half">
                  <span className="sides__bar sides__bar--buy" style={{ width: `${(row.buyers / widest) * 100}%` }} />
                  <span className="sides__count">{row.buyers || ""}</span>
                </span>
              </Link>
            ))}
          </div>
        </section>

        <section className="panel" aria-labelledby="traded-people">
          <h2 className="panel__title" id="traded-people">
            Who disclosed the most trades
          </h2>
          <p className="note">Lines of their reports, not decisions: one report can list a hundred. Funds are left out, because a fund changes thousands of positions a quarter.</p>
          <ul className="tally">
            {data.people.map((person) => (
              <li key={person.actor_id}>
                <Link to={actorHref(person, search)}>{person.actor}</Link>
                <span className="tally__sub">
                  {[person.role, `${plural(person.buys, "purchase")}, ${plural(person.sells, "sale")}, ${person.tickers === 0 ? "no ticker named" : plural(person.tickers, "ticker")}`].filter(Boolean).join(". ")}
                  {hasPortfolio(person) && (
                    <>
                      . <Link to={portfolioHref(person.actor_id, search)}>Portfolio</Link>
                    </>
                  )}
                </span>
                <span className="tally__bar" aria-hidden="true" style={{ width: `${((person.buys + person.sells) / busiest) * 100}%` }}>
                  <span className="tally__buy" style={{ flexGrow: person.buys }} />
                  <span className="tally__sell" style={{ flexGrow: person.sells }} />
                </span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------- moves --
const BOX = { width: 520, height: 34, pad: 8 };

function Box({ spread, side, reach, minSample }: { spread: Spread; side: "buy" | "sell"; reach: number; minSample: number | null }) {
  const label = side === "buy" ? "Bought" : "Sold";
  if (spread.median === null || spread.p10 === null || spread.p25 === null || spread.p75 === null || spread.p90 === null) {
    return (
      <div className="box">
        <span className="box__side">{label}</span>
        <span className="box__few">{spread.n === 0 ? "None measured" : `${plural(spread.n, "trade")}: ${minSample === null ? "too few" : `fewer than ${minSample}, too few`} to summarise`}</span>
      </div>
    );
  }
  const x = (value: number) => BOX.pad + ((Math.max(-reach, Math.min(reach, value)) + reach) / (2 * reach)) * (BOX.width - 2 * BOX.pad);
  const mid = BOX.height / 2;
  const color = side === "buy" ? "var(--buy)" : "var(--sell)";
  return (
    <div className="box">
      <span className="box__side">{label}</span>
      <svg viewBox={`0 0 ${BOX.width} ${BOX.height}`} preserveAspectRatio="none" role="img" aria-label={`${label}: median ${signedPercent(spread.median)}, half of trades between ${signedPercent(spread.p25)} and ${signedPercent(spread.p75)}`}>
        <title>{`${label}, ${spread.n} trades. Median ${signedPercent(spread.median)}. Half between ${signedPercent(spread.p25)} and ${signedPercent(spread.p75)}; eight in ten between ${signedPercent(spread.p10)} and ${signedPercent(spread.p90)}.`}</title>
        <line className="box__zero" x1={x(0)} x2={x(0)} y1={0} y2={BOX.height} />
        <line x1={x(spread.p10)} x2={x(spread.p90)} y1={mid} y2={mid} stroke={color} strokeWidth={2} />
        {/* Bought is solid and sold is an outline, as everywhere else in the app. */}
        <rect x={x(spread.p25)} y={mid - 9} width={Math.max(2, x(spread.p75) - x(spread.p25))} height={18} rx={3} fill={side === "buy" ? color : "var(--panel)"} stroke={color} strokeWidth={2} />
        <line x1={x(spread.median)} x2={x(spread.median)} y1={mid - 13} y2={mid + 13} stroke="var(--ink)" strokeWidth={3} />
      </svg>
      <span className="box__value">
        <strong>{signedPercent(spread.median)}</strong> {plural(spread.n, "trade")}
      </span>
    </div>
  );
}

function Moves({ days, setParam }: { days: number; setParam: (key: string, value: string | null) => void }) {
  const { asOf, search } = useAsOf();
  const rules = useRules();
  const { data, error, loading } = useApi<PriceMovesResponse>("/api/analytics/price-moves", { as_of: asOf, days });
  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  const spreads = data.groups.flatMap((group) => [group.buy, group.sell]);
  const extreme = Math.max(5, ...spreads.flatMap((spread) => [Math.abs(spread.p10 ?? 0), Math.abs(spread.p90 ?? 0)]));
  const reach = Math.ceil(extreme / 5) * 5;
  const people = data.groups.filter((group) => group.key !== "fund" && group.buy.median !== null);
  const lead = [...people].sort((a, b) => (b.buy.median ?? 0) - (a.buy.median ?? 0))[0];
  const headline = !lead
    ? "Too few trades in tickers with a price history to say how far prices moved before disclosure."
    : `By the time a purchase by ${lead.label.toLowerCase()} became public, the price had already moved ${signedPercent(lead.buy.median ?? 0)} at the median.`;

  return (
    <div className={`analytics__body${loading ? " analytics__body--stale" : ""}`}>
      <div className="analytics__lead">
        <p className="brief">{headline}</p>
        <div className="analytics__controls">
          <Window days={days} setParam={setParam} />
        </div>
      </div>

      <div className="analytics__grid">
        <section className="panel" aria-labelledby="moves-spread">
          <h2 className="panel__title" id="moves-spread">
            Price change between the trade and the day it became public
          </h2>
          <p className="note">
            The white line is the median, the box holds the middle half of trades, and the whiskers eight in ten. Measured on {data.measured.toLocaleString("en-US")} of {data.trades.toLocaleString("en-US")} trades: only
            the {data.priced_tickers} tickers the lake holds prices for, all of them large companies. For a fund the clock starts when the quarter closed, not when it traded.
          </p>
          <div className="boxes">
            <div className="boxes__scale" aria-hidden="true">
              <span>−{reach}%</span>
              <span>0</span>
              <span>+{reach}%</span>
            </div>
            {data.groups.map((group) => (
              <div key={group.key} className="boxes__group">
                <h3>{group.label}</h3>
                <Box spread={group.buy} side="buy" reach={reach} minSample={rules?.analytics.min_sample ?? null} />
                <Box spread={group.sell} side="sell" reach={reach} minSample={rules?.analytics.min_sample ?? null} />
              </div>
            ))}
          </div>
        </section>

        <section className="panel" aria-labelledby="moves-missed">
          <h2 className="panel__title" id="moves-missed">
            The moves a reader had missed by the most
          </h2>
          <p className="note">A price that rose before a purchase was public, or fell before a sale was. It says the disclosure came late in the move, not that the trade caused it.</p>
          <ul className="missed">
            {data.examples.map((event) => (
              <li key={event.id}>
                <div className="missed__what">
                  <Mark kind={event.kind} direction={event.direction} />
                  <span>
                    <Link to={actorHref(event, search)}>{event.actor}</Link> {event.verb.toLowerCase()} {event.ticker && <Link to={tickerHref(event.ticker, search)}>{event.ticker}</Link>}
                  </span>
                  <strong>{event.price_move_pct === null ? "" : signedPercent(event.price_move_pct)}</strong>
                </div>
                <LagBar event={event} today={data.today} />
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------- money --
function Totals({ rows, link, wide = false }: { rows: ContractTotal[]; link?: (name: string) => string; wide?: boolean }) {
  const widest = Math.max(1, ...rows.map((row) => Math.abs(row.net_usd)));
  return (
    <ul className={`totals${wide ? " totals--wide" : ""}`}>
      {rows.map((row) => {
        const defense = Math.max(0, Math.min(row.net_usd, row.defense_usd));
        const civilian = Math.max(0, row.net_usd - defense);
        const name = link ? <Link to={link(row.name)}>{row.name}</Link> : <span>{row.name}</span>;
        return (
          <li key={row.name} title={`${row.name}: ${dollars(row.net_usd)} net across ${plural(row.actions, "action")}, ${dollars(row.defense_usd)} of it from the Pentagon`}>
            <span className="totals__name">{name}</span>
            <span className="totals__track">
              {defense > 0 && <span className="totals__bar totals__bar--defense" style={{ width: `${(defense / widest) * 100}%` }} />}
              {civilian > 0 && <span className="totals__bar totals__bar--civilian" style={{ width: `${(civilian / widest) * 100}%` }} />}
            </span>
            <strong className="totals__value">{dollars(row.net_usd)}</strong>
          </li>
        );
      })}
    </ul>
  );
}

const COLS = { width: 560, height: 190, top: 14, bottom: 26 };

function Months({ months }: { months: ContractsAnalytics["months"] }) {
  const high = Math.max(1, ...months.map((month) => month.committed_usd));
  const low = Math.max(0, ...months.map((month) => -month.taken_back_usd));
  const span = high + low;
  const zero = COLS.top + (high / span) * (COLS.height - COLS.top - COLS.bottom);
  const scale = (COLS.height - COLS.top - COLS.bottom) / span;
  const step = COLS.width / Math.max(1, months.length);
  const bar = Math.min(34, step - 8);
  const peak = months.reduce((best, month) => (month.committed_usd > best ? month.committed_usd : best), 0);
  return (
    <svg className="months" viewBox={`0 0 ${COLS.width} ${COLS.height}`} role="img" aria-label="Money committed and taken back, by the month it became public">
      <line className="months__zero" x1={0} x2={COLS.width} y1={zero} y2={zero} />
      {months.map((month, index) => {
        const x = index * step + (step - bar) / 2;
        const up = month.committed_usd * scale;
        const down = -month.taken_back_usd * scale;
        const date = new Date(`${month.month}T00:00:00Z`);
        return (
          <g key={month.month}>
            <title>{`${monthName(date.getUTCMonth())} ${date.getUTCFullYear()}: ${dollars(month.committed_usd)} committed, ${dollars(-month.taken_back_usd)} taken back`}</title>
            <rect x={x} y={zero - up} width={bar} height={Math.max(1, up)} rx={3} fill="var(--ink)" />
            {down > 0.5 && <rect x={x + 1} y={zero + 2} width={bar - 2} height={Math.max(2, down)} rx={2} fill="var(--panel)" stroke="var(--ink)" strokeWidth={2} />}
            {/* One number, on the tallest column: the rest are read against it. */}
            {month.committed_usd === peak && peak > 0 && (
              <text className="months__value" x={x + bar / 2} y={zero - up - 4} textAnchor="middle">
                {dollars(month.committed_usd)}
              </text>
            )}
            <text className="months__label" x={x + bar / 2} y={COLS.height - 8} textAnchor="middle">
              {monthName(date.getUTCMonth()).slice(0, 3)}
              {/* A year runs from one September to the next: say which is which. */}
              {(index === 0 || date.getUTCMonth() === 0) && ` ${String(date.getUTCFullYear()).slice(2)}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function Money() {
  const { asOf, search } = useAsOf();
  const rules = useRules();
  const { data, error, loading } = useApi<ContractsAnalytics>("/api/analytics/contracts", { as_of: asOf });
  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  if (data.actions === 0 && !data.hidden) {
    return (
      <div className="empty">
        <h2>No contract actions had become public by {shortDay(data.today)}.</h2>
        <p>
          Run <code>quantlab data ingest -f usaspending.government_contracts</code>, or choose a later as-of date.
        </p>
      </div>
    );
  }

  const share = data.net_usd > 0 ? Math.round((data.defense_usd / data.net_usd) * 100) : 0;
  const peak = [...data.months].sort((a, b) => b.committed_usd - a.committed_usd)[0];
  const peakDate = peak ? new Date(`${peak.month}T00:00:00Z`) : null;
  return (
    <div className={`analytics__body${loading ? " analytics__body--stale" : ""}`}>
      <div className="analytics__lead">
        <p className="brief">
          The government committed a net {dollars(data.net_usd)} to the listed contractors in the year to {shortDay(data.today)}, {share}% of it from the Pentagon.
          {data.hidden && ` Another ${data.hidden.actions.toLocaleString("en-US")} actions worth ${dollars(data.hidden.net_usd)} had been signed and were not public yet.`}
        </p>
      </div>

      <div className="analytics__grid analytics__grid--money">
        <section className="panel" aria-labelledby="money-companies">
          <h2 className="panel__title" id="money-companies">
            Net committed, by company
          </h2>
          <p className="note">
            Money committed less money taken back, across {data.actions.toLocaleString("en-US")} actions{rules && ` of ${roundDollars(rules.contracts.min_action_usd)} and over`}. It is spent over the life of each contract, which runs to years: it is not revenue.
          </p>
          <div className="totals__key" aria-hidden="true">
            <span className="totals__bar totals__bar--defense" /> Pentagon{rules && `, published ${rules.contracts.defense_embargo_days} days late`}
            <span className="totals__bar totals__bar--civilian" /> Everyone else
          </div>
          <Totals rows={data.companies} link={(name) => tickerHref(name, search)} />
        </section>

        <div className="analytics__stack">
          <section className="panel" aria-labelledby="money-months">
            <h2 className="panel__title" id="money-months">
              {peakDate ? `${monthName(peakDate.getUTCMonth())} was the largest month to reach the public record` : "By the month it became public"}
            </h2>
            <p className="note">
              By the month each action became public, not the month it was signed. Solid is money committed; the outline below the line is money taken back.
              {data.is_live && " The latest month is still filling."}
            </p>
            <Months months={data.months} />
          </section>
          <section className="panel" aria-labelledby="money-agencies">
            <h2 className="panel__title" id="money-agencies">
              Where it came from
            </h2>
            {/* A department's page lists the large contracts its offices signed. */}
            <Totals rows={data.agencies} wide link={(name) => actorHref({ actor_id: `agency:${name.toLowerCase()}` }, search)} />
          </section>
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- following --
const HORIZONS = [30, 90, 180];

/** How the reader should hold the leader board: is it skill, or is it a hundred
 * people and a coin? The sentence at the top answers that before any name. */
function verdict(data: TrackRecords): string {
  if (!data.luck || data.ranked === 0) return "Not enough finished trades to rank anybody yet.";
  const chance = data.luck.as_good_by_chance;
  const best = signedPercent(data.luck.best_mean_excess_pct);
  if (chance >= 0.2)
    return `Nobody stands apart from chance. Dealing the same trades out at random produced a leader as good as ${best} in ${Math.round(chance * 100)}% of ${data.luck.shuffles} shuffles, so this ranking is what ${plural(data.ranked, "filer")} and a coin look like.`;
  return `The leader is hard to explain by chance: dealing the same trades out at random produced someone as good as ${best} in only ${Math.round(chance * 100)}% of ${data.luck.shuffles} shuffles. That is not proof of skill — it is one period, one holding span, and only the trades this lake can price.`;
}

function Following({ horizon, setParam }: { horizon: number; setParam: (key: string, value: string | null) => void }) {
  const { asOf, search } = useAsOf();
  const { data, error, loading } = useApi<TrackRecords>("/api/analytics/track-record", { as_of: asOf, horizon });
  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Measuring…</p>;

  const ranked = data.people.filter((person) => person.ranked);
  const rest = data.people.filter((person) => !person.ranked);

  return (
    <div className={`analytics__body${loading ? " analytics__body--stale" : ""}`}>
      <div className="analytics__lead">
        <p className="brief">{verdict(data)}</p>
        <div className="analytics__controls">
          <label className="select">
            <span className="visually-hidden">Holding span</span>
            <select value={horizon} onChange={(event) => setParam("horizon", event.target.value)}>
              {HORIZONS.map((value) => (
                <option key={value} value={value}>
                  Held {value} days
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {data.measured === 0 ? (
        <div className="empty">
          <h2>Nothing can be measured yet.</h2>
          <p>{data.why_empty ?? `No trade has both a price history and a finished ${horizon}-day window before this date.`}</p>
        </div>
      ) : (
        <div className="analytics__grid analytics__grid--money">
          <section className="panel" aria-labelledby="following-board">
            <h2 className="panel__title" id="following-board">
              Bought on the day it became public, held {data.horizon_days} days, against {data.benchmark}
            </h2>
            <p className="note">
              Every purchase whose {data.horizon_days}-day window had closed by this date, entered at the first price a reader could have paid. What is shown is the difference from putting the same money in {data.benchmark} over
              exactly the same days. {plural(data.measured, "trade")} from {plural(data.filers, "filer")} could be measured; {plural(data.ranked, "filer")} made at least {data.min_trades} of them and are ranked.
              {data.unpriced > 0 && ` ${data.unpriced.toLocaleString("en-US")} more are in tickers the lake cannot price.`}
            </p>
            {ranked.length === 0 ? (
              <p className="note">Nobody has {plural(data.min_trades, "finished trade")} yet.</p>
            ) : (
              <div className="ledger ledger--board">
                <div className="ledger__head" aria-hidden="true">
                  <span>Filer</span>
                  <span className="ledger__num">Trades</span>
                  <span className="ledger__num">A follower got</span>
                  <span className="ledger__num">They got</span>
                  <span className="ledger__num">Beat {data.benchmark}</span>
                </div>
                <ul className="ledger__rows">
                  {ranked.map((person) => (
                    <Line key={person.actor_id} person={person} search={search} />
                  ))}
                </ul>
              </div>
            )}
          </section>

          <div className="analytics__stack">
            <section className="panel" aria-labelledby="following-all">
              <h2 className="panel__title" id="following-all">
                Every measured trade, pooled
              </h2>
              {data.everyone && (
                <p className="note">
                  Across all {plural(data.everyone.trades, "trade")}, following a disclosure was worth{" "}
                  <strong>{signedPercent(data.everyone.mean_excess_pct)}</strong> against {data.benchmark} on average
                  {data.everyone.low_pct !== null && data.everyone.high_pct !== null && ` (between ${signedPercent(data.everyone.low_pct)} and ${signedPercent(data.everyone.high_pct)}, 19 times in 20)`}, with a median of{" "}
                  {signedPercent(data.everyone.median_excess_pct)} and {Math.round(data.everyone.beat_rate * 100)}% of trades beating it.{" "}
                  {!data.everyone.distinguishable && "That interval contains zero: pooled, following everybody is indistinguishable from buying the index."}
                </p>
              )}
              <p className="note">
                {data.standouts === 0
                  ? `No filer's interval is clear of zero.`
                  : `${plural(data.standouts, "filer")} have an interval clear of zero; with ${plural(data.ranked, "filer")} ranked, about ${data.expected_by_luck} would clear it by luck alone.`}
              </p>
            </section>
            {rest.length > 0 && (
              <section className="panel" aria-labelledby="following-thin">
                <h2 className="panel__title" id="following-thin">
                  Measured, too few to rank
                </h2>
                <p className="note">Fewer than {plural(data.min_trades, "finished trade")}. Shown so nobody is missing, not so they can be compared.</p>
                <ul className="tally">
                  {rest.slice(0, 10).map((person) => (
                    <li key={person.actor_id}>
                      <Link to={portfolioHref(person.actor_id, search)}>{person.actor}</Link>
                      <span className="tally__sub">
                        {plural(person.trades, "trade")}, {signedPercent(person.mean_excess_pct)}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Line({ person, search }: { person: FilerRecord; search: string }) {
  return (
    <li className="row row--board" title={`${person.trades} trades between ${person.first} and ${person.last}`}>
      <div className="row__who">
        <Link to={portfolioHref(person.actor_id, search)}>{person.actor}</Link>
        <div className="row__sub">
          {person.role}
          {person.distinguishable && <span className="badge">clear of zero</span>}
        </div>
      </div>
      <div className="ledger__num">{person.trades}</div>
      <div className="ledger__num">
        <strong>{signedPercent(person.mean_excess_pct)}</strong>
        {person.low_pct !== null && person.high_pct !== null && (
          <div className="row__sub">
            {signedPercent(person.low_pct)} to {signedPercent(person.high_pct)}
          </div>
        )}
      </div>
      <div className="ledger__num">{person.own_mean_excess_pct === null ? "" : signedPercent(person.own_mean_excess_pct)}</div>
      <div className="ledger__num">{Math.round(person.beat_rate * 100)}%</div>
    </li>
  );
}

