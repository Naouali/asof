import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { Mark } from "../components/Mark";
import { PriceChart } from "../components/PriceChart";
import type { DataHealth, DisclosureEvent, FeedResponse, PortfolioMembers, PortfolioResponse, TickerResponse, TradedResponse } from "../lib/api";
import { addDays, daysBetween, plural, shortDay } from "../lib/format";
import { useApi, useRules, useTitle } from "../lib/hooks";

const WINDOW_DAYS = 120;
const TIMELINE_ROWS = 36;
/** A small panel cannot carry a year of marks. It shows the latest few, and says so. */
const CHART_MARKS = 12;
const HOLDINGS = 6;

/** The SEC stores manager names in capitals. A sentence on this page does not. */
function readable(name: string): string {
  if (name !== name.toUpperCase()) return name;
  return name.toLowerCase().replace(/\b[a-z]/g, (letter) => letter.toUpperCase()).replace(/\b(Llc|Lp|Plc)\b/g, (word) => word.toUpperCase());
}

function sentence(event: DisclosureEvent, today: string): string {
  const what = `${readable(event.actor)} ${event.verb.toLowerCase()} ${event.size}${event.ticker ? ` of ${event.ticker}` : ""}`;
  return `${what}. ${event.kind === "fund" ? "Counted" : "Traded"} ${shortDay(event.traded_on, today)}, public ${shortDay(event.disclosed_on, today)}.`;
}

/* ------------------------------------------------------------------ timeline -- */

/** A spread of all three kinds, so the field is not thirty rows of one report. */
function sample(events: DisclosureEvent[], start: string): DisclosureEvent[] {
  const take = (kind: DisclosureEvent["kind"], count: number) => {
    // Only trades that began inside the window: a line clipped at the left edge
    // has lost the half of it that matters.
    const pool = events.filter((event) => event.kind === kind && event.traded_on >= start);
    const step = Math.max(1, Math.floor(pool.length / count));
    return pool.filter((_, index) => index % step === 0).slice(0, count);
  };
  // Few fund rows: every position in a quarterly report shares one pair of dates.
  return [...take("insider", 12), ...take("congress", 21), ...take("fund", 3)].sort((a, b) => (a.traded_on < b.traded_on ? -1 : 1)).slice(0, TIMELINE_ROWS);
}

function TradesPanel({ feed }: { feed: FeedResponse }) {
  const today = feed.today;
  const start = addDays(today, -WINDOW_DAYS);
  const events = useMemo(() => feed.groups.flatMap((group) => group.events).filter((event) => !event.noise && event.kind !== "unread" && event.kind !== "contract"), [feed]);
  const rows = useMemo(() => sample(events, start), [events, start]);
  const [back, setBack] = useState(34);
  const [hovered, setHovered] = useState<DisclosureEvent | null>(null);
  const plot = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const asOf = addDays(today, -back);
  const at = (day: string) => Math.max(0, Math.min(100, (daysBetween(start, day) / WINDOW_DAYS) * 100));
  const hidden = events.filter((event) => event.traded_on <= asOf && event.disclosed_on > asOf).length;

  const move = (clientX: number) => {
    const box = plot.current?.getBoundingClientRect();
    if (!box) return;
    const share = Math.max(0, Math.min(1, (clientX - box.left) / box.width));
    setBack(Math.max(0, Math.min(WINDOW_DAYS - 6, Math.round((1 - share) * WINDOW_DAYS))));
  };

  return (
    <section className="lp-panel lp-trades" aria-label="Detected trades">
      <h2 className="lp-panel__title">
        {back === 0 ? (
          <>Everything here is public today. Drag the line back.</>
        ) : (
          <>
            On {shortDay(asOf, today)}, <em>{plural(hidden, "trade")}</em> had already happened that nobody outside could see yet.
          </>
        )}
      </h2>

      <div
        className="lp-trades__plot"
        ref={plot}
        onPointerDown={(event) => {
          dragging.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          move(event.clientX);
        }}
        onPointerMove={(event) => dragging.current && move(event.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => setHovered(null)}
      >
        {rows.map((row) => {
          const state = row.disclosed_on <= asOf ? "known" : row.traded_on <= asOf ? "hidden" : "future";
          return (
            <div key={row.id} className={`lp-row lp-row--${state}`} onPointerEnter={() => setHovered(row)}>
              <span className={`lp-row__line lp-row__line--${row.direction}`} style={{ left: `${at(row.traded_on)}%`, width: `max(3px, ${at(row.disclosed_on) - at(row.traded_on)}%)` }} />
              {row.traded_on >= start && <span className={`lp-row__start lp-row__start--${row.direction}`} style={{ left: `${at(row.traded_on)}%` }} />}
              <span className="lp-row__end" style={{ left: `${at(row.disclosed_on)}%` }}>
                <Mark kind={row.kind} direction={row.direction} />
              </span>
            </div>
          );
        })}
        <div className="lp-trades__curtain" style={{ left: `${at(asOf)}%` }} aria-hidden="true">
          <span>{back === 0 ? "Today" : shortDay(asOf, today)}</span>
        </div>
      </div>

      <div className="lp-trades__foot">
        <label className="lp-trades__slider">
          <span className="visually-hidden">As-of date, in days before today</span>
          <input type="range" min={0} max={WINDOW_DAYS - 6} value={WINDOW_DAYS - 6 - back} onChange={(event) => setBack(WINDOW_DAYS - 6 - Number(event.target.value))} />
        </label>
        <p className="lp-caption">{hovered ? sentence(hovered, today) : "Each line runs from a trade to the day it became public. Drag the yellow line, or point at a trade."}</p>
      </div>
    </section>
  );
}

/* --------------------------------------------------------------------- chart -- */

function ChartPanel({ ticker }: { ticker: TickerResponse }) {
  const all = useMemo(() => ticker.events.filter((event) => !event.noise), [ticker]);
  const events = useMemo(() => all.slice(0, CHART_MARKS), [all]);
  const room = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(200);
  // The room takes whatever the panel has left, and the chart is drawn to fit it:
  // measured at once on mount, and again whenever the room or the window changes.
  useLayoutEffect(() => {
    const node = room.current;
    if (!node) return;
    const fit = () => setHeight(Math.max(140, node.clientHeight));
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(node);
    window.addEventListener("resize", fit);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", fit);
    };
  }, []);
  const [selected, setSelected] = useState<string | null>(null);
  const picked = events.find((event) => event.id === selected) ?? events[0] ?? null;
  const last = ticker.prices[ticker.prices.length - 1];

  return (
    <section className="lp-panel lp-chart" aria-label="Price and disclosures">
      <h2 className="lp-panel__title">
        {ticker.ticker}
        {last ? ` closed at $${last.close.toFixed(2)}.` : "."} Its <em>{all.length > events.length ? `${events.length} latest disclosures` : plural(events.length, "disclosure")}</em>, each drawn from the trade to the day it
        became public.
      </h2>
      <div className="lp-chart__room" ref={room}>
        <PriceChart prices={ticker.prices} events={events} selected={picked?.id ?? null} onSelect={setSelected} today={ticker.today} height={height} />
      </div>
      <p className="lp-caption">
        {picked ? sentence(picked, ticker.today) : "Point at a mark to see who traded."}
        {picked?.price_move_pct != null && ` The price moved ${picked.price_move_pct > 0 ? "+" : "−"}${Math.abs(picked.price_move_pct).toFixed(1)}% before anyone outside knew.`}
      </p>
    </section>
  );
}

/* --------------------------------------------------------------------- clone -- */

interface Holding {
  ticker: string;
  weight: number;
}

/** Move one weight and share the difference among the others, so the total stays 100. */
function reweigh(holdings: Holding[], ticker: string, weight: number): Holding[] {
  const others = holdings.filter((holding) => holding.ticker !== ticker);
  const room = 100 - weight;
  const spread = others.reduce((sum, holding) => sum + holding.weight, 0);
  return holdings.map((holding) => {
    if (holding.ticker === ticker) return { ...holding, weight };
    return { ...holding, weight: spread > 0 ? (holding.weight / spread) * room : room / others.length };
  });
}

function ClonePanel({ portfolio }: { portfolio: PortfolioResponse }) {
  const rules = useRules();
  // The largest holdings of the portfolio the Portfolios page compiles, brought to 100%.
  const original = useMemo(() => {
    const top = portfolio.holdings.slice(0, HOLDINGS);
    const total = top.reduce((sum, holding) => sum + holding.weight_pct, 0) || 1;
    return top.map((holding) => ({ ticker: holding.ticker, weight: (holding.weight_pct / total) * 100 }));
  }, [portfolio]);
  const [holdings, setHoldings] = useState<Holding[]>(original);
  useEffect(() => setHoldings(original), [original]);
  const changed = holdings.some((holding, index) => Math.abs(holding.weight - (original[index]?.weight ?? 0)) > 0.5);

  return (
    <section className="lp-panel lp-clone" aria-label="Clone a portfolio">
      <h2 className="lp-panel__title">
        Clone <em>{portfolio.actor}</em>'s portfolio, then make it yours.
      </h2>
      <ul className="lp-clone__list">
        {holdings.map((holding) => (
          <li key={holding.ticker}>
            <label htmlFor={`weight-${holding.ticker}`}>{holding.ticker}</label>
            <input
              id={`weight-${holding.ticker}`}
              type="range"
              min={0}
              max={100}
              step={1}
              value={Math.round(holding.weight)}
              style={{ "--fill": `${holding.weight}%` } as React.CSSProperties}
              onChange={(event) => setHoldings((current) => reweigh(current, holding.ticker, Number(event.target.value)))}
            />
            <output htmlFor={`weight-${holding.ticker}`}>{Math.round(holding.weight)}%</output>
          </li>
        ))}
      </ul>
      <div className="lp-clone__actions">
        {/* No destination yet: cloning is shown here before it exists in the app. */}
        <button type="button" className="button button--mark">
          Clone this portfolio
        </button>
        {changed && (
          <button type="button" className="button" onClick={() => setHoldings(original)}>
            Reset
          </button>
        )}
      </div>
      <p className="lp-caption">
        Weights are estimated from the ranges members disclose, counted at the middle of each range, over {plural(portfolio.trades, "trade")}. A trade appears when it is reported{rules && `, up to ${rules.deadlines.congress_days} days after it happens`}.
      </p>
    </section>
  );
}

/* ---------------------------------------------------------------------- page -- */

export function LandingPage() {
  const feed = useApi<FeedResponse>("/api/feed", { days: WINDOW_DAYS });
  const lake = useApi<DataHealth>("/api/data", {});

  // The chart shows the most-traded ticker the lake can actually draw: the ranking
  // comes from the disclosures themselves, and the first one with prices wins.
  const ranking = useApi<TradedResponse>("/api/analytics/traded", { days: 365 });
  const candidates = ranking.data?.tickers ?? [];
  const [attempt, setAttempt] = useState(0);
  const ticker = useApi<TickerResponse>(candidates[attempt] ? `/api/tickers/${encodeURIComponent(candidates[attempt].ticker)}` : null, {});
  useEffect(() => {
    if (ticker.data && ticker.data.prices.length < 2 && attempt < candidates.length - 1) setAttempt((value) => value + 1);
  }, [ticker.data, attempt, candidates.length]);

  // The member who has traded the most tickers: the fullest portfolio to show. Both
  // the choice and the portfolio come from the API the Portfolios page uses.
  useTitle(null);
  const members = useApi<PortfolioMembers>("/api/portfolios", {});
  const fullest = members.data?.members[0] ?? null;
  const portfolio = useApi<PortfolioResponse>(fullest ? "/api/portfolio" : null, { actor: fullest?.actor_id });

  // A dataset can have several sources -- the House and the Senate both file into congress_trades.
  const rows = (dataset: string) => (lake.data?.datasets ?? []).filter((set) => set.dataset === dataset).reduce((total, set) => total + set.rows, 0);
  const ready = feed.data && !feed.data.empty_lake;

  return (
    <div className="lp">
      <aside className="lp-statement">
        <span className="brand">
          <span className="brand__mark" aria-hidden="true">
            <span className="brand__dot" />
            <span className="brand__line" />
            <span className="brand__diamond" />
          </span>
          <span className="brand__name">asof</span>
        </span>

        <div className="lp-statement__body">
          <h1>
            <span>The trade happened then.</span> <span>You found out now.</span>
          </h1>
          <p>
            Insiders, funds and members of Congress must disclose what they trade, and they do it days, weeks or months later. asof reads those filings, keeps the two dates apart, draws them on the price, and lets you
            copy a portfolio and change it.
          </p>
          {/* No destination yet: there is no sign-in to send anybody to. */}
          <button type="button" className="button button--mark">
            Request access
          </button>
        </div>

        <p className="lp-statement__foot">
          {lake.data && rows("congress_trades") > 0
            ? `Reading ${rows("insider_transactions").toLocaleString("en-US")} insider transactions, ${rows("congress_trades").toLocaleString("en-US")} congressional trades and ${rows("institutional_holdings").toLocaleString("en-US")} fund positions.`
            : "Reads insider filings, fund holdings and the trades of both chambers of Congress."}
        </p>
      </aside>

      <main className="lp-panels">
        {ready && feed.data ? <TradesPanel feed={feed.data} /> : <Waiting label="Detected trades" empty={Boolean(feed.data?.empty_lake)} />}
        {ticker.data && ticker.data.prices.length > 1 ? <ChartPanel ticker={ticker.data} /> : <Waiting label="Price and disclosures" empty={Boolean(ticker.data)} />}
        {portfolio.data && portfolio.data.holdings.length > 0 ? (
          <ClonePanel portfolio={portfolio.data} />
        ) : (
          <Waiting label="Clone a portfolio" empty={Boolean(members.data && !fullest)} />
        )}
      </main>
    </div>
  );
}

function Waiting({ label, empty }: { label: string; empty: boolean }) {
  return (
    <section className="lp-panel lp-panel--waiting" aria-label={label}>
      <p className="lp-caption">{empty ? `${label}: nothing in the lake to show yet.` : "Reading the lake…"}</p>
    </section>
  );
}
