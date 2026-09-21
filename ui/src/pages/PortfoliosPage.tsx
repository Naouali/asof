import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ReturnChart } from "../components/ReturnChart";
import type { Holding, PortfolioMember, PortfolioMembers, PortfolioResponse, SoldPosition } from "../lib/api";
import { dollars, plural, shortDay, signedPercent } from "../lib/format";
import { useApi, useAsOf, useRules, useTitle } from "../lib/hooks";
import { actorHref, tickerHref } from "../lib/links";
import { Problem } from "./FeedPage";

const CHAMBERS = [
  { key: "", label: "Both chambers" },
  { key: "house", label: "House" },
  { key: "senate", label: "Senate" },
];

/**
 * Each member's portfolio, compiled from the trades they disclosed.
 *
 * It is an estimate of something nobody is required to disclose, and the page says
 * so where it matters: there is no starting position, sizes are ranges, and a
 * return is only real from the day a purchase became public.
 */
export function PortfoliosPage() {
  const { asOf } = useAsOf();
  const [params, setParams] = useSearchParams();
  const chamber = params.get("chamber") ?? "";
  const asked = params.get("actor");
  const [query, setQuery] = useState("");
  const list = useRef<HTMLDivElement>(null);

  const { data, error } = useApi<PortfolioMembers>("/api/portfolios", { as_of: asOf });

  const setParam = (key: string, value: string | null) =>
    setParams((current) => {
      const next = new URLSearchParams(current);
      if (value === null) next.delete(key);
      else next.set(key, value);
      return next;
    });

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (data?.members ?? []).filter((member) => (!chamber || member.chamber === chamber) && (!needle || `${member.actor} ${member.role ?? ""}`.toLowerCase().includes(needle)));
  }, [data, chamber, query]);
  const picked = shown.find((member) => member.actor_id === asked) ?? (data?.members ?? []).find((member) => member.actor_id === asked) ?? shown[0] ?? null;

  useTitle(picked ? `${picked.actor}'s portfolio` : "Portfolios");

  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  const step = (delta: number) => {
    const index = picked ? shown.findIndex((member) => member.actor_id === picked.actor_id) : -1;
    const next = shown[Math.max(0, Math.min(shown.length - 1, index + delta))];
    if (!next) return;
    setParam("actor", next.actor_id);
    list.current?.querySelector<HTMLElement>(`[data-member="${CSS.escape(next.actor_id)}"]`)?.focus();
  };

  return (
    <div className="feed">
      <div className="deskbar">
        <div className="deskbar__title">
          <h1>{data.members.length === 0 ? "No member has disclosed a trade yet" : `Portfolios of ${plural(data.members.length, "member")} of Congress, compiled from their trades`}</h1>
        </div>
        <div className="deskbar__controls">
          <div className="pills" role="group" aria-label="Chamber">
            {CHAMBERS.map((entry) => (
              <button key={entry.key} type="button" aria-pressed={chamber === entry.key} onClick={() => setParam("chamber", entry.key || null)}>
                {entry.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {data.members.length === 0 ? (
        <div className="empty">
          <h2>The lake holds no congressional trades as of this date.</h2>
          <p>
            Run <code>quantlab data ingest -f house_clerk.congress_trades -f senate_efd.congress_trades</code>, or choose a later as-of date.
          </p>
        </div>
      ) : (
        <div className="folio">
          <section className="folio__list" aria-label="Members">
            <label className="folio__find">
              <span className="visually-hidden">Find a member</span>
              <input className="search__input" type="search" placeholder="Find a member or a district" value={query} onChange={(event) => setQuery(event.target.value)} />
            </label>
            {shown.length === 0 ? (
              <p className="note folio__none">Nobody matches.</p>
            ) : (
              <div
                ref={list}
                onKeyDown={(event) => {
                  if (event.key === "ArrowDown") {
                    event.preventDefault();
                    step(1);
                  } else if (event.key === "ArrowUp") {
                    event.preventDefault();
                    step(-1);
                  }
                }}
              >
                <ul>
                  {shown.map((member) => (
                    <li key={member.actor_id}>
                      <Member member={member} today={data.today} on={member.actor_id === picked?.actor_id} onPick={() => setParam("actor", member.actor_id)} />
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
          <main className="folio__main">{picked ? <Portfolio actor={picked.actor_id} /> : <p className="loading">Choose a member on the left.</p>}</main>
        </div>
      )}
    </div>
  );
}

function Member({ member, today, on, onPick }: { member: PortfolioMember; today: string; on: boolean; onPick: () => void }) {
  return (
    <button type="button" className={`item item--member${on ? " item--on" : ""}`} aria-pressed={on} data-member={member.actor_id} onClick={onPick}>
      <span className="item__body">
        <span className="item__who">{member.actor}</span>
        <span className="item__what">
          <span>
            {[member.role, `${plural(member.tickers, "ticker")} in ${plural(member.trades, "trade")}`].filter(Boolean).join(". ")}
            {member.last_disclosed && `, last on ${shortDay(member.last_disclosed, today)}`}
          </span>
        </span>
      </span>
    </button>
  );
}

function range(low: number, high: number): string {
  return low === high ? dollars(low) : `${dollars(low)} to ${dollars(high)}`;
}

function Portfolio({ actor }: { actor: string }) {
  const { asOf, search } = useAsOf();
  const rules = useRules();
  const { data, error, loading } = useApi<PortfolioResponse>("/api/portfolio", { as_of: asOf, actor });
  const [all, setAll] = useState(false);
  useEffect(() => setAll(false), [actor]);

  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Compiling…</p>;

  const today = data.today;
  const widest = Math.max(1, ...data.holdings.map((holding) => holding.weight_pct));
  const holdings = all ? data.holdings : data.holdings.slice(0, FIRST);
  const leftOut = [
    data.left_out.no_ticker > 0 && `${plural(data.left_out.no_ticker, "trade")} in assets with no ticker, such as bonds and funds`,
    data.left_out.options > 0 && `${plural(data.left_out.options, "option trade")}, whose size is a premium and not a position`,
    data.left_out.exchanges > 0 && plural(data.left_out.exchanges, "exchange"),
  ].filter(Boolean);

  return (
    <article className={`folio__body${loading ? " folio__body--stale" : ""}`}>
      <header className="folio__head">
        <p className="record__context">{data.role === "Senate" ? "Sits in the Senate" : `Represents ${data.role} in the House`}</p>
        <h2 className="record__headline">
          {data.holdings.length === 0
            ? `${data.actor} has kept nothing they were seen to buy`
            : `${data.actor} has bought and kept an estimated ${range(data.low_usd, data.high_usd)} in ${plural(data.holdings.length, "ticker")}`}
        </h2>
        <p className="record__lede">
          Compiled from {plural(data.trades, "trade")} in reports filed since {data.since ? shortDay(data.since, today) : "the record began"}. It is what they bought and still hold of it, <strong>not what they own</strong>: a
          report never says what was already held. Each size is a range, because that is all the law asks for; weights use the middle of each range.
          {data.returns &&
            ` On the ${Math.round(data.priced_share * 100)}% of it the lake has prices for, it is ${signedPercent(data.returns.since_bought_pct)} since they bought and ${signedPercent(data.returns.since_public_pct)} since each purchase became public, which is the most a follower could have had.`}
          {!data.returns && rules && data.holdings.length > 0 && ` No overall return is given: the lake has prices for ${Math.round(data.priced_share * 100)}% of it, and ${rules.portfolios.min_priced_share_pct}% is the least that would mean anything.`}
        </p>
        <div className="record__actions">
          <Link className="button" to={actorHref({ actor_id: data.actor_id }, search)}>
            Every trade by {data.actor}
          </Link>
        </div>
      </header>

      {data.performance ? (
        <section className="panel" aria-labelledby="folio-return">
          <h3 className="panel__title" id="folio-return">
            {signedPercent(data.performance.member_pct)} at the prices they traded at
            {data.performance.follower_pct !== null && `, ${signedPercent(data.performance.follower_pct)} for anyone who copied each trade the day it became public`}
          </h3>
          <div className="legend">
            <span className="legend__item">
              <span className="legend__line" aria-hidden="true" />
              {data.actor}, from the day of each trade
            </span>
            {data.performance.follower_pct !== null && (
              <span className="legend__item">
                <span className="legend__line legend__line--follower" aria-hidden="true" />
                Copied on the day each trade became public
              </span>
            )}
          </div>
          <ReturnChart points={data.performance.points} today={today} who={data.actor} />
          <p className="note folio__coverage">
            <strong>
              Drawn from {plural(data.performance.purchases, "purchase")} in {data.performance.tickers} of the {plural(data.holdings.length + data.closed.length, "ticker")} they bought: the ones the lake has prices for.
            </strong>{" "}
            A trade they have closed earns its exit price over its entry price, and then stands still. A trade still open is valued at each day's close. The line is everything gained so far over everything put in so far, sized at
            the middle of each disclosed range. Sales of shares never seen bought take no part, because they have no entry price.
          </p>
        </section>
      ) : (
        data.holdings.length + data.closed.length > 0 && (
          <p className="note">
            No return can be drawn: the lake holds prices for none of the {plural(data.holdings.length + data.closed.length, "ticker")} they bought. Add them to the <code>yahoo.ohlcv_daily</code> job in <code>configs/ingest.yaml</code>.
          </p>
        )
      )}

      {data.holdings.length > 0 && (
        <section aria-labelledby="folio-holdings">
          <div className="sectionhead">
            <h3 id="folio-holdings">Bought and kept</h3>
          </div>
          <div className="ledger ledger--folio">
            <div className="ledger__head" aria-hidden="true">
              <span>Ticker</span>
              <span>Share of the portfolio</span>
              <span className="ledger__num">Estimated size</span>
              <span className="ledger__num">Since they bought</span>
              <span className="ledger__num">Since it was public</span>
            </div>
            <ul className="ledger__rows">
              {holdings.map((holding) => (
                <Row key={holding.ticker} holding={holding} widest={widest} today={today} search={search} />
              ))}
            </ul>
          </div>
          {data.holdings.length > FIRST && (
            <button type="button" className="button folio__more" onClick={() => setAll((value) => !value)}>
              {all ? `Show the ${FIRST} largest` : `Show all ${data.holdings.length}`}
            </button>
          )}
        </section>
      )}

      <div className="folio__pair">
        <Sold title="Sold, and never seen bought" note="They held these before the record begins. How much is left is not known." rows={data.held_before} today={today} search={search} />
        <Sold title="Bought, then sold again" note="Sold for as much as was bought, or more." rows={data.closed} today={today} search={search} />
      </div>

      {leftOut.length > 0 && <p className="note">Left out of the portfolio: {leftOut.join("; ")}.</p>}
    </article>
  );
}

/** How many holdings show before "show all": one screen's worth. */
const FIRST = 12;

function Row({ holding, widest, today, search }: { holding: Holding; widest: number; today: string; search: string }) {
  const others = holding.accounts.filter((account) => account !== "their own");
  return (
    <li className="row row--folio" title={`${holding.ticker}: ${plural(holding.purchases, "purchase")}, ${plural(holding.sales, "sale")}, first bought ${shortDay(holding.first_bought, today)}`}>
      <div className="row__who">
        <Link to={tickerHref(holding.ticker, search)}>{holding.ticker}</Link>
        <div className="row__sub">
          {[`${plural(holding.purchases, "purchase")}${holding.sales > 0 ? `, ${plural(holding.sales, "sale")}` : ""} since ${shortDay(holding.first_bought, today)}`, others.length > 0 && others.join(", ")].filter(Boolean).join(". ")}
        </div>
      </div>
      <div className="weight">
        <span className="weight__track">
          <span className="weight__bar" style={{ width: `${(holding.weight_pct / widest) * 100}%` }} />
        </span>
        <strong>{holding.weight_pct.toFixed(1)}%</strong>
      </div>
      <div className="ledger__num">
        <strong>{dollars(holding.mid_usd)}</strong>
        <div className="row__sub">{range(holding.low_usd, holding.high_usd)}</div>
      </div>
      <div className="ledger__num">{holding.return_since_bought_pct === null ? <span className="row__sub">No prices</span> : signedPercent(holding.return_since_bought_pct)}</div>
      <div className="ledger__num">{holding.return_since_public_pct === null ? "" : <strong>{signedPercent(holding.return_since_public_pct)}</strong>}</div>
    </li>
  );
}

function Sold({ title, note, rows, today, search }: { title: string; note: string; rows: SoldPosition[]; today: string; search: string }) {
  const [all, setAll] = useState(false);
  if (rows.length === 0) return null;
  const shown = all ? rows : rows.slice(0, SOLD_FIRST);
  return (
    <section>
      <div className="sectionhead">
        <h3>
          {title} <span className="folio__count">{rows.length}</span>
        </h3>
      </div>
      <p className="note">{note}</p>
      <ul className="sold">
        {shown.map((row) => (
          <li key={row.ticker}>
            <Link to={tickerHref(row.ticker, search)}>{row.ticker}</Link>
            <span>
              sold {range(row.sold_low_usd, row.sold_high_usd)}
              {row.return_pct !== null && (
                <>
                  , <strong>{signedPercent(row.return_pct)}</strong> exit over entry
                </>
              )}
            </span>
            {row.last_trade && <span className="row__sub">{shortDay(row.last_trade, today)}</span>}
          </li>
        ))}
      </ul>
      {rows.length > SOLD_FIRST && (
        <button type="button" className="button folio__more" onClick={() => setAll((value) => !value)}>
          {all ? "Show fewer" : `Show all ${rows.length}`}
        </button>
      )}
    </section>
  );
}

const SOLD_FIRST = 6;
