import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ContextPane } from "../components/ContextPane";
import { LagBar } from "../components/LagBar";
import { Legend, Mark } from "../components/Mark";
import { RecordPane } from "../components/Record";
import { Timeline } from "../components/Timeline";
import type { DisclosureEvent, FeedResponse, Kind } from "../lib/api";
import { longDay, plural, shortDay } from "../lib/format";
import { useApi, useAsOf } from "../lib/hooks";

type KindFilter = "all" | Exclude<Kind, "unread">;

const FILTERS: { key: KindFilter; label: string }[] = [
  { key: "all", label: "Everyone" },
  { key: "insider", label: "Insiders" },
  { key: "congress", label: "Congress" },
  { key: "fund", label: "Funds" },
  { key: "contract", label: "Contracts" },
];
const WINDOWS = [7, 30, 90, 365];
/** A run by one filer longer than this is folded, so one report cannot bury a day. */
const FOLD_OVER = 4;
const FOLD_TO = 3;

/** Consecutive events by the same filer: usually the lines of one report. */
function runs(events: DisclosureEvent[]): DisclosureEvent[][] {
  const out: DisclosureEvent[][] = [];
  for (const event of events) {
    const last = out[out.length - 1];
    if (last && last[0]?.actor_id === event.actor_id && last[0]?.kind === event.kind) last.push(event);
    else out.push([event]);
  }
  return out;
}

export function FeedPage() {
  const { asOf, search } = useAsOf();
  const [params, setParams] = useSearchParams();
  const actor = params.get("actor");
  const view = params.get("view") === "timeline" ? "timeline" : "desk";
  const days = Number(params.get("days")) || (actor ? 365 : 7);

  // In the URL, like the as-of date: a filtered view is something to send someone.
  const asked = params.get("kind");
  const kind: KindFilter = FILTERS.some((filter) => filter.key === asked) ? (asked as KindFilter) : "all";
  const [noise, setNoise] = useState(false);
  const [unfolded, setUnfolded] = useState<ReadonlySet<string>>(new Set());
  const [selected, setSelected] = useState<string | null>(null);
  const list = useRef<HTMLDivElement>(null);

  const { data, error, loading } = useApi<FeedResponse>("/api/feed", { as_of: asOf, days, actor });

  const setParam = (key: string, value: string | null) =>
    setParams((current) => {
      const next = new URLSearchParams(current);
      if (value === null) next.delete(key);
      else next.set(key, value);
      return next;
    });

  const keep = (event: DisclosureEvent) => {
    if (event.noise && !noise) return false;
    // "Everyone" is everyone who traded. Contracts are a different kind of fact and
    // there are thirty large ones a week, so they are asked for, not mixed in.
    // Unless the page IS an agency's, where its awards are the whole point.
    if (kind === "all") return event.kind !== "contract" || Boolean(actor);
    return event.kind === kind || (event.kind === "unread" && kind === "congress");
  };

  const groups = useMemo(
    () => (data?.groups ?? []).map((group) => ({ date: group.date, events: group.events.filter(keep) })).filter((group) => group.events.length > 0),
    // `keep` is rebuilt from kind and noise, which are listed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data, kind, noise],
  );

  // What the list actually shows, in order: this is what the arrow keys walk.
  const visible = useMemo(
    () =>
      groups.flatMap((group) =>
        runs(group.events).flatMap((run) => {
          const key = `${group.date}:${run[0]?.id}`;
          return !actor && run.length > FOLD_OVER && !unfolded.has(key) ? run.slice(0, FOLD_TO) : run;
        }),
      ),
    [groups, unfolded, actor],
  );
  const shown = groups.flatMap((group) => group.events);
  const picked = visible.find((event) => event.id === selected) ?? visible[0] ?? null;

  useEffect(() => setSelected(null), [asOf, actor, days]);

  if (error) return <Problem message={error} />;
  if (!data) return <p className="loading">Reading the lake…</p>;

  const today = data.today;
  const disclosures = shown.filter((event) => event.kind !== "unread").length;
  const hiddenNoise = data.groups.flatMap((group) => group.events).filter((event) => event.noise).length;
  const noun = kind === "contract" ? "contract action" : "disclosure";
  const heading = data.actor
    ? data.actor.name
    : data.is_live
      ? `${plural(disclosures, noun)} became public ${days === 7 ? "this week" : `in the last ${days} days`}`
      : `${plural(disclosures, noun)} had become public by ${shortDay(today)}`;

  const step = (delta: number) => {
    const index = picked ? visible.findIndex((event) => event.id === picked.id) : -1;
    const next = visible[Math.max(0, Math.min(visible.length - 1, index + delta))];
    if (!next) return;
    setSelected(next.id);
    list.current?.querySelector<HTMLElement>(`[data-event="${CSS.escape(next.id)}"]`)?.focus();
  };

  return (
    <div className={`feed${loading ? " feed--stale" : ""}`}>
      <div className="deskbar">
        <div className="deskbar__title">
          {data.actor && (
            <Link className="crumb" to={`/${search}`}>
              Back to everyone
            </Link>
          )}
          <h1>{heading}</h1>
          {data.actor && (
            <p>
              {[data.actor.role, `${plural(disclosures, "disclosure")} in the last ${days === 365 ? "year" : `${days} days`}`].filter(Boolean).join(". ")}.
            </p>
          )}
        </div>
        <div className="deskbar__controls">
          <div className="pills" role="group" aria-label="Filter by who disclosed">
            {FILTERS.map((filter) => (
              <button key={filter.key} type="button" aria-pressed={kind === filter.key} onClick={() => setParam("kind", filter.key === "all" ? null : filter.key)}>
                {filter.label}
              </button>
            ))}
          </div>
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
          <div className="pills" role="group" aria-label="View">
            <button type="button" aria-pressed={view === "desk"} onClick={() => setParam("view", null)}>
              Desk
            </button>
            <button type="button" aria-pressed={view === "timeline"} onClick={() => setParam("view", "timeline")}>
              Timeline
            </button>
          </div>
        </div>
      </div>

      {data.empty_lake ? (
        <EmptyLake />
      ) : view === "timeline" ? (
        <div className="feed__timeline">
          {kind !== "contract" && <RoutineToggle noise={noise} setNoise={setNoise} hidden={hiddenNoise} />}
          <Timeline events={shown} beyond={data.beyond.filter(keep)} beyondTotal={kind === "contract" ? data.beyond_contracts_total : kind === "all" ? data.beyond_total : data.beyond.filter(keep).length} today={today} isLive={data.is_live} noun={kind === "contract" ? "contract action" : "trade"} />
          <Legend chart contracts={kind === "contract" || Boolean(actor?.startsWith("agency:"))} />
        </div>
      ) : (
        <div className="desk">
          <section className="desk__list" aria-label="Disclosures">
            {kind !== "contract" && <RoutineToggle noise={noise} setNoise={setNoise} hidden={hiddenNoise} />}
            {kind === "contract" && (
              <p className="note desk__note">
                Federal contract actions of $25 million and over, for about forty listed contractors. Pentagon actions appear 90 days after they happen. Each company's page lists its smaller ones.
              </p>
            )}
            {data.is_live &&
              kind !== "congress" &&
              kind !== "fund" &&
              kind !== "contract" &&
              data.insights.map((insight) => (
                <Link className="insight" key={insight.ticker} to={`/t/${encodeURIComponent(insight.ticker)}${search}`}>
                  {insight.text}
                </Link>
              ))}
            {visible.length === 0 ? (
              <div className="empty empty--inline">
                <h2>Nothing was disclosed in this window.</h2>
                <p>Widen the window, include the routine filings, or choose another as-of date.</p>
              </div>
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
                {groups.map((group) => (
                  <section key={group.date} aria-label={longDay(group.date, true)}>
                    <h2 className="desk__day">{longDay(group.date, group.date.slice(0, 4) !== today.slice(0, 4))}</h2>
                    <ul>
                      {runs(group.events).flatMap((run) => {
                        const first = run[0];
                        if (!first) return [];
                        const key = `${group.date}:${first.id}`;
                        const folded = !actor && run.length > FOLD_OVER && !unfolded.has(key);
                        const items = (folded ? run.slice(0, FOLD_TO) : run).map((event) => (
                          <li key={event.id}>
                            <Item event={event} today={today} on={event.id === picked?.id} onPick={() => setSelected(event.id)} />
                          </li>
                        ));
                        if (!folded) return items;
                        return [
                          ...items,
                          <li key={`${key}:more`} className="desk__more">
                            <button type="button" onClick={() => setUnfolded((current) => new Set(current).add(key))}>
                              Show {run.length - FOLD_TO} more from {first.actor}
                            </button>
                          </li>,
                        ];
                      })}
                    </ul>
                  </section>
                ))}
              </div>
            )}
          </section>

          <main className="desk__record">{picked ? <RecordPane event={picked} /> : <p className="loading">Choose a disclosure on the left.</p>}</main>

          <aside className="desk__context" aria-label="Context">
            {picked && <ContextPane event={picked} today={today} />}
          </aside>
        </div>
      )}
    </div>
  );
}

function Item({ event, today, on, onPick }: { event: DisclosureEvent; today: string; on: boolean; onPick: () => void }) {
  const unread = event.kind === "unread";
  return (
    <button type="button" className={`item${on ? " item--on" : ""}${event.noise ? " item--noise" : ""}${unread ? " item--unread" : ""}`} aria-pressed={on} data-event={event.id} onClick={onPick}>
      <span className={`item__ticker${event.ticker ? "" : " item__ticker--none"}`}>{unread ? "Scan" : (event.ticker ?? "None")}</span>
      <span className="item__body">
        <span className="item__who">{event.actor}</span>
        <span className="item__what">
          {!unread && <Mark kind={event.kind} direction={event.direction} />}
          <span>{unread ? "Filed on paper, so it can't be read" : `${event.verb} ${event.size}`}</span>
        </span>
      </span>
      {!unread && <LagBar event={event} today={today} compact />}
    </button>
  );
}

function RoutineToggle({ noise, setNoise, hidden }: { noise: boolean; setNoise: (value: boolean) => void; hidden: number }) {
  return (
    <label className="check">
      <input type="checkbox" checked={noise} onChange={(event) => setNoise(event.target.checked)} />
      <span>
        Include grants, option exercises and pre-scheduled sales
        {hidden > 0 && !noise ? ` (${hidden} hidden)` : ""}
      </span>
    </label>
  );
}

function EmptyLake() {
  return (
    <div className="empty">
      <h2>The lake holds no disclosures yet.</h2>
      <p>This app only reads. Pull the data in, then reload:</p>
      <pre>
        quantlab data ingest -f sec_insider.insider_transactions -f sec_13f.institutional_holdings \{"\n"}
        {"  "}-f sec_ftd.fails_to_deliver -f house_clerk.congress_filings -f house_clerk.congress_trades \{"\n"}
        {"  "}-f senate_efd.congress_filings -f senate_efd.congress_trades
      </pre>
    </div>
  );
}

export function Problem({ message }: { message: string }) {
  return (
    <div className="empty empty--problem" role="alert">
      <h2>The lake could not be read.</h2>
      <p>{message}</p>
      <p>Check that `quantlab serve` is still running, then reload.</p>
    </div>
  );
}
