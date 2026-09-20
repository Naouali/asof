import { Link } from "react-router-dom";

import type { DisclosureEvent } from "../lib/api";
import { addDays, daysBetween, monthName, parseDay, plural, shortDay } from "../lib/format";
import { useAsOf } from "../lib/hooks";
import { Mark } from "./Mark";

const MAX_KNOWN = 40;
const MAX_HIDDEN = 40;
const LOOKBACK_DAYS = 150;
/** A month label this close to the as-of pill would be printed underneath it. */
const PILL_CLEARANCE = 7;

/**
 * The feed as a chart: every disclosure is a line from the day of the trade to
 * the day it became public.
 *
 * When the reader has travelled back in time, a curtain is drawn at the as-of
 * date, and a second block appears: what had ALREADY HAPPENED by then and was not
 * public yet. Those lines start in the open and end behind the curtain. They were
 * not knowable on that date; they are shown, dashed, because how much had already
 * happened in secret is the whole reason to look at the past this way.
 */
export function Timeline({ events, beyond, beyondTotal, today, isLive }: { events: DisclosureEvent[]; beyond: DisclosureEvent[]; beyondTotal: number; today: string; isLive: boolean }) {
  const known = events.filter((event) => event.kind !== "unread");
  const knownRows = known.slice(0, MAX_KNOWN);
  const hiddenRows = beyond.slice(0, MAX_HIDDEN);
  const rows = [...hiddenRows, ...knownRows];
  if (rows.length === 0) {
    return (
      <div className="empty">
        <h2>Nothing to draw in this window.</h2>
        <p>Widen the window, or include the routine filings.</p>
      </div>
    );
  }

  const lastDay = rows.reduce((latest, row) => (row.disclosed_on > latest ? row.disclosed_on : latest), today);
  const earliest = rows.reduce((first, row) => (row.traded_on < first ? row.traded_on : first), today);
  const floor = addDays(today, -LOOKBACK_DAYS);
  const start = earliest < floor ? floor : addDays(earliest, -3);
  const end = addDays(lastDay, 2);
  const span = Math.max(1, daysBetween(start, end));
  const at = (day: string) => Math.max(0, Math.min(100, (daysBetween(start, day) / span) * 100));
  const asOfAt = at(today);

  const months: { left: number; label: string }[] = [];
  const first = parseDay(start);
  for (let cursor = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 1)); cursor <= parseDay(end); cursor = new Date(Date.UTC(cursor.getUTCFullYear(), cursor.getUTCMonth() + 1, 1))) {
    months.push({ left: at(cursor.toISOString().slice(0, 10)), label: monthName(cursor.getUTCMonth()) });
  }

  return (
    <div className="timeline">
      {!isLive && beyond.length > 0 && (
        <p className="timeline__headline">
          On {shortDay(today)}, {plural(beyondTotal, "trade")} had already happened that nobody outside could see yet.
        </p>
      )}

      <div className="timeline__axis">
        <div className="timeline__axis-label">Each line runs from the trade to the day it became public</div>
        <div className="timeline__plot">
          {months
            .filter((month) => Math.abs(month.left - asOfAt) > PILL_CLEARANCE && month.left < 90)
            .map((month) => (
              <span key={`${month.label}${month.left}`} className="timeline__month" style={{ left: `${month.left}%` }}>
                {month.label}
              </span>
            ))}
          <span className="timeline__asof" style={{ left: `${asOfAt}%` }}>
            {isLive ? "Today" : shortDay(today)}
          </span>
        </div>
      </div>

      <div className="timeline__frame">
        {hiddenRows.length > 0 && (
          <Block
            title="Had already happened, not yet public"
            note={beyondTotal > hiddenRows.length ? `The ${hiddenRows.length} that surfaced soonest, of ${beyondTotal.toLocaleString("en-US")}` : plural(beyondTotal, "trade")}
            rows={hiddenRows}
            dashed
            {...{ at, months, start, today }}
          />
        )}
        {knownRows.length > 0 && (
          <Block
            title={isLive ? "Public" : `Public by ${shortDay(today)}`}
            note={known.length > knownRows.length ? `The ${knownRows.length} most recent of ${known.length.toLocaleString("en-US")}` : plural(known.length, "disclosure")}
            rows={knownRows}
            {...{ at, months, start, today }}
          />
        )}
        {!isLive && (
          <div
            className="timeline__curtain"
            style={{ left: `calc(var(--timeline-label) + var(--timeline-gap) + 16px + (100% - var(--timeline-label) - var(--timeline-gap) - 16px) * ${asOfAt / 100})` }}
            aria-hidden="true"
          />
        )}
      </div>
    </div>
  );
}

function Block({
  title,
  note,
  rows,
  dashed = false,
  at,
  months,
  start,
  today,
}: {
  title: string;
  note: string;
  rows: DisclosureEvent[];
  dashed?: boolean;
  at: (day: string) => number;
  months: { left: number; label: string }[];
  start: string;
  today: string;
}) {
  const { search } = useAsOf();
  return (
    <section className="timeline__block">
      <div className="timeline__blockhead">
        <h2>{title}</h2>
        <span>{note}</span>
      </div>
      <ol>
        {rows.map((row) => {
          const from = at(row.traded_on);
          const to = at(row.disclosed_on);
          const clipped = row.traded_on < start;
          return (
            <li key={row.id} className="timeline__row">
              <div className="timeline__label">
                {row.ticker ? (
                  <Link className="timeline__ticker" to={`/t/${encodeURIComponent(row.ticker)}${search}`}>
                    {row.ticker}
                  </Link>
                ) : (
                  <span className="timeline__ticker timeline__ticker--none">None</span>
                )}
                <span className="timeline__text">
                  <strong>{row.actor}</strong> {row.verb.toLowerCase()} {row.size}
                </span>
              </div>
              <div className="timeline__plot">
                {months.map((month) => (
                  <span key={`${month.label}${month.left}`} className="timeline__grid" style={{ left: `${month.left}%` }} />
                ))}
                <span className={`timeline__span timeline__span--${row.direction}${dashed ? " timeline__span--dashed" : ""}`} style={{ left: `${from}%`, width: `max(4px, ${to - from}%)` }} />
                {clipped ? <span className="timeline__since">since {shortDay(row.traded_on, today)}</span> : <span className={`timeline__start timeline__start--${row.direction}`} style={{ left: `${from}%` }} />}
                <span className="timeline__end" style={{ left: `${to}%` }}>
                  <Mark kind={row.kind} direction={row.direction} />
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
