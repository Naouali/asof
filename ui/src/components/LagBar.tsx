import type { DisclosureEvent } from "../lib/api";
import { plural, shortDay } from "../lib/format";

/** The scale every lag bar in the app is drawn to, so their lengths compare. */
export const LAG_SCALE_DAYS = 90;

export function lagText(event: DisclosureEvent): string {
  if (event.late_days > 0) return `${plural(event.lag_days, "day")}, ${event.late_days} late`;
  return plural(event.lag_days, "day");
}

/**
 * The gap between a trade and the day the world could know about it, drawn to
 * one fixed scale: a slow filer is literally long. This is the thing the product
 * is about, so it appears on every row of every list.
 *
 * `compact` is the list's version: the bar and the count, without the dates.
 */
export function LagBar({ event, today, compact = false }: { event: DisclosureEvent; today: string; compact?: boolean }) {
  const share = Math.min(event.lag_days, LAG_SCALE_DAYS) / LAG_SCALE_DAYS;
  const late = event.late_days > 0;
  const from = event.kind === "fund" ? `As of ${shortDay(event.traded_on, today)}` : shortDay(event.traded_on, today);

  return (
    <div className={`lag${late ? " lag--late" : ""}${compact ? " lag--compact" : ""}`}>
      <div className="lag__track" aria-hidden="true">
        {event.deadline_days !== null && <span className="lag__deadline" style={{ left: `${(event.deadline_days / LAG_SCALE_DAYS) * 100}%` }} />}
        <span className="lag__bar" style={{ width: `max(8px, ${share * 100}%)` }} />
      </div>
      <div className="lag__text">
        {!compact && (
          <span className="lag__dates">
            {from} to {shortDay(event.disclosed_on, today)}
          </span>
        )}
        <strong className="lag__days">{lagText(event)}</strong>
      </div>
    </div>
  );
}
