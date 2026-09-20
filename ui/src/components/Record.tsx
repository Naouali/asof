import { Link } from "react-router-dom";

import type { DisclosureEvent } from "../lib/api";
import { dollars, longDay, plural } from "../lib/format";
import { useAsOf } from "../lib/hooks";
import { actorHref, tickerHref } from "../lib/links";

const KIND_LABEL: Record<DisclosureEvent["kind"], string> = {
  insider: "Company insider",
  congress: "House member",
  fund: "Fund manager",
  unread: "House member",
};

/** "OH02" as a reader writes it: "Ohio's 2nd" is more than the data knows, "OH-02" is not. */
function district(code: string): string {
  const match = /^([A-Z]{2})(\d{2})$/.exec(code);
  return match ? `${match[1]}-${match[2]}` : code;
}

/** Who this is, in a sentence, before what they did. */
function standing(event: DisclosureEvent): string | null {
  if (!event.role) return null;
  if (event.kind === "congress" || event.kind === "unread") return `Represents ${district(event.role)} in the House`;
  if (event.kind === "insider") return event.asset ? `${event.role} of ${event.asset}` : event.role;
  return event.role;
}

interface Step {
  day: string;
  when: string;
  title: string;
  note: string;
  tone: "past" | "public" | "limit";
}

function washingtonTime(iso: string): string {
  return new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "America/New_York", hour12: false }).format(new Date(iso));
}

/** The order things happened in: the trade, the day the law required it to be public, and the day it was. */
function chain(event: DisclosureEvent): Step[] {
  const steps: Step[] = [
    {
      day: event.traded_on,
      when: longDay(event.traded_on, true),
      title: event.kind === "fund" ? "The quarter closed and positions were counted" : "The trade",
      note: event.kind === "fund" ? "A quarterly report says what was held on this day, not when it was bought." : "Nobody outside could know.",
      tone: "past",
    },
  ];
  if (event.due_on) {
    steps.push({
      day: event.due_on,
      when: longDay(event.due_on, true),
      title: "The legal deadline",
      note: event.kind === "insider" ? "Two business days after the trade." : `${event.deadline_days} days after ${event.kind === "fund" ? "the quarter closed" : "the trade"}.`,
      tone: "limit",
    });
  }
  const filedWith = event.kind === "congress" ? "The report was filed with the House Clerk" : "The SEC accepted the filing";
  steps.push({
    day: event.disclosed_on,
    // The Clerk publishes a filing date and no time; the SEC stamps the second.
    when: event.kind === "congress" ? longDay(event.disclosed_on, true) : `${longDay(event.disclosed_on, true)}, ${washingtonTime(event.disclosed_at)} in Washington`,
    title: `${filedWith}. This is when it became public`,
    note: event.late_days > 0 ? `${plural(event.lag_days, "day")} after the trade, ${plural(event.late_days, "day")} past the deadline.` : `${plural(event.lag_days, "day")} after ${event.kind === "fund" ? "the quarter closed" : "the trade"}.`,
    tone: "public",
  });
  // A stable sort: when the deadline and the filing share a day, the filing came first or on time.
  return steps.sort((a, b) => (a.day < b.day ? -1 : a.day > b.day ? 1 : a.tone === "public" ? -1 : 1));
}

/** One disclosure, in full: what happened, and how and when the world found out. */
export function RecordPane({ event }: { event: DisclosureEvent }) {
  const { search } = useAsOf();

  if (event.kind === "unread") {
    return (
      <article className="record">
        <p className="record__context">{standing(event) ?? KIND_LABEL[event.kind]}</p>
        <h2 className="record__headline">{event.actor} filed a transaction report on paper</h2>
        <p className="record__lede">
          It was published as a scanned image with no text, so it can't be read automatically. The trades in it are <strong>missing from this app, not zero</strong>. The only way to know what it says is to open it.
        </p>
        <div className="record__actions">
          {event.source_url && (
            <a className="button button--mark" href={event.source_url} target="_blank" rel="noreferrer">
              Open the scanned report
            </a>
          )}
          <Link className="button" to={actorHref(event, search)}>
            Everything by {event.actor}
          </Link>
        </div>
      </article>
    );
  }

  const status = event.due_on === null ? "No deadline applies" : event.late_days > 0 ? `${plural(event.late_days, "day")} late` : "On time";
  const facts = [
    { label: "Size", value: event.value_usd !== null ? dollars(event.value_usd) : event.kind === "congress" ? "Range only" : "Not stated" },
    { label: "Trade to disclosure", value: plural(event.lag_days, "day") },
    { label: "Against the deadline", value: status, late: event.late_days > 0 },
    { label: "Disclosed by", value: KIND_LABEL[event.kind] },
  ];

  return (
    <article className="record">
      <p className="record__context">{[event.ticker, event.asset].filter(Boolean).join(", ") || "No ticker named in the filing"}</p>
      <h2 className="record__headline">
        {event.actor} {event.verb.toLowerCase()} {event.size}
      </h2>
      <p className="record__lede">
        {[standing(event), event.detail].filter(Boolean).join(". ")}
        {event.role || event.detail ? "." : ""}
        {event.kind === "congress" && " The law asks for a range, not an amount, so the exact size is not known."}
        {event.noise && " This is routine: it is hidden from the feed unless you ask for grants, option exercises and pre-scheduled sales."}
      </p>

      <section className="chain" aria-label="How this became public">
        <h3>How this became public</h3>
        <ol>
          {chain(event).map((step) => (
            <li key={step.title} className={`chain__step chain__step--${step.tone}`}>
              <span className="chain__when">{step.when}</span>
              <span className="chain__rail" aria-hidden="true">
                <span className="chain__dot" />
              </span>
              <span className="chain__what">
                <strong>{step.title}</strong>
                <span>{step.note}</span>
              </span>
            </li>
          ))}
        </ol>
      </section>

      <dl className="facts">
        {facts.map((fact) => (
          <div key={fact.label}>
            <dt>{fact.label}</dt>
            <dd className={fact.late ? "facts__late" : undefined}>{fact.value}</dd>
          </div>
        ))}
      </dl>

      <div className="record__actions">
        {event.source_url && (
          <a className="button button--mark" href={event.source_url} target="_blank" rel="noreferrer">
            Read the filing
          </a>
        )}
        {event.ticker && (
          <Link className="button" to={tickerHref(event.ticker, search)}>
            Open {event.ticker}
          </Link>
        )}
        <Link className="button" to={actorHref(event, search)}>
          Everything by {event.actor}
        </Link>
      </div>
    </article>
  );
}
