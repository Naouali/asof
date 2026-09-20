import { Link } from "react-router-dom";

import type { DisclosureEvent, FeedResponse, TickerResponse } from "../lib/api";
import { shortDay } from "../lib/format";
import { useApi, useAsOf } from "../lib/hooks";
import { tickerHref } from "../lib/links";
import { lagText } from "./LagBar";
import { Mark } from "./Mark";

const SHOW = 5;

function Line({ event, today, by }: { event: DisclosureEvent; today: string; by: "actor" | "ticker" }) {
  return (
    <li className="context__line">
      <Mark kind={event.kind} direction={event.direction} />
      <span>
        <strong>{by === "actor" ? event.actor : (event.ticker ?? event.asset ?? "No ticker")}</strong> {event.verb.toLowerCase()} {event.size}
        <span className="context__when">
          {" "}
          {shortDay(event.disclosed_on, today)}, {lagText(event)} after
        </span>
      </span>
    </li>
  );
}

/** What surrounds the selected disclosure: the rest of the company, and the rest of the person. */
export function ContextPane({ event, today }: { event: DisclosureEvent; today: string }) {
  const { asOf, search } = useAsOf();
  const company = useApi<TickerResponse>(event.ticker ? `/api/tickers/${encodeURIComponent(event.ticker)}` : null, { as_of: asOf });
  const person = useApi<FeedResponse>("/api/feed", { as_of: asOf, days: 365, actor: event.actor_id });

  const others = (company.data?.events ?? []).filter((other) => other.id !== event.id && !other.noise && other.actor_id !== event.actor_id);
  const record = (person.data?.groups ?? []).flatMap((group) => group.events).filter((other) => other.id !== event.id && other.kind !== "unread");
  const choices = record.filter((other) => !other.noise);

  return (
    <div className="context">
      {event.ticker && (
        <section>
          <h3>Also at {event.ticker}</h3>
          {company.data && <p className="context__brief">{company.data.brief.join(" ")}</p>}
          {others.length > 0 && (
            <ul>
              {others.slice(0, SHOW).map((other) => (
                <Line key={other.id} event={other} today={today} by="actor" />
              ))}
            </ul>
          )}
          <Link className="context__more" to={tickerHref(event.ticker, search)}>
            Open {event.ticker}, with the price chart
          </Link>
        </section>
      )}

      <section>
        <h3>{event.kind === "fund" ? "This fund's record" : "Their record"}</h3>
        {person.data && (
          <p className="context__brief">
            {record.length === 0
              ? "Nothing else in the last year."
              : `${record.length.toLocaleString("en-US")} other ${record.length === 1 ? "disclosure" : "disclosures"} in the last year${choices.length < record.length ? `, ${choices.length} of them by choice rather than routine` : ""}.`}
          </p>
        )}
        {choices.length > 0 && (
          <ul>
            {choices.slice(0, SHOW).map((other) => (
              <Line key={other.id} event={other} today={today} by="ticker" />
            ))}
          </ul>
        )}
      </section>

      <section>
        <h3>What you can't see here</h3>
        <p className="context__brief">Senate trades, House reports filed on paper, and anything a fund has done since its last quarter end. Short positions are never disclosed at all.</p>
      </section>
    </div>
  );
}
