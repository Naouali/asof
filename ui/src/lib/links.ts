import type { DisclosureEvent } from "./api";

/** The feed, narrowed to one person or fund, keeping the as-of date. */
export function actorHref(actor: Pick<DisclosureEvent, "actor_id">, search: string): string {
  const params = new URLSearchParams(search);
  params.set("actor", actor.actor_id);
  return `/?${params}`;
}

export function tickerHref(ticker: string, search: string): string {
  return `/t/${encodeURIComponent(ticker)}${search}`;
}
