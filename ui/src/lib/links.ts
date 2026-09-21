import type { DisclosureEvent } from "./api";

/** Where the dashboard lives. The bare address redirects here. */
export const DASHBOARD = "/dashboard";

/** The feed, narrowed to one person or fund, keeping the as-of date. */
export function actorHref(actor: Pick<DisclosureEvent, "actor_id">, search: string): string {
  const params = new URLSearchParams(search);
  params.set("actor", actor.actor_id);
  return `${DASHBOARD}?${params}`;
}

/** A member's compiled portfolio. Only members of Congress have one. */
export function portfolioHref(actorId: string, search: string): string {
  const params = new URLSearchParams(search);
  params.set("actor", actorId);
  return `/portfolios?${params}`;
}

export function hasPortfolio(actor: Pick<DisclosureEvent, "actor_id">): boolean {
  return actor.actor_id.startsWith("congress:");
}

export function tickerHref(ticker: string, search: string): string {
  return `/t/${encodeURIComponent(ticker)}${search}`;
}
