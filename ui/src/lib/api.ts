// The API's response shapes. Mirrors src/quantlab/api/models.py, which is the
// contract; /api/docs renders it.

export type Kind = "insider" | "congress" | "fund" | "unread" | "contract";
/** For a contract: "buy" is money committed to the company, "sell" is money taken back. */
export type Direction = "buy" | "sell" | "none";

export interface DisclosureEvent {
  id: string;
  kind: Kind;
  ticker: string | null;
  asset: string | null;
  actor: string;
  actor_id: string;
  role: string | null;
  direction: Direction;
  verb: string;
  size: string;
  detail: string | null;
  value_usd: number | null;
  traded_on: string;
  disclosed_on: string;
  disclosed_at: string;
  lag_days: number;
  deadline_days: number | null;
  due_on: string | null;
  late_days: number;
  noise: boolean;
  source_url: string | null;
  price_move_pct: number | null;
}

export interface FeedResponse {
  as_of: string;
  today: string;
  is_live: boolean;
  days: number;
  actor: { id: string; name: string; role: string | null } | null;
  groups: { date: string; events: DisclosureEvent[] }[];
  insights: { ticker: string; text: string }[];
  /** Had already happened on the as-of date, not public until after it. Only ever drawn behind the curtain. */
  beyond: DisclosureEvent[];
  beyond_total: number;
  /** Contract actions behind the curtain, counted apart from the trades. */
  beyond_contracts_total: number;
  coverage: {
    unread_reports: number;
    fund_period: string | null;
    fund_period_age_days: number | null;
  };
  empty_lake: boolean;
}

export interface Holder {
  manager_id: string;
  manager: string;
  shares: number;
  change: number | null;
  value_usd: number;
  period: string;
  disclosed_on: string;
  age_days: number;
}

/** A company's federal contract actions made public in the last year. */
export interface Contracts {
  actions: number;
  /** Money committed less money taken back. Not revenue: it is spent over years. */
  net_usd: number;
  taken_back: number;
  defense_share: number | null;
  agencies: { name: string; net_usd: number }[];
  /** The largest actions, largest first. */
  events: DisclosureEvent[];
  /** Ids of the ones drawn on the price chart. */
  on_chart: string[];
}

export interface TickerResponse {
  as_of: string;
  today: string;
  ticker: string;
  name: string | null;
  known: boolean;
  prices: { date: string; close: number }[];
  events: DisclosureEvent[];
  holders: Holder[];
  contracts: Contracts | null;
  fails_to_deliver: { quantity: number; settled_on: string; posted_on: string } | null;
  brief: string[];
}

export interface SearchHit {
  kind: "ticker" | "person" | "fund";
  key: string;
  label: string;
  note: string | null;
}

export interface DataHealth {
  datasets: {
    source: string;
    dataset: string;
    rows: number;
    symbols: number;
    first: string | null;
    last: string | null;
    newest_known: string | null;
    age_days: number | null;
    bytes: number;
  }[];
  unread_reports: DisclosureEvent[];
  runs: {
    started_at?: string;
    incremental?: boolean;
    jobs?: { fetcher: string; rows: number; ok: boolean; error: string | null; skipped_reason: string | null }[];
  }[];
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function getJson<T>(path: string, params: Record<string, string | number | null | undefined>, signal?: AbortSignal): Promise<T> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") query.set(key, String(value));
  }
  const suffix = query.size ? `?${query}` : "";
  const response = await fetch(`${path}${suffix}`, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}
