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
  kind: "ticker" | "person" | "fund" | "portfolio" | "agency";
  key: string;
  label: string;
  note: string | null;
}

export interface Unpriced {
  /** Tickers named in disclosures. */
  wanted: number;
  /** How many of them the lake has prices for: every return figure rests on these. */
  priced: number;
  refused: { symbol: string; reason: string; attempts: number; retry_after: string }[];
  refused_total: number;
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
    jobs?: { fetcher: string; rows: number; ok: boolean; error: string | null; skipped_reason: string | null; failed_symbols?: { symbol: string; reason: string }[] }[];
  }[];
  unpriceable: Unpriced;
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

// ---------------------------------------------------------------- portfolios --
export interface PortfolioMember {
  actor_id: string;
  actor: string;
  role: string | null;
  chamber: string;
  trades: number;
  tickers: number;
  last_disclosed: string | null;
}

export interface PortfolioMembers {
  as_of: string;
  today: string;
  members: PortfolioMember[];
}

/** What a member has bought and kept in one ticker: midpoints netted, with the honest low and high beside it. */
export interface Holding {
  ticker: string;
  asset: string | null;
  weight_pct: number;
  mid_usd: number;
  low_usd: number;
  high_usd: number;
  purchases: number;
  sales: number;
  first_bought: string;
  last_trade: string | null;
  accounts: string[];
  /** Null where the lake holds no prices for the ticker. */
  return_since_bought_pct: number | null;
  /** From the day the purchase became public: the return a follower could have had. */
  return_since_public_pct: number | null;
}

export interface SoldPosition {
  ticker: string;
  asset: string | null;
  /** Exit price over entry price. Null without prices, or while shares are still held. */
  return_pct: number | null;
  sold_low_usd: number;
  sold_high_usd: number;
  last_trade: string | null;
}

export interface PortfolioResponse {
  as_of: string;
  today: string;
  actor_id: string;
  actor: string;
  role: string | null;
  chamber: string;
  /** The day the first report the lake holds for this chamber became public. */
  since: string | null;
  trades: number;
  mid_usd: number;
  low_usd: number;
  high_usd: number;
  holdings: Holding[];
  closed: SoldPosition[];
  held_before: SoldPosition[];
  left_out: { options: number; exchanges: number; no_ticker: number };
  /**
   * The return over time: closed trades at their exit over their entry price, open
   * ones marked to each day's close, over everything put in so far. Null without prices.
   */
  performance: {
    points: { date: string; member_pct: number; follower_pct: number | null }[];
    member_pct: number;
    follower_pct: number | null;
    purchases: number;
    tickers: number;
  } | null;
  priced_share: number;
  returns: { since_bought_pct: number; since_public_pct: number } | null;
}

/**
 * Every threshold the app applies, from /api/rules. A sentence that mentions one
 * prints it from here: a number typed into the interface as well would one day
 * disagree with the number the app is actually using.
 */
export interface Rules {
  deadlines: { congress_days: number; fund_days: number; insider_business_days: number };
  contracts: {
    min_action_usd: number;
    feed_min_usd: number;
    publication_lag_days: number;
    defense_embargo_days: number;
    companies: number | null;
    listed_per_ticker: number;
    on_chart: number;
  };
  feed: { fund_changes_per_filing: number; fund_min_change_pct: number };
  analytics: { min_sample: number; max_lag_days: number };
  portfolios: { min_priced_share_pct: number };
}

// ----------------------------------------------------------------- analytics --
export interface TradedResponse {
  as_of: string;
  today: string;
  days: number;
  kind: string | null;
  trades: number;
  tickers_total: number;
  /** Distinct filers on each side: the one count every source supports. */
  tickers: { ticker: string; name: string | null; buyers: number; sellers: number }[];
  people: { actor_id: string; actor: string; role: string | null; kind: "insider" | "congress"; buys: number; sells: number; tickers: number }[];
}

/** Percent price moves between a trade and its disclosure. Null under five trades. */
export interface Spread {
  n: number;
  median: number | null;
  p10: number | null;
  p25: number | null;
  p75: number | null;
  p90: number | null;
}

export interface PriceMovesResponse {
  as_of: string;
  today: string;
  days: number;
  trades: number;
  measured: number;
  priced_tickers: number;
  groups: { key: "insider" | "house" | "senate" | "fund"; label: string; buy: Spread; sell: Spread }[];
  examples: DisclosureEvent[];
}

export interface ContractTotal {
  name: string;
  net_usd: number;
  defense_usd: number;
  actions: number;
}

export interface ContractsAnalytics {
  as_of: string;
  today: string;
  is_live: boolean;
  actions: number;
  net_usd: number;
  defense_usd: number;
  companies: ContractTotal[];
  agencies: ContractTotal[];
  months: { month: string; committed_usd: number; taken_back_usd: number }[];
  /** Signed by the as-of date and not public yet. Null when looking at today. */
  hidden: { actions: number; net_usd: number; defense_actions: number } | null;
}

// -------------------------------------------------------------- track record --
/** A measured record: the mean and what the size of the sample can support. */
export interface Record_ {
  trades: number;
  mean_excess_pct: number;
  low_pct: number | null;
  high_pct: number | null;
  median_excess_pct: number;
  beat_rate: number;
  /** True only when that interval is clear of zero. */
  distinguishable: boolean;
}

export interface FilerRecord extends Record_ {
  actor_id: string;
  actor: string;
  role: string | null;
  /** The same trades entered the day the filer made them: the delay's cost. */
  own_mean_excess_pct: number | null;
  first: string;
  last: string;
  ranked: boolean;
}

export interface TrackRecords {
  as_of: string;
  today: string;
  horizon_days: number;
  benchmark: string;
  min_trades: number;
  measured: number;
  filers: number;
  ranked: number;
  unpriced: number;
  unfinished: number;
  everyone: Record_ | null;
  standouts: number;
  /** How good the leader would look if nobody had any skill at all. */
  luck: { best_mean_excess_pct: number; shuffles: number; as_good_by_chance: number } | null;
  expected_by_luck: number;
  people: FilerRecord[];
  why_empty: string | null;
}

