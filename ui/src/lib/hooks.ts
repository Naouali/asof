import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { getJson } from "./api";
import type { Rules } from "./api";

export interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Fetch JSON whenever the path or its parameters change, dropping stale answers. */
export function useApi<T>(path: string | null, params: Record<string, string | number | null | undefined>): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null, loading: path !== null });
  const key = path === null ? null : `${path}?${JSON.stringify(params)}`;

  useEffect(() => {
    if (path === null) return;
    const controller = new AbortController();
    setState((previous) => ({ ...previous, loading: true, error: null }));
    getJson<T>(path, params, controller.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({ data: null, error: error instanceof Error ? error.message : "Request failed", loading: false });
      });
    return () => controller.abort();
    // `key` is the whole identity of the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return state;
}

/**
 * The browser tab's name. Four tabs all called "asof" cannot be told apart, and
 * neither can a history full of them; a past date is part of what the page is.
 */
export function useTitle(title: string | null): void {
  const { asOf } = useAsOf();
  useEffect(() => {
    const parts = [title, asOf ? `as of ${asOf}` : null].filter(Boolean).join(", ");
    document.title = parts ? `${parts} · asof` : "asof";
  }, [title, asOf]);
}

let rulesRequest: Promise<Rules> | null = null;

/**
 * The app's thresholds, fetched once for the life of the page. Null until they
 * arrive: a sentence that needs one waits for it, and never falls back to a guess.
 */
export function useRules(): Rules | null {
  const [rules, setRules] = useState<Rules | null>(null);
  useEffect(() => {
    let live = true;
    rulesRequest ??= getJson<Rules>("/api/rules", {});
    rulesRequest
      .then((found) => {
        if (live) setRules(found);
      })
      .catch(() => {
        rulesRequest = null; // let the next page try again
      });
    return () => {
      live = false;
    };
  }, []);
  return rules;
}

/**
 * The as-of date lives in the URL, as `?asof=2026-08-14`.
 *
 * It is the one piece of state every page shares, and the URL is the right home
 * for it: a view of the past can be bookmarked and sent to a colleague, it
 * survives a reload, and the back button undoes a jump in time.
 */
export function useAsOf(): { asOf: string | null; setAsOf: (day: string | null) => void; search: string } {
  const [params, setParams] = useSearchParams();
  const raw = params.get("asof");
  const asOf = raw && /^\d{4}-\d{2}-\d{2}$/.test(raw) ? raw : null;

  const setAsOf = useCallback(
    (day: string | null) => {
      setParams((current) => {
        const next = new URLSearchParams(current);
        if (day) next.set("asof", day);
        else next.delete("asof");
        return next;
      });
    },
    [setParams],
  );

  return { asOf, setAsOf, search: asOf ? `?asof=${asOf}` : "" };
}
