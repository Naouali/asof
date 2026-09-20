import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { getJson } from "./api";

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
