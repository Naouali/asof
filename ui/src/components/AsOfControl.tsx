import { useEffect, useId, useRef, useState } from "react";

import { addDays, lastFundDeadline, stampDay } from "../lib/format";
import { useAsOf, useRules } from "../lib/hooks";

/**
 * The global as-of date. Everything on every page is read through it.
 *
 * `today` is the server's idea of today in Washington, not the browser's: near
 * midnight the two disagree, and the server is the one that read the lake.
 */
export function AsOfControl({ today }: { today: string }) {
  const { asOf, setAsOf } = useAsOf();
  const rules = useRules();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const inputId = useId();
  const shown = asOf ?? today;

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", close);
    root.current?.addEventListener("keydown", escape);
    const node = root.current;
    return () => {
      document.removeEventListener("mousedown", close);
      node?.removeEventListener("keydown", escape);
    };
  }, [open]);

  const choose = (day: string | null) => {
    setAsOf(day === null || day >= today ? null : day);
    setOpen(false);
  };

  const presets: { label: string; note: string; day: string | null }[] = [
    { label: "Today", note: "Everything disclosed so far", day: null },
    // Offered once the deadline is known: it comes from the API, not from here.
    ...(rules ? [{ label: stampDay(lastFundDeadline(today, rules.deadlines.fund_days)), note: "The last day quarterly fund holdings were due", day: lastFundDeadline(today, rules.deadlines.fund_days) }] : []),
    { label: stampDay(addDays(today, -30)), note: "One month ago", day: addDays(today, -30) },
    { label: stampDay(addDays(today, -365)), note: "One year ago", day: addDays(today, -365) },
  ];

  return (
    <div className="asof" ref={root}>
      <button
        type="button"
        className={`asof__button${asOf ? " asof__button--past" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="asof__label">As of</span>
        <span className="asof__value">{stampDay(shown)}</span>
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      </button>
      {open && (
        <div className="asof__menu" role="dialog" aria-label="Choose an as-of date">
          <p className="asof__help">See everything as it could be known at the end of a day. Nothing disclosed later is shown.</p>
          {presets.map((preset) => (
            <button key={preset.note} type="button" className="asof__preset" onClick={() => choose(preset.day)}>
              <span className="asof__preset-label">{preset.label}</span>
              <span className="asof__preset-note">{preset.note}</span>
            </button>
          ))}
          <label className="asof__custom" htmlFor={inputId}>
            <span>Another day</span>
            <input
              id={inputId}
              type="date"
              max={today}
              value={shown}
              onChange={(event) => event.target.value && choose(event.target.value)}
            />
          </label>
        </div>
      )}
    </div>
  );
}
