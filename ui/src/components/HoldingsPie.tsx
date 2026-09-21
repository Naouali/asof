import { useState } from "react";
import { Link } from "react-router-dom";

import type { Holding } from "../lib/api";
import { dollars, plural } from "../lib/format";
import { tickerHref } from "../lib/links";

const SIZE = 220;
const OUTER = 104;
const INNER = 62;
/** The gap between slices, in degrees at the outer edge: the ground showing through. */
const GAP = 1.4;

interface Slice {
  key: string;
  label: string;
  percent: number;
  note: string;
  holding: Holding | null;
}

function point(angle: number, radius: number): string {
  const radians = ((angle - 90) * Math.PI) / 180;
  return `${(SIZE / 2 + radius * Math.cos(radians)).toFixed(2)},${(SIZE / 2 + radius * Math.sin(radians)).toFixed(2)}`;
}

/** A ring segment from one angle to another, clockwise from twelve o'clock. */
function ring(from: number, to: number): string {
  const large = to - from > 180 ? 1 : 0;
  return `M${point(from, OUTER)} A${OUTER},${OUTER} 0 ${large} 1 ${point(to, OUTER)} L${point(to, INNER)} A${INNER},${INNER} 0 ${large} 0 ${point(from, INNER)} Z`;
}

/**
 * The largest holdings as shares of the whole portfolio.
 *
 * The slices are shades of one ink, lightest for the largest, because they are
 * ranked amounts and not categories -- and because blue and orange already mean
 * bought and sold in this app. Everything outside the largest few is one hollow
 * slice: without it the ring would claim the top holdings ARE the portfolio. Each
 * slice is named in the list beside the ring, so nothing is told by shade alone.
 */
export function HoldingsPie({ holdings, count, search }: { holdings: Holding[]; count: number; search: string }) {
  const [active, setActive] = useState<string | null>(null);
  const top = holdings.slice(0, count);
  const rest = holdings.length - top.length;
  const shown = top.reduce((sum, holding) => sum + holding.weight_pct, 0);

  const slices: Slice[] = top.map((holding) => ({
    key: holding.ticker,
    label: holding.ticker,
    percent: holding.weight_pct,
    note: `${dollars(holding.mid_usd)}, between ${dollars(holding.low_usd)} and ${dollars(holding.high_usd)}`,
    holding,
  }));
  if (rest > 0) slices.push({ key: "__rest", label: plural(rest, "other"), percent: Math.max(0, 100 - shown), note: `the other ${plural(rest, "ticker")} together`, holding: null });

  const shade = (index: number) => `color-mix(in srgb, var(--ink) ${Math.round(100 - (top.length > 1 ? index / (top.length - 1) : 0) * 100)}%, var(--pie-deep))`;
  const focus = slices.find((slice) => slice.key === active) ?? null;

  let cursor = 0;
  return (
    <figure className="pie" aria-label={`The ${plural(top.length, "largest holding")} as shares of the portfolio`}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={slices.map((slice) => `${slice.label} ${slice.percent.toFixed(1)}%`).join(", ")}>
        {slices.map((slice, index) => {
          const from = cursor;
          const sweep = (slice.percent / 100) * 360;
          cursor += sweep;
          if (sweep <= 0) return null;
          const hollow = slice.holding === null;
          const on = active === slice.key;
          // One slice that is the whole ring cannot be drawn as an arc back to its own start.
          const whole = sweep >= 359.9;
          const gap = whole || slices.length === 1 ? 0 : Math.min(GAP, sweep / 3);
          const shape = whole ? `${ring(0, 180)} ${ring(180, 360)}` : ring(from + gap / 2, from + sweep - gap / 2);
          return (
            <path
              key={slice.key}
              className={`pie__slice${hollow ? " pie__slice--rest" : ""}${on ? " pie__slice--on" : ""}${active && !on ? " pie__slice--dim" : ""}`}
              d={shape}
              fill={hollow ? "transparent" : shade(index)}
              onMouseEnter={() => setActive(slice.key)}
              onMouseLeave={() => setActive(null)}
            >
              <title>{`${slice.label}: ${slice.percent.toFixed(1)}% of the portfolio, ${slice.note}`}</title>
            </path>
          );
        })}
        <text className="pie__figure" x={SIZE / 2} y={SIZE / 2 - 2} textAnchor="middle">
          {(focus ? focus.percent : shown).toFixed(focus ? 1 : 0)}%
        </text>
        <text className="pie__caption" x={SIZE / 2} y={SIZE / 2 + 18} textAnchor="middle">
          {focus ? focus.label : `in the top ${top.length}`}
        </text>
      </svg>
      <figcaption>
        <ol className="pie__list">
          {slices.map((slice, index) => (
            <li key={slice.key} className={active === slice.key ? "pie__row pie__row--on" : "pie__row"} onMouseEnter={() => setActive(slice.key)} onMouseLeave={() => setActive(null)}>
              <span className={`pie__swatch${slice.holding ? "" : " pie__swatch--rest"}`} style={slice.holding ? { background: shade(index) } : undefined} aria-hidden="true" />
              {slice.holding ? <Link to={tickerHref(slice.holding.ticker, search)}>{slice.label}</Link> : <span>{slice.label}</span>}
              <strong>{slice.percent.toFixed(1)}%</strong>
            </li>
          ))}
        </ol>
      </figcaption>
    </figure>
  );
}
