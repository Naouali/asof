import { useEffect, useMemo, useRef, useState } from "react";

import type { DisclosureEvent } from "../lib/api";
import { monthName, parseDay, shortDay } from "../lib/format";

const DEFAULT_HEIGHT = 340;
const PAD = { top: 16, right: 14, bottom: 30, left: 52 };
const BUY = "var(--buy)";
const SELL = "var(--sell)";

interface Point {
  date: string;
  close: number;
}

function niceTicks(low: number, high: number, count = 5): number[] {
  const rough = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(rough || 1));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((value) => value >= rough) ?? power;
  const ticks: number[] = [];
  for (let value = Math.ceil(low / step) * step; value <= high + 1e-9; value += step) ticks.push(Number(value.toFixed(6)));
  return ticks;
}

/**
 * Daily close, with every disclosure drawn as a chord: a hollow dot on the day of
 * the trade, a filled mark on the day it became public, and a line between them.
 * The slope of that line is what the price did before anyone outside could act.
 */
export function PriceChart({
  prices,
  events,
  selected,
  onSelect,
  today,
  height = DEFAULT_HEIGHT,
}: {
  prices: Point[];
  events: DisclosureEvent[];
  selected: string | null;
  onSelect: (id: string) => void;
  today: string;
  /** The landing page draws the same chart in a smaller panel. */
  height?: number;
}) {
  const frame = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(820);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const node = frame.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => entry && setWidth(Math.max(360, Math.round(entry.contentRect.width))));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const model = useMemo(() => {
    const closes = prices.map((point) => point.close);
    const min = Math.min(...closes);
    const max = Math.max(...closes);
    const margin = (max - min) * 0.08 || 1;
    const low = min - margin;
    const high = max + margin;
    const innerW = width - PAD.left - PAD.right;
    const innerH = height - PAD.top - PAD.bottom;
    const x = (index: number) => PAD.left + (innerW * index) / Math.max(1, prices.length - 1);
    const y = (value: number) => PAD.top + innerH - (innerH * (value - low)) / (high - low);

    // The last trading day on or before a calendar day: trades and filings both
    // land on weekends and holidays, when there is no close to pin them to.
    const indexOn = (day: string): number | null => {
      const first = prices[0];
      if (!first || day < first.date) return null;
      let found = 0;
      for (let index = 0; index < prices.length; index += 1) {
        if ((prices[index]?.date ?? "") <= day) found = index;
        else break;
      }
      return found;
    };

    const path = prices.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)} ${y(point.close).toFixed(1)}`).join(" ");
    const months: { x: number; label: string }[] = [];
    prices.forEach((point, index) => {
      const previous = prices[index - 1];
      if (previous && parseDay(point.date).getUTCMonth() !== parseDay(previous.date).getUTCMonth()) {
        months.push({ x: x(index), label: monthName(parseDay(point.date).getUTCMonth()).slice(0, 3) });
      }
    });

    const marks = events.flatMap((event) => {
      const to = indexOn(event.disclosed_on);
      if (to === null) return [];
      const from = indexOn(event.traded_on);
      const end = prices[to];
      const start = from === null ? null : prices[from];
      if (!end) return [];
      return [
        {
          event,
          x1: x(to),
          y1: y(end.close),
          x0: start && from !== null ? x(from) : PAD.left,
          y0: start ? y(start.close) : y(end.close),
          clipped: from === null,
        },
      ];
    });

    return { x, y, path, months, marks, ticks: niceTicks(low, high), innerW };
  }, [prices, events, width, height]);

  const hovered = hover === null ? null : prices[hover];

  return (
    <div className="chart" ref={frame}>
      <svg width={width} height={height} role="img" aria-label={`Daily closing price with ${events.length} disclosures marked. The same disclosures are listed in the table below.`}>
        {model.ticks.map((tick) => (
          <g key={tick}>
            <line className="chart__grid" x1={PAD.left} x2={width - PAD.right} y1={model.y(tick)} y2={model.y(tick)} />
            <text className="chart__tick" x={PAD.left - 8} y={model.y(tick) + 4} textAnchor="end">
              ${tick.toLocaleString("en-US")}
            </text>
          </g>
        ))}
        {model.months.map((month) => (
          <text key={`${month.label}${month.x}`} className="chart__tick" x={month.x} y={height - 8}>
            {month.label}
          </text>
        ))}

        <path className="chart__line" d={model.path} />

        <rect
          x={PAD.left}
          y={PAD.top}
          width={model.innerW}
          height={height - PAD.top - PAD.bottom}
          fill="transparent"
          onMouseMove={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            const share = (event.clientX - box.left) / box.width;
            setHover(Math.max(0, Math.min(prices.length - 1, Math.round(share * (prices.length - 1)))));
          }}
          onMouseLeave={() => setHover(null)}
        />

        {model.marks.map(({ event, x0, y0, x1, y1, clipped }) => {
          const on = event.id === selected;
          // Contracts are drawn in ink: blue and orange mean bought and sold.
          const color = event.kind === "contract" ? "var(--ink)" : event.direction === "sell" ? SELL : BUY;
          const fill = event.direction === "sell" ? "var(--surface)" : color;
          const ring = event.direction === "sell" ? color : "var(--surface)";
          const size = on ? 8 : 6.5;
          return (
            <g key={event.id} className={`chart__mark${on ? " chart__mark--on" : ""}`}>
              {/* A trade from before the price history has no start to draw a chord from. */}
              {!clipped && <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={color} strokeWidth={on ? 3 : 2} opacity={on ? 1 : 0.5} />}
              {!clipped && <circle cx={x0} cy={y0} r={4} fill="var(--surface)" stroke={color} strokeWidth={2} />}
              {event.kind === "insider" && <circle cx={x1} cy={y1} r={size} fill={fill} stroke={ring} strokeWidth={2} />}
              {event.kind === "congress" && <rect x={x1 - size} y={y1 - size} width={size * 2} height={size * 2} fill={fill} stroke={ring} strokeWidth={2} />}
              {event.kind === "fund" && (
                <rect x={x1 - size} y={y1 - size} width={size * 2} height={size * 2} fill={fill} stroke={ring} strokeWidth={2} transform={`rotate(45 ${x1} ${y1})`} />
              )}
              {event.kind === "contract" && (
                <polygon
                  points={`${x1},${y1 - size - 1} ${x1 + size + 1},${y1 + size} ${x1 - size - 1},${y1 + size}`}
                  fill={fill}
                  stroke={event.direction === "sell" ? color : ring}
                  strokeWidth={2}
                  strokeLinejoin="round"
                />
              )}
              <circle className="chart__hit" cx={x1} cy={y1} r={16} onMouseEnter={() => onSelect(event.id)} onClick={() => onSelect(event.id)} />
            </g>
          );
        })}

        {hovered && hover !== null && (
          <g pointerEvents="none">
            <line className="chart__cross" x1={model.x(hover)} x2={model.x(hover)} y1={PAD.top} y2={height - PAD.bottom} />
            <circle cx={model.x(hover)} cy={model.y(hovered.close)} r={4} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
          </g>
        )}
      </svg>
      {hovered && hover !== null && (
        <div className="chart__tip" style={{ left: Math.min(Math.max(model.x(hover) - 70, PAD.left), width - 170) }}>
          <strong>${hovered.close.toFixed(2)}</strong> on {shortDay(hovered.date, today)}
        </div>
      )}
    </div>
  );
}
