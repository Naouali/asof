import { useEffect, useMemo, useRef, useState } from "react";

import { monthName, parseDay, shortDay, signedPercent } from "../lib/format";

const HEIGHT = 300;
/** Narrower than a phone's panel: the drawing must never be wider than its box. */
const MIN_WIDTH = 260;
const PAD = { top: 18, right: 64, bottom: 30, left: 46 };

export interface ReturnPoint {
  date: string;
  member_pct: number;
  follower_pct: number | null;
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
 * A portfolio's return over time, in percent, on one axis.
 *
 * Two lines: the member's, in ink, from the prices on the days they traded; and a
 * follower's, in the yellow this app uses for "public", from the prices on the days
 * each trade was disclosed. The follower's is dashed too, so the two are told apart
 * without colour, and each is labelled at its end. The gap between them is what
 * the delay cost.
 */
export function ReturnChart({ points, today, who }: { points: ReturnPoint[]; today: string; who: string }) {
  const frame = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(820);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const node = frame.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => entry && setWidth(Math.max(MIN_WIDTH, Math.round(entry.contentRect.width))));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const model = useMemo(() => {
    const values = points.flatMap((point) => (point.follower_pct === null ? [point.member_pct] : [point.member_pct, point.follower_pct]));
    const low = Math.min(0, ...values);
    const high = Math.max(0, ...values);
    const margin = (high - low || 1) * 0.08;
    const min = low - margin;
    const max = high + margin;
    const innerW = width - PAD.left - PAD.right;
    const innerH = HEIGHT - PAD.top - PAD.bottom;
    const x = (index: number) => PAD.left + (points.length > 1 ? (index / (points.length - 1)) * innerW : 0);
    const y = (value: number) => PAD.top + (1 - (value - min) / (max - min)) * innerH;

    const line = (pick: (point: ReturnPoint) => number | null) => {
      let path = "";
      let pen = false;
      points.forEach((point, index) => {
        const value = pick(point);
        if (value === null) {
          pen = false;
          return;
        }
        path += `${pen ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
        pen = true;
      });
      return path;
    };

    const months: { x: number; label: string }[] = [];
    let last = -1;
    points.forEach((point, index) => {
      const date = parseDay(point.date);
      const month = date.getUTCMonth();
      if (month === last) return;
      last = month;
      if (index === 0) return;
      // Every month carries its year: a portfolio's line runs across two or three of them.
      months.push({ x: x(index), label: `${monthName(month).slice(0, 3)} ${String(date.getUTCFullYear()).slice(2)}` });
    });
    // Few enough labels that they never touch.
    const every = Math.ceil(months.length / Math.max(1, Math.floor(innerW / 76)));
    return { x, y, innerW, member: line((point) => point.member_pct), follower: line((point) => point.follower_pct), months: months.filter((_, index) => index % every === 0), ticks: niceTicks(min, max) };
  }, [points, width]);

  const end = points[points.length - 1];
  if (!end) return null;
  const hovered = hover === null ? null : points[hover];
  // Keep the two end labels from sitting on top of each other.
  const memberY = model.y(end.member_pct);
  let followerY = end.follower_pct === null ? null : model.y(end.follower_pct);
  if (followerY !== null && Math.abs(followerY - memberY) < 16) followerY = memberY + (followerY >= memberY ? 16 : -16);

  return (
    <div className="chart" ref={frame}>
      <svg width={width} height={HEIGHT} role="img" aria-label={`Return over time. ${who}: ${signedPercent(end.member_pct)}.${end.follower_pct === null ? "" : ` Copied on disclosure: ${signedPercent(end.follower_pct)}.`}`}>
        {model.ticks.map((tick) => (
          <g key={tick}>
            <line className={tick === 0 ? "chart__zero" : "chart__grid"} x1={PAD.left} x2={width - PAD.right} y1={model.y(tick)} y2={model.y(tick)} />
            <text className="chart__tick" x={PAD.left - 8} y={model.y(tick) + 4} textAnchor="end">
              {tick > 0 ? "+" : tick < 0 ? "−" : ""}
              {Math.abs(tick)}%
            </text>
          </g>
        ))}
        {model.months.map((month) => (
          <text key={`${month.label}${month.x}`} className="chart__tick" x={month.x} y={HEIGHT - 8}>
            {month.label}
          </text>
        ))}

        {model.follower && <path className="chart__line chart__line--follower" d={model.follower} />}
        <path className="chart__line" d={model.member} />

        <text className="chart__endlabel" x={width - PAD.right + 8} y={memberY + 4}>
          {signedPercent(end.member_pct)}
        </text>
        {followerY !== null && end.follower_pct !== null && (
          <text className="chart__endlabel chart__endlabel--follower" x={width - PAD.right + 8} y={followerY + 4}>
            {signedPercent(end.follower_pct)}
          </text>
        )}

        <rect
          x={PAD.left}
          y={PAD.top}
          width={model.innerW}
          height={HEIGHT - PAD.top - PAD.bottom}
          fill="transparent"
          onMouseMove={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            const share = (event.clientX - box.left) / box.width;
            setHover(Math.max(0, Math.min(points.length - 1, Math.round(share * (points.length - 1)))));
          }}
          onMouseLeave={() => setHover(null)}
        />
        {hovered && hover !== null && (
          <g pointerEvents="none">
            <line className="chart__cross" x1={model.x(hover)} x2={model.x(hover)} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
            <circle cx={model.x(hover)} cy={model.y(hovered.member_pct)} r={4} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
            {hovered.follower_pct !== null && <circle cx={model.x(hover)} cy={model.y(hovered.follower_pct)} r={4} fill="var(--mark)" stroke="var(--surface)" strokeWidth={2} />}
          </g>
        )}
      </svg>
      {hovered && hover !== null && (
        <div className="chart__tip" style={{ left: Math.min(Math.max(model.x(hover) - 90, PAD.left), width - 230) }}>
          <strong>{signedPercent(hovered.member_pct)}</strong> on {shortDay(hovered.date, today)}
          {hovered.follower_pct !== null && (
            <>
              , <strong>{signedPercent(hovered.follower_pct)}</strong> if copied
            </>
          )}
        </div>
      )}
    </div>
  );
}
