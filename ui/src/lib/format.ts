// Dates arrive as YYYY-MM-DD and mean a Washington calendar day. They are parsed
// and printed in UTC so that a reader in Tokyo and a reader in Lisbon see the same
// day written down, rather than each seeing it shifted by their own offset.

const DAY = 86_400_000;
const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

export function parseDay(iso: string): Date {
  const [year, month, day] = iso.slice(0, 10).split("-").map(Number);
  return new Date(Date.UTC(year ?? 1970, (month ?? 1) - 1, day ?? 1));
}

export function isoDay(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export function addDays(iso: string, days: number): string {
  return isoDay(new Date(parseDay(iso).getTime() + days * DAY));
}

export function daysBetween(fromIso: string, toIso: string): number {
  return Math.round((parseDay(toIso).getTime() - parseDay(fromIso).getTime()) / DAY);
}

/** "Thursday 17 September" */
export function longDay(iso: string, withYear = false): string {
  const date = parseDay(iso);
  const text = `${WEEKDAYS[date.getUTCDay()]} ${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]}`;
  return withYear ? `${text} ${date.getUTCFullYear()}` : text;
}

/** "17 Sep", or "17 Sep 2025" when the year is not the reference year. */
export function shortDay(iso: string, referenceIso?: string): string {
  const date = parseDay(iso);
  const text = `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]?.slice(0, 3)}`;
  if (referenceIso && parseDay(referenceIso).getUTCFullYear() !== date.getUTCFullYear()) {
    return `${text} ${date.getUTCFullYear()}`;
  }
  return text;
}

/** "Fri 18 Sep 2026" */
export function stampDay(iso: string): string {
  const date = parseDay(iso);
  return `${WEEKDAYS[date.getUTCDay()]?.slice(0, 3)} ${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]?.slice(0, 3)} ${date.getUTCFullYear()}`;
}

export function monthName(index: number): string {
  return MONTHS[index] ?? "";
}

export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;
}

export function shares(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return Math.round(value).toLocaleString("en-US");
}

export function dollars(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (magnitude >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e3) return `$${Math.round(value / 1e3)}K`;
  return `$${Math.round(value)}`;
}

export function signedPercent(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(1)}%`;
}

export function bytes(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(value / 1e3))} KB`;
}

/** The most recent day quarterly fund holdings were due: 45 days after a quarter end. */
export function lastFundDeadline(todayIso: string): string {
  const today = parseDay(todayIso);
  const year = today.getUTCFullYear();
  const candidates = [year, year - 1].flatMap((y) => [
    Date.UTC(y, 1, 14),
    Date.UTC(y, 4, 15),
    Date.UTC(y, 7, 14),
    Date.UTC(y, 10, 14),
  ]);
  const past = candidates.filter((time) => time < today.getTime()).sort((a, b) => b - a);
  return isoDay(new Date(past[0] ?? today.getTime()));
}
