import { describe, expect, it } from "vitest";

import { addDays, daysBetween, dollars, lastFundDeadline, longDay, plural, roundDollars, shares, shortDay, signedPercent, stampDay } from "./format";

describe("dates are Washington calendar days, printed the same everywhere", () => {
  it("does not shift a day by the reader's time zone", () => {
    expect(longDay("2026-09-17")).toBe("Thursday 17 September");
    expect(stampDay("2026-09-18")).toBe("Fri 18 Sep 2026");
  });

  it("adds the year only when it differs from the reference", () => {
    expect(shortDay("2026-03-31", "2026-09-18")).toBe("31 Mar");
    expect(shortDay("2025-12-31", "2026-09-18")).toBe("31 Dec 2025");
  });

  it("counts days across a daylight-saving change without losing one", () => {
    expect(daysBetween("2026-03-01", "2026-03-31")).toBe(30);
    expect(daysBetween("2026-07-02", "2026-09-16")).toBe(76);
    expect(addDays("2026-02-27", 3)).toBe("2026-03-02");
  });
});

describe("the last day quarterly fund holdings were due", () => {
  it("is 45 days after the most recent quarter end that has passed", () => {
    expect(lastFundDeadline("2026-09-18", 45)).toBe("2026-08-14");
    expect(lastFundDeadline("2026-08-14", 45)).toBe("2026-05-15");
    expect(lastFundDeadline("2026-01-20", 45)).toBe("2025-11-14");
  });

  it("moves with the deadline it is given, and is never a fixed list of dates", () => {
    expect(lastFundDeadline("2026-09-18", 60)).toBe("2026-08-29");
    expect(roundDollars(25_000_000)).toBe("$25 million");
    expect(roundDollars(1_500_000_000)).toBe("$1.5 billion");
    expect(roundDollars(750_000)).toBe("$750,000");
  });
});

describe("numbers", () => {
  it("abbreviates without inventing precision", () => {
    expect(dollars(646_480_000)).toBe("$646.5M");
    // A figure shows in the unit it fills, and a negative keeps its sign in front.
    expect(dollars(999_999_999)).toBe("$1.00B");
    expect(dollars(-1_234_567)).toBe("-$1.2M");
    expect(dollars(495_600)).toBe("$496K");
    expect(dollars(1_310_000_000)).toBe("$1.31B");
    expect(shares(227_917_808)).toBe("227.9M");
    expect(shares(12_000)).toBe("12,000");
  });

  it("uses a real minus sign, and none for zero", () => {
    expect(signedPercent(-0.6)).toBe("−0.6%");
    expect(signedPercent(13.6)).toBe("+13.6%");
    expect(signedPercent(0)).toBe("0.0%");
  });

  it("pluralises", () => {
    expect(plural(1, "day")).toBe("1 day");
    expect(plural(1_204, "disclosure")).toBe("1,204 disclosures");
  });
});
