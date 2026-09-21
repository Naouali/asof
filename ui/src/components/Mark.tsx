import type { Direction, Kind } from "../lib/api";

/**
 * Who, by shape. Which way, by fill.
 *
 * A circle is an insider, a square a member of Congress, a diamond a fund, and a
 * triangle a federal contract, drawn in ink because public money is neither bought
 * nor sold. Bought is solid and sold is hollow, so the direction survives being printed in grey or
 * read by someone who cannot tell the blue from the orange.
 */
export function Mark({ kind, direction, muted = false }: { kind: Kind; direction: Direction; muted?: boolean }) {
  const classes = ["mark", `mark--${kind}`, `mark--${direction}`, muted ? "mark--muted" : ""].filter(Boolean).join(" ");
  return <span className={classes} aria-hidden="true" />;
}

export function Legend({ chart = false, contracts = false }: { chart?: boolean; contracts?: boolean }) {
  return (
    <div className="legend">
      {chart && (
        <span className="legend__item">
          <span className="legend__trade" aria-hidden="true" />
          Day of the trade
        </span>
      )}
      <span className="legend__item">
        <Mark kind="insider" direction="buy" muted />
        Company insider
      </span>
      <span className="legend__item">
        <Mark kind="congress" direction="buy" muted />
        Member of Congress
      </span>
      <span className="legend__item">
        <Mark kind="fund" direction="buy" muted />
        Fund
      </span>
      {contracts && (
        <span className="legend__item">
          <Mark kind="contract" direction="buy" />
          Federal contract
        </span>
      )}
      <span className="legend__item">
        <span className="legend__swatch legend__swatch--buy" aria-hidden="true" />
        Bought
      </span>
      <span className="legend__item">
        <span className="legend__swatch legend__swatch--sell" aria-hidden="true" />
        Sold
      </span>
    </div>
  );
}
