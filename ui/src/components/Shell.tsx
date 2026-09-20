import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { stampDay } from "../lib/format";
import { useAsOf } from "../lib/hooks";
import { AsOfControl } from "./AsOfControl";
import { Search } from "./Search";

export function Shell({ today, children }: { today: string | null; children: ReactNode }) {
  const { asOf, setAsOf, search } = useAsOf();

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to the content
      </a>
      <header className="topbar">
        <NavLink className="brand" to={`/${search}`}>
          <span className="brand__mark" aria-hidden="true">
            <span className="brand__dot" />
            <span className="brand__line" />
            <span className="brand__diamond" />
          </span>
          <span className="brand__name">asof</span>
        </NavLink>
        <nav className="tabs" aria-label="Main">
          <NavLink className="tabs__tab" to={`/${search}`} end>
            Feed
          </NavLink>
          <NavLink className="tabs__tab" to={`/data${search}`}>
            Data health
          </NavLink>
        </nav>
        <div className="topbar__spacer" />
        <Search />
        {today && <AsOfControl today={today} />}
      </header>

      {asOf && (
        <div className="pastbar" role="status">
          <p>
            <strong>You are looking at the world as it was on {stampDay(asOf)}.</strong> Nothing disclosed after that day is shown.
          </p>
          <button type="button" className="button button--ink" onClick={() => setAsOf(null)}>
            Return to today
          </button>
        </div>
      )}

      <div id="main" className="app__main">
        {children}
      </div>

      <footer className="footnote">
        <span>There is no sign-in yet. This app only answers on this machine.</span>
        <a href="/api/docs">API reference</a>
      </footer>
    </div>
  );
}
