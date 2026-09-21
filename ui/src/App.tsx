import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import { Shell } from "./components/Shell";
import { useApi } from "./lib/hooks";
import { DASHBOARD } from "./lib/links";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { DataPage } from "./pages/DataPage";
import { FeedPage } from "./pages/FeedPage";
import { LandingPage } from "./pages/LandingPage";
import { PortfoliosPage } from "./pages/PortfoliosPage";
import { TickerPage } from "./pages/TickerPage";

interface Health {
  status: string;
  today: string;
}

export function App() {
  // The server's "today", in Washington. The browser's clock is not consulted.
  const { data } = useApi<Health>("/api/health", {});
  const today = data?.today ?? null;
  const { pathname, search } = useLocation();

  // The landing page stands outside the app: no search, no as-of, no way in yet.
  if (pathname === "/welcome") return <LandingPage />;

  return (
    <Shell today={today}>
      <Routes>
        {/* The bare address goes to the dashboard and keeps its query, so an old
            link with ?asof=, ?actor= or ?kind= still lands where it pointed. */}
        <Route path="/" element={<Navigate to={`${DASHBOARD}${search}`} replace />} />
        <Route path={DASHBOARD} element={<FeedPage />} />
        <Route path="/portfolios" element={<PortfoliosPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/t/:ticker" element={<TickerPage />} />
        <Route path="/data" element={today ? <DataPage today={today} /> : <p className="loading">Reading the lake…</p>} />
        <Route
          path="*"
          element={
            <div className="empty">
              <h2>There is no page here.</h2>
              <p>
                <a href={DASHBOARD}>Go to the dashboard</a>
              </p>
            </div>
          }
        />
      </Routes>
    </Shell>
  );
}
