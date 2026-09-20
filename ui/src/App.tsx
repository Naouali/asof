import { Route, Routes, useLocation } from "react-router-dom";

import { Shell } from "./components/Shell";
import { useApi } from "./lib/hooks";
import { DataPage } from "./pages/DataPage";
import { FeedPage } from "./pages/FeedPage";
import { LandingPage } from "./pages/LandingPage";
import { TickerPage } from "./pages/TickerPage";

interface Health {
  status: string;
  today: string;
}

export function App() {
  // The server's "today", in Washington. The browser's clock is not consulted.
  const { data } = useApi<Health>("/api/health", {});
  const today = data?.today ?? null;
  const { pathname } = useLocation();

  // The landing page stands outside the app: no search, no as-of, no way in yet.
  if (pathname === "/welcome") return <LandingPage />;

  return (
    <Shell today={today}>
      <Routes>
        <Route path="/" element={<FeedPage />} />
        <Route path="/t/:ticker" element={<TickerPage />} />
        <Route path="/data" element={today ? <DataPage today={today} /> : <p className="loading">Reading the lake…</p>} />
        <Route
          path="*"
          element={
            <div className="empty">
              <h2>There is no page here.</h2>
              <p>
                <a href="/">Go to the feed</a>
              </p>
            </div>
          }
        />
      </Routes>
    </Shell>
  );
}
