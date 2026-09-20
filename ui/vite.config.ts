import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the interface runs on Vite's port and the API on its own, so
// /api is proxied. In production one process serves both: `quantlab serve`.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8080" },
  },
  build: { outDir: "dist", sourcemap: false, assetsInlineLimit: 0 },
});
