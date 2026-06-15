import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend the dev server proxies /api to. Defaults to :8001 (matches how the
// backend is launched in the runbook); override with VITE_API_PROXY_TARGET,
// e.g. VITE_API_PROXY_TARGET=http://localhost:8000 npm run dev
const apiTarget = process.env.VITE_API_PROXY_TARGET || "http://localhost:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true }
    }
  }
});
