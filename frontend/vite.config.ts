import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port pinned to 5173 because backend CORS allows only localhost:5173 / 127.0.0.1:5173.
// See backend/app/main.py allow_origins.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
