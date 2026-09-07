import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const BACKEND_URL = process.env.VITE_API_URL ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
        // Needed for SSE streaming to work properly
        ws: false,
      },
      '/health': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
})
