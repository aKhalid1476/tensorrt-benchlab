import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/bench': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
      '/telemetry': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
    },
  },
})
