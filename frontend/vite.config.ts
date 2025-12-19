import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'
import { parse as parseYaml } from 'yaml'

// Read backend port from config.yaml
const configPath = path.resolve(__dirname, '../config.yaml')
const configYaml = fs.readFileSync(configPath, 'utf-8')
const config = parseYaml(configYaml)
const backendPort = config.server?.port || 10102

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
      '/ws': {
        target: `ws://localhost:${backendPort}`,
        ws: true,
      },
      '/outputs': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
      '/uploads': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
})
