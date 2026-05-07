import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: '../static/dist',
    assetsDir: 'assets',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: 'main.js',
        chunkFileNames: '[name].js',
      },
    },
  },
  server: {
    proxy: {
      '/runs': 'http://localhost:5000',
      '/predict': 'http://localhost:5000',
      '/static': 'http://localhost:5000',
    }
  }
});
