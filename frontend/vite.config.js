import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { connect } from 'node:net';

// Uvicorn starts listening only after its application lifespan has completed.
function backendReadiness() {
  return {
    name: 'backend-readiness',
    configureServer(server) {
      server.middlewares.use('/__backend_ready', (_request, response) => {
        const socket = connect({ host: '127.0.0.1', port: 8000 });
        let finished = false;
        const reply = ready => {
          if (finished) return;
          finished = true;
          socket.destroy();
          response.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
          response.end(JSON.stringify({ ready }));
        };
        socket.setTimeout(1000);
        socket.once('connect', () => reply(true));
        socket.once('error', () => reply(false));
        socket.once('timeout', () => reply(false));
        response.once('close', () => socket.destroy());
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), backendReadiness()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/images': 'http://127.0.0.1:8000',
    },
  },
});
