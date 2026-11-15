import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const windows = process.platform === 'win32';
const python = path.join(root, '.venv', windows ? 'Scripts/python.exe' : 'bin/python');
const vite = path.join(root, 'frontend/node_modules/vite/bin/vite.js');
const lock = path.join(root, '.cache/dev.lock');
const children = [];
let stopping = false;
let ownsLock = false;

function stop(code) {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.exitCode !== null || child.signalCode !== null) continue;
    if (windows) {
      spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
    } else {
      try { process.kill(-child.pid, 'SIGTERM'); } catch (error) {
        if (error.code !== 'ESRCH') console.error(error.message);
      }
    }
  }
  if (ownsLock) {
    try { unlinkSync(lock); } catch (error) {
      if (error.code !== 'ENOENT') console.error(error.message);
    }
  }
  process.exit(code);
}

function acquireLock() {
  mkdirSync(path.dirname(lock), { recursive: true });
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      writeFileSync(lock, String(process.pid), { flag: 'wx' });
      ownsLock = true;
      return;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      const pid = Number(readFileSync(lock, 'utf8'));
      if (!Number.isInteger(pid) || pid < 1) throw new Error(`Invalid startup lock: ${lock}`);
      try { process.kill(pid, 0); } catch (check) {
        if (check.code === 'ESRCH') { unlinkSync(lock); continue; }
      }
      throw new Error(`Relevia startup is already running (PID ${pid}).`);
    }
  }
  throw new Error('Could not acquire the Relevia startup lock.');
}

function checkPort(port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', error => reject(new Error(`Port ${port} is unavailable: ${error.message}`)));
    server.listen({ host: '127.0.0.1', port, exclusive: true }, () => server.close(resolve));
  });
}

function start(command, args, cwd, name) {
  const child = spawn(command, args, { cwd, stdio: 'inherit', detached: !windows });
  children.push(child);
  child.once('error', error => { console.error(`${name} failed to start: ${error.message}`); stop(1); });
  child.once('exit', (code, signal) => {
    if (!stopping) {
      console.error(`${name} stopped (${signal || `exit ${code}`}); stopping both servers.`);
      stop(code || 1);
    }
  });
}

process.on('SIGINT', () => { console.log('\nStopping Relevia...'); stop(0); });
process.on('SIGTERM', () => stop(0));

try {
  if (!existsSync(python)) throw new Error('Missing .venv Python. Create the virtual environment and install requirements.txt.');
  if (!existsSync(vite)) throw new Error('Missing frontend dependencies. Run npm install in frontend/.');
  acquireLock();
  await Promise.all([checkPort(8000), checkPort(5173)]);
  console.log('Starting Relevia: http://127.0.0.1:5173 (API: http://127.0.0.1:8000). Press Ctrl+C to stop.');
  start(python, ['-m', 'uvicorn', 'backend.api:app', '--host', '127.0.0.1', '--port', '8000'], root, 'FastAPI');
  start(process.execPath, [vite, '--host', '127.0.0.1', '--port', '5173', '--strictPort'], path.join(root, 'frontend'), 'Vite');
} catch (error) {
  console.error(error.message);
  stop(1);
}
