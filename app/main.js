// Electron main process: owns the two worker processes and wires them together.
//
//   Python tracker --(hands + jpeg per frame)--> main --(hands only)--> Rust engine
//                                                 |                         |
//                                                 +--> renderer: frame      +--> renderer: scene

const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

const ROOT = path.join(__dirname, '..');
const isWin = process.platform === 'win32';
const PYTHON = path.join(ROOT, 'tracker', '.venv', isWin ? 'Scripts/python.exe' : 'bin/python');
const TRACKER = path.join(ROOT, 'tracker', 'hand_tracker.py');
const ENGINE = path.join(ROOT, 'engine', 'target', 'release', isWin ? 'tesseract_engine.exe' : 'tesseract_engine');

let win = null;
let tracker = null;
let engine = null;
let shuttingDown = false;

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function status(text, level = 'info') {
  console.log(`[${level}] ${text}`);
  send('status', { text, level });
}

function startEngine() {
  if (!fs.existsSync(ENGINE)) {
    status(`Rust engine not built: ${ENGINE} (run "npm run build:engine")`, 'error');
    return;
  }
  engine = spawn(ENGINE, [], { stdio: ['pipe', 'pipe', 'pipe'] });
  readline.createInterface({ input: engine.stdout }).on('line', (line) => {
    try {
      send('scene', JSON.parse(line));
    } catch {
      /* ignore partial lines */
    }
  });
  engine.stderr.on('data', (d) => console.error(String(d).trim()));
  engine.stdin.on('error', () => {});
  engine.on('exit', (code) => {
    engine = null;
    if (!shuttingDown) status(`Rust engine exited (${code})`, 'error');
  });
}

function startTracker() {
  if (!fs.existsSync(PYTHON)) {
    status(`Python venv missing: ${PYTHON}`, 'error');
    return;
  }
  status('Starting camera + hand tracker…');
  // Anything after `electron .` (e.g. `npm start -- --camera 1`) goes to the tracker.
  tracker = spawn(PYTHON, ['-u', TRACKER, ...process.argv.slice(2)], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  readline.createInterface({ input: tracker.stdout, crlfDelay: Infinity }).on('line', (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    const { jpg, ...data } = msg;
    send('frame', { jpg, hands: data.hands, w: data.w, h: data.h, t: data.t, stats: data.stats });
    if (engine && engine.stdin.writable) engine.stdin.write(JSON.stringify(data) + '\n');
  });

  readline.createInterface({ input: tracker.stderr }).on('line', (line) => {
    if (line.startsWith('[tracker]')) status(line.replace('[tracker] ', ''));
    else console.error(line);
  });

  tracker.on('exit', (code) => {
    tracker = null;
    if (code !== null && code !== 0) {
      status(
        code === 2
          ? 'Could not open the camera. Close other apps using it (Discord, Teams, browser) and restart.'
          : `Tracker exited (${code})`,
        'error'
      );
    }
  });
}

function stopChildren() {
  shuttingDown = true;
  for (const p of [tracker, engine]) {
    if (p && !p.killed) p.kill();
  }
  tracker = engine = null;
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: '#000000',
    title: 'Tesseract Hands',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, 'index.html'));
  win.webContents.on('did-finish-load', () => {
    startEngine();
    startTracker();
  });
  win.webContents.on('before-input-event', (e, input) => {
    if (input.type !== 'keyDown') return;
    if (input.key === 'F11') win.setFullScreen(!win.isFullScreen());
    if (input.key === 'Escape' && win.isFullScreen()) win.setFullScreen(false);
  });
  win.on('closed', () => {
    win = null;
    stopChildren();
  });
}

app.whenReady().then(createWindow);
app.on('before-quit', stopChildren);
app.on('window-all-closed', () => app.quit());
