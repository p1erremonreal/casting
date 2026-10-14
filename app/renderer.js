// Renderer: draws the mirrored camera feed, hand points, and the tesseract
// lines computed by the Rust engine. All incoming coordinates are normalized
// (0..1) in camera-frame space; we map them through the same "cover" fit the
// camera image uses so lines land exactly on your fingers.

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const fx = document.createElement('canvas').getContext('2d'); // tesseract layer
const hud = document.getElementById('hud');
const statusEl = document.getElementById('status');

const state = {
  bitmap: null,
  frameW: 640,
  frameH: 480,
  hands: [],
  scene: null,
  showCamera: true,
  showLandmarks: true,
  decoding: false,
  pendingJpg: null,
  frameTimes: [],
  sceneTimes: [],
};

// ---- incoming data ---------------------------------------------------------

window.bridge.onFrame((f) => {
  state.hands = f.hands || [];
  state.stats = f.stats || null;
  if (f.w) state.frameW = f.w;
  if (f.h) state.frameH = f.h;
  state.frameTimes.push(performance.now());
  if (f.jpg) {
    state.pendingJpg = f.jpg;
    decodeLatest();
  }
});

window.bridge.onScene((s) => {
  state.scene = s;
  state.sceneTimes.push(performance.now());
});

let statusTimer = null;
window.bridge.onStatus(({ text, level }) => {
  statusEl.textContent = text;
  statusEl.className = level === 'error' ? 'error' : '';
  clearTimeout(statusTimer);
  if (level !== 'error') statusTimer = setTimeout(() => statusEl.classList.add('hidden'), 3500);
});

// Decode only the newest frame; drop any that arrive while decoding.
async function decodeLatest() {
  if (state.decoding || !state.pendingJpg) return;
  state.decoding = true;
  const b64 = state.pendingJpg;
  state.pendingJpg = null;
  try {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
    if (state.bitmap) state.bitmap.close();
    state.bitmap = bmp;
  } catch {
    /* skip bad frame */
  }
  state.decoding = false;
  if (state.pendingJpg) decodeLatest();
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'c') state.showCamera = !state.showCamera;
  if (e.key === 'l') state.showLandmarks = !state.showLandmarks;
});

// ---- layout ----------------------------------------------------------------

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
}
window.addEventListener('resize', resize);
resize();

/** Contain-fit of the camera frame into the canvas: the whole image is visible. */
function frameRect() {
  const cw = canvas.width;
  const ch = canvas.height;
  const scale = Math.min(cw / state.frameW, ch / state.frameH);
  const w = state.frameW * scale;
  const h = state.frameH * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

// ---- drawing ---------------------------------------------------------------

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** depth 0 (inner / far in 4D) = violet, 1 (outer / near) = cyan. */
function edgeColor(depth, alpha) {
  const r = Math.round(lerp(190, 80, depth));
  const g = Math.round(lerp(90, 235, depth));
  const b = 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function draw() {
  const { width: cw, height: ch } = canvas;
  const dpr = window.devicePixelRatio || 1;
  const fr = frameRect();
  const P = (x, y) => [fr.x + x * fr.w, fr.y + y * fr.h];

  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cw, ch);

  if (state.showCamera && state.bitmap) {
    ctx.drawImage(state.bitmap, fr.x, fr.y, fr.w, fr.h);
  }

  const scene = state.scene;
  const alpha = scene ? scene.alpha : 0;

  if (scene && alpha > 0.01) {
    // The tesseract is drawn on its own layer so hands can be cut out of it.
    if (fx.canvas.width !== cw || fx.canvas.height !== ch) {
      fx.canvas.width = cw;
      fx.canvas.height = ch;
    }
    fx.globalCompositeOperation = 'source-over';
    fx.clearRect(0, 0, cw, ch);
    fx.globalCompositeOperation = 'lighter';
    fx.lineCap = 'round';

    // Edges: wide soft pass for glow, then a thin bright core.
    for (const pass of [
      { width: 9, a: 0.12 },
      { width: 4, a: 0.25 },
      { width: 1.6, a: 0.95 },
    ]) {
      for (const [x1, y1, x2, y2, depth] of scene.edges) {
        const [ax, ay] = P(x1, y1);
        const [bx, by] = P(x2, y2);
        fx.strokeStyle = edgeColor(depth, pass.a * alpha * lerp(0.55, 1, depth));
        fx.lineWidth = pass.width * dpr * lerp(0.6, 1.2, depth);
        fx.beginPath();
        fx.moveTo(ax, ay);
        fx.lineTo(bx, by);
        fx.stroke();
      }
    }

    // Vertices.
    for (const [x, y, depth] of scene.vertices) {
      const [px, py] = P(x, y);
      fx.fillStyle = edgeColor(depth, 0.9 * alpha);
      fx.beginPath();
      fx.arc(px, py, lerp(1.5, 3.5, depth) * dpr, 0, Math.PI * 2);
      fx.fill();
    }

    // When the back of a hand faces the camera the tesseract is behind it
    // (on the palm side): cut the hand's silhouette out of the layer.
    fx.globalCompositeOperation = 'destination-out';
    for (const hand of scene.hands) {
      if (hand.occlude < 0.01 || hand.outline.length < 3) continue;
      const pts = hand.outline.map(([x, y]) => P(x, y));
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      fx.filter = `blur(${Math.round(extent * 0.03)}px)`; // soft edge
      fx.fillStyle = fx.strokeStyle = `rgba(0,0,0,${hand.occlude})`;
      fx.lineJoin = 'round';
      fx.lineWidth = extent * 0.18; // the hull runs through fingertip centers; pad to the skin edge
      fx.beginPath();
      pts.forEach(([x, y], i) => (i ? fx.lineTo(x, y) : fx.moveTo(x, y)));
      fx.closePath();
      fx.fill();
      fx.stroke();
      fx.filter = 'none';
    }

    ctx.globalCompositeOperation = 'lighter';
    ctx.drawImage(fx.canvas, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
  }

  // Fingertips (index highlighted), at the engine's 60 Hz predicted positions.
  if (state.showLandmarks) {
    for (const hand of scene ? scene.hands : state.hands) {
      hand.tips.forEach(([x, y], i) => {
        const [px, py] = P(x, y);
        ctx.fillStyle = i === 1 ? 'rgba(255,240,120,0.95)' : 'rgba(255,255,255,0.6)';
        ctx.beginPath();
        ctx.arc(px, py, (i === 1 ? 6 : 3.5) * dpr, 0, Math.PI * 2);
        ctx.fill();
        if (i === 1) {
          ctx.strokeStyle = 'rgba(255,240,120,0.5)';
          ctx.lineWidth = 1.5 * dpr;
          ctx.beginPath();
          ctx.arc(px, py, 11 * dpr, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }
  }

  // HUD.
  const now = performance.now();
  state.frameTimes = state.frameTimes.filter((t) => now - t < 1000);
  state.sceneTimes = state.sceneTimes.filter((t) => now - t < 1000);
  const idx = (scene ? scene.hands : state.hands).map((h) => `(${h.tips[1][0].toFixed(3)}, ${h.tips[1][1].toFixed(3)})`);
  const st = state.stats;
  hud.textContent =
    (st
      ? `capture fps ${st.cap_fps}   detect fps ${st.proc_fps}\n` +
        `lag         ${st.lag_ms} ms   dropped ${st.dropped}\n`
      : '') +
    `video fps   ${state.frameTimes.length}   tesseract fps ${state.sceneTimes.length}\n` +
    `hands       ${state.hands.length}\n` +
    `index tips  ${idx.join('  ') || '—'}\n` +
    `facing      ${scene ? scene.hands.map((h) => `${h.facing} (open ${Math.round(h.openness * 100)}%)`).join('  ') || '—' : '—'}\n` +
    `mode        ${scene ? scene.mode : '—'}\n` +
    `[c] camera  [l] landmarks  [F11] fullscreen`;

  requestAnimationFrame(draw);
}

requestAnimationFrame(draw);
