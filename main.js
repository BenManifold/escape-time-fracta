import init, { alloc, dealloc, render_rgba } from "./fractal-wasm/pkg/fractal_wasm.js";

const CANVAS = 1000;
const MIN_BOX_PX = 4;
const ANIM_MS = 450;
const BG = "#0c0c0f";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
const fractalSelect = document.getElementById("fractal");
const maxIterInput = document.getElementById("maxIter");
const iterLabel = document.getElementById("iterLabel");
const statusEl = document.getElementById("status");
const juliaPanel = document.getElementById("juliaPanel");
const juliaReInput = document.getElementById("juliaRe");
const juliaImInput = document.getElementById("juliaIm");
const juliaReLabel = document.getElementById("juliaReLabel");
const juliaImLabel = document.getElementById("juliaImLabel");
const nextPresetBtn = document.getElementById("nextPreset");

/** @type {Array<Array<{ centerX: number; centerY: number; halfW: number }>>} */
const PRESETS = [
  [
    { centerX: -0.5, centerY: 0, halfW: 2 },
    { centerX: -0.75, centerY: 0, halfW: 0.06 },
    { centerX: -0.75, centerY: 0.1, halfW: 0.025 },
    { centerX: -1.25, centerY: 0, halfW: 0.06 },
    { centerX: -0.16, centerY: 1.0405, halfW: 0.012 },
    { centerX: -1.768, centerY: 0, halfW: 0.02 },
  ],
  [
    { centerX: 0, centerY: 0, halfW: 2 },
    { centerX: 0, centerY: 0, halfW: 0.6 },
    { centerX: 0.35, centerY: 0.35, halfW: 0.2 },
    { centerX: -0.4, centerY: 0.6, halfW: 0.15 },
  ],
  [
    { centerX: -1.75, centerY: -0.02, halfW: 0.08 },
    { centerX: -1.94, centerY: -0.02, halfW: 0.02 },
    { centerX: -1.5, centerY: -0.02, halfW: 0.5 },
    { centerX: -0.2, centerY: -0.35, halfW: 0.15 },
  ],
  [
    { centerX: 0, centerY: 0, halfW: 2 },
    { centerX: -0.75, centerY: 0, halfW: 0.08 },
    { centerX: 0.2, centerY: 0.55, halfW: 0.06 },
  ],
];

const presetRound = { 0: 0, 1: 0, 2: 0, 3: 0 };

let wasmMemory;
/** @type {number | null} */
let pixelPtr = null;
let pixelPtrLen = 0;

const view = {
  centerX: -0.5,
  centerY: 0,
  halfW: 2,
};

const julia = {
  re: -0.7269,
  im: 0.1889,
};

let maxIterUser = Number(maxIterInput.value);
let fractalKind = Number(fractalSelect.value);

const cacheCanvas = document.createElement("canvas");
const cacheCtx = cacheCanvas.getContext("2d", { alpha: false });
let cacheFingerprint = "";
/** @type {{ centerX: number; centerY: number; halfW: number }} */
let committedView = { centerX: 0, centerY: 0, halfW: 0 };
let cacheReady = false;

/**
 * @typedef {{ from: typeof view; to: typeof view; snapshotCommitted: typeof committedView; t0: number; duration: number; label: string }} Anim
 * @type {Anim | null}
 */
let anim = null;

/** @type {{ x0: number; y0: number; x1: number; y1: number } | null} */
let rubber = null;
let rubberPointerId = null;

function invalidateCache() {
  cacheReady = false;
  cacheFingerprint = "";
}

function syncIterLabel() {
  iterLabel.textContent = String(maxIterUser);
}

function syncJuliaLabels() {
  julia.re = Number(juliaReInput.value) / 1000;
  julia.im = Number(juliaImInput.value) / 1000;
  juliaReLabel.textContent = julia.re.toFixed(3);
  juliaImLabel.textContent = julia.im.toFixed(3);
}

function updateJuliaPanelVisibility() {
  const isJulia = fractalKind === 1;
  juliaPanel.hidden = !isJulia;
}

function buildFingerprint() {
  return `${CANVAS}x${CANVAS}|${fractalKind}|${maxIterUser}|${julia.re}|${julia.im}`;
}

function viewsEqual(a, b) {
  return a.centerX === b.centerX && a.centerY === b.centerY && a.halfW === b.halfW;
}

function smoothstep(t) {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * @param {number} clientX
 * @param {number} clientY
 */
function canvasCoords(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const sx = ((clientX - r.left) / r.width) * canvas.width;
  const sy = ((clientY - r.top) / r.height) * canvas.height;
  return { sx, sy };
}

function drawWarpedCache(committed, cw, ch) {
  const cx0 = committed.centerX;
  const cy0 = committed.centerY;
  const hw0 = committed.halfW;
  const cx1 = view.centerX;
  const cy1 = view.centerY;
  const hw1 = view.halfW;
  const aspect = ch / cw;
  const hh0 = hw0 * aspect;
  const hh1 = hw1 * aspect;

  const Ku = (cx1 - hw1 - cx0 + hw0) * (cw / (2 * hw0));
  const Kv = (cy1 - hh1 - cy0 + hh0) * (ch / (2 * hh0));
  const s = hw0 / hw1;

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cw, ch);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.setTransform(s, 0, 0, s, -Ku * s, -Kv * s);
  ctx.drawImage(cacheCanvas, 0, 0, cw, ch);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function freeBuffer() {
  if (pixelPtr != null && pixelPtrLen > 0) {
    dealloc(pixelPtr, pixelPtrLen);
    pixelPtr = null;
    pixelPtrLen = 0;
  }
}

function ensureBuffer(byteLen) {
  if (pixelPtr != null && pixelPtrLen === byteLen) return;
  freeBuffer();
  pixelPtr = alloc(byteLen);
  pixelPtrLen = byteLen;
}

/**
 * @param {number} sx
 * @param {number} sy
 */
function screenToComplex(sx, sy) {
  const w = canvas.width;
  const h = canvas.height;
  const aspect = h / w;
  const halfH = view.halfW * aspect;
  const re = view.centerX - view.halfW + (sx / w) * (2 * view.halfW);
  const im = view.centerY - halfH + (sy / h) * (2 * halfH);
  return { re, im };
}

/**
 * Square from drag diagonal, centered on segment midpoint, clamped to canvas.
 */
function squareFromDrag(x0, y0, x1, y1) {
  const minX = Math.min(x0, x1);
  const minY = Math.min(y0, y1);
  const maxX = Math.max(x0, x1);
  const maxY = Math.max(y0, y1);
  let S = Math.max(maxX - minX, maxY - minY);
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  let left = cx - S / 2;
  let top = cy - S / 2;
  if (left < 0) left = 0;
  if (top < 0) top = 0;
  if (left + S > CANVAS) left = CANVAS - S;
  if (top + S > CANVAS) top = CANVAS - S;
  return { left, top, S };
}

/**
 * Map rubber square (pixel coords) to target view using current `view`.
 */
function viewFromSquare(left, top, S) {
  const tl = screenToComplex(left, top);
  const br = screenToComplex(left + S, top + S);
  const reMin = Math.min(tl.re, br.re);
  const reMax = Math.max(tl.re, br.re);
  const imMin = Math.min(tl.im, br.im);
  const imMax = Math.max(tl.im, br.im);
  const centerX = (reMin + reMax) / 2;
  const centerY = (imMin + imMax) / 2;
  const halfW = Math.max(1e-16, (reMax - reMin) / 2);
  return { centerX, centerY, halfW };
}

/**
 * @returns {number} ms
 */
function fullRenderAndCommit() {
  const cw = canvas.width;
  const ch = canvas.height;
  const aspect = ch / cw;
  const byteLen = cw * ch * 4;
  ensureBuffer(byteLen);

  const t0 = performance.now();
  render_rgba(
    pixelPtr,
    byteLen,
    cw,
    ch,
    view.centerX,
    view.centerY,
    view.halfW,
    aspect,
    maxIterUser,
    fractalKind,
    julia.re,
    julia.im
  );
  const t1 = performance.now();

  if (cacheCanvas.width !== cw || cacheCanvas.height !== ch) {
    cacheCanvas.width = cw;
    cacheCanvas.height = ch;
  }

  const src = new Uint8ClampedArray(wasmMemory.buffer, pixelPtr, byteLen);
  const imageData = new ImageData(cw, ch);
  imageData.data.set(src);
  cacheCtx.putImageData(imageData, 0, 0);

  committedView = { centerX: view.centerX, centerY: view.centerY, halfW: view.halfW };
  cacheFingerprint = buildFingerprint();
  cacheReady = true;

  ctx.drawImage(cacheCanvas, 0, 0);

  return t1 - t0;
}

/**
 * @param {typeof view} toView
 * @param {string} label
 */
function startAnimation(toView, label) {
  if (anim !== null) return;

  const fp = buildFingerprint();
  const dimsMatch = cacheCanvas.width === canvas.width && cacheCanvas.height === canvas.height;
  if (!cacheReady || cacheFingerprint !== fp || !dimsMatch || !viewsEqual(view, committedView)) {
    fullRenderAndCommit();
  }

  anim = {
    from: { centerX: view.centerX, centerY: view.centerY, halfW: view.halfW },
    to: { centerX: toView.centerX, centerY: toView.centerY, halfW: toView.halfW },
    snapshotCommitted: { centerX: committedView.centerX, centerY: committedView.centerY, halfW: committedView.halfW },
    t0: performance.now(),
    duration: ANIM_MS,
    label,
  };
  nextPresetBtn.disabled = true;
}

function drawRubberOverlay() {
  if (!rubber) return;
  const { x0, y0, x1, y1 } = rubber;
  const { left, top, S } = squareFromDrag(x0, y0, x1, y1);
  if (S < MIN_BOX_PX) return;
  ctx.strokeStyle = "rgba(110, 181, 255, 0.95)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(left + 0.5, top + 0.5, S - 1, S - 1);
  ctx.setLineDash([]);
}

function frame(now) {
  const cw = canvas.width;
  const ch = canvas.height;

  let wasmMs = 0;
  let mode = "";

  if (anim) {
    const rawT = (now - anim.t0) / anim.duration;
    const t = rawT >= 1 ? 1 : rawT;
    const u = smoothstep(t);

    view.centerX = lerp(anim.from.centerX, anim.to.centerX, u);
    view.centerY = lerp(anim.from.centerY, anim.to.centerY, u);
    view.halfW = lerp(anim.from.halfW, anim.to.halfW, u);

    drawWarpedCache(anim.snapshotCommitted, cw, ch);
    drawRubberOverlay();

    if (t >= 1) {
      view.centerX = anim.to.centerX;
      view.centerY = anim.to.centerY;
      view.halfW = anim.to.halfW;
      const label = anim.label;
      wasmMs = fullRenderAndCommit();
      anim = null;
      nextPresetBtn.disabled = false;
      statusEl.textContent = `${cw}×${ch} · ${label} done · ${maxIterUser} it · ${wasmMs.toFixed(1)} ms`;
    } else {
      statusEl.textContent = `${cw}×${ch} · animating · ${anim.label} · ${(u * 100).toFixed(0)}%`;
    }
    requestAnimationFrame(frame);
    return;
  }

  const fp = buildFingerprint();
  const dimsMatch = cacheCanvas.width === cw && cacheCanvas.height === ch;
  const paramsMatch = cacheReady && cacheFingerprint === fp && dimsMatch;

  if (!paramsMatch) {
    wasmMs = fullRenderAndCommit();
    mode = `render ${maxIterUser} it (cache rebuild)`;
  } else if (!viewsEqual(view, committedView)) {
    wasmMs = fullRenderAndCommit();
    mode = `render ${maxIterUser} it (sync)`;
  } else {
    ctx.drawImage(cacheCanvas, 0, 0);
    mode = `idle · ${maxIterUser} it`;
  }

  drawRubberOverlay();

  statusEl.textContent =
    wasmMs > 0 ? `${cw}×${ch} · ${mode} · ${wasmMs.toFixed(1)} ms` : `${cw}×${ch} · ${mode}`;

  requestAnimationFrame(frame);
}

canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || anim !== null) return;
  canvas.setPointerCapture(e.pointerId);
  rubberPointerId = e.pointerId;
  const { sx, sy } = canvasCoords(e.clientX, e.clientY);
  rubber = { x0: sx, y0: sy, x1: sx, y1: sy };
});

canvas.addEventListener("pointermove", (e) => {
  if (rubber === null || e.pointerId !== rubberPointerId) return;
  const { sx, sy } = canvasCoords(e.clientX, e.clientY);
  rubber.x1 = sx;
  rubber.y1 = sy;
});

canvas.addEventListener("pointerup", (e) => {
  if (rubber === null || e.pointerId !== rubberPointerId) return;
  canvas.releasePointerCapture(e.pointerId);
  rubberPointerId = null;

  const { x0, y0, x1, y1 } = rubber;
  rubber = null;

  if (anim !== null) return;

  const { left, top, S } = squareFromDrag(x0, y0, x1, y1);
  if (S < MIN_BOX_PX) return;

  const toView = viewFromSquare(left, top, S);
  startAnimation(toView, "box zoom");
});

canvas.addEventListener("pointercancel", (e) => {
  if (rubberPointerId === e.pointerId) {
    rubber = null;
    rubberPointerId = null;
  }
});

nextPresetBtn.addEventListener("click", () => {
  if (anim !== null) return;
  const list = PRESETS[fractalKind];
  if (!list || list.length === 0) return;
  const i = presetRound[fractalKind] % list.length;
  presetRound[fractalKind] = i + 1;
  const toView = list[i];
  startAnimation(
    { centerX: toView.centerX, centerY: toView.centerY, halfW: toView.halfW },
    `preset ${i + 1}/${list.length}`
  );
});

fractalSelect.addEventListener("change", () => {
  fractalKind = Number(fractalSelect.value);
  updateJuliaPanelVisibility();
  invalidateCache();
});

juliaReInput.addEventListener("input", () => {
  syncJuliaLabels();
  invalidateCache();
});

juliaImInput.addEventListener("input", () => {
  syncJuliaLabels();
  invalidateCache();
});

maxIterInput.addEventListener("input", () => {
  maxIterUser = Number(maxIterInput.value);
  syncIterLabel();
  invalidateCache();
});

function ensureCanvasSize() {
  if (canvas.width !== CANVAS || canvas.height !== CANVAS) {
    canvas.width = CANVAS;
    canvas.height = CANVAS;
    freeBuffer();
    invalidateCache();
  }
}

async function main() {
  ensureCanvasSize();
  const wasm = await init();
  wasmMemory = wasm.memory;
  syncIterLabel();
  syncJuliaLabels();
  updateJuliaPanelVisibility();
  requestAnimationFrame(frame);
}

main().catch((err) => {
  statusEl.textContent = "WASM failed to load. Run wasm-pack in fractal-wasm/ (see README).";
  console.error(err);
});
