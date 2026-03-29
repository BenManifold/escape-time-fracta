import init, { alloc, dealloc, render_rgba, probe_escape_iter } from "./fractal-wasm/pkg/fractal_wasm.js";

const CANVAS = 1000;
const MIN_BOX_PX = 4;
const BG = "#0c0c0f";

/** Affine segment length in display frames (full WASM only at segment boundaries). */
const CHECKPOINT_FRAMES = 300;
/** Wall-clock duration of the deep zoom toward f64-ish floor. */
const DEEP_ZOOM_DURATION_MS = 20 * 60 * 1000;
/** Pick a new on-screen point to drift toward (complex coords) this often. */
const DEEP_ZOOM_PAN_RETARGET_MS = 20 * 1000;
/**
 * Each checkpoint segment, move the segment end center this fraction closer to `panTarget`
 * (slow pan layered on the zoom; target is always from a pixel currently on screen).
 */
const DEEP_ZOOM_PAN_PER_SEGMENT = 0.08;
/** Random screen probes per retarget; pick the one with escape count nearest the set (highest `n` before bail). */
const DEEP_ZOOM_TARGET_SAMPLES = 48;
/** Side length of screen square scanned for both interior and exterior (clamped to canvas). */
const DEEP_ZOOM_MIXED_REGION_PX = 300;
/** Random placements of that square before fallback scoring. */
const DEEP_ZOOM_MIXED_PLACEMENTS = 36;
/** Pixel step when probing the square (coarse grid, early exit when mixed). */
const DEEP_ZOOM_MIXED_GRID_STEP = 20;
/**
 * Practical lower bound for half-width at 1000² in f64 without perturbation.
 * Below this, coordinates lose too much precision relative to extent.
 */
const DEEP_ZOOM_HALF_W_MIN = 1e-13;

const MAX_ITER_MIN = 32;
const MAX_ITER_MAX = 8192;

/** Default complex-plane window (matches initial load). */
const DEFAULT_VIEW = Object.freeze({
  centerX: -0.5,
  centerY: 0,
  halfW: 2,
});

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
const resetViewBtn = document.getElementById("resetView");
const deepZoomBtn = document.getElementById("deepZoom");
const applyParamsBtn = document.getElementById("applyParams");
const paletteSelect = document.getElementById("palette");

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
  centerX: DEFAULT_VIEW.centerX,
  centerY: DEFAULT_VIEW.centerY,
  halfW: DEFAULT_VIEW.halfW,
};

const julia = {
  re: -0.7269,
  im: 0.1889,
};

let maxIterUser = clampMaxIter(Number(maxIterInput.value));
let fractalKind = Number(fractalSelect.value);

const cacheCanvas = document.createElement("canvas");
const cacheCtx = cacheCanvas.getContext("2d", { alpha: false });
let cacheFingerprint = "";
/** @type {{ centerX: number; centerY: number; halfW: number }} */
let committedView = { centerX: 0, centerY: 0, halfW: 0 };
let cacheReady = false;

/** @type {Worker | null} */
let deepZoomWorker = null;
let deepZoomGen = 0;

/**
 * @typedef {{
 *   gen: number;
 *   t0: number;
 *   panTargetX: number;
 *   panTargetY: number;
 *   segC0x: number;
 *   segC0y: number;
 *   segC1x: number;
 *   segC1y: number;
 *   logHalfW0: number;
 *   logHalfW1: number;
 *   nSeg: number;
 *   segmentIdx: number;
 *   frameInSegment: number;
 *   commitSnapshot: { centerX: number; centerY: number; halfW: number };
 *   hwSegStart: number;
 *   hwSegEnd: number;
 *   catchingUp: boolean;
 *   lastPanRetargetAt: number;
 * }} DeepZoomState
 * @type {DeepZoomState | null}
 */
let deepZoom = null;

/**
 * @typedef {{
 *   gen: number;
 *   segmentIdx: number;
 *   cw: number;
 *   ch: number;
 *   centerX: number;
 *   centerY: number;
 *   halfW: number;
 *   pixels: Uint8ClampedArray;
 * }} PendingCheckpoint
 * @type {PendingCheckpoint | null}
 */
let pendingCheckpoint = null;

/** @type {{ x0: number; y0: number; x1: number; y1: number } | null} */
let rubber = null;
let rubberPointerId = null;

function clampMaxIter(n) {
  const v = Math.floor(Number(n));
  if (!Number.isFinite(v)) return MAX_ITER_MIN;
  return Math.min(MAX_ITER_MAX, Math.max(MAX_ITER_MIN, v));
}

function clampPaletteId(n) {
  const v = Math.floor(Number(n));
  if (!Number.isFinite(v)) return 0;
  return Math.min(3, Math.max(0, v));
}

function currentPaletteId() {
  return clampPaletteId(Number(paletteSelect.value));
}

function readMaxIterFromInput() {
  return clampMaxIter(maxIterInput.value);
}

function invalidateCache() {
  cacheReady = false;
  cacheFingerprint = "";
}

function syncIterLabel() {
  iterLabel.textContent = String(maxIterUser);
}

/** Update Julia readouts from sliders only (does not change `julia` or trigger render). */
function previewJuliaLabelsFromInputs() {
  const re = Number(juliaReInput.value) / 1000;
  const im = Number(juliaImInput.value) / 1000;
  juliaReLabel.textContent = Number.isFinite(re) ? re.toFixed(3) : "—";
  juliaImLabel.textContent = Number.isFinite(im) ? im.toFixed(3) : "—";
}

/** Read inputs into applied `maxIterUser` / `julia` and invalidate cache (call after Apply or on first load). */
function applyRenderParamsFromInputs() {
  maxIterUser = readMaxIterFromInput();
  maxIterInput.value = String(maxIterUser);
  syncIterLabel();
  julia.re = Number(juliaReInput.value) / 1000;
  julia.im = Number(juliaImInput.value) / 1000;
  if (!Number.isFinite(julia.re)) julia.re = 0;
  if (!Number.isFinite(julia.im)) julia.im = 0;
  previewJuliaLabelsFromInputs();
}

function updateJuliaPanelVisibility() {
  const isJulia = fractalKind === 1;
  juliaPanel.hidden = !isJulia;
}

function buildFingerprint() {
  return `${CANVAS}x${CANVAS}|${fractalKind}|${maxIterUser}|${julia.re}|${julia.im}|p${currentPaletteId()}`;
}

function viewsEqual(a, b) {
  return a.centerX === b.centerX && a.centerY === b.centerY && a.halfW === b.halfW;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Uniform random pixel in the canvas (guaranteed on-screen). */
function randomOnScreenPx(cw, ch) {
  const xm = Math.max(0, cw - 1);
  const ym = Math.max(0, ch - 1);
  return { sx: Math.random() * xm, sy: Math.random() * ym };
}

/**
 * Prefer exterior points that escaped late (near the set); de-prioritize interior blobs vs high‑n exterior.
 * `iter` is from `probe_escape_iter`: in [0, maxIter) if escaped, else `maxIter` (inside).
 */
function panTargetScore(iter, maxIter) {
  if (iter < maxIter) return iter;
  return Math.floor(maxIter * 0.78);
}

/**
 * If this axis-aligned square (screen px) contains both bounded and escaping orbits, return complex
 * coords of its center; otherwise null.
 */
function mixedRegionCenterIfComplex(cw, ch, x0, y0, rw, rh) {
  const mi = maxIterUser >>> 0;
  const fk = fractalKind >>> 0;
  const jr = julia.re;
  const ji = julia.im;
  const x1 = x0 + rw;
  const y1 = y0 + rh;
  let inside = false;
  let outside = false;
  const step = Math.max(
    10,
    Math.min(DEEP_ZOOM_MIXED_GRID_STEP, Math.floor(Math.min(rw, rh) / 14)),
  );
  for (let sy = y0 + step * 0.5; sy < y1; sy += step) {
    for (let sx = x0 + step * 0.5; sx < x1; sx += step) {
      const { re, im } = screenToComplex(sx, sy);
      const iter = probe_escape_iter(re, im, mi, fk, jr, ji) >>> 0;
      if (iter >= mi) inside = true;
      else outside = true;
      if (inside && outside) {
        const cx = x0 + rw * 0.5;
        const cy = y0 + rh * 0.5;
        return screenToComplex(cx, cy);
      }
    }
  }
  return null;
}

/**
 * Prefer a 300px window that shows both set interior and exterior; center the pan goal there.
 * Fallback: single-point samples biased toward late escape (boundary).
 */
function pickDeepZoomPanTarget(cw, ch) {
  const mi = maxIterUser >>> 0;
  const rw = Math.min(DEEP_ZOOM_MIXED_REGION_PX, cw);
  const rh = Math.min(DEEP_ZOOM_MIXED_REGION_PX, ch);
  const spanX = Math.max(0, cw - rw);
  const spanY = Math.max(0, ch - rh);

  for (let a = 0; a < DEEP_ZOOM_MIXED_PLACEMENTS; a++) {
    const x0 = spanX > 0 ? Math.floor(Math.random() * (spanX + 1)) : 0;
    const y0 = spanY > 0 ? Math.floor(Math.random() * (spanY + 1)) : 0;
    const p = mixedRegionCenterIfComplex(cw, ch, x0, y0, rw, rh);
    if (p) return p;
  }

  let bestRe = view.centerX;
  let bestIm = view.centerY;
  let bestS = -1;
  for (let i = 0; i < DEEP_ZOOM_TARGET_SAMPLES; i++) {
    const { sx, sy } = randomOnScreenPx(cw, ch);
    const { re, im } = screenToComplex(sx, sy);
    const iter = probe_escape_iter(re, im, mi, fractalKind >>> 0, julia.re, julia.im) >>> 0;
    const s = panTargetScore(iter, mi);
    if (s > bestS) {
      bestS = s;
      bestRe = re;
      bestIm = im;
    }
  }
  return { re: bestRe, im: bestIm };
}

/**
 * New pan goal: center of a 300px screen square that contains both interior and exterior, when found.
 */
function retargetDeepZoomPanGoal(dz, now) {
  const cw = canvas.width;
  const ch = canvas.height;
  const p = pickDeepZoomPanTarget(cw, ch);
  dz.panTargetX = p.re;
  dz.panTargetY = p.im;
  dz.lastPanRetargetAt = now;
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
    julia.im,
    currentPaletteId() >>> 0,
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

function getDeepZoomWorker() {
  if (!deepZoomWorker) {
    deepZoomWorker = new Worker(new URL("./deep-zoom-worker.js", import.meta.url), { type: "module" });
    deepZoomWorker.onmessage = onDeepZoomWorkerMessage;
  }
  return deepZoomWorker;
}

function onDeepZoomWorkerMessage(ev) {
  const msg = ev.data;
  if (msg?.type === "error") {
    console.error("deep-zoom worker:", msg.message);
    return;
  }
  if (msg?.type !== "done") return;
  const dz = deepZoom;
  if (!dz || msg.gen !== dz.gen) return;

  const { buffer, cw, ch, centerX, centerY, halfW, segmentIdx } = msg;
  const pixels = new Uint8ClampedArray(buffer);
  pendingCheckpoint = {
    gen: msg.gen,
    segmentIdx,
    cw,
    ch,
    centerX,
    centerY,
    halfW,
    pixels,
  };
}

function postWorkerCheckpoint(dz) {
  const w = getDeepZoomWorker();
  w.postMessage({
    type: "render",
    gen: dz.gen,
    segmentIdx: dz.segmentIdx,
    cw: canvas.width,
    ch: canvas.height,
    centerX: dz.segC1x,
    centerY: dz.segC1y,
    halfW: dz.hwSegEnd,
    maxIter: maxIterUser,
    fractalKind,
    juliaRe: julia.re,
    juliaIm: julia.im,
    paletteId: currentPaletteId() >>> 0,
  });
}

function computeSegmentHalfWidths(dz) {
  const t0 = dz.segmentIdx / dz.nSeg;
  const t1 = (dz.segmentIdx + 1) / dz.nSeg;
  dz.hwSegStart = Math.exp(lerp(dz.logHalfW0, dz.logHalfW1, t0));
  dz.hwSegEnd = Math.exp(lerp(dz.logHalfW0, dz.logHalfW1, t1));
}

function applyPendingCheckpoint(pr) {
  const { cw, ch } = pr;
  if (cacheCanvas.width !== cw || cacheCanvas.height !== ch) {
    cacheCanvas.width = cw;
    cacheCanvas.height = ch;
  }
  const imageData = new ImageData(pr.pixels, cw, ch);
  cacheCtx.putImageData(imageData, 0, 0);
  committedView = { centerX: pr.centerX, centerY: pr.centerY, halfW: pr.halfW };
  cacheFingerprint = buildFingerprint();
  cacheReady = true;
  view.centerX = pr.centerX;
  view.centerY = pr.centerY;
  view.halfW = pr.halfW;
}

function updateDeepZoomButtonLabel() {
  deepZoomBtn.textContent = deepZoom ? "Stop" : "Deep zoom";
}

function cancelDeepZoom() {
  deepZoomGen += 1;
  deepZoom = null;
  pendingCheckpoint = null;
  nextPresetBtn.disabled = false;
  updateDeepZoomButtonLabel();
}

/**
 * @param {boolean} [finished]
 */
function stopDeepZoom(finished) {
  const was = deepZoom !== null;
  cancelDeepZoom();
  if (!wasmMemory || !was) return;
  const ms = fullRenderAndCommit();
  const tag = finished ? "Zoom done" : "Zoom stopped";
  statusEl.textContent = `${tag} · ${ms.toFixed(0)} ms`;
}

function startDeepZoom() {
  if (!wasmMemory || deepZoom) return;

  const fp = buildFingerprint();
  const dimsMatch = cacheCanvas.width === canvas.width && cacheCanvas.height === canvas.height;
  if (!cacheReady || cacheFingerprint !== fp || !dimsMatch || !viewsEqual(view, committedView)) {
    fullRenderAndCommit();
  }

  if (view.halfW <= DEEP_ZOOM_HALF_W_MIN * 1.0001) {
    statusEl.textContent = "At zoom limit";
    return;
  }

  const logHalfW0 = Math.log(view.halfW);
  const logHalfW1 = Math.log(DEEP_ZOOM_HALF_W_MIN);
  const msPerSegment = (CHECKPOINT_FRAMES / 60) * 1000;
  const nSeg = Math.max(1, Math.ceil(DEEP_ZOOM_DURATION_MS / msPerSegment));

  deepZoomGen += 1;
  const gen = deepZoomGen;

  const tStart = performance.now();
  const c0x = view.centerX;
  const c0y = view.centerY;
  deepZoom = {
    gen,
    t0: tStart,
    panTargetX: c0x,
    panTargetY: c0y,
    segC0x: c0x,
    segC0y: c0y,
    segC1x: c0x,
    segC1y: c0y,
    logHalfW0,
    logHalfW1,
    nSeg,
    segmentIdx: 0,
    frameInSegment: 0,
    commitSnapshot: { centerX: c0x, centerY: c0y, halfW: view.halfW },
    hwSegStart: view.halfW,
    hwSegEnd: view.halfW,
    catchingUp: false,
    lastPanRetargetAt: tStart,
  };
  computeSegmentHalfWidths(deepZoom);
  {
    const dz = deepZoom;
    const p = pickDeepZoomPanTarget(canvas.width, canvas.height);
    dz.panTargetX = p.re;
    dz.panTargetY = p.im;
    dz.segC1x = lerp(dz.segC0x, dz.panTargetX, DEEP_ZOOM_PAN_PER_SEGMENT);
    dz.segC1y = lerp(dz.segC0y, dz.panTargetY, DEEP_ZOOM_PAN_PER_SEGMENT);
    dz.commitSnapshot = {
      centerX: dz.segC0x,
      centerY: dz.segC0y,
      halfW: dz.hwSegStart,
    };
  }

  pendingCheckpoint = null;
  postWorkerCheckpoint(deepZoom);
  nextPresetBtn.disabled = true;
  updateDeepZoomButtonLabel();
}

function formatMmSs(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

function stepDeepZoom(now) {
  const dz = deepZoom;
  if (!dz) return;

  const cw = canvas.width;
  const ch = canvas.height;
  const elapsed = now - dz.t0;
  const lastIx = CHECKPOINT_FRAMES - 1;

  if (elapsed >= DEEP_ZOOM_DURATION_MS || dz.hwSegStart <= DEEP_ZOOM_HALF_W_MIN * 1.0001) {
    stopDeepZoom(true);
    return;
  }

  if (now - dz.lastPanRetargetAt >= DEEP_ZOOM_PAN_RETARGET_MS) {
    retargetDeepZoomPanGoal(dz, now);
  }

  if (dz.frameInSegment === lastIx) {
    const pr = pendingCheckpoint;
    if (pr && pr.gen === dz.gen && pr.segmentIdx === dz.segmentIdx) {
      applyPendingCheckpoint(pr);
      pendingCheckpoint = null;
      dz.segmentIdx += 1;
      dz.frameInSegment = 0;
      dz.catchingUp = false;
      if (dz.segmentIdx >= dz.nSeg) {
        stopDeepZoom(true);
        return;
      }
      computeSegmentHalfWidths(dz);
      dz.segC0x = pr.centerX;
      dz.segC0y = pr.centerY;
      dz.segC1x = lerp(dz.segC0x, dz.panTargetX, DEEP_ZOOM_PAN_PER_SEGMENT);
      dz.segC1y = lerp(dz.segC0y, dz.panTargetY, DEEP_ZOOM_PAN_PER_SEGMENT);
      dz.commitSnapshot = {
        centerX: dz.segC0x,
        centerY: dz.segC0y,
        halfW: dz.hwSegStart,
      };
      postWorkerCheckpoint(dz);
    } else {
      dz.catchingUp = true;
    }
  } else {
    dz.catchingUp = false;
  }

  const f = dz.frameInSegment;
  const alpha = f / lastIx;
  const l0 = Math.log(dz.hwSegStart);
  const l1 = Math.log(dz.hwSegEnd);
  view.centerX = lerp(dz.segC0x, dz.segC1x, alpha);
  view.centerY = lerp(dz.segC0y, dz.segC1y, alpha);
  view.halfW = Math.exp(lerp(l0, l1, alpha));

  drawWarpedCache(dz.commitSnapshot, cw, ch);

  if (dz.frameInSegment < lastIx) {
    dz.frameInSegment += 1;
  }

  const segMsg = dz.catchingUp ? " · …" : "";
  statusEl.textContent = `Zoom · ${formatMmSs(elapsed)}${segMsg}`;
}

function resetToDefaultView() {
  if (!wasmMemory) return;
  cancelDeepZoom();
  rubber = null;
  rubberPointerId = null;
  nextPresetBtn.disabled = false;
  view.centerX = DEFAULT_VIEW.centerX;
  view.centerY = DEFAULT_VIEW.centerY;
  view.halfW = DEFAULT_VIEW.halfW;
  invalidateCache();
  const ms = fullRenderAndCommit();
  statusEl.textContent = `Reset · ${ms.toFixed(0)} ms`;
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

  if (deepZoom) {
    stepDeepZoom(now);
    drawRubberOverlay();
    requestAnimationFrame(frame);
    return;
  }

  const fp = buildFingerprint();
  const dimsMatch = cacheCanvas.width === cw && cacheCanvas.height === ch;
  const paramsMatch = cacheReady && cacheFingerprint === fp && dimsMatch;

  if (!paramsMatch) {
    wasmMs = fullRenderAndCommit();
  } else if (!viewsEqual(view, committedView)) {
    wasmMs = fullRenderAndCommit();
  } else {
    ctx.drawImage(cacheCanvas, 0, 0);
  }

  drawRubberOverlay();

  statusEl.textContent = wasmMs > 0 ? `${cw}×${ch} · ${wasmMs.toFixed(0)} ms` : `${cw}×${ch}`;

  requestAnimationFrame(frame);
}

canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || deepZoom) return;
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

  if (deepZoom) return;

  const { left, top, S } = squareFromDrag(x0, y0, x1, y1);
  if (S < MIN_BOX_PX) return;

  const toView = viewFromSquare(left, top, S);
  view.centerX = toView.centerX;
  view.centerY = toView.centerY;
  view.halfW = toView.halfW;
  fullRenderAndCommit();
});

canvas.addEventListener("pointercancel", (e) => {
  if (rubberPointerId === e.pointerId) {
    rubber = null;
    rubberPointerId = null;
  }
});

resetViewBtn.addEventListener("click", () => {
  resetToDefaultView();
});

deepZoomBtn.addEventListener("click", () => {
  if (deepZoom) stopDeepZoom(false);
  else startDeepZoom();
});

nextPresetBtn.addEventListener("click", () => {
  if (deepZoom) return;
  const list = PRESETS[fractalKind];
  if (!list || list.length === 0) return;
  const i = presetRound[fractalKind] % list.length;
  presetRound[fractalKind] = i + 1;
  const toView = list[i];
  view.centerX = toView.centerX;
  view.centerY = toView.centerY;
  view.halfW = toView.halfW;
  fullRenderAndCommit();
});

fractalSelect.addEventListener("change", () => {
  cancelDeepZoom();
  fractalKind = Number(fractalSelect.value);
  updateJuliaPanelVisibility();
  invalidateCache();
});

paletteSelect.addEventListener("change", () => {
  invalidateCache();
});

juliaReInput.addEventListener("input", () => {
  previewJuliaLabelsFromInputs();
});

juliaImInput.addEventListener("input", () => {
  previewJuliaLabelsFromInputs();
});

applyParamsBtn.addEventListener("click", () => {
  cancelDeepZoom();
  applyRenderParamsFromInputs();
  invalidateCache();
});

maxIterInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") applyParamsBtn.click();
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
  applyRenderParamsFromInputs();
  updateJuliaPanelVisibility();
  updateDeepZoomButtonLabel();
  requestAnimationFrame(frame);
}

main().catch((err) => {
  statusEl.textContent = "WASM failed to load (see README).";
  console.error(err);
});
