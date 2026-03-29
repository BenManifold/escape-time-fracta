import init, { alloc, dealloc, probe_escape_iter, fill_smooth_palette_lut } from "./fractal-wasm/pkg/fractal_wasm.js";
import { createFractalGpuRenderer, webGpuSupported } from "./fractal-webgpu.js";

const MIN_BOX_PX = 4;
/** Clamp drawable width/height (backing-store px) for GPU memory and perf. */
const CANVAS_PX_MIN = 256;
const CANVAS_PX_MAX = 4096;

/** Pan: fraction of the visible complex width (2·halfW) per second while a pan key is held. */
const KEY_PAN_FRAC_PER_SEC = 0.42;
/** Zoom: |d ln(halfW)/dt| while +/− held (exponential, smooth at any frame rate). */
const KEY_ZOOM_LOG_PER_SEC = 0.62;
const KEYBOARD_HALF_W_MIN = 1e-16;
const KEYBOARD_HALF_W_MAX = 96;

/** Frames per affine segment; GPU checkpoint at segment start, warp until segment end. */
const CHECKPOINT_FRAMES = 300;
const DEEP_ZOOM_DURATION_MS = 20 * 60 * 1000;
const DEEP_ZOOM_PAN_RETARGET_MS = 20 * 1000;
const DEEP_ZOOM_PAN_PER_SEGMENT = 0.08;
const DEEP_ZOOM_TARGET_SAMPLES = 48;
const DEEP_ZOOM_MIXED_REGION_PX = 300;
const DEEP_ZOOM_MIXED_PLACEMENTS = 36;
const DEEP_ZOOM_MIXED_GRID_STEP = 20;
const DEEP_ZOOM_HALF_W_MIN = 1e-13;

const MAX_ITER_MIN = 32;
const MAX_ITER_MAX = 8192;

const DEFAULT_VIEW = Object.freeze({
  centerX: -0.5,
  centerY: 0,
  halfW: 2,
});

const canvas = document.getElementById("canvas");
const uiCanvas = document.getElementById("canvasUi");
const uiCtx = uiCanvas.getContext("2d", { alpha: true, desynchronized: true });
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

/**
 * @type {{
 *   drawFull: (p: object, o?: object) => number;
 *   drawCheckpoint: (p: object) => number;
 *   drawWarp: (a: object, b: object) => number;
 *   captureCommit: () => void;
 *   copyWorkToCommit: () => void;
 *   resize: (w: number, h: number) => void;
 *   releaseCommit: () => void;
 *   destroy: () => void;
 * } | null}
 */
let gpuRenderer = null;

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

let cacheFingerprint = "";
/** @type {{ centerX: number; centerY: number; halfW: number }} */
let committedView = { centerX: 0, centerY: 0, halfW: 0 };
let cacheReady = false;

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
 *   checkpointSubmitted: boolean;
 *   lastPanRetargetAt: number;
 * }} DeepZoomState
 * @type {DeepZoomState | null}
 */
let deepZoom = null;

/** @type {{ x0: number; y0: number; x1: number; y1: number } | null} */
let rubber = null;
let rubberPointerId = null;

/** Held keys for smooth pan (arrows / WASD) and zoom (+/−). */
const keysPanZoom = {
  left: false,
  right: false,
  up: false,
  down: false,
  zoomIn: false,
  zoomOut: false,
};
/** @type {number} performance.now() ms; 0 = skip dt on first frame */
let lastKeyboardNavT = 0;

function isTypingFocusTarget(target) {
  if (!target || typeof target !== "object") return false;
  const el = /** @type {HTMLElement} */ (target);
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  return !!el.closest?.("input, textarea, select");
}

function setKeyPanZoomFromCode(code, down) {
  switch (code) {
    case "ArrowLeft":
    case "KeyA":
      keysPanZoom.left = down;
      return true;
    case "ArrowRight":
    case "KeyD":
      keysPanZoom.right = down;
      return true;
    case "ArrowUp":
    case "KeyW":
      keysPanZoom.up = down;
      return true;
    case "ArrowDown":
    case "KeyS":
      keysPanZoom.down = down;
      return true;
    case "Equal":
    case "NumpadAdd":
      keysPanZoom.zoomIn = down;
      return true;
    case "Minus":
    case "NumpadSubtract":
      keysPanZoom.zoomOut = down;
      return true;
    default:
      return false;
  }
}

function clearKeysPanZoom() {
  keysPanZoom.left =
    keysPanZoom.right =
    keysPanZoom.up =
    keysPanZoom.down =
    keysPanZoom.zoomIn =
    keysPanZoom.zoomOut =
      false;
}

/**
 * @param {number} dtSec
 * @returns {boolean} true if `view` changed
 */
function applyKeyboardPanZoom(dtSec) {
  if (!gpuRenderer || deepZoom || dtSec <= 0) return false;
  const k = keysPanZoom;
  if (!k.left && !k.right && !k.up && !k.down && !k.zoomIn && !k.zoomOut) return false;

  const cw = canvas.width;
  const ch = canvas.height;
  if (cw < 1 || ch < 1) return false;
  const aspect = ch / cw;
  const halfH = view.halfW * aspect;
  const spanRe = 2 * view.halfW;
  const spanIm = 2 * halfH;
  const step = KEY_PAN_FRAC_PER_SEC * dtSec;

  let changed = false;
  if (k.left) {
    view.centerX -= spanRe * step;
    changed = true;
  }
  if (k.right) {
    view.centerX += spanRe * step;
    changed = true;
  }
  if (k.up) {
    view.centerY -= spanIm * step;
    changed = true;
  }
  if (k.down) {
    view.centerY += spanIm * step;
    changed = true;
  }

  const z = KEY_ZOOM_LOG_PER_SEC * dtSec;
  if (k.zoomIn && !k.zoomOut) {
    view.halfW *= Math.exp(-z);
    changed = true;
  } else if (k.zoomOut && !k.zoomIn) {
    view.halfW *= Math.exp(z);
    changed = true;
  }
  if (k.zoomIn || k.zoomOut) {
    view.halfW = Math.min(KEYBOARD_HALF_W_MAX, Math.max(KEYBOARD_HALF_W_MIN, view.halfW));
  }
  return changed;
}

function clampMaxIter(n) {
  const v = Math.floor(Number(n));
  if (!Number.isFinite(v)) return MAX_ITER_MIN;
  return Math.min(MAX_ITER_MAX, Math.max(MAX_ITER_MIN, v));
}

function clampPaletteId(n) {
  const v = Math.floor(Number(n));
  if (!Number.isFinite(v)) return 0;
  return Math.min(4, Math.max(0, v));
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

function previewJuliaLabelsFromInputs() {
  const re = Number(juliaReInput.value) / 1000;
  const im = Number(juliaImInput.value) / 1000;
  juliaReLabel.textContent = Number.isFinite(re) ? re.toFixed(3) : "—";
  juliaImLabel.textContent = Number.isFinite(im) ? im.toFixed(3) : "—";
}

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
  juliaPanel.hidden = fractalKind !== 1;
}

function buildFingerprint() {
  const cw = canvas.width;
  const ch = canvas.height;
  return `${cw}x${ch}|${fractalKind}|${maxIterUser}|${julia.re}|${julia.im}|p${currentPaletteId()}`;
}

function viewsEqual(a, b) {
  return a.centerX === b.centerX && a.centerY === b.centerY && a.halfW === b.halfW;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function randomOnScreenPx(cw, ch) {
  const xm = Math.max(0, cw - 1);
  const ym = Math.max(0, ch - 1);
  return { sx: Math.random() * xm, sy: Math.random() * ym };
}

function panTargetScore(iter, maxIter) {
  if (iter < maxIter) return iter;
  return Math.floor(maxIter * 0.78);
}

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

function retargetDeepZoomPanGoal(dz, now) {
  const cw = canvas.width;
  const ch = canvas.height;
  const p = pickDeepZoomPanTarget(cw, ch);
  dz.panTargetX = p.re;
  dz.panTargetY = p.im;
  dz.lastPanRetargetAt = now;
}

function canvasCoords(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const sx = ((clientX - r.left) / r.width) * canvas.width;
  const sy = ((clientY - r.top) / r.height) * canvas.height;
  return { sx, sy };
}

function gpuParams(over = {}) {
  const cw = canvas.width;
  const ch = canvas.height;
  return {
    centerX: over.centerX ?? view.centerX,
    centerY: over.centerY ?? view.centerY,
    halfW: over.halfW ?? view.halfW,
    aspect: ch / cw,
    maxIter: maxIterUser >>> 0,
    paletteId: currentPaletteId() >>> 0,
    fractalKind: fractalKind >>> 0,
    juliaRe: julia.re,
    juliaIm: julia.im,
  };
}

function segmentEndParams(dz) {
  return gpuParams({
    centerX: dz.segC1x,
    centerY: dz.segC1y,
    halfW: dz.hwSegEnd,
  });
}

/**
 * @returns {number} ms
 */
function fullRenderAndCommit() {
  if (!gpuRenderer) return 0;
  const ms = gpuRenderer.drawFull(gpuParams());
  committedView = { centerX: view.centerX, centerY: view.centerY, halfW: view.halfW };
  cacheFingerprint = buildFingerprint();
  cacheReady = true;
  return ms;
}

function screenToComplex(sx, sy) {
  const w = canvas.width;
  const h = canvas.height;
  const aspect = h / w;
  const halfH = view.halfW * aspect;
  const re = view.centerX - view.halfW + (sx / w) * (2 * view.halfW);
  const im = view.centerY - halfH + (sy / h) * (2 * halfH);
  return { re, im };
}

function squareFromDrag(x0, y0, x1, y1) {
  const dim = Math.min(canvas.width, canvas.height);
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
  if (left + S > dim) left = dim - S;
  if (top + S > dim) top = dim - S;
  return { left, top, S };
}

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

function computeSegmentHalfWidths(dz) {
  const t0 = dz.segmentIdx / dz.nSeg;
  const t1 = (dz.segmentIdx + 1) / dz.nSeg;
  dz.hwSegStart = Math.exp(lerp(dz.logHalfW0, dz.logHalfW1, t0));
  dz.hwSegEnd = Math.exp(lerp(dz.logHalfW0, dz.logHalfW1, t1));
}

function updateDeepZoomButtonLabel() {
  deepZoomBtn.textContent = deepZoom ? "Stop" : "Deep zoom";
}

function cancelDeepZoom() {
  deepZoomGen += 1;
  deepZoom = null;
  nextPresetBtn.disabled = false;
  updateDeepZoomButtonLabel();
  gpuRenderer?.releaseCommit();
}

function stopDeepZoom(finished) {
  const was = deepZoom !== null;
  cancelDeepZoom();
  if (!gpuRenderer || !was) return;
  const ms = fullRenderAndCommit();
  const tag = finished ? "Zoom done" : "Zoom stopped";
  statusEl.textContent = `${tag} · ${ms.toFixed(1)} ms`;
}

function startDeepZoom() {
  if (!wasmMemory || !gpuRenderer || deepZoom) return;

  const fp = buildFingerprint();
  if (!cacheReady || cacheFingerprint !== fp || !viewsEqual(view, committedView)) {
    fullRenderAndCommit();
  }
  gpuRenderer.captureCommit();

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
    checkpointSubmitted: false,
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
  if (!dz || !gpuRenderer) return;

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

  if (dz.frameInSegment === 0 && !dz.checkpointSubmitted) {
    gpuRenderer.drawCheckpoint(segmentEndParams(dz));
    dz.checkpointSubmitted = true;
  }

  if (dz.frameInSegment === lastIx) {
    gpuRenderer.copyWorkToCommit();
    view.centerX = dz.segC1x;
    view.centerY = dz.segC1y;
    view.halfW = dz.hwSegEnd;
    committedView = { centerX: view.centerX, centerY: view.centerY, halfW: view.halfW };
    cacheFingerprint = buildFingerprint();
    cacheReady = true;

    dz.segmentIdx += 1;
    dz.frameInSegment = 0;
    if (dz.segmentIdx >= dz.nSeg) {
      stopDeepZoom(true);
      return;
    }
    computeSegmentHalfWidths(dz);
    dz.segC0x = view.centerX;
    dz.segC0y = view.centerY;
    dz.segC1x = lerp(dz.segC0x, dz.panTargetX, DEEP_ZOOM_PAN_PER_SEGMENT);
    dz.segC1y = lerp(dz.segC0y, dz.panTargetY, DEEP_ZOOM_PAN_PER_SEGMENT);
    dz.commitSnapshot = {
      centerX: dz.segC0x,
      centerY: dz.segC0y,
      halfW: dz.hwSegStart,
    };
    gpuRenderer.drawCheckpoint(segmentEndParams(dz));
    dz.checkpointSubmitted = true;
  }

  const f = dz.frameInSegment;
  const alpha = f / lastIx;
  const l0 = Math.log(dz.hwSegStart);
  const l1 = Math.log(dz.hwSegEnd);
  view.centerX = lerp(dz.segC0x, dz.segC1x, alpha);
  view.centerY = lerp(dz.segC0y, dz.segC1y, alpha);
  view.halfW = Math.exp(lerp(l0, l1, alpha));

  gpuRenderer.drawWarp(dz.commitSnapshot, view);

  if (dz.frameInSegment < lastIx) {
    dz.frameInSegment += 1;
  }

  statusEl.textContent = `Zoom · ${formatMmSs(elapsed)}`;
}

function resetToDefaultView() {
  if (!gpuRenderer) return;
  cancelDeepZoom();
  rubber = null;
  rubberPointerId = null;
  nextPresetBtn.disabled = false;
  view.centerX = DEFAULT_VIEW.centerX;
  view.centerY = DEFAULT_VIEW.centerY;
  view.halfW = DEFAULT_VIEW.halfW;
  invalidateCache();
  const ms = fullRenderAndCommit();
  statusEl.textContent = `Reset · ${ms.toFixed(1)} ms`;
}

function drawRubberOverlay() {
  if (!rubber) return;
  const { x0, y0, x1, y1 } = rubber;
  const { left, top, S } = squareFromDrag(x0, y0, x1, y1);
  if (S < MIN_BOX_PX) return;
  uiCtx.strokeStyle = "rgba(110, 181, 255, 0.95)";
  uiCtx.lineWidth = 2;
  uiCtx.setLineDash([6, 4]);
  uiCtx.strokeRect(left + 0.5, top + 0.5, S - 1, S - 1);
  uiCtx.setLineDash([]);
}

function frame() {
  syncStackPixelSize();
  const cw = canvas.width;
  const ch = canvas.height;
  let ms = 0;

  const now = performance.now();
  const dtSec = lastKeyboardNavT > 0 ? Math.min(0.048, (now - lastKeyboardNavT) / 1000) : 0;
  lastKeyboardNavT = now;

  if (deepZoom) {
    stepDeepZoom(now);
    uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
    drawRubberOverlay();
    requestAnimationFrame(frame);
    return;
  }

  applyKeyboardPanZoom(dtSec);

  const fp = buildFingerprint();
  const paramsMatch = cacheReady && cacheFingerprint === fp;
  if (!paramsMatch || !viewsEqual(view, committedView)) {
    ms = fullRenderAndCommit();
  }

  uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
  drawRubberOverlay();

  statusEl.textContent = ms > 0 ? `${cw}×${ch} · ${ms.toFixed(1)} ms` : `${cw}×${ch}`;

  requestAnimationFrame(frame);
}

uiCanvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || deepZoom) return;
  uiCanvas.setPointerCapture(e.pointerId);
  rubberPointerId = e.pointerId;
  const { sx, sy } = canvasCoords(e.clientX, e.clientY);
  rubber = { x0: sx, y0: sy, x1: sx, y1: sy };
});

uiCanvas.addEventListener("pointermove", (e) => {
  if (rubber === null || e.pointerId !== rubberPointerId) return;
  const { sx, sy } = canvasCoords(e.clientX, e.clientY);
  rubber.x1 = sx;
  rubber.y1 = sy;
});

uiCanvas.addEventListener("pointerup", (e) => {
  if (rubber === null || e.pointerId !== rubberPointerId) return;
  uiCanvas.releasePointerCapture(e.pointerId);
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

uiCanvas.addEventListener("pointercancel", (e) => {
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

window.addEventListener("keydown", (e) => {
  if (isTypingFocusTarget(e.target)) return;
  if (!setKeyPanZoomFromCode(e.code, true)) return;
  e.preventDefault();
});

window.addEventListener("keyup", (e) => {
  if (!setKeyPanZoomFromCode(e.code, false)) return;
  e.preventDefault();
});

window.addEventListener("blur", () => {
  clearKeysPanZoom();
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") clearKeysPanZoom();
});

/**
 * Match WebGPU + UI canvas backing store to the on-screen stack and DPR.
 * Re-run after GPU context creation: getContext("webgpu") can reset #canvas dimensions,
 * leaving it out of sync with #canvasUi and breaking box-zoom math.
 */
function syncStackPixelSize() {
  const stack = canvas.parentElement;
  if (!stack) return;
  const rect = stack.getBoundingClientRect();
  if (rect.width < 8 || rect.height < 8) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let nw = Math.round(rect.width * dpr);
  let nh = Math.round(rect.height * dpr);
  nw = Math.min(CANVAS_PX_MAX, Math.max(CANVAS_PX_MIN, nw));
  nh = Math.min(CANVAS_PX_MAX, Math.max(CANVAS_PX_MIN, nh));
  if (
    canvas.width !== nw ||
    canvas.height !== nh ||
    uiCanvas.width !== nw ||
    uiCanvas.height !== nh
  ) {
    canvas.width = nw;
    canvas.height = nh;
    uiCanvas.width = nw;
    uiCanvas.height = nh;
    gpuRenderer?.resize(nw, nh);
    invalidateCache();
  }
}

async function main() {
  if (!webGpuSupported()) {
    statusEl.textContent = "WebGPU is required (enable in browser / use Chromium).";
    return;
  }

  const wasm = await init();
  wasmMemory = wasm.memory;
  applyRenderParamsFromInputs();
  updateJuliaPanelVisibility();
  updateDeepZoomButtonLabel();

  try {
    gpuRenderer = await createFractalGpuRenderer(
      canvas,
      alloc,
      dealloc,
      fill_smooth_palette_lut,
      wasmMemory,
    );
  } catch (err) {
    console.error(err);
    statusEl.textContent = "WebGPU failed to initialize.";
    return;
  }

  syncStackPixelSize();
  const stack = canvas.parentElement;
  if (stack && typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => syncStackPixelSize()).observe(stack);
  }

  // One frame lets layout settle so syncStackPixelSize sees a real .canvas-stack rect (not 256×256).
  requestAnimationFrame(() => {
    syncStackPixelSize();
    requestAnimationFrame(frame);
  });
}

main().catch((err) => {
  statusEl.textContent = "WASM failed to load.";
  console.error(err);
});
