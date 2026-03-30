import init, {
  alloc,
  dealloc,
  fill_smooth_palette_lut,
  render_rgba,
} from "./fractal-wasm/pkg/fractal_wasm.js";
import { createFractalGpuRenderer, webGpuSupported } from "./fractal-webgpu.js";

const MIN_BOX_PX = 4;
/** Clamp drawable width/height (backing-store px) for GPU memory and perf. */
const CANVAS_PX_MIN = 256;
const CANVAS_PX_MAX = 4096;

/** Pan: fraction of the visible complex width (2·halfW) per second while a pan key is held. */
const KEY_PAN_FRAC_PER_SEC = 0.42;
/** Zoom: |d ln(halfW)/dt| while +/− held (exponential, smooth at any frame rate). */
const KEY_ZOOM_LOG_PER_SEC = 0.62;
/** Keyboard zoom clamp (not the real precision limit; GPU f32 breaks down earlier). */
const KEYBOARD_HALF_W_MIN = 1e-16;
const KEYBOARD_HALF_W_MAX = 96;

/**
 * Below this Re half-width, use WASM render_rgba (f64 pixel map). Mandelbrot uses tiled perturb there when
 * half_w &lt; 0.02 (see fractal-wasm perturb). ~5e-5 → zoom ×4e4 from default half-w 2 (before ~1e5 f32 blocks).
 */
const WASM_F64_HALF_W = 5e-5;

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

/** Last `presentCpuRgba` time after an off-thread WASM render (main thread stays responsive during `render_rgba`). */
let lastWasmPresentMs = 0;

/** @type {Worker | null} */
let wasmRenderWorker = null;
/** If worker creation or runtime fails, keep using synchronous WASM render on the main thread. */
let wasmRenderSyncFallback = false;
let wasmRenderSeq = 0;
let wasmRenderInFlight = false;
/** @type {ReturnType<typeof buildWasmRenderJob> | null} */
let wasmRenderPending = null;

/**
 * @type {{
 *   drawFull: (p: object, o?: object) => number;
 *   resize: (w: number, h: number) => void;
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
let fractalKind = (() => {
  let fk = Number(fractalSelect.value);
  if (fk === 3 || !Number.isFinite(fk)) fk = 0;
  fractalSelect.value = String(fk);
  return fk;
})();

let cacheFingerprint = "";
/** @type {{ centerX: number; centerY: number; halfW: number }} */
let committedView = { centerX: 0, centerY: 0, halfW: 0 };
let cacheReady = false;

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
  if (!gpuRenderer || dtSec <= 0) return false;
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

/** Push HUD inputs into `maxIterUser` / `julia` and schedule a redraw. */
function syncParamsFromInputs() {
  applyRenderParamsFromInputs();
  invalidateCache();
}

function updateJuliaPanelVisibility() {
  juliaPanel.hidden = fractalKind !== 1;
}

function buildFingerprint() {
  const cw = canvas.width;
  const ch = canvas.height;
  const jKey = fractalKind === 1 ? `|${julia.re}|${julia.im}` : "";
  return `${cw}x${ch}|${fractalKind}|${maxIterUser}${jKey}|p${currentPaletteId()}`;
}

function viewsEqual(a, b) {
  return a.centerX === b.centerX && a.centerY === b.centerY && a.halfW === b.halfW;
}

function canvasCoords(clientX, clientY) {
  const el = uiCanvas;
  const r = el.getBoundingClientRect();
  const sx = ((clientX - r.left) / r.width) * el.width;
  const sy = ((clientY - r.top) / r.height) * el.height;
  return { sx, sy };
}

function gpuParams(over = {}) {
  const cw = canvas.width;
  const ch = canvas.height;
  const maxIter = over.maxIter !== undefined ? over.maxIter >>> 0 : maxIterUser >>> 0;
  return {
    centerX: over.centerX ?? view.centerX,
    centerY: over.centerY ?? view.centerY,
    halfW: over.halfW ?? view.halfW,
    aspect: ch / cw,
    maxIter,
    paletteId: currentPaletteId() >>> 0,
    fractalKind: fractalKind >>> 0,
    juliaRe: julia.re,
    juliaIm: julia.im,
  };
}

function buildWasmRenderJob() {
  const p = gpuParams();
  const nw = canvas.width | 0;
  const nh = canvas.height | 0;
  return {
    fp: buildFingerprint(),
    committed: { centerX: view.centerX, centerY: view.centerY, halfW: view.halfW },
    nw,
    nh,
    centerX: p.centerX,
    centerY: p.centerY,
    halfW: p.halfW,
    aspect: nh / nw,
    maxIter: p.maxIter >>> 0,
    fractalKind: fractalKind >>> 0,
    juliaRe: julia.re,
    juliaIm: julia.im,
    paletteId: p.paletteId >>> 0,
    perturbMode: fractalKind === 0 ? 2 : 0,
  };
}

function ensureWasmRenderWorker() {
  if (wasmRenderWorker || wasmRenderSyncFallback) return;
  try {
    wasmRenderWorker = new Worker(new URL("./fractal-render-worker.js", import.meta.url), { type: "module" });
    wasmRenderWorker.onmessage = onWasmWorkerMessage;
    wasmRenderWorker.onerror = (e) => {
      console.error(e);
      wasmRenderSyncFallback = true;
      wasmRenderInFlight = false;
      if (wasmRenderWorker) {
        try {
          wasmRenderWorker.terminate();
        } catch (_) {
          /* ignore */
        }
        wasmRenderWorker = null;
      }
    };
  } catch (e) {
    console.error(e);
    wasmRenderSyncFallback = true;
  }
}

/**
 * @param {ReturnType<typeof buildWasmRenderJob>} job
 */
function scheduleWasmRender(job) {
  wasmRenderPending = job;
  pumpWasmRender();
}

function pumpWasmRender() {
  if (wasmRenderInFlight || wasmRenderPending === null || !wasmRenderWorker) return;
  const job = wasmRenderPending;
  wasmRenderPending = null;
  wasmRenderInFlight = true;
  wasmRenderSeq++;
  wasmRenderWorker.postMessage({ type: "wasmRender", seq: wasmRenderSeq, ...job });
}

/**
 * @param {MessageEvent} e
 */
function onWasmWorkerMessage(e) {
  const d = e.data;
  if (d?.type !== "wasmRenderDone" || !gpuRenderer) return;
  wasmRenderInFlight = false;
  if (!d.ok) {
    console.error(d.error);
    wasmRenderSyncFallback = true;
    if (wasmRenderWorker) {
      try {
        wasmRenderWorker.terminate();
      } catch (_) {
        /* ignore */
      }
      wasmRenderWorker = null;
    }
    pumpWasmRender();
    return;
  }
  if (d.seq !== wasmRenderSeq) {
    pumpWasmRender();
    return;
  }
  if (buildFingerprint() !== d.fp || !viewsEqual(view, d.committed)) {
    pumpWasmRender();
    return;
  }
  const u8 = new Uint8Array(d.buffer);
  const ms = gpuRenderer.presentCpuRgba(u8, d.nw, d.nh, { bytesPerRow: d.bytesPerRow });
  lastWasmPresentMs = ms;
  committedView = { centerX: d.committed.centerX, centerY: d.committed.centerY, halfW: d.committed.halfW };
  cacheFingerprint = d.fp;
  cacheReady = true;
  pumpWasmRender();
}

function fullRenderAndCommitSyncWasm() {
  const p = gpuParams();
  const nw = canvas.width | 0;
  const nh = canvas.height | 0;
  const need = nw * nh * 4;
  const ptr = alloc(need);
  try {
    const perturbMode = fractalKind === 0 ? 2 : 0;
    render_rgba(
      ptr,
      need,
      nw >>> 0,
      nh >>> 0,
      p.centerX,
      p.centerY,
      p.halfW,
      nh / nw,
      p.maxIter >>> 0,
      fractalKind >>> 0,
      julia.re,
      julia.im,
      p.paletteId >>> 0,
      perturbMode,
    );
    const u8 = new Uint8Array(wasmMemory.buffer, ptr, need);
    const ms = gpuRenderer.presentCpuRgba(u8, nw, nh);
    lastWasmPresentMs = ms;
    committedView = { centerX: view.centerX, centerY: view.centerY, halfW: view.halfW };
    cacheFingerprint = buildFingerprint();
    cacheReady = true;
    return ms;
  } finally {
    dealloc(ptr, need);
  }
}

/**
 * @returns {number} ms (0 when WASM work was queued to a worker; see `lastWasmPresentMs` for last texture upload time)
 */
function fullRenderAndCommit() {
  if (!gpuRenderer) return 0;
  const p = gpuParams();
  const nw = canvas.width | 0;
  const nh = canvas.height | 0;

  if (p.halfW < WASM_F64_HALF_W && nw > 0 && nh > 0) {
    ensureWasmRenderWorker();
    if (!wasmRenderSyncFallback && wasmRenderWorker) {
      scheduleWasmRender(buildWasmRenderJob());
      return 0;
    }
    return fullRenderAndCommitSyncWasm();
  }

  const ms = gpuRenderer.drawFull(p);
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
  const w = canvas.width;
  const h = canvas.height;
  const dim = Math.min(w, h);
  const minX = Math.min(x0, x1);
  const minY = Math.min(y0, y1);
  const maxX = Math.max(x0, x1);
  const maxY = Math.max(y0, y1);
  let S = Math.max(maxX - minX, maxY - minY);
  S = Math.min(S, dim);
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  let left = cx - S / 2;
  let top = cy - S / 2;
  if (left < 0) left = 0;
  if (top < 0) top = 0;
  if (left + S > w) left = w - S;
  if (top + S > h) top = h - S;
  if (left < 0) left = 0;
  if (top < 0) top = 0;
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

function resetToDefaultView() {
  if (!gpuRenderer) return;
  rubber = null;
  rubberPointerId = null;
  view.centerX = DEFAULT_VIEW.centerX;
  view.centerY = DEFAULT_VIEW.centerY;
  view.halfW = DEFAULT_VIEW.halfW;
  invalidateCache();
  const ms = fullRenderAndCommit();
  const msShow = ms > 0 ? ms : lastWasmPresentMs;
  statusEl.textContent = `Reset · ${msShow.toFixed(1)} ms`;
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

/**
 * @param {number} x
 */
function formatSci(x) {
  if (!Number.isFinite(x)) return "—";
  const ax = Math.abs(x);
  if (ax >= 1e-2 && ax < 1e5) return x.toPrecision(4);
  return x.toExponential(2);
}

/**
 * Cheap zoom readout from `view.halfW` (half-width on Re axis; default view uses 2).
 * @returns {string[]}
 */
function zoomOverlayLines() {
  const hw = view.halfW;
  const zoom = DEFAULT_VIEW.halfW / hw;
  const lg = Math.log10(zoom);
  return [
    `Zoom ×${formatSci(zoom)}  ·  half-w ${formatSci(hw)}`,
    `log10(zoom) ${lg.toFixed(2)}  (vs default half-w ${DEFAULT_VIEW.halfW})`,
  ];
}

/**
 * @param {number} re
 * @param {number} im
 */
function formatLambdaJulia(re, im) {
  const imPart = `${im >= 0 ? "+" : "−"}${formatSci(Math.abs(im))}i`;
  return `${formatSci(re)}${imPart}`;
}

/**
 * @returns {string[]}
 */
function fractalEquationOverlayLines() {
  switch (fractalKind) {
    case 1:
      return [
        "Julia",
        `z' = z^2 + lambda,  z0 = pixel,  lambda = ${formatLambdaJulia(julia.re, julia.im)}`,
      ];
    case 2:
      return [
        "Burning Ship",
        "z' = (|Re z| + i|Im z|)^2 + c,  z0 = 0,  c = pixel",
      ];
    default:
      return ["Mandelbrot", "z' = z^2 + c,  z0 = 0,  c = pixel"];
  }
}

function drawViewOverlay() {
  const w = uiCanvas.width;
  const h = uiCanvas.height;
  if (w < 8 || h < 8) return;

  const pad = 10;
  const lineH = 16;
  const zoomLines = zoomOverlayLines();
  const eqLines = fractalEquationOverlayLines();
  const lines = [...zoomLines, "", ...eqLines];

  uiCtx.save();
  uiCtx.font = '13px ui-monospace, "Cascadia Code", "Segoe UI Mono", Consolas, monospace';
  uiCtx.textBaseline = "top";

  let y = pad;
  for (const line of lines) {
    if (line === "") {
      y += lineH * 0.35;
      continue;
    }
    uiCtx.strokeStyle = "rgba(0, 0, 0, 0.72)";
    uiCtx.lineWidth = 4;
    uiCtx.lineJoin = "round";
    uiCtx.miterLimit = 2;
    uiCtx.strokeText(line, pad, y);
    uiCtx.fillStyle = "rgba(232, 232, 239, 0.94)";
    uiCtx.fillText(line, pad, y);
    y += lineH;
    if (y > h - pad) break;
  }
  uiCtx.restore();
}

function frame() {
  syncStackPixelSize();
  const cw = canvas.width;
  const ch = canvas.height;
  let ms = 0;

  const now = performance.now();
  const dtSec = lastKeyboardNavT > 0 ? Math.min(0.048, (now - lastKeyboardNavT) / 1000) : 0;
  lastKeyboardNavT = now;

  applyKeyboardPanZoom(dtSec);

  const fp = buildFingerprint();
  const paramsMatch = cacheReady && cacheFingerprint === fp;
  if (!paramsMatch || !viewsEqual(view, committedView)) {
    ms = fullRenderAndCommit();
  }

  uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
  drawViewOverlay();
  drawRubberOverlay();

  const tag = gpuRenderer?.usedGpuPerturb ? " · GPU perturb" : "";
  const inWasmBand = view.halfW < WASM_F64_HALF_W;
  const showMs = ms > 0 ? ms : inWasmBand ? lastWasmPresentMs : 0;
  statusEl.textContent =
    showMs > 0 ? `${cw}×${ch}${tag} · ${showMs.toFixed(1)} ms` : `${cw}×${ch}${tag}`;

  requestAnimationFrame(frame);
}

uiCanvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
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

nextPresetBtn.addEventListener("click", () => {
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
  let fk = Number(fractalSelect.value);
  if (fk === 3 || !Number.isFinite(fk)) fk = 0;
  fractalSelect.value = String(fk);
  fractalKind = fk;
  updateJuliaPanelVisibility();
  syncParamsFromInputs();
});

paletteSelect.addEventListener("change", () => {
  invalidateCache();
});

juliaReInput.addEventListener("input", () => {
  syncParamsFromInputs();
});

juliaImInput.addEventListener("input", () => {
  syncParamsFromInputs();
});

maxIterInput.addEventListener("input", () => {
  syncParamsFromInputs();
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
  // Floor avoids round-trip bias that can oscillate with subpixel layout; pairs with jitter guard below.
  let nw = Math.floor(rect.width * dpr);
  let nh = Math.floor(rect.height * dpr);
  nw = Math.min(CANVAS_PX_MAX, Math.max(CANVAS_PX_MIN, nw));
  nh = Math.min(CANVAS_PX_MAX, Math.max(CANVAS_PX_MIN, nh));

  const cw = canvas.width | 0;
  const ch = canvas.height | 0;
  // Layout can wobble 1–2 device px; skip resize to avoid redundant GPU reallocations.
  if (cw > 0 && ch > 0 && Math.abs(nw - cw) <= 2 && Math.abs(nh - ch) <= 2) {
    return;
  }

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

let stackResizeRaf = 0;
function scheduleSyncStackPixelSize() {
  if (stackResizeRaf) return;
  stackResizeRaf = requestAnimationFrame(() => {
    stackResizeRaf = 0;
    syncStackPixelSize();
  });
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
    new ResizeObserver(() => scheduleSyncStackPixelSize()).observe(stack);
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
