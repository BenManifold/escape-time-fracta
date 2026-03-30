import init, { alloc, dealloc, fill_smooth_palette_lut } from "./fractal-wasm/pkg/fractal_wasm.js";
import { createFractalGpuRenderer, webGpuSupported } from "./fractal-webgpu.js";

const MIN_BOX_PX = 4;
/** Clamp drawable width/height (backing-store px) for GPU memory and perf. */
const CANVAS_PX_MIN = 256;
const CANVAS_PX_MAX = 4096;

/** Pan: fraction of the visible complex width (2·halfW) per second while a pan key is held. */
const KEY_PAN_FRAC_PER_SEC = 0.42;
/** Zoom: |d ln(halfW)/dt| while +/− held (exponential, smooth at any frame rate). */
const KEY_ZOOM_LOG_PER_SEC = 0.62;
/** Keyboard zoom clamp. */
const KEYBOARD_HALF_W_MIN = 1e-16;
const KEYBOARD_HALF_W_MAX = 96;

const MAX_ITER_MIN = 32;
const MAX_ITER_MAX = 8192;

/** Julia λ clamp (matches HUD ±2). */
const JULIA_C_LIM = 2;
/** Julia “λ drag” mode: one backing-store pixel of motion changes λ by this much on that axis. */
const JULIA_LAMBDA_PER_CANVAS_PX = 0.0001;
/** With Shift held, scale the above by this factor (0.001× → finer strokes). */
const JULIA_LAMBDA_SHIFT_MULT = 0.001;

/**
 * Julia λ tour: phase advance per animation frame (rad). Re/Im sweep [-JULIA_C_LIM, JULIA_C_LIM]
 * via sin (zero slope at extrema ⇒ smooth rubber). Peak |Δλ|/frame ≈ JULIA_C_LIM * DPHASE (e.g. ~1e-7 when DPHASE=5e-8).
 */
const JULIA_TOUR_DPHASE_RAD = 5e-8;
/** Im uses phase * this factor so Re/Im don’t stay locked in phase. */
const JULIA_TOUR_IM_PHASE_MUL = 0.618033988749895;
/** When |Re λ| and |Im λ| are both within this of 0, phase advances this many times faster. */
const JULIA_TOUR_ORIGIN_HALF = 0.36
const JULIA_TOUR_NEAR_ORIGIN_MULT = 150
/** Upper bound for global φ scan on resync (Lissajous is quasi-periodic; wide span avoids wrong local minima). */
const JULIA_TOUR_RESYNC_PHI_MAX = 3000 * Math.PI

/** Mandelbrot multibrot z^p + c: Δp per backing-store pixel (horizontal drag). */
const MANDEL_EXP_PER_CANVAS_PX = 0.006;
const MANDEL_EXP_SHIFT_MULT = 0.1;
const MANDEL_EXP_MIN = 1.25;
const MANDEL_EXP_MAX = 16;

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
const juliaModeBoxBtn = document.getElementById("juliaModeBox");
const juliaModeLambdaBtn = document.getElementById("juliaModeLambda");
const juliaTourPlayBtn = document.getElementById("juliaTourPlay");
const juliaTourPauseBtn = document.getElementById("juliaTourPause");
const juliaTourDirReBtn = document.getElementById("juliaTourDirRe");
const juliaTourDirImBtn = document.getElementById("juliaTourDirIm");
const juliaTourSpeedBtns = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll(".juliaTourSpeedBtn")
);
const mandelPanel = document.getElementById("mandelPanel");
const mandelModeBoxBtn = document.getElementById("mandelModeBox");
const mandelModeExpBtn = document.getElementById("mandelModeExp");
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

/** Initial camera when picking Reset view (per fractal family). */
function defaultViewForFractalKind(fk) {
  if (fk === 1) return PRESETS[1][0];
  if (fk === 2) return PRESETS[2][0];
  return DEFAULT_VIEW;
}

let wasmMemory;

/**
 * @type {{
 *   drawFull: (p: object, o?: object) => number;
 *   resize: (w: number, h: number) => void;
 *   destroy: () => void;
 * } | null}
 */
let gpuRenderer = null;

const view = {
  centerX: PRESETS[1][0].centerX,
  centerY: PRESETS[1][0].centerY,
  halfW: PRESETS[1][0].halfW,
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

/**
 * @type {{ pointerId: number; lastSx: number; lastSy: number; resumeTourAfter: boolean } | null}
 */
let juliaLambdaDrag = null;

/** True while writing Julia slider DOM from code (avoids `input` handlers stopping a paused λ tour). */
let syncingJuliaInputsFromState = false;

/** @type {"box" | "lambda"} */
let juliaCanvasMode = "lambda";

/** @type {"box" | "exp"} */
let mandelCanvasMode = "box";

/** Multibrot exponent p (Mandelbrot only; default 2). */
let mandelExponent = 2;

/** @type {{ pointerId: number; lastSx: number; lastSy: number } | null} */
let mandelExpDrag = null;

/**
 * Slow λ tour: phase-driven sin sweeps on Re/Im within [reMin,reMax]×[imMin,imMax].
 * @type {{ phase: number; reMin: number; reMax: number; imMin: number; imMax: number; paused: boolean; speedMult: number; dirRe: number; dirIm: number } | null}
 */
let juliaLambdaTour = null;

/** Speed when starting a tour (1× = slowest baseline). */
let juliaTourSpeedMult = 1;
let juliaTourDirRe = 1;
let juliaTourDirIm = 1;

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

function clampJuliaComponent(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.min(JULIA_C_LIM, Math.max(-JULIA_C_LIM, v));
}

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
  const re = clampJuliaComponent(Number(juliaReInput.value));
  const im = clampJuliaComponent(Number(juliaImInput.value));
  juliaReLabel.textContent = Number.isFinite(re) ? re.toFixed(6) : "—";
  juliaImLabel.textContent = Number.isFinite(im) ? im.toFixed(6) : "—";
}

/** String for Julia number fields: full float in [-2,2], no artificial grid. */
function formatJuliaFieldValue(v) {
  if (!Number.isFinite(v)) return "0";
  v = clampJuliaComponent(v);
  const x = parseFloat(v.toFixed(12));
  if (!Number.isFinite(x) || x === 0) return "0";
  return String(x);
}

/** Clamp `julia`, write HUD fields + labels (after λ-drag or programmatic edits). */
function syncJuliaHudFromJuliaState() {
  julia.re = clampJuliaComponent(julia.re);
  julia.im = clampJuliaComponent(julia.im);
  syncingJuliaInputsFromState = true;
  try {
    juliaReInput.value = formatJuliaFieldValue(julia.re);
    juliaImInput.value = formatJuliaFieldValue(julia.im);
  } finally {
    syncingJuliaInputsFromState = false;
  }
  juliaReLabel.textContent = julia.re.toFixed(6);
  juliaImLabel.textContent = julia.im.toFixed(6);
}

function clearCanvasPointerInteraction() {
  rubber = null;
  rubberPointerId = null;
  juliaLambdaDrag = null;
}

function setJuliaCanvasMode(mode) {
  if (mode !== "box" && mode !== "lambda") return;
  juliaCanvasMode = mode;
  const isBox = mode === "box";
  juliaModeBoxBtn?.classList.toggle("juliaModeBtnActive", isBox);
  juliaModeLambdaBtn?.classList.toggle("juliaModeBtnActive", !isBox);
  juliaModeBoxBtn?.setAttribute("aria-pressed", isBox ? "true" : "false");
  juliaModeLambdaBtn?.setAttribute("aria-pressed", isBox ? "false" : "true");
}

function juliaCanvasModeIsLambdaDrag() {
  return juliaCanvasMode === "lambda";
}

function clampMandelExponent(p) {
  if (!Number.isFinite(p)) return 2;
  return Math.min(MANDEL_EXP_MAX, Math.max(MANDEL_EXP_MIN, p));
}

function setMandelCanvasMode(mode) {
  if (mode !== "box" && mode !== "exp") return;
  mandelCanvasMode = mode;
  const isBox = mode === "box";
  mandelModeBoxBtn?.classList.toggle("juliaModeBtnActive", isBox);
  mandelModeExpBtn?.classList.toggle("juliaModeBtnActive", !isBox);
  mandelModeBoxBtn?.setAttribute("aria-pressed", isBox ? "true" : "false");
  mandelModeExpBtn?.setAttribute("aria-pressed", isBox ? "false" : "true");
}

function mandelCanvasModeIsExpDrag() {
  return mandelCanvasMode === "exp";
}

function updateJuliaTourControlsUI() {
  const playing = !!(juliaLambdaTour && !juliaLambdaTour.paused);
  const hasTour = !!juliaLambdaTour;

  if (juliaTourPlayBtn) {
    juliaTourPlayBtn.disabled = playing;
  }
  if (juliaTourPauseBtn) {
    juliaTourPauseBtn.disabled = !hasTour || !!juliaLambdaTour?.paused;
  }

  const speedActive = juliaLambdaTour?.speedMult ?? juliaTourSpeedMult;
  juliaTourSpeedBtns.forEach((btn) => {
    const v = Number(btn.getAttribute("data-speed"));
    btn.classList.toggle("juliaTourSpeedBtnActive", v === speedActive);
  });

  const dr = juliaLambdaTour?.dirRe ?? juliaTourDirRe;
  const di = juliaLambdaTour?.dirIm ?? juliaTourDirIm;
  juliaTourDirReBtn?.setAttribute("aria-pressed", dr < 0 ? "true" : "false");
  juliaTourDirImBtn?.setAttribute("aria-pressed", di < 0 ? "true" : "false");
}

function stopJuliaLambdaTour() {
  if (!juliaLambdaTour) return;
  juliaLambdaTour = null;
  syncJuliaHudFromJuliaState();
  updateJuliaTourControlsUI();
}

/**
 * Start or resume the tour from the **current** λ (only internal `phase` is adjusted; `julia` unchanged).
 */
function playJuliaLambdaTourFromUi() {
  if (fractalKind !== 1) return;
  if (juliaLambdaTour) {
    if (juliaLambdaTour.paused) {
      resyncTourPhaseFromJulia();
      juliaLambdaTour.paused = false;
    }
  } else {
    const w = JULIA_C_LIM;
    juliaLambdaTour = {
      phase: 0,
      reMin: -w,
      reMax: w,
      imMin: -w,
      imMax: w,
      paused: false,
      speedMult: juliaTourSpeedMult,
      dirRe: juliaTourDirRe,
      dirIm: juliaTourDirIm,
    };
    resyncTourPhaseFromJulia();
  }
  updateJuliaTourControlsUI();
  invalidateCache();
}

function pauseJuliaLambdaTourFromUi() {
  if (!juliaLambdaTour || juliaLambdaTour.paused) return;
  juliaLambdaTour.paused = true;
  updateJuliaTourControlsUI();
}

function setJuliaTourSpeedMult(mult) {
  if (!Number.isFinite(mult) || mult <= 0) return;
  juliaTourSpeedMult = mult;
  if (juliaLambdaTour) {
    juliaLambdaTour.speedMult = mult;
    resyncTourPhaseFromJulia();
    invalidateCache();
  }
  updateJuliaTourControlsUI();
}

function toggleJuliaTourDirRe() {
  juliaTourDirRe *= -1;
  if (juliaLambdaTour) {
    juliaLambdaTour.dirRe *= -1;
    resyncTourPhaseFromJulia();
    invalidateCache();
  }
  updateJuliaTourControlsUI();
}

function toggleJuliaTourDirIm() {
  juliaTourDirIm *= -1;
  if (juliaLambdaTour) {
    juliaLambdaTour.dirIm *= -1;
    resyncTourPhaseFromJulia();
    invalidateCache();
  }
  updateJuliaTourControlsUI();
}

function juliaLambdaTourDPhase() {
  if (!juliaLambdaTour) return 0;
  const nearOrigin =
    Math.abs(julia.re) <= JULIA_TOUR_ORIGIN_HALF && Math.abs(julia.im) <= JULIA_TOUR_ORIGIN_HALF;
  const base =
    JULIA_TOUR_DPHASE_RAD * (nearOrigin ? JULIA_TOUR_NEAR_ORIGIN_MULT : 1) * juliaLambdaTour.speedMult;
  return base;
}

/** @param {number} phi @param {{ reMin: number; reMax: number; imMin: number; imMax: number; dirRe?: number; dirIm?: number }} t */
function tourLambdaAtPhase(phi, t) {
  const halfRe = 0.5 * (t.reMax - t.reMin);
  const midRe = 0.5 * (t.reMax + t.reMin);
  const halfIm = 0.5 * (t.imMax - t.imMin);
  const midIm = 0.5 * (t.imMax + t.imMin);
  const dirRe = t.dirRe ?? 1;
  const dirIm = t.dirIm ?? 1;
  return {
    re: clampJuliaComponent(midRe + dirRe * halfRe * Math.sin(phi)),
    im: clampJuliaComponent(midIm + dirIm * halfIm * Math.sin(phi * JULIA_TOUR_IM_PHASE_MUL)),
  };
}

/**
 * Set internal phase so the *next* tick (phase += dPhase) lands on the curve point closest to
 * current λ. Global coarse scan + local refine so resume after drag does not pick a wrong
 * local minimum when `phase` is large.
 */
function resyncTourPhaseFromJulia() {
  if (!juliaLambdaTour) return;
  const t = juliaLambdaTour;
  const tr = clampJuliaComponent(julia.re);
  const ti = clampJuliaComponent(julia.im);
  const dPhase = juliaLambdaTourDPhase();

  function errAtPhi(phi) {
    const raw = tourLambdaAtPhase(phi, t);
    return (raw.re - tr) ** 2 + (raw.im - ti) ** 2;
  }

  let bestPhi = t.phase;
  let bestErr = errAtPhi(t.phase);

  const phiMax = JULIA_TOUR_RESYNC_PHI_MAX;
  const coarseN = 14000;
  for (let i = 0; i <= coarseN; i++) {
    const phi = (phiMax * i) / coarseN;
    const err = errAtPhi(phi);
    if (err < bestErr) {
      bestErr = err;
      bestPhi = phi;
    }
  }

  const refineSpan = 2 * Math.PI;
  const fineN = 2400;
  for (let pass = 0; pass < 2; pass++) {
    for (let j = 0; j <= fineN; j++) {
      const phi = bestPhi - refineSpan + (2 * refineSpan * j) / fineN;
      const err = errAtPhi(phi);
      if (err < bestErr) {
        bestErr = err;
        bestPhi = phi;
      }
    }
  }

  t.phase = bestPhi - dPhase;
}

function resumeJuliaLambdaTourAfterLambdaDrag() {
  if (!juliaLambdaTour?.paused) return;
  juliaLambdaTour.paused = false;
  syncJuliaHudFromJuliaState();
  resyncTourPhaseFromJulia();
  updateJuliaTourControlsUI();
  invalidateCache();
}

/** Advance tour: sin maps phase → [-1,1] with zero derivative at ends (smooth rubber). */
function tickJuliaLambdaTour() {
  if (!juliaLambdaTour || fractalKind !== 1) return;
  const t = juliaLambdaTour;
  if (t.paused) return;
  t.phase += juliaLambdaTourDPhase();
  const { re, im } = tourLambdaAtPhase(t.phase, t);
  julia.re = re;
  julia.im = im;
  juliaReLabel.textContent = julia.re.toFixed(6);
  juliaImLabel.textContent = julia.im.toFixed(6);
  invalidateCache();
}

function applyRenderParamsFromInputs() {
  maxIterUser = readMaxIterFromInput();
  maxIterInput.value = String(maxIterUser);
  syncIterLabel();
  julia.re = clampJuliaComponent(Number(juliaReInput.value));
  julia.im = clampJuliaComponent(Number(juliaImInput.value));
  if (!Number.isFinite(julia.re)) julia.re = 0;
  if (!Number.isFinite(julia.im)) julia.im = 0;
  previewJuliaLabelsFromInputs();
}

/** Push HUD inputs into `maxIterUser` / `julia` and schedule a redraw. */
function syncParamsFromInputs() {
  applyRenderParamsFromInputs();
  invalidateCache();
}

function updateFractalPanelsVisibility() {
  juliaPanel.hidden = fractalKind !== 1;
  if (mandelPanel) {
    mandelPanel.hidden = fractalKind !== 0;
  }
}

function buildFingerprint() {
  const cw = canvas.width;
  const ch = canvas.height;
  const jKey = fractalKind === 1 ? `|${julia.re}|${julia.im}` : "";
  const pKey = fractalKind === 0 ? `|exp${mandelExponent}` : "";
  return `${cw}x${ch}|${fractalKind}|${maxIterUser}${jKey}${pKey}|p${currentPaletteId()}`;
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
    mandelExponent: fractalKind === 0 ? mandelExponent : 2,
  };
}

/**
 * @returns {number} ms GPU time for compute + present
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
  stopJuliaLambdaTour();
  clearCanvasPointerInteraction();
  const d = defaultViewForFractalKind(fractalKind);
  view.centerX = d.centerX;
  view.centerY = d.centerY;
  view.halfW = d.halfW;
  if (fractalKind === 0) {
    mandelExponent = 2;
  }
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
    default: {
      const p = mandelExponent;
      const pStr = Math.abs(p - Math.round(p)) < 1e-4 ? String(Math.round(p)) : p.toFixed(3);
      const title = Math.abs(p - 2) < 1e-5 ? "Mandelbrot" : "Mandelbrot (multibrot)";
      return [title, `z' = z^${pStr} + c,  z0 = 0,  c = pixel`];
    }
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

  if (juliaLambdaTour && fractalKind === 1) {
    tickJuliaLambdaTour();
  }

  const fp = buildFingerprint();
  const paramsMatch = cacheReady && cacheFingerprint === fp;
  if (!paramsMatch || !viewsEqual(view, committedView)) {
    ms = fullRenderAndCommit();
  }

  uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
  drawViewOverlay();
  drawRubberOverlay();

  const tag = gpuRenderer?.usedGpuPerturb ? " · GPU perturb" : "";
  const tourTag =
    juliaLambdaTour && fractalKind === 1
      ? juliaLambdaTour.paused
        ? " · λ tour (paused)"
        : " · λ tour"
      : "";
  statusEl.textContent =
    ms > 0
      ? `${cw}×${ch}${tag} · ${ms.toFixed(1)} ms${tourTag}`
      : `${cw}×${ch}${tag}${tourTag}`;

  requestAnimationFrame(frame);
}

uiCanvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  uiCanvas.setPointerCapture(e.pointerId);
  rubberPointerId = e.pointerId;
  const { sx, sy } = canvasCoords(e.clientX, e.clientY);
  if (fractalKind === 1 && juliaCanvasModeIsLambdaDrag()) {
    rubber = null;
    let resumeTourAfter = false;
    if (juliaLambdaTour) {
      resumeTourAfter = !juliaLambdaTour.paused;
      juliaLambdaTour.paused = true;
      updateJuliaTourControlsUI();
    }
    juliaLambdaDrag = { pointerId: e.pointerId, lastSx: sx, lastSy: sy, resumeTourAfter };
    return;
  }
  if (fractalKind === 0 && mandelCanvasModeIsExpDrag()) {
    rubber = null;
    juliaLambdaDrag = null;
    mandelExpDrag = { pointerId: e.pointerId, lastSx: sx, lastSy: sy };
    return;
  }
  juliaLambdaDrag = null;
  mandelExpDrag = null;
  rubber = { x0: sx, y0: sy, x1: sx, y1: sy };
});

uiCanvas.addEventListener("pointermove", (e) => {
  if (e.pointerId !== rubberPointerId) return;
  const { sx, sy } = canvasCoords(e.clientX, e.clientY);
  if (juliaLambdaDrag) {
    const dx = sx - juliaLambdaDrag.lastSx;
    const dy = sy - juliaLambdaDrag.lastSy;
    const mult = e.shiftKey ? JULIA_LAMBDA_SHIFT_MULT : 1;
    const step = JULIA_LAMBDA_PER_CANVAS_PX * mult;
    julia.re = clampJuliaComponent(julia.re + dx * step);
    julia.im = clampJuliaComponent(julia.im + dy * step);
    syncJuliaHudFromJuliaState();
    juliaLambdaDrag.lastSx = sx;
    juliaLambdaDrag.lastSy = sy;
    invalidateCache();
    return;
  }
  if (mandelExpDrag) {
    const dx = sx - mandelExpDrag.lastSx;
    const mult = e.shiftKey ? MANDEL_EXP_SHIFT_MULT : 1;
    mandelExponent = clampMandelExponent(mandelExponent + dx * MANDEL_EXP_PER_CANVAS_PX * mult);
    mandelExpDrag.lastSx = sx;
    mandelExpDrag.lastSy = sy;
    invalidateCache();
    return;
  }
  if (rubber === null) return;
  rubber.x1 = sx;
  rubber.y1 = sy;
});

uiCanvas.addEventListener("pointerup", (e) => {
  if (e.pointerId !== rubberPointerId) return;
  uiCanvas.releasePointerCapture(e.pointerId);
  rubberPointerId = null;

  if (juliaLambdaDrag) {
    const resumeTour = juliaLambdaDrag.resumeTourAfter;
    juliaLambdaDrag = null;
    if (resumeTour) {
      resumeJuliaLambdaTourAfterLambdaDrag();
    }
    return;
  }

  if (mandelExpDrag) {
    mandelExpDrag = null;
    return;
  }

  if (rubber === null) return;

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
  if (rubberPointerId !== e.pointerId) return;
  try {
    uiCanvas.releasePointerCapture(e.pointerId);
  } catch {
    /* already released */
  }
  const resumeTour = juliaLambdaDrag?.resumeTourAfter;
  const hadJuliaLambdaDrag = !!juliaLambdaDrag;
  clearCanvasPointerInteraction();
  if (hadJuliaLambdaDrag && resumeTour) {
    resumeJuliaLambdaTourAfterLambdaDrag();
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
  if (fractalKind === 0) {
    mandelExponent = 2;
  }
  if (fractalKind === 1) {
    stopJuliaLambdaTour();
  }
  fullRenderAndCommit();
});

fractalSelect.addEventListener("change", () => {
  let fk = Number(fractalSelect.value);
  if (fk === 3 || !Number.isFinite(fk)) fk = 0;
  fractalSelect.value = String(fk);
  fractalKind = fk;
  mandelExponent = 2;
  stopJuliaLambdaTour();
  clearCanvasPointerInteraction();
  updateFractalPanelsVisibility();
  syncParamsFromInputs();
});

paletteSelect.addEventListener("change", () => {
  invalidateCache();
});

juliaReInput.addEventListener("input", () => {
  if (syncingJuliaInputsFromState) return;
  stopJuliaLambdaTour();
  syncParamsFromInputs();
});

juliaImInput.addEventListener("input", () => {
  if (syncingJuliaInputsFromState) return;
  stopJuliaLambdaTour();
  syncParamsFromInputs();
});

juliaModeBoxBtn?.addEventListener("click", () => setJuliaCanvasMode("box"));
juliaModeLambdaBtn?.addEventListener("click", () => setJuliaCanvasMode("lambda"));

juliaTourPlayBtn?.addEventListener("click", () => {
  playJuliaLambdaTourFromUi();
});

juliaTourPauseBtn?.addEventListener("click", () => {
  pauseJuliaLambdaTourFromUi();
});

juliaTourDirReBtn?.addEventListener("click", () => {
  toggleJuliaTourDirRe();
});

juliaTourDirImBtn?.addEventListener("click", () => {
  toggleJuliaTourDirIm();
});

juliaTourSpeedBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const v = Number(btn.getAttribute("data-speed"));
    setJuliaTourSpeedMult(v);
  });
});

mandelModeBoxBtn?.addEventListener("click", () => setMandelCanvasMode("box"));
mandelModeExpBtn?.addEventListener("click", () => setMandelCanvasMode("exp"));

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
  updateFractalPanelsVisibility();
  setJuliaCanvasMode("lambda");
  setMandelCanvasMode("box");
  updateJuliaTourControlsUI();

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
