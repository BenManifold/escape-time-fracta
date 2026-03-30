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
/** λ disk: keep pointer mapping inside the rim (px). */
const JULIA_LAMBDA_WHEEL_INSET = 8;
/** Knob offset from center as % of disk width/height (matches pointer mapping). */
const JULIA_LAMBDA_WHEEL_KNOB_FRAC = 42;
/** Julia “λ drag” mode: one backing-store pixel of motion changes λ by this much on that axis. */
const JULIA_LAMBDA_PER_CANVAS_PX = 0.0001;
/** With Shift held, scale the above by this factor (0.001× → finer strokes). */
const JULIA_LAMBDA_SHIFT_MULT = 0.001;

/** Damped spring toward release point after λ-drag (a = k·(p₀−p) − c·v). Defaults; live values from HUD sliders. */
const JIGGLE_DEFAULT_SPRING_K = 265;
const JIGGLE_DEFAULT_DAMPING_C = 9;
const JIGGLE_DEFAULT_VELOCITY_GAIN = 3;
const JIGGLE_DEFAULT_VEL_CAP = 70;
/** Below this speed (in λ units/sec, after gain), skip jiggle. */
const JIGGLE_SPEED_THRESHOLD = 4e-5;
/** Impulse scale when λ tour step (−1/0/1 per axis) changes and jiggle snap is on. */
const JIGGLE_DIR_KICK_PER_STEP = 0.014;
const JIGGLE_POS_EPS = 1e-7;
const JIGGLE_VEL_EPS = 2e-6;
const JIGGLE_MAX_SEC = 3.2;

/**
 * Julia λ tour: phase advance per animation frame (rad). Re/Im sweep [-JULIA_C_LIM, JULIA_C_LIM]
 * via sin (zero slope at extrema ⇒ smooth rubber). Peak |Δλ|/frame ≈ JULIA_C_LIM * DPHASE (e.g. ~1e-7 when DPHASE=5e-8).
 */
const JULIA_TOUR_DPHASE_RAD = 5e-8;
/** Im uses phase * this factor so Re/Im don’t stay locked in phase. */
const JULIA_TOUR_IM_PHASE_MUL = 0.618033988749895;
/** Upper bound for global φ scan on resync (Lissajous is quasi-periodic; wide span avoids wrong local minima). */
const JULIA_TOUR_RESYNC_PHI_MAX = 3000 * Math.PI
/** Min half-width of Lissajous box per axis (λ at a face still gets a tiny oscillation). */
const JULIA_TOUR_MIN_BOX_HALF = 0.02

/** Random-walk: ms between independent Re/Im step picks (−1, 0, +1 per axis). */
const JULIA_RW_CHOICE_MS = 5000;
/** Random-walk: seconds to ease heading / step magnitude after a discrete direction change. */
const JULIA_RW_DIR_BLEND_SEC = 0.15;

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
const juliaLambdaWheel = document.getElementById("juliaLambdaWheel");
const juliaLambdaWheelKnob = document.getElementById("juliaLambdaWheelKnob");
const juliaReLabel = document.getElementById("juliaReLabel");
const juliaImLabel = document.getElementById("juliaImLabel");
const juliaModeBoxBtn = document.getElementById("juliaModeBox");
const juliaModeLambdaBtn = document.getElementById("juliaModeLambda");
const juliaTourPlayBtn = document.getElementById("juliaTourPlay");
const juliaTourPauseBtn = document.getElementById("juliaTourPause");
const juliaTourDirPad = document.getElementById("juliaTourDirPad");
const juliaPathRandomWalkBtn = document.getElementById("juliaPathRandomWalk");
const juliaTourSpeedBtns = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll(".juliaTourSpeedBtn")
);
const juliaJiggleSnapBtn = document.getElementById("juliaJiggleSnapBtn");
const juliaJiggleSpringInput = document.getElementById("juliaJiggleSpring");
const juliaJiggleDampingInput = document.getElementById("juliaJiggleDamping");
const juliaJiggleCarryInput = document.getElementById("juliaJiggleCarry");
const juliaJiggleCapInput = document.getElementById("juliaJiggleCap");
const juliaJiggleSpringVal = document.getElementById("juliaJiggleSpringVal");
const juliaJiggleDampingVal = document.getElementById("juliaJiggleDampingVal");
const juliaJiggleCarryVal = document.getElementById("juliaJiggleCarryVal");
const juliaJiggleCapVal = document.getElementById("juliaJiggleCapVal");
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

/** Default λ for Julia (matches HUD initial inputs). */
const DEFAULT_JULIA_RE = -0.7269;
const DEFAULT_JULIA_IM = 0.1889;

const julia = {
  re: DEFAULT_JULIA_RE,
  im: DEFAULT_JULIA_IM,
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
 * @type {{
 *   pointerId: number;
 *   lastSx: number;
 *   lastSy: number;
 *   resumeTourAfter: boolean;
 *   lastMoveT: number;
 *   velRe: number;
 *   velIm: number;
 *   lastDRe: number;
 *   lastDIm: number;
 * } | null}
 */
let juliaLambdaDrag = null;

/**
 * HUD λ disk drag: velocity for jiggle on release (same physics as canvas λ-drag).
 * @type {{
 *   lastMoveT: number;
 *   velRe: number;
 *   velIm: number;
 *   lastDRe: number;
 *   lastDIm: number;
 * } | null}
 */
let juliaLambdaWheelDrag = null;

/** Canvas λ-drag release: damped settle toward drop point. */
let juliaJiggleSnapEnabled = true;

let jiggleSpringK = JIGGLE_DEFAULT_SPRING_K;
let jiggleDampingC = JIGGLE_DEFAULT_DAMPING_C;
let jiggleVelocityGain = JIGGLE_DEFAULT_VELOCITY_GAIN;
let jiggleVelCap = JIGGLE_DEFAULT_VEL_CAP;

/**
 * @type {{
 *   targetRe: number;
 *   targetIm: number;
 *   velRe: number;
 *   velIm: number;
 *   t0: number;
 * } | null}
 */
let juliaJiggle = null;

/** Tour resume deferred until jiggle finishes (when drag had paused a playing tour). */
let pendingTourResumeAfterJiggle = false;

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
 * Slow λ tour: Lissajous sin sweeps or random-walk axis steps.
 * @type {{
 *   phase: number;
 *   reMin: number;
 *   reMax: number;
 *   imMin: number;
 *   imMax: number;
 *   paused: boolean;
 *   speedMult: number;
 *   dirRe: number;
 *   dirIm: number;
 *   pathMode: "lissajous" | "randomWalk";
 *   rwChoiceAccumMs?: number;
 *   rwDirBlendRemainSec?: number;
 *   rwBlendMode?: "turn" | "fromIdle" | "toIdle";
 *   rwBlendFromUx?: number;
 *   rwBlendFromUy?: number;
 *   rwBlendToUx?: number;
 *   rwBlendToUy?: number;
 * } | null}
 */
let juliaLambdaTour = null;

/** Speed when starting a tour (1× = slowest baseline). */
let juliaTourSpeedMult = 1;
let juliaTourDirRe = 1;
let juliaTourDirIm = 1;

/** HUD intent for new / resumed tours (random walk off ⇒ Lissajous box sweep). */
let juliaTourPathIntent = /** @type {"lissajous" | "randomWalk"} */ ("lissajous");

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
  updateJuliaLambdaWheelVisual();
}

/**
 * Move the λ disk knob and aria to match `julia` (disk maps Re/Im to horizontal/vertical in ±JULIA_C_LIM).
 */
function updateJuliaLambdaWheelVisual() {
  const knob = juliaLambdaWheelKnob;
  const wheel = juliaLambdaWheel;
  if (!knob || !wheel) return;
  let nx = julia.re / JULIA_C_LIM;
  let nyScr = -julia.im / JULIA_C_LIM;
  if (!Number.isFinite(nx)) nx = 0;
  if (!Number.isFinite(nyScr)) nyScr = 0;
  const d = Math.hypot(nx, nyScr);
  if (d > 1 && d > 1e-12) {
    nx /= d;
    nyScr /= d;
  }
  const fr = JULIA_LAMBDA_WHEEL_KNOB_FRAC;
  knob.style.left = `calc(50% + ${nx * fr}%)`;
  knob.style.top = `calc(50% + ${nyScr * fr}%)`;
  wheel.setAttribute("aria-valuetext", `Re ${julia.re.toFixed(4)}, Im ${julia.im.toFixed(4)}`);
}

/**
 * @param {number} clientX
 * @param {number} clientY
 */
function lambdaWheelClientToReIm(clientX, clientY) {
  const el = juliaLambdaWheel;
  if (!el) {
    return { re: julia.re, im: julia.im };
  }
  const rect = el.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const radiusPx = Math.max(
    10,
    Math.min(rect.width, rect.height) / 2 - JULIA_LAMBDA_WHEEL_INSET,
  );
  let dx = clientX - cx;
  let dy = clientY - cy;
  const dist = Math.hypot(dx, dy);
  if (dist > radiusPx && dist > 1e-6) {
    dx *= radiusPx / dist;
    dy *= radiusPx / dist;
  }
  const nx = dx / radiusPx;
  const ny = dy / radiusPx;
  return {
    re: clampJuliaComponent(nx * JULIA_C_LIM),
    im: clampJuliaComponent(-ny * JULIA_C_LIM),
  };
}

/**
 * @param {number} clientX
 * @param {number} clientY
 * @param {{ trackVelocity?: boolean }} [opts]
 */
function applyJuliaLambdaFromWheelClient(clientX, clientY, opts) {
  if (fractalKind !== 1) return;
  const prevRe = julia.re;
  const prevIm = julia.im;
  const { re, im } = lambdaWheelClientToReIm(clientX, clientY);
  julia.re = re;
  julia.im = im;
  if (opts?.trackVelocity && juliaLambdaWheelDrag) {
    const now = performance.now();
    const dRe = re - prevRe;
    const dIm = im - prevIm;
    if (juliaLambdaWheelDrag.lastMoveT > 0) {
      const dtSec = Math.max(1e-4, (now - juliaLambdaWheelDrag.lastMoveT) / 1000);
      juliaLambdaWheelDrag.velRe = dRe / dtSec;
      juliaLambdaWheelDrag.velIm = dIm / dtSec;
    }
    juliaLambdaWheelDrag.lastMoveT = now;
    juliaLambdaWheelDrag.lastDRe = dRe;
    juliaLambdaWheelDrag.lastDIm = dIm;
  }
  syncJuliaHudFromJuliaState();
  invalidateCache();
}

function clearCanvasPointerInteraction() {
  rubber = null;
  rubberPointerId = null;
  juliaLambdaDrag = null;
}

function clearJiggleSnapState() {
  if (juliaJiggle) {
    julia.re = clampJuliaComponent(juliaJiggle.targetRe);
    julia.im = clampJuliaComponent(juliaJiggle.targetIm);
    juliaJiggle = null;
    syncJuliaHudFromJuliaState();
  }
  pendingTourResumeAfterJiggle = false;
}

function updateJuliaJiggleSnapToggleUI() {
  if (!juliaJiggleSnapBtn) return;
  juliaJiggleSnapBtn.classList.toggle("juliaModeBtnActive", juliaJiggleSnapEnabled);
  juliaJiggleSnapBtn.setAttribute("aria-pressed", juliaJiggleSnapEnabled ? "true" : "false");
}

function updateJiggleParamValueLabels() {
  if (juliaJiggleSpringVal) {
    juliaJiggleSpringVal.textContent = String(Math.round(jiggleSpringK));
  }
  if (juliaJiggleDampingVal) {
    juliaJiggleDampingVal.textContent = String(Math.round(jiggleDampingC));
  }
  if (juliaJiggleCarryVal) {
    juliaJiggleCarryVal.textContent =
      Math.abs(jiggleVelocityGain - Math.round(jiggleVelocityGain)) < 1e-6
        ? String(Math.round(jiggleVelocityGain))
        : jiggleVelocityGain.toFixed(1);
  }
  if (juliaJiggleCapVal) {
    juliaJiggleCapVal.textContent = String(Math.round(jiggleVelCap));
  }
}

/** Read elastic-release sliders into runtime spring parameters. */
function syncJigglePhysicsFromInputs() {
  if (juliaJiggleSpringInput) {
    const v = Number(juliaJiggleSpringInput.value);
    jiggleSpringK = Number.isFinite(v) ? v : JIGGLE_DEFAULT_SPRING_K;
  }
  if (juliaJiggleDampingInput) {
    const v = Number(juliaJiggleDampingInput.value);
    jiggleDampingC = Number.isFinite(v) ? v : JIGGLE_DEFAULT_DAMPING_C;
  }
  if (juliaJiggleCarryInput) {
    const v = Number(juliaJiggleCarryInput.value) / 10;
    jiggleVelocityGain = Number.isFinite(v) ? v : JIGGLE_DEFAULT_VELOCITY_GAIN;
  }
  if (juliaJiggleCapInput) {
    const v = Number(juliaJiggleCapInput.value);
    jiggleVelCap = Number.isFinite(v) ? v : JIGGLE_DEFAULT_VEL_CAP;
  }
  updateJiggleParamValueLabels();
}

/** No elastic jiggle while random-walk path is selected or the tour is in random-walk mode. */
function juliaJiggleDisabledForRandomWalk() {
  return (
    juliaTourPathIntent === "randomWalk" ||
    (juliaLambdaTour?.pathMode ?? "lissajous") === "randomWalk"
  );
}

/**
 * Start jiggle from pointer-drag λ velocity (canvas drag or λ disk).
 * @param {{ velRe?: number; velIm?: number; lastDRe: number; lastDIm: number }} velState
 * @param {boolean} pendingTourResumeAfter
 * @returns {boolean} true if jiggle started
 */
function tryJuliaJiggleAfterPointerDragVelocity(velState, pendingTourResumeAfter) {
  if (
    !juliaJiggleSnapEnabled ||
    fractalKind !== 1 ||
    juliaJiggle ||
    juliaJiggleDisabledForRandomWalk()
  ) {
    return false;
  }
  let vRe = (velState.velRe ?? 0) * jiggleVelocityGain;
  let vIm = (velState.velIm ?? 0) * jiggleVelocityGain;
  let speed = Math.hypot(vRe, vIm);
  if (speed < JIGGLE_SPEED_THRESHOLD && (velState.lastDRe !== 0 || velState.lastDIm !== 0)) {
    const fallbackDt = 1 / 60;
    vRe = (velState.lastDRe / fallbackDt) * jiggleVelocityGain;
    vIm = (velState.lastDIm / fallbackDt) * jiggleVelocityGain;
    speed = Math.hypot(vRe, vIm);
  }
  if (speed < JIGGLE_SPEED_THRESHOLD) return false;
  juliaJiggle = {
    targetRe: julia.re,
    targetIm: julia.im,
    velRe: Math.max(-jiggleVelCap, Math.min(jiggleVelCap, vRe)),
    velIm: Math.max(-jiggleVelCap, Math.min(jiggleVelCap, vIm)),
    t0: performance.now(),
  };
  pendingTourResumeAfterJiggle = pendingTourResumeAfter;
  invalidateCache();
  return true;
}

/**
 * End canvas λ-drag: optional damped jiggle toward release point, or immediate tour resume.
 * @param {boolean} resumeTourAfter
 */
function endJuliaLambdaDragFromPointer(resumeTourAfter) {
  const drag = juliaLambdaDrag;
  juliaLambdaDrag = null;
  if (!drag) {
    if (resumeTourAfter) {
      resumeJuliaLambdaTourAfterLambdaDrag();
    }
    invalidateCache();
    return;
  }

  if (tryJuliaJiggleAfterPointerDragVelocity(drag, resumeTourAfter)) {
    return;
  }

  if (resumeTourAfter) {
    resumeJuliaLambdaTourAfterLambdaDrag();
  }
  invalidateCache();
}

function endJuliaLambdaWheelDragFromPointer() {
  const d = juliaLambdaWheelDrag;
  juliaLambdaWheelDrag = null;
  if (!d) {
    invalidateCache();
    return;
  }
  if (tryJuliaJiggleAfterPointerDragVelocity(d, false)) {
    return;
  }
  invalidateCache();
}

/**
 * If jiggle snap is enabled, start a short spring settle at the current λ with an impulse from the
 * tour direction change (same physics as drag release). Skips if the pad step did not change.
 * @param {number} prevRe
 * @param {number} prevIm
 * @param {number} nextRe
 * @param {number} nextIm
 */
function tryStartJuliaJiggleAfterTourDirChange(prevRe, prevIm, nextRe, nextIm) {
  if (!juliaJiggleSnapEnabled || fractalKind !== 1 || juliaJiggleDisabledForRandomWalk()) return;
  const pr = juliaTourPadStep(prevRe);
  const pi = juliaTourPadStep(prevIm);
  const nr = juliaTourPadStep(nextRe);
  const ni = juliaTourPadStep(nextIm);
  if (pr === nr && pi === ni) return;

  let vRe = (nr - pr) * JIGGLE_DIR_KICK_PER_STEP * jiggleVelocityGain;
  let vIm = (ni - pi) * JIGGLE_DIR_KICK_PER_STEP * jiggleVelocityGain;
  const speed = Math.hypot(vRe, vIm);
  if (speed < JIGGLE_SPEED_THRESHOLD) return;

  if (juliaJiggle) {
    julia.re = clampJuliaComponent(juliaJiggle.targetRe);
    julia.im = clampJuliaComponent(juliaJiggle.targetIm);
    juliaJiggle = null;
    syncJuliaHudFromJuliaState();
  }

  juliaJiggle = {
    targetRe: julia.re,
    targetIm: julia.im,
    velRe: Math.max(-jiggleVelCap, Math.min(jiggleVelCap, vRe)),
    velIm: Math.max(-jiggleVelCap, Math.min(jiggleVelCap, vIm)),
    t0: performance.now(),
  };
  pendingTourResumeAfterJiggle = false;
  invalidateCache();
}

/** @param {number} dtSec */
function tickJuliaJiggleSnap(dtSec) {
  if (!juliaJiggle || fractalKind !== 1) return;
  const dt = Math.max(1e-4, Math.min(0.048, dtSec > 0 ? dtSec : 1 / 60));

  const j = juliaJiggle;
  let { velRe, velIm } = j;
  let re = julia.re;
  let im = julia.im;
  const { targetRe, targetIm } = j;

  const aRe = jiggleSpringK * (targetRe - re) - jiggleDampingC * velRe;
  const aIm = jiggleSpringK * (targetIm - im) - jiggleDampingC * velIm;
  velRe += aRe * dt;
  velIm += aIm * dt;
  re += velRe * dt;
  im += velIm * dt;
  re = clampJuliaComponent(re);
  im = clampJuliaComponent(im);

  julia.re = re;
  julia.im = im;
  j.velRe = velRe;
  j.velIm = velIm;

  const err = Math.hypot(re - targetRe, im - targetIm);
  const spd = Math.hypot(velRe, velIm);
  const elapsed = (performance.now() - j.t0) / 1000;

  if ((err < JIGGLE_POS_EPS && spd < JIGGLE_VEL_EPS) || elapsed > JIGGLE_MAX_SEC) {
    julia.re = clampJuliaComponent(targetRe);
    julia.im = clampJuliaComponent(targetIm);
    juliaJiggle = null;
    syncJuliaHudFromJuliaState();
    if (pendingTourResumeAfterJiggle) {
      pendingTourResumeAfterJiggle = false;
      resumeJuliaLambdaTourAfterLambdaDrag();
    }
    invalidateCache();
    return;
  }

  syncJuliaHudFromJuliaState();
  invalidateCache();
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

/** Quantize tour dir to pad cell (−1, 0, +1). */
function juliaTourPadStep(v) {
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
}

function juliaTourAxisDeltaLabel(axis, step) {
  if (step > 0) return `${axis} increasing`;
  if (step < 0) return `${axis} decreasing`;
  return `${axis} steady`;
}

function updateJuliaPathModeButtonsUI() {
  const rw = juliaTourPathIntent === "randomWalk";
  juliaPathRandomWalkBtn?.classList.toggle("juliaModeBtnActive", rw);
  juliaPathRandomWalkBtn?.setAttribute("aria-pressed", rw ? "true" : "false");
}

/** Lissajous sweep uses ±1 only. */
function normalizeLissajousDirSign(d) {
  if (d === 0 || !Number.isFinite(d)) return 1;
  return d > 0 ? 1 : -1;
}

/** Uniform −1, 0, +1 (e.g. enabling random walk). */
function randomWalkAxisStepUnconstrained() {
  const u = Math.random() * 3;
  if (u < 1) return -1;
  if (u < 2) return 0;
  return 1;
}

/**
 * Random-walk axis pick: if this axis was idle (0), next is only −1 or +1; else −1, 0, or +1.
 * @param {number} prev
 */
function randomWalkAxisStepFromPrev(prev) {
  if (prev === 0) return Math.random() < 0.5 ? -1 : 1;
  const u = Math.random() * 3;
  if (u < 1) return -1;
  if (u < 2) return 0;
  return 1;
}

/** Auto-rolls never use (0,0): full stop is only via the λ-step pad. */
function avoidRandomWalkCenterGrid(nextRe, nextIm) {
  if (nextRe !== 0 || nextIm !== 0) return { re: nextRe, im: nextIm };
  if (Math.random() < 0.5) {
    return { re: Math.random() < 0.5 ? -1 : 1, im: 0 };
  }
  return { re: 0, im: Math.random() < 0.5 ? -1 : 1 };
}

function randomWalkNormalizedDir(dre, dim) {
  const sx = dre ?? 0;
  const sy = dim ?? 0;
  const len = Math.hypot(sx, sy);
  if (len < 1e-15) return { ux: 0, uy: 0, len: 0 };
  return { ux: sx / len, uy: sy / len, len };
}

function clearRandomWalkDirBlend(t) {
  delete t.rwDirBlendRemainSec;
  delete t.rwBlendMode;
  delete t.rwBlendFromUx;
  delete t.rwBlendFromUy;
  delete t.rwBlendToUx;
  delete t.rwBlendToUy;
}

/**
 * After discrete (dirRe, dirIm) changes, ease motion for `JULIA_RW_DIR_BLEND_SEC` (heading lerp and step scale).
 * `t` already holds the new dir; `prevRe`/`prevIm` are the old pad steps.
 * @param {{ rwDirBlendRemainSec?: number; rwBlendMode?: string; dirRe?: number; dirIm?: number }} t
 * @param {number} prevRe
 * @param {number} prevIm
 */
function startRandomWalkDirBlend(t, prevRe, prevIm) {
  const from = randomWalkNormalizedDir(prevRe, prevIm);
  const to = randomWalkNormalizedDir(t.dirRe ?? 0, t.dirIm ?? 0);
  if (from.len < 1e-15 && to.len < 1e-15) return;
  const sameDir =
    from.len > 1e-15 &&
    to.len > 1e-15 &&
    Math.abs(from.ux - to.ux) < 1e-8 &&
    Math.abs(from.uy - to.uy) < 1e-8;
  if (sameDir) return;

  t.rwDirBlendRemainSec = JULIA_RW_DIR_BLEND_SEC;
  if (from.len < 1e-15) {
    t.rwBlendMode = "fromIdle";
    t.rwBlendToUx = to.ux;
    t.rwBlendToUy = to.uy;
  } else if (to.len < 1e-15) {
    t.rwBlendMode = "toIdle";
    t.rwBlendFromUx = from.ux;
    t.rwBlendFromUy = from.uy;
  } else {
    t.rwBlendMode = "turn";
    t.rwBlendFromUx = from.ux;
    t.rwBlendFromUy = from.uy;
    t.rwBlendToUx = to.ux;
    t.rwBlendToUy = to.uy;
  }
}

function rollRandomWalkBothAxes(t) {
  const prevRe = juliaTourPadStep(t.dirRe ?? 0);
  const prevIm = juliaTourPadStep(t.dirIm ?? 0);
  let nextRe = randomWalkAxisStepFromPrev(t.dirRe ?? 0);
  let nextIm = randomWalkAxisStepFromPrev(t.dirIm ?? 0);
  const adj = avoidRandomWalkCenterGrid(nextRe, nextIm);
  nextRe = adj.re;
  nextIm = adj.im;
  t.dirRe = nextRe;
  t.dirIm = nextIm;
  juliaTourDirRe = nextRe;
  juliaTourDirIm = nextIm;
  if (prevRe !== nextRe || prevIm !== nextIm) {
    startRandomWalkDirBlend(t, prevRe, prevIm);
  }
  tryStartJuliaJiggleAfterTourDirChange(prevRe, prevIm, nextRe, nextIm);
  updateJuliaTourControlsUI();
}

/**
 * Advance random-walk reroll clock (wall time). Only called when the tour path is advancing (not during
 * jiggle snap), so rerolls stay aligned with “moving along the path.”
 * @param {{ rwChoiceAccumMs?: number; paused?: boolean; pathMode?: string }} t
 * @param {number} dtSec
 */
function processRandomWalkDirectionRolls(t, dtSec) {
  if (t.paused) return;
  t.rwChoiceAccumMs = (t.rwChoiceAccumMs ?? 0) + dtSec * 1000;
  while (t.rwChoiceAccumMs >= JULIA_RW_CHOICE_MS) {
    t.rwChoiceAccumMs -= JULIA_RW_CHOICE_MS;
    rollRandomWalkBothAxes(t);
    if (juliaJiggle) break;
  }
}

/** Center the Lissajous rectangle on current λ with the largest symmetric half-ranges that fit in ±JULIA_C_LIM. */
function refreshLissajousTourBoundsFromJulia(t) {
  const mr = clampJuliaComponent(julia.re);
  const mi = clampJuliaComponent(julia.im);
  let halfRe = Math.min(JULIA_C_LIM - mr, mr + JULIA_C_LIM);
  let halfIm = Math.min(JULIA_C_LIM - mi, mi + JULIA_C_LIM);
  halfRe = Math.max(JULIA_TOUR_MIN_BOX_HALF, halfRe);
  halfIm = Math.max(JULIA_TOUR_MIN_BOX_HALF, halfIm);
  t.reMin = clampJuliaComponent(mr - halfRe);
  t.reMax = clampJuliaComponent(mr + halfRe);
  t.imMin = clampJuliaComponent(mi - halfIm);
  t.imMax = clampJuliaComponent(mi + halfIm);
}

function randomizeJuliaTourDirQuadrant() {
  const oldRe = juliaTourPadStep(juliaTourDirRe);
  const oldIm = juliaTourPadStep(juliaTourDirIm);
  let a = randomWalkAxisStepUnconstrained();
  let b = randomWalkAxisStepUnconstrained();
  const adj = avoidRandomWalkCenterGrid(a, b);
  a = adj.re;
  b = adj.im;
  juliaTourDirRe = a;
  juliaTourDirIm = b;
  if (juliaLambdaTour) {
    juliaLambdaTour.dirRe = a;
    juliaLambdaTour.dirIm = b;
  }
  tryStartJuliaJiggleAfterTourDirChange(oldRe, oldIm, a, b);
}

/**
 * Apply path shape intent to HUD and, if a tour is active, its `pathMode` and phase/velocity.
 * Random walk skips phase resync so λ is not moved by mode changes alone.
 */
function setJuliaTourPathIntent(intent) {
  juliaTourPathIntent = intent;
  updateJuliaPathModeButtonsUI();
  if (intent === "lissajous") {
    juliaTourDirRe = normalizeLissajousDirSign(juliaTourDirRe);
    juliaTourDirIm = normalizeLissajousDirSign(juliaTourDirIm);
  }
  if (juliaLambdaTour) {
    juliaLambdaTour.pathMode = intent;
    if (intent === "randomWalk") {
      juliaLambdaTour.rwChoiceAccumMs = juliaLambdaTour.rwChoiceAccumMs ?? 0;
    } else {
      clearRandomWalkDirBlend(juliaLambdaTour);
      juliaLambdaTour.dirRe = normalizeLissajousDirSign(juliaLambdaTour.dirRe ?? 1);
      juliaLambdaTour.dirIm = normalizeLissajousDirSign(juliaLambdaTour.dirIm ?? 1);
      if (intent === "lissajous") {
        refreshLissajousTourBoundsFromJulia(juliaLambdaTour);
      }
      resyncTourPhaseFromJulia();
    }
    invalidateCache();
  }
  updateJuliaTourControlsUI();
}

function onJuliaPathRandomWalkClick() {
  if (juliaTourPathIntent === "randomWalk") {
    setJuliaTourPathIntent("lissajous");
  } else {
    setJuliaTourPathIntent("randomWalk");
    randomizeJuliaTourDirQuadrant();
  }
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

  const dr = juliaTourPadStep(juliaLambdaTour?.dirRe ?? juliaTourDirRe);
  const di = juliaTourPadStep(juliaLambdaTour?.dirIm ?? juliaTourDirIm);

  juliaTourDirPad?.setAttribute(
    "aria-label",
    `λ step pad. Active: ${juliaTourAxisDeltaLabel("Re", dr)}, ${juliaTourAxisDeltaLabel("Im", di)}`,
  );

  juliaTourDirPad?.querySelectorAll(".juliaTourDirCell").forEach((cell) => {
    const el = /** @type {HTMLElement} */ (cell);
    const cr = juliaTourPadStep(Number(el.getAttribute("data-dre")));
    const ci = juliaTourPadStep(Number(el.getAttribute("data-dim")));
    el.classList.toggle("juliaTourDirCellActive", cr === dr && ci === di);
  });
}

function stopJuliaLambdaTour() {
  clearJiggleSnapState();
  if (!juliaLambdaTour) return;
  juliaLambdaTour = null;
  syncJuliaHudFromJuliaState();
  updateJuliaTourControlsUI();
}

/**
 * Start or resume the tour from the **current** λ (phase resync for Lissajous; random walk keeps λ until ticks).
 */
function playJuliaLambdaTourFromUi() {
  if (fractalKind !== 1) return;
  clearJiggleSnapState();
  if (juliaLambdaTour) {
    if (juliaLambdaTour.paused) {
      juliaLambdaTour.pathMode = juliaTourPathIntent;
      if (juliaTourPathIntent === "randomWalk") {
        juliaLambdaTour.rwChoiceAccumMs = juliaLambdaTour.rwChoiceAccumMs ?? 0;
      } else {
        juliaLambdaTour.dirRe = normalizeLissajousDirSign(juliaLambdaTour.dirRe ?? 1);
        juliaLambdaTour.dirIm = normalizeLissajousDirSign(juliaLambdaTour.dirIm ?? 1);
        if (juliaTourPathIntent === "lissajous") {
          refreshLissajousTourBoundsFromJulia(juliaLambdaTour);
        }
        resyncTourPhaseFromJulia();
      }
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
      pathMode: juliaTourPathIntent,
      ...(juliaTourPathIntent === "randomWalk" ? { rwChoiceAccumMs: 0 } : {}),
    };
    if (juliaTourPathIntent === "lissajous") {
      refreshLissajousTourBoundsFromJulia(juliaLambdaTour);
    }
    if (juliaTourPathIntent !== "randomWalk") {
      resyncTourPhaseFromJulia();
    }
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
    if (juliaLambdaTour.pathMode !== "randomWalk") {
      resyncTourPhaseFromJulia();
    }
    invalidateCache();
  }
  updateJuliaTourControlsUI();
}

/**
 * @param {number} cellRe
 * @param {number} cellIm
 */
function applyJuliaTourDirFromPadClick(cellRe, cellIm) {
  const prevRe = juliaTourPadStep(juliaLambdaTour?.dirRe ?? juliaTourDirRe);
  const prevIm = juliaTourPadStep(juliaLambdaTour?.dirIm ?? juliaTourDirIm);
  const rw = juliaTourPathIntent === "randomWalk";
  let re = juliaTourPadStep(cellRe);
  let im = juliaTourPadStep(cellIm);
  if (!rw) {
    re = re === 0 ? 1 : re;
    im = im === 0 ? 1 : im;
  }
  const changed = prevRe !== re || prevIm !== im;
  juliaTourDirRe = re;
  juliaTourDirIm = im;
  if (juliaLambdaTour) {
    juliaLambdaTour.dirRe = re;
    juliaLambdaTour.dirIm = im;
    if (changed && rw && juliaLambdaTour.pathMode === "randomWalk") {
      startRandomWalkDirBlend(juliaLambdaTour, prevRe, prevIm);
    }
  }
  if (changed) {
    tryStartJuliaJiggleAfterTourDirChange(prevRe, prevIm, re, im);
  }
  if (juliaLambdaTour) {
    if (!rw) {
      resyncTourPhaseFromJulia();
    }
    invalidateCache();
  }
  updateJuliaTourControlsUI();
}

function juliaLambdaTourDPhase() {
  if (!juliaLambdaTour) return 0;
  return JULIA_TOUR_DPHASE_RAD * juliaLambdaTour.speedMult;
}

/**
 * @param {number} phi
 * @param {{
 *   reMin: number;
 *   reMax: number;
 *   imMin: number;
 *   imMax: number;
 *   dirRe?: number;
 *   dirIm?: number;
 *   pathMode?: "lissajous" | "randomWalk";
 * }} t
 */
function tourLambdaAtPhase(phi, t) {
  const mode = t.pathMode ?? "lissajous";
  if (mode === "randomWalk") {
    return {
      re: clampJuliaComponent(julia.re),
      im: clampJuliaComponent(julia.im),
    };
  }
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
  const mode = t.pathMode ?? "lissajous";
  if (mode === "randomWalk") return;

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
  const mode = juliaLambdaTour.pathMode ?? "lissajous";
  if (mode === "randomWalk") {
    /* keep λ where the user released */
  } else {
    if (mode === "lissajous") {
      refreshLissajousTourBoundsFromJulia(juliaLambdaTour);
    }
    resyncTourPhaseFromJulia();
  }
  updateJuliaTourControlsUI();
  invalidateCache();
}

/**
 * Random walk: motion from current `dirRe`/`dirIm` (−1, 0, +1 per axis). Rerolls are handled by
 * `processRandomWalkDirectionRolls`. After each discrete dir change, `JULIA_RW_DIR_BLEND_SEC` eases heading
 * and step scale (short deceleration-style window).
 * @param {number} dtSec
 */
function tickJuliaLambdaTourRandomWalk(t, dtSec) {
  const dPhase = juliaLambdaTourDPhase();
  const baseStep = dPhase * JULIA_C_LIM * 2;
  const dt = Math.max(0, dtSec);

  let ux = 0;
  let uy = 0;
  let stepScale = 1;
  let blending = (t.rwDirBlendRemainSec ?? 0) > 0;

  if (blending) {
    t.rwDirBlendRemainSec = Math.max(0, (t.rwDirBlendRemainSec ?? 0) - dt);
    const elapsed = JULIA_RW_DIR_BLEND_SEC - t.rwDirBlendRemainSec;
    const u = Math.min(1, elapsed / JULIA_RW_DIR_BLEND_SEC);
    const e = u * u * (3 - 2 * u);

    const mode = t.rwBlendMode;
    if (mode === "fromIdle") {
      ux = t.rwBlendToUx ?? 0;
      uy = t.rwBlendToUy ?? 0;
      stepScale = e;
    } else if (mode === "toIdle") {
      ux = t.rwBlendFromUx ?? 0;
      uy = t.rwBlendFromUy ?? 0;
      stepScale = 1 - e;
    } else {
      const fx = t.rwBlendFromUx ?? 0;
      const fy = t.rwBlendFromUy ?? 0;
      const tx = t.rwBlendToUx ?? 0;
      const ty = t.rwBlendToUy ?? 0;
      let bx = fx + (tx - fx) * e;
      let by = fy + (ty - fy) * e;
      const blen = Math.hypot(bx, by);
      if (blen > 1e-15) {
        ux = bx / blen;
        uy = by / blen;
      } else {
        ux = tx;
        uy = ty;
      }
      stepScale = e * (2 - e);
    }

    if (t.rwDirBlendRemainSec <= 0) {
      clearRandomWalkDirBlend(t);
      blending = false;
    }
  }

  if (!blending) {
    const sx = t.dirRe ?? 0;
    const sy = t.dirIm ?? 0;
    const len = Math.hypot(sx, sy);
    if (len < 1e-15) {
      juliaTourDirRe = sx;
      juliaTourDirIm = sy;
      updateJuliaTourControlsUI();
      return;
    }
    ux = sx / len;
    uy = sy / len;
    stepScale = 1;
  }

  const step = baseStep * stepScale;
  const dre0 = t.dirRe ?? 0;
  const dim0 = t.dirIm ?? 0;
  let re = julia.re + ux * step;
  let im = julia.im + uy * step;
  const lim = JULIA_C_LIM;
  if (re > lim) {
    re = lim;
    if ((t.dirRe ?? 0) > 0) t.dirRe = -1;
  } else if (re < -lim) {
    re = -lim;
    if ((t.dirRe ?? 0) < 0) t.dirRe = 1;
  }
  if (im > lim) {
    im = lim;
    if ((t.dirIm ?? 0) > 0) t.dirIm = -1;
  } else if (im < -lim) {
    im = -lim;
    if ((t.dirIm ?? 0) < 0) t.dirIm = 1;
  }
  if ((t.dirRe !== dre0 || t.dirIm !== dim0)) {
    startRandomWalkDirBlend(t, dre0, dim0);
  }
  juliaTourDirRe = t.dirRe ?? 0;
  juliaTourDirIm = t.dirIm ?? 0;
  julia.re = clampJuliaComponent(re);
  julia.im = clampJuliaComponent(im);
  juliaReLabel.textContent = julia.re.toFixed(6);
  juliaImLabel.textContent = julia.im.toFixed(6);
  updateJuliaLambdaWheelVisual();
  updateJuliaTourControlsUI();
  invalidateCache();
}

/** Advance tour: Lissajous or random-walk step. */
function tickJuliaLambdaTour(dtSec) {
  if (!juliaLambdaTour || fractalKind !== 1) return;
  const t = juliaLambdaTour;
  if (t.paused) return;
  const mode = t.pathMode ?? "lissajous";
  if (mode === "randomWalk") {
    tickJuliaLambdaTourRandomWalk(t, dtSec);
    return;
  }
  const dPhase = juliaLambdaTourDPhase();
  t.phase += dPhase;
  const { re, im } = tourLambdaAtPhase(t.phase, t);
  julia.re = re;
  julia.im = im;
  juliaReLabel.textContent = julia.re.toFixed(6);
  juliaImLabel.textContent = julia.im.toFixed(6);
  updateJuliaLambdaWheelVisual();
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
  updateJuliaLambdaWheelVisual();
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
  if (fractalKind === 1) {
    julia.re = clampJuliaComponent(DEFAULT_JULIA_RE);
    julia.im = clampJuliaComponent(DEFAULT_JULIA_IM);
    syncJuliaHudFromJuliaState();
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

  tickJuliaJiggleSnap(dtSec);

  /* While jiggle snap is active, λ does not advance along the tour path and random-walk rerolls wait. */
  const pathAdvances = juliaLambdaTour && fractalKind === 1 && !juliaJiggle;
  const rwTour =
    pathAdvances &&
    !juliaLambdaTour.paused &&
    (juliaLambdaTour.pathMode ?? "lissajous") === "randomWalk";
  if (rwTour) {
    processRandomWalkDirectionRolls(juliaLambdaTour, dtSec);
  }

  if (pathAdvances) {
    tickJuliaLambdaTour(dtSec);
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
    clearJiggleSnapState();
    let resumeTourAfter = false;
    if (juliaLambdaTour) {
      resumeTourAfter = !juliaLambdaTour.paused;
      juliaLambdaTour.paused = true;
      updateJuliaTourControlsUI();
    }
    juliaLambdaDrag = {
      pointerId: e.pointerId,
      lastSx: sx,
      lastSy: sy,
      resumeTourAfter,
      lastMoveT: 0,
      velRe: 0,
      velIm: 0,
      lastDRe: 0,
      lastDIm: 0,
    };
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
    const dRe = dx * step;
    const dIm = dy * step;
    const now = performance.now();
    if (juliaLambdaDrag.lastMoveT > 0) {
      const dtSec = Math.max(1e-4, (now - juliaLambdaDrag.lastMoveT) / 1000);
      juliaLambdaDrag.velRe = dRe / dtSec;
      juliaLambdaDrag.velIm = dIm / dtSec;
    }
    juliaLambdaDrag.lastMoveT = now;
    juliaLambdaDrag.lastDRe = dRe;
    juliaLambdaDrag.lastDIm = dIm;
    julia.re = clampJuliaComponent(julia.re + dRe);
    julia.im = clampJuliaComponent(julia.im + dIm);
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
    endJuliaLambdaDragFromPointer(resumeTour);
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
  rubberPointerId = null;
  if (hadJuliaLambdaDrag) {
    endJuliaLambdaDragFromPointer(!!resumeTour);
  }
  clearCanvasPointerInteraction();
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
  juliaLambdaWheelDrag = null;
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

juliaLambdaWheel?.addEventListener("pointerdown", (e) => {
  if (fractalKind !== 1) return;
  if (e.button != null && e.button !== 0) return;
  e.preventDefault();
  stopJuliaLambdaTour();
  juliaLambdaWheelDrag = {
    lastMoveT: 0,
    velRe: 0,
    velIm: 0,
    lastDRe: 0,
    lastDIm: 0,
  };
  juliaLambdaWheel.setPointerCapture(e.pointerId);
  applyJuliaLambdaFromWheelClient(e.clientX, e.clientY);
});

juliaLambdaWheel?.addEventListener("pointermove", (e) => {
  if (!juliaLambdaWheel?.hasPointerCapture(e.pointerId)) return;
  e.preventDefault();
  applyJuliaLambdaFromWheelClient(e.clientX, e.clientY, { trackVelocity: true });
});

function releaseJuliaLambdaWheelPointer(e) {
  if (!juliaLambdaWheel?.hasPointerCapture(e.pointerId)) return;
  juliaLambdaWheel.releasePointerCapture(e.pointerId);
  endJuliaLambdaWheelDragFromPointer();
}

juliaLambdaWheel?.addEventListener("pointerup", releaseJuliaLambdaWheelPointer);
juliaLambdaWheel?.addEventListener("pointercancel", releaseJuliaLambdaWheelPointer);

juliaModeBoxBtn?.addEventListener("click", () => setJuliaCanvasMode("box"));
juliaModeLambdaBtn?.addEventListener("click", () => setJuliaCanvasMode("lambda"));

juliaPathRandomWalkBtn?.addEventListener("click", onJuliaPathRandomWalkClick);

juliaTourPlayBtn?.addEventListener("click", () => {
  playJuliaLambdaTourFromUi();
});

juliaTourPauseBtn?.addEventListener("click", () => {
  pauseJuliaLambdaTourFromUi();
});

juliaTourDirPad?.addEventListener("click", (e) => {
  const btn = e.target?.closest?.(".juliaTourDirCell");
  if (!btn || !juliaTourDirPad?.contains(btn)) return;
  const re = Number(btn.getAttribute("data-dre"));
  const im = Number(btn.getAttribute("data-dim"));
  if (!Number.isFinite(re) || !Number.isFinite(im)) return;
  applyJuliaTourDirFromPadClick(re, im);
});

juliaTourSpeedBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const v = Number(btn.getAttribute("data-speed"));
    setJuliaTourSpeedMult(v);
  });
});

juliaJiggleSnapBtn?.addEventListener("click", () => {
  if (juliaJiggleSnapEnabled) {
    juliaJiggleSnapEnabled = false;
    if (juliaJiggle) {
      julia.re = clampJuliaComponent(juliaJiggle.targetRe);
      julia.im = clampJuliaComponent(juliaJiggle.targetIm);
      juliaJiggle = null;
      syncJuliaHudFromJuliaState();
      if (pendingTourResumeAfterJiggle) {
        pendingTourResumeAfterJiggle = false;
        resumeJuliaLambdaTourAfterLambdaDrag();
      }
      invalidateCache();
    }
  } else {
    juliaJiggleSnapEnabled = true;
  }
  updateJuliaJiggleSnapToggleUI();
});

function wireJigglePhysicsSliders() {
  const onInput = () => {
    syncJigglePhysicsFromInputs();
    invalidateCache();
  };
  juliaJiggleSpringInput?.addEventListener("input", onInput);
  juliaJiggleDampingInput?.addEventListener("input", onInput);
  juliaJiggleCarryInput?.addEventListener("input", onInput);
  juliaJiggleCapInput?.addEventListener("input", onInput);
}

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
  updateJuliaPathModeButtonsUI();
  updateJuliaTourControlsUI();
  wireJigglePhysicsSliders();
  syncJigglePhysicsFromInputs();
  updateJuliaJiggleSnapToggleUI();

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
