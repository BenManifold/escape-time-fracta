import init, { alloc, dealloc, render_rgba } from "./fractal-wasm/pkg/fractal_wasm.js";

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

const MOTION_SETTLE_MS = 140;
const LOD_SCALE = 2;
const MOTION_MAX_ITER_CAP = 96;
const PAN_SPEED = 0.12;

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
let lastInputAt = 0;
let dragging = false;
let dragLastX = 0;
let dragLastY = 0;
const keys = new Set();

function markInput() {
  lastInputAt = performance.now();
}

function isInMotion(now) {
  return now - lastInputAt < MOTION_SETTLE_MS || dragging || keys.size > 0;
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

function resizeCanvas() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.floor(window.innerWidth * dpr);
  const h = Math.floor(window.innerHeight * dpr);
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  freeBuffer();
  markInput();
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
 * @param {number} sx
 * @param {number} sy
 * @param {number} deltaY
 */
function zoomAtScreen(sx, sy, deltaY) {
  const factor = Math.exp(-deltaY * 0.0018);
  const newHalfW = Math.min(1e6, Math.max(1e-16, view.halfW * factor));
  const { re: cre, im: cim } = screenToComplex(sx, sy);
  const w = canvas.width;
  const h = canvas.height;
  const aspect = h / w;
  const newHalfH = newHalfW * aspect;
  view.centerX = cre + newHalfW - (sx / w) * (2 * newHalfW);
  view.centerY = cim + newHalfH - (sy / h) * (2 * newHalfH);
  view.halfW = newHalfW;
  markInput();
}

function panPixels(dx, dy, fine) {
  const w = canvas.width;
  const h = canvas.height;
  const aspect = h / w;
  const halfH = view.halfW * aspect;
  const scale = fine ? 0.25 : 1;
  const nx = (dx / w) * (2 * view.halfW) * PAN_SPEED * scale;
  const ny = (dy / h) * (2 * halfH) * PAN_SPEED * scale;
  view.centerX -= nx;
  view.centerY -= ny;
  markInput();
}

function applyKeyMovement(now) {
  if (keys.size === 0) return;
  const fine = keys.has("shift");
  let dx = 0;
  let dy = 0;
  if (keys.has("arrowleft") || keys.has("a")) dx += 6;
  if (keys.has("arrowright") || keys.has("d")) dx -= 6;
  if (keys.has("arrowup") || keys.has("w")) dy += 6;
  if (keys.has("arrowdown") || keys.has("s")) dy -= 6;
  if (dx !== 0 || dy !== 0) panPixels(dx * 4, dy * 4, fine);
  if (keys.has("+") || keys.has("=")) {
    zoomAtScreen(canvas.width / 2, canvas.height / 2, -120 * (fine ? 0.35 : 1));
  }
  if (keys.has("-") || keys.has("_")) {
    zoomAtScreen(canvas.width / 2, canvas.height / 2, 120 * (fine ? 0.35 : 1));
  }
}

function frame(now) {
  resizeCanvas();
  applyKeyMovement(now);

  const cw = canvas.width;
  const ch = canvas.height;
  const moving = isInMotion(now);
  const scale = moving ? LOD_SCALE : 1;
  const bufW = Math.max(1, Math.ceil(cw / scale));
  const bufH = Math.max(1, Math.ceil(ch / scale));
  const byteLen = bufW * bufH * 4;
  ensureBuffer(byteLen);

  const aspect = ch / cw;
  let maxIter = maxIterUser;
  if (moving) {
    maxIter = Math.min(maxIter, MOTION_MAX_ITER_CAP);
  }

  const t0 = performance.now();
  render_rgba(
    pixelPtr,
    byteLen,
    bufW,
    bufH,
    view.centerX,
    view.centerY,
    view.halfW,
    aspect,
    maxIter,
    fractalKind,
    julia.re,
    julia.im
  );
  const t1 = performance.now();

  const src = new Uint8ClampedArray(wasmMemory.buffer, pixelPtr, byteLen);
  const imageData = new ImageData(bufW, bufH);
  imageData.data.set(src);

  ctx.imageSmoothingEnabled = scale > 1;
  ctx.imageSmoothingQuality = "low";
  bitmapToCanvas(imageData, bufW, bufH, cw, ch);

  const ms = (t1 - t0).toFixed(1);
  const mode = moving ? `preview · ${maxIter} it` : `${maxIter} it`;
  statusEl.textContent = `${bufW}×${bufH} · ${mode} · ${ms} ms`;

  requestAnimationFrame(frame);
}

/**
 * @param {ImageData} imageData
 * @param {number} bufW
 * @param {number} bufH
 * @param {number} cw
 * @param {number} ch
 */
function bitmapToCanvas(imageData, bufW, bufH, cw, ch) {
  if (bufW === cw && bufH === ch) {
    ctx.putImageData(imageData, 0, 0);
    return;
  }
  if (!bitmapToCanvas._scratch || bitmapToCanvas._scratch.width !== bufW || bitmapToCanvas._scratch.height !== bufH) {
    const c = document.createElement("canvas");
    c.width = bufW;
    c.height = bufH;
    bitmapToCanvas._scratch = c;
  }
  const oc = bitmapToCanvas._scratch;
  const octx = oc.getContext("2d", { alpha: false });
  octx.putImageData(imageData, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(oc, 0, 0, bufW, bufH, 0, 0, cw, ch);
}

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width;
    const sx = (e.clientX - rect.left) * dpr;
    const sy = (e.clientY - rect.top) * dpr;
    zoomAtScreen(sx, sy, e.deltaY);
  },
  { passive: false }
);

canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  dragging = true;
  dragLastX = e.clientX;
  dragLastY = e.clientY;
  markInput();
});

window.addEventListener("mouseup", () => {
  dragging = false;
});

window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  const dx = e.clientX - dragLastX;
  const dy = e.clientY - dragLastY;
  dragLastX = e.clientX;
  dragLastY = e.clientY;
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  panPixels(dx * scaleX, dy * scaleY, e.shiftKey);
});

window.addEventListener(
  "keydown",
  (e) => {
    const k = e.key.toLowerCase();
    if (
      ["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d", "+", "=", "-", "_", "shift"].includes(
        k
      ) ||
      e.key === "+"
    ) {
      keys.add(k);
      if (e.key === "Shift") keys.add("shift");
      markInput();
    }
  },
  { passive: true }
);

window.addEventListener(
  "keyup",
  (e) => {
    const k = e.key.toLowerCase();
    keys.delete(k);
    keys.delete(e.key);
    if (e.key === "Shift") keys.delete("shift");
  },
  { passive: true }
);

fractalSelect.addEventListener("change", () => {
  fractalKind = Number(fractalSelect.value);
  updateJuliaPanelVisibility();
  markInput();
});

juliaReInput.addEventListener("input", () => {
  syncJuliaLabels();
  markInput();
});

juliaImInput.addEventListener("input", () => {
  syncJuliaLabels();
  markInput();
});

maxIterInput.addEventListener("input", () => {
  maxIterUser = Number(maxIterInput.value);
  syncIterLabel();
  markInput();
});

window.addEventListener("resize", markInput);

async function main() {
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
