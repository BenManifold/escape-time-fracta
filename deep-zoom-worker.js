import init, { alloc, dealloc, render_rgba } from "./fractal-wasm/pkg/fractal_wasm.js";

let wasmMemory;
let ready = false;

async function ensureReady() {
  if (ready) return;
  const wasm = await init();
  wasmMemory = wasm.memory;
  ready = true;
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg?.type !== "render") return;

  try {
    await ensureReady();
  } catch (e) {
    self.postMessage({ type: "error", gen: msg.gen, segmentIdx: msg.segmentIdx, message: String(e) });
    return;
  }

  const {
    gen,
    segmentIdx,
    cw,
    ch,
    centerX,
    centerY,
    halfW,
    maxIter,
    fractalKind,
    juliaRe,
    juliaIm,
    paletteId: pid = 0,
  } = msg;

  const aspect = ch / cw;
  const byteLen = cw * ch * 4;
  const ptr = alloc(byteLen);
  try {
    render_rgba(
      ptr,
      byteLen,
      cw,
      ch,
      centerX,
      centerY,
      halfW,
      aspect,
      maxIter,
      fractalKind,
      juliaRe,
      juliaIm,
      pid >>> 0
    );
    const src = new Uint8ClampedArray(wasmMemory.buffer, ptr, byteLen);
    const copy = new Uint8Array(byteLen);
    copy.set(src);
    const buffer = copy.buffer;
    self.postMessage(
      { type: "done", gen, segmentIdx, cw, ch, centerX, centerY, halfW, buffer },
      [buffer]
    );
  } finally {
    dealloc(ptr, byteLen);
  }
};
