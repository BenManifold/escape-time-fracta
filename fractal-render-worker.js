import init, { alloc, dealloc, render_rgba } from "./fractal-wasm/pkg/fractal_wasm.js";

/** @type {import("./fractal-wasm/pkg/fractal_wasm.js").InitOutput | null} */
let wasmOut = null;

async function ensureInit() {
  if (wasmOut) return wasmOut;
  wasmOut = await init();
  return wasmOut;
}

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m?.type !== "wasmRender") return;
  const { seq } = m;
  try {
    const out = await ensureInit();
    const {
      nw,
      nh,
      centerX,
      centerY,
      halfW,
      aspect,
      maxIter,
      fractalKind,
      juliaRe,
      juliaIm,
      paletteId,
      perturbMode,
      fp,
      committed,
    } = m;
    const need = nw * nh * 4;
    const ptr = alloc(need);
    render_rgba(
      ptr,
      need,
      nw >>> 0,
      nh >>> 0,
      centerX,
      centerY,
      halfW,
      aspect,
      maxIter >>> 0,
      fractalKind >>> 0,
      juliaRe,
      juliaIm,
      paletteId >>> 0,
      perturbMode >>> 0,
    );
    const tight = new Uint8Array(need);
    tight.set(new Uint8Array(out.memory.buffer, ptr, need));
    dealloc(ptr, need);
    const rowBytes = nw * 4;
    const bytesPerRow = (rowBytes + 255) & ~255;
    const paddedBytes = bytesPerRow * nh;
    const padded = new Uint8Array(paddedBytes);
    for (let y = 0; y < nh; y++) {
      padded.set(tight.subarray(y * rowBytes, y * rowBytes + rowBytes), y * bytesPerRow);
    }
    self.postMessage(
      {
        type: "wasmRenderDone",
        seq,
        ok: true,
        buffer: padded.buffer,
        nw,
        nh,
        bytesPerRow,
        fp,
        committed,
      },
      [padded.buffer],
    );
  } catch (err) {
    self.postMessage({
      type: "wasmRenderDone",
      seq,
      ok: false,
      error: String(err),
    });
  }
};
