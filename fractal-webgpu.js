/**
 * Single WebGPU path: escape-time fractals (f32), LUT from WASM, optional affine warp for deep zoom.
 */

const LUT_W = 4096;

/**
 * Template literals copy CRLF verbatim on Windows; WGSL rejects stray \\r (Tint: "Parsing error").
 */
function wgslSource(src) {
  return src.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

const SHADER = /* wgsl */ `
const BAILOUT: f32 = 4.0;
const BG_R: f32 = 12.0 / 255.0;
const BG_G: f32 = 12.0 / 255.0;
const BG_B: f32 = 15.0 / 255.0;
const VEIL_GRAY: f32 = 0.35;

struct Params {
  view: vec4<f32>,
  iter_pal_wh: vec4<u32>,
  kind_j: vec4<f32>,
  tm: vec4<f32>
}

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var palTex: texture_2d<f32>;
@group(0) @binding(2) var palSam: sampler;
@group(0) @binding(3) var outImg: texture_storage_2d<rgba8unorm, write>;

fn interior_nebula(zr: f32, zi: f32) -> vec3<f32> {
  let ang = atan2(zi, zr);
  let w = 0.5 + 0.5 * sin(ang);
  let r2 = zr * zr + zi * zi;
  let rho = clamp(sqrt(r2), 0.0, 2.0) * 0.5;
  var r = 0.052 + 0.035 * w + 0.018 * rho;
  var g = 0.038 + 0.022 * (1.0 - w) + 0.014 * rho;
  var b = 0.11 + 0.05 * w + 0.028 * rho;
  let VEIL_IN = 0.35;
  r = r * (1.0 - VEIL_IN) + BG_R * VEIL_IN;
  g = g * (1.0 - VEIL_IN) + BG_G * VEIL_IN;
  b = b * (1.0 - VEIL_IN) + BG_B * VEIL_IN;
  return vec3<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

fn interior_flotilla(zr: f32, zi: f32) -> vec3<f32> {
  let ang = atan2(zi, zr);
  let w = 0.5 + 0.5 * sin(ang);
  let r2 = zr * zr + zi * zi;
  let rho = clamp(sqrt(r2), 0.0, 2.0) * 0.5;
  var r = 0.045 + 0.04 * w + 0.04 * rho;
  var g = 0.075 + 0.055 * (1.0 - w) + 0.05 * rho;
  var b = 0.11 + 0.07 * w + 0.055 * rho;
  r += 0.025 * rho * max(cos(ang * 2.1), 0.0);
  g += 0.02 * rho * max(sin(ang * 2.1), 0.0);
  let VEIL_IN = 0.26;
  r = r * (1.0 - VEIL_IN) + BG_R * VEIL_IN;
  g = g * (1.0 - VEIL_IN) + BG_G * VEIL_IN;
  b = b * (1.0 - VEIL_IN) + BG_B * VEIL_IN;
  return vec3<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

fn interior_ghost_ship(zr: f32, zi: f32) -> vec3<f32> {
  let ang = atan2(zi, zr);
  let w = 0.5 + 0.5 * sin(ang);
  let r2 = zr * zr + zi * zi;
  let rho = clamp(sqrt(r2), 0.0, 2.0) * 0.5;
  var r = 0.038 + 0.05 * w + 0.065 * rho;
  var g = 0.072 + 0.055 * (1.0 - w) + 0.085 * rho;
  var b = 0.11 + 0.08 * w + 0.055 * rho;
  g += 0.045 * rho * max(sin(ang * 2.05), 0.0);
  b += 0.055 * rho * max(cos(ang * 2.05), 0.0);
  r += 0.028 * rho * max(cos(ang * 1.65), 0.0);
  let VEIL_IN = 0.24;
  r = r * (1.0 - VEIL_IN) + BG_R * VEIL_IN;
  g = g * (1.0 - VEIL_IN) + BG_G * VEIL_IN;
  b = b * (1.0 - VEIL_IN) + BG_B * VEIL_IN;
  return vec3<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

fn interior_gray(zr: f32, zi: f32) -> vec3<f32> {
  let ang = atan2(zi, zr);
  let v = 0.04 + 0.06 * (0.5 + 0.5 * sin(ang));
  let x = v * (1.0 - VEIL_GRAY) + 0.05 * VEIL_GRAY;
  return vec3<f32>(x, x, x);
}

fn interior_rgb(zr: f32, zi: f32, pid: u32) -> vec3<f32> {
  if (pid == 1u) { return interior_flotilla(zr, zi); }
  if (pid == 2u) { return vec3<f32>(0.0, 0.0, 0.0); }
  if (pid == 3u) { return interior_gray(zr, zi); }
  if (pid == 4u) { return interior_ghost_ship(zr, zi); }
  return interior_nebula(zr, zi);
}

fn step_escape(zr: f32, zi: f32, cr: f32, ci: f32, fk: u32) -> vec2<f32> {
  if (fk == 2u) {
    let azr = abs(zr);
    let azi = abs(zi);
    return vec2<f32>(azr * azr - azi * azi + cr, 2.0 * azr * azi + ci);
  }
  if (fk == 3u) {
    let zrc = -zi;
    return vec2<f32>(zr * zr - zrc * zrc + cr, 2.0 * zr * zrc + ci);
  }
  return vec2<f32>(zr * zr - zi * zi + cr, 2.0 * zr * zi + ci);
}

@compute @workgroup_size(8, 8)
fn escape_cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.iter_pal_wh.z;
  let h = u.iter_pal_wh.w;
  if (gid.x >= w || gid.y >= h) { return; }

  let cx = u.view.x;
  let cy = u.view.y;
  let half_w = u.view.z;
  let aspect = u.view.w;
  let half_h = half_w * aspect;
  let max_iter = u.iter_pal_wh.x;
  let palette_id = u.iter_pal_wh.y;
  let t_max = max(u.tm.x, 1.0);
  let fk = u32(u.kind_j.x + 0.5);
  let jre = u.kind_j.y;
  let jim = u.kind_j.z;

  let px = f32(gid.x) + 0.5;
  let py = f32(gid.y) + 0.5;
  let wf = f32(w);
  let hf = f32(h);
  let re = cx - half_w + (px / wf) * (2.0 * half_w);
  let im = cy - half_h + (py / hf) * (2.0 * half_h);

  var zr: f32;
  var zi: f32;
  var cr: f32;
  var ci: f32;
  if (fk == 1u) {
    zr = re;
    zi = im;
    cr = jre;
    ci = jim;
  } else {
    zr = 0.0;
    zi = 0.0;
    cr = re;
    ci = im;
  }

  var n = 0u;
  loop {
    if (n >= max_iter) { break; }
    let r2 = zr * zr + zi * zi;
    if (r2 >= BAILOUT) {
      let zmag = max(sqrt(r2), 4.0000001);
      let smooth_iter = f32(n) + 1.0 - log2(log2(zmag));
      let tu = clamp(smooth_iter / t_max, 0.0, 1.0);
      let c = textureSampleLevel(palTex, palSam, vec2<f32>(tu, 0.5), 0.0);
      textureStore(outImg, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(c.r, c.g, c.b, 1.0));
      return;
    }
    let nz = step_escape(zr, zi, cr, ci, fk);
    zr = nz.x;
    zi = nz.y;
    n += 1u;
  }

  let rgb = interior_rgb(zr, zi, palette_id);
  textureStore(outImg, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb.x, rgb.y, rgb.z, 1.0));
}

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var o: VsOut;
  let p = pos[vi];
  o.pos = vec4<f32>(p.x, p.y, 0.0, 1.0);
  o.uv = uv[vi];
  return o;
}

@group(1) @binding(0) var blitSrc: texture_2d<f32>;
@group(1) @binding(1) var blitSam: sampler;

@fragment
fn fs_blit(@location(0) _uv: vec2<f32>, @builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  // Framebuffer coords: origin top-left, y down (matches canvas + screenToComplex). Avoids NDC/triangle UV mismatch.
  // Do not take VsOut here: it already carries @builtin(position); duplicating pos is invalid WGSL.
  let td = textureDimensions(blitSrc);
  let dims = vec2<f32>(f32(td.x), f32(td.y));
  let uv = pos.xy / dims;
  return textureSample(blitSrc, blitSam, uv);
}

struct WarpU {
  ku_kv_s: vec4<f32>,
  wh: vec2<f32>
}

@group(2) @binding(0) var<uniform> warp: WarpU;
@group(2) @binding(1) var warpSrc: texture_2d<f32>;
@group(2) @binding(2) var warpSam: sampler;

@fragment
fn fs_warp(@location(0) _uv: vec2<f32>, @builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let cw = warp.wh.x;
  let ch = warp.wh.y;
  let dx = pos.x;
  let dy = pos.y;
  // Ku/Kv derived with inverted vertical pixel coord (top → ch). pos.y is 0 at top, increases down.
  let dy_warp = ch - dy;
  let s = warp.ku_kv_s.z;
  let src_x = dx / s + warp.ku_kv_s.x;
  let src_y = dy_warp / s + warp.ku_kv_s.y;
  let su = src_x / cw;
  let sv = src_y / ch;
  if (su < 0.0 || su > 1.0 || sv < 0.0 || sv > 1.0) {
    return vec4<f32>(BG_R, BG_G, BG_B, 1.0);
  }
  return textureSampleLevel(warpSrc, warpSam, vec2<f32>(su, sv), 0.0);
}
`;

/**
 * @param {HTMLCanvasElement} canvas
 * @param {(len: number) => number} alloc
 * @param {(ptr: number, len: number) => void} dealloc
 * @param {(ptr: number, len: number, n: number, paletteId: number, tMax: number) => void} fillLut
 * @param {WebAssembly.Memory} wasmMemory
 */
export async function createFractalGpuRenderer(canvas, alloc, dealloc, fillLut, wasmMemory) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const device = await adapter.requestDevice();

  const ctx = canvas.getContext("webgpu");
  if (!ctx) throw new Error("No WebGPU canvas context");

  function configureSwapChain() {
    const nw = Math.max(1, canvas.width | 0);
    const nh = Math.max(1, canvas.height | 0);
    const base = {
      device,
      format,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    };
    try {
      ctx.configure({ ...base, width: nw, height: nh });
    } catch {
      ctx.configure(base);
    }
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  const wgsl = wgslSource(SHADER);
  const module = device.createShaderModule({ code: wgsl });

  const bindComputeLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: "write-only", format: "rgba8unorm" },
      },
    ],
  });

  const bindBlitLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
    ],
  });

  const bindWarpLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
    ],
  });

  const layoutCompute = device.createPipelineLayout({
    bindGroupLayouts: [bindComputeLayout],
  });

  const emptyBindGroupLayout = device.createBindGroupLayout({ entries: [] });
  const emptyBindGroup = device.createBindGroup({
    layout: emptyBindGroupLayout,
    entries: [],
  });

  const layoutBlit = device.createPipelineLayout({
    bindGroupLayouts: [emptyBindGroupLayout, bindBlitLayout],
  });

  const layoutWarp = device.createPipelineLayout({
    bindGroupLayouts: [emptyBindGroupLayout, emptyBindGroupLayout, bindWarpLayout],
  });

  const computePipeline = device.createComputePipeline({
    layout: layoutCompute,
    compute: { module, entryPoint: "escape_cs" },
  });

  const blitPipeline = device.createRenderPipeline({
    layout: layoutBlit,
    vertex: { module, entryPoint: "vs_fullscreen" },
    fragment: {
      module,
      entryPoint: "fs_blit",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const warpPipeline = device.createRenderPipeline({
    layout: layoutWarp,
    vertex: { module, entryPoint: "vs_fullscreen" },
    fragment: {
      module,
      entryPoint: "fs_warp",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  /** WebGPU requires uniform bindings to use at least 256 bytes of a buffer. */
  const UNIFORM_ALIGN = 256;

  const uniformBuffer = device.createBuffer({
    size: UNIFORM_ALIGN,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const warpUniformBuffer = device.createBuffer({
    size: UNIFORM_ALIGN,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const palSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  const blitSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  let w = 0;
  let h = 0;
  let lastPaletteId = 0;
  let lastMaxIter = 1024;
  /** @type {GPUTexture | null} */
  let workTex = null;
  /** @type {GPUTexture | null} */
  let commitTex = null;
  let commitW = 0;
  let commitH = 0;
  /** @type {GPUBindGroup | null} */
  let bindCompute = null;
  /** @type {GPUBindGroup | null} */
  let bindBlit = null;
  /** @type {GPUBindGroup | null} */
  let bindWarp = null;
  /** @type {GPUTexture | null} */
  let paletteTex = null;
  let lutKey = "";

  function destroyWork() {
    workTex?.destroy();
    workTex = null;
    bindCompute = null;
    bindBlit = null;
  }

  function destroyCommit() {
    commitTex?.destroy();
    commitTex = null;
    commitW = 0;
    commitH = 0;
    bindWarp = null;
  }

  function destroyPalette() {
    paletteTex?.destroy();
    paletteTex = null;
    lutKey = "";
  }

  function refreshComputeBindGroup() {
    if (!workTex || !paletteTex) return;
    bindCompute = device.createBindGroup({
      layout: bindComputeLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: paletteTex.createView() },
        { binding: 2, resource: palSampler },
        { binding: 3, resource: workTex.createView() },
      ],
    });
  }

  function refreshWarpBindGroup() {
    if (!commitTex) return;
    bindWarp = device.createBindGroup({
      layout: bindWarpLayout,
      entries: [
        { binding: 0, resource: { buffer: warpUniformBuffer } },
        { binding: 1, resource: commitTex.createView() },
        { binding: 2, resource: blitSampler },
      ],
    });
  }

  function ensureWork(nw, nh) {
    if (nw === w && nh === h && workTex) return;
    destroyWork();
    w = nw;
    h = nh;
    if (w === 0 || h === 0 || !paletteTex) return;

    workTex = device.createTexture({
      size: [w, h],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });

    refreshComputeBindGroup();

    bindBlit = device.createBindGroup({
      layout: bindBlitLayout,
      entries: [
        { binding: 0, resource: workTex.createView() },
        { binding: 1, resource: blitSampler },
      ],
    });
  }

  function ensureCommit(nw, nh) {
    if (commitTex && commitW === nw && commitH === nh) return;
    destroyCommit();
    if (nw < 1 || nh < 1) return;
    commitW = nw;
    commitH = nh;
    commitTex = device.createTexture({
      size: [nw, nh],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    refreshWarpBindGroup();
  }

  function ensurePalette(paletteId, maxIter) {
    const tMax = Math.max(maxIter * 1.5, 64);
    const key = `${paletteId}|${maxIter}`;
    if (paletteTex && lutKey === key) return;

    destroyPalette();
    lutKey = key;

    const need = LUT_W * 4;
    const ptr = alloc(need);
    try {
      fillLut(ptr, need, LUT_W, paletteId >>> 0, tMax);
      const data = new Uint8Array(wasmMemory.buffer, ptr, need);

      paletteTex = device.createTexture({
        size: [LUT_W, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: paletteTex },
        data,
        { bytesPerRow: LUT_W * 4 },
        { width: LUT_W, height: 1 },
      );
      refreshComputeBindGroup();
    } finally {
      dealloc(ptr, need);
    }
  }

  function writeParams(p) {
    const nw = canvas.width;
    const nh = canvas.height;
    const tMax = Math.max((p.maxIter >>> 0) * 1.5, 64);
    const ubo = new ArrayBuffer(64);
    new Float32Array(ubo, 0, 4).set([p.centerX, p.centerY, p.halfW, p.aspect]);
    new Uint32Array(ubo, 16, 4).set([p.maxIter >>> 0, p.paletteId >>> 0, nw, nh]);
    new Float32Array(ubo, 32, 4).set([p.fractalKind >>> 0, p.juliaRe, p.juliaIm, 0]);
    new Float32Array(ubo, 48, 4).set([tMax, 0, 0, 0]);
    device.queue.writeBuffer(uniformBuffer, 0, ubo);
  }

  /**
   * Full-quality frame: compute → swapchain.
   * @param {object} p
   * @param {{ present?: boolean }} [opts] default present true
   */
  function drawFull(p, opts = {}) {
    const present = opts.present !== false;
    const nw = canvas.width;
    const nh = canvas.height;
    if (nw < 1 || nh < 1) return 0;

    lastPaletteId = p.paletteId >>> 0;
    lastMaxIter = p.maxIter >>> 0;

    ensurePalette(lastPaletteId, lastMaxIter);
    ensureWork(nw, nh);
    if (!bindCompute || !workTex || !paletteTex) return 0;

    configureSwapChain();
    writeParams(p);

    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, bindCompute);
      pass.dispatchWorkgroups(Math.ceil(nw / 8), Math.ceil(nh / 8));
      pass.end();
    }
    if (present && bindBlit) {
      const view = ctx.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(blitPipeline);
      pass.setBindGroup(0, emptyBindGroup);
      pass.setBindGroup(1, bindBlit);
      pass.draw(3);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    return performance.now() - t0;
  }

  /** Copy last compute result to commit texture (deep-zoom keyframe). */
  function copyWorkToCommit() {
    if (!workTex || !commitTex) return;
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture(
      { texture: workTex },
      { texture: commitTex },
      [canvas.width, canvas.height],
    );
    device.queue.submit([encoder.finish()]);
  }

  /**
   * After drawFull, refresh commit from work (same view).
   */
  function captureCommit() {
    ensureCommit(canvas.width, canvas.height);
    copyWorkToCommit();
  }

  /**
   * Affine warp of commit texture (matches former 2D drawWarpedCache).
   * @param {{ centerX: number; centerY: number; halfW: number }} committed
   * @param {{ centerX: number; centerY: number; halfW: number }} current
   */
  function drawWarp(committed, current) {
    const cw = canvas.width;
    const ch = canvas.height;
    if (!commitTex || cw < 1 || ch < 1 || !bindWarp) return 0;

    const cx0 = committed.centerX;
    const cy0 = committed.centerY;
    const hw0 = committed.halfW;
    const cx1 = current.centerX;
    const cy1 = current.centerY;
    const hw1 = current.halfW;
    const aspect = ch / cw;
    const hh0 = hw0 * aspect;
    const hh1 = hw1 * aspect;

    const Ku = (cx1 - hw1 - cx0 + hw0) * (cw / (2 * hw0));
    const Kv = (cy1 - hh1 - cy0 + hh0) * (ch / (2 * hh0));
    const s = hw0 / hw1;

    const wubo = new ArrayBuffer(32);
    new Float32Array(wubo, 0, 4).set([Ku, Kv, s, 0]);
    new Float32Array(wubo, 16, 2).set([cw, ch]);
    device.queue.writeBuffer(warpUniformBuffer, 0, wubo);

    configureSwapChain();

    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    const view = ctx.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(warpPipeline);
    pass.setBindGroup(0, emptyBindGroup);
    pass.setBindGroup(1, emptyBindGroup);
    pass.setBindGroup(2, bindWarp);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return performance.now() - t0;
  }

  /**
   * Segment checkpoint: render end-of-segment view into work (no present). Call copyWorkToCommit on apply.
   */
  function drawCheckpoint(p) {
    return drawFull(p, { present: false });
  }

  function presentWork() {
    if (!bindBlit || !workTex) return;
    configureSwapChain();
    const encoder = device.createCommandEncoder();
    const view = ctx.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(blitPipeline);
    pass.setBindGroup(0, emptyBindGroup);
    pass.setBindGroup(1, bindBlit);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function resize(nw, nh) {
    if (nw === w && nh === h && workTex) return;
    destroyWork();
    destroyCommit();
    w = nw;
    h = nh;
    commitW = 0;
    commitH = 0;
    if (w > 0 && h > 0 && paletteTex) {
      workTex = device.createTexture({
        size: [w, h],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      refreshComputeBindGroup();
      bindBlit = device.createBindGroup({
        layout: bindBlitLayout,
        entries: [
          { binding: 0, resource: workTex.createView() },
          { binding: 1, resource: blitSampler },
        ],
      });
    }
  }

  /** Drop deep-zoom snapshot texture (~width×height×4 bytes). Call when deep zoom ends or is cancelled. */
  function releaseCommit() {
    destroyCommit();
  }

  function destroy() {
    destroyWork();
    destroyCommit();
    destroyPalette();
    uniformBuffer.destroy();
    warpUniformBuffer.destroy();
  }

  return {
    drawFull,
    drawCheckpoint,
    presentWork,
    drawWarp,
    captureCommit,
    copyWorkToCommit,
    resize,
    releaseCommit,
    destroy,
  };
}

export function webGpuSupported() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}
