/**
 * WebGPU: single compute pass — double-single `c` per pixel, double-single orbit `z` + `c` in the iteration + blit.
 */

const LUT_W = 4096;

function wgslSource(src) {
  return src.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function splitToDoubleSingle(x) {
  const hi = Math.fround(x);
  const lo = Math.fround(x - hi);
  return [hi, lo];
}

const WGSL_PARAMS_AND_DS = /* wgsl */ `
const BAILOUT: f32 = 4.0;

struct Params {
  view: vec4<f32>,
  iter_pal_wh: vec4<u32>,
  kind_j: vec4<f32>,
  tm: vec4<f32>,
  julia_lambda: vec4<f32>,
  c_base: vec4<f32>,
  c_step: vec4<f32>,
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  let bb = s - a;
  let err = (a - (s - bb)) + (b - bb);
  return vec2<f32>(s, err);
}

fn ds_add(ah: f32, al: f32, bh: f32, bl: f32) -> vec2<f32> {
  let t = two_sum(ah, bh);
  var sh = t.x;
  var sl = t.y + al + bl;
  let t2 = two_sum(sh, sl);
  return vec2<f32>(t2.x, t2.y);
}

fn ds_sub(ah: f32, al: f32, bh: f32, bl: f32) -> vec2<f32> {
  return ds_add(ah, al, -bh, -bl);
}

fn ds_mul(ah: f32, al: f32, bh: f32, bl: f32) -> vec2<f32> {
  let p1 = ah * bh;
  let p2 = fma(ah, bh, -p1);
  let p3 = ah * bl + al * bh + al * bl + p2;
  return two_sum(p1, p3);
}

// Veltkamp split: represent the scalar as double-single (hi, lo) for extended products/sums.
fn split_f32(a: f32) -> vec2<f32> {
  let t = 4097.0 * a;
  let th = t - (t - a);
  let tl = a - th;
  return vec2<f32>(th, tl);
}

fn mul_u32_ds(n: u32, dh: f32, dl: f32) -> vec2<f32> {
  let nf = f32(n);
  let hi = nf * dh;
  let lo = fma(nf, dh, -hi) + nf * dl;
  return two_sum(hi, lo);
}

fn ds_scale_s(ah: f32, al: f32, s: f32) -> vec2<f32> {
  let hi = ah * s;
  let lo = fma(ah, s, -hi) + al * s;
  return two_sum(hi, lo);
}

fn ds_neg(ah: f32, al: f32) -> vec2<f32> {
  return ds_sub(0.0, 0.0, ah, al);
}

fn ds_abs_pair(ah: f32, al: f32) -> vec2<f32> {
  let s = ah + al;
  if (s >= 0.0) {
    return vec2<f32>(ah, al);
  }
  return ds_neg(ah, al);
}

fn cplx_sqr_ds(zrh: f32, zrl: f32, zih: f32, zil: f32) -> vec4<f32> {
  let tr = ds_mul(zrh, zrl, zrh, zrl);
  let ti = ds_mul(zih, zil, zih, zil);
  let re = ds_sub(tr.x, tr.y, ti.x, ti.y);
  let zrzi = ds_mul(zrh, zrl, zih, zil);
  let im = ds_scale_s(zrzi.x, zrzi.y, 2.0);
  return vec4<f32>(re.x, re.y, im.x, im.y);
}

fn cplx_add_ds(
  arh: f32, arl: f32, aih: f32, ail: f32,
  brh: f32, brl: f32, bih: f32, bil: f32,
) -> vec4<f32> {
  let r = ds_add(arh, arl, brh, brl);
  let i = ds_add(aih, ail, bih, bil);
  return vec4<f32>(r.x, r.y, i.x, i.y);
}

/// Multibrot z^p + c: magnitude/phase in f32, then split and add c in double-single.
fn cplx_pow_f32_add_ds(
  zr: f32,
  zi: f32,
  p: f32,
  crh: f32,
  crl: f32,
  cih: f32,
  cil: f32,
) -> vec4<f32> {
  let r = sqrt(zr * zr + zi * zi);
  if (r < 1e-30) {
    return cplx_add_ds(0.0, 0.0, 0.0, 0.0, crh, crl, cih, cil);
  }
  let phi = atan2(zi, zr);
  let rp = pow(r, p);
  let nph = p * phi;
  let pr = rp * cos(nph);
  let pi = rp * sin(nph);
  let pr_s = split_f32(pr);
  let pi_s = split_f32(pi);
  return cplx_add_ds(pr_s.x, pr_s.y, pi_s.x, pi_s.y, crh, crl, cih, cil);
}

fn step_escape_ds(
  zrh: f32, zrl: f32, zih: f32, zil: f32,
  crh: f32, crl: f32, cih: f32, cil: f32,
  fk: u32,
  mandel_exp: f32,
) -> vec4<f32> {
  var ar = zrh;
  var arl = zrl;
  var ai = zih;
  var ail = zil;
  if (fk == 2u) {
    let ax = ds_abs_pair(zrh, zrl);
    let ay = ds_abs_pair(zih, zil);
    ar = ax.x;
    arl = ax.y;
    ai = ay.x;
    ail = ay.y;
    let sq = cplx_sqr_ds(ar, arl, ai, ail);
    return cplx_add_ds(sq.x, sq.y, sq.z, sq.w, crh, crl, cih, cil);
  }
  if (fk == 1u) {
    let sq = cplx_sqr_ds(ar, arl, ai, ail);
    return cplx_add_ds(sq.x, sq.y, sq.z, sq.w, crh, crl, cih, cil);
  }
  if (abs(mandel_exp - 2.0) < 0.00005) {
    let sq = cplx_sqr_ds(ar, arl, ai, ail);
    return cplx_add_ds(sq.x, sq.y, sq.z, sq.w, crh, crl, cih, cil);
  }
  let zr = ar + arl;
  let zi = ai + ail;
  return cplx_pow_f32_add_ds(zr, zi, mandel_exp, crh, crl, cih, cil);
}
`;

const WGSL_INTERIOR = /* wgsl */ `
const BG_R: f32 = 12.0 / 255.0;
const BG_G: f32 = 12.0 / 255.0;
const BG_B: f32 = 15.0 / 255.0;
const VEIL_GRAY: f32 = 0.35;

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
`;

const WGSL_ESCAPE = /* wgsl */ `
@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var palTex: texture_2d<f32>;
@group(0) @binding(2) var palSam: sampler;
@group(0) @binding(3) var outImg: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn escape_cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u.iter_pal_wh.z;
  let h = u.iter_pal_wh.w;
  if (gid.x >= w || gid.y >= h) { return; }

  let max_iter = u.iter_pal_wh.x;
  let palette_id = u.iter_pal_wh.y;
  let t_max = max(u.tm.x, 1.0);
  let fk = u32(u.kind_j.x + 0.5);
  let mandel_exp = u.kind_j.y;

  let off_re = mul_u32_ds(gid.x, u.c_step.x, u.c_step.y);
  let re_ds = ds_add(u.c_base.x, u.c_base.y, off_re.x, off_re.y);

  let off_im = mul_u32_ds(gid.y, u.c_step.z, u.c_step.w);
  let im_ds = ds_add(u.c_base.z, u.c_base.w, off_im.x, off_im.y);

  var zrh: f32;
  var zrl: f32;
  var zih: f32;
  var zil: f32;
  var crh: f32;
  var crl: f32;
  var cih: f32;
  var cil: f32;

  if (fk == 1u) {
    zrh = re_ds.x;
    zrl = re_ds.y;
    zih = im_ds.x;
    zil = im_ds.y;
    crh = u.julia_lambda.x;
    crl = u.julia_lambda.y;
    cih = u.julia_lambda.z;
    cil = u.julia_lambda.w;
  } else {
    zrh = 0.0;
    zrl = 0.0;
    zih = 0.0;
    zil = 0.0;
    crh = re_ds.x;
    crl = re_ds.y;
    cih = im_ds.x;
    cil = im_ds.y;
  }

  var n = 0u;
  loop {
    if (n >= max_iter) { break; }
    let zrs = zrh + zrl;
    let zis = zih + zil;
    let r2 = zrs * zrs + zis * zis;
    if (r2 >= BAILOUT) {
      let zmag = max(sqrt(r2), 4.0000001);
      let smooth_iter = f32(n) + 1.0 - log2(log2(zmag));
      let tu = clamp(smooth_iter / t_max, 0.0, 1.0);
      let c = textureSampleLevel(palTex, palSam, vec2<f32>(tu, 0.5), 0.0);
      textureStore(outImg, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(c.r, c.g, c.b, 1.0));
      return;
    }
    let nz = step_escape_ds(zrh, zrl, zih, zil, crh, crl, cih, cil, fk, mandel_exp);
    zrh = nz.x;
    zrl = nz.y;
    zih = nz.z;
    zil = nz.w;
    n = n + 1u;
  }

  let zrs = zrh + zrl;
  let zis = zih + zil;
  let rgb = interior_rgb(zrs, zis, palette_id);
  textureStore(outImg, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb.x, rgb.y, rgb.z, 1.0));
}
`;

const WGSL_BLIT = /* wgsl */ `
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

@group(0) @binding(0) var blitSrc: texture_2d<f32>;
@group(0) @binding(1) var blitSam: sampler;

@fragment
fn fs_blit(@location(0) _uv: vec2<f32>, @builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let td = textureDimensions(blitSrc);
  let dims = vec2<f32>(f32(td.x), f32(td.y));
  let uv = pos.xy / dims;
  return textureSample(blitSrc, blitSam, uv);
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

  const format = navigator.gpu.getPreferredCanvasFormat();
  const maxStorage = device.limits.maxStorageBufferBindingSize;

  let swapConfiguredW = -1;
  let swapConfiguredH = -1;

  function configureSwapChain() {
    const nw = Math.max(1, canvas.width | 0);
    const nh = Math.max(1, canvas.height | 0);
    if (nw === swapConfiguredW && nh === swapConfiguredH) {
      return;
    }
    swapConfiguredW = nw;
    swapConfiguredH = nh;
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

  const modEscape = device.createShaderModule({
    code: wgslSource(WGSL_PARAMS_AND_DS + WGSL_INTERIOR + WGSL_ESCAPE),
  });
  const modBlit = device.createShaderModule({
    code: wgslSource(WGSL_BLIT),
  });

  const bindEscapeLayout = device.createBindGroupLayout({
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

  const layoutEscape = device.createPipelineLayout({ bindGroupLayouts: [bindEscapeLayout] });
  const layoutBlit = device.createPipelineLayout({ bindGroupLayouts: [bindBlitLayout] });

  const escapePipeline = device.createComputePipeline({
    layout: layoutEscape,
    compute: { module: modEscape, entryPoint: "escape_cs" },
  });

  const blitPipeline = device.createRenderPipeline({
    layout: layoutBlit,
    vertex: { module: modBlit, entryPoint: "vs_fullscreen" },
    fragment: {
      module: modBlit,
      entryPoint: "fs_blit",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const UNIFORM_ALIGN = 256;
  const uniformBuffer = device.createBuffer({
    size: UNIFORM_ALIGN,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const paramScratch = new ArrayBuffer(256);

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
  /** @type {GPUBindGroup | null} */
  let bindEscape = null;
  /** @type {GPUBindGroup | null} */
  let bindBlit = null;
  /** @type {GPUTexture | null} */
  let paletteTex = null;
  let lutKey = "";
  function destroyWork() {
    workTex?.destroy();
    workTex = null;
    bindEscape = null;
    bindBlit = null;
  }

  function destroyPalette() {
    paletteTex?.destroy();
    paletteTex = null;
    lutKey = "";
  }

  function refreshEscapeBindGroup() {
    if (!workTex || !paletteTex) return;
    bindEscape = device.createBindGroup({
      layout: bindEscapeLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: paletteTex.createView() },
        { binding: 2, resource: palSampler },
        { binding: 3, resource: workTex.createView() },
      ],
    });
  }

  function ensureWork(nw, nh) {
    if (nw === w && nh === h && workTex) {
      if (paletteTex && !bindEscape) {
        refreshEscapeBindGroup();
      }
      return;
    }
    destroyWork();
    w = nw;
    h = nh;
    if (w === 0 || h === 0) return;

    workTex = device.createTexture({
      size: [w, h],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });

    if (paletteTex) {
      refreshEscapeBindGroup();
    }

    bindBlit = device.createBindGroup({
      layout: bindBlitLayout,
      entries: [
        { binding: 0, resource: workTex.createView() },
        { binding: 1, resource: blitSampler },
      ],
    });
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
      refreshEscapeBindGroup();
    } finally {
      dealloc(ptr, need);
    }
  }

  /**
   * @param {object} p
   */
  function writeParams(p) {
    const nw = canvas.width;
    const nh = canvas.height;
    const maxIter = p.maxIter >>> 0;
    const lutIter =
      p.paletteIter !== undefined && p.paletteIter !== null ? p.paletteIter >>> 0 : maxIter;
    const tMax = Math.max(lutIter * 1.5, 64);
    const cx = p.centerX;
    const cy = p.centerY;
    const halfW = p.halfW;
    const aspect = nh / nw;
    const halfH = halfW * aspect;
    const du = (2 * halfW) / nw;
    const dv = (2 * halfH) / nh;
    const oRe = cx - halfW + 0.5 * du;
    const oIm = cy - halfH + 0.5 * dv;
    const [oRh, oRl] = splitToDoubleSingle(oRe);
    const [oIh, oIl] = splitToDoubleSingle(oIm);
    const [duRh, duRl] = splitToDoubleSingle(du);
    const [dvIh, dvIl] = splitToDoubleSingle(dv);

    const [jrH, jrL] = splitToDoubleSingle(p.juliaRe);
    const [jiH, jiL] = splitToDoubleSingle(p.juliaIm);

    new Float32Array(paramScratch, 0, 4).set([cx, cy, halfW, aspect]);
    new Uint32Array(paramScratch, 16, 4).set([maxIter, p.paletteId >>> 0, nw, nh]);
    const mandelExp =
      typeof p.mandelExponent === "number" && Number.isFinite(p.mandelExponent) ? p.mandelExponent : 2;
    new Float32Array(paramScratch, 32, 4).set([p.fractalKind >>> 0, mandelExp, 0, 0]);
    new Float32Array(paramScratch, 48, 4).set([tMax, 0, 0, 0]);
    new Float32Array(paramScratch, 64, 4).set([jrH, jrL, jiH, jiL]);
    new Float32Array(paramScratch, 80, 4).set([oRh, oRl, oIh, oIl]);
    new Float32Array(paramScratch, 96, 4).set([duRh, duRl, dvIh, dvIl]);
    device.queue.writeBuffer(uniformBuffer, 0, new Uint8Array(paramScratch));
  }

  /**
   * @param {object} p
   * @param {{ present?: boolean }} [opts]
   */
  function drawFull(p, opts = {}) {
    const present = opts.present !== false;
    const nw = canvas.width;
    const nh = canvas.height;
    if (nw < 1 || nh < 1) return 0;

    lastPaletteId = p.paletteId >>> 0;
    const lutIter =
      p.paletteIter !== undefined && p.paletteIter !== null ? p.paletteIter >>> 0 : p.maxIter >>> 0;
    lastMaxIter = lutIter;

    ensurePalette(lastPaletteId, lastMaxIter);
    ensureWork(nw, nh);
    if (!workTex || !paletteTex) {
      return 0;
    }

    configureSwapChain();
    writeParams(p);

    const t0 = performance.now();
    const encoder = device.createCommandEncoder();

    if (!bindEscape) refreshEscapeBindGroup();
    if (!bindEscape) return 0;
    const pass = encoder.beginComputePass();
    pass.setPipeline(escapePipeline);
    pass.setBindGroup(0, bindEscape);
    pass.dispatchWorkgroups(Math.ceil(nw / 8), Math.ceil(nh / 8));
    pass.end();

    if (present && bindBlit) {
      const view = ctx.getCurrentTexture().createView();
      const rpass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      rpass.setPipeline(blitPipeline);
      rpass.setBindGroup(0, bindBlit);
      rpass.draw(3);
      rpass.end();
    }
    device.queue.submit([encoder.finish()]);
    return performance.now() - t0;
  }

  function resize(nw, nh) {
    if (nw === w && nh === h && workTex) return;
    destroyWork();
    w = nw;
    h = nh;
    if (w > 0 && h > 0) {
      workTex = device.createTexture({
        size: [w, h],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      if (paletteTex) {
        refreshEscapeBindGroup();
      }
      bindBlit = device.createBindGroup({
        layout: bindBlitLayout,
        entries: [
          { binding: 0, resource: workTex.createView() },
          { binding: 1, resource: blitSampler },
        ],
      });
    }
  }

  function destroy() {
    destroyWork();
    destroyPalette();
    uniformBuffer.destroy();
    swapConfiguredW = -1;
    swapConfiguredH = -1;
  }

  return {
    drawFull,
    resize,
    destroy,
    /** @readonly */
    limits: { maxStorageBufferBindingSize: maxStorage },
    get usedGpuPerturb() {
      return false;
    },
  };
}

export function webGpuSupported() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}
