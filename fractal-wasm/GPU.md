# GPU acceleration (browser) — options for this project

The app today is **CPU (WASM)** → `ImageData` → **Canvas2D**. Moving hot loops to the GPU is a separate path; perturbation and GPU are **orthogonal** (you can do either, both, or neither).

## WebGPU (preferred when available)

- **Compute pass**: one **workgroup per pixel** (or 8×8 tiles). Inner loop = escape or perturbation step; output RGBA8 to a `storage-buffer` or `rgba8unorm` texture.
- **Perturbation on GPU**: upload reference orbit `Z_n` as a **storage buffer** (`vec2<f32>` × `(max_iter+1)`) or a **1D texture** `rg32float`. Per pixel: `δc` from uniform view + `global_id`, run the same δ recurrence as `perturb.rs`, **glitch branch** can write a sentinel and a cheap second pass fixes those pixels on CPU—or use **atomics** / skip (more complex).
- **Series approximation**: precompute polynomial coefficients on CPU (or prior compute pass), evaluate per pixel in WGSL (Fractalshades-style).
- **Interop**: readback via `copyBufferToTexture` / `mapAsync` into existing canvas, or present with **WebGPU canvas context**.

Detection: `navigator.gpu?.requestAdapter()`. Safari / older browsers may lack WebGPU; keep WASM fallback.

## WebGL2 (wider reach)

- **Fragment shader** full-screen quad: `v_uv` → complex `c`, loop `max_iter` in shader (unrolled bands or dynamic loop with `break` if driver allows). **f64 is not available** — use **two f32** (emulated double) for deep zoom, or stay shallow.
- **Perturbation**: bind reference orbit as **2×N texture** or **data texture**; sample `Z_n` by iteration index (loop counter as float index — awkward). Often easier to ship **non-perturb** GPU first, WASM perturb for deep zoom only.

## Hybrid (practical)

1. **GPU** for interactive pan/zoom at moderate depth (`half_w` not tiny).
2. **WASM + perturb** when `half_w` is small or user forces quality.
3. **Worker** already used for deep-zoom checkpoints; GPU readback can feed the same `ImageData` path.

## References

- [WebGPU specification](https://www.w3.org/TR/webgpu/)  
- [WebGPU samples — compute](https://github.com/webgpu/webgpu-samples)  
- Shadertoy Mandelbrot shaders (GLSL ideas; adapt to WGSL)

