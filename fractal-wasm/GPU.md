# GPU (WebGPU) in this project

The live renderer is **WebGPU**: WGSL **f32** escape-time per pixel, LUT from WASM, optional affine **deep zoom** between checkpoints. There is no Canvas2D fractal path in the main UI anymore.

## How deep you can zoom (limits)

1. **Real limit — `f32`**  
   `centerX`, `centerY`, `halfW`, and each pixel’s `c` are **32-bit floats** (~7 decimal digits). When `halfW` gets small, the step between adjacent pixels’ `c` values loses precision relative to the orbit; the set **breaks up** (blocky/wrong) well before any JavaScript constant stops you. That is why the old WASM path used **f64** and **perturbation** for very deep Mandelbrot zooms.

2. **Code clamps in `main.js`** (soft / UX, not physics)  
   - **Keyboard +/−:** `halfW` clamped to about **`1e-16` … `96`** (`KEYBOARD_HALF_W_MIN` / `KEYBOARD_HALF_W_MAX`). The lower bound is far beyond what f32 can represent usefully for this map; it only prevents `0` or negative width.  
   - **Deep zoom button:** animation targets down to **`halfW ≈ 1e-13`** (`DEEP_ZOOM_HALF_W_MIN`), then stops with “At zoom limit”. That matches “very zoomed” without pretending f32 stays accurate there.  
   - **Box zoom:** `halfW` floored at **`1e-16`** when deriving the new view.

3. **Hybrid path (implemented)**  
   When `half_w < 0.02` (same threshold as `perturb::PERTURB_AUTO_HALF_W`), the app calls WASM **`render_rgba`** with **`perturb_mode = 2` (auto)**: **Mandelbrot** uses **tiled f64 perturbation** (`perturb.rs`); **Julia / Burning Ship / Tricorn** use **plain f64** escape-time (still far deeper than GPU f32). The RGBA buffer is **`writeTexture`**’d into the WebGPU work texture and **blit**’d to the swapchain. Large views can take noticeable CPU time; the status line shows **`· f64`** in that regime. Deep-zoom warps still run on the GPU but sample the CPU-rendered commit.

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

