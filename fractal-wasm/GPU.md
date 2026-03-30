# GPU rendering (WebGPU)

The interactive app draws with **WebGPU** ([fractal-webgpu.js](../fractal-webgpu.js)):

- **Compute** `escape_cs`: one pass per pixel, f32 iteration, **double-single** mapping from pixel index to **c**.
- **Blit** to the canvas swapchain.

When the view **half-width on Re** drops below **`WASM_F64_HALF_W`** in [main.js](../main.js), **all** fractals switch to WASM **`render_rgba`** (full **f64** complex plane sampling) and upload RGBA to the GPU work texture, then blit. That avoids f32 uniform quantization (visible around zoom ×1e5 for Mandelbrot on GPU alone).

**Mandelbrot (WASM only):** `render_rgba` with **`perturb_mode == 2`** uses **tiled f64 perturbation** ([perturb.rs](src/perturb.rs)) when `half_w < 0.02`, same as before the GPU perturb experiment. Julia and Burning Ship use direct **f64** escape in WASM.

Palette: WASM fills an **RGBA8 LUT**; the GPU samples it when using the escape shader.
