# Manual regression: WASM Mandelbrot perturb + GPU handoff

1. **Mandelbrot**, default view: WebGPU only; image matches prior single-pass look.
2. Zoom until **half-w** drops below **`WASM_F64_HALF_W`** (~5e-5, zoom ×~4e4 from half-w 2): status may show higher frame time; image should **not** show large f32-style pixel blocks. Mandelbrot uses **tiled f64 perturb** inside `render_rgba` (`perturb_mode` 2).
3. **Julia** / **Burning Ship** past the same threshold: WASM **direct f64** escape (no perturb).
