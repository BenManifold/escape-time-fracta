# GPU rendering (WebGPU)

The interactive app draws entirely on the **GPU** ([fractal-webgpu.js](../fractal-webgpu.js)):

- **Compute** `escape_cs`: one pass per pixel. **Pixel `c`** is built with **double-single** arithmetic from corner + per-pixel steps; the **orbit `z`** and addend **`c`** in the iteration use **double-single** complex squaring and addition (Julia λ is also double-single). Burning Ship applies `abs` per component in double-single before squaring. This avoids plain **f32** orbit drift at deep zoom without a WASM pixel path.
- **Blit** to the canvas swapchain.

**WASM** ([fractal-wasm](.)) is used only to fill the **palette LUT** (`fill_smooth_palette_lut`) on the CPU; pixels are not rendered there in the web app.
