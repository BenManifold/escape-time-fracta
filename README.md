# escape-time-fracta

Fractal explorer for the browser. The **interactive UI** draws with **WebGPU** ([`fractal-webgpu.js`](fractal-webgpu.js)): WGSL uses **double-single** arithmetic for per-pixel **c** and the orbit so deep zoom stays coherent (unlike a plain **f32**-only orbit). **WebAssembly** loads the [`fractal-wasm`](fractal-wasm) crate and runs **`fill_smooth_palette_lut`** on the CPU; the hosted app does **not** run the per-pixel escape loop in WASM.

The **`fractal-wasm`** library still implements **f64** iteration and optional **Mandelbrot perturbation** in `render_rgba` for tooling or future use ([`PERTURBATION.md`](fractal-wasm/PERTURBATION.md), [`GPU.md`](fractal-wasm/GPU.md)).

## Prerequisites (development)

1. **Rust** — Install with [rustup](https://rustup.rs/) (includes `cargo` and `rustc`). On Windows, the default `x86_64-pc-windows-msvc` toolchain needs the **Visual Studio C++ build tools** (or full VS with the “Desktop development with C++” workload) so the linker is available.

2. **WebAssembly target** — After installing Rust:

   ```bash
   rustup target add wasm32-unknown-unknown
   ```

3. **wasm-pack** — Builds the crate and generates JS glue:

   ```bash
   cargo install wasm-pack
   ```

4. **PATH** — If `cargo` or `wasm-pack` are not found in a new terminal, ensure `%USERPROFILE%\.cargo\bin` is on your PATH (rustup adds this for new shells; restart Cursor/VS Code or open a new terminal after install).

## Build the WASM package

From the `fractal-wasm` directory:

```bash
wasm-pack build --target web --release
```

From the **repository root** (positional crate path — same effect as `cd fractal-wasm`):

```bash
wasm-pack build fractal-wasm --target web --release
```

**Do not** pass Cargo’s `--manifest-path` to `wasm-pack build`. `wasm-pack` does not treat it as a first-class flag; it ends up in extra options forwarded to `cargo`. The build logic also scans those options for `--target` and treats the next token as a **Rust target triple**. If `--target web` (wasm-pack’s *output* mode for ES modules) lands in that list, it is misread as the triple `web`, which leads to `rustup target add web` and the error *toolchain does not support target 'web'*.

The renderer does not require WASM SIMD or extra `RUSTFLAGS`.

Output is written to `fractal-wasm/pkg/`.

## Serve the static site

Browsers require correct MIME types for `.wasm` files. Use any static server from the repo root (where `index.html` lives), for example:

```bash
npx --yes serve .
```

or Python:

```bash
python -m http.server 8080
```

Then open the URL shown in the terminal.

Do not open `index.html` directly from the file system (`file://`): ES module imports and WASM fetch require a local HTTP server.

### HTTPS on your LAN (phones, WebGPU)

WebGPU is only exposed in a **secure context**. Plain `http://192.168.x.x` from `npx serve` hides `navigator.gpu` on the phone even when GitHub Pages works.

From the repo root:

```bash
npm install
npm run dev:https
```

Open the printed `https://YOUR-LAN-IP:3443` on the phone. The first run creates a **self-signed** cert in `dev-certs/` (gitignored) with SAN entries for `localhost`, `127.0.0.1`, and your current LAN IPv4 addresses, plus standard TLS extensions so strict clients (e.g. iOS Safari) accept the connection. **Trust `dev-certs/cert.pem` on the phone** (e.g. AirDrop the file to iPhone, install the profile, then enable full trust under Settings → General → About → Certificate Trust Settings). If your Wi‑Fi IP changes, run `npm run dev:https:regen` and trust the new cert.

If you still see a TLS / “secure connection failed” error after pulling updates, run `npm run dev:https` again (certs auto-regenerate when the dev format version changes) and re-install trust for the new `cert.pem`.

Optional: set `DEV_HTTPS_PORT` to use a port other than `3443`.

## Browser note

The live app **requires WebGPU** (recent Chromium, Edge, or another WebGPU-capable browser). If the GPU path fails to start, the status line shows an error.

You still need **WebAssembly** support so the WASM module can load. If initialization fails, check the console for fetch/MIME or instantiation errors.
